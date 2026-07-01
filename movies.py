"""
PIXEL CINEMA — the "Cinema" model directs movies that play in the Claudecade engine.

You give it a CONCEPT (or a public-domain book). The model writes the story AND
directs the animation as a structured "movie script" (shots: narration + a cast of
sprites + motion). The Claudecade engine plays it. Movies save out as JSON and
replay with no model.

    python movies.py --concept "a lonely robot who finds a cat"
    python movies.py --book 84                 # Frankenstein (Project Gutenberg id)
    python movies.py --play saved/<file>.json  # replay a saved movie
    python movies.py --concept "..." --dry     # just print the script (no terminal)

Model:  CLAUDEMOVIES_LLM_URL / _KEY / _MODEL  (OpenAI-compatible, e.g. Modal vLLM)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import TYPE_CHECKING

import ascii_sprites as A
import characters
import craft_kb
from llm_client import llm  # the director model (leaf module; re-exported as movies.llm)

if TYPE_CHECKING:
    from schema import RGB, FloorProfile, MovieSpec, ScenePlan, Shot

# ── timing: films run ~2 minutes ──────────────────────────────────────────────
# One canonical frame = FRAME_MS; every player (gif, web stage, terminal) targets it.
# A shot = a short MOVE window (entrance + subtitle types on) + a HOLD window sized
# to the reading time of its text, so subtitles stay up long enough to read.
FRAME_MS = 150                # ms per frame  (≈6.7 fps timeline)
TARGET_SECONDS = 90           # ~90-second stories
ENGINE_FPS = 14               # terminal player tick rate (gif/web use the FRAME_MS timeline)
MOVE_FRAMES = 14              # entrance + type-on (~2.1s)
_READ_CPS = 8.0              # chars/sec a viewer reads subtitles (lower = beats dwell longer)
_HOLD_MIN, _HOLD_MAX = 28, 60  # 4.2s .. 9.0s of dwell  (9 beats land near ~90s)


def shot_budget(shot: Shot) -> tuple[int, int]:
    """(move, hold) frames for a shot. HOLD scales with how much text must be read
    (narration + dialogue), clamped, so dense beats dwell longer and sparse beats
    don't drag. 13 such beats land the film near TARGET_SECONDS."""
    chars = len(shot.get("narration", "")) + len(shot.get("dialogue", ""))
    read = int((chars / _READ_CPS) * (1000 / FRAME_MS))
    return MOVE_FRAMES, max(_HOLD_MIN, min(_HOLD_MAX, read))


# ── the FLOOR: draw the ground as what it actually is, with shade/block glyphs ──
# Each floor = a textured SURFACE row + a FILL row beneath it. Unicode shading
# blocks (░▒▓█), lower blocks (▁▂▃), and wave glyphs give depth; `anim` shimmers.
FLOOR: dict[str, FloorProfile | None] = {
    "water":  dict(surf="~≈  ~ ≈ ", fill="░", scol=(120, 180, 235), fcol=(45, 95, 165),  tcol="BLUE",   anim=True),
    "grass":  dict(surf="▂▁▃▁▂▁",   fill="▒", scol=(130, 205, 95),  fcol=(70, 140, 60),  tcol="GREEN",  anim=False),
    "forest": dict(surf="▃▁▂▁▃▁",   fill="▓", scol=(110, 170, 80),  fcol=(60, 110, 55),  tcol="GREEN",  anim=False),
    "sand":   dict(surf="▁▁▂▁▁▁",   fill="░", scol=(225, 205, 150), fcol=(200, 178, 120), tcol="YELLOW", anim=False),
    "snow":   dict(surf="▁▂▁▁▁▂",   fill="░", scol=(238, 242, 250), fcol=(205, 215, 232), tcol="WHITE",  anim=False),
    "stone":  dict(surf="▔▔ ▔ ▔ ",  fill="▓", scol=(155, 155, 165), fcol=(95, 95, 105),  tcol="WHITE",  anim=False),
    "road":   dict(surf="─  ─  ─ ",  fill="▒", scol=(125, 125, 135), fcol=(80, 80, 90),   tcol="WHITE",  anim=False),
    "lava":   dict(surf="~▒░~ ▒ ",  fill="▓", scol=(255, 150, 50),  fcol=(175, 45, 25),  tcol="RED",    anim=True),
    "earth":  dict(surf="▔▔▔▔▔▔",   fill="▒", scol=(160, 120, 80),  fcol=(95, 68, 42),   tcol="YELLOW", anim=False),
    "indoor": dict(surf="═══════",  fill="▒", scol=(190, 160, 120), fcol=(120, 92, 62),  tcol="YELLOW", anim=False),
    "sky":    None,                                              # open air — draw no floor
}
SETTINGS = tuple(FLOOR)                                          # valid director "setting" values

_FLOOR_KW = [
    ("indoor", "indoor indoors inside room kitchen bedroom bathroom hallway hall attic cellar basement parlor study office classroom bakery workshop cottage cabin cozy fireplace hearth bed couch sofa desk shelf carpet rug"),
    ("water",  "water sea ocean waves wave river lake pond rain flood tide harbor shore splash drain puddle stream marsh swamp creek waterfall dock pier"),
    ("lava",   "lava volcano magma ember inferno furnace molten"),
    ("snow",   "snow ice frost arctic tundra glacier winter frozen blizzard icy"),
    ("sand",   "sand desert beach dune oasis"),
    ("grass",  "grass meadow field lawn garden hill prairie pasture savanna valley"),
    ("forest", "forest wood woods glade thicket jungle grove"),
    ("stone",  "stone rock cave cavern dungeon castle mountain cliff cobble temple ruin crypt fortress tunnel"),
    ("road",   "road street path sidewalk alley pavement trail lane avenue gutter curb"),
    ("sky",    "sky cloud clouds space stars flying soar heavens cosmos orbit"),
]
_WATER_CAST = {"ship", "boat", "fish", "whale", "dolphin", "shark", "crab", "duck", "swan", "octopus", "mermaid", "penguin"}


def floor_kind(narration: str, cast=(), setting: str | None = None) -> str:
    """Pick a floor for a shot: an explicit director `setting` wins; else infer it
    from keywords in the narration; else from a water-dwelling cast; else 'earth'."""
    if setting in FLOOR:
        return setting
    low = " " + (narration or "").lower() + " "
    for kind, words in _FLOOR_KW:
        if any(re.search(r"\b" + w + r"\b", low) for w in words.split()):
            return kind
    if _WATER_CAST & {str(c).lower() for c in cast}:
        return "water"
    return "earth"

# ── colour: every sprite gets a sensible colour, shared by gif/web (RGB) and
#    terminal (curses). A named colour resolves to an RGB and a curses Color name.
PALETTE = {                              # name: (rgb, curses-color-name)
    "white":   ((230, 237, 243), "WHITE"),
    "gray":    ((175, 180, 190), "WHITE"),
    "red":     ((255, 107, 107), "RED"),
    "orange":  ((255, 165, 70),  "YELLOW"),
    "yellow":  ((255, 214, 92),  "YELLOW"),
    "green":   ((124, 220, 140), "GREEN"),
    "cyan":    ((121, 192, 255), "CYAN"),
    "blue":    ((110, 150, 235), "BLUE"),
    "magenta": ((235, 140, 210), "MAGENTA"),
    "pink":    ((255, 150, 190), "MAGENTA"),
    "brown":   ((185, 135, 90),  "YELLOW"),
    "tan":     ((220, 200, 150), "YELLOW"),
    "purple":  ((190, 150, 235), "MAGENTA"),
}
SPRITE_COLOR = {                         # canonical sprite label -> palette name
    # people
    "person": "cyan", "child": "cyan", "knight": "gray", "king": "yellow", "wizard": "purple",
    "robot": "gray", "pirate": "red", "ninja": "gray", "witch": "purple", "fairy": "pink",
    "angel": "white", "skeleton": "white", "zombie": "green", "vampire": "red", "alien": "green",
    "astronaut": "white", "princess": "pink", "farmer": "brown", "chef": "white", "clown": "red",
    "viking": "brown", "samurai": "red", "mermaid": "cyan", "genie": "cyan", "snowman": "white",
    "scarecrow": "brown", "devil": "red", "elf": "green", "dwarf": "brown", "troll": "green",
    # animals
    "cat": "yellow", "dog": "brown", "lion": "orange", "tiger": "orange", "bear": "brown",
    "panda": "white", "wolf": "gray", "fox": "orange", "deer": "brown", "horse": "brown",
    "cow": "white", "pig": "pink", "sheep": "white", "goat": "white", "elephant": "gray",
    "monkey": "brown", "rabbit": "white", "mouse": "gray", "frog": "green", "turtle": "green",
    "snake": "green", "bee": "yellow", "butterfly": "magenta", "ladybug": "red", "spider": "gray",
    "crab": "red", "octopus": "magenta", "fish": "cyan", "dolphin": "cyan", "whale": "blue",
    "shark": "gray", "penguin": "white", "duck": "yellow", "chicken": "white", "bird": "cyan",
    "owl": "brown", "parrot": "green", "swan": "white", "hedgehog": "brown", "bat": "gray",
    "dragon": "red", "monster": "magenta", "ghost": "white",
    # nature / objects
    "tree": "green", "flower": "pink", "mountain": "gray", "moon": "yellow", "sun": "yellow",
    "star": "yellow", "cloud": "white", "house": "yellow", "castle": "gray", "door": "brown",
    "ship": "cyan", "key": "yellow", "book": "cyan", "sword": "gray", "candle": "yellow",
    "crown": "yellow", "clock": "white", "oven": "orange",
    "cactus": "green", "pine": "green", "bush": "green", "rock": "gray", "bigtree": "green",
    "campfire": "orange", "torch": "orange",
}

# inanimate "props" the cast can gather AROUND — staged centre-stage so characters flank them
PROPS = {"campfire", "torch", "fireplace", "bed", "table", "chair", "well", "cauldron",
         "tent", "oven", "chest", "treasure", "cake", "bread", "pie"}


def _color_name(label):
    """Palette name for a sprite label (resolving aliases). Unknown/model-drawn
    characters get a stable colour derived from the name, so they vary but stay
    consistent across shots."""
    label = (label or "").strip().lower()
    saved = characters.color(label)                 # a saved character's chosen colour wins
    if saved in PALETTE:
        return saved
    canon = A.ALIASES.get(label, label)
    if canon in SPRITE_COLOR:
        return SPRITE_COLOR[canon]
    keys = list(PALETTE)
    return keys[sum(ord(c) for c in label or "x") % len(keys)]


def sprite_rgb(label: str) -> RGB:
    return PALETTE[_color_name(label)][0]


def sprite_curses(label: str) -> str:
    return PALETTE[_color_name(label)][1]


def known_sprite(label: str) -> bool:
    """True if `label` resolves to a built-in or saved character (incl. aliases)."""
    return A.get(label) is not None or characters.get(label) is not None


# ── expression: a small emote glyph drawn over a character's head ──────────────
MOOD_EMOTE = {                           # mood -> (glyph, palette colour name)
    "happy": ("♪", "yellow"), "joyful": ("♪", "yellow"), "excited": ("!", "yellow"),
    "sad": ("'", "cyan"), "crying": ("'", "cyan"),
    "angry": ("!", "red"), "mad": ("!", "red"), "furious": ("!", "red"),
    "scared": ("°", "white"), "afraid": ("°", "white"), "fear": ("°", "white"),
    "fearful": ("°", "white"), "terrified": ("°", "white"), "nervous": ("°", "white"),
    "worried": ("°", "white"), "anxious": ("°", "white"),
    "dreamy": ("z", "gray"), "weird": ("?", "white"),
    "hopeful": ("♪", "yellow"), "proud": ("!", "orange"), "lonely": ("'", "cyan"),
    "love": ("♥", "pink"), "smitten": ("♥", "pink"),
    "surprised": ("!", "yellow"), "shocked": ("!", "yellow"),
    "confused": ("?", "white"), "curious": ("?", "cyan"),
    "sleepy": ("z", "gray"), "tired": ("z", "gray"),
    "brave": ("!", "orange"), "determined": ("!", "orange"),
}


def mood_emote(mood: str | None):
    """(glyph, rgb) to float over a character's head for a mood, or None."""
    e = MOOD_EMOTE.get((mood or "").strip().lower())
    return (e[0], PALETTE[e[1]][0]) if e else None


def mood_curses(mood: str | None) -> str:
    e = MOOD_EMOTE.get((mood or "").strip().lower())
    return PALETTE[e[1]][1] if e else "WHITE"


# ── camera: a per-shot move applied to the cast (and tracked scenery) ──────────
CAMERAS = ("static", "pan", "push", "shake")


def camera_offset(camera: str, p: float, f: int) -> tuple[int, int]:
    """(dx, dy) cell offset at full-shot progress p in [0,1], frame index f.
    pan = drift across the stage; push = ease down (feels closer); shake = jitter."""
    if camera == "pan":
        return int(round((p - 0.5) * 12)), 0
    if camera == "push":
        return 0, int(round(p * 2))
    if camera == "shake":
        return (1 if f % 2 else -1), (1 if f % 3 == 0 else 0)
    return 0, 0


# ── shared staging helpers (used by both the gif/web renderer and the terminal) ─
def appearance_order(spec: MovieSpec) -> list[str]:
    """First-appearance order of every character, so a recurring character keeps
    its left-to-right place shot to shot."""
    order = []
    for sh in spec.get("shots", []):
        for nm in sh.get("cast", []):
            if nm not in order:
                order.append(nm)
    return order


def home_columns(cast: list[str], W: int = 64, order: list[str] | None = None) -> dict[str, int]:
    """Even, non-overlapping x for the characters present in ONE shot (>=14 cols
    apart, sprites are ~10 wide). `order` keeps placement stable across shots."""
    if order:
        cast = [n for n in order if n in cast] or list(cast)
    cast = list(dict.fromkeys(cast)) or ["tree"]
    # stage any props (a campfire, table, ...) CENTRE-stage so characters flank them
    props = [c for c in cast if c in PROPS]
    chars = [c for c in cast if c not in PROPS]
    if props and chars:
        h = len(chars) // 2
        cast = chars[:h] + props + chars[h:]
    n = len(cast)
    left, right = 7, W - 11
    span = right - left
    if n == 1:
        return {cast[0]: left + span // 2 - 4}
    step = max(14, span // (n - 1))
    total = step * (n - 1)
    start = max(2, left + (span - total) // 2) if total < span else left
    return dict(zip(cast, [start + i * step for i in range(n)]))


def ease(t: float) -> float:
    """Smoothstep: gentle accelerate/decelerate, clamped to [0, 1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def stamped_slug(title: str) -> str:
    """Filename stem `slug-YYYYMMDD-HHMMSS` so saved files never overwrite."""
    slug = re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")[:40] or "movie"
    return f"{slug}-{time.strftime('%Y%m%d-%H%M%S')}"


def saved_dir() -> str:
    """The (created) directory where movies and renders are written."""
    d = os.path.join(os.path.dirname(__file__), "saved")
    os.makedirs(d, exist_ok=True)
    return d


# ── scenery: set-dressing behind the cast (sun/moon, clouds, stars, trees) ─────
_NIGHT_KW = ("night", "midnight", "moonlit", "moonlight", "starry", "stars",
             "evening", "dusk", "twilight", "dark")


def scenery(shot, kind):
    """Decide the set-dressing for a shot from its floor `kind` + narration:
      sky:    'sun' | 'moon' | None        (drawn in a top corner)
      clouds: how many drifting clouds      stars: how many night stars
      ground: [(label, frac_x), ...]        static props drawn BEHIND the cast
    Auto, but consistent while a scene stays in one place."""
    narr = (shot.get("narration", "") or "").lower()
    night = any(w in narr for w in _NIGHT_KW) or "moon" in shot.get("cast", [])
    p = {"sky": None, "clouds": 0, "stars": 0, "ground": []}
    if kind == "sky":
        p.update(sky="moon" if night else "sun", clouds=3, stars=6 if night else 0)
    elif kind in ("grass", "forest", "earth", "road"):
        trees = [("bigtree", 0.0), ("tree", 0.94)] if kind in ("grass", "forest") else [("tree", 0.95)]
        p.update(sky="moon" if night else "sun", clouds=0 if night else 2,
                 stars=7 if night else 0, ground=trees)
    elif kind == "sand":
        p.update(sky="sun", clouds=1, ground=[("cactus", 0.02), ("cactus", 0.95)])
    elif kind == "snow":
        p.update(sky="moon" if night else "sun", clouds=0 if night else 1,
                 stars=8 if night else 0, ground=[("pine", 0.0), ("pine", 0.93)])
    elif kind == "water":
        p.update(sky="moon" if night else "sun", clouds=0 if night else 2, stars=6 if night else 0)
    elif kind == "stone":
        p.update(ground=[("rock", 0.04), ("mountain", 0.9)])
    # lava / unknown: no extra dressing
    return p
SYN = {"person": ["man", "woman", "girl", "boy", "child", "victor", "alice", "knight", "king", "queen"],
       "robot": ["robot", "machine", "android", "bot"], "cat": ["cat", "kitten"],
       "rabbit": ["rabbit", "hare", "bunny"], "monster": ["monster", "creature", "beast", "wretch", "demon"],
       "tree": ["tree", "forest", "wood", "garden"], "house": ["house", "home", "cottage"],
       "moon": ["moon", "night"], "ship": ["ship", "boat", "sea", "ocean"],
       "castle": ["castle", "palace", "tower"], "bird": ["bird", "raven", "owl"],
       "flower": ["flower", "rose", "bloom"]}


def _detect(text: str) -> list[str]:
    low = " " + text.lower() + " "
    out = [n for n, syns in SYN.items()
           if any(re.search(r"\b" + re.escape(w) + r"\b", low) for w in [n] + syns)]
    return out[:4] or ["person"]


# ── airtight prose↔visuals: every drawable thing the narration names ends up on screen ──
# Common concrete objects models mention that aren't in the sprite library but the ASCII
# artist can still draw on demand (drawn once, then cached).
_EXTRA_DRAWABLE = [
    "fire", "fireplace", "campfire", "bonfire", "torch", "bed", "window", "table", "chair",
    "lamp", "cup", "mug", "kettle", "cake", "bread", "pie", "cookie", "umbrella", "kite",
    "balloon", "gift", "map", "letter", "mirror", "bell", "lighthouse", "bridge", "fence",
    "well", "cauldron", "pumpkin", "mushroom", "acorn", "leaf", "car", "train", "rocket",
    "treasure", "chest", "crystal", "gem", "drum", "guitar", "violin", "anchor", "compass",
    "telescope", "hat", "boot", "ladder", "tent", "campsite", "snowflake", "raindrop",
]

# word -> canonical sprite label. ONLY real library sprites + aliases — so narrated objects are
# auto-cast only when we have a proper sprite for them. Loose props (window, table, bread, ...) are
# NOT auto-cast, so they never get drawn as standing "characters".
_VOCAB: dict[str, str] = {lab: lab for lab in A.SPRITES}
for _al, _canon in A.ALIASES.items():
    _VOCAB.setdefault(_al, _canon)
_VOCAB_RE = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, _VOCAB), key=len, reverse=True)) + r")\b", re.I)


def _mentioned(text: str) -> list[str]:
    """Every drawable subject named in `text`, as canonical sprite labels, in order."""
    seen, out = set(), []
    for m in _VOCAB_RE.finditer(text or ""):
        canon = _VOCAB[m.group(1).lower()]
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def _loads_lenient(cand: str) -> dict | None:
    """json.loads with repairs for small-model artifacts: strip code fences, fix
    the common 'every inner quote backslash-escaped' bug (\\"narration\\" -> "narration"),
    and drop trailing commas before } or ]. Returns None if nothing parses."""
    cand = cand.strip()
    for fix in (lambda x: x,                                   # 1. as-is
                lambda x: x.replace('\\"', '"'),              # 2. un-escape stray \"
                lambda x: re.sub(r",(\s*[}\]])", r"\1",       # 3. (after 2) trailing commas
                                 x.replace('\\"', '"'))):
        try:
            out = json.loads(fix(cand))
            if isinstance(out, dict):
                return out
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _extract_json(s: str) -> dict | None:
    s = re.sub(r"```(?:json)?|```", "", s)        # drop markdown code fences if present
    i = s.find("{")
    while i != -1:
        depth = 0
        for j in range(i, len(s)):
            depth += (s[j] == "{") - (s[j] == "}")
            if depth == 0:
                out = _loads_lenient(s[i:j + 1])
                if out is not None:
                    return out
                break                             # candidate unrepairable: scan to next '{'
        i = s.find("{", i + 1)
    return None


def _salvage(s: str) -> dict | None:
    """Last-ditch parse for messy/truncated model JSON: pull the title (if any) and
    every individual shot object that parses, ignoring a broken/cut-off tail. This
    survives token-truncation, stray backslash-escapes, and array-valued fields —
    the three ways the small director model tends to mangle its JSON output."""
    s = re.sub(r"```(?:json)?|```", "", s)
    arr = re.search(r'"shots"\s*:\s*\[', s)             # only scan inside the shots array
    start = arr.end() if arr else (s.find("{") + 1)
    shots, depth, obj_start, in_str, esc = [], 0, None, False, False
    for j in range(start, len(s)):
        ch = s[j]
        if in_str:                                      # don't count braces inside strings
            esc = (ch == "\\" and not esc)
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                obj_start = j
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                obj = _loads_lenient(s[obj_start:j + 1])
                if isinstance(obj, dict):
                    shots.append(obj)
        elif ch == "]" and depth == 0:
            break
    if not shots:
        return None
    tm = re.search(r'"title"\s*:\s*"([^"]+)"', s)
    return {"title": tm.group(1).strip() if tm else "", "shots": shots}


ACTIONS = ("enter", "rise", "gather", "exit")
N_SHOTS = 9                                      # the director writes a tight 9-beat film


def good_shot_count(n):
    """A well-formed film has roughly the 9 beats; accept a tolerant band."""
    return 6 <= n <= 12

# A 9-shot example sets the quality bar AND locks the ~90-second length and the tight
# 3-act beat structure (the example format IS the output format).
_EXAMPLE = ('{"title":"The Paper Boat","logline":"A folded paper boat braves the storm drains to reach the sea it has only dreamed of.",'
            '"shots":['
            '{"narration":"Rain hammers the gutter; a folded paper boat shivers against the cold curb.","cast":["ship"],"action":"rise","setting":"road"},'
            '{"narration":"All it has ever wanted is the sea — a blue it knows only from stories.","cast":["ship"],"action":"gather","setting":"road","mood":["dreamy"],"dialogue":"someday..."},'
            '{"narration":"A swell tears it loose and drags it spinning into the roaring drain.","cast":["ship"],"action":"enter","setting":"water","camera":"pan"},'
            '{"narration":"In the black tunnel a lost goldfish surfaces beside it, fins trembling.","cast":["ship","fish"],"action":"enter","setting":"water","mood":["","scared"],"dialogue":"lost too?"},'
            '{"narration":"Side by side they ride the current, daring the dark together.","cast":["ship","fish"],"action":"gather","setting":"water"},'
            '{"narration":"A rusted grate slams the only salt-smelling path shut.","cast":["ship","fish"],"action":"gather","setting":"stone","dialogue":"no way through..."},'
            '{"narration":"The little boat sags, soaked soft, its paper hull peeling apart.","cast":["ship"],"action":"gather","setting":"water","mood":["sad"]},'
            '{"narration":"The goldfish heaves it onto a leaf and shoves it through the bars.","cast":["ship","fish"],"action":"rise","setting":"water","camera":"shake"},'
            '{"narration":"The tunnel opens — and there it is, the endless silver sea.","cast":["ship","moon"],"action":"gather","setting":"water","camera":"push","mood":["happy",""],"dialogue":"we made it."}]}')

DIRECTOR_SYS = (
    "You are a master storyteller and ASCII-animation director. You INVENT ORIGINAL stories — never "
    "retell, adapt, or reference existing works. From a concept, write ONE COMPLETE ~90-second film "
    "with a clear beginning, middle, and end.\n\n"
    "STRUCTURE — write EXACTLY 9 shots, one per beat, a tight three-act arc:\n"
    " ACT I  — 1 Opening image: establish the hero and their world\n"
    "          2 The want: the one thing the hero yearns for (make us care)\n"
    "          3 Inciting incident: a disruption forces them to act\n"
    " ACT II — 4 The leap + ally/obstacle: they commit; someone or something enters\n"
    "          5 Rising action: they pursue the want, the world pushes back\n"
    "          6 Midpoint turn: the stakes jump or the plan breaks\n"
    "          7 Low point: hope collapses, all seems lost\n"
    " ACT III— 8 Climax: the decisive choice or confrontation\n"
    "          9 Resolution / final image: the earned ending that echoes the opening\n\n"
    "CRAFT — every line must earn its place:\n"
    "- A clear emotional core — ONE protagonist, OR a central pair (e.g. two friends) — with ONE "
    "clear want and ONE real obstacle. If the concept names two characters, BOTH are leads: keep "
    "them both on screen and let their relationship drive the story. Keep the SAME characters throughout.\n"
    "- Cause and effect: each beat is caused by the last. The ending pays off the want from beat 2.\n"
    "- Prose: present tense, concrete and sensory. Specific nouns and strong verbs, not adjectives. "
    "Show feeling through action, don't name it. No clichés, no narrating 'our story begins'. "
    "Write CLEAR, grammatical English — every line must make literal sense; vivid, never surreal word-salad.\n"
    "- Each narration is ONE vivid sentence, 55-95 characters.\n"
    "- Dialogue is rare and only when it lands (3-4 shots at most), <=40 chars, lowercase, natural.\n"
    "- One true beat of humor or heart. Earn the ending — surprising yet inevitable.\n\n"
    "OUTPUT — strict JSON only, no prose, no markdown:\n"
    '{"title":str,"logline":one sentence,"shots":[{"narration":"55-95 chars, present tense, one sentence",'
    '"cast":[1-3 subjects on screen],"action":"' + "|".join(ACTIONS) + '",'
    '"setting":"' + "|".join(SETTINGS) + '","camera":"' + "|".join(CAMERAS) + '",'
    '"mood":[optional feeling per cast member],"dialogue":"optional, <=40 chars"}]}\n'
    "CAST: each entry is ONE lowercase noun for a character/object (e.g. fox, knight, robot, moon, "
    "ship, dragon, snowman). Any common subject works — it will be drawn in ASCII. Keep the same "
    "characters across shots so the audience can follow them.\n"
    "VISUALS RULE — ONLY the cast subjects are drawn on screen. If your narration names an important "
    "visible thing (a fireplace, a door, a tree, the moon, a candle), it MUST also appear in that "
    "shot's cast or the audience won't see it. Never describe an object or character that isn't in "
    "the cast. Keep the cast small (1-3) and consistent.\n"
    "SETTING: where the shot takes place — pick what FITS the story, not always outdoors. Use 'indoor' "
    "for any room/house/kitchen/shop interior; water/grass/forest/sand/snow/stone/road/lava/earth for "
    "outdoor terrain; 'sky' for floating/flying. Keep it consistent while the scene stays in one place.\n"
    "CAMERA: 'static' for most shots; 'pan' for travel/reveal, 'push' to lean into an emotional beat, "
    "'shake' for impact (use on the climax). Use sparingly.\n"
    "MOOD: optional — one feeling per cast member, SAME order and length as cast (use \"\" for none); "
    "drawn as an emote over the character. Options: " + ", ".join(MOOD_EMOTE) + ". Save it for real "
    "emotional beats.\n\n"
    "EXAMPLE (match this format, length, structure, and prose quality):\n" + _EXAMPLE)


PITCH_SYS = (
    "You are REEL, a witty late-night host for an ASCII animation channel. Pitch THREE ORIGINAL short "
    "films for tonight. Make the three feel different in tone — one heartfelt, one funny, one strange. "
    "Original only; never reference real films. Output JSON only: "
    '{"films":[{"title":str,"premise":"one tantalizing sentence"}]}')


def pitch(mood: str = "") -> list[dict[str, str]]:
    """Host suggests 3 original films (title + premise). Used for 'what's on tonight?'."""
    raw = llm(PITCH_SYS, f"The viewer is in the mood for: {mood or 'anything good'}. What's on tonight?",
              max_tokens=320, temperature=1.05)
    data = _extract_json(raw) if raw else None
    films = (data or {}).get("films") if isinstance(data, dict) else None
    if films and isinstance(films, list):
        clean = [{"title": str(f.get("title", "Untitled"))[:40], "premise": str(f.get("premise", ""))[:120]}
                 for f in films[:3] if f.get("premise")]
        if clean:
            return clean
    return [{"title": "The Long Way Home", "premise": "A lost robot follows a cat through a haunted wood."},
            {"title": "Moonstruck Mouse", "premise": "A mouse builds a ladder to steal a slice of the moon."},
            {"title": "Tea With a Dragon", "premise": "A knight and a dragon would rather share tea than fight."}]


def _clip(s: str, n: int) -> str:
    """Clip `s` to `n` chars at a WORD boundary (with an ellipsis), never mid-word."""
    if len(s) <= n:
        return s
    cut = s[:n - 1]
    if " " in cut:
        cut = cut[:cut.rfind(" ")]
    return cut.rstrip(" ,;:") + "…"


def _flat(x) -> str:
    """Coerce a model field that may be a str OR a list-of-strings into one string."""
    if isinstance(x, list):
        return " ".join(str(i).strip() for i in x if str(i).strip())
    return str(x or "")


def _sanitize_shot(sh: dict, concept: str = "") -> dict:
    """Coerce a model-produced shot dict into a safe, playable Shot."""
    narr = _flat(sh.get("narration"))
    cast = [str(c).strip().lower() for c in sh.get("cast", []) if str(c).strip()][:4] \
        or _detect(narr or concept)
    for m in _mentioned(narr):          # airtight: draw every drawable thing the line names
        if m not in cast and len(cast) < 4:
            cast.append(m)
    sh["cast"] = cast
    sh["action"] = sh.get("action") if sh.get("action") in ACTIONS else "gather"
    sh["narration"] = _clip(narr, 300)                           # full sentence(s); paginated subtitles
    sh["dialogue"] = (_clip(_flat(sh["dialogue"]), 60) if sh.get("dialogue") else "")
    sh["setting"] = floor_kind(sh["narration"], sh["cast"], sh.get("setting"))
    sh["camera"] = sh.get("camera") if sh.get("camera") in CAMERAS else "static"
    m = sh.get("mood")
    sh["mood"] = [str(x).strip().lower() for x in m][:len(sh["cast"])] if isinstance(m, list) else []
    return sh


def direct(concept: str, title: str | None = None):
    """Concept -> an ORIGINAL movie script (the model writes + directs). Always
    returns a playable spec (deterministic fallback if the model is off/garbled)."""
    ref = craft_kb.brief(concept)                  # cross-reference the craft KB while writing
    want = _mentioned(concept)                      # the concrete drawable subjects named in the concept
    leads = ""
    if len(want) >= 2:                              # a relationship concept — both must be on screen
        leads = (f"\nLEADS: this story is about {', '.join(want)} — ALL of them must appear on screen "
                 f"(in the cast) and interact; do not drop any. Name them concretely, never 'they'/'it'.\n")
    user_msg = (f"CRAFT REFERENCE (apply this 3-act structure and craft):\n{ref}\n{leads}\n"
                f"CONCEPT: {concept}\n\nWrite the original ~90-second film (all 9 beats) as JSON.")
    # the endpoint scales to zero; a cold start can exceed one timeout. Retry with a long
    # read timeout so a real, original script comes back instead of the generic fallback.
    raw = ""
    for attempt in range(3):
        try:
            raw = llm(DIRECTOR_SYS, user_msg, max_tokens=1700, temperature=0.7, timeout=170)
        except Exception:
            raw = ""
        if raw:
            break
        time.sleep(2)
    # parse the whole object if it gives a real film; else salvage individual shots
    # (the small model often truncates at the token cap or mangles inner quotes)
    def _shots(d):
        s = d.get("shots") if isinstance(d, dict) else None
        return [x for x in s if isinstance(x, dict)] if isinstance(s, list) else []
    spec = _extract_json(raw) if raw else None
    if len(_shots(spec)) < 6 and raw:                  # truncated / mangled -> salvage
        spec = _salvage(raw) or spec
    shots = _shots(spec)[:11]                           # ~90s ceiling; drop any overflow
    if len(shots) >= 6:
        spec["shots"] = [_sanitize_shot(sh, concept) for sh in shots]
        # guarantee every subject named in the concept actually gets drawn: if the model dropped
        # one (e.g. wrote a cat story and forgot the dog), bring it on from the first act's end so
        # the relationship pays off on screen.
        present = {c for sh in spec["shots"] for c in sh["cast"]}
        missing = [w for w in want if w not in present]
        if missing:
            for i, sh in enumerate(spec["shots"]):
                if i >= len(spec["shots"]) // 3:
                    for w in missing:
                        if w not in sh["cast"] and len(sh["cast"]) < 4:
                            sh["cast"].append(w)
        spec["title"] = title or spec.get("title") or _clip(concept, 30)
        spec.setdefault("logline", concept)
        return spec
    # deterministic ~90-second fallback: a coherent 9-beat arc around the concept
    cast = want or _detect(concept)
    hero = (cast[0] if cast else "hero")
    beats = [
        ("rise",   f"A small {hero} wakes in a world that feels a little too quiet."),
        ("gather", f"More than anything, the {hero} wants what lies just out of reach."),
        ("enter",  "Then one ordinary morning, something arrives that changes everything."),
        ("rise",   f"Heart pounding, the {hero} leaves the safe and known behind."),
        ("gather", "The road gives back as good as it takes — wonder, and warning."),
        ("gather", "Halfway through, the ground shifts and the easy plan falls apart."),
        ("exit",   "At the lowest turn, it all seems lost, and the dark leans close."),
        ("rise",   f"So the {hero} digs deep and makes the one brave, costly choice."),
        ("gather", "And the world, changed for good, is quietly never the same again."),
    ]
    return {"title": title or _clip(concept, 30), "logline": concept,
            "shots": [{"narration": n, "cast": cast, "action": a, "dialogue": "",
                       "setting": floor_kind(n, cast)} for a, n in beats]}


def save_movie(spec: MovieSpec) -> str:
    """Save the movie spec, TIMESTAMPED so we never overwrite an earlier cut."""
    path = os.path.join(saved_dir(), stamped_slug(spec["title"]) + ".json")
    with open(path, "w") as f:
        json.dump(spec, f, indent=2)
    return path


# ── choose-your-own-adventure: an interactive, branching film ─────────────────
ADVENTURE_SYS = (
    "You are the director of an INTERACTIVE branching ASCII film. Tell it in CHUNKS of 3-4 shots. "
    "Continue naturally from the story-so-far and the viewer's chosen direction. Original only.\n\n"
    "Each chunk: 3-4 vivid shots (same craft + format as a normal film), then EITHER stop on a "
    "cliffhanger / decision point with 2-3 DISTINCT choices the viewer picks, OR — if the story has "
    "reached a satisfying, earned ending — finish it (set \"ending\":true, \"choices\":[]). Aim to end "
    "within about 4-6 chunks; don't drag.\n\n"
    "Each shot: " + '{"narration":"60-90 chars","cast":[1-3 nouns],"action":"' + "|".join(ACTIONS) + '",'
    '"setting":"' + "|".join(SETTINGS) + '","camera":"' + "|".join(CAMERAS) + '","mood":[per cast],'
    '"dialogue":"optional"}.\n'
    "OUTPUT — strict JSON only:\n"
    '{"title":"only on the first chunk","shots":[...3-4 shots...],'
    '"choices":[{"key":"a","label":"a short option, <=40 chars"},{"key":"b","label":"..."}],'
    '"ending":false}\n'
    "Keep the SAME characters across chunks so the viewer can follow them.")


def direct_branch(concept: str, history: str = "", choice: str = "") -> dict:
    """One chunk of an interactive film. `history` is the story so far; `choice` is
    what the viewer just picked (a label or their own typed direction). Returns
    {title?, shots:[...], choices:[{key,label}], ending:bool}."""
    if not history:
        user = f"CONCEPT: {concept}\n\nBegin the adventure: write the FIRST chunk."
    else:
        user = (f"CONCEPT: {concept}\n\nSTORY SO FAR:\n{history}\n\n"
                f"THE VIEWER CHOSE: {choice}\n\nContinue: write the NEXT chunk.")
    # adventure may use a separate, more general endpoint (CLAUDEMOVIES_ADV_*); branching
    # is poor on the movie-only fine-tune. Falls back to the main endpoint if ADV is unset.
    raw = llm(ADVENTURE_SYS, user, max_tokens=900, temperature=0.9, prefix="CLAUDEMOVIES_ADV")
    data = _extract_json(raw) if raw else None
    if data and isinstance(data.get("shots"), list) and any(isinstance(s, dict) for s in data["shots"]):
        shots = [_sanitize_shot(sh, concept) for sh in data["shots"] if isinstance(sh, dict)][:4]
        choices = []
        if isinstance(data.get("choices"), list):
            for i, ch in enumerate(data["choices"][:3]):
                label = str(ch.get("label", "") if isinstance(ch, dict) else ch)[:40].strip()
                if label:
                    choices.append({"key": "abc"[i], "label": label})
        ending = bool(data.get("ending"))
        if len(history) < 500:                       # never end in the first chunk or two
            ending = False
        if not ending and not choices:               # keep it interactive even if the model gave none
            choices = [{"key": "a", "label": "press deeper into the unknown"},
                       {"key": "b", "label": "turn back the way you came"}]
        out = {"shots": shots, "choices": [] if ending else choices, "ending": ending}
        if not history:
            out["title"] = str(data.get("title") or concept[:30])
        return out
    # deterministic fallback (model off/garbled): one beat + a couple of generic forks
    cast = _detect(choice or concept)
    n = (choice or concept)[:90] or "The story continues into the unknown."
    ending = bool(history) and len(history) > 600           # wrap up once it has run a while
    return {
        "shots": [{"narration": n, "cast": cast, "action": "enter", "dialogue": "",
                   "setting": floor_kind(n, cast), "camera": "static", "mood": []}],
        "choices": [] if ending else [{"key": "a", "label": "press on bravely"},
                                      {"key": "b", "label": "turn back"}],
        "ending": ending,
        **({"title": concept[:30]} if not history else {}),
    }


# ── the player (Claudecade engine) ────────────────────────────────────────────
def play(spec):
    import curses
    from claudcade_engine import Engine, Scene, Renderer, Input, AnimSprite, Color  # noqa
    import draw

    order = appearance_order(spec)              # stable left-to-right character order

    class Movie(Scene):
        def on_enter(self):
            self.shots = spec["shots"]
            self.i, self.prev = 0, set()
            self._load()

        def _load(self):
            sh = self.shots[self.i]
            self.action = sh["action"]
            self.cast = sh["cast"]
            # any character — library or freshly drawn (cached for consistency)
            self.actors = [(n, AnimSprite({"idle": [draw.draw_sprite(n)]}, ticks_per_frame=8))
                           for n in self.cast]
            self.narr = sh["narration"]
            self.dialogue = sh.get("dialogue", "")
            self.setting = sh.get("setting")
            self.t = 0
            mv, hd = shot_budget(sh)                        # same per-shot budget as the gif/stage
            self.limit = int((mv + hd) * (FRAME_MS / 1000) * ENGINE_FPS)  # frames -> engine ticks

        def update(self, inp: Input, tick: int, dt: float):
            if inp.pause:
                return "quit"
            self.t += 1
            for _, a in self.actors:
                a.tick()
            if inp.confirm or self.t > self.limit:         # next shot (hold so text is readable)
                self.prev = set(self.cast)                 # everyone here is now "already on stage"
                self.i += 1
                if self.i >= len(self.shots):
                    return "quit"
                self._load()
            return None

        def draw(self, r: Renderer, tick: int):
            H, W = self.engine.H, self.engine.W
            homes = home_columns(self.cast, W, order)
            r.header(spec["title"], right="ENTER skip · ESC quit", color=Color.CYAN)
            r.outer_border(Color.BLUE)
            ground = H - 5
            prof = FLOOR.get(floor_kind(self.narr, self.cast, self.setting))
            if prof:                                       # textured floor under the cast
                surf = prof["surf"]
                shift = tick if prof.get("anim") else 0
                line = "".join(surf[(c + shift) % len(surf)] for c in range(W - 2))
                col = getattr(Color, prof["tcol"], Color.WHITE)
                r.sprite(ground, 1, [line], col)
                r.sprite(ground + 1, 1, [prof["fill"] * (W - 2)], col)
            try:                                           # set-dressing behind the cast
                plan = scenery(self.shots[self.i],
                               floor_kind(self.narr, self.cast, self.setting))
                for i in range(plan["stars"]):
                    r.sprite(1 + i % 2, (5 + i * 11) % (W - 2) + 1, ["*"], Color.YELLOW)
                if plan["sky"]:
                    art = draw.draw_sprite(plan["sky"])
                    r.sprite(1, W - len(art[0]) - 2, art, getattr(Color, sprite_curses(plan["sky"]), Color.YELLOW))
                if plan["clouds"]:
                    cl = draw.draw_sprite("cloud")
                    for i in range(plan["clouds"]):
                        r.sprite(1, (i * 24 + tick // 2) % (W + 12) - 10, cl, Color.WHITE)
                for label, frac in plan["ground"]:
                    art = draw.draw_sprite(label)
                    r.sprite(ground - len(art), int(frac * (W - 1)) - len(art[0]) // 2, art,
                             getattr(Color, sprite_curses(label), Color.GREEN))
            except curses.error:                           # only tolerate drawing past the screen edge
                pass
            e = ease(min(1.0, self.t / 30))                # smoothstep entrance progress
            for name, a in self.actors:
                frame = a.current()
                col = getattr(Color, sprite_curses(name), Color.WHITE)
                home, base = homes.get(name, W // 2), ground - len(frame)
                if self.action == "exit":                  # leaving: drift off to the side
                    row, x = base, int(home + e * (W - 2 - home))
                elif name in self.prev:                    # carried over -> LOCKED, no re-entry
                    row, x = base, home
                else:                                      # new: always slide in from the nearer SIDE
                    start = -len(frame[0]) - 1 if home < W // 2 else W + 1
                    row, x = base, int(start + e * (home - start))
                r.sprite(row, x, frame, col)
            if self.dialogue:
                r.center(H - 3, "“" + self.dialogue + "”", Color.YELLOW)
            shown = self.narr[:int(len(self.narr) * min(1.0, self.t / 24)) + 1]
            r.center(H - 2, shown, Color.WHITE)

    Engine(spec["title"][:24], fps=ENGINE_FPS).scene("movie", Movie()).run("movie")


# ── cli ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", help="your idea — the model writes an ORIGINAL movie from it")
    ap.add_argument("--play", help="replay a saved movie JSON")
    ap.add_argument("--dry", action="store_true", help="print the script, don't open the terminal")
    a = ap.parse_args()

    if a.play:
        spec = json.load(open(a.play))
    elif a.concept:
        spec = direct(a.concept)
    else:
        ap.error("give --concept or --play")

    if not a.play:
        print("saved:", save_movie(spec))
    if a.dry:
        print(json.dumps(spec, indent=2))
    else:
        play(spec)


if __name__ == "__main__":
    main()
