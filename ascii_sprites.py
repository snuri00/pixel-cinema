"""
ASCII drawing knowledge base — labeled, hand-authored sprites in one consistent
small style. Triple duty:
  1. TRAINING DATA for the "draw" skill (label/description -> ASCII art).
  2. CACHE/library the director reuses so a character looks the same every shot.
  3. STYLE REFERENCE the model cross-references when drawing a NEW character.

Style rules (kept consistent so the model learns a coherent hand):
  • <= 7 rows, <= 14 columns, pure ASCII, front-facing, sits on a ground line.

The big animal/character set lives in sprites.txt (plain text so backslashes are
literal) and is merged in at import; add new art there, no Python escaping needed.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from schema import Sprite

# label -> list of lines (the canonical drawing)
SPRITES = {
    "person":  ["  (o) ", "  /|\\ ", "  / \\ "],
    "child":   ["  (o)", "  /|\\", "  / \\"],
    "knight":  [".[+].", "(o_o)", "/|#|\\", "_/ \\_"],
    "king":    [" \\.W./ ", "  (o)  ", "  /|\\  ", "  / \\  "],
    "wizard":  ["   /\\  ", "  (..) ", "  /||\\ ", " ~/  \\~"],
    "robot":   [" [o_o]", " /[ ]\\", "  | | "],
    "cat":     [" /\\_/\\", "( o.o )", " > ^ < "],
    "dog":     [" /^ ^\\", "( o.o )", " (___) ", " /   \\"],
    "rabbit":  [" (\\_/)", " (o.o)", ' ("")("")'],
    "mouse":   ["  oo  ", " (..) ", " /~~\\ "],
    "fox":     [" /\\_/\\", "(>'.'<)", " ^^ ^^ "],
    "bird":    [" _    ", "<o \\  ", " \\__> "],
    "owl":     [" {O,O}", " |)_)|", "  ^ ^ "],
    "fish":    ["  ><>  ", "><(((*> "],
    "dragon":  ["  /\\__/\\ ", " ( o..o )", "  )vvvv( ", " /|    |\\"],
    "monster": [" ,,,,, ", "(o   o)", "( === )", "/|   |\\"],
    "ghost":   [" .---. ", "( o o )", "(  ~  )", " '|||' "],
    "tree":    ["  ###  ", " ##### ", "#####. ", "   ||  "],
    "flower":  ["  @  ", " \\|/ ", "  |  "],
    "mountain": ["   /\\   ", "  /  \\  ", " /    \\ ", "/______\\"],
    "moon":    [" _.._ ", "( (  )", " `--' "],
    "sun":     [" \\ | / ", "- (#) -", " / | \\ "],
    "star":    ["  .  ", " ./|\\.", "  \\|/ ", "  '  "],
    "cloud":   [" .--.  ", "(    )) ", " `--'  "],
    "house":   ["  /\\  ", " /__\\ ", " |[]| ", " |__| "],
    "castle":  ["|^|^|^|", "|     |", "|_| |_|"],
    "door":    [" +--+ ", " |  | ", " | o| ", " +--+ "],
    "ship":    ["  |   ", " /|\\  ", "_|_|_ ", "\\___/ "],
    "key":     [" o==-"],
    "book":    [" ____ ", "|    |", "| == |", "|____|"],
    "sword":   ["  /\\ ", "  || ", " <##>", "  || "],
    "candle":  ["  (  ", "  )  ", "  |  ", " |_| "],
    "crown":   [" \\o^o^o/", " \\___/ "],
    "clock":   [" .--. ", "|10\\ |", "|  '/|", " `--' "],
}

def _load_txt(path: str) -> dict[str, Sprite]:
    """Parse sprites.txt: '@label' lines start a sprite; following lines are its art."""
    out: dict[str, Sprite] = {}
    name: str | None = None
    lines: list[str] = []
    try:
        raw = open(path, encoding="utf-8").read().splitlines()
    except OSError as e:                                 # the library ships with the repo; its absence is notable
        print(f"ascii_sprites: could not read {path}: {e}", file=sys.stderr)
        return out
    for ln in raw:
        if ln.startswith("# "):                         # a '# ' comment line (never art)
            continue
        if ln.startswith("@"):
            if name and lines:
                out[name] = lines
            name, lines = ln[1:].strip().lower(), []
        elif name is not None and ln.strip():           # art line (keep leading spaces)
            lines.append(ln[:14])                       # clamp to house-style width
    if name and lines:
        out[name] = lines
    return out


# merge the big text library (animals + characters); inline art above wins on conflict
for _n, _art in _load_txt(os.path.join(os.path.dirname(__file__), "sprites.txt")).items():
    SPRITES.setdefault(_n, _art[:7])


# alternate names that map to the same drawing (so labels generalize)
ALIASES = {
    "man": "person", "woman": "person", "girl": "child", "boy": "child", "kid": "child",
    "hero": "knight", "soldier": "knight", "queen": "king", "ruler": "king",
    "mage": "wizard", "sorcerer": "wizard", "android": "robot", "machine": "robot",
    "kitten": "cat", "puppy": "dog", "hound": "dog", "bunny": "rabbit", "hare": "rabbit",
    "rat": "mouse", "raven": "bird", "crow": "bird", "beast": "monster", "creature": "monster",
    "spirit": "ghost", "phantom": "ghost", "forest": "tree", "rose": "flower", "bloom": "flower",
    "hill": "mountain", "palace": "castle", "tower": "castle", "boat": "ship", "vessel": "ship",
    "gate": "door", "blade": "sword", "lantern": "candle",
    "shadow": "monster", "shade": "monster", "figure": "monster",
    "lizard": "frog", "gecko": "frog", "newt": "frog",
    "human": "person", "humans": "person", "people": "person", "guy": "person", "lady": "person",
    "villager": "person", "folk": "person", "stranger": "person", "men": "person", "women": "person",
    "boys": "child", "girls": "child", "kids": "child", "children": "child",
    "fire": "campfire", "bonfire": "campfire", "fireplace": "campfire", "hearth": "campfire",
    "flame": "campfire", "flames": "campfire", "fires": "campfire", "embers": "campfire",
    # animals
    "lioness": "lion", "cub": "lion", "kitty": "cat", "tabby": "cat",
    "grizzly": "bear", "cubbear": "bear", "wolves": "wolf", "vixen": "fox",
    "stag": "deer", "doe": "deer", "pony": "horse", "mare": "horse", "stallion": "horse",
    "calf": "cow", "ox": "cow", "bull": "cow", "piglet": "pig", "hog": "pig",
    "lamb": "sheep", "ewe": "sheep", "ram": "goat", "billy": "goat",
    "toad": "frog", "tadpole": "frog", "tortoise": "turtle", "serpent": "snake", "cobra": "snake",
    "bumblebee": "bee", "moth": "butterfly", "beetle": "ladybug",
    "tarantula": "spider", "lobster": "crab", "squid": "octopus",
    "orca": "whale", "porpoise": "dolphin", "duckling": "duck", "drake": "duck",
    "hen": "chicken", "rooster": "chicken", "macaw": "parrot", "cygnet": "swan",
    "porcupine": "hedgehog", "monkeys": "monkey", "ape": "monkey",
    "trunk": "elephant", "tusker": "elephant",
    # characters
    "buccaneer": "pirate", "corsair": "pirate", "assassin": "ninja", "shinobi": "ninja",
    "hag": "witch", "sorceress": "witch", "pixie": "fairy", "sprite": "fairy",
    "seraph": "angel", "cherub": "angel", "skull": "skeleton", "bones": "skeleton",
    "undead": "zombie", "dracula": "vampire", "nosferatu": "vampire",
    "martian": "alien", "ufo": "alien", "spaceman": "astronaut", "cosmonaut": "astronaut",
    "cook": "chef", "jester": "clown", "fool": "clown", "norseman": "viking",
    "warrior": "samurai", "ronin": "samurai", "siren": "mermaid",
    "djinn": "genie", "wish": "genie", "snowperson": "snowman",
    "imp": "devil", "demon2": "devil", "sprite2": "elf", "gnome": "dwarf",
    "ogre": "troll", "giant": "troll", "farmhand": "farmer",
    # scenery
    "cacti": "cactus", "saguaro": "cactus", "evergreen": "pine", "fir": "pine",
    "spruce": "pine", "shrub": "bush", "hedge": "bush", "boulder": "rock",
    "oak": "bigtree", "pinetree": "pine",
}


def style_reference(n: int = 6) -> str:
    """A few labeled examples to show the model the house style when drawing new art."""
    sample = list(SPRITES.items())[:n]
    return "\n\n".join(f"{name}:\n" + "\n".join(lines) for name, lines in sample)


def get(label: str | None) -> Sprite | None:
    label = (label or "").strip().lower()
    if label in SPRITES:
        return SPRITES[label]
    if label in ALIASES:
        return SPRITES[ALIASES[label]]
    return None
