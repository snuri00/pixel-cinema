"""
pixelscene.py — render a film spec as an animated PIXEL scene (the Level-1 alternative
to the ASCII stage). Same MovieSpec the director already produces (title, logline,
shots[ narration, cast, action, setting, camera, mood, dialogue ]).

The stage is a theatre, not a slideshow: characters CHOREOGRAPH their shot action
(enter walks in from off-screen, exit leaves, chase pursues, rise floats...), the
camera pans / pushes / shakes over an oversized world canvas, and the environment
lives — parallax silhouette layers, drifting clouds, stars and a moon at night,
weather particles (rain / snow / embers / fireflies / dust), torch flicker indoors,
mood emotes and speech bubbles over the cast — with a subtitle bar and a THE END card.

Nothing here touches the ASCII engine (render.py / stage.py / server_app.py). It reuses
ONLY the leaf asset modules, so the two render paths live side by side.

    python pixelscene.py                       # render showcase/knight.json -> *_pixel.gif
    python pixelscene.py showcase/ghost.json
    python pixelscene.py --concept "a raccoon who runs a midnight noodle stand"   # needs the model
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import random

from PIL import Image, ImageDraw, ImageEnhance

import pixelsprites as PS
import wfc

W, H = 320, 180
MX, MY = 32, 18
BW, BH = W + 2 * MX, H + 2 * MY
GROUND_Y = int(H * 0.64)
WGROUND = MY + GROUND_Y

SKY = {
    "lava":   ((60, 20, 28), (150, 60, 40)),
    "snow":   ((150, 170, 205), (215, 228, 240)),
    "water":  ((70, 120, 175), (150, 195, 225)),
    "sky":    ((70, 130, 200), (180, 215, 240)),
    "stone":  ((60, 66, 86), (120, 130, 150)),
    "indoor": ((40, 34, 46), (78, 64, 74)),
    "grass":  ((88, 148, 210), (190, 220, 240)),
    "forest": ((70, 118, 170), (150, 190, 205)),
    "sand":   ((120, 160, 215), (235, 215, 170)),
    "road":   ((95, 120, 160), (180, 195, 215)),
}
SKY_DEFAULT = ((78, 132, 196), (188, 214, 236))

_SIL_STYLE = {"stone": "peaks", "snow": "peaks", "earth": "hills", "grass": "hills",
              "forest": "trees", "sand": "dunes", "water": "isles", "road": "roofs",
              "lava": "spikes"}


def _seed(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def _variant(spec: dict) -> int:
    return _seed("film:" + (spec.get("title") or "untitled")) % 3


def _cast_variant(spec: dict) -> int:
    """A per-film sprite look: the same subject stays consistent within one film but
    gets a different design in the next (0 would mean the canonical cached sprite)."""
    return _seed("cast:" + (spec.get("title") or "untitled"))


def _ease(p: float) -> float:
    return p * p * (3 - 2 * p)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _mix(a, b, t: float):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _night(c):
    return (int(c[0] * 0.30) + 5, int(c[1] * 0.32) + 8, int(c[2] * 0.48) + 22)


def _shot_flags(shot: dict, setting: str) -> dict:
    """Read the shot's text once and derive lighting + weather for the whole stage:
    brightness, warm/cold tint, night sky, rain/snow/ember particles, torch flicker."""
    text = ((shot.get("narration") or "") + " " + (shot.get("dialogue") or "")).lower()

    def has(*ws):
        return any(w in text for w in ws)

    night = has("night", "midnight", "moonlit", "moonlight", " moon", "starlit",
                "stars", "nightfall")
    dark = has("dark", "shadow", "dim", "cave", "gloom", "black")
    bright = has("bright", "sunny", "noon", "daylight", "blaze", "glow")
    b = 0.62 if dark else (0.72 if night else (1.12 if bright else 1.0))
    tint = None
    if setting == "lava" or has("dawn", "dusk", "sunset", "sunrise", "ember", "warm", "fire"):
        tint = (1.12, 1.0, 0.82)
    elif has("cold", "frost", "ice", "snow", "winter", "frozen") or (night and setting == "snow"):
        tint = (0.86, 0.94, 1.14)
    rain = has("rain", "storm", "drizzle", "pour", "downpour")
    snowfx = setting == "snow" or has("snow", "blizzard")
    embers = setting == "lava" or has("ember", "campfire", "bonfire", "furnace")
    flicker = embers or setting == "indoor" or has("torch", "candle", "fire")
    return dict(night=night, b=b, tint=tint, rain=rain, snow=snowfx,
                embers=embers, flicker=flicker)


def _apply_grade(img: Image.Image, b: float, tint) -> Image.Image:
    rgba = img if img.mode == "RGBA" else img.convert("RGBA")
    if b == 1.0 and tint is None:
        return rgba
    a = rgba.getchannel("A")
    rgb = rgba.convert("RGB")
    if b != 1.0:
        rgb = ImageEnhance.Brightness(rgb).enhance(b)
    if tint:
        r, g, bl = rgb.split()
        r = r.point(lambda v: min(255, int(v * tint[0])))
        g = g.point(lambda v: min(255, int(v * tint[1])))
        bl = bl.point(lambda v: min(255, int(v * tint[2])))
        rgb = Image.merge("RGB", (r, g, bl))
    out = rgb.convert("RGBA")
    out.putalpha(a)
    return out


_SKY_CACHE: dict = {}


def _sky_img(setting: str, night: bool) -> Image.Image:
    """The gradient sky as a 1px column stretched to the canvas (fast) — cached."""
    key = (setting, night)
    if key not in _SKY_CACHE:
        top, bot = SKY.get(setting, SKY_DEFAULT)
        if night:
            top, bot = _night(top), _night(bot)
        horizon = BH if setting == "sky" else WGROUND
        col = Image.new("RGB", (1, BH))
        for y in range(BH):
            t = min(1.0, y / max(1, horizon))
            col.putpixel((0, y), tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3)))
        _SKY_CACHE[key] = col.resize((BW, BH), Image.NEAREST).convert("RGBA")
    return _SKY_CACHE[key].copy()


def _celestial(img: Image.Image, setting: str, night: bool, variant: int):
    if setting in ("indoor", "lava"):
        return
    d = ImageDraw.Draw(img)
    x = BW - 78 - (variant * 23) % 40
    y = 24
    if night:
        d.ellipse([x, y, x + 24, y + 24], fill=(226, 232, 244, 255))
        cover = img.getpixel((min(BW - 1, x + 34), y + 2))
        d.ellipse([x + 7, y - 4, x + 29, y + 18], fill=cover)
    else:
        disc = (240, 246, 255) if setting == "snow" else (255, 244, 210)
        d.ellipse([x, y, x + 26, y + 26], fill=disc + (255,))


def _sil_heights(style: str, w: int, hmax: int, rng, amp: float) -> list[int]:
    hs = [0] * w
    if style == "peaks":
        h = rng.randint(10, 22)
        for x in range(w):
            h += rng.randint(-3, 3)
            if rng.random() < 0.02:
                h += rng.randint(6, 14) * rng.choice((-1, 1))
            h = max(4, min(hmax - 2, h))
            hs[x] = h
    elif style == "hills":
        p1, p2 = rng.uniform(0, 9), rng.uniform(0, 9)
        w1, w2 = rng.uniform(40, 90), rng.uniform(18, 34)
        base = rng.randint(8, 12)
        for x in range(w):
            hs[x] = max(2, int(base + 6 * math.sin(x / w1 + p1) + 3 * math.sin(x / w2 + p2)))
    elif style == "trees":
        x = 0
        while x < w:
            seg = rng.randint(5, 12)
            th = rng.randint(8, 18) if rng.random() > 0.15 else rng.randint(2, 4)
            for i in range(min(seg, w - x)):
                hs[x + i] = max(1, th + rng.randint(-2, 1))
            x += seg
    elif style == "dunes":
        p1 = rng.uniform(0, 9)
        w1 = rng.uniform(50, 110)
        base = rng.randint(5, 9)
        for x in range(w):
            hs[x] = max(1, int(base + 5 * math.sin(x / w1 + p1) + 2 * math.sin(x / 23 + p1 * 2)))
    elif style == "spikes":
        x = 0
        while x < w:
            if rng.random() < 0.12:
                sw, sh = rng.randint(2, 5), rng.randint(14, 30)
                for i in range(min(sw, w - x)):
                    hs[x + i] = max(3, int(sh * (1 - abs(i - sw / 2) / (sw / 2 + 0.01))))
                x += sw
            else:
                hs[x] = rng.randint(2, 5)
                x += 1
    elif style == "roofs":
        x = 0
        while x < w:
            seg = rng.randint(12, 30)
            th = rng.randint(6, 20)
            for i in range(min(seg, w - x)):
                hs[x + i] = th
            if rng.random() < 0.5:
                for i in range(2):
                    if x + seg - 4 + i < w:
                        hs[x + seg - 4 + i] = th + 5
            x += seg
    elif style == "isles":
        x = 0
        while x < w:
            if rng.random() < 0.05:
                iw, ih = rng.randint(8, 18), rng.randint(3, 7)
                for i in range(min(iw, w - x)):
                    hs[x + i] = max(0, int(ih * math.sin(math.pi * i / iw)))
                x += iw + rng.randint(10, 60)
            else:
                x += 1
    return [int(h * amp) for h in hs]


def _sil_layer(setting: str, seed: int, near: bool, flags: dict):
    """One parallax silhouette strip (distant mountains / treeline / rooftops...)
    sitting on the horizon, wider than the canvas so the camera can pan across it."""
    style = _SIL_STYLE.get(setting)
    if not style:
        return None
    rng = random.Random(seed)
    hmax, wpx = 52, BW + 80
    hs = _sil_heights(style, wpx, hmax, rng, 1.0 if near else 0.62)
    _, bot = SKY.get(setting, SKY_DEFAULT)
    gbase = wfc.GROUND.get(setting, wfc.GROUND["earth"])[0]
    dark = tuple(int(g * 0.5 + d * 0.5) for g, d in zip(gbase, (30, 34, 52)))
    col = _mix(bot, dark, 0.72 if near else 0.45)
    if flags["night"]:
        col = tuple(int(c * 0.5) for c in col)
    img = Image.new("RGBA", (wpx, hmax), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for x, hh in enumerate(hs):
        if hh > 0:
            d.line([(x, hmax - hh), (x, hmax - 1)], fill=col + (255,))
    return img


def _cloud_defs(seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    return [dict(x=rng.uniform(0, BW), y=rng.uniform(8, 52), s=rng.uniform(0.7, 1.5),
                 v=rng.uniform(0.12, 0.3)) for _ in range(n)]


def _draw_clouds(frame: Image.Image, defs, k: int, night: bool, shift: float):
    ov = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    col = (176, 186, 208, 44) if night else (236, 242, 248, 60)
    span = BW + 140
    for c in defs:
        cx = (c["x"] + k * c["v"]) % span - 70 + shift
        cy, s = c["y"], c["s"]
        d.ellipse([cx - 16 * s, cy - 5 * s, cx + 16 * s, cy + 5 * s], fill=col)
        d.ellipse([cx - 7 * s, cy - 9 * s, cx + 6 * s, cy + 2 * s], fill=col)
        d.ellipse([cx + 2 * s, cy - 7 * s, cx + 14 * s, cy + 3 * s], fill=col)
    frame.alpha_composite(ov)


def _draw_stars(frame: Image.Image, seed: int, k: int):
    rng = random.Random(seed)
    px = frame.load()
    for j in range(70):
        x, y = rng.randint(0, BW - 1), rng.randint(0, max(1, WGROUND - 16))
        lv = (150, 200, 245)[(j + k // 2) % 3]
        px[x, y] = (lv, lv, min(255, lv + 10), 255)


_GCACHE: dict = {}


def _ground_img(setting: str, seed: int, flags: dict) -> Image.Image:
    key = (setting, seed, flags["b"], flags["tint"])
    if key not in _GCACHE:
        if len(_GCACHE) > 24:
            _GCACHE.clear()
        g = wfc.render_ground(setting, BW, BH - WGROUND, seed=seed, tile=8)
        _GCACHE[key] = _apply_grade(g, flags["b"], flags["tint"])
    return _GCACHE[key]


def _build_stage(shot: dict, index: int, variant: int) -> dict:
    """Everything static for the shot: graded sky + celestial + ground, the parallax
    silhouette layers and cloud drift definitions (SD backgrounds replace all of it)."""
    setting = shot.get("setting", "earth")
    flags = _shot_flags(shot, setting)
    stage = dict(setting=setting, flags=flags, sd=False, ground=None, sils=(),
                 clouds=(), star_seed=0)
    if os.environ.get("CLAUDEMOVIES_BG_BACKEND") == "sd":
        try:
            import sdpixel
            base = sdpixel.background_sd(setting, w=BW, h=BH, variant=variant).convert("RGBA")
            stage["base"] = _apply_grade(base, flags["b"], flags["tint"])
            stage["sd"] = True
            return stage
        except Exception:
            pass
    base = _sky_img(setting, flags["night"])
    _celestial(base, setting, flags["night"], variant)
    stage["base"] = _apply_grade(base, flags["b"], flags["tint"])
    seed = _seed(f"{variant}:{setting}")
    stage["star_seed"] = seed + 4
    if setting != "sky":
        stage["ground"] = _ground_img(setting, seed, flags)
    sils = []
    for off, near, f in ((1, False, 0.4), (2, True, 0.7)):
        sil = _sil_layer(setting, seed + off, near, flags)
        if sil is not None:
            sils.append((_apply_grade(sil, flags["b"], flags["tint"]), f))
    stage["sils"] = tuple(sils)
    if setting not in ("indoor", "lava"):
        stage["clouds"] = _cloud_defs(seed + 3, 4 if setting == "sky" else 3)
    return stage


def _target_px(label: str) -> int:
    s = PS.asset_spec(label).get("scale", 0.4)
    return int(round(36 + max(0.1, min(1.0, s)) * 78))


def _cells_img(cells, target_h: int):
    h, w = len(cells), len(cells[0])
    base = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    px = base.load()
    for cy in range(h):
        for cx in range(w):
            if cells[cy][cx] is not None:
                px[cx, cy] = cells[cy][cx] + (255,)
    tw = max(1, round(w * target_h / max(1, h)))
    return tw, target_h, base.resize((tw, target_h), Image.NEAREST)


def _cast_setup(shot: dict, cast_variant: int = 0) -> list[dict]:
    """Pre-render every cast member's idle AND walk cycles (both facings) at their
    fitted on-screen size, so the per-tick loop only pastes cached images."""
    cast = [c for c in (shot.get("cast") or [])][:3] or ["hero"]
    moods = shot.get("mood") if isinstance(shot.get("mood"), list) else []
    idle = [PS.anim_cells_for(c, mode="idle", frames=6, variant=cast_variant) for c in cast]
    walk = [PS.anim_cells_for(c, mode="walk", frames=6, variant=cast_variant) for c in cast]
    n = len(cast)
    targets = [_target_px(c) for c in cast]
    dims = [(len(a[0][0]), len(a[0])) for a in idle]
    maxh = GROUND_Y - 6
    fit = 1.0
    widths = [dims[i][0] * targets[i] / max(1, dims[i][1]) for i in range(n)]
    for i in range(n):
        if targets[i] > maxh:
            fit = max(fit, targets[i] / maxh)
    if sum(widths) + 10 * (n + 1) > W:
        fit = max(fit, (sum(widths) + 10 * (n + 1)) / W)
    if fit > 1.0:
        targets = [max(20, int(t / fit)) for t in targets]
    members = []
    for i, c in enumerate(cast):
        frames = {}
        for mode, anim in (("idle", idle[i]), ("walk", walk[i])):
            fr = []
            for cells in anim:
                tw, th, img = _cells_img(cells, targets[i])
                fr.append((tw, th, img, img.transpose(Image.FLIP_LEFT_RIGHT)))
            frames[mode] = fr
        mood = str(moods[i] if i < len(moods) else "").strip().lower()
        members.append(dict(label=c, frames=frames, mood=mood))
    return members


def _choreo(action: str, i: int, n: int, p: float, direction: int):
    """Screen-space blocking for cast member i at shot progress p in [0,1]:
    returns (x, y_offset, moving, facing). This is what turns a shot's `action`
    into actual stage movement instead of a stationary loop."""
    slot = W * (i + 1) / (n + 1)
    if action == "enter":
        start = -50.0 if direction > 0 else W + 50.0
        q = _clamp01((p - 0.06 * i) / 0.55)
        return _lerp(start, slot, _ease(q)), 0.0, q < 1.0, direction
    if action == "exit":
        end = W + 50.0 if direction > 0 else -50.0
        q = _clamp01((p - 0.18 - 0.06 * i) / 0.6)
        return _lerp(slot, end, q * q), 0.0, 0.0 < q < 1.0, direction
    if action in ("walk", "travel"):
        return slot + direction * (p - 0.5) * 0.34 * W, 0.0, True, direction
    if action == "run":
        return slot + direction * (p - 0.5) * 0.6 * W, 0.0, True, direction
    if action == "chase":
        a, b = (0.16 * W, 0.84 * W) if direction > 0 else (0.84 * W, 0.16 * W)
        lead = _lerp(a, b, p)
        if i == 0:
            return lead, 0.0, True, direction
        gap = (26.0 + 16.0 * i) * (1 - 0.4 * _ease(p))
        return lead - direction * gap, 0.0, True, direction
    if action == "flee":
        away = direction if n == 1 else (1 if slot >= W / 2 else -1)
        return slot + away * _ease(p) * 0.55 * W, 0.0, True, away
    if action == "rise":
        return slot, -20.0 * _ease(min(1.0, p * 1.3)), False, direction
    if action == "fall":
        drop = -26.0 * (1 - _ease(min(1.0, p * 1.4)))
        if p > 0.75:
            drop = -2.0 * abs(math.sin((p - 0.75) * 12))
        return slot, drop, False, direction
    if action == "gather":
        centre = W / 2
        wide = slot + (slot - centre) * 0.4
        tight = slot - (slot - centre) * 0.22
        q = _ease(_clamp01(p * 1.5))
        face = 1 if centre >= slot else -1
        return _lerp(wide, tight, q), 0.0, q < 1.0 and n > 1, face
    return slot, 0.0, False, direction


def _camera(camera: str, p: float, k: int, index: int):
    """(camx, camy, zoom) for the crop window over the oversized canvas: pan drifts
    across the world, push slowly zooms in (Ken Burns), shake jitters."""
    direction = 1 if index % 2 == 0 else -1
    if camera == "pan":
        return direction * (p - 0.5) * 48.0, 0.0, 1.0
    if camera == "push":
        return 0.0, 0.0, 1.0 + 0.13 * _ease(p)
    if camera == "shake":
        return float((-2, 0, 2, 0, -1, 1)[k % 6]), float((0, 1, 0, -1, 1, 0)[k % 6]), 1.0
    return 0.0, 0.0, 1.0


_EMOTE_MASKS = {
    "heart": (".X.X.", "XXXXX", "XXXXX", ".XXX.", "..X.."),
    "excl":  ("XX", "XX", "XX", "..", "XX"),
    "quest": (".XX.", "X..X", "..X.", "....", ".X.."),
    "drop":  (".X.", "XXX", "XXX", ".X."),
    "zzz":   ("XXX", "..X", ".X.", "X..", "XXX"),
    "note":  (".XX", ".X.", ".X.", "XX.", "XX."),
}

_EMOTE_OF: dict = {}
for _words, _key, _rgb in (
    ("happy joyful hopeful", "note", (255, 214, 92)),
    ("excited surprised shocked", "excl", (255, 214, 92)),
    ("angry mad furious", "excl", (255, 107, 107)),
    ("proud brave determined", "excl", (255, 165, 70)),
    ("sad crying lonely", "drop", (121, 192, 255)),
    ("scared afraid fear fearful terrified nervous worried anxious", "drop", (230, 237, 243)),
    ("dreamy sleepy tired", "zzz", (175, 180, 190)),
    ("weird confused", "quest", (230, 237, 243)),
    ("curious", "quest", (121, 192, 255)),
    ("love smitten", "heart", (255, 150, 190)),
):
    for _w in _words.split():
        _EMOTE_OF[_w] = (_key, _rgb)


def _draw_emote(view: Image.Image, key: str, rgb, cx: int, top: int, k: int):
    mask = _EMOTE_MASKS[key]
    s = 2
    bob = -1 if (k // 3) % 2 else 0
    x0 = int(cx - len(mask[0]) * s / 2)
    y0 = top - len(mask) * s - 3 + bob
    d = ImageDraw.Draw(view)
    for ry, row in enumerate(mask):
        for rx, ch in enumerate(row):
            if ch == "X":
                d.rectangle([x0 + rx * s, y0 + ry * s, x0 + rx * s + s - 1, y0 + ry * s + s - 1],
                            fill=rgb + (255,))


def _fold(text: str, maxlen: int) -> str:
    for a, b in (("—", "-"), ("–", "-"), ("…", "..."), ("“", '"'), ("”", '"'),
                 ("‘", "'"), ("’", "'")):
        text = text.replace(a, b)
    text = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in text)
    text = " ".join(text.split())
    if len(text) > maxlen:
        cut = text[:maxlen - 3]
        text = (cut[:cut.rfind(" ")] if " " in cut else cut).rstrip(" ,;:") + "..."
    return text


def _bubble(view: Image.Image, text: str, cx: int, top: int):
    """A little speech bubble with a tail, hovering over the speaker's head."""
    text = _fold(text, 40)
    lines, cur = [], ""
    for wd in text.split():
        if not cur or len(cur) + len(wd) + 1 <= 19:
            cur = (cur + " " + wd).strip()
        elif len(lines) < 1:
            lines.append(cur)
            cur = wd
        else:
            break
    if cur and len(lines) < 2:
        lines.append(cur)
    if not lines:
        return
    ov = Image.new("RGBA", view.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    tw = max(int(d.textlength(l)) for l in lines)
    bw, bh = tw + 10, 11 * len(lines) + 5
    bx = max(3, min(W - 3 - bw, cx - bw // 2))
    by = max(3, top - bh - 8)
    d.rounded_rectangle([bx, by, bx + bw, by + bh], radius=4,
                        fill=(243, 246, 250, 235), outline=(70, 80, 95, 255))
    tx = max(bx + 5, min(bx + bw - 5, cx))
    d.polygon([(tx - 3, by + bh), (tx + 3, by + bh), (tx, min(top - 2, by + bh + 5))],
              fill=(243, 246, 250, 235))
    for li, l in enumerate(lines):
        d.text((bx + 5, by + 3 + 11 * li), l, fill=(24, 30, 40, 255))
    view.alpha_composite(ov)


def _particles(view: Image.Image, stage: dict, k: int):
    """Deterministic per-tick weather/ambience: rain streaks, snow, rising embers,
    blinking fireflies on night meadows, drifting dust motes indoors."""
    flags, setting = stage["flags"], stage["setting"]
    fireflies = flags["night"] and setting in ("grass", "forest", "earth") and not flags["rain"]
    if not (flags["rain"] or flags["snow"] or flags["embers"] or fireflies or setting == "indoor"):
        return
    ov = Image.new("RGBA", view.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    if flags["rain"]:
        for j in range(44):
            x = (j * 97 - k * 3) % (W + 30) - 15
            y = (j * 53 + k * 11) % (H - 18)
            d.line([(x, y), (x + 2, y + 6)], fill=(170, 192, 224, 150))
    if flags["snow"]:
        for j in range(30):
            y = (j * 47 + k * 2 + (j * j) % 13) % (H - 14)
            x = (j * 83 + int(4 * math.sin((k + j * 7) / 9.0))) % W
            d.point((x, y), fill=(244, 248, 255, 210))
    if flags["embers"]:
        for j in range(18):
            y = H - 6 - ((j * 37 + k * 4) % (H - 30))
            x = (j * 71 + int(3 * math.sin((k + j) / 4.0))) % W
            if (k + j) % 5:
                d.point((x, y), fill=(255, 196, 90, 220) if j % 3 else (255, 120, 50, 220))
    if fireflies:
        for j in range(9):
            x = (j * 59 + 17) % (W - 20) + 10 + int(9 * math.sin(k / 7.0 + j))
            y = GROUND_Y - 6 - (j * 23) % 46 + int(4 * math.sin(k / 5.0 + j * 2))
            if (k + j * 3) % 8 < 5:
                d.point((x, y), fill=(216, 240, 130, 230))
    if setting == "indoor":
        for j in range(12):
            x = (j * 67 + k) % W
            y = (j * 41 + k // 2) % (H - 30)
            d.point((x, y), fill=(210, 205, 190, 60))
    view.alpha_composite(ov)


def _subtitle(scene: Image.Image, text: str):
    if not text:
        return
    text = _fold(text, 58)
    d = ImageDraw.Draw(scene)
    d.rectangle([0, H - 22, W, H], fill=(8, 10, 14, 235))
    tw = d.textlength(text)
    d.text(((W - tw) // 2, H - 16), text, fill=(225, 232, 240))


def _hold(shot: dict) -> int:
    """How many displayed frames this shot dwells — scales with text so subtitles stay
    up long enough to read (the player ticks ~150ms/frame, matching the ASCII path)."""
    chars = len(shot.get("narration", "")) + len(shot.get("dialogue", ""))
    return max(16, min(48, 14 + chars // 3))


def _shot_tick_frames(shot: dict, index: int, variant: int = 0,
                      cast_variant: int = 0) -> list[Image.Image]:
    """Every displayed frame of one shot, fully staged: choreographed cast over the
    living background, seen through the moving camera, with emotes/bubble/subtitle."""
    stage = _build_stage(shot, index, variant)
    members = _cast_setup(shot, cast_variant)
    setting, flags = stage["setting"], stage["flags"]
    n = len(members)
    action = shot.get("action") or "gather"
    camera = shot.get("camera") or "static"
    direction = 1 if index % 2 == 0 else -1
    hold = _hold(shot)
    dialogue = (shot.get("dialogue") or "").strip()
    speaker = next((i for i, m in enumerate(members) if m["mood"]), 0)

    frames = []
    for k in range(hold):
        p = k / max(1, hold - 1)
        camx, camy, z = _camera(camera, p, k, index)
        frame = stage["base"].copy()
        if not stage["sd"]:
            if flags["night"] and setting != "indoor":
                _draw_stars(frame, stage["star_seed"], k)
            for sil, f in stage["sils"]:
                frame.paste(sil, (int(-40 + camx * (1 - f)), WGROUND - sil.height), sil)
            if stage["clouds"]:
                _draw_clouds(frame, stage["clouds"], k, flags["night"], camx * 0.7)
            if stage["ground"] is not None:
                frame.paste(stage["ground"], (0, WGROUND), stage["ground"])
        shadow_ov = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(shadow_ov)
        placements = []
        for i, m in enumerate(members):
            x, yoff, moving, face = _choreo(action, i, n, p, direction)
            if setting == "sky":
                yoff += -10 + 2.5 * math.sin(k / 2.6 + i * 1.7)
            elif setting == "water":
                yoff += math.sin(k / 2.2 + i * 2.0)
            if x < -60 or x > W + 60:
                placements.append(None)
                continue
            fr = m["frames"]["walk" if moving else "idle"]
            speed = 2 if action in ("run", "chase", "flee") else 1
            fi = (k * speed if moving else k // 2) % len(fr)
            tw, th, img, img_fl = fr[fi]
            spr = img if face >= 0 else img_fl
            wx = MX + x
            foot = WGROUND + 2 + yoff
            if setting != "sky":
                lift = max(0.0, -yoff)
                shr = max(0.35, 1.0 - lift / 50.0)
                rw = max(4, int(tw * 0.42 * shr))
                rh = max(2, rw // 3)
                sdraw.ellipse([wx - rw, WGROUND + 3 - rh, wx + rw, WGROUND + 3 + rh],
                              fill=(0, 0, 0, int(95 * shr)))
            placements.append((wx, foot, tw, th, spr, i))
        frame.alpha_composite(shadow_ov)
        for pl in placements:
            if pl:
                wx, foot, tw, th, spr, _i = pl
                frame.paste(spr, (int(wx - tw / 2), int(foot - th)), spr)
        cw, ch = int(round(W / z)), int(round(H / z))
        cx0 = max(0, min(BW - cw, int(round(MX + camx + (W - cw) / 2))))
        cy0 = max(0, min(BH - ch, int(round(MY + camy + (H - ch) / 2))))
        view = frame.crop((cx0, cy0, cx0 + cw, cy0 + ch))
        if (cw, ch) != (W, H):
            view = view.resize((W, H), Image.NEAREST)
        if flags["flicker"]:
            fl = 1.0 + 0.045 * math.sin(k * 1.9 + index * 2.1) + (((k * 37 + index * 11) % 5) - 2) * 0.006
            view = ImageEnhance.Brightness(view).enhance(fl)
        _particles(view, stage, k)
        for pl in placements:
            if not pl:
                continue
            wx, foot, tw, th, spr, i = pl
            m = members[i]
            sx = int((wx - cx0) * W / cw)
            top = int((foot - th - cy0) * H / ch)
            talking = dialogue and i == speaker and 0.18 < p < 0.94
            if talking:
                _bubble(view, dialogue, sx, top)
            elif m["mood"] and 0.1 < p < 0.85:
                em = _EMOTE_OF.get(m["mood"])
                if em:
                    _draw_emote(view, em[0], em[1], sx, top - 2, k)
        _subtitle(view, shot.get("narration", ""))
        frames.append(view)
    return frames


def render_film(spec: dict, scale: int = 2, step: int = 1) -> list[Image.Image]:
    """All shots -> one list of upscaled RGBA frames (every `step`-th staged tick)."""
    variant, cvar = _variant(spec), _cast_variant(spec)
    frames: list[Image.Image] = []
    for i, shot in enumerate(spec.get("shots", [])):
        frames += _shot_tick_frames(shot, i, variant, cvar)[::max(1, step)]
    return [f.resize((W * scale, H * scale), Image.NEAREST) for f in frames]


def save_gif(frames, path: str, duration: int = 150):
    flat = [f.convert("RGB").convert("P", palette=Image.ADAPTIVE) for f in frames]
    flat[0].save(path, save_all=True, append_images=flat[1:], loop=0, duration=duration)


def _datauri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _datauri_rgba(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _tint_of(img: Image.Image):
    """Average colour + brightness 'energy' of a frame, for the reactive cinema glow."""
    r, g, b = img.convert("RGB").resize((1, 1), Image.BILINEAR).getpixel((0, 0))
    lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return [r, g, b], round(min(1.0, lum * 1.15), 3)


def _big_text(text: str, fill) -> Image.Image:
    tmp = Image.new("RGBA", (240, 24), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((2, 2), text, fill=fill)
    box = tmp.getbbox()
    if not box:
        return tmp
    t = tmp.crop(box)
    return t.resize((t.width * 2, t.height * 2), Image.NEAREST)


def _end_card(spec: dict, scale: int = 2) -> Image.Image:
    """A poster-style closing card: chunky THE END, the title, and the cast's own
    sprites lined up along the bottom like a curtain call."""
    img = Image.new("RGBA", (W, H), (10, 12, 16, 255))
    d = ImageDraw.Draw(img)
    big = _big_text("THE END", (236, 240, 246))
    img.alpha_composite(big, ((W - big.width) // 2, 42))
    title = _fold((spec.get("title") or "").upper(), 44)
    if title:
        tw = d.textlength('"' + title + '"')
        d.text(((W - tw) // 2, 42 + big.height + 10), '"' + title + '"', fill=(150, 160, 172))
    labels: list[str] = []
    for shot in spec.get("shots", []):
        for c in (shot.get("cast") or []):
            if c and c not in labels:
                labels.append(c)
    cvar = _cast_variant(spec)
    row = []
    for c in labels[:4]:
        try:
            row.append(_cells_img(PS.cells_for(c, variant=cvar), 34)[2])
        except Exception:
            pass
    if row:
        total = sum(s.width for s in row) + 14 * (len(row) - 1)
        x = (W - total) // 2
        for s in row:
            img.alpha_composite(s, (x, H - 62))
            x += s.width + 14
        cap = _fold("starring " + ", ".join(labels[:4]), 54)
        cw2 = d.textlength(cap)
        d.text(((W - cw2) // 2, H - 22), cap, fill=(120, 130, 142))
    return img if scale == 1 else img.resize((W * scale, H * scale), Image.NEAREST)


def iter_film_frames(spec: dict, scale: int = 1, end: bool = True):
    """Yield (data_uri, tint, lvl, shot_index) per displayed frame. Every tick is a
    unique staged frame (motion progresses through the shot), streamed at native
    320x180 — the browser upscales with image-rendering:pixelated, so bytes stay
    small. The closing card (shot_index == len(shots)) is skipped when end=False,
    so adventure chunks can flow into the next choice."""
    import sys
    import traceback
    variant, cvar = _variant(spec), _cast_variant(spec)
    for i, shot in enumerate(spec.get("shots", [])):
        try:
            frames = _shot_tick_frames(shot, i, variant, cvar)
        except Exception:
            print(f"pixelscene: shot {i} failed, skipping:", file=sys.stderr)
            traceback.print_exc()
            continue
        for f in frames:
            tint, lvl = _tint_of(f)
            img = f if scale == 1 else f.resize((W * scale, H * scale), Image.NEAREST)
            yield (_datauri(img), tint, lvl, i)
    if end:
        endu = _datauri(_end_card(spec, scale))
        n = len(spec.get("shots", []))
        for _ in range(20):
            yield (endu, [12, 14, 18], 0.0, n)


def iter_prewarm(spec: dict, first_only: bool = False):
    """Generate every asset the film needs, yielding one progress event per asset:
    {"kind":"set","label":...} for backgrounds and {"kind":"sprite","label":...,
    "img": <small data-uri or None>} for cast members — the frontend turns these
    into live opening-credits lines with sprite previews."""
    seen_c, seen_s = set(), set()
    bg_sd = os.environ.get("CLAUDEMOVIES_BG_BACKEND") == "sd"
    variant, cvar = _variant(spec), _cast_variant(spec)
    for shot in spec.get("shots", []):
        s = shot.get("setting", "earth")
        if bg_sd and s not in seen_s:
            seen_s.add(s)
            try:
                import sdpixel
                sdpixel.background_sd(s, w=BW, h=BH, variant=variant)
            except Exception:
                pass
            yield {"kind": "set", "label": s}
        for c in (shot.get("cast") or [])[:3]:
            if c and c not in seen_c:
                seen_c.add(c)
                img = None
                try:
                    PS.asset_spec(c)
                    cells = PS.cells_for(c, variant=cvar)
                    img = _datauri_rgba(PS.to_image(cells, scale=3))
                except Exception:
                    pass
                yield {"kind": "sprite", "label": c, "img": img}
        if first_only:
            return


def main():
    ap = argparse.ArgumentParser(description="Render a film spec as an animated pixel scene.")
    ap.add_argument("spec", nargs="?", default="showcase/knight.json", help="path to a MovieSpec JSON")
    ap.add_argument("--concept", help="generate a fresh film via the model instead of a JSON file")
    ap.add_argument("--step", type=int, default=2, help="keep every Nth tick in the GIF (1 = all)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.concept:
        import movies
        spec = movies.direct(args.concept)
        stem = "concept"
    else:
        spec = json.load(open(args.spec))
        stem = os.path.splitext(os.path.basename(args.spec))[0]

    frames = render_film(spec, step=args.step)
    out = args.out or f"{stem}_pixel.gif"
    save_gif(frames, out, duration=150 * max(1, args.step))
    print(f"wrote {out}  ({len(frames)} frames) — '{spec.get('title')}', {len(spec.get('shots', []))} shots")


if __name__ == "__main__":
    main()
