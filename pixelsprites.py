"""
pixelsprites.py — Level 1 (algorithmic, fully local) PROCEDURAL PIXEL SPRITES.

An ALTERNATIVE asset path that sits BESIDE the ASCII engine without touching it.
Two generators, both deterministic per (label or seed) so the same subject always
yields the SAME sprite (cache-friendly, like ascii_sprites.py):

  * procedural blobs  — Dave-Bollinger / pixel-sprite-generator idea: random half-mask
                        -> mirror -> cellular-automata smoothing -> outline -> shade.
                        Good for 'creature' and 'ship'.
  * template shapes   — a hand-authored half-mask with random ('?') cells, mirrored,
                        for recognisable silhouettes: 'humanoid', 'quadruped', 'tree',
                        'object'. Two colour materials (primary + secondary, e.g. a
                        green canopy on a brown trunk).

Plus simple per-frame ANIMATION (idle bob / walk sway) and a GIF writer.

    python pixelsprites.py                       # contact sheet of all styles -> PNG
    python pixelsprites.py --label dragon        # one sprite -> dragon.png
    python pixelsprites.py --anim knight --mode walk   # animated -> knight.gif

No model, no GPU, no network: microseconds per sprite. Leaf module (stdlib + Pillow).
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import os
import random

from PIL import Image

from llm_client import llm

EMPTY, PRIM, SEC = 0, 1, 2
BORDER_PRIM, BORDER_SEC = 3, 4

_STYLES = ("humanoid", "quadruped", "creature", "ship", "tree", "object")

ASSET_BACKEND = os.environ.get("CLAUDEMOVIES_ASSET_BACKEND", "llm").strip().lower()
VARIANTS = max(1, int(os.environ.get("CLAUDEMOVIES_SPRITE_VARIANTS", "3")))


def _sd():
    import sdpixel
    return sdpixel

_SPEC_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assetspec_cache.json")
try:
    _SPEC_CACHE = json.load(open(_SPEC_CACHE_PATH))
except Exception:
    _SPEC_CACHE = {}

_KW = {
    "humanoid": "knight king queen prince princess person man woman child kid robot wizard witch "
                "hero guard soldier pirate ninja farmer chef clown angel devil viking samurai elf dwarf",
    "quadruped": "cat dog fox wolf lion tiger bear horse cow pig sheep goat deer rabbit mouse panda "
                 "elephant monkey hedgehog raccoon dragon dinosaur",
    "tree": "tree bush shrub flower plant fern mushroom cactus",
    "ship": "ship boat raft rocket car cart wagon plane submarine train",
    "object": "gem crystal potion box chest key star heart coin book lamp candle bell egg moon sun",
}

_STYLE_SCALE = {"humanoid": 0.5, "quadruped": 0.4, "creature": 0.4, "ship": 0.7, "tree": 0.9, "object": 0.3}

_CLASSIFY_SYS = (
    "Classify a subject for a procedural pixel-sprite generator. Reply STRICT JSON only, no prose:\n"
    '{"style":"humanoid"|"quadruped"|"creature"|"ship"|"tree"|"object","hue":0.0-1.0,"scale":0.1-1.0}\n'
    "Pick the style that fits the subject's body plan (humanoid = upright two-legged; quadruped = "
    "four-legged animal; ship = vehicle; tree = plant; object = inanimate item; creature = anything "
    "else). hue is a colour-wheel position: 0.0 red, 0.08 brown/orange, 0.15 yellow, 0.33 green, "
    "0.5 cyan, 0.6 blue, 0.75 purple, 0.9 pink — the subject's typical colour. scale is real-world "
    "HEIGHT relative to others: 0.15 tiny (mouse, insect, coin, snail), 0.3 small (cat, rabbit, "
    "bird, fish), 0.5 human-sized (person, knight, robot, dog), 0.8 large (horse, bear, ship), "
    "1.0 huge (elephant, tree, dragon, whale, house).")


def _heuristic(label: str) -> dict:
    toks = label.lower().replace("-", " ").split()
    for style, words in _KW.items():
        ws = set(words.split())
        if any(t in ws for t in toks):
            return {"style": style, "hue": (_seed_from(label) % 1000) / 1000.0, "scale": _STYLE_SCALE[style]}
    return {"style": "creature", "hue": (_seed_from(label) % 1000) / 1000.0, "scale": 0.4}


def asset_spec(label: str, use_llm: bool = True) -> dict:
    """{style, hue, scale} for any subject — model-classified (cached to disk), heuristic if offline."""
    key = label.lower().strip()
    if key in _SPEC_CACHE and "scale" in _SPEC_CACHE[key]:
        return _SPEC_CACHE[key]
    spec = None
    if use_llm and os.environ.get("CLAUDEMOVIES_LLM_URL"):
        try:
            raw = llm(_CLASSIFY_SYS, f"Subject: {key}", max_tokens=80, temperature=0.0)
            data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1]) if raw and "{" in raw else None
            if isinstance(data, dict) and data.get("style") in _STYLES:
                spec = {"style": data["style"], "hue": max(0.0, min(1.0, float(data.get("hue", 0.5)))),
                        "scale": max(0.1, min(1.0, float(data.get("scale", _STYLE_SCALE[data["style"]]))))}
        except Exception:
            spec = None
    if spec is None:
        spec = _heuristic(label)
    _SPEC_CACHE[key] = spec
    try:
        json.dump(_SPEC_CACHE, open(_SPEC_CACHE_PATH, "w"), indent=0)
    except Exception:
        pass
    return spec

TEMPLATES = {
    "humanoid": [
        "...o#", "..o##", "..o##", ".o###", "o####",
        "o####", "o####", ".####", ".####", ".###o",
        ".##o.", ".##..", ".##..", ".#o..", ".#...",
    ],
    "quadruped": [
        "...o#", "..o##", ".o###", "o####", "o####",
        "o####", "o####", ".####", ".#.#.", ".#.#.", ".#.#.",
    ],
    "tree": [
        "..oo#", ".oo##", "ooo##", "#####", "#####",
        "#####", "ooo##", ".oo##", "...TT", "...TT",
        "...TT", "...TT", "...TT",
    ],
    "object": [
        "...#", "..o#", ".o##", "####", "####", ".o##", "..o#", "...#",
    ],
}


def _hsl(h: float, s: float, l: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hls_to_rgb(h % 1.0, max(0.0, min(1.0, l)), max(0.0, min(1.0, s)))
    return (int(r * 255), int(g * 255), int(b * 255))


def _seed_from(label: str) -> int:
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)


_PX_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drawn_pixels_cache.json")
try:
    _PX_CACHE = json.load(open(_PX_CACHE_PATH))
except Exception:
    _PX_CACHE = {}

_EXAMPLE_FISH = (
    '{"w":12,"h":7,"palette":{"K":"22303a","O":"e0913a","W":"ffffff"},'
    '"rows":["............","...KKKK.....","..KOOOOK..K.",".KOOOWOOKKK.",'
    '"..KOOOOK..K.","...KKKK.....","............"]}')

_DRAW_PX_SYS = (
    "You are a pixel-art artist. Draw a SMALL, instantly recognisable pixel sprite of the subject — "
    "front or side view, centred, on a transparent background. Use bold simple shapes and a dark "
    "outline so it reads at tiny size, and make the silhouette unmistakable (ears, tail, fins, "
    "wings, limbs as the subject needs). Canvas 12-18 wide and 12-18 tall, up to 6 palette colours.\n"
    "Output STRICT JSON only, no prose:\n"
    '{"w":W,"h":H,"palette":{"K":"rrggbb",...},"rows":[H strings, each EXACTLY W chars]}\n'
    "Each character is a palette key (a letter) or '.' for an empty/transparent pixel.\n"
    "EXAMPLE (a fish):\n" + _EXAMPLE_FISH)


def _grid_from_drawspec(d: dict):
    """Build an RGB cell grid (None = transparent) from a {w,h,palette,rows} draw spec."""
    try:
        w, h = int(d["w"]), int(d["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (4 <= w <= 40 and 4 <= h <= 40):
        return None
    pal = {}
    for k, v in (d.get("palette") or {}).items():
        v = str(v).lstrip("#")
        if len(v) == 6:
            try:
                pal[k] = (int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16))
            except ValueError:
                pass
    rows = d.get("rows")
    if not isinstance(rows, list) or not rows:
        return None
    grid = []
    for r in range(h):
        line = rows[r] if r < len(rows) and isinstance(rows[r], str) else ""
        grid.append([pal.get(line[c]) if c < len(line) else None for c in range(w)])
    if not any(c for row in grid for c in row):
        return None
    return grid


def draw_pixels(label: str, desc: str = "", variant: int = 0):
    """Model-drawn sprite cell grid for any subject (cached to disk). None if no endpoint
    or the model didn't return usable pixel art. `variant` (bucketed to VARIANTS) asks
    for a visibly different design of the same subject, cached separately."""
    if not os.environ.get("CLAUDEMOVIES_LLM_URL"):
        return None
    b = variant % VARIANTS
    key = label.lower().strip() + (f"#{b}" if b else "")
    if key in _PX_CACHE:
        return _grid_from_drawspec(_PX_CACHE[key])
    ask = f"Draw: {label}. {desc}".strip()
    if b:
        ask += f" Variation {b}: same subject, but a clearly different design — new palette, pose and details."
    try:
        raw = llm(_DRAW_PX_SYS, ask, max_tokens=1100, temperature=0.5)
    except Exception:
        raw = ""
    if not raw or "{" not in raw:
        return None
    try:
        spec = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    except Exception:
        return None
    grid = _grid_from_drawspec(spec)
    if grid is None:
        return None
    _PX_CACHE[key] = spec
    try:
        json.dump(_PX_CACHE, open(_PX_CACHE_PATH, "w"))
    except Exception:
        pass
    return grid


def style_for(label: str) -> str:
    return asset_spec(label)["style"]


def _mirror(half: list[list[int]], odd: bool) -> list[list[int]]:
    return [row + (row[-2::-1] if odd else row[::-1]) for row in half]


def _shape_procedural(rng, w, h, style) -> list[list[int]]:
    """Random half-mask -> mirror -> cellular-automata smoothing. Returns a material grid."""
    hw = (w + 1) // 2
    half = [[EMPTY] * hw for _ in range(h)]
    for y in range(h):
        vy = 1.0 - abs(y / (h - 1) - 0.5) * 2 if h > 1 else 1.0
        for x in range(hw):
            hxn = (x + 1) / hw
            p = 0.34 + 0.55 * hxn if style == "ship" else (0.46 + 0.42 * hxn) * (0.58 + 0.42 * vy)
            half[y][x] = PRIM if rng.random() < p else EMPTY
    full = _mirror(half, w % 2 == 1)
    W = len(full[0])
    for _ in range(3):
        nxt = [row[:] for row in full]
        for y in range(h):
            for x in range(W):
                n = sum(1 for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                        if not (dx == 0 and dy == 0)
                        and 0 <= y + dy < h and 0 <= x + dx < W and full[y + dy][x + dx])
                nxt[y][x] = PRIM if (n >= 3 if full[y][x] else n >= 5) else EMPTY
        full = nxt
    return full


def _shape_template(rng, style) -> list[list[int]]:
    tmpl = TEMPLATES[style]
    odd = True
    code = {".": EMPTY, "#": PRIM, "T": SEC}
    half = []
    for line in tmpl:
        row = []
        for ch in line:
            if ch == "o":
                row.append(PRIM if rng.random() < 0.85 else EMPTY)
            elif ch == "?":
                row.append(PRIM if rng.random() < 0.5 else EMPTY)
            elif ch == "t":
                row.append(SEC if rng.random() < 0.85 else EMPTY)
            else:
                row.append(code[ch])
        half.append(row)
    return _mirror(half, odd)


def generate(w: int = 9, h: int = 12, seed: int = 0, style: str = "creature",
             hue: float | None = None) -> list[list[tuple[int, int, int] | None]]:
    """Return an h x W grid of RGB cells (None = transparent), left-right symmetric."""
    rng = random.Random(seed)
    mat = _shape_template(rng, style) if style in TEMPLATES else _shape_procedural(rng, w, h, style)
    H, W = len(mat), len(mat[0])

    grid = [row[:] for row in mat]
    for y in range(H):
        for x in range(W):
            if mat[y][x] in (PRIM, SEC) and any(
                    not (0 <= y + dy < H and 0 <= x + dx < W) or not mat[y + dy][x + dx]
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                grid[y][x] = BORDER_PRIM if mat[y][x] == PRIM else BORDER_SEC

    if hue is None:
        hue = rng.random()
    sec_hue = 0.08
    px: list[list[tuple[int, int, int] | None]] = [[None] * W for _ in range(H)]
    for y in range(H):
        for x in range(W):
            c = grid[y][x]
            if c == EMPTY:
                continue
            shade = 0.74 - 0.34 * (y / (H - 1) if H > 1 else 0) + rng.uniform(-0.05, 0.05)
            if c == PRIM:
                px[y][x] = _hsl(hue, 0.55, shade)
            elif c == SEC:
                px[y][x] = _hsl(sec_hue, 0.45, max(0.28, shade - 0.10))
            elif c == BORDER_PRIM:
                px[y][x] = _hsl(hue, 0.60, 0.18)
            else:
                px[y][x] = _hsl(sec_hue, 0.55, 0.14)
    return px


def to_image(px, scale: int = 14) -> Image.Image:
    """Nearest-neighbour-upscaled RGBA image of a cell grid."""
    h, W = len(px), len(px[0])
    base = Image.new("RGBA", (W, h), (0, 0, 0, 0))
    for y in range(h):
        for x in range(W):
            if px[y][x] is not None:
                base.putpixel((x, y), px[y][x] + (255,))
    return base.resize((W * scale, h * scale), Image.NEAREST)


def cells_for(label: str, w: int = 9, h: int = 12, variant: int = 0):
    """One sprite cell grid for a subject, via the configured backend with graceful fallthrough:
    local SD -> model-drawn palette grid -> procedural blob (always works). `variant` keys the
    look to a film, so the same subject differs between films but stays consistent within one:
    SD/LLM bucket it to VARIANTS distinct cached designs, the free procedural path uses the
    full value (unbounded shapes, plus a small hue drift)."""
    if ASSET_BACKEND == "sd":
        try:
            g = _sd().draw_pixels_sd(label, variant=variant)
            if g:
                return g
        except Exception:
            pass
    if ASSET_BACKEND in ("sd", "llm"):
        drawn = draw_pixels(label, variant=variant)
        if drawn is not None:
            return drawn
    spec = asset_spec(label)
    seed = _seed_from(f"{variant}:{label}") if variant else _seed_from(label)
    hue = (spec["hue"] + ((variant % 13) - 6) * 0.01) % 1.0 if variant else spec["hue"]
    return generate(w=w, h=h, seed=seed, style=spec["style"], hue=hue)


def anim_cells_for(label: str, mode: str = "idle", frames: int = 6, variant: int = 0):
    """Animation frames for a subject. With the SD backend we use the model's OWN sprite-sheet
    frames (a real walk/idle cycle); otherwise we synthesise motion from the single sprite."""
    if ASSET_BACKEND == "sd":
        try:
            fr = _sd().frames_sd(label, variant=variant)
            if fr and len(fr) >= 2:
                return fr
        except Exception:
            pass
    return animate_cells(cells_for(label, variant=variant), mode=mode, frames=frames)


def sprite_for(label: str, scale: int = 14, w: int = 9, h: int = 12, variant: int = 0) -> Image.Image:
    return to_image(cells_for(label, w, h, variant=variant), scale=scale)


def _shift(px, dx, dy):
    """Return a copy of the cell grid shifted by (dx, dy), padded with transparency."""
    h, W = len(px), len(px[0])
    out = [[None] * W for _ in range(h)]
    for y in range(h):
        for x in range(W):
            sy, sx = y - dy, x - dx
            if 0 <= sy < h and 0 <= sx < W:
                out[y][x] = px[sy][sx]
    return out


def _leg_lift(px, left: bool):
    """A step pose: the bottom row on one side folds up one pixel (a bent leg),
    instead of vanishing — the silhouette keeps its mass while striding."""
    h, w = len(px), len(px[0])
    out = [row[:] for row in px]
    xs = range(0, w // 2) if left else range((w + 1) // 2, w)
    for x in xs:
        if out[h - 1][x] is not None:
            if h >= 2 and out[h - 2][x] is None:
                out[h - 2][x] = out[h - 1][x]
            out[h - 1][x] = None
    return out


def animate_cells(px, mode: str = "idle", frames: int = 4):
    """Yield `frames` cell grids for a looping idle/walk cycle. Pure 2D transforms so the
    silhouette stays consistent (no re-rolling the random shape per frame).
    Walk = a 4-pose stride (contact, left step, contact, right step) with bob + sway."""
    out = []
    if mode == "walk":
        poses = (_shift(px, 0, 0),
                 _shift(_leg_lift(px, True), -1, -1),
                 _shift(px, 0, 0),
                 _shift(_leg_lift(px, False), 1, -1))
        for i in range(frames):
            out.append(poses[i % 4])
    else:
        for i in range(frames):
            out.append(_shift(px, 0, 0 if i % 2 == 0 else -1))
    return out


def animate(label: str, mode: str = "idle", frames: int = 4, scale: int = 14, variant: int = 0):
    return [to_image(c, scale=scale)
            for c in animate_cells(cells_for(label, variant=variant), mode=mode, frames=frames)]


def save_gif(frames, path: str, duration: int = 180):
    bg = [Image.new("RGBA", f.size, (18, 20, 26, 255)) for f in frames]
    flat = [Image.alpha_composite(b, f).convert("P", palette=Image.ADAPTIVE) for b, f in zip(bg, frames)]
    flat[0].save(path, save_all=True, append_images=flat[1:], loop=0, duration=duration, disposal=2)


def contact_sheet(labels, scale: int = 12, cols: int = 6, pad: int = 12) -> Image.Image:
    imgs = [(lbl, sprite_for(lbl, scale=scale)) for lbl in labels]
    cw = max(im.width for _, im in imgs) + pad
    ch = max(im.height for _, im in imgs) + pad
    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new("RGBA", (cw * min(cols, len(imgs)), ch * rows), (18, 20, 26, 255))
    for i, (lbl, im) in enumerate(imgs):
        cx, cy = (i % cols) * cw, (i // cols) * ch
        sheet.alpha_composite(im, (cx + (cw - im.width) // 2, cy + (ch - im.height) // 2))
    return sheet


def main():
    ap = argparse.ArgumentParser(description="Procedural pixel sprites (Level 1, local).")
    ap.add_argument("--label", help="one named sprite -> <label>.png")
    ap.add_argument("--anim", help="animate a named sprite -> <label>.gif")
    ap.add_argument("--mode", default="idle", choices=["idle", "walk"])
    ap.add_argument("--scale", type=int, default=14)
    ap.add_argument("--variant", type=int, default=0, help="per-film look variant (0 = canonical)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.anim:
        frames = animate(args.anim, mode=args.mode, scale=args.scale, variant=args.variant)
        out = args.out or f"{args.anim}.gif"
        save_gif(frames, out)
        print(f"wrote {out}  ({len(frames)} frames, {args.mode})")
        return
    if args.label:
        img = sprite_for(args.label, scale=args.scale, variant=args.variant)
        out = args.out or f"{args.label}.png"
        img.save(out)
        print(f"wrote {out}  ({img.width}x{img.height}, style={style_for(args.label)})")
        return

    labels = ["knight", "wizard", "robot", "cat", "fox", "bear",
              "dragon", "frog", "alien", "tree", "ship", "gem"]
    contact_sheet(labels).save(args.out or "pixelsprites_demo.png")
    print(f"wrote pixelsprites_demo.png — {len(labels)} sprites across {len(set(style_for(l) for l in labels))} styles")


if __name__ == "__main__":
    main()
