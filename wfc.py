"""
wfc.py — Wave Function Collapse (simple tiled model) + a top-down GROUND generator.

A compact, correct WFC: each cell starts as the full set of possible tiles; we
repeatedly collapse the lowest-entropy cell to one weighted-random tile and propagate
adjacency constraints, restarting on the rare contradiction. Used here to lay out a
coherent, varied floor (base shades + scattered detail tiles) for the pixel scene —
the Hotline-Miami-style top-down look — but the solver itself is fully generic.

Leaf module: stdlib + Pillow only. Nothing imports the ASCII engine.

    python wfc.py            # preview floors for several settings -> wfc_demo.png
"""

from __future__ import annotations

import random

from PIL import Image

_DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
_OPP = [2, 3, 0, 1]


def solve(w: int, h: int, ntiles: int, weights, allowed, rng, retries: int = 30):
    """Return a w*h grid of tile ids. `allowed[d]` is a set of (a, b) pairs meaning tile
    `a` may sit with tile `b` to its `d` side. Restarts on contradiction up to `retries`."""
    for _ in range(retries):
        wave = [[set(range(ntiles)) for _ in range(w)] for _ in range(h)]
        if _run(wave, w, h, weights, allowed, rng):
            return [[next(iter(wave[y][x])) for x in range(w)] for y in range(h)]
    best = max(range(ntiles), key=lambda t: weights[t])
    return [[best] * w for _ in range(h)]


def _run(wave, w, h, weights, allowed, rng):
    while True:
        cell = _min_entropy(wave, w, h, rng)
        if cell is None:
            return True
        x, y = cell
        opts = list(wave[y][x])
        if not opts:
            return False
        wts = [weights[t] for t in opts]
        wave[y][x] = {rng.choices(opts, weights=wts, k=1)[0]}
        if not _propagate(wave, w, h, x, y, allowed):
            return False


def _min_entropy(wave, w, h, rng):
    best, choice = None, None
    for y in range(h):
        for x in range(w):
            n = len(wave[y][x])
            if n > 1 and (best is None or n < best or (n == best and rng.random() < 0.3)):
                best, choice = n, (x, y)
    return choice


def _propagate(wave, w, h, x, y, allowed):
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        for d, (dx, dy) in enumerate(_DIRS):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            possible = set()
            for a in wave[cy][cx]:
                possible |= {b for (aa, b) in allowed[d] if aa == a}
            new = wave[ny][nx] & possible
            if not new:
                return False
            if len(new) < len(wave[ny][nx]):
                wave[ny][nx] = new
                stack.append((nx, ny))
    return True


GROUND = {
    "grass":  [(96, 158, 70), (110, 176, 84), (74, 130, 56)],
    "forest": [(70, 120, 60), (86, 140, 72), (52, 96, 48)],
    "water":  [(58, 110, 178), (74, 132, 200), (40, 86, 150)],
    "sand":   [(214, 196, 140), (226, 210, 158), (196, 176, 120)],
    "snow":   [(224, 230, 240), (238, 242, 250), (200, 210, 228)],
    "stone":  [(120, 122, 132), (140, 142, 152), (96, 98, 108)],
    "road":   [(96, 98, 108), (110, 112, 122), (78, 80, 90)],
    "lava":   [(200, 80, 36), (236, 130, 40), (150, 40, 24)],
    "earth":  [(140, 102, 64), (160, 120, 78), (112, 80, 48)],
    "indoor": [(150, 120, 84), (168, 138, 100), (124, 96, 64)],
}

_GW = [16, 7, 2]


def _ground_rules():
    allowed = [set() for _ in range(4)]
    for d in range(4):
        for a in range(3):
            for b in range(3):
                if a == 2 and b == 2:
                    continue
                allowed[d].add((a, b))
    return allowed


_GROUND_RULES = _ground_rules()


def _shade(rgb, f):
    return tuple(max(0, min(255, int(c * f))) for c in rgb)


def render_ground(setting: str, w_px: int, h_px: int, seed: int, tile: int = 8) -> Image.Image:
    """A WFC-laid floor filling w_px x h_px for the given terrain setting.
    Texture comes from a small pool of pre-noised tile variants pasted per cell,
    so the cost is a few hundred pastes instead of one putpixel per pixel."""
    pal = GROUND.get(setting, GROUND["earth"])
    rng = random.Random(seed)
    tw, th = (w_px + tile - 1) // tile, (h_px + tile - 1) // tile
    grid = solve(tw, th, 3, _GW, _GROUND_RULES, rng)
    variants = []
    for t in range(3):
        vs = []
        for _ in range(5):
            v = Image.new("RGBA", (tile, tile))
            vp = v.load()
            for py in range(tile):
                f0 = 1.0 + 0.05 * ((py / tile) - 0.5)
                for px in range(tile):
                    vp[px, py] = _shade(pal[t], f0 + rng.uniform(-0.04, 0.04)) + (255,)
            vs.append(v)
        variants.append(vs)
    img = Image.new("RGBA", (tw * tile, th * tile), (0, 0, 0, 255))
    for ty in range(th):
        for tx in range(tw):
            t = grid[ty][tx]
            img.paste(rng.choice(variants[t]), (tx * tile, ty * tile))
            if t == 2:
                cx, cy = tx * tile + tile // 2, ty * tile + tile // 2
                dark = _shade(pal[t], 0.7) + (255,)
                img.putpixel((cx, cy), dark)
                img.putpixel((cx, cy + 1), dark)
    return img.crop((0, 0, w_px, h_px))


def main():
    settings = ["grass", "water", "stone", "sand", "snow", "lava"]
    pad, w, h = 8, 220, 90
    sheet = Image.new("RGBA", ((w + pad) * 3 + pad, (h + pad) * 2 + pad), (18, 20, 26, 255))
    for i, s in enumerate(settings):
        g = render_ground(s, w, h, seed=i + 1)
        x, y = pad + (i % 3) * (w + pad), pad + (i // 3) * (h + pad)
        sheet.alpha_composite(g, (x, y))
    sheet.save("wfc_demo.png")
    print(f"wrote wfc_demo.png — {len(settings)} WFC floors")


if __name__ == "__main__":
    main()
