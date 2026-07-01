"""
Render a PIXEL CINEMA script to an actual video (animated GIF) + a poster frame,
so it can be watched outside a terminal. Same scenes/animation the engine plays.

    python render.py --concept "a tiny knight afraid of the dark"
"""

from __future__ import annotations

import argparse
import json
import os
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

import draw as drawer
import movies

if TYPE_CHECKING:
    from schema import RGB, Grid, MovieSpec, ScenePlan, Shot, Sprite

W, H = 80, 18
CW, CH, PAD = 10, 20, 10
BG = (40, 42, 46)          # charcoal, like a terminal background
FG = (124, 252, 154)
WHITE = (230, 237, 243)
YELLOW = (255, 228, 92)
# per-sprite colours live in movies.sprite_rgb (shared with the terminal player)


def _font(size=16):
    for p in ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf",
              "/Library/Fonts/Andale Mono.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


FONT = _font(16)


def _stamp(grid: Grid, lines: Sprite, x: int, y: int, rgb: RGB) -> None:
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            yy, xx = y + r, x + c
            if 0 <= yy < H and 0 <= xx < W and ch != " ":
                grid[yy][xx] = [ch, rgb]


def _stamp_occlude(grid: Grid, lines: Sprite, x: int, y: int, rgb: RGB) -> None:
    """Like _stamp, but the sprite's whole bounding box is OPAQUE: non-space cells
    draw the character, space cells clear to background. Drawing the cast back-to-
    front, a front character thus fully covers (occludes) the ones behind it — the
    top-z character is never overlapped; lower-z characters get covered."""
    for r, line in enumerate(lines):
        yy = y + r
        if not (0 <= yy < H):
            continue
        for c, ch in enumerate(line):
            xx = x + c
            if 0 <= xx < W:
                grid[yy][xx] = [ch, rgb] if ch != " " else [" ", FG]


def _text(grid: Grid, row: int, s: str, rgb: RGB) -> None:
    s = s[:W - 2]
    start = max(0, (W - len(s)) // 2)
    for i, ch in enumerate(s):
        if 0 <= start + i < W and 0 <= row < H:
            grid[row][start + i] = [ch, rgb]


def _wrap(s: str, width: int = W - 2, lines: int = 2) -> list[str]:
    """Word-wrap `s` into at most `lines` centred subtitle lines; anything that
    still overflows the last line is clipped with an ellipsis, never dropped."""
    out, cur = [], ""
    for w in s.split():
        if cur and len(cur) + 1 + len(w) > width:
            out.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        out.append(cur)
    if len(out) > lines:
        rest = " ".join(out[lines - 1:])
        out = out[:lines - 1] + [rest[:width - 1].rstrip() + "…"]
    return out


def _floor(grid: Grid, kind: str, f: int) -> None:
    """Draw the ground as textured terrain: a SURFACE row of shade/wave glyphs over
    a FILL row, coloured per `kind` (water/grass/sand/snow/stone/road/lava/...).
    `f` shimmers animated floors (water, lava). 'sky' draws nothing (open air)."""
    prof = movies.FLOOR.get(kind)
    if not prof:
        return
    surf, fill, scol, fcol = prof["surf"], prof["fill"], prof["scol"], prof["fcol"]
    shift = f if prof.get("anim") else 0
    g = H - 6                                       # surface row; fill sits just below
    for c in range(W):
        grid[g][c] = [surf[(c + shift) % len(surf)], scol]
        grid[g + 1][c] = [fill, fcol]


def _dim(rgb, k=0.6):
    return tuple(int(c * k) for c in rgb)


def _scenery(grid: Grid, plan: ScenePlan, f: int, dx: int = 0) -> None:
    """Set-dressing drawn BEHIND the cast: stars, a corner sun/moon, drifting
    clouds, and dimmed background props (trees/cactus/pine/rock/mountain). `dx`
    tracks the ground props with a panning camera."""
    for i in range(plan["stars"]):                  # stable night-star scatter
        grid[i % 2][(5 + i * 11) % (W - 2) + 1] = ["*", (205, 205, 165)]
    if plan["sky"]:                                 # sun / moon in the top-right corner
        art = drawer.draw_sprite(plan["sky"])
        _stamp(grid, art, W - len(art[0]) - 2, 0, movies.sprite_rgb(plan["sky"]))
    if plan["clouds"]:                              # clouds drift slowly across the top
        cloud = drawer.draw_sprite("cloud")
        for i in range(plan["clouds"]):
            _stamp(grid, cloud, (i * 24 + f // 2) % (W + 12) - 10, i % 2, (200, 205, 215))
    g = H - 6                                       # background props stand on the floor line
    for label, frac in plan["ground"]:
        art = drawer.draw_sprite(label)
        _stamp(grid, art, int(frac * (W - 1)) - len(art[0]) // 2 + dx, g - len(art),
               _dim(movies.sprite_rgb(label)))


def _emote(grid: Grid, glyph: str, rgb: RGB, cx: int, top_row: int) -> None:
    """Float a mood glyph one row above a character's head, centred on the sprite."""
    y = top_row - 1
    if y >= 0:
        _stamp(grid, [glyph], cx, y, rgb)


def _page_hold(text: str) -> int:
    """Frames to HOLD one subtitle page, sized to its reading time (clamped)."""
    read = int((len(text) / movies._READ_CPS) * (1000 / movies.FRAME_MS))
    return max(movies._HOLD_MIN, min(movies._HOLD_MAX, read))


def _shot_frames(shot: Shot, layout=None, prev_cast=(), move=None, hold=None) -> list[Grid]:
    """Frames for ONE shot. Characters already on stage (in prev_cast) stay LOCKED;
    only NEW characters animate in. The narration is split into PAGES that each fit the
    subtitle area — a long sentence CONTINUES on the next page, and the scene holds on
    each page long enough to read it, so text is never cut off mid-sentence."""
    cast = shot.get("cast") or ["tree"]
    layout = layout or movies.home_columns(cast, W)
    action = shot.get("action", "gather")
    camera = shot.get("camera", "static")
    moods = shot.get("mood") or []
    kind = movies.floor_kind(shot.get("narration", ""), cast, shot.get("setting"))
    plan = movies.scenery(shot, kind)
    sprites = {nm: drawer.draw_sprite(nm) for nm in cast}
    ground = H - 6
    narr, dlg = shot.get("narration", ""), shot.get("dialogue", "")

    maxlines = 2 if dlg else 3
    wrapped = _wrap(narr, lines=10_000) or [""]              # ALL lines, never clipped
    pages = [wrapped[i:i + maxlines] for i in range(0, len(wrapped), maxlines)]
    # one (move, hold, lines, text) window per page; entrance only on the first page
    windows = []
    for pi, plines in enumerate(pages):
        ptext = " ".join(plines)
        mv = (move if move is not None else movies.MOVE_FRAMES) if pi == 0 else 8
        hd = hold if (hold is not None and pi == 0) else _page_hold(ptext)
        windows.append((mv, hd, plines, ptext))
    total = sum(mv + hd for mv, hd, _, _ in windows)

    out, gf = [], 0
    for pi, (mv, hd, plines, ptext) in enumerate(windows):
        settled = (set(prev_cast) if pi == 0 else set(cast))   # pages after the first: all settled
        for f in range(mv + hd):
            te = movies.ease(f / max(1, mv))
            p = gf / max(1, total - 1)                          # full-shot progress (camera)
            cdx, cdy = movies.camera_offset(camera, p, gf)
            grid = [[[" ", FG] for _ in range(W)] for _ in range(H)]
            _floor(grid, kind, gf)
            _scenery(grid, plan, gf, cdx)
            for i, nm in enumerate(cast):
                spr = sprites[nm]
                h, home = len(spr), layout.get(nm, W // 2)
                sw = max(len(r) for r in spr)                     # widest row
                home = max(0, min(home, W - sw))                  # keep the whole sprite on screen
                base = ground - h
                if action == "exit" and pi == len(windows) - 1:   # leaving on the last page
                    x, row = int(home + te * (W - 2 - home)), base
                elif nm in settled:                               # locked, no re-entry
                    x, row = home, base
                else:                                             # new: slide in from nearer side
                    start = -len(spr[0]) - 1 if home < W // 2 else W + 1
                    x, row = int(start + te * (home - start)), base
                if (nm in settled or f >= mv) and (gf // 5) % 2 == 0:
                    row -= 1                                      # gentle idle "breathing"
                x, row = x + cdx, row + cdy
                _stamp_occlude(grid, spr, x, row, movies.sprite_rgb(nm))
                em = movies.mood_emote(moods[i]) if i < len(moods) else None
                if em:
                    _emote(grid, em[0], em[1], x + len(spr[0]) // 2, row)
            # this page's subtitle, typed on over its move window then held
            base = max(H - len(plines) - 1, H - 3)              # keep a blank row above the floor
            reveal = int(len(ptext) * te) + 1
            for k, line in enumerate(plines):
                _text(grid, base + k, line[:max(0, reveal)], WHITE)
                reveal -= len(line) + 1
            if dlg:
                _text(grid, base - 1, "“" + dlg + "”", YELLOW)
            out.append(grid)
            gf += 1
    return out


def _card(lines: list[tuple[str, RGB]]) -> Grid:
    """A centred text card on the charcoal background (title/end cards)."""
    grid = [[[" ", FG] for _ in range(W)] for _ in range(H)]
    top = max(0, (H - len(lines)) // 2 - 1)
    for k, (text, rgb) in enumerate(lines):
        _text(grid, top + k, text, rgb)
    return grid


def _dim_grid(grid: Grid, k: float) -> Grid:
    """A copy of `grid` with every colour scaled toward black (for fades)."""
    return [[[cell[0], tuple(int(c * k) for c in cell[1])] for cell in row] for row in grid]


def _title_card(spec: MovieSpec) -> list[Grid]:
    title = (spec.get("title") or "untitled").upper()[:30]
    logline = (spec.get("logline") or "")[:54]
    card = _card([("─ ─   P I X E L   C I N E M A   ─ ─", WHITE), ("", FG),
                  ("“" + title + "”", WHITE), (logline, (170, 176, 186))])
    return [_dim_grid(card, .3), _dim_grid(card, .65)] + [card] * 14 \
        + [_dim_grid(card, .5), _dim_grid(card, .2)]


def _end_card(spec: MovieSpec) -> list[Grid]:
    title = (spec.get("title") or "").upper()[:30]
    end = _card([("T H E   E N D", WHITE), ("", FG), ("“" + title + "”", WHITE)])
    return ([_dim_grid(end, .3), _dim_grid(end, .65)] + [end] * 14    # fade in + hold
            + [_dim_grid(end, .4), _dim_grid(end, .15)])             # fade THE END out


def iter_movie_frames(spec: MovieSpec, cards: bool = True):
    """Yield the film one frame at a time, building each shot just-in-time. Streaming
    players MUST use this (not the list builder): the first frame appears instantly and
    the worker yields between shots, so a long film never blocks long enough to stall the
    connection and get the generator restarted from the top."""
    order = movies.appearance_order(spec)
    if cards:
        yield from _title_card(spec)
    prev: set[str] = set()
    shots = spec["shots"]
    for j, sh in enumerate(shots):
        cast = [n for n in order if n in sh.get("cast", [])]
        fr = _shot_frames(sh, movies.home_columns(cast, W), prev)
        yield from fr
        if j < len(shots) - 1:                             # soft dip between scenes
            yield _dim_grid(fr[-1], .5)
            yield _dim_grid(fr[-1], .2)
        prev = set(sh.get("cast", []))
    if cards:
        yield from _end_card(spec)


def movie_frames(spec: MovieSpec, cards: bool = True) -> list[Grid]:
    """Every frame of the film as a list (for the GIF/filmstrip). Players that stream
    inline should use iter_movie_frames instead so they don't block building it all."""
    return list(iter_movie_frames(spec, cards))


def _img(grid):
    im = Image.new("RGB", (W * CW + 2 * PAD, H * CH + 2 * PAD), BG)
    d = ImageDraw.Draw(im)
    for r in range(H):
        for c in range(W):
            ch, rgb = grid[r][c]
            if ch != " ":
                d.text((PAD + c * CW, PAD + r * CH), ch, font=FONT, fill=rgb)
    return im


def _ascii(grid: Grid) -> str:
    return "\n".join("".join(cell[0] for cell in row).rstrip() for row in grid)


def settled_frames(spec: MovieSpec) -> list[Grid]:
    """The final (settled) frame of each shot — the filmstrip / ascii preview."""
    order = movies.appearance_order(spec)
    out: list[Grid] = []
    prev: set[str] = set()
    for shot in spec["shots"]:
        cast = [n for n in order if n in shot.get("cast", [])]
        out.append(_shot_frames(shot, movies.home_columns(cast), prev)[-1])
        prev = set(shot.get("cast", []))
    return out


def render_spec(spec: MovieSpec) -> tuple[str, str, int]:
    """Render a spec to a timestamped GIF + filmstrip PNG. Returns (gif, png, n_frames)."""
    imgs = [_img(g) for g in movie_frames(spec)]      # full film: cards + shots + fades
    here, stamp = movies.saved_dir(), movies.stamped_slug(spec["title"])
    gif = os.path.join(here, stamp + ".gif")
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=movies.FRAME_MS, loop=0)

    shot_imgs = [_img(g) for g in settled_frames(spec)]
    strip = Image.new("RGB", (shot_imgs[0].width, sum(im.height + 6 for im in shot_imgs)), BG)
    y = 0
    for im in shot_imgs:
        strip.paste(im, (0, y))
        y += im.height + 6
    film = os.path.join(here, stamp + ".png")
    strip.save(film)
    return gif, film, len(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept")
    ap.add_argument("--play", help="render a saved movie JSON (no model needed)")
    a = ap.parse_args()

    if a.play:
        spec = json.load(open(a.play))
    elif a.concept:
        spec = movies.direct(a.concept)
        movies.save_movie(spec)
    else:
        ap.error("give --concept or --play")
    print(f"{spec['title']} — {spec.get('logline','')}\n")

    gif, film, n = render_spec(spec)
    for g in settled_frames(spec):
        print(_ascii(g))
        print()
    print(f"saved video: {gif}  ({n} frames)\nfilmstrip: {film}")


if __name__ == "__main__":
    main()
