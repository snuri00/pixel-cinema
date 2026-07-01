"""
pixelscene.py — render a film spec as an animated PIXEL scene (the Level-1 alternative
to the ASCII stage). Same MovieSpec the director already produces (title, logline,
shots[ narration, cast, action, setting, camera ]) — but instead of an ASCII grid it
composes: a sky gradient + a WaveFunctionCollapse ground (wfc.py) + animated procedural
sprites (pixelsprites.py), with a subtitle bar. Output is a looping GIF.

Nothing here touches the ASCII engine (render.py / stage.py / server_app.py). It reuses
ONLY the leaf asset modules, so the two render paths live side by side.

    python pixelscene.py                       # render showcase/knight.json -> *_pixel.gif
    python pixelscene.py showcase/ghost.json
    python pixelscene.py --concept "a raccoon who runs a midnight noodle stand"   # needs the model
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os

from PIL import Image, ImageDraw, ImageEnhance

import pixelsprites as PS
import wfc

W, H = 320, 180
GROUND_Y = int(H * 0.64)
CELL = 6
FRAMES_PER_SHOT = 6
WALK_ACTIONS = {"enter", "exit", "run", "walk", "chase", "flee", "travel"}

SKY = {
    "lava":  ((60, 20, 28), (150, 60, 40)),
    "snow":  ((150, 170, 205), (215, 228, 240)),
    "water": ((70, 120, 175), (150, 195, 225)),
    "sky":   ((70, 130, 200), (180, 215, 240)),
    "stone": ((60, 66, 86), (120, 130, 150)),
    "indoor": ((40, 34, 46), (78, 64, 74)),
}
SKY_DEFAULT = ((78, 132, 196), (188, 214, 236))


def _sky(setting: str) -> Image.Image:
    top, bot = SKY.get(setting, SKY_DEFAULT)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    horizon = H if setting == "sky" else GROUND_Y
    for y in range(H):
        t = min(1.0, y / max(1, horizon))
        c = tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3))
        for x in range(W):
            img.putpixel((x, y), c + (255,))
    return img


def _background(shot: dict, seed: int) -> Image.Image:
    setting = shot.get("setting", "earth")
    if os.environ.get("CLAUDEMOVIES_BG_BACKEND") == "sd":
        try:
            import sdpixel
            return sdpixel.background_sd(setting, w=W, h=H).convert("RGBA")
        except Exception:
            pass
    bg = _sky(setting)
    if setting != "sky":
        ground = wfc.render_ground(setting, W, H - GROUND_Y, seed=seed, tile=8)
        bg.alpha_composite(ground, (0, GROUND_Y))
    if setting not in ("indoor", "lava"):
        d = ImageDraw.Draw(bg)
        disc = (255, 244, 210) if setting != "snow" else (240, 246, 255)
        d.ellipse([W - 54, 18, W - 26, 46], fill=disc + (255,))
    return bg


def _shadow(scene: Image.Image, cx: int, y: int, w: int):
    ov = Image.new("RGBA", scene.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    rw = max(5, int(w * 0.42))
    rh = max(2, rw // 3)
    d.ellipse([cx - rw, y - rh, cx + rw, y + rh], fill=(0, 0, 0, 95))
    scene.alpha_composite(ov)


def _grade(bg: Image.Image, shot: dict, index: int) -> Image.Image:
    narr = (shot.get("narration", "") + " " + shot.get("dialogue", "")).lower()
    img = bg if bg.mode == "RGBA" else bg.convert("RGBA")
    if index % 2 == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    b = 1.0
    if any(k in narr for k in ("dark", "night", "shadow", "dim", "cave", "black", "gloom")):
        b = 0.6
    elif any(k in narr for k in ("bright", "sunny", "noon", "daylight", "blaze", "glow")):
        b = 1.12
    tint = None
    if any(k in narr for k in ("dawn", "dusk", "sunset", "sunrise", "ember", "warm", "fire", "lava")):
        tint = (1.12, 1.0, 0.82)
    elif any(k in narr for k in ("cold", "frost", "ice", "snow", "moon", "winter", "frozen")):
        tint = (0.86, 0.94, 1.14)
    b *= 1.0 + (((index * 13) % 7) - 3) * 0.02
    img = ImageEnhance.Brightness(img).enhance(b)
    if tint:
        r, g, bl, a = img.split()
        r = r.point(lambda v: min(255, int(v * tint[0])))
        g = g.point(lambda v: min(255, int(v * tint[1])))
        bl = bl.point(lambda v: min(255, int(v * tint[2])))
        img = Image.merge("RGBA", (r, g, bl, a))
    return img


def _sprite_px_size(cells, target_h: int):
    h, w = len(cells), len(cells[0])
    return max(1, round(w * target_h / max(1, h))), target_h


def _blit_cells(scene: Image.Image, cells, foot_x: int, foot_y: int, target_h: int):
    """Draw a sprite cell grid at a real on-screen HEIGHT (px), feet at foot_y, centred on foot_x."""
    h, w = len(cells), len(cells[0])
    base = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for cy in range(h):
        for cx in range(w):
            if cells[cy][cx] is not None:
                base.putpixel((cx, cy), cells[cy][cx] + (255,))
    tw, th = _sprite_px_size(cells, target_h)
    spr = base.resize((tw, th), Image.NEAREST)
    scene.alpha_composite(spr, (foot_x - tw // 2, foot_y - th))


def _subtitle(scene: Image.Image, text: str):
    if not text:
        return
    d = ImageDraw.Draw(scene)
    for a, b in (("—", "-"), ("–", "-"), ("…", "..."), ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'")):
        text = text.replace(a, b)
    text = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in text)
    if len(text) > 58:
        cut = text[:55]
        text = (cut[:cut.rfind(" ")] if " " in cut else cut).rstrip(" ,;:") + "..."
    d.rectangle([0, H - 22, W, H], fill=(8, 10, 14, 235))
    tw = d.textlength(text)
    d.text(((W - tw) // 2, H - 16), text, fill=(225, 232, 240))


def _target_px(label: str) -> int:
    s = PS.asset_spec(label).get("scale", 0.4)
    return int(round(36 + max(0.1, min(1.0, s)) * 78))


def _shot_frames(shot: dict, index: int) -> list[Image.Image]:
    cast = [c for c in (shot.get("cast") or [])][:3] or ["hero"]
    mode = "walk" if shot.get("action") in WALK_ACTIONS else "idle"
    bg = _grade(_background(shot, seed=index * 97 + 13), shot, index)

    anims = [PS.anim_cells_for(c, mode=mode, frames=FRAMES_PER_SHOT) for c in cast]
    n = len(cast)
    xs = [int(W * (i + 1) / (n + 1)) for i in range(n)]
    targets = [_target_px(c) for c in cast]
    dims = [(len(a[0][0]), len(a[0])) for a in anims]
    maxh = GROUND_Y - 6
    fit = 1.0
    widths = [dims[i][0] * targets[i] / max(1, dims[i][1]) for i in range(n)]
    for i in range(n):
        if targets[i] > maxh:
            fit = max(fit, targets[i] / maxh)
    total_w = sum(widths) + 10 * (n + 1)
    if total_w > W:
        fit = max(fit, total_w / W)
    if fit > 1.0:
        targets = [max(20, int(t / fit)) for t in targets]
    nframes = max((len(a) for a in anims), default=FRAMES_PER_SHOT)
    shake = shot.get("camera") == "shake"

    frames = []
    for f in range(nframes):
        scene = bg.copy()
        jx, jy = (0, 0)
        if shake:
            jx, jy = ((-2, 0, 2, 0, -1, 1)[f % 6], (0, 1, 0, -1, 1, 0)[f % 6])
        for i, anim in enumerate(anims):
            cells = anim[f % len(anim)]
            tw, _ = _sprite_px_size(cells, targets[i])
            _shadow(scene, xs[i] + jx, GROUND_Y + 3 + jy, tw)
            _blit_cells(scene, cells, xs[i] + jx, GROUND_Y + 2 + jy, targets[i])
        _subtitle(scene, shot.get("narration", ""))
        frames.append(scene)
    return frames


def render_film(spec: dict, scale: int = 2) -> list[Image.Image]:
    """All shots -> one list of upscaled RGBA frames (each shot animates then the next plays)."""
    frames: list[Image.Image] = []
    for i, shot in enumerate(spec.get("shots", [])):
        frames += _shot_frames(shot, i)
    return [f.resize((W * scale, H * scale), Image.NEAREST) for f in frames]


def save_gif(frames, path: str, duration: int = 150):
    flat = [f.convert("RGB").convert("P", palette=Image.ADAPTIVE) for f in frames]
    flat[0].save(path, save_all=True, append_images=flat[1:], loop=0, duration=duration)


def _datauri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _tint_of(img: Image.Image):
    """Average colour + brightness 'energy' of a frame, for the reactive cinema glow."""
    r, g, b = img.convert("RGB").resize((1, 1), Image.BILINEAR).getpixel((0, 0))
    lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return [r, g, b], round(min(1.0, lum * 1.15), 3)


def _hold(shot: dict) -> int:
    """How many displayed frames this shot dwells — scales with text so subtitles stay
    up long enough to read (the player ticks ~150ms/frame, matching the ASCII path)."""
    chars = len(shot.get("narration", "")) + len(shot.get("dialogue", ""))
    return max(16, min(48, 14 + chars // 3))


def _end_card(spec: dict, scale: int = 2) -> Image.Image:
    img = Image.new("RGBA", (W, H), (10, 12, 16, 255))
    d = ImageDraw.Draw(img)
    end = "THE END"
    title = (spec.get("title") or "").upper()[:44]
    ew = d.textlength(end)
    d.text(((W - ew) // 2, H // 2 - 18), end, fill=(236, 240, 246))
    if title:
        tw = d.textlength(title)
        d.text(((W - tw) // 2, H // 2 + 2), '"' + title + '"', fill=(150, 160, 172))
    return img.resize((W * scale, H * scale), Image.NEAREST)


def iter_film_frames(spec: dict, scale: int = 2):
    """Yield (data_uri, tint, lvl) per displayed frame. Each shot's short animation cycle is
    encoded once, then repeated for its hold window — so the whole film is only ~6 PNGs/shot."""
    import traceback
    for i, shot in enumerate(spec.get("shots", [])):
        try:
            cyc = []
            for f in _shot_frames(shot, i):
                tint, lvl = _tint_of(f)
                cyc.append((_datauri(f.resize((W * scale, H * scale), Image.NEAREST)), tint, lvl))
        except Exception:
            print(f"pixelscene: shot {i} failed, skipping:", file=__import__("sys").stderr)
            traceback.print_exc()
            continue
        for k in range(_hold(shot)):
            yield cyc[k % len(cyc)]
    end = _datauri(_end_card(spec, scale))
    for _ in range(20):
        yield (end, [12, 14, 18], 0.0)


def iter_prewarm(spec: dict, first_only: bool = False):
    seen_c, seen_s = set(), set()
    bg_sd = os.environ.get("CLAUDEMOVIES_BG_BACKEND") == "sd"
    for shot in spec.get("shots", []):
        s = shot.get("setting", "earth")
        if bg_sd and s not in seen_s:
            seen_s.add(s)
            try:
                import sdpixel
                sdpixel.background_sd(s, w=W, h=H)
            except Exception:
                pass
            yield "bg:" + s
        for c in (shot.get("cast") or [])[:3]:
            if c and c not in seen_c:
                seen_c.add(c)
                try:
                    PS.asset_spec(c)
                    PS.cells_for(c)
                except Exception:
                    pass
                yield "sprite:" + c
        if first_only:
            return


def main():
    ap = argparse.ArgumentParser(description="Render a film spec as an animated pixel scene.")
    ap.add_argument("spec", nargs="?", default="showcase/knight.json", help="path to a MovieSpec JSON")
    ap.add_argument("--concept", help="generate a fresh film via the model instead of a JSON file")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.concept:
        import movies
        spec = movies.direct(args.concept)
        stem = "concept"
    else:
        spec = json.load(open(args.spec))
        stem = os.path.splitext(os.path.basename(args.spec))[0]

    frames = render_film(spec)
    out = args.out or f"{stem}_pixel.gif"
    save_gif(frames, out)
    print(f"wrote {out}  ({len(frames)} frames) — '{spec.get('title')}', {len(spec.get('shots', []))} shots")


if __name__ == "__main__":
    main()
