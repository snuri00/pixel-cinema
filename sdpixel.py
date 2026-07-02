"""
sdpixel.py — LOCAL Stable-Diffusion pixel-art asset backend (the "ticari servislerin
yaptığı" yöntemin local/ücretsiz hali): a pixel-tuned SD 1.5 model draws the subject, then
a deterministic PIXELIZATION pass (downscale -> palette quantize -> background removal)
turns it into a clean sprite cell grid compatible with pixelsprites/pixelscene.

Runs on a 4 GB GPU: fp16 + attention/VAE slicing + (optional) CPU offload. Generation is
slow-ish but every sprite is cached to disk, so a film only pays it once.

    python sdpixel.py --label cat            # raw + pixelized preview PNGs
    python sdpixel.py --label "red dragon"

Config (env): CLAUDEMOVIES_SD_MODEL, CLAUDEMOVIES_SD_PROMPT (use {subject}),
CLAUDEMOVIES_SD_STEPS, CLAUDEMOVIES_SD_OFFLOAD=1 to force CPU offload.
Leaf-ish: imports torch/diffusers/Pillow lazily; nothing imports the ASCII engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading

from PIL import Image

_GEN_LOCK = threading.Lock()

MODEL = os.environ.get("CLAUDEMOVIES_SD_MODEL", "PublicPrompts/All-In-One-Pixel-Model")
PROMPT = os.environ.get("CLAUDEMOVIES_SD_PROMPT",
                        "{subject}, pixelsprite, full body, side view, solid magenta background")
NEG = os.environ.get("CLAUDEMOVIES_SD_NEG",
                     "blurry, realistic, photo, 3d render, text, watermark, multiple, cropped, "
                     "portrait, close-up, sprite sheet")
LCM = os.environ.get("CLAUDEMOVIES_SD_LCM", "1") == "1"
LCM_MODEL = os.environ.get("CLAUDEMOVIES_SD_LCM_MODEL", "latent-consistency/lcm-lora-sdv1-5")
STEPS = int(os.environ.get("CLAUDEMOVIES_SD_STEPS", "8" if LCM else "26"))
GUIDANCE = float(os.environ.get("CLAUDEMOVIES_SD_GUIDANCE", "1.6" if LCM else "7.5"))
GEN = int(os.environ.get("CLAUDEMOVIES_SD_GEN", "512"))
TARGET_H = int(os.environ.get("CLAUDEMOVIES_SD_PXH", "48"))
COLORS = int(os.environ.get("CLAUDEMOVIES_SD_COLORS", "16"))

_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drawn_sd_cache.json")
try:
    _CACHE = json.load(open(_CACHE_PATH))
except Exception:
    _CACHE = {}

_PIPE = None


def _seed(label: str) -> int:
    return int(hashlib.sha256(label.encode()).hexdigest()[:8], 16) % (2 ** 31)


def _load_pipe():
    """Load the SD pipeline once, tuned for a 4 GB card."""
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    import torch
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    if LCM:
        from diffusers import LCMScheduler
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights(LCM_MODEL)
        pipe.fuse_lora()
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass
    if os.environ.get("CLAUDEMOVIES_SD_OFFLOAD", "1") != "0":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
    _PIPE = pipe
    torch.cuda.empty_cache()
    return pipe


def generate_raw(subject: str, seed: int | None = None, prompt: str | None = None) -> Image.Image:
    """One 512px SD render. Uses the sprite PROMPT for a subject, or a raw `prompt` verbatim."""
    import torch
    pipe = _load_pipe()
    g = torch.Generator(device="cpu").manual_seed(seed if seed is not None else _seed(subject))
    text = prompt if prompt else PROMPT.format(subject=subject)
    try:
        img = pipe(text, negative_prompt=NEG,
                   num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                   width=GEN, height=GEN, generator=g).images[0]
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return img.convert("RGB")


TOL = int(os.environ.get("CLAUDEMOVIES_SD_TOL", "60"))


def _corners(im, k=3):
    from collections import Counter
    w, h = im.size
    edge = ([im.getpixel((x, 0)) for x in range(w)] + [im.getpixel((x, h - 1)) for x in range(w)]
            + [im.getpixel((0, y)) for y in range(h)] + [im.getpixel((w - 1, y)) for y in range(h)])
    return [c for c, _ in Counter(edge).most_common(k)]


def _is_bg(c, corners, tol=TOL):
    return any(abs(c[0] - b[0]) + abs(c[1] - b[1]) + abs(c[2] - b[2]) <= tol for b in corners)


def _runs(mask):
    runs, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            runs.append([s, i - 1]); s = None
    if s is not None:
        runs.append([s, len(mask) - 1])
    return runs


def _merge(runs, gap):
    if not runs:
        return runs
    out = [runs[0][:]]
    for s, e in runs[1:]:
        if s - out[-1][1] - 1 <= gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return out


def _frame_grid(sm: Image.Image):
    """One quantized frame image -> tight cell grid: flood-remove the edge background,
    drop lone speckle pixels, crop to content."""
    W, H = sm.size
    cor = _corners(sm)
    grid = [[sm.getpixel((x, y)) for x in range(W)] for y in range(H)]
    seen = [[False] * W for _ in range(H)]
    stack = [(x, 0) for x in range(W)] + [(x, H - 1) for x in range(W)] \
        + [(0, y) for y in range(H)] + [(W - 1, y) for y in range(H)]
    while stack:
        x, y = stack.pop()
        if not (0 <= x < W and 0 <= y < H) or seen[y][x]:
            continue
        seen[y][x] = True
        if _is_bg(grid[y][x], cor):
            grid[y][x] = None
            stack += [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        else:
            grid[y][x] = grid[y][x] if grid[y][x] else None
    for y in range(H):
        for x in range(W):
            if grid[y][x] is not None and seen[y][x] and _is_bg(grid[y][x], cor):
                grid[y][x] = None
    den = [[grid[y][x] for x in range(W)] for y in range(H)]
    for y in range(H):
        for x in range(W):
            if grid[y][x] is not None:
                nb = sum(1 for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                         if (dx or dy) and 0 <= y + dy < H and 0 <= x + dx < W and grid[y + dy][x + dx] is not None)
                if nb <= 1:
                    den[y][x] = None
    rows = [y for y in range(H) if any(den[y])]
    cols = [x for x in range(W) if any(den[y][x] for y in range(H))]
    if not rows or not cols:
        return None
    return [[den[y][x] for x in range(min(cols), max(cols) + 1)] for y in range(min(rows), max(rows) + 1)]


def _normalize(frames):
    """Pad frames to a common canvas, centred horizontally and bottom-aligned, so a multi-frame
    sheet plays as a stable in-place animation."""
    frames = [f for f in frames if f]
    if not frames:
        return frames
    mw = max(len(f[0]) for f in frames)
    mh = max(len(f) for f in frames)
    out = []
    for f in frames:
        h, w = len(f), len(f[0])
        ox, oy = (mw - w) // 2, mh - h
        canvas = [[None] * mw for _ in range(mh)]
        for y in range(h):
            for x in range(w):
                canvas[oy + y][ox + x] = f[y][x]
        out.append(canvas)
    return out


def pixelize_frames(img: Image.Image, target_h: int = TARGET_H, colors: int = COLORS):
    """SD render (often a horizontal sprite SHEET) -> a list of clean, aligned frame grids.
    Splits the sheet at empty background columns; a single-subject render yields one frame."""
    med = img.resize((256, 256), Image.BILINEAR).quantize(colors=colors, method=Image.MEDIANCUT).convert("RGB")
    cor = _corners(med)
    xs = [x for x in range(256) if any(not _is_bg(med.getpixel((x, y)), cor) for y in range(256))]
    ys = [y for y in range(256) if any(not _is_bg(med.getpixel((x, y)), cor) for x in range(256))]
    if not xs or not ys:
        return []
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    cw, ch = maxx - minx + 1, maxy - miny + 1
    if os.environ.get("CLAUDEMOVIES_SD_SHEET", "0") == "1":
        n = max(1, min(6, round((cw / ch) / 0.45)))
    else:
        n = 1
    W0, H0 = img.size
    sx, sy0, sy1 = W0 / 256.0, miny / 256.0 * H0, (maxy + 1) / 256.0 * H0
    frames = []
    for k in range(n):
        x0 = (minx + k * cw / n) * sx
        x1 = (minx + (k + 1) * cw / n) * sx
        crop = img.crop((int(x0), int(sy0), max(int(x0) + 1, int(x1)), int(sy1)))
        tw = max(8, round(target_h * crop.width / crop.height))
        sm = crop.resize((tw, target_h), Image.BILINEAR).quantize(colors=colors, method=Image.MEDIANCUT).convert("RGB")
        g = _frame_grid(sm)
        if g:
            frames.append(g)
    return _normalize(frames)


def pixelize(img: Image.Image, target_h: int = TARGET_H, colors: int = COLORS):
    """Back-compat: a single sprite cell grid (the first frame of the sheet)."""
    frames = pixelize_frames(img, target_h, colors)
    return frames[0] if frames else None


def _grid_to_spec(grid):
    keys, pal = {}, {}
    rows = []
    nextk = 0
    abc = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    for row in grid:
        s = ""
        for c in row:
            if c is None:
                s += "."
            else:
                hx = "%02x%02x%02x" % c
                if hx not in keys:
                    keys[hx] = abc[nextk % len(abc)]
                    pal[keys[hx]] = hx
                    nextk += 1
                s += keys[hx]
        rows.append(s)
    return {"w": len(grid[0]), "h": len(grid), "palette": pal, "rows": rows}


def _spec_to_grid(spec):
    pal = {k: (int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)) for k, v in spec["palette"].items()}
    return [[pal.get(ch) for ch in row] for row in spec["rows"]]


def frames_sd(label: str, desc: str = "", variant: int = 0):
    """All animation frames for a subject via LOCAL SD (the sprite sheet split into frames),
    cached to disk. `variant` (bucketed to CLAUDEMOVIES_SPRITE_VARIANTS looks) reseeds the
    render so different films get different designs of the same subject, each cached once.
    Returns a list of aligned cell grids, or None on failure."""
    nvar = max(1, int(os.environ.get("CLAUDEMOVIES_SPRITE_VARIANTS", "3")))
    b = variant % nvar
    key = label.lower().strip() + (f"#{b}" if b else "")
    if key in _CACHE:
        return [_spec_to_grid(s) for s in _CACHE[key]["frames"]]
    with _GEN_LOCK:
        if key in _CACHE:
            return [_spec_to_grid(s) for s in _CACHE[key]["frames"]]
        try:
            raw = generate_raw(f"{label} {desc}".strip(), seed=_seed(key))
            frames = pixelize_frames(raw)
        except Exception as e:
            import sys
            print(f"sdpixel: generation failed for {label!r}: {type(e).__name__}: {e}", file=sys.stderr)
            return None
        frames = [f for f in (frames or []) if f and any(c for row in f for c in row)]
        if not frames:
            return None
        _CACHE[key] = {"frames": [_grid_to_spec(f) for f in frames]}
        try:
            json.dump(_CACHE, open(_CACHE_PATH, "w"))
        except Exception:
            pass
        return frames


def draw_pixels_sd(label: str, desc: str = "", variant: int = 0):
    """A single sprite cell grid for a subject (first SD frame). None on failure."""
    frames = frames_sd(label, desc, variant=variant)
    return frames[0] if frames else None


SCENES = {
    "indoor": "dark medieval stone room interior with torches",
    "earth": "grassy plains under an open sky",
    "grass": "green rolling meadow landscape",
    "forest": "lush green forest clearing",
    "sand": "desert dunes landscape",
    "snow": "snowy winter landscape",
    "stone": "rocky mountain cavern",
    "road": "cobblestone road through a town",
    "water": "wide ocean surface under the sky",
    "lava": "volcanic cavern with rivers of lava",
    "sky": "bright sky with soft clouds",
}
_BG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bg_cache")


def _bg_load(path: str, w: int, h: int):
    img = Image.open(path).convert("RGB")
    return img if img.size == (w, h) else img.resize((w, h), Image.NEAREST)


def background_sd(setting: str, w: int = 320, h: int = 180, seed: int | None = None,
                  variant: int = 0):
    """A pixelized SD backdrop for a setting. `variant` picks between differently
    seeded renders of the same scene (cached per variant), so films don't all
    share one identical background per setting."""
    name = setting if not variant else f"{setting}_{variant}"
    path = os.path.join(_BG_DIR, f"{name}.png")
    if os.path.exists(path):
        return _bg_load(path, w, h)
    with _GEN_LOCK:
        if os.path.exists(path):
            return _bg_load(path, w, h)
        scene = SCENES.get(setting, "open landscape")
        img = generate_raw("bg:" + name, seed=seed if seed is not None else _seed("bg:" + name),
                           prompt=f"16bitscene, {scene}, detailed environment, no characters, no people")
        small = (img.resize((w // 2, h // 2), Image.BILINEAR)
                 .quantize(colors=32, method=Image.MEDIANCUT).convert("RGB")
                 .resize((w, h), Image.NEAREST))
        os.makedirs(_BG_DIR, exist_ok=True)
        small.save(path)
        return small


def main():
    import pixelsprites as PS
    ap = argparse.ArgumentParser(description="Local SD pixel-art sprite backend.")
    ap.add_argument("--label", required=True)
    ap.add_argument("--scale", type=int, default=8)
    args = ap.parse_args()

    raw = generate_raw(args.label)
    stem = args.label.replace(" ", "_")
    raw.save(f"_sd_raw_{stem}.png")
    frames = pixelize_frames(raw)
    PS.to_image(frames[0], scale=args.scale).save(f"_sd_px_{stem}.png")
    if len(frames) > 1:
        PS.save_gif([PS.to_image(f, scale=args.scale) for f in frames], f"_sd_anim_{stem}.gif", duration=160)
    print(f"wrote _sd_raw_{stem}.png (512) + _sd_px_{stem}.png  "
          f"({len(frames[0][0])}x{len(frames[0])} sprite, {len(frames)} frame(s))")


if __name__ == "__main__":
    main()
