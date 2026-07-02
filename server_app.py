"""
PIXEL CINEMA — CUSTOM frontend on gr.Server (Gradio's FastAPI API engine).

This is the "off-brand" build: instead of Gradio components, we serve our OWN
HTML/CSS/JS cinema and stream frames over Server-Sent Events from FastAPI routes
on a gr.Server instance. The film engine (movies + render) is reused as-is.

Run:    python server_app.py
Deploy: HF Space (set app_file: server_app.py). Same CLAUDEMOVIES_LLM_* secrets.
"""

import html
import json
import os
import re
import threading
import time
import uuid

import gradio as gr
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

import movies
import pixelscene
import render
import tts

W, H = render.W, render.H
HERE = os.path.dirname(__file__)
SHOW = os.path.join(HERE, "showcase")
MADE = os.path.join(HERE, "made")
POSTERS = os.path.join(HERE, "static", "posters")
os.makedirs(MADE, exist_ok=True)
os.makedirs(POSTERS, exist_ok=True)
_MADE_RE = re.compile(r"^[0-9a-f]{12}$")
_POSTER_LOCK = threading.Lock()
ADV: dict = {}
GENRES = ("comedy", "action", "adventure", "drama", "romance", "mystery",
          "spooky", "sci-fi", "fairy tale", "western", "noir", "slice of life")


def _genre(g):
    g = (g or "").strip().lower()
    return g if g in GENRES else ""

# Which renderer stages the film: the new PIXEL scene (default) or the original ASCII grid.
# Override per deploy with CLAUDEMOVIES_RENDER=ascii|pixel.
RENDER_MODE = os.environ.get("CLAUDEMOVIES_RENDER", "pixel").strip().lower()

# ── content filter (profanity / slurs / hate) ──
_BANNED = re.compile(r"\b(" + "|".join([
    "fuck", "fucks", "fucking", "fucker", "shit", "bitch", "cunt", "asshole", "bastard",
    "dick", "piss", "slut", "whore", "nigger", "nigga", "faggot", "fag", "kike", "spic",
    "chink", "gook", "wetback", "tranny", "retard", "coon", "dyke", "paki", "beaner",
    "heil hitler", "sieg heil", "white power", "kkk", "gas the jews",
]) + r")\w*", re.I)        # \w* catches suffix variants (shitty, fucking, dickhead); leading \b avoids embeds


def _blocked(t):
    return bool(_BANNED.search(t or ""))


_MAXLEN = 60
# common prompt-injection / jailbreak patterns — neutralised so nobody can steer the model
_INJECT = re.compile(
    r"(ignore\s+(all\s+|the\s+)?(previous|prior|above|earlier)"
    r"|disregard\s+(all|the|previous|prior)"
    r"|system\s+prompt|you\s+are\s+now|act\s+as\s+(a|an|if)"
    r"|jailbreak|developer\s+mode|override\s+(the|your)"
    r"|forget\s+(everything|all|the)|new\s+instructions)", re.I)


def _clean_concept(s):
    """Authoritative server-side sanitiser: strip control chars / markup, collapse whitespace,
    and HARD-cap to 60 chars so a single short film idea is all the model ever receives."""
    s = re.sub(r"[\x00-\x1f\x7f]", " ", s or "")        # control chars, newlines, tabs
    s = "".join(ch for ch in s if ch.isprintable())
    s = re.sub(r"[<>{}`\\]", "", s)                      # markup / structural chars
    s = re.sub(r"\s+", " ", s).strip()
    return s[:_MAXLEN]


def _films(folder):
    out = {}
    order = ["knight.json", "paper-boat.json", "rain-cat.json", "troll-dance.json",
             "snail-race.json", "ghost.json", "bread-dragon.json"]
    paths = sorted(__import__("glob").glob(os.path.join(folder, "*.json")),
                   key=lambda p: (order.index(os.path.basename(p)) if os.path.basename(p) in order else 99,
                                  os.path.basename(p)))
    for p in paths:
        try:
            out[json.load(open(p)).get("title") or os.path.basename(p)[:-5]] = p
        except Exception:
            continue
    return out


def grid_to_html(grid, title=""):
    rows = []
    for r in range(H):
        cells, row, i = grid[r], "", 0
        while i < W:
            col, j, seg = cells[i][1], i, ""
            while j < W and cells[j][1] == col:
                seg += cells[j][0]
                j += 1
            hexc = "#%02x%02x%02x" % col if isinstance(col, tuple) else "#7df9a6"
            esc = html.escape(seg)
            row += f"<span style='color:{hexc}'>{esc}</span>" if seg.strip() else esc
            i = j
        rows.append(row)
    bar = (f"<div class='bar'><i></i><i></i><i></i><b>now_playing: {html.escape(title)}</b></div>")
    return f"{bar}<pre>{chr(10).join(rows)}</pre>"


def _tint(grid):
    """The frame's lit colour and 'energy' — a saturation/brightness-weighted average of the
    drawn cells, so the audience glow becomes a low-opacity MIRROR of the action on screen.
    Energy is keyed to how much is lit, so sparse frames (the title/credit cards) glow ~0."""
    tr = tg = tb = tw = 0.0
    lit = 0
    for row in grid:
        for cell in row:
            ch = cell[0]
            if ch == " " or ch == "":
                continue
            r, g, b = cell[1]
            lit += 1
            mx, mn = max(r, g, b), min(r, g, b)
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            sat = (mx - mn) / mx if mx else 0.0
            w = lum * (0.35 + sat)                       # vivid + bright cells lead the colour
            tr += r * w; tg += g * w; tb += b * w; tw += w
    cov = lit / float(W * H)
    energy = max(0.0, min(1.0, (cov - 0.10) * 6.0))      # cards are sparse -> ~0 -> no glow
    if tw <= 0:
        return [150, 180, 235], 0.0
    return [int(tr / tw), int(tg / tw), int(tb / tw)], round(energy, 3)


def _payload(grid, title=""):
    tint, energy = _tint(grid)
    return {"html": grid_to_html(grid, title), "tint": tint, "lvl": energy}


def _pix_payload(uri, tint, lvl, shot=None, total=None):
    """A pixel-scene frame: a PNG data-URI shown as an <img>, plus the same reactive-glow
    fields (tint/lvl) the ASCII frames carry, so the cinema lighting still mirrors the
    action — and the shot index so the player can show progress and cue sounds."""
    p = {"html": "<img class=pixfilm src='" + uri + "'>", "pix": True, "tint": tint, "lvl": lvl}
    if shot is not None:
        p["shot"], p["shots"] = shot, total
    return p


def _film_payloads(spec, end=True):
    """Yield SSE frame payloads for a film in the configured render mode (pixel | ascii)."""
    if RENDER_MODE == "ascii":
        for grid in render.iter_movie_frames(spec):
            yield _payload(grid, spec.get("title", ""))
    else:
        total = len(spec.get("shots", []))
        for uri, tint, lvl, shot in pixelscene.iter_film_frames(spec, scale=1, end=end):
            yield _pix_payload(uri, tint, lvl, shot, total)


def _prewarm_payloads(spec):
    """Turn asset prewarm events into credits-screen SSE lines (casting / set building)."""
    for ev in pixelscene.iter_prewarm(spec):
        if not isinstance(ev, dict):
            yield {"phase": "writing"}
        elif ev.get("kind") == "sprite":
            p = {"phase": "cast", "label": str(ev.get("label") or "")[:40]}
            if ev.get("img"):
                p["img"] = ev["img"]
            yield p
        else:
            yield {"phase": "set", "label": str(ev.get("label") or "")[:40]}


def _speaker(sh):
    """Which cast member delivers the shot's dialogue — mirrors the pixel stage's
    rule: the first member with a mood, else the first member."""
    cast = [c for c in (sh.get("cast") or [])][:3] or ["hero"]
    moods = sh.get("mood") if isinstance(sh.get("mood"), list) else []
    for i, c in enumerate(cast):
        if i < len(moods) and str(moods[i] or "").strip():
            return c
    return cast[0]


def _film_payloads_with_audio(spec, end=True):
    """Frames + per-shot audio. The narrator (male, CLAUDEMOVIES_TTS_VOICE) reads the
    narration, then the speaking character delivers its dialogue in its own stable
    voice. Kokoro synthesis runs in a background thread (each line cached on disk);
    finished clips are interleaved into the frame stream as {"audio": ..., "shot": i}."""
    if RENDER_MODE == "ascii" or not tts.available():
        yield from _film_payloads(spec, end)
        return
    results: dict = {}

    def worker():
        for i, sh in enumerate(spec.get("shots", [])):
            segs = []
            narr = (sh.get("narration") or "").strip()
            dlg = (sh.get("dialogue") or "").strip().strip('"')
            if narr:
                segs.append((narr, ""))
            if dlg:
                segs.append((dlg, tts.voice_for(_speaker(sh))))
            results[i] = tts.say_segments(segs) if segs else None
    threading.Thread(target=worker, daemon=True).start()
    sent = set()

    def flush():
        for i in sorted(results.keys()):
            if i not in sent:
                sent.add(i)
                if results[i]:
                    yield {"audio": results[i], "shot": i}
    for p in _film_payloads(spec, end):
        yield from flush()
        yield p
    yield from flush()


def _meta_payload(spec, made_id=None):
    m = {"title": spec.get("title", ""), "shots": len(spec.get("shots", []))}
    if made_id:
        m["id"] = made_id
    return {"meta": m}


def _save_made(spec):
    mid = uuid.uuid4().hex[:12]
    try:
        json.dump(spec, open(os.path.join(MADE, mid + ".json"), "w"))
        return mid
    except Exception:
        return None


def _screen(msg, label="ready"):
    bar = f"<div class='bar'><i></i><i></i><i></i><b>{html.escape(label)}</b></div>"
    return f"{bar}<pre class='msg'>{html.escape(msg)}</pre>"


def _sse(frames):
    try:
        for fr in frames:
            payload = fr if isinstance(fr, dict) else {"html": fr}
            yield "data: " + json.dumps(payload) + "\n\n"
    except Exception:
        import sys
        import traceback
        traceback.print_exc(file=sys.stderr)
    finally:
        yield "data: " + json.dumps({"done": True}) + "\n\n"


# ── the custom frontend: a real cinema-hall photo as the backdrop, the ASCII film
#    projected onto the screen, and all controls docked along the bottom ──
PAGE = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>PIXEL CINEMA</title>
<meta name=description content="A small model writes, draws and directs original ~90-second ASCII films.">
<meta property="og:title" content="PIXEL CINEMA">
<meta property="og:description" content="A small model writes, draws and directs original pixel films.">
<meta property="og:type" content="website">
<meta property="og:image" content="static/share.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="PIXEL CINEMA">
<meta name="twitter:description" content="A small model writes, draws and directs original pixel films.">
<meta name="twitter:image" content="static/share.png">
<link rel=preconnect href=https://fonts.googleapis.com>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap" rel=stylesheet>
<style>
:root{--accent:#3ddc97;--accent2:#36c6e0;--ink:#e7ebf0}
*{box-sizing:border-box;margin:0}
html,body{height:100%;font-family:'JetBrains Mono',ui-monospace,monospace;background:#04060a;color:var(--ink);overflow:hidden}
/* the photo (with a transparent screen cut-out) sits ON TOP; the film screen is BEHIND
   it and shows through the hole — pixel-perfect. While a film plays we dim the PHOTO itself
   (its transparent hole stays clear so the screen stays bright) and pool soft light below. */
#bg{position:fixed;inset:0;width:100%;height:100%;object-fit:cover;z-index:1;
  filter:brightness(.62) saturate(.9);transition:filter 1.4s ease}
/* while a film plays the whole room goes really dark */
body.playing #bg{filter:brightness(.3) saturate(.78)}
/* the projection, BEHIND the photo, fills the cut-out hole */
#screen{position:fixed;left:22.3%;top:11.1%;width:57.7%;height:43.9%;z-index:0;background:#04060a;
  display:flex;align-items:center;justify-content:center;overflow:hidden}
#screen .bar{display:none}
/* REVEAL: a second copy of the photo at near-full brightness, shown ONLY through a soft,
   very blurry blob rising from UNDER the screen. The projector light "uncovers" the dark
   room (reveals the photo's own opacity) rather than painting white on top of it. */
#reveal{position:fixed;inset:0;z-index:2;pointer-events:none;opacity:0;
  background:url(static/cinema.png) center/cover no-repeat;
  filter:brightness(1.12) saturate(1.04) blur(8px);
  -webkit-mask-image:
     linear-gradient(to bottom, transparent 0 58%, #000 80%),
     radial-gradient(56% 48% at 51% 75%, #000 0%, rgba(0,0,0,.5) 44%, transparent 80%);
  -webkit-mask-composite:source-in;
  mask-image:
     linear-gradient(to bottom, transparent 0 58%, #000 80%),
     radial-gradient(56% 48% at 51% 75%, #000 0%, rgba(0,0,0,.5) 44%, transparent 80%);
  mask-composite:intersect;
  transition:opacity 1.4s ease, filter .22s linear}
/* the colour-mirror bloom over the reveal — JS sets its colour per frame (transparent until then) */
#spill{position:fixed;inset:0;z-index:3;pointer-events:none;opacity:0;mix-blend-mode:screen;
  background:transparent;filter:blur(40px);transition:opacity 1.4s ease, filter .22s linear;
  -webkit-mask-image:linear-gradient(to bottom, transparent 0 58%, #000 76%);
  mask-image:linear-gradient(to bottom, transparent 0 58%, #000 76%)}
body.playing #reveal{opacity:1}
body.playing #spill{opacity:1}
#brand{position:fixed;top:16px;left:20px;z-index:4;font-weight:700;letter-spacing:.06em;font-size:15px;
  text-shadow:0 1px 8px #000}#brand b{color:var(--accent)}
#screen pre{margin:0;white-space:pre;text-align:center;color:#cfd6df;line-height:1.12;
  font-size:clamp(11px,1.5vw,19px);text-shadow:none}
#screen pre.msg{color:var(--ink)}
/* pixel-render frames: a PNG that fills the screen box, crisp (no smoothing) */
#screen img.pixfilm{display:block;width:100%;height:100%;object-fit:contain;
  image-rendering:pixelated;image-rendering:crisp-edges}
/* opening-credits sequence, projected on the screen while the film renders */
#credwrap{width:100%;height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:20px;text-align:center;padding:0 6%;opacity:0;transition:opacity 1.1s ease;color:#eef2f6}
#credwrap.show{opacity:1}
.cr-s{font-size:clamp(10px,1.45vw,17px);letter-spacing:.42em;text-transform:uppercase;color:#cfd6df;line-height:1.9}
.cr-s b{color:#fff;font-weight:700}
.cr-big{font-size:clamp(30px,6.6vw,82px);font-weight:700;letter-spacing:.07em;line-height:.96;
  text-shadow:0 0 42px rgba(238,242,246,.22)}
/* the ANSI-shadow ASCII title — WHITE, sized per context (#screen pre would otherwise
   force its green film-frame style, so these need the #screen id for specificity) */
#screen pre.cr-title{margin:0;white-space:pre;line-height:1.0;color:#eef2f6;text-align:center;
  font-size:clamp(4px,0.78vw,9px);text-shadow:0 0 26px rgba(238,242,246,.2)}
#screen pre.wel-title{margin:0;white-space:pre;line-height:1.0;color:#eef2f6;text-align:center;
  font-size:clamp(4px,0.72vw,8px);text-shadow:0 0 16px rgba(238,242,246,.16)}
#screen pre.cr-art{font-size:clamp(7px,1.05vw,13px);line-height:1.05;color:#8b94a2;margin-top:4px;
  white-space:pre;text-shadow:none}
.cr-sub{font-size:clamp(9px,1.2vw,14px);letter-spacing:.5em;text-transform:uppercase;color:#9aa3b0;
  animation:lpulse 1.8s ease-in-out infinite}
/* the idle "now showing" welcome card on the theatre screen */
.scene-center{width:100%;height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:14px;text-align:center;padding:0 6%;color:#eef2f6}
.w-dir{font-size:clamp(8px,1.15vw,13px);color:#aab2bd;letter-spacing:.03em;line-height:1.7;max-width:84%;margin-top:2px}
.w-dir .wlink{color:#cfd6df;text-decoration:underline}.w-dir .wlink:hover{color:#fff}
/* controls docked along the bottom, over the audience */
#controls{position:fixed;left:0;right:0;bottom:0;z-index:4;padding:18px 22px 20px;
  background:linear-gradient(to top,rgba(4,6,10,.97),rgba(4,6,10,.78) 65%,transparent)}
#crow{display:flex;gap:10px;align-items:center;max-width:1120px;margin:0 auto 10px;width:100%}
#concept{flex:1;min-width:0;background:rgba(11,14,18,.92);border:1px solid #2b3540;border-radius:11px;
  color:var(--ink);padding:12px 15px;font:14px 'JetBrains Mono',monospace}
#concept:focus{outline:none;border-color:var(--accent)}
#controls button{border:none;border-radius:11px;padding:12px 18px;cursor:pointer;white-space:nowrap;
  font:700 13px 'JetBrains Mono',monospace;color:var(--ink);background:#222b36}
#controls button:hover{filter:brightness(1.18)}
#go{background:linear-gradient(92deg,var(--accent),var(--accent2));color:#04130d}
#stop{color:#ff8a84;background:#16191f;border:1px solid #3a2a2c}
#genre{background:rgba(11,14,18,.92);border:1px solid #2b3540;border-radius:11px;color:var(--ink);
  padding:12px 10px;font:13px 'JetBrains Mono',monospace;cursor:pointer;max-width:150px}
#genre:focus{outline:none;border-color:var(--accent)}
#shelves{display:flex;gap:14px;max-width:1120px;margin:0 auto;width:100%;align-items:flex-start}
.shelf{flex:1;min-width:0}
.shelf h4{margin:0 0 6px;font-size:10px;letter-spacing:.22em;text-transform:uppercase;
  color:#7d8896;font-weight:700}
.shelf .list{display:flex;flex-direction:column;gap:6px;max-height:128px;overflow-y:auto;
  padding-right:4px;scrollbar-width:thin}
.shelf .list::-webkit-scrollbar{width:6px}
.shelf .list::-webkit-scrollbar-thumb{background:#2a3340;border-radius:3px}
.shelf .film{display:flex;align-items:center;gap:10px;width:100%;text-align:left;
  background:rgba(12,15,20,.8);border:1px solid #2b3540;color:#aeb6c2;
  border-radius:9px;padding:5px 8px;font-size:12px;cursor:pointer}
.shelf .film:hover{color:#fff;border-color:var(--accent)}
.shelf .film img{width:64px;height:36px;border-radius:4px;display:block;flex:0 0 auto;
  image-rendering:pixelated;background:#0a0e14}
.shelf .film span{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#curtains{position:fixed;z-index:2;pointer-events:none;overflow:hidden}
#curtains .cl,#curtains .cr{position:absolute;top:0;bottom:0;width:52%;
  background:linear-gradient(90deg,#081824,#123245 26%,#0a2130 52%,#123245 78%,#081824);
  box-shadow:0 0 30px rgba(0,0,0,.6);transition:transform 1.35s cubic-bezier(.65,.05,.35,1)}
#curtains .cl{left:0;transform:translateX(-103%)}
#curtains .cr{right:0;transform:translateX(103%)}
#curtains.shut .cl,#curtains.shut .cr{transform:none}
#marquee{position:fixed;z-index:4;text-align:center;font:700 12px 'JetBrains Mono',monospace;
  letter-spacing:.26em;color:#e8c66a;text-shadow:0 0 16px rgba(232,198,106,.35);opacity:0;
  transition:opacity .9s ease;pointer-events:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
body.playing #marquee{opacity:1}
#choices{position:fixed;z-index:5;display:none;flex-direction:column;align-items:center;
  justify-content:flex-end;gap:8px;padding-bottom:16px}
#choices.on{display:flex}
#choices .chead{font-size:12px;letter-spacing:.32em;text-transform:uppercase;color:#eef2f6;
  text-shadow:0 1px 10px #000}
#choices .chb{background:rgba(10,14,20,.94);border:1px solid var(--accent);color:#dfe6ee;
  border-radius:9px;padding:9px 16px;font:600 12px 'JetBrains Mono',monospace;cursor:pointer;max-width:86%}
#choices .chb:hover{background:var(--accent);color:#04130d}
#progrow{display:none;align-items:center;gap:10px;max-width:1120px;margin:0 auto 8px;width:100%}
body.playing #progrow{display:flex}
#pbar{flex:1;height:5px;background:#1a222c;border-radius:3px;cursor:pointer;overflow:hidden}
#pfill{height:100%;width:0;background:linear-gradient(90deg,var(--accent),var(--accent2))}
#shotlab{font-size:11px;color:#93a0ad;min-width:52px;text-align:right}
#queuechip{max-width:1120px;margin:0 auto;width:100%;font-size:11px;color:#8b96a4}
#queuechip:not(:empty){margin-bottom:6px}
#credcast{display:flex;gap:16px;justify-content:center;flex-wrap:wrap;margin-top:2px}
#credcast .cred{display:flex;flex-direction:column;align-items:center;gap:5px;font-size:9px;
  letter-spacing:.16em;text-transform:uppercase;color:#aab4c0}
#credcast .cred img{height:42px;image-rendering:pixelated}
#pause,#poster,#mute,#fs{padding:12px 12px}
/* ── mobile: stack the input over the buttons so nothing is cut off, and keep clear of the
   phone's home indicator with the safe-area inset ── */
@media (max-width:680px){
  #marquee{display:none}
  #shelves{flex-direction:column;gap:8px}
  .shelf .list{max-height:92px}
  .shelf .film img{width:52px;height:30px}
  #genre{flex:1 1 100%;max-width:none;font-size:12px;padding:10px}
  #choices .chb{font-size:11px;padding:8px 10px}
  /* on mobile, placeScreen() zooms the photo so the theatre screen fills the width and pins
     #screen to it (nothing cut off). The photo-reveal glow assumes desktop cover, so hide it. */
  #reveal{display:none}
  /* the theatre screen is short & wide on mobile — shrink intro content so it fits, not clips */
  .scene-center{gap:7px;padding:0 4%}
  #screen pre.wel-title,#screen pre.cr-title{font-size:clamp(3px,1.5vw,6px)}
  #screen pre.cr-art{display:none}                 /* drop the decorative reel on mobile */
  .cr-s{font-size:9px;letter-spacing:.22em;line-height:1.45}
  .cr-sub{font-size:8px;letter-spacing:.26em}
  .w-dir{font-size:8.5px;line-height:1.4;max-width:94%}
  #controls{padding:10px 10px calc(10px + env(safe-area-inset-bottom,0px))}
  #crow{flex-wrap:wrap;gap:7px;margin-bottom:8px}
  #concept{flex:1 1 100%;font-size:13px;padding:11px 13px;border-radius:9px}
  #controls button{flex:1 1 auto;padding:11px 8px;font-size:12px;border-radius:9px}
  #brand{font-size:12px;top:10px;left:12px}
  #gallery{gap:6px}
  #gallery .film{font-size:11px;padding:7px 10px}
}
/* full-screen loader so the scene reveals all at once (no elements popping in) */
#loader{position:fixed;inset:0;z-index:99;background:#04060a;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:20px;text-align:center;transition:opacity 1.1s ease}
#loader.hide{opacity:0;pointer-events:none}
#loader .lsmall{color:#eef2f6;font-size:12px;letter-spacing:.55em;text-transform:uppercase;
  animation:lpulse 1.8s ease-in-out infinite}
#loader .lbig{font-weight:700;letter-spacing:.04em;color:#eef2f6;
  text-shadow:0 0 34px rgba(238,242,246,.22)}
/* the ANSI-shadow ASCII title (PIXEL CINEMA) — rendered as a monospace block */
pre#loadtitle{margin:0;white-space:pre;line-height:1.0;font-size:clamp(5px,1.0vw,11px);
  text-align:center}
#loader .lbar{width:min(280px,52vw);height:2px;background:#161b22;border-radius:2px;overflow:hidden}
#loader .lbar::after{content:"";display:block;height:100%;width:42%;
  background:linear-gradient(90deg,transparent,#eef2f6,transparent);animation:lslide 1.25s linear infinite}
@keyframes lpulse{0%,100%{opacity:.45}50%{opacity:1}}
@keyframes lslide{0%{transform:translateX(-120%)}100%{transform:translateX(330%)}}
</style></head><body>
<div id=loader>
  <pre id=loadtitle class=lbig>PIXEL CINEMA</pre>
  <div class=lbar></div>
</div>
<div id=screen></div>
<img id=bg src="static/cinema.png" alt="">
<div id=curtains><div class=cl></div><div class=cr></div></div>
<div id=reveal></div>
<div id=spill></div>
<div id=marquee></div>
<div id=choices></div>
<div id=brand>&#9654; PIXEL <b>CINEMA</b></div>
<div id=controls>
  <div id=progrow><div id=pbar><div id=pfill></div></div><span id=shotlab></span></div>
  <div id=crow>
    <input id=concept maxlength=60 autocomplete=off spellcheck=false
      placeholder="Describe your film — e.g. a tiny knight afraid of the dark">
    <select id=genre>
      <option value="">any genre</option>
      <option>comedy</option><option>action</option><option>adventure</option>
      <option>drama</option><option>romance</option><option>mystery</option>
      <option>spooky</option><option>sci-fi</option><option>fairy tale</option>
      <option>western</option><option>noir</option><option>slice of life</option>
    </select>
    <button id=go>&#9654; Make &amp; play</button>
    <button id=adv>&#8943; Adventure</button>
    <button id=surprise>Surprise</button>
    <button id=replay>&#8635; Replay</button>
    <button id=pause>&#9208; Pause</button>
    <button id=stop>Stop</button>
    <button id=poster>Poster</button>
    <button id=mute>Sound: on</button>
    <button id=fs>&#9974;</button>
  </div>
  <div id=queuechip></div>
  <div id=shelves>
    <div class=shelf id=galshelf><h4>tonight's programme</h4><div class=list id=gallery></div></div>
    <div class=shelf id=myshelf style="display:none"><h4>your films</h4><div class=list id=myrow></div></div>
  </div>
</div>
<script>
const screen=document.getElementById('screen');
let GLYPH=null;
function asciiWord(t){if(!GLYPH)return t;t=t.toUpperCase();
  let rows=[];for(let r=0;r<6;r++){let line="";
    for(const c of t){line+= c===' ' ? '    ' : (GLYPH[c]?GLYPH[c][r]:'    ');}rows.push(line);}
  return rows.join('\n');}
function asciiTitle(){return asciiWord('PIXEL')+'\n'+asciiWord('CINEMA');}
function paintLoaderTitle(){const el=document.getElementById('loadtitle');if(el&&GLYPH)el.textContent=asciiTitle();}
fetch('static/ansifont.json').then(r=>r.json()).then(g=>{GLYPH=g;paintLoaderTitle();
  if(!es&&!playTimer&&!phase)welcome();}).catch(()=>{});
const SURPRISES=["a tiny knight who is afraid of the dark","a lonely robot who finds a stray cat",
"a dragon who would rather bake than fight","a snail who dreams of racing the wind",
"a little ghost afraid of humans","a candle racing the dawn"];
let es=null;
const loader=document.getElementById('loader');
let bgReady=false,galReady=false,minDone=false;
function reveal(){if(bgReady&&galReady&&minDone){loader.classList.add('hide');
  setTimeout(()=>{loader.style.display='none';},1200);}}
(function(){const b=document.getElementById('bg');
  if(b.complete&&b.naturalWidth>0)bgReady=true; else b.addEventListener('load',()=>{bgReady=true;reveal();});
  setTimeout(()=>{minDone=true;reveal();},1600);
  setTimeout(()=>{bgReady=galReady=minDone=true;reveal();},7000);})();
const revealEl=document.getElementById('reveal'), spillEl=document.getElementById('spill'),
  curtainsEl=document.getElementById('curtains'), marqueeEl=document.getElementById('marquee'),
  choicesEl=document.getElementById('choices'), pfill=document.getElementById('pfill'),
  shotlab=document.getElementById('shotlab'), pbar=document.getElementById('pbar'),
  pauseBtn=document.getElementById('pause'), queueEl=document.getElementById('queuechip');
let _fs=null;
function fit(force){const p=screen.querySelector('pre'); if(!p||p.classList.contains('msg'))return;
  if(_fs&&!force){p.style.fontSize=_fs+'px';return;}
  p.style.fontSize='10px';
  const cw=p.scrollWidth/10, ch=p.scrollHeight/10;
  if(!cw||!ch)return;
  _fs=(Math.min(screen.clientWidth/cw, screen.clientHeight/ch)*0.99).toFixed(2);
  p.style.fontSize=_fs+'px';}
const SX=596,SY=172,SW=1507,SH=636,IW=2638,IH=1478;
function placeAt(l,t,w,h){screen.style.left=l+'px';screen.style.top=t+'px';
  screen.style.width=w+'px';screen.style.height=h+'px';
  [curtainsEl,choicesEl].forEach(el=>{el.style.left=l+'px';el.style.top=t+'px';
    el.style.width=w+'px';el.style.height=h+'px';});
  marqueeEl.style.left=l+'px';marqueeEl.style.width=w+'px';marqueeEl.style.top=Math.max(4,t-26)+'px';}
function placeScreen(){const bg=document.getElementById('bg'),vw=window.innerWidth,vh=window.innerHeight;
  if(vw<=680){
    const s=(vw*0.96)/SW, lx=vw*0.02, ty=vh*0.16;
    bg.style.position='fixed';bg.style.objectFit='fill';bg.style.right='auto';bg.style.bottom='auto';
    bg.style.width=(IW*s)+'px';bg.style.height=(IH*s)+'px';
    bg.style.left=(lx-SX*s)+'px';bg.style.top=(ty-SY*s)+'px';
    placeAt(lx,ty,SW*s,SH*s);
  }else{
    bg.style.position='';bg.style.objectFit='';bg.style.right='';bg.style.bottom='';
    bg.style.width='';bg.style.height='';bg.style.left='';bg.style.top='';
    const s=Math.max(vw/IW,vh/IH), ox=(vw-IW*s)/2, oy=(vh-IH*s)/2;
    placeAt(ox+SX*s,oy+SY*s,SW*s,SH*s);
  }}
placeScreen();
window.addEventListener('resize',()=>{placeScreen();_fs=null;if(screen.querySelector('pre'))fit(true);});
function show(h){screen.innerHTML=h;fit();}
function curt(open){curtainsEl.classList.toggle('shut',!open);}
function playing(on){document.body.classList.toggle('playing',on);
  if(!on){revealEl.style.opacity='';revealEl.style.filter='';spillEl.style.background='';}}
let AC=null,master=null,humGain=null,muted=localStorage.getItem('pc_mute')==='1';
function initAudio(){if(AC)return;
  try{AC=new (window.AudioContext||window.webkitAudioContext)();
    master=AC.createGain();master.gain.value=muted?0:1;master.connect(AC.destination);
    const len=AC.sampleRate*2,nb=AC.createBuffer(1,len,AC.sampleRate),ch=nb.getChannelData(0);
    for(let i=0;i<len;i++)ch[i]=Math.random()*2-1;
    const src=AC.createBufferSource();src.buffer=nb;src.loop=true;
    const lp=AC.createBiquadFilter();lp.type='lowpass';lp.frequency.value=130;
    humGain=AC.createGain();humGain.gain.value=0;
    src.connect(lp);lp.connect(humGain);humGain.connect(master);src.start();
  }catch(e){AC=null;}}
function hum(on){if(AC&&humGain)humGain.gain.linearRampToValueAtTime(on?0.045:0,AC.currentTime+0.8);}
function blip(f,dur,vol){if(!AC)return;
  try{const o=AC.createOscillator(),g=AC.createGain();o.type='triangle';o.frequency.value=f;
    g.gain.setValueAtTime(vol,AC.currentTime);g.gain.exponentialRampToValueAtTime(0.0001,AC.currentTime+dur);
    o.connect(g);g.connect(master);o.start();o.stop(AC.currentTime+dur);}catch(e){}}
function endChord(){[262,330,392].forEach((f,i)=>setTimeout(()=>blip(f,1.2,0.06),i*90));}
function paintMute(){document.getElementById('mute').textContent=muted?'Sound: off':'Sound: on';}
document.getElementById('mute').onclick=()=>{muted=!muted;localStorage.setItem('pc_mute',muted?'1':'0');
  if(master)master.gain.value=muted?0:1;paintMute();};
paintMute();
function welcomeHTML(){const title=GLYPH?("<pre class='wel-title'>"+asciiTitle()+"</pre>")
    :"<div class='cr-big'>PIXEL CINEMA</div>";
  return "<div class='scene-center'>"+title
    +"<div class='w-dir'>Type an idea below and press &ldquo;Make &amp; play&rdquo; &mdash; a small model will "
    +"write, draw and direct your film. &ldquo;Adventure&rdquo; makes it interactive: you pick what happens next. "
    +"Or pick one from the gallery. "
    +"You can also <a class='wlink' href='https://huggingface.co/spaces/build-small-hackathon/llm-cinema/blob/main/LOCAL_LLAMACPP.md' target='_blank' rel='noopener'>run it off the grid, locally on llama.cpp</a>.</div></div>";}
function welcome(){playing(false);hideChoices();curt(true);hum(false);marqueeEl.textContent='';
  screen.innerHTML=welcomeHTML();}
function esc(t){return String(t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function renderFilmFrame(d){show(d.html);playing(true);
  if(typeof d.shot==='number'&&d.shot!==lastShot){
    if(lastShot>=0){if(curShots&&d.shot>=curShots)endChord();else blip(1180,0.05,0.028);}
    lastShot=d.shot;
    if(!curShots||d.shot<curShots)playNarr(d.shot);else stopNarr();}
  if(d.tint){const r=d.tint[0],g=d.tint[1],b=d.tint[2],e=(d.lvl||0);
    const a=(0.15*e).toFixed(3), a2=(0.05*e).toFixed(3);
    spillEl.style.background='radial-gradient(76% 56% at 51% 70%, rgba('+r+','+g+','+b+','+a+') 0%, '
      +'rgba('+r+','+g+','+b+','+a2+') 46%, transparent 78%)';
    revealEl.style.opacity=Math.min(0.7,e*1.4).toFixed(3);
    revealEl.style.filter='brightness('+(1.06+0.12*e).toFixed(3)+') saturate(1.05) blur(8px)';}}
let castLog=[],writeMsg='';
function writingHTML(){const t=GLYPH?("<pre class='cr-title'>"+asciiTitle()+"</pre>"):"<div class='cr-big'>PIXEL CINEMA</div>";
  const items=castLog.slice(-6).map(c=>"<span class='cred'>"+(c.img?"<img src='"+c.img+"'>":"")
    +esc(c.label)+"</span>").join('');
  return t+"<div class='cr-sub'>"+esc(writeMsg)+"</div>"+(items?"<div id=credcast>"+items+"</div>":"");}
function showCard(inner){screen.innerHTML="<div id=credwrap>"+inner+"</div>";
  const w=document.getElementById('credwrap');requestAnimationFrame(()=>{if(w)w.classList.add('show');});}
function fadeCard(){const w=document.getElementById('credwrap');if(w)w.classList.remove('show');}
function startWriting(msg){phase='writing';playing(true);curt(true);castLog=[];writeMsg=msg;
  revealEl.style.opacity='0';spillEl.style.background='';
  showCard(writingHTML());}
function updWriting(msg){if(phase!=='writing')return;writeMsg=msg||writeMsg;
  const w=document.getElementById('credwrap');
  if(w){w.innerHTML=writingHTML();w.classList.add('show');}else showCard(writingHTML());}
const BUMPER=["<div class='cr-s'>PIXEL&nbsp;CINEMA<br><b>presents</b></div>"];
function playBumper(done){let i=0;
  (function step(){
    if(i>=BUMPER.length){fadeCard();creditsTimer=setTimeout(done,900);return;}
    showCard(BUMPER[i]);
    creditsTimer=setTimeout(()=>{fadeCard();creditsTimer=setTimeout(()=>{i++;step();},700);},2200);})();}
let phase=null,buf=[],playTimer=null,creditsTimer=null,streamDone=false;
let paused=false,idx=0,lastShot=-1,curTitle='',curShots=0,pendingBranch=null,advId=null;
let queue=[],audioByShot={},narrBufs={},narrSrc=null;
function stopNarr(){if(narrSrc){try{narrSrc.stop();}catch(e){}narrSrc=null;}}
function playNarr(s){if(!AC||!(s in audioByShot))return;const u=audioByShot[s];if(!u)return;
  stopNarr();
  const go=b=>{if(lastShot!==s||paused)return;const src=AC.createBufferSource();src.buffer=b;
    const g=AC.createGain();g.gain.value=0.9;src.connect(g);g.connect(master);src.start();narrSrc=src;};
  if(narrBufs[s]){go(narrBufs[s]);return;}
  fetch(u).then(r=>r.arrayBuffer()).then(ab=>AC.decodeAudioData(ab))
    .then(b=>{narrBufs[s]=b;go(b);}).catch(()=>{});}
function curtainReveal(){phase='reveal';fadeCard();curt(false);
  creditsTimer=setTimeout(()=>{if(buf.length)show(buf[0].html);
    creditsTimer=setTimeout(()=>{curt(true);
      creditsTimer=setTimeout(startBufferedPlay,600);},400);},1500);}
function updQueue(){queueEl.textContent=queue.length?('next up: '+queue.map(q=>q.c).join(' · ')):'';}
function updProg(){if(!buf.length){pfill.style.width='0%';shotlab.textContent='';return;}
  const j=Math.max(0,Math.min(idx,buf.length-1)),d=buf[j];
  pfill.style.width=(100*Math.min(1,(j+1)/buf.length)).toFixed(1)+'%';
  shotlab.textContent=(d&&typeof d.shot==='number'&&curShots)
    ?(Math.min(d.shot+1,curShots)+'/'+curShots):'';}
function paintPause(){pauseBtn.innerHTML=paused?'&#9654; Resume':'&#9208; Pause';}
function togglePause(){if(!playTimer)return;paused=!paused;paintPause();hum(!paused);
  if(AC){try{paused?AC.suspend():AC.resume();}catch(e){}}}
function finishPlayback(){if(playTimer){clearInterval(playTimer);playTimer=null;}
  phase=null;hum(false);updProg();
  if(pendingBranch){const br=pendingBranch;pendingBranch=null;showChoices(br);return;}
  playing(false);curt(true);
  if(queue.length){const q=queue.shift();updQueue();startMake(q.c,q.g);}}
function playTick(){if(paused)return;
  if(idx<buf.length){renderFilmFrame(buf[idx++]);updProg();}
  else if(streamDone)finishPlayback();}
function startBufferedPlay(){phase='playing';idx=0;paused=false;lastShot=-1;paintPause();
  curt(true);hum(true);
  playTimer=setInterval(playTick,150);}
function seek(ev){if(!buf.length)return;
  const r=pbar.getBoundingClientRect(),f=Math.max(0,Math.min(1,(ev.clientX-r.left)/r.width));
  idx=Math.floor(f*buf.length);lastShot=-1;stopNarr();
  if(!playTimer)renderFilmFrame(buf[Math.min(idx,buf.length-1)]);
  updProg();}
pbar.addEventListener('click',seek);
function stopStream(){if(es){es.close();es=null;}
  if(playTimer){clearInterval(playTimer);playTimer=null;}
  if(creditsTimer){clearTimeout(creditsTimer);creditsTimer=null;}}
function stop(){stopStream();stopNarr();if(AC){try{AC.resume();}catch(e){}}
  phase=null;paused=false;pendingBranch=null;queue=[];updQueue();welcome();}
function showChoices(br){advId=br.id;const opts=(br.choices||[]).slice(0,3);
  if(!opts.length){playing(false);curt(true);return;}
  choicesEl.innerHTML="<div class='chead'>What happens next?</div>"
    +opts.map(o=>"<button class='chb' data-c='"+encodeURIComponent(o.label)+"'>"
      +esc(o.label)+"</button>").join('');
  choicesEl.classList.add('on');
  choicesEl.querySelectorAll('.chb').forEach(b=>{b.onclick=()=>{hideChoices();
    stream('api/adventure_next?id='+advId+'&choice='+b.dataset.c,'the story continues, please wait',true);};});}
function hideChoices(){choicesEl.classList.remove('on');choicesEl.innerHTML='';}
function stream(url,msg,noBumper){stopStream();phase=null;buf=[];streamDone=false;idx=0;paused=false;
  pendingBranch=null;hideChoices();curTitle='';curShots=0;lastShot=-1;updProg();paintPause();
  audioByShot={};narrBufs={};stopNarr();
  startWriting(msg||'drawing the cast & sets, please wait');
  es=new EventSource(url);
  es.onmessage=e=>{const d=JSON.parse(e.data);
    if(d.done){streamDone=true;if(es){es.close();es=null;}
      if(phase==null&&!playTimer){playing(false);curt(true);} return;}
    if(d.meta){curTitle=d.meta.title||'';curShots=d.meta.shots||0;
      if(d.meta.id)saveMyFilm(d.meta.id,curTitle);
      marqueeEl.textContent=curTitle?('NOW SHOWING — “'+curTitle.toUpperCase()+'”'):'';
      return;}
    if(d.branch){pendingBranch=d.branch;return;}
    if(d.audio){audioByShot[d.shot]=d.audio;
      if(playTimer&&lastShot===d.shot&&!narrSrc)playNarr(d.shot);
      return;}
    if(d.phase){
      if(d.phase==='cast'){castLog.push({label:d.label||'',img:d.img||null});updWriting('casting the players');}
      else if(d.phase==='set'){castLog.push({label:'set: '+(d.label||''),img:null});updWriting('building the sets');}
      return;}
    if(!d.html)return;
    const film=d.pix||d.html.indexOf('<span')>=0;
    if(!film){stopStream();phase=null;show(d.html);playing(false);curt(true);return;}
    buf.push(d);
    if(phase==='writing'){phase='bumper';
      if(noBumper)curtainReveal();else playBumper(curtainReveal);}
    else if(!playTimer&&phase==null){startBufferedPlay();}};
  es.onerror=()=>{if(es){es.close();es=null;}streamDone=true;if(phase==null&&!playTimer)playing(false);};}
function replay(){if(!buf.length)return;stopStream();streamDone=true;pendingBranch=null;hideChoices();
  startBufferedPlay();}
function poster(){const last=buf[buf.length-1];if(!last||!last.pix)return;
  const m=last.html.match(/src='([^']+)'/);if(!m)return;
  const a=document.createElement('a');a.href=m[1];
  a.download=((curTitle||'pixel-cinema').toLowerCase().replace(/[^a-z0-9]+/g,'-')+'-poster.png');
  document.body.appendChild(a);a.click();a.remove();}
const concept=document.getElementById('concept'), genreSel=document.getElementById('genre');
function active(){return !!(es||playTimer||creditsTimer||phase);}
function gpart(g){return g?('&genre='+encodeURIComponent(g)):'';}
function startMake(c,g){initAudio();stream('api/make?concept='+encodeURIComponent(c)+gpart(g));}
function make(){const c=concept.value.trim().slice(0,60),g=genreSel.value;
  if(!c){concept.focus();return;}
  if(active()){queue.push({c:c,g:g});updQueue();concept.value='';return;}
  startMake(c,g);}
document.getElementById('go').onclick=make;
concept.addEventListener('keydown',e=>{if(e.key==='Enter')make();});
document.getElementById('adv').onclick=()=>{const c=concept.value.trim().slice(0,60),g=genreSel.value;
  if(!c){concept.focus();return;}
  initAudio();stream('api/adventure?concept='+encodeURIComponent(c)+gpart(g),
    'writing your adventure, please wait');};
document.getElementById('surprise').onclick=()=>{const c=SURPRISES[Math.floor(Math.random()*SURPRISES.length)];
  concept.value=c;make();};
document.getElementById('replay').onclick=replay;
document.getElementById('stop').onclick=stop;
document.getElementById('pause').onclick=togglePause;
document.getElementById('poster').onclick=poster;
document.getElementById('fs').onclick=()=>{const d=document, el=d.documentElement;
  if(d.fullscreenElement||d.webkitFullscreenElement){(d.exitFullscreen||d.webkitExitFullscreen||function(){}).call(d);return;}
  const req=el.requestFullscreen||el.webkitRequestFullscreen;
  if(req){const p=req.call(el); const lock=()=>{try{window.screen.orientation&&window.screen.orientation.lock&&
      window.screen.orientation.lock('landscape').catch(()=>{});}catch(e){}};
    if(p&&p.then)p.then(lock).catch(()=>{}); else lock();}};
document.addEventListener('keydown',e=>{if(e.target===concept)return;
  if(e.code==='Space'){e.preventDefault();togglePause();}
  else if(e.key==='r'||e.key==='R')replay();
  else if(e.key==='Escape')stop();});
function filmChip(title,posterUrl,onclick){const b=document.createElement('button');b.className='film';
  const im=document.createElement('img');im.loading='lazy';im.src=posterUrl;im.alt='';
  im.onerror=()=>{im.style.display='none';};
  const sp=document.createElement('span');sp.textContent=title;
  b.appendChild(im);b.appendChild(sp);b.onclick=onclick;return b;}
function myFilms(){try{return JSON.parse(localStorage.getItem('pc_films')||'[]');}catch(e){return [];}}
function saveMyFilm(id,title){const l=myFilms().filter(f=>f.id!==id);
  l.unshift({id:id,title:title||'untitled'});
  try{localStorage.setItem('pc_films',JSON.stringify(l.slice(0,12)));}catch(e){}
  renderMy();}
function renderMy(){const row=document.getElementById('myrow'),shelf=document.getElementById('myshelf');
  const l=myFilms();row.innerHTML='';shelf.style.display=l.length?'':'none';
  l.forEach(f=>row.appendChild(filmChip(f.title,'api/poster?made='+f.id,
    ()=>{initAudio();stream('api/play_made?id='+f.id);})));}
renderMy();
fetch('api/gallery').then(r=>r.json()).then(films=>{const g=document.getElementById('gallery');
  films.forEach(n=>g.appendChild(filmChip(n,'api/poster?name='+encodeURIComponent(n),
    ()=>{initAudio();stream('api/play?name='+encodeURIComponent(n));})));
  galReady=true;reveal();}).catch(()=>{galReady=true;reveal();});
welcome();
</script></body></html>"""


server = gr.Server(title="PIXEL CINEMA")


@server.get("/")
def index():
    return HTMLResponse(PAGE)


@server.get("/static/cinema.png")
def bg_image():
    return FileResponse(os.path.join(HERE, "static", "cinema.png"))


@server.get("/static/ansifont.json")
def ansifont():
    return FileResponse(os.path.join(HERE, "static", "ansifont.json"), media_type="application/json")


@server.get("/static/share.png")
def share_image():
    return FileResponse(os.path.join(HERE, "static", "share.png"), media_type="image/png")


@server.get("/api/gallery")
def gallery():
    return JSONResponse(list(_films(SHOW).keys()))


def _spec_stream(spec, made_id=None):
    def frames():
        for p in _prewarm_payloads(spec):
            yield p
        yield _meta_payload(spec, made_id)
        for p in _film_payloads_with_audio(spec):
            yield p
            time.sleep(movies.FRAME_MS * 0.6 / 1000)
    return StreamingResponse(_sse(frames()), media_type="text/event-stream")


@server.get("/api/play")
def play(name: str):
    path = _films(SHOW).get(name)
    if not path:
        return JSONResponse({"error": "not found"}, status_code=404)
    return _spec_stream(json.load(open(path)))


@server.get("/api/play_made")
def play_made(id: str):
    if not _MADE_RE.match(id or ""):
        return JSONResponse({"error": "bad id"}, status_code=400)
    path = os.path.join(MADE, id + ".json")
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return _spec_stream(json.load(open(path)), made_id=id)


@server.get("/api/poster")
def poster(name: str = "", made: str = ""):
    if made:
        if not _MADE_RE.match(made):
            return JSONResponse({"error": "bad id"}, status_code=400)
        path, key = os.path.join(MADE, made + ".json"), "made_" + made
    else:
        path = _films(SHOW).get(name)
        key = "show_" + re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")[:48]
    if not path or not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    out = os.path.join(POSTERS, key + ".png")
    if not os.path.exists(out):
        with _POSTER_LOCK:
            if not os.path.exists(out):
                spec = json.load(open(path))
                shots = spec.get("shots") or []
                if not shots:
                    return JSONResponse({"error": "empty"}, status_code=404)
                try:
                    fr = pixelscene._shot_tick_frames(
                        shots[0], 0, pixelscene._variant(spec), pixelscene._cast_variant(spec))
                    fr[len(fr) // 2].convert("RGB").save(out)
                except Exception:
                    return JSONResponse({"error": "render failed"}, status_code=500)
    return FileResponse(out, media_type="image/png")


@server.get("/api/make")
def make(concept: str, genre: str = ""):
    concept = _clean_concept(concept)               # sanitise + hard 60-char cap (authoritative)
    genre = _genre(genre)

    def frames():
        if not concept:
            yield _screen("Type a short film idea to begin.", "ready")
            return
        if _blocked(concept) or _INJECT.search(concept):
            yield _screen("Let's keep it friendly — try a kind, fun film idea.", "blocked")
            return
        result = {}

        def work():
            try:
                result["spec"] = movies.direct(concept, genre=genre)
            except Exception as e:
                result["error"] = f"{type(e).__name__}: {e}"
        th = threading.Thread(target=work, daemon=True)
        th.start()
        t0 = time.time()
        while th.is_alive():
            # heartbeats keep the SSE alive while the client plays the opening credits
            yield {"phase": "writing", "t": int(time.time() - t0)}
            th.join(timeout=0.7)
        if result.get("error") or not (result.get("spec") or {}).get("shots"):
            yield _screen("Could not reach the model. Please try again.", "error")
            return
        spec = result["spec"]
        mid = _save_made(spec)
        for p in _prewarm_payloads(spec):
            yield p
        yield _meta_payload(spec, mid)
        for p in _film_payloads_with_audio(spec):
            yield p
            time.sleep(movies.FRAME_MS * 0.6 / 1000)
    return StreamingResponse(_sse(frames()), media_type="text/event-stream")


def _adv_stream(sid: str, concept: str, history: str, choice: str, first: bool, genre: str = ""):
    def frames():
        result = {}

        def work():
            try:
                result["chunk"] = movies.direct_branch(concept, history, choice, genre=genre)
            except Exception as e:
                result["error"] = f"{type(e).__name__}: {e}"
        th = threading.Thread(target=work, daemon=True)
        th.start()
        t0 = time.time()
        while th.is_alive():
            yield {"phase": "writing", "t": int(time.time() - t0)}
            th.join(timeout=0.7)
        chunk = result.get("chunk")
        if result.get("error") or not chunk or not chunk.get("shots"):
            yield _screen("Could not reach the model. Please try again.", "error")
            return
        sess = ADV.get(sid)
        if sess is None:
            yield _screen("This adventure has expired — start a new one.", "error")
            return
        if first:
            sess["title"] = str(chunk.get("title") or concept[:30])
        sess["history"] += "\n".join(s.get("narration", "") for s in chunk["shots"]) + "\n"
        spec = {"title": sess["title"], "logline": concept, "shots": chunk["shots"]}
        for p in _prewarm_payloads(spec):
            yield p
        yield _meta_payload(spec)
        ending = bool(chunk.get("ending"))
        for p in _film_payloads_with_audio(spec, end=ending):
            yield p
            time.sleep(movies.FRAME_MS * 0.6 / 1000)
        if not ending:
            yield {"branch": {"id": sid, "choices": (chunk.get("choices") or [])[:3]}}
    return StreamingResponse(_sse(frames()), media_type="text/event-stream")


@server.get("/api/adventure")
def adventure(concept: str, genre: str = ""):
    concept = _clean_concept(concept)
    genre = _genre(genre)
    if not concept or _blocked(concept) or _INJECT.search(concept):
        msg = ("Type a short film idea to begin.", "ready") if not concept else \
              ("Let's keep it friendly — try a kind, fun film idea.", "blocked")

        def guard():
            yield _screen(*msg)
        return StreamingResponse(_sse(guard()), media_type="text/event-stream")
    if len(ADV) > 200:
        ADV.clear()
    sid = uuid.uuid4().hex[:12]
    ADV[sid] = {"concept": concept, "history": "", "title": concept[:30], "genre": genre}
    return _adv_stream(sid, concept, "", "", True, genre)


@server.get("/api/adventure_next")
def adventure_next(id: str, choice: str = ""):
    sess = ADV.get(id)
    if not sess or not _MADE_RE.match(id or ""):
        return JSONResponse({"error": "unknown adventure"}, status_code=404)
    choice = _clean_concept(choice)[:40]
    if _blocked(choice) or _INJECT.search(choice):
        choice = "press on"
    sess["history"] += f"\nTHE VIEWER CHOSE: {choice}\n"
    return _adv_stream(id, sess["concept"], sess["history"], choice, False, sess.get("genre", ""))


if __name__ == "__main__":
    # HF Spaces runs this file as a script and injects GRADIO_SERVER_NAME/PORT.
    server.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
                  server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
