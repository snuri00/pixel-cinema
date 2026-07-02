"""
tts.py — LOCAL narration + character voices via Kokoro-82M (the same realtime
pipeline as the small-talk project). Each shot is voiced as: a MALE NARRATOR reads
the narration, then the speaking character delivers its dialogue in its OWN voice —
picked deterministically from a pool, so a character keeps the same voice across
shots, chunks and replays. Segments are cached per (voice, line) as 24 kHz mono WAVs
in tts_cache/ and shipped to the browser as one data-URI per shot.

Defaults to CPU so the little 4 GB GPU stays free for the SD asset backend —
Kokoro-82M is near-realtime on CPU. Override per deploy:

    CLAUDEMOVIES_TTS=0                # disable narration entirely
    CLAUDEMOVIES_TTS_DEVICE=cpu      # cpu (default) | cuda | auto (cuda if >=2 GB free)
    CLAUDEMOVIES_TTS_VOICE=bm_george # narrator voice id (male storyteller by default)

Leaf module: imports kokoro/torch lazily; degrades to silence if they're missing.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import os
import threading

ENABLED = os.environ.get("CLAUDEMOVIES_TTS", "1") != "0"
DEVICE = os.environ.get("CLAUDEMOVIES_TTS_DEVICE", "cpu").strip().lower()
NARRATOR = os.environ.get("CLAUDEMOVIES_TTS_VOICE", "bm_george").strip()

CAST_POOL = ("af_heart", "af_bella", "af_nicole", "af_sky", "am_adam", "am_michael",
             "am_fenrir", "am_puck", "am_onyx", "bf_emma", "bf_isabella",
             "bm_lewis", "bm_daniel", "bm_fable")

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_cache")
_LOCK = threading.RLock()
_PIPES: dict = {}
_FAILED = False


def available() -> bool:
    if not ENABLED or _FAILED:
        return False
    return (importlib.util.find_spec("kokoro") is not None
            and importlib.util.find_spec("soundfile") is not None)


def voice_for(label: str) -> str:
    """A stable character voice: the same subject always speaks with the same voice
    (hash-picked from the pool, never the narrator's own voice)."""
    pool = [v for v in CAST_POOL if v != NARRATOR] or list(CAST_POOL)
    h = int(hashlib.sha256(("voice:" + (label or "").lower().strip()).encode()).hexdigest()[:8], 16)
    return pool[h % len(pool)]


def _device():
    import torch
    device = DEVICE
    if device == "auto":
        free = 0
        if torch.cuda.is_available():
            try:
                free = torch.cuda.mem_get_info()[0] >> 20
            except Exception:
                free = 0
        device = "cuda" if free >= 2048 else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    return device


def _pipe(lang: str):
    """One KPipeline per language code ('a' American, 'b' British), sharing a single
    loaded KModel so a mixed-accent cast doesn't cost 2x the memory."""
    global _FAILED
    pipe = _PIPES.get(lang)
    if pipe is not None:
        return pipe
    with _LOCK:
        pipe = _PIPES.get(lang)
        if pipe is not None:
            return pipe
        try:
            from kokoro import KPipeline
            shared = next((p.model for p in _PIPES.values() if getattr(p, "model", None)), None)
            if shared is not None:
                pipe = KPipeline(lang_code=lang, model=shared)
            else:
                pipe = KPipeline(lang_code=lang, device=_device())
            _PIPES[lang] = pipe
        except Exception:
            _FAILED = True
            raise
    return pipe


def _synth(text: str, voice: str):
    import numpy as np
    pipe = _pipe(voice[0])
    chunks = []
    with _LOCK:
        for item in pipe(text, voice=voice):
            audio = getattr(item, "audio", None)
            if audio is None:
                audio = item[2]
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            chunks.append(np.asarray(audio, dtype=np.float32))
    if not chunks:
        return None
    return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]


def _segment_wav(text: str, voice: str):
    """The synthesised float32 array for one (voice, line), cached on disk."""
    import soundfile as sf
    key = hashlib.sha256((voice + "|" + text).encode("utf-8")).hexdigest()[:20]
    path = os.path.join(_DIR, key + ".wav")
    if not os.path.exists(path):
        wav = _synth(text, voice)
        if wav is None:
            return None
        os.makedirs(_DIR, exist_ok=True)
        sf.write(path, wav, 24000)
        return wav
    data, _ = sf.read(path, dtype="float32")
    return data


def say_segments(segments) -> str | None:
    """[(text, voice_or_empty), ...] -> one 'data:audio/wav;base64,...' clip with a
    short beat between segments. Empty voice means the narrator. None on failure."""
    if not available():
        return None
    import numpy as np
    parts = []
    for text, voice in segments:
        text = " ".join((text or "").split())[:300]
        if not text:
            continue
        v = (voice or NARRATOR).strip()
        try:
            wav = _segment_wav(text, v)
        except Exception:
            wav = None
        if wav is not None and len(wav):
            if parts:
                parts.append(np.zeros(int(24000 * 0.28), dtype=np.float32))
            parts.append(wav)
    if not parts:
        return None
    try:
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, np.concatenate(parts) if len(parts) > 1 else parts[0],
                 24000, format="WAV")
        return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def say_datauri(text: str, voice: str = "") -> str | None:
    return say_segments([(text, voice)])


def main():
    import argparse
    import time
    ap = argparse.ArgumentParser(description="Voice a line with the local Kokoro narrator.")
    ap.add_argument("text", nargs="?", default="In the shadows, a tiny knight sits alone, his heart heavy.")
    ap.add_argument("--voice", default="")
    ap.add_argument("--label", default="", help="voice a line AS this character instead")
    args = ap.parse_args()
    v = args.voice or (voice_for(args.label) if args.label else "")
    t0 = time.time()
    uri = say_datauri(args.text, v)
    if not uri:
        print("tts unavailable (is `pip install kokoro soundfile` done, CLAUDEMOVIES_TTS=1?)")
        return
    print(f"ok in {time.time() - t0:.2f}s — voice={v or NARRATOR}, {len(uri)} chars of data-uri")


if __name__ == "__main__":
    main()
