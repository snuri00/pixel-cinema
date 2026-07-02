"""
tts.py — LOCAL narration voice via Kokoro-82M (the same realtime pipeline as the
small-talk project). Each shot's narration is synthesised once to a 24 kHz mono WAV,
cached on disk, and shipped to the browser as a data-URI so the film speaks.

Defaults to CPU so the little 4 GB GPU stays free for the SD asset backend —
Kokoro-82M is near-realtime on CPU. Override per deploy:

    CLAUDEMOVIES_TTS=0                # disable narration entirely
    CLAUDEMOVIES_TTS_DEVICE=cpu      # cpu (default) | cuda | auto (cuda if >=2 GB free)
    CLAUDEMOVIES_TTS_VOICE=af_heart  # any Kokoro voice id

Leaf module: imports kokoro/torch lazily; degrades to silence if they're missing.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import os
import threading

ENABLED = os.environ.get("CLAUDEMOVIES_TTS", "1") != "0"
DEVICE = os.environ.get("CLAUDEMOVIES_TTS_DEVICE", "cpu").strip().lower()
VOICE = os.environ.get("CLAUDEMOVIES_TTS_VOICE", "af_heart").strip()

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_cache")
_LOCK = threading.RLock()
_PIPE = None
_FAILED = False


def available() -> bool:
    if not ENABLED or _FAILED:
        return False
    return (importlib.util.find_spec("kokoro") is not None
            and importlib.util.find_spec("soundfile") is not None)


def _pipe():
    global _PIPE, _FAILED
    if _PIPE is not None:
        return _PIPE
    with _LOCK:
        if _PIPE is not None:
            return _PIPE
        try:
            import torch
            from kokoro import KPipeline
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
            _PIPE = KPipeline(lang_code=VOICE[0], device=device)
        except Exception:
            _FAILED = True
            raise
    return _PIPE


def _synth(text: str, voice: str):
    import numpy as np
    pipe = _pipe()
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


def say_datauri(text: str, voice: str = "") -> str | None:
    """One narration line -> 'data:audio/wav;base64,...' (24 kHz mono, disk-cached),
    or None when TTS is off / unavailable / the line is empty."""
    if not available():
        return None
    text = " ".join((text or "").split())[:300]
    if not text:
        return None
    v = voice or VOICE
    key = hashlib.sha256((v + "|" + text).encode("utf-8")).hexdigest()[:20]
    path = os.path.join(_DIR, key + ".wav")
    if not os.path.exists(path):
        try:
            import soundfile as sf
            wav = _synth(text, v)
            if wav is None:
                return None
            os.makedirs(_DIR, exist_ok=True)
            sf.write(path, wav, 24000)
        except Exception:
            return None
    try:
        data = open(path, "rb").read()
    except Exception:
        return None
    return "data:audio/wav;base64," + base64.b64encode(data).decode()


def main():
    import argparse
    import time
    ap = argparse.ArgumentParser(description="Voice one line with the local Kokoro narrator.")
    ap.add_argument("text", nargs="?", default="In the shadows, a tiny knight sits alone, his heart heavy.")
    ap.add_argument("--voice", default="")
    args = ap.parse_args()
    t0 = time.time()
    uri = say_datauri(args.text, args.voice)
    if not uri:
        print("tts unavailable (is `pip install kokoro soundfile` done, CLAUDEMOVIES_TTS=1?)")
        return
    print(f"ok in {time.time() - t0:.2f}s — {len(uri)} chars of data-uri (cached in tts_cache/)")


if __name__ == "__main__":
    main()
