"""
Draw-on-demand: give any character/object, get a small ASCII sprite.

  • known thing -> the hand-authored library (consistent).
  • new thing  -> the model draws it (matching the house style), then it's
                  NORMALIZED and CACHED so it looks the same every shot.

So any story the model invents can be staged: it makes it up, draws the cast,
and animates — even characters that were never in the library.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
from typing import TYPE_CHECKING

import ascii_sprites as A
import characters
from llm_client import llm

if TYPE_CHECKING:
    from schema import Sprite

CACHE = os.path.join(os.path.dirname(__file__), "drawn_cache.json")
MAX_W, MAX_H = 14, 7
PLACEHOLDER = ["  (?) ", "  /|\\ ", "  / \\ "]


def _normalize(lines: list[str]) -> Sprite:
    lines = [ln.replace("\t", " ") for ln in lines if ln.strip()][:MAX_H]
    lines = [ln[:MAX_W] for ln in lines] or ["  (?) ", "  /|\\ ", "  / \\ "]
    w = max(len(ln) for ln in lines)
    return [ln.ljust(w) for ln in lines]


def _parse(raw: str | None) -> Sprite:
    m = re.search(r"```[a-z]*\n?(.*?)```", raw or "", re.S)
    body = (m.group(1) if m else (raw or "")).strip("\n")
    return _normalize(body.split("\n"))


_CODEY = re.compile(r"\b(for|while|print|range|def|import|return|class|lambda|elif|None|True|False|self)\b", re.I)


def _is_art(spr: Sprite) -> bool:
    """Reject model output that's prose/code, not a sprite (a 4B sometimes emits Python or a
    sentence). Real sprites are mostly symbols; text is mostly letters."""
    joined = "\n".join(spr)
    nonspace = [c for c in joined if not c.isspace()]
    if not nonspace:
        return False
    if sum(c.isalpha() for c in nonspace) / len(nonspace) > 0.5:
        return False
    return not _CODEY.search(joined)


def _load() -> dict[str, Sprite]:
    if not os.path.exists(CACHE):
        return {}
    try:
        return json.load(open(CACHE))
    except (OSError, json.JSONDecodeError) as e:        # corrupt cache: warn, don't silently wipe
        print(f"draw: ignoring unreadable {CACHE}: {e}", file=sys.stderr)
        return {}


def draw_sprite(label: str, desc: str = "") -> Sprite:
    """Return a sprite for any label: saved character, built-in library, cache, or
    freshly drawn (a saved custom character takes precedence over the library)."""
    saved = characters.get(label)
    if saved:
        return saved
    base = A.get(label)
    if base:
        return base
    key = label.strip().lower()
    cache = _load()
    if key in cache:
        return cache[key]
    raw = ""
    try:                                          # ask the model to draw a new one, in style
        raw = llm(
            "You are an ASCII artist. Draw a small black-and-white ASCII sprite (max 7 rows, max 14 "
            "columns, pure ASCII, front-facing) of the subject, in THIS house style:\n\n"
            + A.style_reference(4) + "\n\nOutput ONLY the ASCII art.",
            f"Draw: {label}. {desc}".strip(), max_tokens=160, temperature=0.7)
    except (urllib.error.URLError, TimeoutError, OSError, KeyError, json.JSONDecodeError) as e:
        print(f"draw_sprite: model call failed for {label!r}: {e}", file=sys.stderr)
    if not raw:                                   # no endpoint or failed call -> placeholder, NOT cached
        return PLACEHOLDER
    sprite = _parse(raw)
    if not _is_art(sprite):                        # model emitted prose/code, not art -> neutral, NOT cached
        return PLACEHOLDER
    cache[key] = sprite                           # cache only real, model-drawn art
    json.dump(cache, open(CACHE, "w"))
    return sprite
