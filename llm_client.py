"""
Chat client for the director model. Speaks TWO protocols behind one `llm()` call:

  * OpenAI-compatible  /v1/chat/completions   (Modal vLLM, local llama.cpp, etc.)
  * Anthropic Messages /v1/messages           (Claude, or an Anthropic-compatible
                                                DeepSeek endpoint, via the `anthropic` SDK)

A leaf module: it imports nothing from the package, so both `movies` (story
director) and `draw` (ASCII artist) can use it without an import cycle.

Config via env:  CLAUDEMOVIES_LLM_URL / _KEY / _MODEL  (the main director: Cinema).
The Adventure mode may use a second endpoint via CLAUDEMOVIES_ADV_* (a general model
handles branching better); it falls back to the main endpoint if ADV is unset.

Protocol is auto-detected:
  * explicit:  CLAUDEMOVIES_LLM_API=anthropic|openai  (or CLAUDEMOVIES_ADV_API)
  * otherwise: a URL containing "anthropic" -> Anthropic, else OpenAI.

DeepSeek (Anthropic-compatible) example:
    export CLAUDEMOVIES_LLM_URL=https://api.deepseek.com/anthropic
    export CLAUDEMOVIES_LLM_KEY=sk-...            # your DeepSeek key
    export CLAUDEMOVIES_LLM_MODEL=deepseek-chat   # or deepseek-reasoner
"""

import json
import os
import urllib.request


def _load_dotenv():
    """Load KEY=VALUE pairs from a `.env` next to this file into os.environ, WITHOUT
    overriding anything already set in the real environment. Stdlib-only (no
    python-dotenv dependency); ignores blank lines, `#` comments and surrounding quotes."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:                # real env wins over the file
                    os.environ[k] = v
    except FileNotFoundError:
        pass


_load_dotenv()


def _api_kind(prefix: str, url: str) -> str:
    """'anthropic' or 'openai' — explicit env override wins, else sniff the URL."""
    explicit = os.environ.get(prefix + "_API", "").strip().lower()
    if explicit in ("anthropic", "openai"):
        return explicit
    return "anthropic" if "anthropic" in url.lower() else "openai"


def _anthropic(prefix, url, key, model, system, user, max_tokens, temperature, timeout):
    """Call the Anthropic Messages API via the official SDK. The base_url must NOT
    include the '/v1/messages' suffix — the SDK appends it (DeepSeek's base is
    '.../anthropic'). Temperature is clamped to Anthropic's [0, 1] range.

    Thinking (reasoning) is DISABLED by default: a reasoning model like deepseek-v4-pro
    spends the max_tokens budget on hidden thinking, which on the app's short-budget
    calls (draw, judge, pitch) leaves no text at all and silently drops us to the generic
    fallback film. Override with CLAUDEMOVIES_LLM_THINKING=enabled (and optionally
    CLAUDEMOVIES_LLM_THINKING_BUDGET)."""
    import anthropic                                       # imported lazily: only when used

    base = url.rstrip("/")
    if base.endswith("/v1"):                               # tolerate an OpenAI-style trailing /v1
        base = base[:-3]

    mode = os.environ.get(prefix + "_THINKING", "disabled").strip().lower()
    if mode in ("on", "enabled", "true", "1"):
        budget = int(os.environ.get(prefix + "_THINKING_BUDGET", "1024") or 1024)
        thinking = {"type": "enabled", "budget_tokens": budget}
    else:
        thinking = {"type": "disabled"}

    client = anthropic.Anthropic(api_key=key, base_url=base, timeout=float(timeout))
    msg = client.messages.create(
        model=model or "deepseek-chat",
        max_tokens=max_tokens,
        temperature=max(0.0, min(1.0, temperature)),       # Anthropic caps temperature at 1.0
        system=system,                                     # system is a top-level param, not a message
        messages=[{"role": "user", "content": user}],
        extra_body={"thinking": thinking},                 # DeepSeek reads this Anthropic-format toggle
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def _openai(url, key, model, system, user, max_tokens, temperature, timeout):
    """Call an OpenAI-compatible /chat/completions endpoint with stdlib urllib."""
    base = url.rstrip("/") + ("" if url.rstrip("/").endswith("/v1") else "/v1")
    payload = json.dumps({"model": model or "Qwen/Qwen2.5-3B-Instruct",
                          "messages": [{"role": "system", "content": system},
                                       {"role": "user", "content": user}],
                          "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(base + "/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())["choices"][0]["message"]["content"]


def llm(system, user, max_tokens=600, temperature=0.8, prefix="CLAUDEMOVIES_LLM", timeout=90):
    """Return the model's reply, or "" if no endpoint is configured. `prefix` selects
    the env-var family (CLAUDEMOVIES_LLM_* by default, CLAUDEMOVIES_ADV_* for adventure).
    `timeout` is the per-request read timeout — raise it for a cold scale-to-zero endpoint."""
    if prefix == "CLAUDEMOVIES_ADV" and not os.environ.get("CLAUDEMOVIES_ADV_URL"):
        prefix = "CLAUDEMOVIES_LLM"                       # no separate adventure endpoint -> use the main one
    url = os.environ.get(prefix + "_URL", "")
    if not url:
        return ""
    key = os.environ.get(prefix + "_KEY", "")
    model = os.environ.get(prefix + "_MODEL", "")
    if _api_kind(prefix, url) == "anthropic":
        return _anthropic(prefix, url, key, model, system, user, max_tokens, temperature, timeout)
    return _openai(url, key, model, system, user, max_tokens, temperature, timeout)
