# 🎬 PIXEL CINEMA

A small language model **writes, directs, and stages original short films** as animated
**pixel-art scenes**, streamed live in the browser — frame by frame, on a little cinema screen.

From a single concept the model invents a beat-by-beat story, casts the characters, picks
the setting, camera and mood for each shot. The scene is then assembled locally: each
character is drawn as a pixel sprite, placed on a generated pixel background, scaled to its
real-world size, grounded with a soft shadow, and played with subtitles — ending on a
proper **THE END** card.

## How it works

- **Director (LLM).** A concept → a strict-JSON film script: shots with `narration`, `cast`,
  `setting`, `action`, `camera`, `mood`. Uses any OpenAI- or Anthropic-compatible endpoint
  (e.g. DeepSeek) via `llm_client.py`.
- **Assets (local, cached).** Character sprites and backgrounds are generated on your machine,
  best → simplest with automatic fallthrough:
  - **`sd`** — a pixel-art Stable-Diffusion 1.5 model (+ LCM for fast sampling); the raw render
    is pixelized (downscale → palette quantize → background cutout) into a clean sprite.
    Backgrounds use the model's scene mode.
  - **`llm`** — the director draws the sprite directly as a palette grid.
  - **`procedural`** — an algorithmic sprite/WaveFunctionCollapse-ground generator (no GPU, offline).
- **Stage (web UI).** A custom `gr.Server` frontend streams frames over SSE onto a cinema-hall
  screen with reactive lighting. Subjects are proportionally sized (a mouse < a person < an
  elephant), fit to the frame, and shaded onto the ground. Every asset is generated once and
  cached, so replays are instant.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` (never commit it) with your director endpoint:

```bash
CLAUDEMOVIES_LLM_URL=https://api.deepseek.com/anthropic
CLAUDEMOVIES_LLM_KEY=sk-your-key
CLAUDEMOVIES_LLM_MODEL=deepseek-v4-pro
CLAUDEMOVIES_RENDER=pixel          # or "ascii" for the grid renderer
CLAUDEMOVIES_ASSET_BACKEND=sd      # sd | llm | procedural
CLAUDEMOVIES_BG_BACKEND=sd         # sd | procedural
```

Run it:

```bash
python server_app.py               # open http://localhost:7860
```

**Local SD backend (optional, recommended).** For the `sd` sprite/background backend you also
need a PyTorch + diffusers stack:

```bash
pip install torch diffusers transformers accelerate peft safetensors
```

A CUDA GPU is recommended. On a small (≤4 GB) card set `CLAUDEMOVIES_SD_OFFLOAD=1` (default) to
stream weights from CPU; on a larger GPU set it to `0` for full-speed generation. The pixel
model and LCM LoRA download automatically on first use. Without a GPU/SD stack the app falls
back to the `llm` / `procedural` backends and the pre-written gallery still plays.

## Configuration

| Variable | Purpose |
|---|---|
| `CLAUDEMOVIES_LLM_URL/_KEY/_MODEL` | Director endpoint (OpenAI- or Anthropic-compatible; auto-detected) |
| `CLAUDEMOVIES_RENDER` | `pixel` (default) or `ascii` |
| `CLAUDEMOVIES_ASSET_BACKEND` | `sd` \| `llm` \| `procedural` |
| `CLAUDEMOVIES_BG_BACKEND` | `sd` \| `procedural` |
| `CLAUDEMOVIES_SD_MODEL` | Pixel-art SD 1.5 model id |
| `CLAUDEMOVIES_SD_OFFLOAD` | `1` CPU-offload (low VRAM) · `0` full GPU (fast) |
| `CLAUDEMOVIES_SD_GEN` | Generation resolution (lower = less VRAM) |
| `CLAUDEMOVIES_SD_LCM` | `1` fast LCM sampling · `0` standard |
