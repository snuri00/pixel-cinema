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
- **A real cinema, not just a stream.** Curtains part when the film starts, a NOW SHOWING
  marquee lights up over the screen, and the opening credits cast the players live (each
  sprite appears as it's drawn). The player has a scrubbable progress bar with a shot counter,
  pause/resume (Space), replay (R), stop (Esc), a queue for your next idea, and a downloadable
  end-card poster starring the cast. Films you make are kept on a "your films" shelf (with
  thumbnails, like the gallery) so you can replay them any time. Optional projector-hum +
  chiptune cues via Web Audio, with a mute toggle.
- **Adventure mode.** "Adventure" turns your idea into an interactive film told in chunks:
  at each cliffhanger the audience picks what happens next, until the story earns its ending.
- **Narration (local TTS).** A male narrator (**Kokoro-82M**, `bm_george` by default — the same
  realtime pipeline as small-talk) reads each shot's narration, then the speaking character
  delivers its dialogue in its **own voice**, hash-picked per subject from a pool so a character
  sounds the same across shots, chunks and replays. Synthesised in a background thread while
  the film streams, cached per line in `tts_cache/`. Runs on CPU by default so the GPU stays
  free for the SD backend; needs `pip install kokoro soundfile` (falls back to silence).
- **Genre picker.** A dropdown next to the concept box (comedy, action, spooky, noir, ...)
  steers the director: the chosen genre is written into the screenplay brief, for both normal
  films and adventures.
- **Theatrical staging.** Each shot is choreographed, not looped: `enter` walks in from
  off-screen, `exit` leaves, `chase` pursues, `rise` floats up — with facing flips and a real
  stride cycle. The camera pans / pushes (Ken Burns) / shakes over an oversized world canvas.
  The environment lives: parallax silhouette skylines per setting, drifting clouds, stars and
  a crescent moon at night, rain / snow / embers / fireflies / dust, torch flicker indoors,
  mood emotes and speech bubbles over the cast.

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
| `CLAUDEMOVIES_SPRITE_VARIANTS` | Distinct SD/LLM sprite looks cached per subject (default `3`) — each film picks one, so "cat" isn't the same cat in every film |
| `CLAUDEMOVIES_TTS` | `1` narration voice on (default) · `0` off |
| `CLAUDEMOVIES_TTS_DEVICE` | `cpu` (default, keeps VRAM for SD) \| `cuda` \| `auto` (cuda if ≥2 GB free) |
| `CLAUDEMOVIES_TTS_VOICE` | Narrator voice id (default `bm_george`, a male storyteller; characters pick their own voices from a pool) |
| `CLAUDEMOVIES_SD_MODEL` | Pixel-art SD 1.5 model id |
| `CLAUDEMOVIES_SD_OFFLOAD` | `1` CPU-offload (low VRAM) · `0` full GPU (fast) |
| `CLAUDEMOVIES_SD_GEN` | Generation resolution (lower = less VRAM) |
| `CLAUDEMOVIES_SD_LCM` | `1` fast LCM sampling · `0` standard |
