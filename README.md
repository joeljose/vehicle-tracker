# Vehicle Tracker — Traffic Junction Monitor

GPU-accelerated vehicle tracking for traffic junction monitoring. Detects vehicles, determines directional transit (entry arm to exit arm), and generates alerts with best-quality vehicle photos — all from a single consumer GPU.

![Review phase demo](docs/demo_videos/review_demo.gif)

*Review phase — alerts populate the feed on the right; clicking one replays the clip with bounding-box overlays and shows the best-photo snapshot.*

## Features

- **Directional transit detection** — entry/exit directions (e.g., "west to north")
- **Stagnant vehicle alerts** — vehicles stopped for configurable duration (default 2.5 min)
- **Best-photo capture** — highest quality frame of each tracked vehicle
- **Multi-channel** — multiple sources share a single GPU pipeline (saves ~280 MB VRAM per added channel vs separate pipelines). N-channel support is architectural (no hard limit in the code); see *Hardware sizing* below for what we've actually validated.
- **YouTube Live support** — monitor live traffic camera streams directly
- **4-phase workflow** — Setup (draw ROI/lines) -> Analytics (run detection) -> Review (replay alerts)
- **Browser UI** — live MJPEG video, ROI/line drawing tools, alert feed, video replay

## Prerequisites

- **NVIDIA GPU** with driver >= 580.x (tested on RTX 4050 Max-Q, 6 GB VRAM)
- **Docker Engine** >= 24.0 with Docker Compose v2
- **NVIDIA Container Toolkit** ([install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
- **make** (`sudo apt install make`)

## Hardware sizing

The code architecturally supports N concurrent channels (see PRD F-03 / F-08j) — there is no hard limit on `add_channel`. What you can actually run depends on the GPU you're on, the source resolution / FPS, and detection density:

- **What we've validated:** 1 channel of 1080p @ 60 FPS Analytics on RTX 4050 (6 GB VRAM) runs at real-time (≈1.02× source duration, both backends). 30 FPS sources have substantial headroom.
- **What we've measured under stress:** 2 concurrent 1080p Analytics streams on the same 4050 saturate the GPU — the heavier (60 FPS, dense) channel slows to a fraction of source rate. Two light channels (30 FPS, sparse scenes) co-exist comfortably.
- **What we haven't tested:** ≥3 concurrent channels on this hardware, or any channel count on bigger GPUs (A4000/A100/4090). The "8 channels" mentioned in design notes is the *target ceiling* on a server-class GPU, not a 4050 number.

Practical guidance: budget ~17 ms per processed frame per Analytics channel (1080p, ~20 vehicles, FP16) on a 4050. If you have N channels at S source FPS each, you need N × S × 17 ms ≤ 1000 ms of GPU/frame time. Two 60 FPS channels on a 4050 = ~2.0 s of GPU work per second of wall-clock. The custom backend caps processing at 30 FPS (decoding still happens at native rate, but inference/render is rate-limited), so plug N × 30 × 17 ms ≈ 510 ms × N — meaning ~2 channels is the comfortable limit on a 4050 even after the cap.

## Quick Start

```bash
git clone git@github.com:joeljose/vehicle-tracker.git
cd vehicle-tracker

# 1. First-time setup (creates .env, builds Docker images)
make setup

# 2. Start containers (backend idle, frontend hot-reloading)
make dev

# 3. Launch the backend server
make start

# 4. Open the UI
#    http://localhost:5173
```

**That's it.** The UI will show a "Start Pipeline" button. Click it, then add a video source.

### Test sources

| Source | Path / URL |
|--------|------------|
| File (1 min, Lytle South) | `/data/test_clips/lytle_south_1min.mp4` |
| File (10s, 741 & 73) | `/data/test_clips/741_73_10s.mp4` |
| YouTube Live | Paste any YouTube Live traffic cam URL |

> File paths are **inside the container** — the `data_collection/` directory is mounted at `/data/`.

### Workflow (per channel)

1. **Setup** — Video loops. Draw an ROI polygon, then draw entry/exit lines. Load a saved config with the "Load Site" dropdown if available.
2. **Start Analytics** — Click "Start Analytics". The pipeline runs detection/tracking on the video (plays once for files, continuous for live streams).
3. **Review** — After the video ends (or you click "Stop Analytics"), alerts appear. Click an alert to replay the clip with bounding box overlay.

## Day-to-Day Development

All commands go through the **Makefile**. Never call `docker compose` directly (causes root-owned files).

### Lifecycle

```bash
make dev              # Start containers (run once per session)
make start            # Launch backend server
make restart          # Restart backend (after Python code changes)
make stop-server      # Stop backend (container stays up)
make down             # Stop everything
```

> **After editing Python files**, run `make restart` to pick up changes. The frontend hot-reloads automatically — no restart needed for JS/JSX changes.

### Testing & Quality

```bash
make test             # Run all tests (367 on DS, 296 on custom; no GPU needed)
make test-v           # Verbose (show test names)
make lint             # Ruff check
make format           # Ruff auto-fix
```

### Debugging

```bash
make logs             # Tail backend server logs
make status           # Container health + GPU memory
make shell            # Bash into backend container
make shell-frontend   # Shell into frontend container
```

### Data

```bash
# Extract a test clip from a full video
make clip SRC="/data/site_videos/full_video.mp4" START=30 DURATION=60 OUT="/data/test_clips/clip.mp4"
```

### Release

```bash
make version          # Show current version (from git tags)
make tag              # Show recent tags + instructions
# To release: git tag vX.Y.Z && git push origin vX.Y.Z
```

## Architecture

```
Browser (React) <-> FastAPI (REST + WebSocket + MJPEG) <-> Pipeline backend (GPU)
                                                            |
                                              +-------------+-------------+
                                              |                           |
                                       DeepStream backend           Custom backend
                                       (default, M5+)               (M7+, hackable)
```

The FastAPI layer talks to the pipeline through a `PipelineBackend` Protocol so the API stays pipeline-agnostic. Pick the backend at startup with `VT_BACKEND=deepstream` (default) or `VT_BACKEND=custom`. Both run the same fine-tuned single-class YOLOv8s student (M8-P1.5 v2).

**DeepStream backend topology** (shared across channels):
```
src_0 (nvurisrcbin) --+
src_1 (nvurisrcbin) --+--> nvstreammux -> nvinfer -> nvtracker -> nvstreamdemux
                      |                                              |-> ch0: OSD -> MJPEG
                      |                                              |-> ch1: OSD -> MJPEG
                      |   BatchMetadataRouter routes per-source metadata; phase
                      |   transitions on one channel use _soft_remove_source so
                      |   they don't disturb other channels.
```

**Custom backend topology** (per channel):
```
NVDEC (PyNvVideoCodec) --> CuPy preprocess --> direct TensorRT --> CPU NMS
                                                                       |
                                                  BoT-SORT (HSV ReID) <+
                                                          |
                                                          +--> analytics + render JPEG
                                                                       |
                                                          display_queue (PTS-paced)
                                                                       |
                                                                     MJPEG
```
Processing is capped at 30 FPS; the source is decoded at native rate but every other frame is dropped on 60 FPS sources, matching the tracker's frame_rate=30 assumption and saving ~50% GPU/CPU. Replay clips in Review are produced by ffmpeg directly from the original source file at full source FPS.

## Project Structure

```
vehicle-tracker/
  backend/
    api/                 # FastAPI routes, WebSocket (per-client queues), MJPEG
    config/              # Site config model + saved junction configs
    pipeline/
      deepstream/        # DeepStream adapter (shared GPU pipeline)
      custom/            # Custom adapter: NVDEC + CuPy + TRT + BoT-SORT
      protocol.py        # PipelineBackend Protocol interface
      direction.py       # Line-crossing + direction state machine
      roi.py             # ROI polygon filtering
      alerts.py          # AlertStore (transit + stagnant)
      snapshot.py        # Best-photo capture
      clip_extractor.py  # ffmpeg clip extraction for replay (race-safe)
      fake.py            # Test backend (no GPU needed)
    tests/               # 367 tests on DS, 296 on custom (backend-gated)
    main.py              # FastAPI app factory
    Dockerfile.deepstream
    Dockerfile.custom
  frontend/
    src/
      pages/             # Home, Channel
      components/        # AlertFeed, ReplayView, DrawingCanvas, StatsBar
      contexts/          # React Context state management
      api/               # REST + WebSocket clients
  docs/                  # PRD, design doc, changelog, spike results
  scripts/               # Developer utility scripts (draw_config.py)
  plans/                 # Milestone implementation plans
  experiments/           # Pipeline validation experiments + results
  config/deepstream/     # DeepStream model configs
  data_collection/       # Test clips + site videos (gitignored)
  models/                # TensorRT engines (gitignored)
  docker-compose.dev.yml
  Makefile               # All dev commands (run `make help`)
```

## Milestones

| Milestone | Tag | Tests |
|-----------|-----|-------|
| M1 — DeepStream pipeline | v0.1.0 | 177 |
| M2 — API layer (FastAPI REST/WS/MJPEG) | v0.2.0 | 237 |
| M3 — React UI (4-phase workflow) | v0.3.0 | 286 |
| M4 — DeepStream-FastAPI integration | v0.4.0 | 312 |
| M5 — Multi-channel shared pipeline | v0.5.0 | 336 |
| M6 — YouTube Live streams | v0.6.0 | 371 |
| M7 — Custom GPU-resident pipeline | v0.7.0 | 291 (custom) + 363 (DS) |
| M8 — Custom model training (P1.5 v2) | v0.8.0-dev | same — both backends share the new student |
| Post-M8 hardening | (unreleased) | 296 (custom) + 367 (DS) — playback pacing, WS HoL, clip-extractor race, HLS init bound, DS soft-remove |

## Target Junctions

- 741 & 73
- 741 & Lytle South
- Drugmart & 73

## Documentation

- [Product Requirements Document](docs/prd.md)
- [Design Document](docs/design.md)
- [Changelog](docs/changelog.md)

## License

MIT
