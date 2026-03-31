# Vehicle Tracker — Traffic Junction Monitor

GPU-accelerated vehicle tracking for traffic junction monitoring. Detects vehicles, determines directional transit (entry arm to exit arm), and generates alerts with best-quality vehicle photos — all from a single consumer GPU.

## Features

- **Directional transit detection** — entry/exit directions (e.g., "west to north")
- **Stagnant vehicle alerts** — vehicles stopped for configurable duration (default 2.5 min)
- **Best-photo capture** — highest quality frame of each tracked vehicle
- **Multi-channel** — up to 8 channels sharing a single GPU pipeline (~280 MB VRAM saved per channel)
- **YouTube Live support** — monitor live traffic camera streams directly
- **4-phase workflow** — Setup (draw ROI/lines) -> Analytics (run detection) -> Review (replay alerts)
- **Browser UI** — live MJPEG video, ROI/line drawing tools, alert feed, video replay

## Prerequisites

- **NVIDIA GPU** with driver >= 580.x (tested on RTX 4050 Max-Q, 6 GB VRAM)
- **Docker Engine** >= 24.0 with Docker Compose v2
- **NVIDIA Container Toolkit** ([install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
- **make** (`sudo apt install make`)

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
make test             # Run all tests (372 tests, no GPU needed)
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
Browser (React) <-> FastAPI (REST + WebSocket + MJPEG) <-> DeepStream Pipeline (GPU)
```

The FastAPI layer communicates with the pipeline through a `PipelineBackend` Protocol interface, keeping the API layer pipeline-agnostic.

**Shared pipeline topology:**
```
src_0 (nvurisrcbin) --+
src_1 (nvurisrcbin) --+--> nvstreammux -> nvinfer -> nvtracker -> nvstreamdemux
                      |    (TrafficCamNet)  (NvDCF)                  |-> ch0: OSD -> MJPEG
                      |                                              |-> ch1: OSD -> MJPEG
                      |   BatchMetadataRouter routes per-source metadata
```

## Project Structure

```
vehicle-tracker/
  backend/
    api/                 # FastAPI routes, WebSocket, MJPEG streaming
    config/              # Site config model + saved junction configs
    pipeline/
      deepstream/        # DeepStream adapter (GPU pipeline)
      protocol.py        # PipelineBackend Protocol interface
      direction.py       # Line-crossing + direction state machine
      roi.py             # ROI polygon filtering
      alerts.py          # AlertStore (transit + stagnant)
      snapshot.py        # Best-photo capture
      clip_extractor.py  # ffmpeg clip extraction for replay
      fake.py            # Test backend (no GPU needed)
    tests/               # 372 tests
    main.py              # FastAPI app factory
    Dockerfile
  frontend/
    src/
      pages/             # Home, Channel
      components/        # AlertFeed, ReplayView, DrawingCanvas, StatsBar
      contexts/          # React Context state management
      api/               # REST + WebSocket clients
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

## Target Junctions

- 741 & 73
- 741 & Lytle South
- Drugmart & 73

## Documentation

- [Product Requirements Document](prd.md)
- [Design Document](vehicle_tracker_design.md)
- [Changelog](CHANGELOG.md)

## License

MIT
