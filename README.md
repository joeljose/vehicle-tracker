# Vehicle Tracker — Traffic Junction Monitor

A GPU-accelerated vehicle tracking system for traffic junction monitoring. Detects vehicles at road junctions, determines their directional transit (entry arm to exit arm), and generates alerts with best-quality vehicle photos — all from a single consumer GPU.

## Features

- **Directional transit detection** — tracks vehicles through a junction and reports entry/exit directions (e.g., "west to north")
- **Stagnant vehicle alerts** — detects vehicles stopped for configurable duration (default 2.5 min)
- **Best-photo capture** — automatically saves the highest quality frame of each tracked vehicle
- **Multi-channel** — up to 8 channels sharing a single GPU pipeline with independent phase transitions (~280 MB VRAM saved per channel vs separate pipelines)
- **4-phase workflow** — Setup (draw ROI/lines) -> Analytics (run detection) -> Review (replay alerts) -> Teardown
- **Browser-based UI** — React frontend with live MJPEG video, ROI/line drawing tools, alert feed, and video replay
- **Real-time streaming** — MJPEG video with detection overlays at ~15fps, WebSocket alerts, stats bar
- **Video replay** — clip extraction with bounding box overlay for reviewing transit events
- **Site configs** — save and load ROI/line configurations per junction

## Architecture

```
Browser (React) <-> FastAPI (REST + WebSocket + MJPEG) <-> DeepStream Pipeline (GPU)
```

The FastAPI layer communicates with the pipeline through a `PipelineBackend` Protocol interface, keeping the API layer pipeline-agnostic.

**Shared pipeline topology (M5):**
```
src_0 (nvurisrcbin) ─┐
src_1 (nvurisrcbin) ─┤
  ...                ├─> nvstreammux -> nvinfer (TrafficCamNet) -> nvtracker (NvDCF) -> nvstreamdemux
                     │                                                                    ├─> ch0: OSD -> MJPEG (~15fps)
                     │                                                                    ├─> ch1: OSD -> MJPEG (~15fps)
                     │   BatchMetadataRouter probe routes per-source metadata              ...
```

## Milestones

| Milestone | Status | Tag | Tests |
|-----------|--------|-----|-------|
| M1 — DeepStream pipeline | Complete | v0.1.0 | 177 |
| M2 — API layer (FastAPI REST/WS/MJPEG) | Complete | v0.2.0 | 237 |
| M3 — React UI (4-phase workflow, drawing tools) | Complete | v0.3.0 | 286 |
| M4 — DeepStream-FastAPI integration | Complete | v0.4.0 | 312 |
| M5 — Multi-channel shared pipeline | Complete | v0.5.0 | 336 |
| M6 — YouTube Live streams | Planned | — | — |

## Prerequisites

- **NVIDIA GPU** with driver >= 580.x (tested on RTX 4050 Max-Q, 6 GB VRAM)
- **Docker Engine** >= 24.0 with Docker Compose v2
- **NVIDIA Container Toolkit** ([install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
- **make** (`sudo apt install make`)

## Quick Start

```bash
git clone git@github.com:joeljose/vehicle-tracker.git
cd vehicle-tracker

make setup    # Create .env, cache dirs, build Docker images
make dev      # Start containers (backend idle, frontend hot-reload)
make start    # Launch backend server (uvicorn)

# Open http://localhost:5173 in your browser
```

## Development Workflow

All commands go through the Makefile. Run `make help` to see all targets.

### Setup & Lifecycle

```bash
make setup          # First-time setup: .env, cache dirs, build images
make dev            # Start dev containers
make down           # Stop all containers
make clean          # Stop + remove caches, snapshots, build artifacts
make rebuild        # Force rebuild Docker images (after Dockerfile changes)
```

### Backend Server

```bash
make start          # Start uvicorn inside backend container
make stop-server    # Stop uvicorn (container keeps running)
make restart        # Stop + start uvicorn
make logs           # Tail backend server logs
```

### Development

```bash
make shell          # Bash into backend container
make shell-frontend # Shell into frontend container
make test           # Run pytest (336 tests)
make test-v         # Run pytest verbose
make lint           # Run ruff check
make format         # Run ruff format (auto-fix)
```

### Data & Debugging

```bash
make status         # Container status, server health, GPU memory
make clip SRC="/data/site_videos/video.mp4" START=30 DURATION=60 OUT="/data/test_clips/clip.mp4"
```

### Release

```bash
make version        # Show current version (from git tags via setuptools-scm)
make tag            # Show version info, instructions for tagging
```

## Testing

```bash
make test           # Run all tests
make test-v         # Verbose output with test names
make lint           # Check for lint issues
```

Tests run inside the Docker container using pytest. The test suite uses a `FakeBackend` that implements the same `PipelineBackend` Protocol as the real DeepStream pipeline, so no GPU is needed for tests.

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
      stitch.py          # Track stitching across ID switches
      clip_extractor.py  # ffmpeg clip extraction for replay
      fake.py            # Test backend (no GPU)
    tests/               # 336 tests
    main.py              # FastAPI app factory
    Dockerfile           # Dev image (DeepStream + ffmpeg + pip deps)
    requirements.txt
  frontend/
    src/
      pages/             # Home, Channel
      components/        # AlertFeed, ReplayView, DrawingCanvas, StatsBar
      contexts/          # React Context for state management
      api/               # REST + WebSocket clients
  config/deepstream/     # DeepStream model configs
  data_collection/       # Test clips + site videos (gitignored)
  models/                # TensorRT engines (gitignored)
  docker-compose.dev.yml
  Makefile               # All dev commands (run `make help`)
  prd.md                 # Product Requirements Document
  vehicle_tracker_design.md  # Technical Design Document
```

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
