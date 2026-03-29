# Vehicle Tracker — Traffic Junction Monitor

A GPU-accelerated vehicle tracking system for traffic junction monitoring. Detects vehicles at road junctions, determines their directional transit (entry arm to exit arm), and generates alerts with best-quality vehicle photos.

## Features

- **Directional transit detection** — tracks vehicles through a junction and reports entry/exit directions (e.g., "741-North to 73-East")
- **Stagnant vehicle alerts** — detects vehicles stopped for configurable duration (default 2.5 min)
- **Best-photo capture** — automatically captures the highest quality frame of each tracked vehicle
- **Two pipeline backends** — NVIDIA DeepStream (primary) and custom NVDEC+TensorRT+ByteTrack (secondary), behind a unified API
- **4-phase workflow** — Setup (draw ROI/lines) → Analytics (run detection) → Review (replay alerts) → Teardown
- **Browser-based UI** — React frontend with alert feed, ROI/line drawing tools, and video replay
- **Recorded video and RTSP** — analyze pre-recorded footage or monitor live camera streams

## Status

- **M1 — DeepStream pipeline:** Complete (v0.1.0) — single-channel detection, tracking, direction FSM, alerts, best-photo capture. 177 tests.
- **M2 — API layer:** Complete (v0.2.0) — FastAPI REST/WebSocket/MJPEG endpoints, AlertStore, PipelineBackend Protocol. 237 tests.
- **M3 — React UI:** Next

## Target Hardware

- NVIDIA GeForce RTX 4050 Max-Q (Mobile), 6 GB VRAM
- Single machine, single GPU, fully offline after model download

## Architecture

```
Browser (React) ←→ FastAPI (REST + WebSocket + MJPEG) ←→ Pipeline (DeepStream OR Custom)
```

Both pipeline backends implement the same `PipelineBackend` Protocol. The FastAPI layer and React frontend are pipeline-agnostic.

## Target Junctions

- 741 & 73
- 741 & Lytle South
- Drugmart & 73

## Documentation

- [Product Requirements Document](prd.md)
- [Design Document](vehicle_tracker_design.md)

## License

MIT
