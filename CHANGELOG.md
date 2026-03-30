# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.4.0] - 2026-03-30

### Added
- DeepStreamPipeline adapter class implementing PipelineBackend Protocol
- Per-channel pipeline lifecycle: preview (Setup, loops) -> analytics (plays once, ROI) -> review (EOS)
- MjpegExtractor BufferOperator for GPU-to-JPEG frame extraction at ~15fps
- Real-time alert delivery via callbacks (transit + stagnant)
- Per-frame bbox/centroid/confidence data for replay overlay
- EOS auto-transition with `register_phase_callback()` for WS notification + clip extraction
- Best-photo snapshot disk fallback for finalized tracks

### Fixed
- Pipeline stop crash when called from asyncio event loop (deferred stop via background thread)
- Alert frame metadata (first_seen_frame, last_seen_frame, trajectory) now included in alert dicts
- Clip extraction works end-to-end with ffmpeg

## [0.3.0] - 2026-03-29

### Added
- React frontend with 4-phase workflow (Setup / Analytics / Review / Teardown)
- Home page with channel cards showing status and alert counts
- Channel page with MJPEG video panel and phase tabs
- ROI polygon drawing tool with canvas overlay
- Entry/exit line drawing tool (up to 4 lines per junction)
- Site configuration save/load (per-junction ROI + lines)
- Alert feed sidebar with All/Transit/Stagnant filter tabs
- Stats bar (FPS, active tracks, inference time, phase, alert count)
- Replay backend: clip extraction via ffmpeg, replay endpoint
- Review phase: clickable alert cards with video replay and bbox overlay
- WebSocket integration for real-time alerts, stats, and phase changes

## [0.2.0] - 2026-03-27

### Added
- FastAPI application with REST, WebSocket, and MJPEG endpoints
- PipelineBackend Protocol for pipeline-agnostic API boundary
- In-memory AlertStore with transit and stagnant alert support
- MJPEG streaming endpoint (`/stream/{channel_id}`)
- WebSocket broadcaster for frame data, alerts, stats, and phase changes
- Pipeline control endpoints (start, stop, add/remove channel, phase transitions)
- Alert and snapshot REST endpoints
- Site config REST endpoints (save, load, list)
- FakeBackend for testing without GPU

## [0.1.0] - 2026-03-25

### Added
- DeepStream pipeline: decode, TrafficCamNet detection, NvDCF tracking
- ROI polygon filtering with ray-casting point-in-polygon test
- Entry/exit line-crossing detection with auto-calibration
- Direction state machine (entry/exit/inferred transit detection)
- Stagnant vehicle detection (configurable threshold, default 2.5 min)
- Best-photo capture with area + confidence scoring
- Proximity-based track stitching across tracker ID switches
- Idle optimization (reduce inference interval when no detections)
- TrajectoryBuffer with configurable capacity
