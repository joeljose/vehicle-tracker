# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased] — M8-P1.5 v2

### Added
- Training container (`training/Dockerfile`) + Docker Compose service with
  full auto-label → fine-tune → export workflow
- Training scripts: frame extraction, PHash dedup, auto-labeling with
  pluggable teachers, stratified split, YOLOv8s fine-tune (fresh + continue),
  ONNX export, test-split evaluation, student-vs-teacher diff renderer
- Teacher implementations: YOLOv8x, YOLO26x, YOLO12x, RF-DETR-M, Grounding DINO-T
- 1000-frame teacher bake-off harness (`build_comparison_set.py`,
  `run_comparison.py`, `compute_comparison_stats.py`) with qualitative
  LS-based review instead of manual ground-truth mAP
- Label Studio workflow (`ls_setup.py` / `ls_export.py`) with
  `docker-compose.label-studio.yml` and legacy-token auto-init in the
  `train-ls-up` Makefile target
- Makefile `train-*` targets (`train-extract/dedup/label/split/student/
  student-continue/export-onnx/compare*/ls-*`)
- `plans/m8-custom-model-training.md` pointer plan
- Canonical site ROI + entry/exit line configs under `backend/config/sites/`
  for 741_73, drugmart, and lytle

### Changed
- Both backends now run a **single-class fine-tuned YOLOv8s** (project class
  0 = `vehicle`) instead of the COCO-pretrained 80-class yolov8s
- DeepStream YOLOv8 C++ parser rewritten for a (1,5,8400) tensor: no per-
  anchor argmax, no COCO vehicle-id filter, always emits classId 0
- Custom detector postprocess rewritten for the same 1-class tensor shape;
  `COCO_VEHICLE_CLASSES` replaced with `PROJECT_CLASS_NAMES`
- DeepStream `pgie_config.yml` rewritten: `num-detected-classes: 1`,
  single-line `labels.txt` (`vehicle`), `class-attrs-all: pre-cluster-
  threshold: 0.25` matching the parser's `CONF_THRESH`, deprecated
  `parse-func: 0` removed
- Custom engine builder defaults to `yolov8s_rfdetr_v1_cont.onnx` /
  `..._custom.engine`
- Top-level PRD/design M8 rows rewritten for the P1.5 v2 scope
  (10 h balanced inventory, RF-DETR-M teacher, single class)

### Fixed
- DeepStream `model-engine-file` path now matches nvinfer's auto-generated
  `<onnx>_b1_gpu0_fp16.engine` — previously the cache lookup missed on
  every restart and TRT rebuilt the engine (~150 s) every time
- Backend server `logging.basicConfig` wired up in `backend/main.py` so
  `backend.pipeline.*` INFO-level lines (e.g. "New track #N") actually
  surface under uvicorn; level is overridable via `VT_LOG_LEVEL`

## [0.7.0] - 2026-04-05

### Added
- Custom GPU-resident pipeline: NVDEC + CuPy + direct TensorRT + BoT-SORT
- Vendored BoT-SORT tracker (no ultralytics/torch dependency)
- HSV histogram ReID encoder for appearance-based track matching
- Container split: Dockerfile.custom (5.9 GB) vs Dockerfile.deepstream (37.8 GB)
- YouTube Live HLS decode via GStreamer FIFO + PyNvVideoCodec
- Stream recovery with CircuitBreaker (reused from DeepStream)
- Shared pipeline utilities (shared.py): get_snapshot, check_crossings, finalize_lost_track, parse_lines
- Decoupled processing/display threads with PTS-based FIFO for real-time playback
- YOLOv8s custom C++ bbox parser for DeepStream nvinfer (both backends now use same model)
- ROI polygon overlay in Analytics and Review (frontend-only, common to both backends)
- exp15 resource comparison benchmark
- Custom model training PRD and design doc (training/docs/)

### Changed
- DeepStream detector swapped from TrafficCamNet to YOLOv8s for fair comparison
- Display rate: source FPS via PTS timing (was hardcoded 15fps)
- per_frame_data timestamps use decoder PTS (was computed from frame count)
- Clip extraction uses per_frame_data timestamps (fps-agnostic)
- Docker image optimized: 10.6 GB → 5.9 GB (--no-install-recommends, strip .so, no ultralytics/torch)

### Fixed
- Same-arm crossing rejection (prevents false N→N / E→E alerts from bbox jitter)
- COMPLETED state in DirectionStateMachine (prevents duplicate alerts)
- Snapshot RGB→BGR conversion + disk fallback
- Clean last_frame.jpg for frozen-frame replay (no analytics overlays)
- Stitcher merge inherits per_frame_data (replay animation for merged tracks)
- Replay bbox timing (only between first/last detection)
- MJPEG reconnect on phase transition (cache-busting query param)
- PTS conversion: 90kHz MPEG ticks → milliseconds

## [0.6.0] - 2026-03-31

### Added
- YouTube Live stream support: accept YouTube URLs as channel sources
- Source resolver module (`source_resolver.py`) with yt-dlp URL resolution to HLS
- Async channel add: `POST /channel/add` returns 202 for YouTube URLs, WS notification on completion
- Stream recovery: retry 3x with backoff → liveness check → URL re-extraction → reconnect
- Circuit breaker: auto-eject channel after 3 pipeline rebuilds in 10 minutes
- Serialized yt-dlp calls via asyncio.Lock to prevent YouTube rate limiting
- Last frame buffer: MjpegExtractor keeps most recent JPEG, persisted at Phase 2→3 transition
- Frozen-frame replay for YouTube Live: animated bbox on static last frame (transit), static bbox (stagnant)
- `GET /channel/{id}/last_frame` endpoint for Phase 3 replay image
- "Live" badge with red pulse indicator on channel cards for YouTube sources
- yt-dlp + deno precompiled binaries in Dockerfile for YouTube JS extraction
- Validation spike results documented in `docs/spikes/m6-spike-results.md`

### Changed
- `add_channel()` now accepts `source_type` and `original_url` kwargs (Protocol + all backends)
- `nvurisrcbin` source URI: raw URL for HLS streams, `file://` prefix for local files
- `file-loop` property only set for file sources (YouTube HLS streams don't loop)
- Design doc Section 3.4 rewritten with grill session decisions
- Quality selection removed from PRD (F-04a) — always best available quality

## [0.5.0] - 2026-03-31

### Added
- Multi-channel shared pipeline: nvstreammux → nvinfer → nvtracker → nvstreamdemux → per-channel OSD/MJPEG
- BatchMetadataRouter for per-source metadata dispatch to channel-specific TrackingReporters
- Dynamic channel add/remove via pipeline rebuild (nvurisrcbin limitation workaround)
- Per-source EOS detection via frame-gap watchdog in BatchMetadataRouter
- Independent phase transitions per channel (Setup/Analytics/Review)
- Per-channel snapshot directories (snapshots/{channel_id}/) with cleanup on channel removal
- Stop Analytics button for manual analytics→review transition
- BroadcastChannel cross-tab notification on channel removal
- Snapshot extraction skipped during setup phase (no wasted crops from looping preview)
- 2-channel E2E validated on RTX 4050 (~459 MB shared VRAM vs ~738 MB separate)

### Fixed
- phase_callback now fires on all transitions (was only firing on EOS auto-transition)
- Video scroll during analytics: fixed viewport overflow with h-screen layout
- Snapshot thumbnails and replay clips load correctly in review phase

### Removed
- Legacy per-channel pipeline builders (_start_preview_pipeline, _start_analytics_pipeline, _stop_channel_pipeline)

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
