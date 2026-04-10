# M7 Implementation Plan: Custom GPU-Resident Pipeline

**Milestone:** M7 — Custom pipeline
**Version target:** 0.7.0
**Tests at completion:** 291 (custom) + 363 (deepstream)

**Development notes:**
- All development and testing happens inside Docker containers
- Two separate Docker images: `Dockerfile.custom` (5.9 GB) and `Dockerfile.deepstream` (37.8 GB)
- Backend selection via `BACKEND_TYPE` Makefile arg or `VT_BACKEND` env var (default: custom)

**Key design decisions:**
- GPU-resident data flow: NVDEC → CuPy → TensorRT → CPU NMS → BoT-SORT (CPU)
- Vendored BoT-SORT from ultralytics v8.4 — stripped torch/ultralytics dependency entirely
- HSV histogram ReID encoder (24-dim, no torch) for appearance-based track matching
- Decoupled processing/display threads: processing at full GPU speed, display at source FPS via PTS-based FIFO
- Both backends use same YOLOv8s model (DeepStream via custom C++ bbox parser)
- Shared analytics utilities in `backend/pipeline/shared.py`
- ROI polygon and entry/exit lines rendered by frontend (common to both backends)
- PTS-based timestamps from decoder (fps-agnostic, correct for 30/60fps sources)

---

## Phase 1: Single-Channel Adapter

**Goal:** CustomPipeline class implementing PipelineBackend Protocol with NVDEC decode, CuPy preprocess, direct TRT inference, BoT-SORT tracking.

**Key files:**
- `backend/pipeline/custom/adapter.py` — main adapter
- `backend/pipeline/custom/decoder.py` — NVDEC hardware decoder
- `backend/pipeline/custom/detector.py` — direct TRT inference + CPU NMS
- `backend/pipeline/custom/preprocess.py` — GPU preprocess (NV12→RGB, letterbox, normalize)
- `backend/pipeline/custom/tracker.py` — BoT-SORT wrapper
- `backend/pipeline/custom/renderer.py` — bbox drawing + JPEG encode
- `backend/pipeline/custom/engine_builder.py` — TRT engine from ONNX

**Acceptance criteria:**
- [x] CustomPipeline implements PipelineBackend Protocol
- [x] Single-channel file playback with vehicle detection + tracking
- [x] MJPEG streaming with bounding boxes
- [x] Phase transitions (Setup → Analytics → Review)
- [x] All shared analytics modules integrated (ROI, direction, snapshot, stitch, trajectory, idle)

## Phase 2: MJPEG + Frame Callbacks

**Goal:** Frame rendering with bbox annotation, MJPEG streaming, stats via WebSocket.

**Acceptance criteria:**
- [x] NV12→BGR GPU conversion for frame rendering
- [x] JPEG encoding (~4ms per frame)
- [x] FrameResult callbacks with stats (FPS, track count, inference time)
- [x] MJPEG streaming to browser

## Phase 3: Multi-Channel + Batching

**Goal:** Per-channel decoders and trackers, dynamic add/remove.

**Acceptance criteria:**
- [x] Per-channel NVDEC decoders and BoT-SORT trackers
- [x] Dynamic add/remove channels without pipeline restart
- [x] Per-channel phase transitions
- [x] Sequential inference (1 channel at a time)

## Phase 4: YouTube Live + Stream Recovery + Container Split

**Goal:** HLS stream support, recovery, separate Docker images.

**Key files:**
- `backend/Dockerfile.custom` — slim CUDA + GStreamer + TRT image
- `backend/Dockerfile.deepstream` — full DeepStream image
- `docker-compose.custom.yml` / `docker-compose.deepstream.yml`

**Acceptance criteria:**
- [x] GStreamer subprocess handles HLS demux → FIFO → PyNvVideoCodec
- [x] Stream recovery with CircuitBreaker (reused from DeepStream)
- [x] Container split: custom (5.9 GB) vs deepstream (37.8 GB)
- [x] Backend selection via Makefile `BACKEND_TYPE` arg
- [x] Test reorganization (custom/ vs deepstream/ directories)

## Phase 5: Integration, Validation, Bug Fixes

**Goal:** E2E validation, both backends aligned, bugs fixed.

**Acceptance criteria:**
- [x] E2E: 741_73, drugmart_73, lytle_south clips validated
- [x] E2E: YouTube Live stream validated
- [x] E2E: 2-channel operation validated
- [x] E2E: Phase transitions + Review replay validated
- [x] Both backends use YOLOv8s (DeepStream via custom C++ bbox parser)
- [x] Vendored BoT-SORT (no ultralytics/torch dependency)
- [x] HSV histogram ReID encoder enabled
- [x] Shared utilities extracted (shared.py)
- [x] Direction FSM: same-arm rejection, COMPLETED state (no duplicate alerts)
- [x] Frame pacing: decoupled processing/display threads, PTS-based FIFO
- [x] Snapshot fixes: RGB→BGR, disk fallback, clean last_frame.jpg
- [x] Stitcher merge inherits per_frame_data
- [x] Replay: bbox timing, clip extraction fps-agnostic, PTS conversion fix
- [x] ROI/lines rendering moved to frontend-only overlays
- [x] Docker image optimized: 10.6 GB → 5.9 GB
- [x] exp15 resource comparison benchmark
- [x] All docs updated for M7 completion

## Performance (RTX 4050 Laptop, single channel)

| Metric | Custom | DeepStream |
|---|---|---|
| Wall-clock (1-min video) | 58.4s | 60.6s |
| Avg CPU % | 53.1 | 401.7 |
| Avg GPU memory (MB) | 336 | 1679 |
| Avg GPU utilization % | 16.2 | 37.7 |
| Avg GPU power (W) | 17.3 | 30.8 |
| Alert count | 40 | 40 |
| Docker image size | 5.9 GB | 37.8 GB |
