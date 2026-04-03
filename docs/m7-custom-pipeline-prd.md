# Product Requirements Document
## M7 — Custom Pipeline (NVDEC + TensorRT + BoT-SORT)

**Version:** 0.2
**Status:** Draft (updated after grill session)
**Last updated:** April 2026
**Parent project:** Vehicle Tracker (v0.6.0)
**Target version:** v0.7.0

---

## 1. Purpose

Build an alternative vehicle tracking pipeline that replaces DeepStream's proprietary GStreamer elements with open, Python-native components: NVDEC hardware decoding, YOLOv8s inference via direct TensorRT API, and BoT-SORT multi-object tracking — with a GPU-resident data flow that minimizes PCIe transfers.

The custom pipeline implements the same `PipelineBackend` Protocol as the DeepStream adapter. The FastAPI layer, React frontend, and all downstream consumers are unaware which pipeline is running. Backend selection is a startup-time configuration choice.

---

## 2. Problem Statement

The DeepStream pipeline (M1-M6) works well but has significant drawbacks:

1. **Vendor lock-in:** DeepStream is NVIDIA-proprietary. The pyservicemaker Python bindings, GStreamer element behavior, and metadata formats are undocumented or poorly documented. Debugging requires reverse-engineering C/C++ internals.
2. **Deployment complexity:** The DeepStream container image is 37 GB. It bundles Triton, GStreamer, and dozens of unused components.
3. **Limited flexibility:** Adding custom processing stages (e.g., vehicle color extraction, plate detection) requires writing GStreamer plugins in C/C++ or fighting with the probe-based Python API.
4. **Fragile upgrades:** DeepStream 8.0 requires driver ≥580.x, DeepStream 9.0 requires ≥590.x. Version upgrades cascade through the entire stack.

The custom pipeline solves these by using standard, well-documented Python libraries (TensorRT, PyNvVideoCodec, CuPy) that run on any CUDA-capable GPU without a 37 GB container. The pipeline keeps frames on the GPU from decode through inference, matching DeepStream's GPU-resident data flow without DeepStream's proprietary stack.

---

## 3. Goals

- Implement `PipelineBackend` Protocol with identical behavior to the DeepStream adapter
- GPU-resident data flow: NVDEC decode → GPU preprocess → direct TensorRT inference → GPU NMS — frame never leaves GPU until MJPEG output
- Use NVDEC hardware decoder (via PyNvVideoCodec) for video decoding — frames stay as CUDA surfaces
- Use YOLOv8s TensorRT FP16 for detection via direct TensorRT Python API (no Ultralytics wrapper)
- Use standalone BoT-SORT for multi-object tracking on CPU (operates on bbox coords only, ~1KB per frame)
- Only GPU→CPU transfer per frame: bbox array (~1KB) for tracking + analytics
- Batch inference across channels (batch=N for N active channels)
- Reuse all shared pipeline modules: direction FSM, ROI filtering, alerts, snapshots, trajectory, stitch, idle optimization
- Pass all existing tests that use FakeBackend (API/WebSocket contract tests)
- Validate against the same test videos used for DeepStream (741_73, lytle_south, drugmart_73)
- Run in the same Docker container as the DeepStream backend — backend selection via config

---

## 4. Non-Goals

- Running both pipelines simultaneously (one backend at a time)
- Matching DeepStream's exact detection/tracking output frame-by-frame (different models produce different results — behavior must be equivalent, not identical)
- Exceeding DeepStream's performance (meeting real-time 30fps per channel is sufficient)
- Supporting more than 2 simultaneous channels in v1 (same constraint as DeepStream pipeline)
- Building a custom nvinfer bbox parser (that's a DeepStream concern — custom pipeline uses Ultralytics natively)
- Night-time or adverse weather support

---

## 5. Architecture

### 5.1 Container Strategy

**Same container, swappable backend.** Both pipeline implementations live in the backend container. New dependencies (`ultralytics`, `pynvvideocodec`) are added to `requirements.txt` and the Dockerfile. An environment variable selects the active backend at startup.

```
docker-compose.dev.yml:
  backend:
    environment:
      - PIPELINE_BACKEND=custom    # or "deepstream" (default)
```

```python
# main.py — backend selection
if os.environ.get("PIPELINE_BACKEND") == "custom":
    from backend.pipeline.custom.adapter import CustomPipeline as Backend
else:
    from backend.pipeline.deepstream.adapter import DeepStreamPipeline as Backend
```

**Rationale:** A separate container would duplicate the entire FastAPI app, WebSocket layer, AlertStore, and frontend proxy — all for switching one component. The Protocol abstraction exists precisely to avoid this.

### 5.2 Pipeline Topology

```
GPU ─────────────────────────────────────────────────────────────
  Per-channel NVDEC decode (PyNvVideoCodec)
    ch0: NVDEC → CUDA surface (NV12)
    ch1: NVDEC → CUDA surface (NV12)
           │                    │
           ▼                    ▼
    GPU preprocess (CuPy/CUDA kernel)
      color convert (NV12→RGB) → resize → normalize → float32 tensor
           │                    │
           ▼                    ▼
    ┌──────────────────────────────────────┐
    │  Batch: stack GPU tensors → (N,3,640,640) │
    │  TensorRT inference (YOLOv8s FP16)        │
    │  EfficientNMS plugin (on-GPU NMS)         │
    │  Split results per channel                │
    └──────────────────────────────────────┘
           │                    │
           ▼                    ▼
    bbox arrays (~1KB each) ─── only GPU→CPU transfer ───
─────────────────────────────────────────────────────────────────
CPU ─────────────────────────────────────────────────────────────
    Per-channel tracking (standalone BoT-SORT, ~1ms)
    ch0: tracker.update(dets)   ch1: tracker.update(dets)
           │                    │
           ▼                    ▼
    Per-channel post-processing (shared modules, Phase 2 only)
    ├── ROI filtering (roi.py)
    ├── Direction FSM (direction.py)
    ├── Best-photo scoring (snapshot.py)
    ├── Stagnation detection (idle.py)
    ├── Track stitching (stitch.py)
    └── Alert generation → callbacks
           │                    │
           ▼                    ▼
    MJPEG output (every other frame, ~15fps)
      GPU→CPU frame download → OpenCV draw bboxes → cv2.imencode
─────────────────────────────────────────────────────────────────
```

**Key design:** Frames stay on GPU from decode through inference. The only GPU→CPU transfer per inference cycle is the bbox array (~1KB). For MJPEG frames (~15fps, every other inference frame), one full frame is downloaded to CPU for annotation and JPEG encoding. This matches DeepStream's GPU-resident pattern without the proprietary stack.

### 5.3 Threading Model

```
Main thread:     asyncio event loop (uvicorn/FastAPI)
                    ↑ call_soon_threadsafe()
                    │
Pipeline thread: decode → infer → track → process → callbacks
```

A dedicated pipeline thread reads frames from all channels' `cv2.VideoCapture` instances, batches them for inference, runs tracking, and dispatches callbacks to the asyncio event loop via `call_soon_threadsafe()` — same pattern as the DeepStream adapter.

### 5.4 Component Selection

| Component | Library | Why |
|-----------|---------|-----|
| Decode | PyNvVideoCodec (NVDEC) | Hardware decode to CUDA surface — frame never touches CPU. Handles files via demuxer. HLS/RTSP TBD (spike). |
| GPU preprocess | CuPy | NV12→RGB color conversion, resize, normalize on GPU. No CPU round-trip. |
| Detection | TensorRT Python API (direct) | YOLOv8s FP16 engine. Direct GPU tensor in/out. No Ultralytics wrapper overhead or CPU↔GPU bouncing. |
| NMS | TensorRT EfficientNMS plugin | Baked into the TRT engine at export. Outputs final boxes on GPU. |
| Tracking | Standalone BoT-SORT (no ReID, no GMC) | Kalman + Hungarian on CPU. Operates on bbox coords only (~1KB). 103.6 fps validated in exp10. Imported from ultralytics.trackers or reimplemented. |
| Frame rendering | OpenCV | Draw bboxes on CPU numpy frame, encode JPEG via cv2.imencode. Only for MJPEG output frames (~15fps). |
| Best-photo crop | CuPy or CPU | Crop region from CUDA surface (GPU) or download frame to CPU for crop. TBD in spike. |

**Dropped from v0.1 PRD:** Ultralytics as inference/tracking wrapper. The high-level `model.track()` API forces CPU↔GPU round-trips every frame, limiting 2-channel throughput to 18.75 fps/channel (below 30fps target). Direct TensorRT + standalone tracker eliminates this bottleneck.

---

## 6. Protocol Contract

The custom pipeline must implement `PipelineBackend` Protocol exactly. This section summarizes the contract — see `backend/pipeline/protocol.py` for the canonical definition.

### 6.1 Lifecycle Methods

| Method | Behavior |
|--------|----------|
| `start()` | Load TensorRT engine, allocate GPU. Capture asyncio event loop. |
| `stop()` | Finalize all channels (save tracks/photos). Release GPU. Clear state. |

### 6.2 Channel Management

| Method | Behavior |
|--------|----------|
| `add_channel(channel_id, source, *, source_type, original_url)` | Create decoder for source. Add to channel dict. Start decode. |
| `remove_channel(channel_id)` | Finalize tracks/photos. Destroy decoder. Clean snapshots. |
| `configure_channel(channel_id, roi_polygon, entry_exit_lines)` | Store ROI/lines for ANALYTICS phase. |
| `set_channel_phase(channel_id, phase)` | Transition phase. On ANALYTICS: enable detection/tracking. On REVIEW: finalize. Fire phase callback. |

### 6.3 Callbacks (registered once at init)

| Callback | Signature | When Fired |
|----------|-----------|------------|
| Frame | `(FrameResult) -> None` | Every frame (~15fps for MJPEG) |
| Alert | `(dict) -> None` | Transit crossing or stagnation detected |
| Track ended | `(dict) -> None` | Track lost (no detections for N frames) |
| Phase | `(channel_id, new_phase, old_phase) -> None` | Phase transition (explicit or EOS auto) |

### 6.4 Data Structures

**FrameResult:** `channel_id`, `frame_number`, `timestamp_ms`, `detections: list[Detection]`, `annotated_jpeg: bytes`, `inference_ms`, `phase`, `idle_mode`

**Detection:** `track_id` (sequential int), `class_name`, `bbox` (x, y, w, h pixels), `confidence`, `centroid` (cx, cy)

**Alert dict:** `track_id`, `label`, `entry_arm`, `entry_label`, `exit_arm`, `exit_label`, `method`, `was_stagnant`, `first_seen_frame`, `last_seen_frame`, `trajectory`, `per_frame_data`, `channel`

### 6.5 Track ID Management

BoT-SORT produces integer track IDs that reset per-tracker instance. The adapter must maintain a mapping to sequential IDs (1, 2, 3...) consistent within a pipeline session, same as the DeepStream adapter's `_ds_to_seq` mapping.

---

## 7. Requirements

### 7.1 Decoding

| ID | Requirement | Priority |
|----|-------------|----------|
| C-01 | Decode video using NVDEC hardware decoder (PyNvVideoCodec) — frames stay as CUDA surfaces | Must |
| C-02 | Support file sources (mp4, mkv, avi) via PyNvVideoCodec demuxer | Must |
| C-02a | Support HLS streams (YouTube Live) — PyNvVideoCodec or OpenCV fallback (spike TBD) | Must |
| C-03 | Normalize all sources to 30fps for inference | Must |
| C-04 | File sources loop in SETUP phase, play once in ANALYTICS | Must |
| C-05 | Detect end-of-stream (EOS) for file sources and trigger auto-transition to REVIEW | Must |
| C-06 | OpenCV VideoCapture as fallback decoder for sources NVDEC can't handle | Should |

### 7.2 Detection

| ID | Requirement | Priority |
|----|-------------|----------|
| C-07 | Run YOLOv8s TensorRT FP16 inference via direct TensorRT Python API (no Ultralytics wrapper) | Must |
| C-07a | Export YOLOv8s engine with EfficientNMS plugin baked in (NMS on GPU) | Must |
| C-07b | GPU preprocess: NV12→RGB color convert, resize to 640x640, normalize — all on GPU via CuPy | Must |
| C-08 | Batch inference across active channels (batch=N) | Must |
| C-09 | Support runtime confidence threshold changes | Must |
| C-10 | Support runtime inference interval changes (idle optimization) | Must |
| C-11 | Detection output: class_name, bbox (x,y,w,h), confidence, per object per frame | Must |

### 7.3 Tracking

| ID | Requirement | Priority |
|----|-------------|----------|
| C-12 | Track objects using standalone BoT-SORT (no ReID, no GMC) on CPU | Must |
| C-13 | Per-channel tracker instances (no cross-channel ID collisions) | Must |
| C-14 | Map tracker IDs to sequential integers (1, 2, 3...) | Must |
| C-15 | Reset tracker on SETUP→ANALYTICS transition (fresh IDs) | Must |
| C-16 | Track buffer ≥150 frames for occlusion recovery (~5s at 30fps) | Should |

### 7.4 Shared Module Integration

| ID | Requirement | Priority |
|----|-------------|----------|
| C-17 | ROI filtering via `point_in_polygon()` from `roi.py` | Must |
| C-18 | Direction detection via `DirectionStateMachine` from `direction.py` | Must |
| C-19 | Best-photo scoring and extraction via `snapshot.py` | Must |
| C-20 | Track stitching via `stitch.py` for ID switch recovery | Must |
| C-21 | Trajectory tracking via `TrajectoryBuffer` from `trajectory.py` | Must |
| C-22 | Stagnation detection via displacement threshold from `direction.py` | Must |
| C-23 | Idle optimization: reduce inference interval when no detections | Should |

### 7.5 Frame Output

| ID | Requirement | Priority |
|----|-------------|----------|
| C-24 | Generate annotated JPEG frames with bounding boxes and track IDs drawn | Must |
| C-25 | MJPEG frame rate ~15fps (encode every other inference frame) | Must |
| C-26 | Frame callback fires with `FrameResult` dataclass | Must |

### 7.6 Pipeline Lifecycle

| ID | Requirement | Priority |
|----|-------------|----------|
| C-27 | `start()` is non-blocking — processing runs on background thread | Must |
| C-28 | `stop()` cleanly shuts down all decoders, releases GPU | Must |
| C-29 | Dynamic channel add/remove without restarting pipeline | Must |
| C-30 | Phase transitions work identically to DeepStream adapter | Must |
| C-31 | EOS auto-transition from ANALYTICS to REVIEW for file sources | Must |
| C-32 | YouTube Live stream recovery: retry + URL re-extraction + circuit breaker | Must |

### 7.7 Integration

| ID | Requirement | Priority |
|----|-------------|----------|
| C-33 | Backend selection via `PIPELINE_BACKEND` env var | Must |
| C-34 | All dependencies installed in existing backend Docker container | Must |
| C-35 | Makefile targets unchanged — `make start`, `make test`, etc. work for both backends | Must |
| C-36 | All existing FakeBackend tests pass (API contract tests) | Must |
| C-37 | New integration tests validate custom pipeline on test clips | Must |

---

## 8. Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Inference FPS (single channel) | ≥ 30 fps | Real-time requirement. Must match DeepStream. |
| Inference FPS (2 channels, batched) | ≥ 60 fps total | 30fps × 2 channels. DeepStream achieves this with GPU-resident flow. |
| Decode latency | < 5ms per frame | NVDEC hardware decode should be faster than OpenCV's 1.3ms (exp09). |
| End-to-end latency (decode → alert) | < 100ms | Alert delivery NF-05 in parent PRD |
| GPU→CPU transfer per inference frame | ~1 KB | Bbox array only. Full frame download only for MJPEG output (~15fps). |
| VRAM usage (2 channels) | < 1 GB | YOLOv8s FP16 (~380 MB) + NVDEC surfaces + preprocess buffers. Well within 6 GB. |
| MJPEG frame rate | ~15 fps | Matches DeepStream adapter |
| Container image size | < 15 GB | Significantly smaller than DeepStream's 37 GB |

---

## 9. Success Criteria

| Criteria | How to Verify |
|----------|---------------|
| All FakeBackend tests pass | `make test` — existing 371 tests |
| Custom pipeline produces transit alerts on 741_73_1min.mp4 | Run pipeline, verify alerts in UI |
| Custom pipeline produces transit alerts on lytle_south_1min.mp4 | Run pipeline, verify alerts in UI |
| Direction detection matches DeepStream results qualitatively | Same entry/exit arms detected for same vehicles |
| YouTube Live streaming works | Add YouTube Live URL, verify MJPEG stream and alerts |
| 2-channel operation works | Add 2 file sources, verify both produce alerts independently |
| Backend swap is transparent | Switch env var, restart — frontend works unchanged |
| MJPEG stream visible in browser | Navigate to channel, verify live video with bboxes |
| Best-photo snapshots captured | Check snapshots/ directory after analytics run |
| Replay works in Review phase | Click alert, verify replay overlay |

---

## 10. Milestones

| Phase | Deliverable | Dependencies |
|-------|-------------|--------------|
| P0 — Spike: GPU-resident pipeline | **Exp12:** NVDEC decode (PyNvVideoCodec) → GPU preprocess (CuPy) → direct TensorRT inference → EfficientNMS. Measure fps, latency, memory. Compare against exp11 baseline. **Exp13:** nvJPEG GPU encoding vs cv2.imencode. Validates GPU-resident tech stack on RTX 4050. | None |
| P1 — Single-channel adapter | `CustomPipeline` implements `PipelineBackend`. Single file source: NVDEC decode → GPU infer → standalone BoT-SORT → ROI → direction → alerts. All shared modules integrated. FakeBackend tests pass. | P0 |
| P2 — MJPEG + frame callbacks | Frame download for annotation, OpenCV draw, JPEG encode, frame callback, MJPEG streaming to browser. Stats (FPS, tracks, inference time). | P1 |
| P3 — Multi-channel + batching | Batch inference for 2 channels. Per-channel trackers. Dynamic add/remove. Phase transitions per channel. | P2 |
| P4 — YouTube Live + stream recovery | HLS stream support (PyNvVideoCodec or OpenCV fallback). Source resolver integration. Stream recovery with circuit breaker. | P3 |
| P5 — Integration + validation | Backend selection env var. Dockerfile updates. E2E validation on all 3 test clips + YouTube Live. Performance benchmarks. Container size optimization. | P4 |

---

## 11. Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| PyNvVideoCodec not available or broken in DeepStream container | Medium | High | Fall back to cv2.cudacodec.VideoReader (OpenCV CUDA decode). If neither works, OpenCV CPU decode + GPU upload is the worst-case fallback. Test in P0 spike. |
| PyNvVideoCodec can't handle HLS streams | Medium | Medium | Use OpenCV VideoCapture for HLS/YouTube Live sources, NVDEC for local files. Hybrid decode strategy. |
| Direct TensorRT API complexity (engine loading, bindings, pre/post-processing) | Medium | Medium | Well-documented API. Many reference implementations. Export engine with EfficientNMS to simplify postprocessing. Spike validates feasibility. |
| EfficientNMS plugin export fails or behaves differently | Low | Medium | Fall back to CuPy NMS on GPU, or worst-case CPU NMS (still fast at <100 detections per frame). |
| BoT-SORT standalone import from ultralytics.trackers has hidden dependencies | Medium | Medium | Reimplement Kalman + Hungarian from scratch if import is too tangled — it's ~200 lines of well-understood code. |
| GPU memory contention between NVDEC + inference | Low | Medium | NVDEC uses dedicated hardware (not CUDA cores). YOLOv8s FP16 uses ~380 MB. exp11 measured 152 MB for 2-channel. Well within 6 GB. |
| NV12→RGB color conversion adds GPU overhead | Low | Low | Standard CUDA kernel, well-optimized in CuPy. Negligible vs inference time. |
| CuPy preprocess (resize + normalize) produces different results than Ultralytics preprocess | Medium | Medium | Must match letterbox padding + normalization exactly. Validate by comparing TRT outputs with identical input frame. |

---

## 12. Files to Create/Modify

### New files (in `backend/pipeline/custom/`)

```
backend/pipeline/custom/
  __init__.py              # already exists (empty)
  adapter.py               # CustomPipeline implementing PipelineBackend
  decoder.py               # NVDEC decoder wrapper (PyNvVideoCodec, returns CUDA surfaces)
  preprocess.py            # GPU preprocess: NV12→RGB, resize, normalize via CuPy
  detector.py              # Direct TensorRT inference (batch-capable, GPU tensor in/out)
  tracker.py               # Standalone BoT-SORT wrapper (CPU, bbox coords only)
  renderer.py              # CPU bbox drawing + JPEG encoding (for MJPEG output)
```

### Modified files

```
backend/Dockerfile         # add cupy, pynvvideocodec to pip install
backend/requirements.txt   # add cupy, pynvvideocodec
backend/main.py            # backend selection via PIPELINE_BACKEND env var
docker-compose.dev.yml     # add PIPELINE_BACKEND env var
Makefile                   # optionally: make start-custom / make start-deepstream shortcuts
```

### Test files

```
backend/tests/
  test_custom_adapter.py   # unit tests for CustomPipeline
  test_custom_decoder.py   # NVDEC decoder wrapper tests
  test_custom_preprocess.py # GPU preprocess tests
  test_custom_detector.py  # direct TensorRT inference tests
  test_custom_tracker.py   # standalone BoT-SORT wrapper tests
  test_m7_integration.py   # E2E: custom pipeline on test clips
```

---

## 13. Out of Scope for M7

- Training a custom model (that's M8)
- INT8 quantization (FP16 is sufficient for real-time)
- More than 2 simultaneous channels
- Replacing the DeepStream backend entirely (both coexist)
- Custom GStreamer plugins or nvinfer parsers
- Mobile or edge deployment
- Night-time footage support
