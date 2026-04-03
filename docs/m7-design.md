# M7 — Custom Pipeline Design Document

**Version:** 1.0
**Status:** Ready for implementation
**Last updated:** April 2026
**Parent:** [M7 PRD](m7-custom-pipeline-prd.md) (v0.2)
**Informed by:** exp09-exp13 spike results, grill session

---

## 1. Summary

A GPU-resident custom pipeline that replaces DeepStream's proprietary GStreamer stack with direct NVDEC + TensorRT + BoT-SORT. Frames stay on GPU from decode through inference. The only GPU→CPU transfer per inference cycle is the bbox array (~1KB). For MJPEG output (~15fps), one frame is downloaded to CPU for annotation and JPEG encoding.

**Validated performance (exp12):**

| Metric | Result | Target |
|--------|--------|--------|
| Single channel | 159.4 fps | ≥30 fps |
| Dual channel | 82.6 fps total (41.3 fps/ch) | ≥60 fps total |
| GPU memory (dual) | 210 MB | <1 GB |

---

## 2. Architecture

### 2.1 Data Flow

```
GPU ───────────────────────────────────────────────────────────────
  NVDEC decode (PyNvVideoCodec 2.1)
    ch0: demux → decode → CUDA NV12 surface (1620×1920, uint8)
    ch1: demux → decode → CUDA NV12 surface
           │                    │
           ▼                    ▼
  GPU preprocess (CuPy CUDA kernel + cupyx.scipy.ndimage)
    NV12→RGB (custom CUDA kernel) → resize (bilinear) → normalize → float32
           │                    │
           ▼                    ▼
  TensorRT inference (direct Python API, pip TRT 10.16)
    input: (1, 3, 640, 640) float32 per channel (sequential, not batched*)
    output: (1, 84, 8400) float32 per channel
           │                    │
           ▼                    ▼
  bbox download ─── ~2.7 MB per output tensor ─── GPU→CPU ───────
───────────────────────────────────────────────────────────────────
CPU ───────────────────────────────────────────────────────────────
  NMS (numpy, ~1.8ms)
    conf filter → cxcywh→xyxy → IoU NMS → scale to original coords
           │                    │
           ▼                    ▼
  BoT-SORT tracking (standalone, ~0.9ms per channel)
    per-channel tracker instances, no cross-channel ID collision
           │                    │
           ▼                    ▼
  Phase routing ("always infer, conditionally analyze")
    SETUP: render boxes only, no analytics
    ANALYTICS: ROI filter → direction FSM → alerts → best-photo → stitch
           │                    │
           ▼                    ▼
  MJPEG output (every other inference frame, ~15fps)
    GPU→CPU frame download (1.3ms) → OpenCV draw bboxes (1.1ms)
    → cv2.imencode q=80 (4.1ms) → FrameResult callback
───────────────────────────────────────────────────────────────────
```

*Sequential inference (one `execute_async_v3` per channel) rather than batch inference. The engine is exported with batch=1. Batch=N would require re-exporting the engine with dynamic batch and adds complexity for marginal gain — sequential 2-channel inference is 4.85ms total (exp12), well within budget.

### 2.2 Component Stack

| Component | Library | Validated in |
|-----------|---------|-------------|
| Decode | PyNvVideoCodec 2.1.0 — NVDEC hardware, NV12 CUDA surfaces | exp12: 902 fps |
| Color convert | CuPy RawKernel — NV12→RGB on GPU | exp12: part of 1.13ms preprocess |
| Resize + normalize | cupyx.scipy.ndimage.zoom + CuPy ops | exp12: part of 1.13ms preprocess |
| Inference | TensorRT 10.16 Python API — direct `execute_async_v3` | exp12: 1.63ms single |
| NMS | numpy on CPU | exp12: 1.77ms |
| Tracking | BoT-SORT (no ReID, no GMC) via ultralytics.trackers | exp12: 0.89ms |
| Frame annotation | OpenCV `cv2.rectangle` + `cv2.putText` on CPU | exp11: 1.1ms |
| JPEG encode | `cv2.imencode` q=80 on CPU | exp13: 4.15ms (5.36ms with GPU download) |
| Analytics | Shared modules: direction.py, roi.py, snapshot.py, stitch.py, idle.py, alerts.py | M1-M6 |

### 2.3 TensorRT Engine

The engine is built from `yolov8s.onnx` using the pip TensorRT 10.16 Python API (`trt.Builder`). **Not** using `trtexec` (system TRT 10.9 — version mismatch) or Ultralytics export (wraps engine in JSON metadata).

| Property | Value |
|----------|-------|
| Model | YOLOv8s |
| Precision | FP16 |
| Input | `images`: (1, 3, 640, 640) float32 |
| Output | `output0`: (1, 84, 8400) float32 |
| NMS | CPU-side (not baked into engine) |
| Size | 25.2 MB |
| Path | `/app/models/yolov8s_direct.engine` |

**Engine build:** Part of container setup (Dockerfile or first-run script). Build takes ~150s. Engine is GPU-specific — must be rebuilt if GPU changes.

**Output format:** (1, 84, 8400) where 84 = 4 bbox coords (cx, cy, w, h) + 80 COCO class scores. Postprocessed on CPU: confidence filter → class filter (car=2, motorcycle=3, bus=5, truck=7) → cxcywh→xyxy → NMS → scale to original coordinates.

---

## 3. Module Design

### 3.1 File Layout

```
backend/pipeline/custom/
    __init__.py
    adapter.py          # CustomPipeline — implements PipelineBackend Protocol
    decoder.py          # NvDecoder — NVDEC decode wrapper
    preprocess.py       # GPU preprocess: NV12→RGB, letterbox resize, normalize
    detector.py         # TRTDetector — direct TensorRT inference + CPU NMS
    tracker.py          # TrackerWrapper — standalone BoT-SORT + ID mapping
    renderer.py         # FrameRenderer — bbox drawing + JPEG encoding
    engine_builder.py   # Build TRT engine from ONNX (first-run or Dockerfile)
```

### 3.2 NvDecoder (`decoder.py`)

Wraps PyNvVideoCodec demuxer + decoder. One instance per channel.

```python
class NvDecoder:
    """NVDEC hardware decoder. Returns NV12 CuPy arrays on GPU."""

    def __init__(self, source: str, *, loop: bool = False):
        """
        Args:
            source: File path (mp4/mkv/avi) or HLS URL.
            loop: If True, seek to start on EOS (Setup phase looping).
        """

    def read(self) -> tuple[cp.ndarray | None, int]:
        """Decode next frame.
        Returns: (nv12_gpu_frame, pts_ms) or (None, -1) on EOS.
        """

    def release(self) -> None:
        """Release NVDEC resources."""

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def fps(self) -> float: ...
```

**Looping (Setup phase):** On EOS, call `demuxer.Seek(0)` to restart from beginning. On Setup→Analytics transition, the decoder is destroyed and recreated with `loop=False`.

**HLS streams:** PyNvVideoCodec may not support HLS directly. Fallback: use `ffmpeg` subprocess to remux HLS→pipe, feed raw H.264 packets to decoder. If that fails, fall back to OpenCV VideoCapture (CPU decode + GPU upload). Determine in P4.

**Frame rate normalization:** Source is 60fps, inference at 30fps. Decoder reads every frame; the pipeline loop skips every other frame for inference (`frame_count % 2 != 0 → skip`). MJPEG encodes every other inference frame (~15fps).

### 3.3 GPU Preprocess (`preprocess.py`)

```python
class GpuPreprocessor:
    """GPU-resident preprocess: NV12→RGB, letterbox resize, normalize."""

    def __init__(self, input_size: int = 640):
        """Compile CUDA kernel, pre-allocate output buffer."""

    def __call__(self, nv12_frame: cp.ndarray, height: int, width: int
                 ) -> tuple[cp.ndarray, ScaleInfo]:
        """
        Returns:
            input_tensor: (1, 3, 640, 640) float32 CuPy array on GPU.
            scale_info: (scale, pad_top, pad_left, orig_h, orig_w) for
                        mapping detections back to original coordinates.
        """
```

**NV12→RGB:** Custom CuPy `RawKernel` (BT.601 color conversion). 32×32 thread blocks, one thread per pixel. Validated in exp12.

**Letterbox resize:** `cupyx.scipy.ndimage.zoom` with bilinear interpolation. Pads to 640×640 with 0.5 fill (matching Ultralytics convention). Aspect ratio preserved.

**Normalize:** `/ 255.0` to [0, 1] float32. No ImageNet mean/std subtraction (YOLOv8 uses [0, 1] normalization).

**Pre-allocated buffers:** The output tensor `(1, 3, 640, 640)` and padded intermediate `(640, 640, 3)` are allocated once and reused to avoid per-frame GPU allocations.

### 3.4 TRT Detector (`detector.py`)

```python
class TRTDetector:
    """Direct TensorRT inference + CPU-side NMS."""

    def __init__(self, engine_path: str, conf_thresh: float = 0.25):
        """Load engine, create execution context, allocate GPU output buffer."""

    def detect(self, input_tensor: cp.ndarray, scale_info: ScaleInfo,
               ) -> np.ndarray:
        """
        Run inference + NMS.

        Args:
            input_tensor: (1, 3, 640, 640) CuPy float32 on GPU.
            scale_info: For mapping boxes to original frame coords.

        Returns:
            detections: (N, 6) numpy array [x1, y1, x2, y2, conf, cls]
        """

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update confidence threshold at runtime."""
```

**Inference:** `context.set_tensor_address()` + `context.execute_async_v3()` on a dedicated CuPy CUDA stream. Stream is synchronized after execution.

**NMS:** CPU-side. Download output tensor (2.7 MB), transpose (8400, 84), filter by confidence + vehicle classes, convert cxcywh→xyxy, run greedy NMS, scale back to original coordinates. ~1.8ms total.

**Confidence threshold:** Stored as instance variable, applied during NMS confidence filter. Runtime-changeable via `set_confidence_threshold()`.

### 3.5 Tracker Wrapper (`tracker.py`)

```python
class TrackerWrapper:
    """Per-channel BoT-SORT tracker with sequential ID mapping."""

    def __init__(self):
        """Create BoT-SORT instance, initialize ID map."""

    def update(self, detections: np.ndarray
               ) -> list[tuple[int, np.ndarray, float, str]]:
        """
        Feed detections to tracker.

        Args:
            detections: (N, 6) array [x1, y1, x2, y2, conf, cls].

        Returns:
            List of (seq_track_id, bbox_xywh, confidence, class_name).
        """

    def reset(self) -> None:
        """Reset tracker state. Called on Setup→Analytics transition."""
```

**ID mapping:** BoT-SORT produces arbitrary integer IDs that can reset. The wrapper maintains a `dict[int, int]` mapping tracker IDs to sequential IDs (1, 2, 3...) consistent within a pipeline session. Same pattern as DeepStream adapter's `_ds_to_seq` mapping.

**DetectionResults adapter:** BoT-SORT expects a Results-like object with `.conf`, `.xyxy`, `.xywh`, `.cls` attributes and `__getitem__`/`__len__`. A lightweight `DetectionResults` dataclass wraps the numpy array. Validated in exp12.

### 3.6 Frame Renderer (`renderer.py`)

```python
class FrameRenderer:
    """Renders annotated frames for MJPEG output."""

    def render(self, gpu_frame: cp.ndarray, height: int, width: int,
               tracks: list, phase: ChannelPhase,
               roi_polygon: list | None = None,
               entry_exit_lines: dict | None = None,
               ) -> bytes:
        """
        Download frame to CPU, draw bboxes/IDs, encode JPEG.

        Returns: JPEG bytes.
        """
```

**Flow:** NV12→BGR on GPU (CuPy kernel) → `cp.asnumpy()` (1.3ms download) → `cv2.rectangle` + `cv2.putText` for each track → `cv2.imencode('.jpg', quality=80)` → bytes.

**Rate limiting:** Called every other inference frame (~15fps). The adapter controls the rate, not the renderer.

**ROI/lines overlay:** During Setup and Analytics, draw ROI polygon outline and entry/exit lines on the frame (thin green lines). Same visual feedback as DeepStream OSD.

### 3.7 Engine Builder (`engine_builder.py`)

```python
def build_engine(onnx_path: str, engine_path: str, fp16: bool = True) -> None:
    """Build TRT engine from ONNX using pip TRT Python API."""
```

Called at container startup if the engine doesn't exist or the ONNX has changed (checksum comparison). Takes ~150s on RTX 4050. Logs progress.

---

## 4. Adapter Design (`adapter.py`)

`CustomPipeline` implements `PipelineBackend` Protocol. It orchestrates all components.

### 4.1 State

```python
@dataclass
class ChannelState:
    """Per-channel mutable state."""
    decoder: NvDecoder
    tracker: TrackerWrapper
    phase: ChannelPhase = ChannelPhase.SETUP
    source: str = ""
    source_type: str = "file"       # "file" or "youtube_live"
    original_url: str | None = None
    roi_polygon: list | None = None
    entry_exit_lines: dict | None = None
    frame_count: int = 0
    infer_count: int = 0
    # Shared module instances (per-channel)
    reporter: TrackingReporter | None = None  # ROI, direction, alerts, stitch
    photo_tracker: BestPhotoTracker | None = None
```

### 4.2 Pipeline Loop

A single background thread runs the main loop:

```python
def _pipeline_loop(self):
    """Main pipeline loop — runs on background thread."""
    while self._running:
        # 1. Drain phase transition queue
        self._apply_pending_transitions()

        # 2. Decode one frame per active channel
        frames = {}
        for ch_id, state in self._channels.items():
            nv12, pts = state.decoder.read()
            if nv12 is None:
                self._handle_eos(ch_id)
                continue
            state.frame_count += 1
            # Frame rate normalization: skip odd frames (60→30fps)
            if state.frame_count % 2 != 0:
                continue
            frames[ch_id] = (nv12, pts, state)

        if not frames:
            time.sleep(0.001)  # avoid busy spin when no channels
            continue

        # 3. Preprocess + infer + NMS + track per channel
        for ch_id, (nv12, pts, state) in frames.items():
            # GPU preprocess
            input_tensor, scale_info = self._preprocess(nv12,
                                                         state.decoder.height,
                                                         state.decoder.width)
            # TRT inference + NMS
            detections = self._detector.detect(input_tensor, scale_info)

            # BoT-SORT tracking
            tracks = state.tracker.update(detections)

            state.infer_count += 1

            # 4. Phase routing
            if state.phase == ChannelPhase.ANALYTICS and state.reporter:
                # Full analytics: ROI, direction, alerts, best-photo, stitch
                self._process_analytics(ch_id, state, tracks, nv12, pts)

            # 5. MJPEG frame output (~15fps: every other inference frame)
            if state.infer_count % 2 == 0 and self._frame_callback:
                jpeg = self._renderer.render(nv12,
                                             state.decoder.height,
                                             state.decoder.width,
                                             tracks, state.phase,
                                             state.roi_polygon,
                                             state.entry_exit_lines)
                result = FrameResult(
                    channel_id=ch_id,
                    frame_number=state.infer_count,
                    timestamp_ms=pts,
                    detections=self._to_detection_list(tracks),
                    annotated_jpeg=jpeg,
                    inference_ms=...,
                    phase=state.phase.value,
                )
                self._loop.call_soon_threadsafe(self._frame_callback, result)
```

### 4.3 Phase Transitions

Phase transitions are requested from the API thread via a thread-safe queue. The pipeline loop drains the queue between frames (before decode).

| Transition | Action |
|---|---|
| `add_channel` | Create `NvDecoder(source, loop=True)`, `TrackerWrapper()`. Phase=SETUP. |
| Setup → Analytics | Destroy decoder. Create new `NvDecoder(source, loop=False)`. Reset tracker. Create `TrackingReporter` with ROI/lines. Create `BestPhotoTracker`. Phase=ANALYTICS. |
| Analytics → Review (EOS) | Finalize all tracks (fire track_ended for remaining). Destroy decoder. Fire phase callback. Phase=REVIEW. |
| Analytics → Review (manual) | Same as EOS transition. |
| `remove_channel` | Destroy decoder. Clear all state. Remove from channels dict. |

### 4.4 Threading

```
Main thread:     asyncio event loop (uvicorn/FastAPI)
                   ↑ call_soon_threadsafe()
                   │
Pipeline thread: decode → preprocess → infer → track → analytics → render
                   (single thread, all channels sequential)
```

Single pipeline thread processes all channels sequentially within each loop iteration. This is simpler than DeepStream's multi-threaded GStreamer approach and sufficient at 82.6 fps total.

**Thread safety:**
- Phase transitions: `queue.Queue` (thread-safe) drained at loop start.
- Callbacks: `loop.call_soon_threadsafe()` from pipeline thread to asyncio thread.
- `channels` dict: only modified from pipeline thread (after draining transition queue). API thread reads are safe (Python GIL on dict reads).

### 4.5 Idle Optimization

When no detections are found for N consecutive frames, increase the inference skip interval (process every 15th frame instead of every frame). Same logic as DeepStream's `nvinfer` interval property, but implemented in the pipeline loop.

```python
# In pipeline loop, after tracking:
if len(tracks) == 0:
    state.idle_frames += 1
    if state.idle_frames > IDLE_THRESHOLD:
        state.inference_interval = 15  # skip 14 of every 15 frames
else:
    state.idle_frames = 0
    state.inference_interval = 1
```

### 4.6 Best-Photo Integration

Best-photo scoring needs the **clean frame** (no bboxes drawn). The pipeline has the NV12 CUDA surface available before rendering. For best-photo crops:

1. Track scorer identifies a track that needs a crop (higher score than current best).
2. Download the NV12 frame to CPU (or just the crop region via CuPy slicing on GPU → download small region).
3. Convert to BGR, crop bbox region, encode JPEG.

This only fires when a better photo is found — not every frame. Typical frequency: a few times per track lifetime.

---

## 5. Integration with Shared Modules

The custom pipeline reuses all shared modules from `backend/pipeline/`:

| Module | How it's called |
|--------|----------------|
| `TrackingReporter` | Created per-channel on Setup→Analytics. Called with tracks + frame data each inference frame. Owns direction FSM, ROI filter, alert generation, per-frame data storage. |
| `BestPhotoTracker` | Created per-channel on Setup→Analytics. Called with tracks + frame to evaluate best-photo scores. |
| `direction.py` | Called by TrackingReporter. Line-crossing + state machine. |
| `roi.py` | Called by TrackingReporter. Point-in-polygon filter on centroids. |
| `snapshot.py` | Best-photo scoring (blur, size, centering). Called by BestPhotoTracker. |
| `stitch.py` | Track stitching for ID switch recovery. Called by TrackingReporter. |
| `idle.py` | Idle mode detection. Called by pipeline loop. |
| `trajectory.py` | Per-track centroid ring buffer. Managed by TrackingReporter. |
| `alerts.py` | Alert generation. Called by TrackingReporter. |
| `clip_extractor.py` | Post-analytics clip extraction for replay. Called on Analytics→Review. |
| `source_resolver.py` | YouTube URL resolution. Called before `add_channel`. |
| `stream_recovery.py` | Circuit breaker for stream reconnection. Used for YouTube Live. |

**Key point:** The custom pipeline adapter calls `TrackingReporter` with the same inputs as the DeepStream adapter does — a list of tracks with (track_id, bbox, confidence, class_name, centroid) plus frame metadata. The reporter doesn't know which pipeline produced the data.

---

## 6. Differences from DeepStream Pipeline

| Aspect | DeepStream | Custom |
|--------|-----------|--------|
| Decode | nvurisrcbin (GStreamer, GPU NVDEC) | PyNvVideoCodec (direct NVDEC) |
| Inference | nvinfer (GStreamer element) | Direct TensorRT Python API |
| Tracking | nvtracker (NvDCF, GPU) | BoT-SORT on CPU (standalone) |
| Detection model | TrafficCamNet (NVIDIA pretrained) | YOLOv8s TensorRT FP16 |
| OSD / Drawing | nvdsosd (GPU) | OpenCV on CPU |
| Multi-channel | nvstreammux batches natively | Sequential decode + sequential infer |
| Phase routing | GStreamer pad probe on source_id | if/elif in Python loop |
| MJPEG | BufferOperator probe + cupy→cv2 | GPU→CPU download + cv2.imencode |
| Threading | GStreamer manages threads | Single pipeline thread |
| Pipeline rebuild on channel change | Yes (M5 shared pipeline rebuild) | No — just add/remove decoder + tracker |
| Setup→Analytics transition | Remove+re-add nvurisrcbin to mux | Destroy+recreate NvDecoder |
| Container size | 37 GB (DeepStream stack) | ~15 GB target (TRT + CuPy + PyNvVideoCodec) |

---

## 7. Implementation Phases

### P1 — Single-channel adapter

**Goal:** `CustomPipeline` implements `PipelineBackend`. Single file source working end-to-end: decode → detect → track → ROI → direction → alerts.

**Deliverables:**
- `decoder.py` — NvDecoder with loop support
- `preprocess.py` — GPU NV12→RGB + letterbox resize + normalize
- `detector.py` — TRTDetector (load engine, infer, NMS)
- `tracker.py` — TrackerWrapper (BoT-SORT + ID mapping)
- `adapter.py` — CustomPipeline with pipeline loop, phase transitions, TrackingReporter integration
- `engine_builder.py` — Build TRT engine from ONNX
- All FakeBackend tests pass
- Integration test on 741_73_1min.mp4: correct transit alerts generated

**Not in P1:** MJPEG streaming, multi-channel, YouTube Live.

### P2 — MJPEG + frame callbacks

**Goal:** Annotated video visible in browser.

**Deliverables:**
- `renderer.py` — FrameRenderer (NV12→BGR, draw, JPEG encode)
- Frame callback wiring → FastAPI MJPEG endpoint
- Stats broadcasting (fps, tracks, inference_ms)
- ROI/line overlay on frames

### P3 — Multi-channel + batching

**Goal:** Two channels running simultaneously.

**Deliverables:**
- Multi-channel decode (per-channel NvDecoder instances)
- Sequential inference (same TRT context, sequential `execute_async_v3`)
- Per-channel tracker/reporter/photo instances
- Dynamic add/remove channels
- Phase transitions per channel independently

### P4 — YouTube Live + stream recovery

**Goal:** YouTube Live streams work.

**Deliverables:**
- HLS stream support (PyNvVideoCodec or OpenCV fallback)
- Source resolver integration
- Stream recovery with circuit breaker
- Frozen-frame buffer for Phase 3 replay

### P5 — Integration + validation

**Goal:** Production-ready, validated.

**Deliverables:**
- Backend selection via `PIPELINE_BACKEND` env var
- Dockerfile updates (pip deps)
- E2E validation on all 3 test clips + YouTube Live
- Performance benchmarks (fps, latency, memory)
- Container size measurement

---

## 8. Risk Mitigations

| Risk | Mitigation | Validated? |
|------|-----------|-----------|
| PyNvVideoCodec can't handle HLS | OpenCV VideoCapture fallback for HLS sources | TBD (P4) |
| TRT version mismatch (pip 10.16 vs system 10.9) | Use pip TRT for both build and load. Don't use trtexec. | Yes (exp12) |
| BoT-SORT API expects Results object | DetectionResults wrapper with .conf/.xyxy/.xywh/.cls | Yes (exp12) |
| CuPy preprocess differs from Ultralytics | Letterbox + /255.0 matches Ultralytics convention. Verified by detection quality in exp12 (30 IDs vs exp11's 48 — different frames, different count is expected) | Partially |
| NV12→RGB color accuracy | BT.601 conversion kernel validated visually in exp13 | Yes |
| GPU memory with NVDEC + TRT | 210 MB for dual channel, well within 6 GB | Yes (exp12) |

---

## 9. Performance Budget (Dual Channel)

Per loop iteration (both channels processed):

| Stage | Time (ms) | Source |
|-------|-----------|--------|
| Decode 2 channels | 1.12 | exp12 |
| Preprocess 2 channels | 2.03 | exp12 |
| Infer 2 channels (sequential) | 4.85 | exp12 |
| NMS 2 channels | 2.04 | exp12 |
| Track 2 channels | 1.56 | exp12 |
| **Subtotal (inference path)** | **11.60** | |
| MJPEG render 1 frame (every other iter) | ~3.25* | exp13 |
| **Total avg** | **~12.1** | exp12 measured |
| **FPS** | **82.6 total / 41.3 per channel** | |

*MJPEG render: 1.3ms download + 1.1ms draw + 4.1ms encode = 6.5ms, but only every other frame = ~3.25ms amortized. Runs on same thread but only 15fps.

**Headroom:** 41.3 fps per channel vs 30 fps target = 1.38x headroom. Sufficient for production. Analytics post-processing (ROI, direction, alerts) adds <1ms per frame and runs on CPU — negligible impact.
