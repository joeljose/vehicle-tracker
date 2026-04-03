# Vehicle Tracker System — Design Document

## 1. Project Overview

A GPU-accelerated traffic junction monitoring system that detects vehicles, tracks them through the junction, determines their directional transit (entry arm to exit arm), and generates alerts with best-quality vehicle photos. The system supports both recorded video files and YouTube Live streams from traffic cameras at real traffic junctions.

**Target junctions:** 741 & 73, 741 & Lytle South, Drugmart & 73.

**Core deliverable:** Per-vehicle directional transit alerts — "a car entered from 741-North and exited on 73-East" — accompanied by the best-quality photo captured during its transit, delivered to a browser-based UI in real time.

The system provides two independent pipeline implementations (DeepStream and custom) behind a single API boundary, a four-phase per-channel workflow (Setup, Analytics, Review, Teardown), and a React frontend with alert feed, ROI/line drawing tools, and video replay.

---

## 2. Hardware Target

| Property | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Max-Q (Mobile) |
| Architecture | Ada Lovelace (AD107M) |
| VRAM | 6 GB GDDR6 |
| NVDEC gen | 5 (AV1, H.264, H.265 hardware decode) |
| NVENC | Yes (hardware H.264/H.265 encode) |
| TensorRT | 10.x compatible |

---

## 3. Architecture Overview

Two pipeline implementations exist — DeepStream (primary) and custom (secondary). They are built separately, never mixed at runtime. Both expose the identical FastAPI API boundary so the frontend and all downstream consumers are pipeline-agnostic.

**Architecture principle:** Single process, multi-channel. One persistent GStreamer pipeline batches all active sources through shared `nvinfer` → `nvtracker`, then `nvstreamdemux` splits back to per-channel branches for OSD/MJPEG extraction. Experimentally validated savings: ~280 MB VRAM for 2 channels (+459 MB shared vs +738 MB separate).

```
                       +-----------------------+
                       |     React Frontend    |
                       |  (pipeline-agnostic)  |
                       +-----------+-----------+
                                   |
                        REST / WS / MJPEG / HTTP
                                   |
                       +-----------+-----------+
                       |   FastAPI (localhost)  |
                       |  Unified API boundary  |
                       +-----------+-----------+
                                   |
                     +-------------+-------------+
                     |                           |
          +----------+----------+     +----------+----------+
          | DeepStream Pipeline |     |  Custom Pipeline    |
          |    (primary, M1)    |     |   (secondary, M9)   |
          +---------------------+     +---------------------+
```

### 3.1 DeepStream Pipeline (M5 Shared Architecture)

```
src_0 (nvurisrcbin, file-loop=varies) ──┐
                                        ├→ nvstreammux → nvinfer → nvtracker ──→ nvstreamdemux
src_1 (nvurisrcbin, file-loop=varies) ──┘   (shared)     (shared)    (shared)         │
                                                            │                         │
                                               [BatchMetadataRouter probe]            │
                                               routes frame_meta.source_id            │
                                               to per-channel TrackingReporter        │
                                                                                      │
                                        ┌─────────────────────────────────────────────┘
                                        │
                              ┌─────────┴─────────┐
                              ▼                   ▼
                      [ch0 branch]          [ch1 branch]
                      queue → nvdsosd       queue → nvdsosd
                      → nvvideoconvert      → nvvideoconvert
                      → capsfilter(RGB)     → capsfilter(RGB)
                      → fakesink            → fakesink
                      [MjpegExtractor]      [MjpegExtractor]
                      [FrameExtractor]      [FrameExtractor]
```

**Shared part (one instance for all channels):**

- `nvstreammux` batches frames from all active `nvurisrcbin` sources. Sources are added/removed dynamically — `add_channel` links a new source to the mux, `remove_channel` unlinks it.
- `nvinfer` runs TrafficCamNet on the batch — one inference call serves all channels.
- `nvtracker` runs NvDCF, tracking independently per `source_id` (zero track ID overlap between channels, experimentally validated).
- **`BatchMetadataRouter`** probe after tracker iterates `batch_meta.frame_items`, reads `source_id`, dispatches each frame's metadata to the correct channel's `TrackingReporter` for per-channel post-processing (ROI, direction, alerts, best-photo, idle optimization).

**Per-channel part (after `nvstreamdemux`):**

- `nvstreamdemux` splits the batched stream back to per-channel outputs (batch_size=1 each). This is required because pyservicemaker's `BufferOperator` (C++ pybind11) cannot handle batch_size > 1.
- Each channel branch has its own `nvdsosd` → `nvvideoconvert` → `capsfilter(RGB)` → `fakesink` chain.
- `MjpegExtractor` (BufferOperator probe) produces annotated JPEG frames at ~15fps per channel.
- `FrameExtractor` (BufferOperator probe) extracts clean RGB frames for best-photo snapshots.
- These are the exact same probe classes from M4, reused unchanged.

**Source lifecycle:**

| Event | Action |
|-------|--------|
| `add_channel(id, uri)` | Create `nvurisrcbin(file-loop=True)`, link to mux. Create demux output branch. Phase = Setup. |
| Setup → Analytics | Remove looping source from mux. Add same URI with `file-loop=False`. Reinitialize Reporter with ROI/lines/callbacks. |
| Analytics EOS | Per-source EOS detected (mux handles this). Remove source from mux. Finalize tracks. Phase = Review. |
| `remove_channel(id)` | Remove source from mux. Tear down demux branch. Clean up state. |

**Experimental validation (exp07-08):** 466 batches/10s, ~10fps MJPEG per channel, both channels flowing continuously. VRAM: +459 MB for 2 channels shared.

### 3.2 Custom Pipeline

```
Sources ---> NVDEC ---> batch ---> TensorRT ---> ByteTrack ---> [if/elif per channel phase] ---> annotator ---> output
```

- NVDEC hardware decode per source (one decode session per channel).
- Frames batched and fed to TensorRT YOLOv8s INT8.
- ByteTrack runs per-channel on CPU.
- Phase routing is a simple `if/elif` block after inference, before post-processing.
- CUDA annotation kernel draws overlays. Output encoded to MJPEG.

### 3.3 Pipeline Interface Contract

Both pipeline implementations (DeepStream and custom) must implement the same Python Protocol. The FastAPI layer imports and calls this interface — it never imports DeepStream or custom pipeline modules directly. Swapping backends is a single config change.

```python
from typing import Protocol, Callable
from dataclasses import dataclass
from enum import Enum

class ChannelPhase(Enum):
    SETUP = "setup"
    ANALYTICS = "analytics"
    REVIEW = "review"

@dataclass
class Detection:
    track_id: int
    class_name: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    centroid: tuple[int, int]

@dataclass
class FrameResult:
    channel_id: int
    frame_number: int
    timestamp_ms: int
    detections: list[Detection]
    annotated_jpeg: bytes  # MJPEG frame with overlays baked in

class PipelineBackend(Protocol):
    """Interface that both DeepStream and custom pipelines must implement."""

    def start(self) -> None:
        """Boot the inference pipeline (load model, allocate GPU resources)."""
        ...

    def stop(self) -> None:
        """Stop pipeline, release all GPU resources cleanly."""
        ...

    def add_channel(self, channel_id: int, source: str) -> None:
        """Add a video source (file path or resolved HLS URL) to the pipeline."""
        ...

    def remove_channel(self, channel_id: int) -> None:
        """Remove a channel, release its decoder and tracker resources."""
        ...

    def configure_channel(
        self, channel_id: int,
        roi_polygon: list[tuple[float, float]],
        entry_exit_lines: dict,
    ) -> None:
        """Set ROI and entry/exit lines for a channel. Called before analytics start."""
        ...

    def set_channel_phase(self, channel_id: int, phase: ChannelPhase) -> None:
        """Transition a channel's phase. Queued and applied between frames."""
        ...

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update detection confidence threshold at runtime."""
        ...

    def set_inference_interval(self, interval: int) -> None:
        """Set nvinfer interval (0=every frame, 15=idle mode)."""
        ...

    def register_frame_callback(self, callback: Callable[[FrameResult], None]) -> None:
        """Register callback invoked per-channel per-frame with detection results.
        The FastAPI layer uses this to feed WebSocket broadcaster and MJPEG endpoints."""
        ...

    def register_alert_callback(self, callback: Callable[[dict], None]) -> None:
        """Register callback invoked when a transit or stagnant alert is generated.
        The dict matches the WebSocket alert schema."""
        ...

    def register_track_ended_callback(self, callback: Callable[[dict], None]) -> None:
        """Register callback invoked when a track ends.
        The dict matches the track_ended WebSocket schema."""
        ...

    def register_phase_callback(
        self, callback: Callable[[int, ChannelPhase, ChannelPhase], None]
    ) -> None:
        """Register callback invoked when a channel auto-transitions phase.
        Called with (channel_id, new_phase, previous_phase). Used by the API
        layer to push WS notifications and trigger clip extraction on EOS."""
        ...

    def get_snapshot(self, track_id: int) -> bytes | None:
        """Return the best-photo JPEG crop for a given track, or None."""
        ...

    def get_alert(self, alert_id: str) -> dict | None:
        """Return full alert metadata including per-frame data, or None."""
        ...

    def get_alerts(self, limit: int = 50, alert_type: str | None = None) -> list[dict]:
        """Return recent alerts (summary format, paginated)."""
        ...

    def export_video(self, channel_id: int) -> str:
        """Render annotated video with stored overlays. Returns output file path."""
        ...
```

**FastAPI wiring:**
```python
# main.py
from pipeline.deepstream.pipeline import DeepStreamPipeline
# from pipeline.custom.pipeline import CustomPipeline  # swap in for custom backend

def create_app(backend: str = "deepstream") -> FastAPI:
    if backend == "deepstream":
        pipeline = DeepStreamPipeline()
    else:
        pipeline = CustomPipeline()

    app = FastAPI()
    # All routes use `pipeline` via the Protocol interface
    # No DeepStream-specific or custom-specific imports in api/
    setup_routes(app, pipeline)
    setup_websocket(app, pipeline)
    setup_mjpeg(app, pipeline)
    return app
```

**Key design rules:**
- The `api/` package NEVER imports from `pipeline/deepstream/` or `pipeline/custom/` directly — only through the `PipelineBackend` Protocol.
- Shared modules (`direction.py`, `snapshot.py`, `roi.py`, `alerts.py`, `phase.py`, `idle.py`, `stitch.py`) are used by BOTH pipeline implementations. They live in `pipeline/` (not inside `deepstream/` or `custom/`).
- The backend selection is a startup config (`--backend deepstream` or `--backend custom`), not a runtime switch.

---

## 3.4 Source Resolver — YouTube Live URL Resolution

The system accepts two source types: local file paths and YouTube Live URLs. File paths are passed directly to the pipeline. YouTube URLs require resolution to an HLS stream URL before the pipeline can consume them.

**Resolution flow:**
```
Operator pastes YouTube URL in UI
    → POST /channel/add { "source": "https://youtube.com/watch?v=..." }
    → Backend detects YouTube URL pattern
    → Returns 202 Accepted immediately (channel not yet created)
    → Background task: yt-dlp extracts best HLS URL
    → Success: channel created, WS notification sent to frontend
    → Failure: WS error notification with reason, no channel created
    → Resolved HLS URL passed to pipeline (nvurisrcbin handles HLS natively)
```

**Async resolution:** YouTube URL resolution via `yt-dlp` takes 10-30 seconds (deno JS extraction). To avoid blocking the API, resolution runs as a background task. The channel is only created after successful resolution — there is no "resolving" state in the channel lifecycle. If resolution fails (invalid URL, private stream, geo-blocked, stream offline), a WebSocket error event is sent and the operator can retry with a different URL.

**Source resolver module:**
```python
# backend/pipeline/source_resolver.py

@dataclass
class ResolvedSource:
    """Result of resolving a user-provided source string."""
    source_type: Literal["file", "youtube_live"]
    stream_url: str                    # file path or resolved HLS URL
    original_url: str                  # what the user provided

def resolve_source(source: str) -> ResolvedSource:
    """Resolve a user-provided source to a pipeline-consumable URL.

    For file paths: validates the file exists, returns as-is.
    For YouTube URLs: extracts best-quality HLS URL via yt-dlp.
    Always resolves to best available quality — no quality selection.
    """
    ...

def check_stream_liveness(original_url: str) -> bool:
    """Check if a YouTube Live stream is still broadcasting.
    Used during stream recovery to distinguish 'stream ended' from 'network issue'."""
    ...

def re_extract_url(original_url: str) -> str:
    """Re-extract a fresh HLS URL from a YouTube Live URL.
    Used during stream recovery when the existing HLS URL has expired."""
    ...
```

**yt-dlp call serialization:** All `yt-dlp` subprocess calls (initial resolution and recovery) are serialized through a single `asyncio.Lock`. Only one `yt-dlp` process runs at a time. This prevents concurrent calls from triggering YouTube rate limiting, which would make recovery worse. With a 2-3 channel target, contention is rare; when it occurs (e.g., network blip causing simultaneous recovery), queuing adds a few seconds — acceptable for background operations.

**Stream recovery strategy** (implemented in the source manager, invoked when the pipeline detects a stream interruption):

1. **Retry existing HLS URL** — 3 attempts with backoff (5s, 10s, 20s). Most interruptions are transient network issues; the HLS URL is still valid.
2. **Check liveness** — If retries fail, call `check_stream_liveness()` via `yt-dlp --dump-json`.
   - **Stream ended:** Transition channel to Phase 3 automatically.
   - **Stream still live:** The HLS URL may have expired. Proceed to re-extraction.
3. **Validate and reconnect** — Call `re_extract_url()` to get a fresh HLS URL. Validate the new URL (HEAD request or equivalent) before triggering a shared pipeline rebuild. This avoids tearing down other channels' streams for a URL that will immediately fail again.
4. **Give up** — If re-extraction or validation fails, surface the error to the operator via a `pipeline_event` WebSocket message. Operator can manually retry or remove the channel.

**Circuit breaker:** If the same channel triggers 3 pipeline rebuilds within 10 minutes, it is automatically ejected (transitioned to error state and removed from the shared pipeline). The operator is notified via WebSocket. This prevents one flaky stream from repeatedly disrupting all channels in the shared pipeline.

**Stream interruption detection:** Two mechanisms:
- **GStreamer error messages** from `nvurisrcbin` on HLS failure (404, timeout, connection reset).
- **Frame-gap watchdog** (from M5) as a fallback — if no new frames arrive for a channel within N seconds, treat it as a stream interruption. This catches silent freezes where `nvurisrcbin` does not emit an error.

**Note:** The exact GStreamer messages emitted by `nvurisrcbin` on HLS failure must be validated experimentally before implementation (see validation spike below).

**Last frame buffer for Phase 3 replay:** The MJPEG encoder already produces a JPEG for every frame. A single `last_frame` reference per channel is kept in memory (overwritten each frame, near-zero cost). At Phase 2→3 transition — whether from clean stream end or stream death — this buffer is persisted to disk as the frozen review frame for canvas-based bbox replay.

**yt-dlp invocation for URL resolution:**
```python
# Extract best HLS URL
result = subprocess.run(
    ["yt-dlp", "-f", "bv*+ba/b", "--get-url", youtube_url],
    capture_output=True, text=True, timeout=30
)
hls_url = result.stdout.strip()

# Check liveness (used during recovery)
result = subprocess.run(
    ["yt-dlp", "--dump-json", "--no-download", youtube_url],
    capture_output=True, text=True, timeout=30
)
info = json.loads(result.stdout)
# info["is_live"] → True if stream is still broadcasting
```

**Critical yt-dlp flags** (learned from production use in youtube-downloader project):
- `--no-live-from-start` — Do NOT use `--live-from-start`. Old HLS fragments get purged by YouTube causing 404/500 errors.
- `--hls-use-mpegts` — Required for HLS stream compatibility.
- `--wait-for-video 5` — Robustness for streams that haven't fully started.
- Format selection: `bv*+ba/b` for best quality (always).

**Note:** These flags apply when yt-dlp is used as a downloader. For URL resolution (`--dump-json`, `--get-url`), only the format filter (`-f`) is needed. The HLS URL returned by `--get-url` is fed directly to DeepStream's `nvurisrcbin`, which handles HLS consumption natively.

**Dependencies:**
- `yt-dlp` precompiled binary downloaded in the Dockerfile (latest release from GitHub, same approach as youtube-downloader project).
- `deno` precompiled binary downloaded in the Dockerfile — **required by yt-dlp for YouTube JavaScript extraction**. Without deno, many streams fail or return limited formats. Made available on PATH so yt-dlp finds it automatically.
- `ffmpeg` available in the container (already present in the DeepStream base image).
- Called as a subprocess (`subprocess.run(["yt-dlp", ...])`) — not as a Python library — for isolation and easy version updates.

**Key constraints:**
- YouTube HLS URLs expire after ~6 hours. Sessions are expected to be ~1 hour max, so expiry is unlikely but handled via re-extraction.
- `yt-dlp` requires periodic updates as YouTube changes its API. Use latest release in Dockerfile; rebuild image when streams start failing.
- `deno` is similarly updated — needed for yt-dlp's YouTube extractor to run JavaScript from YouTube's player.

**Validation spike (must complete before implementation):**
1. Can `nvurisrcbin` consume a YouTube HLS URL resolved by `yt-dlp`?
2. What GStreamer bus messages does `nvurisrcbin` emit on HLS stream failure (404, timeout, silent freeze)?
3. Does the M5 frame-gap watchdog reliably detect HLS stream drops as a fallback?

---

## 4. Phase Routing Design

**Principle: "Always infer, conditionally analyze."**

Detection and tracking run continuously on every active channel regardless of phase. This ensures stable bounding boxes with IDs are visible during Phase 1 (Setup), giving the operator visual feedback while drawing ROI and lines. Post-processing (alerts, best-photo capture, direction detection) only runs for channels in Phase 2 (Analytics).

### 4.1 Phase State Dict

```python
# Shared state, one entry per active channel
channel_phases: dict[int, ChannelPhase] = {
    0: ChannelPhase.SETUP,
    1: ChannelPhase.ANALYTICS,
}
```

### 4.2 DeepStream Phase Routing (Probe Function)

**Note:** DeepStream 8.0 uses `pyservicemaker` (not the legacy `pyds` API). The `pyservicemaker` module provides `Pipeline`, `Flow`, `Probe`, `BatchMetadataOperator`, `FrameMetadata`, `ObjectMetadata`, etc. The probe pattern below uses the DS 8.0 API — exact method signatures may differ from `pyds` examples found in older DeepStream documentation.

```python
from pyservicemaker import Probe
from pyservicemaker._pydeepstream import BatchMetadata, FrameMetadata, ObjectMetadata

def analytics_probe(batch_meta: BatchMetadata):
    """Probe callback invoked per batch. DS 8.0 pyservicemaker API."""
    for frame_meta in batch_meta.frame_metadata_list:
        source_id = frame_meta.source_id
        phase = channel_phases.get(source_id, ChannelPhase.SETUP)

        if phase == ChannelPhase.ANALYTICS:
            process_direction(frame_meta)
            process_best_photo(frame_meta)
            process_alerts(frame_meta)
        # Phase 1 (SETUP): boxes rendered by OSD, no post-processing
        # Phase 3 (REVIEW): source already removed from pipeline
```

### 4.3 Custom Pipeline Phase Routing

```python
for channel_id, detections in batch_results.items():
    tracks = trackers[channel_id].update(detections)
    phase = channel_phases.get(channel_id, ChannelPhase.SETUP)

    if phase == ChannelPhase.ANALYTICS:
        direction_engine.process(channel_id, tracks)
        snapshot_engine.process(channel_id, tracks, frame)
        alert_engine.process(channel_id, tracks)
    # Phase 1: tracks rendered, no analytics
```

### 4.4 Phase Transition Mechanics

- Phase transitions are requested from the API thread (e.g., `POST /channel/{id}/phase`).
- Each channel has two pipeline instances across its lifecycle:
  1. **Preview pipeline (Setup):** decode → infer → track → OSD → MJPEG. No ROI, no alerts. Video loops. sync=True.
  2. **Analytics pipeline (Analytics):** same + ROI filtering + direction detection + alerts + best-photo. Video plays once. sync=True.
- `pipeline.start()` is non-blocking (pyservicemaker spawns its own GLib MainLoop thread). Probe callbacks fire on the GStreamer streaming thread and bridge to the FastAPI asyncio event loop via `asyncio.loop.call_soon_threadsafe()`.
- **Phase 1 (Setup):** Preview pipeline starts at `add_channel()`. Operator sees live detections at real-time pace while drawing ROI/lines.
- **Phase 1 to 2 transition:** Preview pipeline stopped. The frontend sends the operator's currently drawn ROI polygon and entry/exit lines with the phase transition request. The backend calls `configure_channel()` with these coordinates, then `set_channel_phase()`. Analytics pipeline starts with ROI/lines, fresh tracker, playback from frame 0 (no loop).
- **Phase 2 to 3 transition:** For recorded video, triggered automatically at EOS via pipeline `on_message` callback. For YouTube Live, triggered by stream end/drop (after recovery attempts fail) or operator click "Stop". Analytics pipeline stopped.
- **Phase 3 to 4 transition:** Removes the channel entirely. All in-memory data (alerts, snapshots, replay data) is cleared.

### 4.5 Threading Architecture

```
Main Thread:  asyncio event loop (uvicorn/FastAPI)
                 ↑ call_soon_threadsafe()
                 |
GStreamer Thread 1:  pipeline.start() → channel 0 probes
GStreamer Thread 2:  pipeline.start() → channel 1 probes
```

- GStreamer probe callbacks (BatchMetadataOperator, BufferOperator) run on the streaming thread.
- Callbacks push FrameResult / alerts to the asyncio thread via `call_soon_threadsafe()`.
- AlertStore requires `threading.Lock` for cross-thread writes.
- MJPEG frames: BufferOperator after nvdsosd → cupy.asnumpy() → cv2.imencode('.jpg') at ~15fps (skip every other frame).
- **Phase 3 to 4 transition:** Removes the channel entirely. All in-memory data (alerts, snapshots, replay data) is cleared.

```python
class PhaseTransition:
    """Queued from API thread, consumed by pipeline thread."""
    channel_id: int
    target_phase: ChannelPhase
    timestamp: float

# Pipeline thread loop
def process_frame_batch():
    # Drain transition queue between batches
    with phase_lock:
        while not transition_queue.empty():
            t = transition_queue.get()
            apply_transition(t)

    # Process next batch...
```

---

## 5. Application Workflow

### Phase 1 — Setup

| Aspect | Detail |
|---|---|
| Trigger | Operator adds a video source (file path or YouTube Live URL) |
| Video behavior | Plays immediately; recorded files loop continuously |
| Inference | Detection + tracking active (preview overlay with boxes and IDs) |
| Operator actions | Draw ROI polygon, draw entry/exit lines, label arms, adjust thresholds |
| Video transport | MJPEG over HTTP |
| Alerts | None generated |
| Purpose | Visual feedback for configuration — operator sees tracked vehicles while setting up ROI and lines |

### Phase 2 — Analytics

| Aspect | Detail |
|---|---|
| Trigger | Operator clicks "Start Analytics" |
| Video behavior | Recorded: resets to frame 0, plays once (no loop). YouTube Live: starts from current time |
| Inference | Full pipeline: detection + tracking + direction + alerts + best-photo |
| Tracker | Fresh instance created (all prior track IDs cleared) |
| Alerts | Transit alerts on line crossing or track exit. Stagnant alerts after 2.5 minutes stopped |
| Video transport | MJPEG over HTTP with full overlays (boxes, IDs, trajectories, direction indicators) |
| Data collection | Per-frame bbox+centroid stored per track for replay |

### Phase 3 — Review

| Aspect | Detail |
|---|---|
| Trigger | Video end (recorded), YouTube Live stream end/drop (after recovery fails), or operator clicks "Stop" |
| Phase transition | Background clip extraction begins (transit alerts only, NVENC). Review UI opens immediately; clips become available progressively. |
| Alert feed | Persists — all alerts from Phase 2 remain visible and clickable |
| Replay (transit, recorded) | Click alert to play a short looping clip (entry to exit + padding) with bbox overlay on canvas. Clip pre-extracted at phase transition. Spinner shown if clip not yet ready. |
| Replay (transit, YouTube Live) | Frozen last frame displayed as `<img>`. Stored bboxes animated on `<canvas>` overlay along stored trajectory at original timing |
| Replay (stagnant, recorded) | Single full frame extracted at best-photo timestamp with bbox drawn. Static image, no video. |
| Replay (stagnant, YouTube Live) | Bbox drawn on frozen last frame at vehicle's position. Static image. |
| Export | Option to render annotated video with fading trajectory overlay (recorded only, uses NVENC) |

### Phase 4 — Teardown

| Aspect | Detail |
|---|---|
| Trigger | Operator clicks "Remove Stream" |
| Action | Channel removed from pipeline. All in-memory data (alerts, snapshots, replay data, tracker state) cleared |

---

## 6. Direction Detection Architecture

### 6.1 Entry/Exit Lines

Each junction arm has one entry/exit line defined as a line segment with two endpoints and an editable label. Default labels follow the N/E/S/W convention (top=North, clockwise), but operators can rename them to road names (e.g., "741-North", "73-East").

Lines are drawn by the operator in Phase 1 on the canvas overlay. A vehicle's centroid crossing a line triggers direction state transitions.

### 6.2 Line-Crossing Algorithm

Line crossing is detected by computing the sign of the cross product of consecutive centroids relative to the line segment.

```
Given line segment AB and consecutive centroids P1, P2:

  cross1 = (B - A) x (P1 - A)    # 2D cross product (scalar)
  cross2 = (B - A) x (P2 - A)

  If sign(cross1) != sign(cross2), the centroid crossed the line between these frames.
```

The direction of crossing (into or out of the junction) is determined by the sign of the cross product. The sign's meaning depends on which side of the line is "inside" the junction.

**User-specified junction side:** When the operator draws an entry/exit line during Setup, a junction-side indicator (small triangle) appears on one side of the line, showing which side is considered "inside the junction." The operator can flip this with one click. The convention:

- **Left side** of the start→end vector = positive cross product
- **Right side** = negative cross product
- `junction_side: "left" | "right"` is stored per line in the site config

This eliminates the need for any auto-calibration delay — alerts start immediately when analytics begins.

```python
def check_line_crossing(line: LineSeg, prev: Point, curr: Point) -> int | None:
    """Returns +1 (neg→pos), -1 (pos→neg), or None."""
    cross_prev = (B - A) x (prev - A)
    cross_curr = (B - A) x (curr - A)
    if sign(cross_prev) != sign(cross_curr):
        return +1 if cross_prev < 0 else -1
    return None

def classify_crossing(line: LineSeg, crossing_direction: int) -> str:
    """Deterministic entry/exit from user-specified junction side."""
    junction_is_positive = (line.junction_side == "left")
    if junction_is_positive:
        return "entry" if crossing_direction == +1 else "exit"
    else:
        return "entry" if crossing_direction == -1 else "exit"
```

### 6.3 Per-Track Direction State Machine

```
                +-------------------+
   first seen   |     UNKNOWN       |
   ------------>| (no line crossed) |
                +--------+----------+
                         |
                         | crosses entry line of arm X
                         v
                +-------------------+
                |   ENTERED(dir)    |
                | entry_arm = X     |
                | entry_label = "X" |
                +--------+----------+
                         |
                         | crosses exit line of arm Y
                         v
                +-------------------+
                |   EXITED(dir)     |
                | exit_arm = Y      |       -----> generate transit_alert
                | exit_label = "Y"  |              (method: "confirmed")
                +-------------------+


    Alternative paths (inferred direction):

    UNKNOWN ---> track lost (no entry line crossed)
                 |
                 v
                 infer entry direction from trajectory heading
                 generate transit_alert (method: "inferred")

    ENTERED(dir) ---> track lost (no exit line crossed)
                      |
                      v
                      infer exit direction from trajectory heading
                      generate transit_alert (method: "inferred")
```

### 6.4 Direction Inference

When a track ends without crossing both lines (common for vehicles that appear after the entry line or disappear before the exit line), direction is inferred by **ray intersection** — a ray is cast from the trajectory endpoint along the heading direction, and the first arm line it intersects determines the arm.

For entry inference: the ray is cast backwards (opposite to direction of travel) from the first few centroids. For exit inference: the ray is cast forwards from the last few centroids.

**Sanitization:** The trajectory is first trimmed for sudden position jumps (>80px between consecutive frames), which indicate tracker ID reassignment during occlusion. For exit inference, the tail heading is cross-checked against the stable mid-trajectory heading (25%-75%) — if they diverge by more than 60°, the stable heading is used instead (gradual tracker drift during occlusion).

**Fallback:** If the heading is too weak (<5px displacement) or the ray doesn't intersect any line, falls back to pure proximity (closest line segment to the anchor point).

```python
def nearest_arm(trajectory, arms, use_start):
    # 1. Trim trajectory jumps (hard ID switches)
    # 2. Compute heading from first/last N centroids
    # 3. For exit: cross-check tail heading against stable heading
    # 4. Cast ray from anchor along heading direction
    for arm_id, line in arms.items():
        t = ray_segment_intersection(
            anchor, ray_direction, line.start, line.end
        )
        # Pick the line with smallest t (first intersection)

    # 5. Fallback: pure proximity if ray misses all lines
    for arm_id, line in arms.items():
        dist = point_to_segment_distance(anchor, line)
        # Pick closest
```

### 6.5 Track ID Mapping

DeepStream assigns very large 64-bit track IDs (e.g., `7531586948696113201`) that exceed JavaScript's `Number.MAX_SAFE_INTEGER` (`2^53 - 1`). To avoid precision loss in JSON serialization, `TrackingReporter` maintains a mapping from DeepStream IDs to sequential integers starting from 1. All externally-visible IDs (alerts, snapshots, WebSocket messages) use the sequential ID. The mapping is stored in `_ds_to_seq` and assigned monotonically via `_seq_id()`.

---

## 7. Vehicle State Machine

Each tracked vehicle transitions through states based on motion and line crossings.

```
                      +------------------+
            detect    |     UNKNOWN      |
            --------->|  (no entry yet)  |
                      +--------+---------+
                               |
                               | crosses entry line
                               v
                      +------------------+
                      |   IN-TRANSIT     |<----+
                      |  (has entry arm) |     | starts moving again
                      +--------+---------+     |
                               |               |
                   +-----------+-----------+   |
                   |                       |   |
           centroid moving          stopped (centroid displacement
                   |                < M pixels over window)
                   |                       |
                   |               +-------v--------+
                   |               |    WAITING     |
                   |               | (< 2.5 min)   |
                   |               +-------+--------+
                   |                       |
                   |               still stopped > 2.5 min (150s)
                   |                       |
                   |               +-------v--------+
                   |               |   STAGNANT     |----> stagnant_alert
                   |               +-------+--------+
                   |                       |
                   |               starts moving again
                   |                       |
                   v                       v
            crosses exit line       crosses exit line
                   |                       |
                   v                       v
            transit_alert           transit_alert
            (confirmed)             (confirmed + was_stagnant=true)
```

**Stagnant detection algorithm:**

```python
def check_stagnant(track: Track, config: SiteConfig) -> bool:
    """
    Check if a track's centroid displacement over the stagnant window
    is below the motion threshold.
    """
    window_frames = int(config.stagnant_threshold_seconds * config.inference_fps)
    if len(track.trajectory) < window_frames:
        return False

    oldest = track.trajectory[-window_frames]
    newest = track.trajectory[-1]
    displacement = math.sqrt(
        (newest[0] - oldest[0])**2 + (newest[1] - oldest[1])**2
    )
    return displacement < config.motion_threshold_pixels
```

---

## 8. Best-Photo Pipeline

**Scoring function:** `score = bbox_area * confidence`

This naturally favors frames where the vehicle is large (close, unoccluded) and detected with high confidence. No Laplacian blur detection or frame buffering — the scoring is cheap enough to run per-frame.

```python
class BestPhotoTracker:
    """Maintains one best crop per active track."""

    def __init__(self):
        self.best: dict[int, BestPhoto] = {}  # track_id -> BestPhoto

    def update(self, track_id: int, bbox: tuple, confidence: float, full_frame: np.ndarray):
        x, y, w, h = bbox
        area = w * h
        score = area * confidence

        current = self.best.get(track_id)
        if current is None or score > current.score:
            # Crop from full-resolution frame (not inference tensor)
            crop = full_frame[y:y+h, x:x+w].copy()
            self.best[track_id] = BestPhoto(
                score=score,
                crop=crop,          # ~50KB as JPEG
                bbox=bbox,
                confidence=confidence,
            )

    def finalize(self, track_id: int) -> BestPhoto | None:
        """Called on track end. Returns the best photo and removes from active dict."""
        return self.best.pop(track_id, None)
```

**Storage:** One best crop per track is held in memory (~50 KB JPEG). On track end, the crop is written to disk and served via `GET /snapshot/{track_id}`. The URL is included in the alert payload.

---

## 9. ROI Design

Two independent geometric layers gate what detections produce alerts.

### Layer 1: ROI Polygon (Noise Filter)

A broad polygon encompassing the junction area of interest. Drawn by the operator in Phase 1.

- **Detection runs on the full frame** — always. The ROI does not restrict where inference happens.
- After detection, each vehicle's centroid is tested against the ROI polygon using the ray casting (point-in-polygon) algorithm.
- Vehicles outside the ROI are still rendered on the video (boxes + IDs) but are **not tracked for alerts**.
- Purpose: filter out distant vehicles, parked cars on shoulders, and pedestrians at the periphery.

```python
def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Ray casting algorithm. O(n) where n = number of polygon vertices."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside
```

### Layer 2: Entry/Exit Lines (Event Detection)

Four line segments (one per junction arm) positioned at the borders of the junction. Only centroids that pass the ROI polygon test are checked for line crossings.

```
  +------------------------------------------+
  |              Full Frame                  |
  |                                          |
  |    +------- ROI Polygon --------+        |
  |    |                            |        |
  |    |   ====== North Line ====== |        |   <-- entry/exit lines at
  |    |   ||                    || |        |       junction boundaries
  |    |   || West    East      || |        |
  |    |   || Line    Line      || |        |
  |    |   ||                    || |        |
  |    |   ====== South Line ====== |        |
  |    |                            |        |
  |    +----------------------------+        |
  |                                          |
  +------------------------------------------+
```

---

## 10. ID Switch Mitigation

### 10.1 DeepStream: NvDCF with ReID

NvDCF (NVIDIA Discriminative Correlation Filter) tracker with visual ReID features enabled. Key config parameters:

```ini
[tracker]
tracker-width=640
tracker-height=384
gpu-id=0
ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
ll-config-file=tracker_config.yml
enable-past-frame=1
enable-batch-process=1
```

```yaml
# tracker_config.yml (key parameters)
NvDCF:
  useUniqueID: 1
  maxTargetsPerStream: 50
  maxShadowTrackingAge: 150   # ~5s at 30fps; keeps "shadow" tracks for stopped/occluded vehicles
  probationAge: 3
  earlyTerminationAge: 1
  minIouDiff4NewTarget: 0.7
  featureImgSizeLevel: 3      # ReID feature extraction resolution
  useColorNames: 1
  useHog: 1
```

`maxShadowTrackingAge=60-90` frames (2-3 seconds at 30fps) maintains a prediction of occluded vehicles' positions. When a vehicle reappears near the predicted location, the original track ID is preserved rather than creating a new ID.

### 10.2 Custom Pipeline: ByteTrack

ByteTrack with extended `track_buffer` for the same purpose:

```python
tracker = BYTETracker(
    track_thresh=0.5,
    track_buffer=90,      # frames to keep lost tracks (3s at 30fps)
    match_thresh=0.8,
    frame_rate=30,
)
```

### 10.3 Proximity-Based Track Stitching (Both Pipelines)

A safety net for unrecoverable ID switches. If a new track appears within 100 pixels and 2 seconds of a lost track's predicted position, the tracks are merged.

```python
def stitch_tracks(new_track: Track, lost_tracks: list[Track], max_dist: float = 100, max_gap_s: float = 2.0):
    """
    Check if new_track is likely a continuation of a recently lost track.
    If so, merge: preserve the lost track's entry direction and trajectory history.
    """
    for lost in lost_tracks:
        time_gap = new_track.first_seen_time - lost.last_seen_time
        if time_gap > max_gap_s:
            continue

        # Predict where the lost track would be now using its last velocity
        predicted_pos = lost.predict_position(time_gap)
        dist = euclidean_distance(new_track.centroid, predicted_pos)

        if dist < max_dist:
            # Merge: new track inherits lost track's direction state
            new_track.entry_arm = lost.entry_arm
            new_track.entry_label = lost.entry_label
            new_track.direction_state = lost.direction_state
            new_track.trajectory = lost.trajectory + new_track.trajectory
            # Note: if inherited state is UNKNOWN (lost track never crossed
            # an entry line) and the stitched track later crosses an exit
            # line, apply nearest_arm() on the combined trajectory's start
            # proximity to infer entry direction (method="inferred").
            return True
    return False
```

### 10.4 Direction Inference as Ultimate Fallback

When an ID switch is unrecoverable (no proximity match), the direction inference algorithm (Section 6.4) uses the partial trajectory to estimate the vehicle's entry or exit arm. This produces an "inferred" alert rather than a "confirmed" one.

---

## 11. Idle Optimization

When a scene is empty (e.g., all traffic stopped at a red light off-camera), running full inference every frame wastes GPU cycles.

```
                   +-----------------------+
                   |   ACTIVE MODE         |
                   |   inference every     |
          +------->|   frame (30fps)       |
          |        +-----------+-----------+
          |                    |
          |        0 detections for 30 consecutive
          |        frames (1 second)
          |                    |
          |                    v
          |        +-----------------------+
          |        |   IDLE MODE           |
          +--------+   inference every     |
       any         |   15th frame (~2fps)  |
       detection   +-----------------------+
```

**Rules:**
- Zero-detection counter increments each frame with 0 detections.
- After 30 consecutive zero-detection frames (1 second at 30fps), switch to idle mode.
- Idle mode: run inference every 15th frame (effectively ~2fps).
- **Any detection** in an idle-mode frame immediately returns to full rate.
- **Never skip inference while active tracks exist** — even if a single frame has 0 detections, if the tracker still has active tracks, stay at full rate.

**Implementation:**

- DeepStream: `pipeline["nvinfer_name"].set({"interval": 15})` for idle, `.set({"interval": 0})` for active (pyservicemaker Node API — dict syntax, not key-value).
- Custom: skip the TensorRT inference call on non-inference frames, pass empty detections to tracker.

---

## 12. Frame Rate Normalization

All inference is normalized to 30fps regardless of source frame rate.

| Source FPS | Decode FPS | Inference FPS | Method |
|---|---|---|---|
| 30 | 30 | 30 | Every frame |
| 60 | 60 | 30 | Skip every other frame for inference |
| 25 | 25 | 25 | Every frame (close enough to 30) |
| 15 | 15 | 15 | Every frame (low-fps source) |

- **Decode stays at native FPS** for smooth video display.
- 60fps sources: DeepStream uses `interval=1` on `nvinfer` (process every 2nd frame). Custom pipeline skips every other frame before TensorRT call.
- **All time-based configurations are in seconds**, not frame counts. Conversion to frame counts uses the source's actual FPS at runtime.

```python
# Example: stagnant threshold conversion
stagnant_frames = int(config.stagnant_threshold_seconds * source_fps)
# 150 seconds * 30 fps = 4500 frames
# 150 seconds * 60 fps = 9000 frames (but inference at 30fps = 4500 inference frames)
```

---

## 13. Video Transport Design

### 13.1 Phase 1 & 2 — Live View (MJPEG)

```
Backend pipeline:
  frame -> draw bboxes/IDs/trajectories -> JPEG encode -> multipart/x-mixed-replace

Frontend:
  <img src="/stream/{channel_id}">     <-- MJPEG stream, auto-updates
  +
  <canvas> overlay (transparent)        <-- ROI polygon + line drawing (Phase 1 only)
```

MJPEG is chosen for live view because:
- Zero client-side decode overhead (browser natively renders JPEGs).
- Backend already has the annotated frame in memory — just encode and push.
- Works in any browser without WebRTC/MSE complexity.
- Latency is frame-level (~33ms at 30fps).

The `<canvas>` overlay sits on top of the `<img>` element, pixel-aligned. In Phase 1, the operator draws ROI polygon and entry/exit lines on this canvas. In Phase 2, the canvas is used for any frontend-side overlays (e.g., direction indicators).

### 13.2 Phase 3 Replay — Transit Alert (Recorded Video)

Backend extracts a short clip per transit alert at the Analytics → Review phase transition:

```
Phase transition (background job):
  For each transit alert:
    ffmpeg -ss {first_seen_ts - padding} -to {last_seen_ts + padding} -i {source_file} -c:v h264_nvenc clip_{alert_id}.mp4
  Clips stored in a temp directory, cleared on channel teardown.

Frontend:
  <video src="/alert/{alert_id}/replay" loop autoplay>   <-- short clip, loops
  +
  <canvas> overlay                                        <-- draws stored per-frame bboxes + trajectory
```

Clip extraction runs in the background — Review phase opens immediately. Transit alert cards show a spinner until their clip is ready. Clips are 5-15 seconds each (track duration + padding).

**Synchronization:** The clip starts at time 0 which corresponds to `first_seen_timestamp_ms - padding`. The canvas overlay uses `requestVideoFrameCallback()` with `metadata.mediaTime` matched against `timestamp_ms` in `per_frame_data` via binary search. Since the clip is extracted from the same video file, `mediaTime` maps directly to the file's PTS. The `per_frame_data` stores `timestamp_ms` (video PTS in milliseconds) per frame for this mapping.

**Clip storage estimate:** ~800 transit alerts x 5s avg x 1MB = ~1-2GB typical, ~4GB worst case. Cleared on channel teardown (Phase 4).

### 13.3 Phase 3 Replay — Stagnant Alert (Recorded Video)

```
Backend:
  Extract single frame at best-photo timestamp, draw bbox overlay
  GET /alert/{alert_id}/replay  -->  JPEG image (full frame with bbox)

Frontend:
  <img src="/alert/{alert_id}/replay">   <-- static image, no video
```

A stagnant vehicle sitting still for 2.5 minutes has no meaningful video to replay. The operator sees the full frame with the vehicle highlighted by a bounding box, plus the metadata (duration, position, best photo).

### 13.4 Phase 3 Replay — YouTube Live (No Original Video)

Since there is no recorded file for YouTube Live sources, replay uses stored data only:

**Transit alerts:**
```
Frontend:
  <img src="{last_frame_jpeg}">         <-- frozen last frame from stream
  +
  <canvas> overlay                      <-- animated bbox sliding along stored trajectory
```

The canvas animates the stored per-frame bboxes at original timing using `requestAnimationFrame()`. The operator sees the vehicle's tracked path replayed on top of the last captured frame.

**Stagnant alerts:**
```
Frontend:
  <img src="{last_frame_jpeg}">         <-- frozen last frame from stream
  +
  <canvas> overlay                      <-- bbox drawn at vehicle's position
```

Static bbox drawn on the frozen last frame. No animation needed.

---

## 14. Per-Alert Replay Data

Each alert stores per-frame data for every frame the tracked vehicle was active.

**Per-frame record:**
```python
@dataclass
class FrameRecord:
    frame: int              # frame number (4 bytes)
    timestamp_ms: int       # video PTS in milliseconds (8 bytes) — from GStreamer buffer PTS
    bbox: tuple[int, int, int, int]  # x, y, w, h (16 bytes)
    centroid: tuple[int, int]        # x, y (8 bytes)
    # ~36 bytes per record, ~48 bytes with Python object overhead
```

**Note:** `timestamp_ms` is the video file's PTS (presentation timestamp), NOT wall-clock time. For recorded files, this matches the browser's `mediaTime * 1000` in `requestVideoFrameCallback()`. For YouTube Live, PTS from the HLS stream. This field is essential for replay synchronization — see Section 13.2.

**Memory estimation:**
- Average track duration: ~10 seconds at 30fps = 300 frames
- Per alert: 300 frames x 40 bytes = ~12 KB
- 45-minute session at a busy junction: ~1,350 transit alerts
- Total: 1,350 x 12 KB = ~16 MB raw frame data
- Including alert metadata + snapshot references: ~85 MB total

**Delivery:** Per-frame data is NOT sent via WebSocket push. The lightweight WS alert contains only summary fields. Full per-frame data is fetched on demand via `GET /alert/{alert_id}` when the operator clicks to expand an alert or initiate replay.

---

## 15. Annotated Video Export

Available for recorded video sources only, triggered in Phase 3 before teardown.

**Process:**
1. `POST /video/export` triggers the export job.
2. Backend reads the original video file frame by frame.
3. For each frame, looks up all alerts that were active at that frame number.
4. Draws stored bounding boxes + fading trajectory overlay.
5. Encodes the annotated frame using NVENC (hardware H.264 encoder).
6. Writes to output file.

**Limitation:** Only vehicles that generated alerts (transit or stagnant) have stored per-frame data. Vehicles that were tracked but did not complete a transit (e.g., lost mid-junction, outside ROI, or still in-transit when Phase 2 ended) will NOT have overlays in the exported video. This means some visible vehicles may lack bounding boxes in the export. Accepted for v1 — re-running inference during export is expensive and would duplicate the pipeline.

**Fading trajectory trail:**
```python
def draw_fading_trail(frame: np.ndarray, trajectory: list[Point], current_idx: int, trail_length: int = 30):
    """
    Draw the last N centroids with linear opacity decay.
    Newest point: fully opaque. Oldest point: nearly transparent.
    """
    start_idx = max(0, current_idx - trail_length)
    points = trajectory[start_idx:current_idx + 1]

    for i, (x, y) in enumerate(points):
        alpha = (i + 1) / len(points)  # 0.0 (oldest) -> 1.0 (newest)
        color = (0, 255, 0, int(alpha * 255))  # green with fading alpha
        cv2.circle(frame, (x, y), 3, color[:3], -1)  # simplified; real impl uses alpha blending

    # Draw lines connecting points
    for i in range(1, len(points)):
        alpha = (i + 1) / len(points)
        cv2.line(frame, points[i-1], points[i], (0, int(255 * alpha), 0), 2)
```

---

## 16. Alert Feed UI Design

### 16.0 UI Architecture — Tab-Per-Channel

The frontend uses a **tab-per-channel** model:

- **Home page** (`/`): Channel manager. Shows pipeline start/stop button, "Add Channel" form, and text-based status cards per active channel (phase, alert count). No video panels.
- **Channel tab** (`/channel/{id}`): Opens in a new browser tab. Full phase-driven layout for a single channel. Self-contained: own WebSocket connection, own MJPEG stream, own alert feed.

This eliminates multi-channel layout conflicts — each tab is single-channel, single-phase. The operator focuses on one junction at a time and uses the home page for bird's-eye status.

**Home page state sync:** The home page connects to the WebSocket and listens for `phase_changed` and `pipeline_event` messages. On initial load, it fetches `GET /channels` for current state.

**Tab lifecycle:** Closing a channel tab does not stop the channel — it continues running on the backend. Reopening the tab reconnects the WebSocket and backfills alerts via `GET /alerts?channel={id}`.

### 16.1 Channel Tab Layout

```
+------------------------------------------------------------+
|  Control Panel (top bar)                                    |
+--------------------------------------------+---------------+
|                                            |  Alert Feed   |
|          Video Panel                       |  (~300px)     |
|     (single channel, MJPEG or replay)      |               |
|                                            | [Filter: All] |
|                                            |               |
|                                            | +--card-----+ |
|                                            | | thumb | N->S| |
|                                            | | #567  8.5s | |
|                                            | +----------+ |
|                                            |               |
|                                            | +--card-----+ |
|                                            | | thumb |STAG| |
|                                            | | #571  2.5m | |
|                                            | +----------+ |
|                                            |               |
+--------------------------------------------+---------------+
|  Stats Bar (FPS, track count, inference ms)                 |
+------------------------------------------------------------+
```

### 16.2 Compact Card

Each alert is a compact card in the right sidebar:
- **Thumbnail:** Best photo (64x64 crop).
- **Direction indicator:** Arrow or label (e.g., "N -> S" or "741-North -> 741-South").
- **Track ID:** `#567`.
- **Duration:** Time in junction (e.g., "8.5s").
- **Stagnant highlight:** Orange/yellow background for stagnant alerts.
- **Timestamp:** Relative time (e.g., "2m ago").

### 16.3 Expanded View (on Click)

Clicking a card expands it (accordion style) to show:
- Full-size best photo.
- Complete metadata: vehicle class, entry/exit timestamps, duration, average confidence, max confidence, frames tracked, stationary time.
- Trajectory minimap: small junction snapshot with the vehicle's trajectory drawn as a colored line.
- "Download Photo" button.
- "Replay" button: jumps the video panel to the alert's timestamp and replays the segment with stored bbox overlay.

### 16.4 Filters

Three filter buttons above the card list:
- **All** — show all alerts.
- **Transit** — transit alerts only.
- **Stagnant** — stagnant alerts only.

### 16.5 Data Flow

```
Alert generated (backend)
        |
        v
WebSocket push: lightweight summary
{ alert_id, track_id, class, direction, best_photo_url, timestamp }
        |
        v
Frontend appends to in-memory alert array
        |
        v
Virtual scroller (@tanstack/react-virtual) renders only visible cards (~20-30)
        |
        v  (user clicks card in Review phase)
GET /alert/{alert_id}/replay  -->  clip (transit) or frame (stagnant)
        |
        v
Frontend renders replay (clip loop or static frame with bbox)
```

**Alert feed loading:** During Analytics, alerts accumulate client-side from WebSocket. On tab reopen or page refresh, `GET /alerts?channel={id}` returns the full alert array (~200-400KB for 1,000-2,000 alerts). No pagination — all alerts loaded into client memory. Virtual scrolling ensures DOM performance regardless of alert count.

**Auto-scroll:** Feed auto-scrolls to newest alert when the user is at the top of the list. If the user has scrolled down, auto-scrolling stops and a "New alerts" badge appears to jump back to the top.

---

## 17. Site Config Schema

Each junction has a saved configuration file. Loaded when the same site is added again.

```json
{
  "site_id": "741_73",
  "roi_polygon": [[120, 80], [520, 80], [620, 400], [20, 400]],
  "entry_exit_lines": {
    "north": {
      "label": "741-North",
      "start": [200, 90],
      "end": [450, 90]
    },
    "east": {
      "label": "73-East",
      "start": [610, 150],
      "end": [610, 350]
    },
    "south": {
      "label": "741-South",
      "start": [200, 390],
      "end": [450, 390]
    },
    "west": {
      "label": "73-West",
      "start": [30, 150],
      "end": [30, 350]
    }
  },
  "stagnant_threshold_seconds": 150,
  "motion_threshold_pixels": 10,
  "idle_threshold_seconds": 1,
  "inference_fps": 30
}
```

**Persistence:** Saved as JSON to `backend/config/sites/{site_id}.json`. Loaded automatically when a video file matching the site pattern is added, or manually via the UI.

---

## 18. Data Contracts

### 18.1 WebSocket Messages (Push from Backend)

**frame_data** (per frame, per channel, ~30/s):
```json
{
  "type": "frame_data",
  "channel": 0,
  "frame": 1042,
  "timestamp_ms": 34733,
  "tracks": [
    {
      "id": 7,
      "class": "car",
      "bbox": [412, 305, 541, 378],
      "confidence": 0.91,
      "centroid": [476, 341],
      "trajectory": [[470, 338], [473, 340], [476, 341]]
    }
  ]
}
```

**stats_update** (every 1 second, per channel):
```json
{
  "type": "stats_update",
  "channel": 0,
  "fps": 29.4,
  "active_tracks": 11,
  "inference_ms": 8.2,
  "dropped_frames": 0,
  "phase": "analytics",
  "idle_mode": false
}
```

**Implementation:** Emitted from the pipeline frame callback. A time check (`time.time() - last_stats_time >= 1.0`) triggers collection: fps from frame counter / elapsed, active_tracks from tracker dict length, inference_ms from probe timing. No extra thread needed — piggybacks on existing ~30/sec frame callback. ~15 lines of code.

**pipeline_event:**
```json
{
  "type": "pipeline_event",
  "event": "started",
  "detail": "Pipeline started with 2 channels"
}
```

**track_ended:**
```json
{
  "type": "track_ended",
  "channel": 0,
  "track_id": 7,
  "class": "car",
  "state": "completed",
  "entry_direction": "north",
  "entry_label": "741-North",
  "exit_direction": "south",
  "exit_label": "741-South",
  "method": "confirmed",
  "best_photo_url": "/snapshot/7",
  "full_trajectory": [[470, 338], [473, 340], [476, 341]],
  "first_seen_frame": 1001,
  "last_seen_frame": 1042,
  "duration_ms": 1367,
  "frames_stationary": 0
}
```

**transit_alert** (lightweight push):
```json
{
  "type": "transit_alert",
  "channel": 0,
  "alert_id": "a1b2c3",
  "track_id": 567,
  "class": "car",
  "entry_direction": "north",
  "entry_label": "741-North",
  "exit_direction": "south",
  "exit_label": "741-South",
  "method": "confirmed",
  "best_photo_url": "/snapshot/567",
  "duration_ms": 8500,
  "timestamp_ms": 1711612345000
}
```

**stagnant_alert** (lightweight push):
```json
{
  "type": "stagnant_alert",
  "channel": 0,
  "alert_id": "d4e5f6",
  "track_id": 571,
  "class": "truck",
  "best_photo_url": "/snapshot/571",
  "stationary_duration_ms": 150000,
  "position": [450, 320],
  "timestamp_ms": 1711612645000
}
```

**phase_changed** (on phase transition):
```json
{
  "type": "phase_changed",
  "channel": 0,
  "phase": "review",
  "previous_phase": "analytics"
}
```

### 18.2 REST Responses

**GET /channels** (home page initial load):
```json
{
  "channels": [
    {
      "channel_id": 0,
      "source": "741_73.mp4",
      "phase": "analytics",
      "alert_count": 47
    }
  ],
  "pipeline_started": true
}
```

**GET /alert/{alert_id}** (full metadata on demand):
```json
{
  "alert_id": "a1b2c3",
  "type": "transit_alert",
  "channel": 0,
  "track_id": 567,
  "class": "car",
  "entry_direction": "north",
  "entry_label": "741-North",
  "exit_direction": "south",
  "exit_label": "741-South",
  "method": "confirmed",
  "best_photo_url": "/snapshot/567",
  "full_trajectory": [[470, 338], [473, 340], [476, 341]],
  "per_frame_data": [
    {"frame": 1001, "timestamp_ms": 1410, "bbox": [412, 305, 129, 73], "centroid": [476, 341]},
    {"frame": 1002, "timestamp_ms": 1443, "bbox": [414, 306, 130, 74], "centroid": [479, 343]}
  ],
  "first_seen_frame": 1001,
  "last_seen_frame": 1296,
  "entry_timestamp_ms": 1711612336500,
  "exit_timestamp_ms": 1711612345000,
  "duration_ms": 8500,
  "avg_confidence": 0.89,
  "max_confidence": 0.95,
  "frames_tracked": 295,
  "frames_stationary": 0
}
```

**GET /alerts?channel=0&type=transit**:
```json
{
  "alerts": [
    {
      "alert_id": "a1b2c3",
      "type": "transit_alert",
      "channel": 0,
      "track_id": 567,
      "class": "car",
      "entry_label": "741-North",
      "exit_label": "741-South",
      "method": "confirmed",
      "best_photo_url": "/snapshot/567",
      "duration_ms": 8500,
      "timestamp_ms": 1711612345000
    }
  ],
  "total": 127
}
```

---

## 19. API Endpoints

### Pipeline Control

| Method | Path | Request Body | Response | Notes |
|---|---|---|---|---|
| POST | `/pipeline/start` | — | `{"status": "started"}` | Boots the inference pipeline |
| POST | `/pipeline/stop` | — | `{"status": "stopped"}` | Stops all inference |
| GET | `/channels` | — | `{"channels": [...], "pipeline_started": true}` | List all channels with phase, alert count. Used by home page on load. |
| POST | `/channel/add` | `{"source": "https://youtube.com/watch?v=..." \| "/path/file.mp4"}` | File: `{"channel_id": 0}`. YouTube: `{"status": "resolving", "request_id": "..."}` (202) | Adds source. File sources enter Phase 1 immediately. YouTube URLs resolve asynchronously — channel created on success via WS notification, error sent on failure. |
| POST | `/channel/remove` | `{"channel_id": 0}` | `{"status": "removed"}` | Phase 4 teardown, clears extracted clips |
| POST | `/channel/{id}/phase` | `{"phase": "setup" \| "analytics" \| "review"}` | `{"status": "ok", "phase": "analytics"}` | Transition channel phase. Emits `phase_changed` WS event. Analytics→Review triggers background clip extraction. |
| PATCH | `/config` | `{"confidence_threshold": 0.5}` | `{"status": "updated"}` | Runtime config update |

### Video

| Method | Path | Request Body | Response | Notes |
|---|---|---|---|---|
| GET | `/stream/{channel_id}` | — | `multipart/x-mixed-replace` (MJPEG) | Phase 1/2 live stream |

### Alerts & Snapshots

| Method | Path | Request Body | Response | Notes |
|---|---|---|---|---|
| GET | `/alerts` | Query: `channel`, `type` | Full alert array for channel (no pagination) | Returns all alerts; ~200-400KB for 1,000-2,000 alerts. `channel` filter required for channel tab backfill. |
| GET | `/alert/{alert_id}` | — | Full alert metadata + per-frame data | On-demand detail |
| GET | `/alert/{alert_id}/replay` | — | Video clip (transit) or JPEG frame with bbox (stagnant) | Transit: pre-extracted clip, returns 202 if not yet ready. Stagnant: full frame with bbox overlay. |
| GET | `/snapshot/{track_id}` | — | JPEG image | Best-photo crop |
| POST | `/video/export` | `{"channel_id": 0}` | `{"status": "exporting", "job_id": "..."}` | Annotated video export (recorded only) |

### Site Config

| Method | Path | Request Body | Response | Notes |
|---|---|---|---|---|
| POST | `/site/config` | `SiteConfigRequest` Pydantic model (see Section 17) | `{"status": "saved"}` | Validated: site_id, roi_polygon (min 3 vertices), entry_exit_lines with label+start+end per arm |
| GET | `/site/config` | Query: `site_id` | Site config JSON | Load config |
| GET | `/site/configs` | — | `{"sites": ["741_73", "drugmart_73", ...]}` | List saved configs |

### WebSocket

| Path | Direction | Messages |
|---|---|---|
| GET `/ws?channels=0,1&types=transit_alert,stagnant_alert,stats_update,pipeline_event,phase_changed` | Server -> Client | `frame_data`, `stats_update`, `pipeline_event`, `phase_changed`, `track_ended`, `transit_alert`, `stagnant_alert`. Optional `channels` query param filters to specific channels server-side. Optional `types` query param filters message types server-side — only subscribed types are sent over the wire. If omitted, all channels/types are streamed. |

**M3 WebSocket subscriptions:**
- Home page: `?types=pipeline_event,phase_changed` (no channel filter — receives all channels)
- Channel tab: `?channels={id}&types=transit_alert,stagnant_alert,stats_update,pipeline_event,phase_changed` (no `frame_data` — annotations baked into MJPEG)

---

## 20. Component Map

### Backend

```
backend/
├── main.py                        # FastAPI app, uvicorn entrypoint, CORSMiddleware (allow localhost:5173)
├── pipeline/
│   ├── deepstream/                # DeepStream pipeline (primary, M1-M2)
│   │   ├── pipeline.py            # GStreamer pipeline construction + element linking
│   │   ├── probes.py              # Pad probe functions (phase routing, analytics)
│   │   └── config.py              # DeepStream config file generation (nvinfer, tracker)
│   ├── custom/                    # Custom pipeline (secondary, M9)
│   │   ├── pipeline.py            # Main loop: decode -> batch -> infer -> track -> route
│   │   ├── decoder.py             # NVDEC wrapper (hardware decode)
│   │   ├── preprocess.py          # CUDA NV12->RGB, resize, normalize
│   │   ├── detector.py            # TensorRT YOLOv8s inference
│   │   └── tracker.py             # ByteTrack wrapper
│   ├── source_resolver.py         # YouTube URL → HLS resolution via yt-dlp, liveness check, quality listing
│   ├── direction.py               # Line-crossing detection, direction state machine, inference
│   ├── snapshot.py                # Best-photo scoring, cropping, in-memory storage
│   ├── roi.py                     # Point-in-polygon filtering (ray casting)
│   ├── alerts.py                  # Alert generation, in-memory buffer, replay data storage
│   ├── phase.py                   # Per-channel phase state machine + transition queue
│   ├── idle.py                    # Zero-detection idle optimization logic
│   └── stitch.py                  # Proximity-based track stitching
├── api/
│   ├── routes.py                  # REST endpoints (pipeline control, alerts, config)
│   ├── websocket.py               # WS handler + broadcaster (multiplexed channels)
│   ├── mjpeg.py                   # MJPEG stream endpoint (Phase 1/2)
│   └── video.py                   # Range-request video serving (Phase 3)
├── config/
│   ├── site_config.py             # Per-site config load/save logic
│   └── sites/                     # Saved site JSON configs
│       ├── 741_73.json
│       ├── 741_lytle_south.json
│       └── drugmart_73.json
└── models/
    └── best.engine                # TensorRT model (YOLOv8s or TrafficCamNet)
```

### Frontend

```
frontend/src/
├── api/
│   ├── rest.js                    # fetch wrappers for all REST endpoints
│   └── ws.js                      # WebSocket singleton, event emitter, reconnect logic, types filter
├── contexts/
│   ├── ChannelContext.jsx          # Per-channel tab: phase, channel_id, source, pipeline status
│   ├── AlertContext.jsx            # Per-channel tab: alert array, filters
│   └── HomeContext.jsx             # Home page: channel list, pipeline status
├── pages/
│   ├── Home.jsx                   # Channel manager: add channel, status cards, pipeline start/stop
│   └── Channel.jsx                # Channel tab: phase-driven layout for single channel
├── components/
│   ├── ControlPanel/              # Phase controls, config sliders (channel tab only)
│   ├── VideoPanel/                # MJPEG <img> (Phase 1/2), clip <video>+canvas (Phase 3 transit)
│   ├── StatsBar/                  # FPS, track count, inference ms — from stats_update WS messages
│   ├── AlertFeed/                 # Right sidebar: virtual-scrolled cards (@tanstack/react-virtual), filters
│   ├── DrawingTools/              # ROI polygon + entry/exit line drawing on raw canvas overlay
│   └── ReplayView/                # Transit: looping clip + bbox canvas. Stagnant: full frame + bbox.
├── widgets/
│   ├── TrajectoryWidget/          # Fading centroid trails on canvas
│   └── TrackCountChart/           # Rolling time-series of active vehicle count (from stats_update)
└── lib/
    └── coords.js                  # Canvas coordinate mapping: object-fit:contain offset calc, original↔display conversion
```

**Tech stack:** React + Vite, shadcn/ui + Tailwind CSS (dark mode), @tanstack/react-virtual (alert feed), Recharts (charts), Lucide (icons). Native fetch (no Axios). State management via React Context + useReducer (two contexts per channel tab, one for home page).

**Drawing tools coordinate system:** All points stored in original pixel space (e.g., 1920x1080). Mouse clicks converted immediately using `object-fit: contain` geometry (letterbox offset + scale). Canvas re-renders on resize via ResizeObserver — no coordinate recalculation. See `lib/coords.js`.

---

## 21. Memory Budget

### GPU VRAM

| Allocation | Size | Notes |
|---|---|---|
| TensorRT model weights | ~380 MB | Shared across all channels |
| Frame buffers (double buffered, 2 channels) | ~50 MB | NV12 + RGB per channel |
| NVDEC decode surfaces (2 channels) | ~30 MB | Hardware decoder state |
| TensorRT inference workspace | ~200 MB | Scratch memory |
| CUDA context | ~300 MB | Driver-level overhead |
| OS / driver overhead | ~400 MB | Display + system |
| **Total GPU** | **~1.36 GB** | |
| **VRAM headroom** | **~4.64 GB** | Out of 6 GB |

### CPU Memory

| Allocation | Size | Notes |
|---|---|---|
| Per-track best crops (40 active tracks x 50 KB) | ~2 MB | In-memory, flushed on track end |
| Alert replay data (1,350 alerts x 12 KB) | ~16 MB | Per-frame bbox+centroid |
| Alert snapshots (1,350 alerts x 50 KB) | ~67 MB | Best photos, kept until teardown |
| Trajectory ring buffers (40 tracks x 300 entries) | ~2 MB | 10s history per track |
| **Total CPU (45-min session)** | **~87 MB** | |

---

## 22. Detection Model

### DeepStream Pipeline: TrafficCamNet

TrafficCamNet is the default model for the DeepStream pipeline. It is purpose-built for traffic scenes and ships with DeepStream, requiring no finetuning for v1.

- **Classes:** car, bicycle, person, road_sign (4 classes; filter to car only for junction monitoring). Note: TrafficCamNet groups all vehicle types (cars, trucks, buses, motorcycles) under a single "car" class — typed vehicle classification requires a secondary classifier (VehicleTypeNet) or finetuning (v2).
- **Input:** 960x544, FP16/INT8
- **Architecture:** DetectNet_v2 (ResNet18 backbone)
- **Integration:** Native `nvinfer` config, no custom parser needed

### Custom Pipeline: YOLOv8s (COCO)

YOLOv8s pretrained on COCO is used for the custom pipeline. Vehicle classes (car, truck, bus, motorcycle) are filtered from the 80-class COCO output. No finetuning for v1.

### Model Comparison

| Model | Params | mAP50 (COCO) | TRT INT8 fps* | VRAM | Pipeline |
|---|---|---|---|---|---|
| TrafficCamNet | ~12 M | N/A (proprietary) | ~150 fps | ~350 MB | DeepStream |
| YOLOv8n | 3.2 M | 37.3 | ~180 fps | ~280 MB | Custom (alt) |
| **YOLOv8s** | **11.2 M** | **44.9** | **~120 fps** | **~380 MB** | **Custom** |
| YOLOv8m | 25.9 M | 50.2 | ~75 fps | ~560 MB | — |
| YOLOv9s | 7.1 M | 46.8 | ~110 fps | ~340 MB | — |
| RT-DETR-s | 20 M | 48.1 | ~60 fps | ~620 MB | — |

*Estimated on RTX 4050 Mobile, 640x640 input, batch=1*

### Finetuning (v2)

Finetuning is deferred to v2. When pursued:
- Collect 500-2000 frames from target junctions.
- Annotate with Roboflow or CVAT.
- Finetune YOLOv8s with frozen backbone (first 10 layers).
- Export to TensorRT INT8 using scene-specific calibration images.

---

## 23. Tracking

### DeepStream: NvDCF

NvDCF (NVIDIA Discriminative Correlation Filter) with visual ReID features. Runs on GPU as part of the DeepStream pipeline.

Key configuration:
- `maxShadowTrackingAge`: 60-90 frames (maintains predicted position for occluded vehicles)
- `maxTargetsPerStream`: 50
- ReID features: color names + HOG descriptors
- Batch processing across all channels

### Custom: ByteTrack

ByteTrack runs on CPU. One instance per channel, stored in a dict keyed by channel_id.

- At 20-40 active tracks, CPU execution is faster than GPU kernel launch overhead.
- `track_buffer=90` frames (3 seconds) for lost track retention.
- Kalman filter for position prediction.

### Trajectory Ring Buffer

Per active track, a ring buffer stores the last 300 centroids (x, y, frame_number). At 30fps, this covers 10 seconds of history.

```python
from collections import deque

class TrajectoryBuffer:
    def __init__(self, max_len: int = 300):
        self.buffer: deque[tuple[int, int, int]] = deque(maxlen=max_len)

    def append(self, x: int, y: int, frame: int):
        self.buffer.append((x, y, frame))

    def get_trajectory(self) -> list[tuple[int, int]]:
        """Return (x, y) pairs for drawing."""
        return [(x, y) for x, y, _ in self.buffer]

    def get_displacement(self, window_frames: int) -> float:
        """Centroid displacement over the last N frames."""
        if len(self.buffer) < window_frames:
            return float('inf')  # not enough data
        oldest = self.buffer[-window_frames]
        newest = self.buffer[-1]
        return math.sqrt((newest[0] - oldest[0])**2 + (newest[1] - oldest[1])**2)
```

---

## 24. Technology Stack

| Layer | Choice | Notes |
|---|---|---|
| Backend language | Python 3.12 | FastAPI async, pyservicemaker (DS 8.0 Python API) |
| Video decode (DeepStream) | nvv4l2decoder | GStreamer element, hardware NVDEC |
| Video decode (custom) | NVDEC via Video Codec SDK | Direct hardware decode |
| Pre-process (DeepStream) | nvvideoconvert + nvinfer | Built-in scaling + normalization |
| Pre-process (custom) | CUDA kernel or NVIDIA NPP | NV12->RGB, resize, normalize |
| Detection (DeepStream) | TrafficCamNet via nvinfer | Native DeepStream model |
| Detection (custom) | YOLOv8s -> TensorRT INT8 | Ultralytics export |
| Tracking (DeepStream) | NvDCF (nvtracker) | GPU, ReID features |
| Tracking (custom) | ByteTrack | CPU, Kalman + IoU |
| Line analytics (DeepStream) | Custom Python probe (direction.py) | Cross-product line-crossing algorithm |
| Line analytics (custom) | Custom Python (direction.py) | Cross-product algorithm |
| Annotation (DeepStream) | nvdsosd | GStreamer element, GPU rendering |
| Annotation (custom) | CUDA draw kernel | Boxes + IDs + trails on GPU frame |
| API server | FastAPI + uvicorn | REST + WebSocket + MJPEG + range requests |
| Video streaming (live) | MJPEG over HTTP | multipart/x-mixed-replace |
| Video streaming (replay) | HTTP range requests | Native `<video>` element |
| Video export | NVENC via OpenCV/FFmpeg | Hardware H.264 encode |
| Frontend | React + Vite | |
| Charts | recharts | Lightweight, React-native |
| Canvas drawing | HTML5 Canvas API | ROI, lines, trajectory, replay overlays |
| Config persistence | JSON files | Per-site config |

---

## 25. Project Structure

```
vehicle-tracker/
├── backend/
│   ├── main.py
│   ├── pipeline/
│   │   ├── deepstream/
│   │   │   ├── pipeline.py
│   │   │   ├── probes.py
│   │   │   └── config.py
│   │   ├── custom/
│   │   │   ├── pipeline.py
│   │   │   ├── decoder.py
│   │   │   ├── preprocess.py
│   │   │   ├── detector.py
│   │   │   └── tracker.py
│   │   ├── direction.py
│   │   ├── snapshot.py
│   │   ├── roi.py
│   │   ├── alerts.py
│   │   ├── phase.py
│   │   ├── idle.py
│   │   └── stitch.py
│   ├── api/
│   │   ├── routes.py
│   │   ├── websocket.py
│   │   ├── mjpeg.py
│   │   └── video.py
│   ├── config/
│   │   ├── site_config.py
│   │   └── sites/
│   │       ├── 741_73.json
│   │       ├── 741_lytle_south.json
│   │       └── drugmart_73.json
│   └── models/
│       └── best.engine
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   │   ├── rest.js
│   │   │   └── stream.js
│   │   ├── components/
│   │   │   ├── ControlPanel/
│   │   │   ├── VideoPanel/
│   │   │   ├── StatsBar/
│   │   │   ├── AlertFeed/
│   │   │   └── DrawingTools/
│   │   ├── widgets/
│   │   │   ├── TrajectoryWidget/
│   │   │   └── TrackCountChart/
│   │   └── layouts/
│   └── package.json
├── docs/
│   ├── prd.md
│   ├── design.md
│   ├── changelog.md
│   └── spikes/
├── scripts/
│   └── draw_config.py
├── plans/
├── experiments/
└── data_collection/
    ├── site_videos/
    └── test_clips/
```

---

## 26. Versioning & Release

### Versioning

SemVer (Major.Minor.Patch). Version is derived from **git tags only** via `setuptools-scm` — no version strings in source files.

```
git tag v0.1.0    →  version = "0.1.0"
3 commits later   →  version = "0.1.0.dev3+g1a2b3c4" (auto-generated)
git tag v0.2.0    →  version = "0.2.0"
```

**Milestone to version mapping:**

| Milestone | Version | Significance |
|---|---|---|
| M1 | 0.1.0 | DeepStream pipeline — single-channel file input |
| M2 | 0.2.0 | API layer — FastAPI REST/WS/MJPEG |
| M3 | 0.3.0 | React UI — 4-phase workflow |
| M4 | 0.4.0 | DeepStream-FastAPI integration |
| M5 | 0.5.0 | Multi-channel shared pipeline |
| M6 | 0.6.0 | YouTube Live streams |
| M7 | 0.7.0 | Custom pipeline (NVDEC + TensorRT + ByteTrack) |
| M8 | 0.8.0 | Custom model training (YOLOv8s fine-tune, 4 vehicle classes) |
| M9 | 0.9.0 | Polish (trajectory overlay, profiling, error handling) |

### Release Mechanism: Docker on GitHub Container Registry

GPU-dependent software cannot be distributed as pip packages or standalone binaries. Docker is the release mechanism.

**Base image:** `nvcr.io/nvidia/deepstream:8.0-gc-triton-devel` (includes DeepStream, TensorRT, CUDA, GStreamer).

**Dockerfile structure:**
```dockerfile
FROM nvcr.io/nvidia/deepstream:8.0-gc-triton-devel

# System dependencies + yt-dlp + deno (required for YouTube JS extraction)
RUN apt-get update && apt-get install -y python3-pip nodejs npm curl unzip \
    && pip3 install --break-system-packages yt-dlp \
    && curl -fsSL https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip -o /tmp/deno.zip \
    && unzip /tmp/deno.zip -d /usr/local/bin && rm /tmp/deno.zip

# Backend
COPY backend/ /app/backend/
RUN pip3 install -r /app/backend/requirements.txt

# Frontend (build step)
COPY frontend/ /app/frontend/
RUN cd /app/frontend && npm ci && npm run build

# Models
COPY models/ /app/models/

WORKDIR /app
EXPOSE 8000
CMD ["python3", "backend/main.py"]
```

**Published to:** `ghcr.io/joeljose/vehicle-tracker`

```bash
# User runs:
docker run --gpus all \
  -p 8000:8000 \
  -v /path/to/videos:/data \
  -v /path/to/config:/app/config/sites \
  ghcr.io/joeljose/vehicle-tracker:0.2.0
```

**GitHub Release page contains:**
- Changelog / release notes
- Link to the Docker image tag
- `docker-compose.yml` for easy setup
- Hardware requirements

**Release process:**
1. `git tag v0.2.0 && git push --tags`
2. GitHub Actions workflow triggers on tag push
3. Builds Docker image from the tagged commit
4. Pushes to `ghcr.io/joeljose/vehicle-tracker:0.2.0` and `:latest`
5. Creates a GitHub Release with auto-generated notes

**GitHub Container Registry:** Free for public repos. No storage or bandwidth limits.

---

## 27. Development Environment

**All development happens inside containers. Nothing is installed on the host system.**

### Dev Container

A development container based on the same DeepStream base image used for production. Source code is bind-mounted for live editing.

```yaml
# docker-compose.dev.yml
services:
  backend:
    image: nvcr.io/nvidia/deepstream:8.0-gc-triton-devel
    runtime: nvidia
    volumes:
      - ./backend:/app/backend
      - ./models:/app/models
      - ./data_collection:/data
      - ./config:/app/config
    ports:
      - "8000:8000"
    working_dir: /app
    command: bash -c "pip install -r backend/requirements.txt && python3 backend/main.py --reload"
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

  frontend:
    image: node:22-alpine
    volumes:
      - ./frontend:/app
    ports:
      - "5173:5173"
    working_dir: /app
    command: sh -c "npm ci && npm run dev -- --host"
```

```bash
# Start dev environment
docker compose -f docker-compose.dev.yml up

# Backend: http://localhost:8000
# Frontend: http://localhost:5173
```

**Key points:**
- Backend container has GPU access (`runtime: nvidia`) with DeepStream, TensorRT, CUDA pre-installed
- Source code is bind-mounted — edit on host, changes reflected immediately
- Frontend runs in a lightweight Node container
- `pip install` and `npm ci` run inside the containers, not on the host
- The `--reload` flag on uvicorn enables hot-reload during development
- Video files are bind-mounted from `data_collection/` to `/data` inside the container

### Custom Pipeline Dev Container

When implementing the custom pipeline (M9), the DeepStream base image still works since it includes TensorRT and CUDA. Alternatively, a lighter image can be used:

```yaml
  backend-custom:
    image: nvcr.io/nvidia/tensorrt:24.01-py3
    runtime: nvidia
    volumes:
      - ./backend:/app/backend
      - ./models:/app/models
      - ./data_collection:/data
    ports:
      - "8000:8000"
    working_dir: /app
    command: bash -c "pip install -r backend/requirements.txt && python3 backend/main.py --backend custom --reload"
```

### Development Workflow

1. Clone repo
2. `docker compose -f docker-compose.dev.yml up`
3. Edit code on host (any editor/IDE)
4. Backend auto-reloads, frontend hot-reloads
5. Test in browser at `localhost:5173`
6. Run tests: `docker compose exec backend pytest`
7. No `pip install`, `npm install`, or any package manager invocation on the host — ever

---

## 28. Next Steps (Implementation Order)

Milestones are ordered for incremental, demonstrable progress. Each milestone produces a working system with more capability than the last.

| Milestone | Description | Key Deliverables |
|---|---|---|
| **M1** | DeepStream pipeline — single-channel file input | Detection + NvDCF tracking + ROI + line-crossing + direction FSM + stagnant detection + best-photo capture + track stitching + idle optimization. 177 tests. **COMPLETE (v0.1.0).** |
| **M2** | API layer | FastAPI REST + WebSocket + MJPEG endpoints. In-memory AlertStore. PipelineBackend Protocol. Pipeline controllable from curl, MJPEG viewable in browser. 237 tests. **COMPLETE (v0.2.0).** |
| **M3** | React UI | Video panels (MJPEG), alert feed sidebar, stats bar, phase controls, ROI polygon + entry/exit line drawing tools, site config save/load, WebSocket integration. 286 tests. **COMPLETE (v0.3.0).** |
| **M4** | DeepStream-FastAPI integration | DeepStreamPipeline adapter (PipelineBackend Protocol), per-channel pipeline lifecycle, MjpegExtractor for GPU->JPEG, real-time alerts, clip extraction for replay, EOS auto-transition. 312 tests. **COMPLETE (v0.4.0).** |
| **M5** | Multi-channel | Shared pipeline: nvstreammux → nvinfer → nvtracker → nvstreamdemux → per-channel OSD/MJPEG. Dynamic source add/remove. BatchMetadataRouter. Per-source EOS. 336 tests. **COMPLETE (v0.5.0).** |
| **M6** | YouTube Live streams | YouTube URL resolution via `yt-dlp`, HLS stream consumption, stream recovery with circuit breaker, serialized yt-dlp calls, last-frame buffer for Phase 3 frozen-frame replay. 371 tests. **COMPLETE (v0.6.0).** |
| **M7** | Custom pipeline | NVDEC + TensorRT + ByteTrack pipeline, same API contract as DeepStream |
| **M8** | Custom model training | Auto-label junction video, fine-tune YOLOv8s (4 classes: car/truck/bus/motorcycle), TensorRT FP16 engine. See `training/docs/`. |
| **M9** | Polish | Remaining widgets (trajectory overlay, track count chart), performance profiling with Nsight Systems, edge case handling |
