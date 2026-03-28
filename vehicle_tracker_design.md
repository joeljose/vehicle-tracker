# Vehicle Tracker System — Design Document

## 1. Project Overview

A GPU-accelerated traffic junction monitoring system that detects vehicles, tracks them through the junction, determines their directional transit (entry arm to exit arm), and generates alerts with best-quality vehicle photos. The system supports both recorded video files and live RTSP streams from IP cameras at real traffic junctions.

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

**Architecture principle:** Single process, multi-channel. DeepStream's `nvstreammux` batches all sources for shared inference, saving ~630 MB VRAM compared to separate processes. The custom pipeline similarly batches frames from all channels into a single TensorRT call.

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

### 3.1 DeepStream Pipeline

```
Sources ---> nvstreammux ---> nvinfer ---> nvtracker(NvDCF) ---> nvdsanalytics ---> [PROBE] ---> nvdsosd ---> sinks
                                                                                      |
                                                                          Phase routing:
                                                                          source_id -> channel_state dict
                                                                          Phase 1: render boxes only
                                                                          Phase 2: alerts, photos, direction
                                                                          Phase 3: source removed
```

- `nvstreammux` batches frames from all active channels into a single batch tensor.
- `nvinfer` runs TrafficCamNet (or YOLOv8s) on the batch — one inference call serves all channels.
- `nvtracker` runs NvDCF with ReID features for robust tracking.
- `nvdsanalytics` handles line-crossing primitive detection.
- The **pad probe** function after `nvdsanalytics` is the phase routing point. It inspects `frame_meta.source_id`, looks up the channel's current phase in a shared state dict, and conditionally runs post-processing (direction logic, alert generation, best-photo scoring).
- `nvdsosd` draws bounding boxes, IDs, and trajectory overlays onto the frame.
- Sinks encode to MJPEG for HTTP streaming.

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
        """Add a video source (file path or RTSP URL) to the pipeline."""
        ...

    def remove_channel(self, channel_id: int) -> None:
        """Remove a channel, release its decoder and tracker resources."""
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

```python
def analytics_probe(pad, info):
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(info.get_buffer()))
    frame = batch_meta.frame_meta_list

    while frame is not None:
        frame_meta = pyds.NvDsFrameMeta.cast(frame.data)
        source_id = frame_meta.source_id
        phase = channel_phases.get(source_id, ChannelPhase.SETUP)

        if phase == ChannelPhase.ANALYTICS:
            process_direction(frame_meta)
            process_best_photo(frame_meta)
            process_alerts(frame_meta)
        # Phase 1 (SETUP): boxes rendered by nvdsosd, no post-processing
        # Phase 3 (REVIEW): source already removed from pipeline

        frame = frame.next
    return Gst.PadProbeReturn.OK
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
- Requests are queued in a thread-safe queue.
- The pipeline thread drains the queue **between frame batches**, under a mutex, ensuring no mid-frame state corruption.
- **Phase 1 to 2 transition:** Creates a fresh tracker instance for that channel (clears all existing track IDs). For DeepStream, this means resetting the tracker's per-source state. For recorded video, playback resets to frame 0 and plays once (no loop).
- **Phase 2 to 3 transition:** For recorded video, triggered automatically at video end or manually by the operator. For RTSP, triggered by stream drop or operator click "Stop". The source is removed from the pipeline (DeepStream: runtime source delete via `nvstreammux`).
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
| Trigger | Operator adds a video source (file or RTSP URL) |
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
| Video behavior | Recorded: resets to frame 0, plays once (no loop). RTSP: starts from current time |
| Inference | Full pipeline: detection + tracking + direction + alerts + best-photo |
| Tracker | Fresh instance created (all prior track IDs cleared) |
| Alerts | Transit alerts on line crossing or track exit. Stagnant alerts after 2.5 minutes stopped |
| Video transport | MJPEG over HTTP with full overlays (boxes, IDs, trajectories, direction indicators) |
| Data collection | Per-frame bbox+centroid stored per track for replay |

### Phase 3 — Review

| Aspect | Detail |
|---|---|
| Trigger | Video end (recorded), RTSP stream drop, or operator clicks "Stop" |
| Alert feed | Persists — all alerts from Phase 2 remain visible and clickable |
| Replay (recorded) | Click alert to jump to timestamp in original video. `<video>` element with canvas overlay draws stored per-frame bboxes and trajectory |
| Replay (RTSP) | Frozen last frame displayed as `<img>`. Stored bboxes animated on `<canvas>` overlay at original timing |
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

The direction of crossing (into or out of the junction) is determined by the sign of the cross product. However, the sign's meaning depends on line endpoint drawing order (left-to-right vs right-to-left), which the operator should not need to care about.

**Auto-calibration:** During Phase 1, the system observes the first 5-10 vehicles crossing each line. For each crossing, it records which side the vehicle came from (based on the prior centroid) and which side it went to. The majority vote determines which sign corresponds to "entry" (approaching from outside) vs "exit" (leaving toward outside). This polarity is stored in the site config and can be manually overridden. Auto-calibration requires vehicles to cross each line during Phase 1, which is expected since the operator watches traffic flow while drawing lines.

```python
def check_line_crossing(line: LineSeg, prev_centroid: Point, curr_centroid: Point) -> str | None:
    """Returns 'entry', 'exit', or None."""
    ax, ay = line.start
    bx, by = line.end
    cross_prev = (bx - ax) * (prev_centroid.y - ay) - (by - ay) * (prev_centroid.x - ax)
    cross_curr = (bx - ax) * (curr_centroid.y - ay) - (by - ay) * (curr_centroid.x - ax)

    if cross_prev * cross_curr < 0:  # sign change
        return "entry" if cross_prev > 0 else "exit"
    return None
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

When a track ends without crossing both lines (common for vehicles that appear after the entry line or disappear before the exit line), direction is inferred from trajectory heading.

```python
def nearest_arm(trajectory: list[Point], arms: dict[str, LineSeg], use_start: bool) -> str:
    """
    Average the first or last 5-10 centroids for a stable direction vector.
    Compute dot product of trajectory heading vs each arm's direction vector.
    Return the arm with the highest alignment (largest dot product).
    """
    n = min(10, len(trajectory) // 2)
    if use_start:
        segment = trajectory[:n]    # first N centroids -> infer entry
    else:
        segment = trajectory[-n:]   # last N centroids -> infer exit

    # Direction vector from segment
    heading = (segment[-1][0] - segment[0][0], segment[-1][1] - segment[0][1])
    heading_mag = math.sqrt(heading[0]**2 + heading[1]**2)
    if heading_mag < 1e-6:
        return "unknown"
    heading_norm = (heading[0] / heading_mag, heading[1] / heading_mag)

    best_arm = None
    best_dot = -float('inf')
    for arm_name, line in arms.items():
        arm_dir = (line.end[0] - line.start[0], line.end[1] - line.start[1])
        arm_mag = math.sqrt(arm_dir[0]**2 + arm_dir[1]**2)
        arm_norm = (arm_dir[0] / arm_mag, arm_dir[1] / arm_mag)
        dot = heading_norm[0] * arm_norm[0] + heading_norm[1] * arm_norm[1]
        if dot > best_dot:
            best_dot = dot
            best_arm = arm_name

    return best_arm
```

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
  maxShadowTrackingAge: 75    # 60-90 frames; keeps "shadow" tracks for occluded vehicles
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
            # line, apply nearest_arm() on the combined trajectory's initial
            # heading to infer entry direction (method="inferred").
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

- DeepStream: `pgie.set_property('interval', 15)` for idle, `pgie.set_property('interval', 0)` for active.
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

### 13.2 Phase 3 Replay — Recorded Video

```
Backend:
  FastAPI serves original video file via HTTP range requests
  GET /video/{filename}  (Accept-Ranges: bytes)

Frontend:
  <video src="/video/{filename}">       <-- native HTML5 player with seek/scrub
  +
  <canvas> overlay                      <-- draws stored per-frame bboxes + trajectory
```

The `<video>` element provides native controls (play, pause, seek, scrub). The `<canvas>` overlay draws stored per-frame bounding boxes synchronized via `requestVideoFrameCallback()`.

**Keyframe seeking limitation:** H.264 video can only seek to keyframes (I-frames), which may be 2-5 seconds apart in traffic camera recordings. When seeking to an alert's `first_seen_frame`, the video may land at the nearest prior keyframe. The canvas overlay simply draws nothing until `presentedFrames` matches the first entry in `per_frame_data`, so the user sees raw video for a brief moment before the bbox overlay begins. This is acceptable for v1. If precise seeking is needed in v2, the video can be re-muxed with frequent keyframes during Phase 2 at the cost of disk space.

Synchronization code:

```javascript
function syncOverlay(video, canvas, alertData) {
    const ctx = canvas.getContext('2d');

    video.requestVideoFrameCallback(function onFrame(now, metadata) {
        // Use presentedFrames (monotonic frame count) rather than
        // mediaTime * fps, which can be inaccurate with non-zero start
        // times or variable frame rate containers from traffic cameras.
        const frameNum = metadata.presentedFrames;
        // Find nearest per_frame_data entry (binary search on frame number)
        const frameData = findNearestFrame(alertData.per_frame_data, frameNum);

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (frameData) {
            drawBbox(ctx, frameData.bbox);
            drawTrajectory(ctx, alertData.full_trajectory, frameNum);
        }

        video.requestVideoFrameCallback(onFrame);
    });
}
```

### 13.3 Phase 3 Replay — RTSP (No Original Video)

Since there is no recorded file for RTSP sources, replay uses stored data only:

```
Frontend:
  <img src="{last_frame_jpeg}">         <-- frozen last frame from stream
  +
  <canvas> overlay                      <-- animated bbox sliding along stored trajectory
```

The canvas animates the stored per-frame bboxes at original timing using `requestAnimationFrame()`. The operator sees the vehicle's tracked path replayed on top of the last captured frame.

---

## 14. Per-Alert Replay Data

Each alert stores per-frame data for every frame the tracked vehicle was active.

**Per-frame record:**
```python
@dataclass
class FrameRecord:
    frame: int              # frame number (4 bytes)
    bbox: tuple[int, int, int, int]  # x, y, w, h (16 bytes)
    centroid: tuple[int, int]        # x, y (8 bytes)
    # ~28 bytes per record, ~40 bytes with Python object overhead
```

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

### 16.1 Layout

```
+------------------------------------------------------------+
|  Control Panel (top bar)                                    |
+--------------------------------------------+---------------+
|                                            |  Alert Feed   |
|          Video Panels                      |  (~300px)     |
|     (1-N channels, MJPEG or <video>)       |               |
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
Frontend appends compact card to alert feed
        |
        v  (user clicks card)
GET /alert/{alert_id}  -->  full metadata + per_frame_data
        |
        v
Frontend renders expanded view + enables replay
```

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

**stats_update** (every 1 second):
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

### 18.2 REST Responses

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
    {"frame": 1001, "bbox": [412, 305, 129, 73], "centroid": [476, 341]},
    {"frame": 1002, "bbox": [414, 306, 130, 74], "centroid": [479, 343]}
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

**GET /alerts?limit=50&type=transit**:
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
  "total": 127,
  "has_more": true
}
```

---

## 19. API Endpoints

### Pipeline Control

| Method | Path | Request Body | Response | Notes |
|---|---|---|---|---|
| POST | `/pipeline/start` | — | `{"status": "started"}` | Boots the inference pipeline |
| POST | `/pipeline/stop` | — | `{"status": "stopped"}` | Stops all inference |
| POST | `/channel/add` | `{"source": "rtsp://..." \| "/path/file.mp4"}` | `{"channel_id": 0}` | Adds source, enters Phase 1 |
| POST | `/channel/remove` | `{"channel_id": 0}` | `{"status": "removed"}` | Phase 4 teardown |
| POST | `/channel/{id}/phase` | `{"phase": "setup" \| "analytics" \| "review"}` | `{"status": "ok", "phase": "analytics"}` | Transition channel phase |
| PATCH | `/config` | `{"confidence_threshold": 0.5}` | `{"status": "updated"}` | Runtime config update |

### Video

| Method | Path | Request Body | Response | Notes |
|---|---|---|---|---|
| GET | `/stream/{channel_id}` | — | `multipart/x-mixed-replace` (MJPEG) | Phase 1/2 live stream |
| GET | `/video/{filename}` | — | Video file (HTTP range requests) | Phase 3 recorded replay |

### Alerts & Snapshots

| Method | Path | Request Body | Response | Notes |
|---|---|---|---|---|
| GET | `/alerts` | Query: `limit`, `type` | Paginated alert summaries | Historical alerts |
| GET | `/alert/{alert_id}` | — | Full alert metadata + per-frame data | On-demand detail |
| GET | `/snapshot/{track_id}` | — | JPEG image | Best-photo crop |
| POST | `/video/export` | `{"channel_id": 0}` | `{"status": "exporting", "job_id": "..."}` | Annotated video export (recorded only) |

### Site Config

| Method | Path | Request Body | Response | Notes |
|---|---|---|---|---|
| POST | `/site/config` | Site config JSON (see Section 17) | `{"status": "saved"}` | Save ROI, lines, labels |
| GET | `/site/config` | Query: `site_id` | Site config JSON | Load config |
| GET | `/site/configs` | — | `{"sites": ["741_73", "drugmart_73", ...]}` | List saved configs |

### WebSocket

| Path | Direction | Messages |
|---|---|---|
| GET `/ws?channels=0,1` | Server -> Client | `frame_data`, `stats_update`, `pipeline_event`, `track_ended`, `transit_alert`, `stagnant_alert`. Optional `channels` query param filters to specific channels server-side. If omitted, all channels are streamed. |

---

## 20. Component Map

### Backend

```
backend/
├── main.py                        # FastAPI app, uvicorn entrypoint
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
│   ├── rest.js                    # Axios wrappers for all REST endpoints
│   └── stream.js                  # WebSocket singleton, event emitter, reconnect logic
├── components/
│   ├── ControlPanel/              # Channel management, phase controls, config sliders
│   ├── VideoPanel/                # MJPEG <img> (Phase 1/2), <video>+canvas (Phase 3)
│   ├── StatsBar/                  # FPS, track count, inference ms, phase indicator
│   ├── AlertFeed/                 # Right sidebar: compact cards, filters, expand on click
│   └── DrawingTools/              # ROI polygon + entry/exit line drawing on canvas overlay
├── widgets/
│   ├── TrajectoryWidget/          # Fading centroid trails on canvas
│   └── TrackCountChart/           # Rolling time-series of active vehicle count
└── layouts/                       # Saved layout JSON configs
```

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
| Backend language | Python | FastAPI async, GStreamer/DeepStream bindings |
| Video decode (DeepStream) | nvv4l2decoder | GStreamer element, hardware NVDEC |
| Video decode (custom) | NVDEC via Video Codec SDK | Direct hardware decode |
| Pre-process (DeepStream) | nvvideoconvert + nvinfer | Built-in scaling + normalization |
| Pre-process (custom) | CUDA kernel or NVIDIA NPP | NV12->RGB, resize, normalize |
| Detection (DeepStream) | TrafficCamNet via nvinfer | Native DeepStream model |
| Detection (custom) | YOLOv8s -> TensorRT INT8 | Ultralytics export |
| Tracking (DeepStream) | NvDCF (nvtracker) | GPU, ReID features |
| Tracking (custom) | ByteTrack | CPU, Kalman + IoU |
| Line analytics (DeepStream) | nvdsanalytics + custom probe | Line-crossing primitives |
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
│   └── design.md
└── data/
    └── videos/
        ├── 741_73/
        ├── 741_lytle_south/
        └── drugmart_73/
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
| M1-M2 | 0.1.0 | First working pipeline (DeepStream, single channel, alerts) |
| M3-M4 | 0.2.0 | API + UI working end-to-end |
| M5-M6 | 0.3.0 | Full workflow (setup, analytics, review with replay) |
| M7-M8 | 0.4.0 | Multi-channel + RTSP |
| M9 | 0.5.0 | Custom pipeline alternative |
| M10 | 1.0.0 | Production-ready release |

### Release Mechanism: Docker on GitHub Container Registry

GPU-dependent software cannot be distributed as pip packages or standalone binaries. Docker is the release mechanism.

**Base image:** `nvcr.io/nvidia/deepstream:7.0-triton-multiarch` (includes DeepStream, TensorRT, CUDA, GStreamer).

**Dockerfile structure:**
```dockerfile
FROM nvcr.io/nvidia/deepstream:7.0-triton-multiarch

# System dependencies
RUN apt-get update && apt-get install -y python3-pip nodejs npm

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

## 27. Next Steps (Implementation Order)

Milestones are ordered for incremental, demonstrable progress. Each milestone produces a working system with more capability than the last.

| Milestone | Description | Key Deliverables |
|---|---|---|
| **M1** | DeepStream pipeline — single-channel file input | Detection + NvDCF tracking + ROI polygon + line-crossing primitives working end-to-end |
| **M2** | Direction state machine + alerts | Direction state machine, stagnant detection, best-photo capture via probe functions |
| **M3** | FastAPI API layer | All REST endpoints, WebSocket broadcaster, MJPEG streaming — pipeline-agnostic API boundary |
| **M4** | React shell | Video panels (MJPEG), alert feed sidebar, stats bar, phase controls |
| **M5** | ROI + line drawing UI | Canvas-based ROI polygon + entry/exit line drawing tools, site config persistence |
| **M6** | Phase 3 replay | `<video>` + canvas overlay for recorded, animated overlay for RTSP, per-alert replay data |
| **M7** | Multi-channel | Single process, nvstreammux batching, independent per-channel phases |
| **M8** | RTSP with reconnect | Live RTSP source support with automatic reconnection |
| **M9** | Custom pipeline | NVDEC + TensorRT + ByteTrack pipeline, same API contract as DeepStream |
| **M10** | Polish | Remaining widgets (trajectory overlay, track count chart), performance profiling with Nsight Systems, edge case handling |
