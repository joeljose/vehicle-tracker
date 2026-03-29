# Product Requirements Document
## Vehicle Tracker System — Traffic Junction Monitor

**Version:** 0.3
**Status:** M2 Complete
**Last updated:** March 2026

---

## 1. Purpose

This document defines the product requirements for a GPU-accelerated vehicle tracking system designed specifically for **traffic junction monitoring**. The system detects vehicles at road junctions, determines their directional transit (which arm they entered from and exited to), and generates alerts with best-quality vehicle photos. It supports both recorded video analysis and YouTube Live stream monitoring from a single consumer GPU machine with a browser-based UI.

This PRD serves as the single source of truth for development decisions and milestone planning.

---

## 2. Problem Statement

Traffic engineers and analysts monitoring road junctions need to understand vehicle movement patterns: which directions vehicles travel through the junction, how long they take, and whether any vehicles become stagnant (blocking an arm for extended periods). Current approaches either require expensive commercial systems, rely on manual observation, or use generic tracking tools that provide raw bounding boxes without junction-specific intelligence.

This project solves that by running the full pipeline -- decode, detect, track, classify direction, generate alerts -- on a single consumer GPU machine. The system produces directional transit alerts with vehicle photos, supports YouTube Live streams and recorded footage, and provides a browser UI for ROI configuration, alert monitoring, and video replay.

**Target junctions (initial deployment):**
- 741 & 73
- 741 & Lytle South
- Drugmart & 73

Seven videos from these three junctions have been collected for development and validation.

---

## 3. Goals

- Detect and track vehicles at traffic junctions using pre-trained models (no finetuning required for v1)
- Determine directional transit (entry arm to exit arm) for each vehicle passing through the junction
- Generate alerts with a best-quality vehicle photo on transit completion or stagnant detection
- Support recorded video files (analyze then review) and YouTube Live streams (live monitoring) through the same pipeline
- Provide a browser-based UI with an alert feed, ROI/line drawing tools, and video replay capability
- Keep the entire system self-contained on a single local machine with a consumer-grade NVIDIA GPU (RTX 4050 Mobile, 6 GB VRAM)

---

## 4. Non-Goals

- Cloud deployment or multi-machine distribution (out of scope for v1)
- More than 2 simultaneous channels (revisit in v2)
- Mixing pipeline backends simultaneously (DeepStream and custom pipeline are built and run separately, never combined)
- Cross-camera vehicle re-identification
- Night-time operation (v1 targets daytime footage only)
- Model finetuning on custom data (v1 uses pre-trained models only)
- U-turn detection (not reliably observable at these junctions)
- Audio processing
- License plate recognition
- Mobile or native app UI

---

## 5. Users

### Primary user -- operator / developer
The person setting up and running the system. Comfortable with terminal and Python. Will configure sources, draw ROI polygons and entry/exit lines, label junction arms, and monitor alerts. Needs clear API contracts and a reliable pipeline.

### Secondary user -- analyst / viewer
Someone who uses the UI to monitor live feeds, review alerts, and replay vehicle transits. Does not touch configuration files or code. Needs an intuitive control panel, readable alert cards, and straightforward video replay.

---

## 6. Application Workflow

Each channel independently progresses through four phases. Multiple channels can be in different phases simultaneously.

### Phase 1 -- Setup
- Operator adds a video source (file path or YouTube Live URL). Video plays immediately (loops for recorded files).
- Detection and tracking run as a live preview -- bounding boxes with track IDs are drawn on the video. This provides visual feedback for configuration.
- Operator draws a broad ROI polygon and entry/exit lines on the video using canvas drawing tools. Operator labels each arm (defaults: N/E/S/W, editable to road names like "741-North").
- No alerts, direction detection, or best-photo capture occur in this phase.
- Site configuration (ROI, lines, labels) can be saved to disk and reloaded for the same junction. Saved configs are a convenience for persistence — they are never auto-loaded by the pipeline.

### Phase 2 -- Analytics
- Operator clicks "Start Analytics". The frontend sends the currently drawn ROI polygon and entry/exit lines as part of the phase transition request. The pipeline always uses the operator's current drawing — never a saved site config.
- For recorded video: playback restarts from frame 0 and plays once. For YouTube Live: analytics starts from the current moment.
- Tracker is reset (fresh instance created for the channel) to avoid stale tracks from Phase 1.
- Full pipeline runs: detection, tracking, direction detection (line crossing), best-photo capture, vehicle state classification, and alert generation.
- Transit alerts and stagnant alerts appear in the alert feed sidebar in real time.

### Phase 3 -- Review
- Triggered by: recorded video reaching end-of-file, YouTube Live stream ending or dropping, or operator clicking "Stop".
- Alert feed remains visible and interactive. No new alerts are generated.
- **Recorded video:** Click an alert to jump to that timestamp in the video. Video replays the segment with stored per-frame bounding box and trajectory overlays drawn on a canvas layer. Native `<video>` controls available.
- **YouTube Live (no recording):** Click an alert to see stored bounding boxes animated on a frozen last frame, replaying the vehicle's trajectory at original timing.
- Option to export an annotated video with fading trajectory overlays (recorded video sources only). Export is triggered on exiting Phase 3.

### Phase 4 -- Teardown
- Operator removes the stream. All in-memory data for that channel (alerts, snapshots, replay data) is cleared. GPU resources for the channel are released.

---

## 7. Site & Direction Model

| ID | Requirement | Priority |
|---|---|---|
| S-01 | Default direction convention: top of frame = North, clockwise (N, E, S, W) | Must |
| S-02 | Direction labels are editable per arm -- operator can rename "north" to "741-North", "east" to "73-East", etc. | Must |
| S-03 | Each site has a configuration consisting of: ROI polygon, 4 entry/exit lines with labels, and tuning parameters | Must |
| S-04 | Site config is persisted to disk as JSON and reloaded when the same site is loaded again | Must |
| S-05 | Three known sites pre-defined: `741_73`, `741_lytle_south`, `drugmart_73` | Should |
| S-06 | Site config schema includes: `site_id`, `roi_polygon`, `entry_exit_lines` (per arm: label, start point, end point), `stagnant_threshold_seconds`, `motion_threshold_pixels`, `idle_threshold_seconds`, `inference_fps` | Must |

---

## 8. Functional Requirements

### 8.1 Input Sources

| ID | Requirement | Priority |
|---|---|---|
| F-01 | System accepts YouTube Live stream URLs as input sources. Backend resolves the YouTube URL to an HLS stream URL via `yt-dlp` at channel add time. | Must |
| F-02 | System accepts local video files (mp4, mkv, avi) as input sources | Must |
| F-03 | Architecture supports N simultaneous channels -- the channel limit is a runtime config value, not a structural constraint | Must |
| F-04 | YouTube Live stream recovery: on stream interruption, retry the existing HLS URL up to 3 times (5s, 10s, 20s backoff). If retries fail, check if the stream is still live via `yt-dlp`. If the stream ended, move to Phase 3 automatically. If the stream is still live but the URL is stale, re-extract the URL once and reconnect. If still failing, surface the error to the operator. | Must |
| F-04a | Stream quality selection: best available quality by default. Operator can change quality from the UI per channel. Quality options are fetched from `yt-dlp` at channel add time. | Must |
| F-05 | File sources play once during Phase 2 (analytics); loop during Phase 1 (setup preview) | Must |
| F-06 | Channels can be added or removed at runtime via the UI without restarting the pipeline | Must |
| F-07 | Each channel progresses through phases independently -- one channel in Setup while another is in Analytics | Must |
| F-08 | All inference is normalized to 30 fps regardless of source frame rate; 60 fps sources skip every other frame for inference while decoding at native fps for smooth display | Must |

### 8.2 Detection

| ID | Requirement | Priority |
|---|---|---|
| F-09 | v1 uses pre-trained models only -- no finetuning required to start | Must |
| F-10 | DeepStream pipeline uses NVIDIA TrafficCamNet as the primary detector | Must |
| F-11 | Custom pipeline uses YOLOv8s (COCO pre-trained) compiled to TensorRT INT8 | Must |
| F-12 | Video is decoded using hardware decoder (NVDEC for custom pipeline, DeepStream's internal decoder for DS pipeline) | Must |
| F-13 | Detection confidence threshold is configurable at runtime from the UI | Must |
| F-14 | The model can be replaced by dropping a new model file -- no code change required | Should |
| F-15 | Vehicle classes detected depend on the chosen model: TrafficCamNet (DeepStream) detects a single "car" class encompassing all vehicle types; YOLOv8s COCO (custom pipeline) detects car, truck, bus, motorcycle separately. Typed vehicle classes with DeepStream require finetuning or a secondary classifier (v2). | Must |
| F-16 | Idle optimization: when zero detections persist for 1 second (30 frames), inference drops to every 15th frame; any detection immediately restores full-rate inference; inference is never skipped while active tracks exist | Must |

### 8.3 Tracking

| ID | Requirement | Priority |
|---|---|---|
| F-17 | DeepStream pipeline: NvDCF tracker with ReID, `maxShadowTrackingAge` set to 60-90 frames for occlusion handling | Must |
| F-18 | Custom pipeline: ByteTrack tracker on CPU with extended `track_buffer` for occlusion handling | Must |
| F-19 | Each detected vehicle is assigned a persistent integer track ID that remains stable across frames | Must |
| F-20 | Per-track centroid trajectory is maintained in a ring buffer holding 300 entries (10 seconds at 30 fps) | Must |
| F-21 | When a track is lost, a `track_ended` event is emitted with the full trajectory, direction state (entry/exit arms, method), best_photo_url, and vehicle state (completed/stagnant/lost) | Must |
| F-22 | Proximity-based track stitching: if a new track appears within 100 px and 2 seconds of a lost track's predicted position, the tracks are merged and the original entry direction is preserved | Must |
| F-23 | Phase 1 to Phase 2 transition creates a fresh tracker instance for that channel to avoid stale tracks from the setup phase | Must |
| F-24 | Phase transitions are queued from the API thread and applied between frame batches (mutex-protected) | Must |

### 8.4 Direction Detection

| ID | Requirement | Priority |
|---|---|---|
| F-25 | Direction detection uses entry/exit line crossing per junction arm | Must |
| F-26 | Line crossing is detected via sign change of cross-product on consecutive centroids relative to the line segment | Must |
| F-26a | Line polarity (which side is junction interior) is auto-calibrated by observing the first 5-10 vehicles crossing each line during Phase 1 -- operator does not need to specify drawing direction | Must |
| F-27 | Per-track direction state machine: UNKNOWN -> ENTERED(arm) -> EXITED(arm) -> ALERT | Must |
| F-28 | Confirmed transit: vehicle crossed both an entry line and an exit line -- entry and exit arms are known precisely | Must |
| F-29 | Inferred transit: when a track is lost before crossing an exit line (or appears after an entry line), trajectory heading is used to infer the missing direction via dot product against arm direction vectors | Should |
| F-30 | `nearest_arm()` calculation averages the first/last 5-10 centroids for stable direction estimation | Should |

### 8.5 Vehicle States

| ID | Requirement | Priority |
|---|---|---|
| F-31 | Three vehicle states: in-transit (moving through junction), waiting (stopped for less than 2.5 minutes), stagnant (stopped for 2.5 minutes or more) | Must |
| F-32 | Stagnant detection: centroid displacement below a configurable pixel threshold over 150 seconds (2.5 minutes) triggers a stagnant alert | Must |
| F-33 | A vehicle that was stagnant and then resumes moving retains its stagnant history -- if it later completes a transit, the alert includes both transit and stagnant information | Should |
| F-34 | All time-based thresholds are configured in seconds, not frame counts; conversion to frames uses the source fps | Must |

### 8.6 Best-Photo Capture

| ID | Requirement | Priority |
|---|---|---|
| F-35 | Best-photo score per frame per tracked vehicle: `bbox_area * confidence` | Must |
| F-36 | Only the highest-scoring crop is stored per track (approximately 50 KB per crop) | Must |
| F-37 | Crop is taken from the full-resolution decoded frame, not the inference-resolution tensor | Must |
| F-38 | On track end, the best photo is written to disk and its URL is included in the alert payload | Must |

### 8.7 ROI

| ID | Requirement | Priority |
|---|---|---|
| F-39 | Broad ROI polygon serves as a noise filter -- only vehicles with centroids inside the ROI polygon enter direction detection, alert generation, and best-photo capture. Tracking itself runs on ALL detections regardless of ROI (needed for stable IDs and Phase 1 preview). | Must |
| F-40 | Entry/exit lines (one per junction arm) serve as event triggers for direction detection | Must |
| F-41 | Detection and tracking run on the full frame at all times -- the ROI only gates which tracked vehicles enter direction detection, alert generation, and best-photo capture | Must |
| F-42 | Vehicles outside the ROI are still rendered with bounding boxes on the annotated video but are not tracked for alerts | Should |
| F-43 | ROI polygon and entry/exit lines are drawn in the UI during Phase 1 and persisted as part of the site configuration | Must |

### 8.8 Alert System

| ID | Requirement | Priority |
|---|---|---|
| F-44 | Two alert types: transit alerts (vehicle completed a directional transit) and stagnant alerts (vehicle stationary for over 2.5 minutes) | Must |
| F-45 | Alerts are pushed to the UI via WebSocket as lightweight summaries: track_id, direction labels, best_photo_url, timestamp, and alert type | Must |
| F-46 | Full alert metadata (per-frame bounding boxes, full trajectory, timing details, confidence stats) is fetched on demand via `GET /alert/{alert_id}` | Must |
| F-47 | All alerts and snapshots are kept in memory until the stream is removed (Phase 4) -- no disk persistence of alert metadata required | Must |
| F-48 | Per-frame replay data stored per alert: frame_number, bbox (x, y, w, h), centroid (x, y) for every frame the track was active -- approximately 12 KB per alert, approximately 85 MB total for a 45-minute session with 1,350 alerts | Must |

### 8.9 Annotated Video Export

| ID | Requirement | Priority |
|---|---|---|
| F-49 | For recorded video sources only: system can render an annotated video with stored bounding box and fading trajectory overlays | Should |
| F-50 | Export is triggered via `POST /video/export` and produces a downloadable video file | Should |

### 8.10 Output -- Backend

| ID | Requirement | Priority |
|---|---|---|
| F-51 | Annotated video frames (bounding boxes + track IDs) are served as MJPEG streams over HTTP, one endpoint per channel (Phase 1 and Phase 2) | Must |
| F-52 | Recorded video files are served via HTTP with range-request support for Phase 3 replay (seeking, partial content) | Must |
| F-53 | Per-frame detection and tracking data is broadcast over WebSocket as `frame_data` messages | Must |
| F-54 | Per-second pipeline stats (fps, active tracks, inference latency) are broadcast as `stats_update` messages | Must |
| F-55 | Pipeline lifecycle events (started, stopped, error) are broadcast as `pipeline_event` messages | Must |
| F-56 | Track termination data is broadcast as `track_ended` messages | Must |
| F-57 | Transit and stagnant alerts are broadcast as lightweight WebSocket messages | Must |

### 8.11 API

| ID | Requirement | Priority |
|---|---|---|
| F-58 | `POST /pipeline/start` -- start the pipeline | Must |
| F-59 | `POST /pipeline/stop` -- stop the pipeline and clean up GPU resources | Must |
| F-60 | `POST /channel/add` -- add a new source channel (`{ source: "https://youtube.com/watch?v=..." \| "/path/file.mp4" }`) | Must |
| F-61 | `POST /channel/remove` -- remove a channel (`{ channel_id: 0 }`). Clears extracted clips. | Must |
| F-62 | `POST /channel/{id}/phase` -- transition a channel's phase. Setup→Analytics request includes `roi_polygon` and `entry_exit_lines` from the operator's drawing. Emits `phase_changed` WS event. Analytics→Review triggers background clip extraction. | Must |
| F-63 | `PATCH /config` -- update runtime parameters (e.g., confidence threshold) | Must |
| F-64 | `GET /stream/{channel_id}` -- MJPEG annotated video stream (Phase 1 and 2) | Must |
| F-65 | `GET /channels` -- list all channels with phase and alert count (home page) | Must |
| F-66 | `GET /alerts?channel={id}&type=transit\|stagnant` -- full alert array for channel, no pagination (~200-400KB) | Must |
| F-67 | `GET /alert/{alert_id}` -- full alert metadata including per-frame bounding box data (with `timestamp_ms` per frame for replay sync) | Must |
| F-67b | `GET /alert/{alert_id}/replay` -- transit: pre-extracted clip (202 if not ready), stagnant: full frame with bbox JPEG | Must |
| F-68 | `GET /snapshot/{track_id}` -- best-photo JPEG for a given track | Must |
| F-69 | `POST /site/config` -- save site configuration via validated Pydantic model (ROI, lines, labels) | Must |
| F-70 | `GET /site/config` -- load current site configuration | Must |
| F-71 | `GET /site/configs` -- list all saved site configurations | Should |
| F-73 | `POST /video/export` -- render annotated video with stored overlays (recorded only) | Should |
| F-74 | `GET /ws?channels=0,1&types=...` -- WebSocket with optional `channels` and `types` server-side filters. Types: `frame_data`, `stats_update`, `pipeline_event`, `phase_changed`, `track_ended`, `transit_alert`, `stagnant_alert`. M3 frontend subscribes without `frame_data`. | Must |

### 8.12 UI -- Core

**Architecture:** Tab-per-channel model. Home page (`/`) is the channel manager — pipeline start/stop, add channel, text status cards per channel. Each channel opens in its own browser tab (`/channel/{id}`) with the full phase-driven layout. Eliminates multi-channel layout conflicts.

**Layout:** Phase-driven UI per channel tab. The main content area adapts to the current phase (Setup / Analytics / Review). Video panel and alert feed are always side-by-side. Phase indicator at the top serves as navigation.

**Design:** Dark mode default (monitoring dashboard standard). shadcn/ui + Tailwind CSS component library. Color scheme: background `#0f1117`, cards `#1a1d23`, transit=blue `#60a5fa`, stagnant=amber `#fbbf24`, active=green `#4ade80`, error=red `#f87171`.

**State management:** React Context + useReducer. Two contexts per channel tab (ChannelContext, AlertContext), one for home page (HomeContext). No external state library.

**Best-photo thumbnails:** Always square (long_side × long_side). Vehicle occupies ≥50% of visible area. Scene expansion for context, black padding for edge cases and remaining gap.

| ID | Requirement | Priority |
|---|---|---|
| F-75 | Home page allows adding and removing source channels (YouTube Live URL or file path) | Must |
| F-76 | Home page allows starting and stopping the pipeline | Must |
| F-77 | Channel tab exposes a confidence threshold slider that updates the backend in real time | Must |
| F-78 | Phase indicator displayed in channel tab showing current phase (Setup / Analytics / Review). Acts as top-level navigation -- UI content adapts to current phase. | Must |
| F-79 | Start/Stop Analytics button in channel tab for phase transitions | Must |
| F-80 | Each channel opens in its own browser tab with a single video panel -- tab-per-channel model | Must |
| F-81 | Video panel renders MJPEG via `<img>` tag during Setup and Analytics phases | Must |
| F-82 | Video panel renders looping clip via `<video>` with `<canvas>` overlay for transit alert Review (recorded video). Clips pre-extracted at phase transition, spinner if not yet ready. | Must |
| F-83 | Video panel renders frozen last frame as `<img>` with animated `<canvas>` overlay for transit alert Review (YouTube Live) | Must |
| F-82b | Stagnant alert Review (recorded): full frame with bbox overlay as static `<img>` | Must |
| F-83b | Stagnant alert Review (YouTube Live): bbox drawn on frozen last frame as static `<img>` | Must |
| F-84 | Transparent `<canvas>` overlay layer on top of video panel for ROI and line drawing in Setup phase. Coordinates stored in original pixel space, converted via object-fit:contain geometry. | Must |
| F-85 | Stats bar below video shows fps, active track count, and inference latency (from `stats_update` WS messages, Analytics phase) | Must |
| F-86 | Channel tab connects to backend WebSocket with `types` filter (no `frame_data`) and reconnects automatically on drop. Home page connects with `types=pipeline_event,phase_changed`. | Must |
| F-100 | Alert feed only shows completed events (transit alerts, stagnant alerts) -- not raw detections or active tracks. Active tracking is visible in the video panel. | Must |
| F-101 | Home page shows text status cards per channel: source name, current phase, alert count. Updated via `phase_changed` WS events. | Must |
| F-102 | Home page fetches `GET /channels` on load for initial state. | Must |
| F-103 | CORS middleware on backend to allow frontend (localhost:5173) requests. | Must |

### 8.13 UI -- Alert Feed

| ID | Requirement | Priority |
|---|---|---|
| F-87 | Alert feed displayed as a persistent right sidebar in channel tab, approximately 300 px wide | Must |
| F-88 | Compact alert cards showing: square thumbnail (best photo), direction labels, track ID, duration, method, relative timestamp. Minimal format — feed must be scannable at high alert rates. | Must |
| F-89 | Stagnant alerts visually distinguished with amber/yellow highlight (transit = blue, stagnant = amber) | Must |
| F-90 | Filter buttons at top of feed: All, Transit, Stagnant (client-side filtering, all alerts in memory) | Must |
| F-91 | Alert cards are read-only during Analytics phase (no click interaction). Clickable only during Review phase (after analytics stops or video ends). | Must |
| F-92 | Click a transit alert during Review (recorded): play pre-extracted clip loop with bbox overlay via `requestVideoFrameCallback()` + `mediaTime` sync | Must |
| F-93 | Click a transit alert during Review (YouTube Live): animated bounding box replays on frozen last frame at original timing | Must |
| F-92b | Click a stagnant alert during Review (recorded): show full frame with bbox at best-photo timestamp | Must |
| F-93b | Click a stagnant alert during Review (YouTube Live): show bbox on frozen last frame | Must |
| F-99 | Alert feed auto-scrolls to newest when user is at the top. If user has scrolled down, stop auto-scrolling and show a "New alerts" badge to jump back. | Must |
| F-104 | Alert feed uses virtual scrolling (@tanstack/react-virtual) to render only visible cards. Handles 1,000-2,000 alerts without DOM performance issues. | Must |
| F-105 | On tab open/reopen, backfill alerts via `GET /alerts?channel={id}` (full array, no pagination). | Must |

### 8.14 UI -- Widgets (v1 Scope)

| ID | Requirement | Priority |
|---|---|---|
| F-94 | Alert feed sidebar (described in 8.13) | Must |
| F-95 | ROI polygon and entry/exit line drawing tool with raw canvas overlay during Phase 1. Coordinates in original pixel space, ResizeObserver for responsive redraw. | Must |
| F-96 | Fading trajectory overlay -- centroid trails with configurable fade length, drawn on the video panel canvas | Should |
| F-97 | Live track count chart -- rolling time-series of active vehicle count per channel (from `stats_update`) | Should |
| F-98 | All other widgets (confidence histogram, speed estimator, zone counter, heatmap, track timeline, headless replay, alert rules) are deferred to v2 | -- |

---

## 9. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NF-01 | Pipeline sustains 30 fps inference per channel at the 2-channel config on RTX 4050 Mobile, 720p |
| NF-02 | Total GPU VRAM usage stays below 2 GB at 2-channel config (out of 6 GB available) |
| NF-03 | Inference normalized to 30 fps; 60 fps sources skip every other frame for inference while decoding at native fps |
| NF-04 | Idle optimization: zero detections for 1 second triggers inference every 15th frame; any detection restores full rate; never skipped while active tracks exist |
| NF-05 | Alert delivery latency: transit and stagnant alerts delivered to the UI within 500 ms of the triggering event (track exit or stagnant threshold crossed) |
| NF-06 | Per-alert replay data approximately 12 KB (40 bytes/frame x 300 frames); total approximately 85 MB for a 45-minute session |
| NF-07 | Best-photo snapshots approximately 50 KB each, kept in memory until stream removed (Phase 4) |
| NF-08 | Single-process architecture: FastAPI server and pipeline run in the same Python process on localhost |
| NF-09 | Multi-channel uses single `nvstreammux` (DeepStream) or single batch loop (custom) sharing one inference engine -- saves approximately 630 MB VRAM vs separate processes |
| NF-10 | Double buffering per channel prevents GPU stalls between frame transfers |
| NF-11 | Pinned (page-locked) host memory used for all CPU-GPU transfers |
| NF-12 | WebSocket `frame_data` messages delivered within one frame period (~33 ms) of inference completion |
| NF-13 | Graceful shutdown: GPU resources and decoder sessions released cleanly on stop |
| NF-14 | Offline capable for file sources after initial model download. YouTube Live sources require internet access for `yt-dlp` URL resolution and HLS stream consumption. |
| NF-15 | Frontend runs in any modern Chromium-based browser without plugins |

---

## 10. Architecture Summary

### Pipeline design: always infer, conditionally analyze

Detection and tracking run continuously on all active channels regardless of phase. Post-processing (direction detection, alert generation, best-photo capture) only runs for channels in Phase 2. This is the "always infer, conditionally analyze" pattern.

- **DeepStream pipeline:** Phase routing implemented in a GStreamer pad probe function. The probe checks `frame_meta.source_id` against a channel state dictionary. Phase 1 channels get rendered boxes only. Phase 2 channels get full post-processing. Phase 3 channels are removed from the pipeline.
- **Custom pipeline:** Phase routing implemented as an if/elif block after inference and before post-processing.

### Single process, multi-channel

All channels share a single inference engine. DeepStream uses `nvstreammux` to batch frames from multiple sources for shared inference. The custom pipeline batches frames from multiple decoders into a single TensorRT call. This saves approximately 630 MB VRAM compared to running separate processes per channel.

### API boundary and pipeline interface

Both pipeline implementations conform to the same `PipelineBackend` Python Protocol. The FastAPI layer imports only this interface — never DeepStream or custom pipeline modules directly. Swapping backends is a startup config flag (`--backend deepstream` or `--backend custom`), not a code change. Shared modules (direction detection, ROI, alerts, best-photo, idle optimization, track stitching) are used by both backends and live outside the backend-specific directories.

The UI is fully pipeline-agnostic — it works identically regardless of which pipeline backend is running.

### System diagram

```
Browser (React, localhost)
    |
    |-- REST   -> FastAPI  (control: pipeline, channels, config, site, alerts, export)
    |-- WS     -> FastAPI  (data: frame_data, stats, events, track_ended, alerts)
    |-- MJPEG  -> FastAPI  (annotated video: /stream/{channel_id}, Phase 1/2)
    +-- HTTP   -> FastAPI  (recorded video: /video/{filename}, Phase 3 range requests)
                   |
                   v
            Pipeline core (DeepStream OR Custom -- never both)
            |-- Source manager    (YouTube Live / file, multi-channel mux)
            |-- Decoder           (NVDEC / DeepStream internal)
            |-- Inference engine  (TensorRT, shared across channels)
            |-- Tracker           (NvDCF or ByteTrack, one instance per channel)
            |-- Phase router      (probe function or if/elif)
            |-- Direction engine  (line crossing + state machine, Phase 2 only)
            |-- Best-photo engine (score + crop, Phase 2 only)
            |-- Alert engine      (transit + stagnant, Phase 2 only)
            |-- Idle optimizer    (skip inference when scene empty)
            +-- Annotator + MJPEG encoder
```

### Lifecycle

Browser opens -> WebSocket connects -> operator adds channels -> draws ROI/lines (Phase 1) -> starts analytics (Phase 2) -> alerts stream in -> video ends or operator stops (Phase 3) -> reviews alerts with replay -> removes stream (Phase 4) -> all data cleared.

---

## 11. Pipeline Comparison: DeepStream vs Custom

| Aspect | DeepStream Pipeline | Custom Pipeline |
|---|---|---|
| **Framework** | NVIDIA DeepStream SDK (GStreamer-based) | NVDEC + TensorRT + ByteTrack (manual integration) |
| **Detector** | TrafficCamNet (built-in, no finetuning needed) | YOLOv8s COCO (TensorRT INT8) |
| **Tracker** | NvDCF with ReID, `maxShadowTrackingAge=60-90` | ByteTrack on CPU with extended `track_buffer` |
| **Decode** | DeepStream internal (nvv4l2decoder) | NVDEC via Video Codec SDK or PyNvCodec |
| **Multi-channel** | `nvstreammux` batches sources natively | Manual batching of decoded frames |
| **Phase routing** | GStreamer pad probe on `frame_meta.source_id` | if/elif after inference |
| **Idle optimization** | `node.set({"interval": 15})` via pyservicemaker | Skip TensorRT call |
| **Runtime source add/remove** | Supported natively | Manual pipeline reconfiguration |
| **Learning curve** | High (GStreamer + DeepStream APIs) | Moderate (direct CUDA/TRT APIs) |
| **Flexibility** | Constrained by GStreamer plugin model | Full control over every step |
| **YouTube Live handling** | `yt-dlp` resolves URL → HLS → `nvurisrcbin` handles natively | `yt-dlp` resolves URL → HLS → manual decode |

**Decision:** Build the DeepStream pipeline first (M1-M8). Build the custom pipeline second (M9) as an alternative implementation behind the same API. Both are standalone -- they are never mixed or run simultaneously.

---

## 12. Data Contracts

### 12.1 WebSocket Messages

All messages share a `type` discriminator field.

**transit_alert (lightweight push):**
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

**stagnant_alert (lightweight push):**
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

**frame_data:**
```
type: "frame_data"
channel, frame, timestamp_ms, tracks[]
  track: id, class, bbox[4], confidence, centroid[2], trajectory[][2]
```

**stats_update:**
```
type: "stats_update"
channel, fps, active_tracks, inference_ms, dropped_frames
```

**pipeline_event:**
```
type: "pipeline_event"
event (started|stopped|error), detail
```

**track_ended:**
```
type: "track_ended"
channel, track_id, class, state (completed|stagnant|lost),
entry_direction, entry_label, exit_direction, exit_label,
method (confirmed|inferred), best_photo_url,
full_trajectory[][2], first_seen_frame, last_seen_frame,
duration_ms, frames_stationary
```

### 12.2 REST -- Full Alert Metadata

**GET /alert/{alert_id} response:**
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
  "full_trajectory": [[470, 338], [473, 340]],
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

### 12.3 Data Flow Summary

- **WebSocket push:** Lightweight alert summaries (transit and stagnant) pushed in real time for immediate UI display.
- **REST on demand:** Full metadata with per-frame bounding boxes fetched only when the user expands an alert card (`GET /alert/{alert_id}`).
- **REST historical:** `GET /alerts` for paginated alert summaries on page reload or initial load.
- **Per-frame replay data:** Stored per alert at approximately 40 bytes/frame x 300 frames = 12 KB per alert. Sent to frontend only via the REST endpoint, never via WebSocket.

---

## 13. Milestones

| Milestone | Deliverable | Dependencies |
|---|---|---|
| M1 -- DeepStream single-channel | Single-channel file input with full DeepStream pipeline: decode, TrafficCamNet detection, NvDCF tracking, ROI polygon filtering, entry/exit line-crossing, direction state machine, stagnant detection, best-photo capture, proximity-based track stitching, idle optimization. All implemented as GStreamer pad probe functions. Structured logging. 153 tests. **COMPLETE.** | None |
| M2 -- API layer | FastAPI server with REST + WebSocket + MJPEG endpoints. In-memory AlertStore for transit/stagnant alerts. PipelineBackend Protocol for pipeline-agnostic API boundary. Pipeline controllable from curl, MJPEG viewable in browser. 237 tests. **COMPLETE.** | M1 |
| M3 -- React UI | Control panel, video panels (MJPEG), alert feed sidebar, stats bar, phase controls, WebSocket integration, ROI polygon and entry/exit line drawing tools on canvas overlay, site config save/load. | M2 |
| M4 -- Phase 3 replay | `<video>` + canvas overlay for recorded video replay. Animated overlay on frozen frame for YouTube Live. `requestVideoFrameCallback()` for frame-accurate bbox rendering. | M3 |
| M5 -- Multi-channel | Single process, `nvstreammux` batching, two channels sharing one inference engine. Independent phase control per channel. | M4 |
| M6 -- YouTube Live streams | YouTube Live URL resolution via `yt-dlp`, HLS stream consumption, quality selection, stream recovery logic. Tested on live YouTube traffic camera feeds. | M5 |
| M7 -- Custom pipeline | Alternative pipeline: NVDEC + TensorRT + ByteTrack. Same API contract as DeepStream pipeline. Verified against same test videos. | M2 |
| M8 -- Polish | Remaining widgets (trajectory overlay, track count chart), annotated video export, profiling, error states in UI, graceful shutdown. | M6 |

---

## 14. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| DeepStream learning curve slows initial development (GStreamer + DS APIs are complex) | High | Medium | Budget extra time for M1. Prototype with simple pipeline first. Fall back to custom pipeline if DS proves unworkable. |
| NvDCF tracker has ID switches at busy junctions despite ReID | Medium | Medium | Proximity-based track stitching as safety net. Direction inference as fallback for unrecoverable switches. |
| TensorRT INT8 calibration degrades mAP significantly | Medium | High | Validate mAP before and after calibration. Fall back to FP16 if needed. TrafficCamNet avoids this for DS pipeline. |
| Pre-trained models miss vehicles in target junction footage (unusual angles/distances) | Medium | Medium | Evaluate TrafficCamNet and YOLOv8s COCO on all 7 target videos before committing. Finetuning deferred to v2 but is the planned escape hatch. |
| Per-frame bbox storage for replay consumes too much memory at scale | Low | Medium | 85 MB for 45 min is acceptable. Monitor actual usage. Can downsample to every Nth frame if needed. |
| Python GIL causes frame drops on 2-channel load | Low | Medium | Profile first. Move decode loop to subprocess or C extension if needed. DeepStream handles threading internally. |
| MJPEG latency unacceptable for YouTube Live sources | Low | Low | YouTube Live already has 5-15s HLS latency; MJPEG adds negligible overhead on top. WebRTC as v2 upgrade path if needed. |
| `yt-dlp` breaks due to YouTube API changes | Medium | High | Pin `yt-dlp` version, update periodically. Fallback: operator can manually extract HLS URL and paste it directly. |
| YouTube Live HLS latency (5-15s) unacceptable for alerting | Low | Low | Acceptable for traffic monitoring — alerts are for review, not real-time response. |

---

## 15. Out of Scope for v1 (Future Consideration)

- More than 2 simultaneous channels
- Night-time operation (IR / low-light video)
- Model finetuning on custom junction data
- Cross-camera vehicle re-identification
- U-turn detection
- Cloud or multi-machine deployment
- License plate or driver recognition
- Persistent database storage of alert history (v1 keeps alerts in memory only)
- JSON export of alert data
- User authentication on the UI
- Mobile UI
- RTSP IP camera sources (v1 targets YouTube Live streams only; RTSP can be added in v2)
- WebRTC video transport
- StrongSORT / BoT-SORT tracker upgrades
- C++ port of pipeline hot paths
- Confidence histogram, speed estimator, zone counter, heatmap, track timeline, headless replay, alert rules widgets
