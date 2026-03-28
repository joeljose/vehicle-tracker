# M1 Implementation Plan: DeepStream Single-Channel Pipeline

**Milestone:** M1 — DeepStream pipeline, single-channel file input
**Version target:** 0.1.0
**PRD requirements covered:** F-02, F-05, F-08, F-09, F-10, F-12, F-13, F-15, F-16, F-17, F-19, F-20, F-22, F-25, F-26, F-26a, F-39, F-40, F-41, S-06
**Design doc sections:** 3.1, 3.3, 4, 6, 7, 8, 9, 10, 11, 12, 17, 27

**Development notes:**
- All development and testing happens inside Docker containers — nothing installed on host
- Tests run via `docker compose exec backend pytest`
- Version comes from git tags (`setuptools-scm`) — no version field in any file to bump
- Release = `git tag vX.Y.Z` → GitHub Actions builds Docker image → pushes to ghcr.io

---

## Phase 0: Dev Environment Scaffolding

**Goal:** Working dev container with DeepStream, GPU access, and project structure — verify the GPU pipeline works at all before writing any application code.

**Slice:** Container boots → can run `deepstream-app` sample → confirms NVDEC, nvinfer, nvtracker all work on your RTX 4050.

**Files to create:**
```
docker-compose.dev.yml
backend/Dockerfile.dev
backend/requirements.txt
backend/__init__.py
backend/pipeline/__init__.py
backend/pipeline/deepstream/__init__.py
backend/pipeline/custom/__init__.py
backend/api/__init__.py
backend/config/__init__.py
backend/config/sites/
```

**Acceptance criteria:**
- [ ] `docker compose -f docker-compose.dev.yml up backend` starts the container with GPU access
- [ ] `nvidia-smi` inside container shows RTX 4050
- [ ] `deepstream-test1-app` sample runs successfully on one of the junction videos
- [ ] Project directory structure matches design doc Section 20 (Component Map)
- [ ] Source code bind-mounted — editing on host reflects inside container

**Estimated complexity:** S

**Risks:**
- DeepStream container image may not support your exact GPU driver version. Check compatibility matrix.
- NVIDIA Container Toolkit must be installed on host (this is the ONE host dependency — it's the bridge between Docker and GPU).

---

## Phase 0.5: Experimentation — Validate DeepStream + TrafficCamNet on Junction Footage

**Goal:** Before writing any application code, validate that the core tech stack works on your actual footage. Answer the critical unknowns early.

**Experiments to run (all inside the dev container):**

### Experiment 1: TrafficCamNet detection quality
```bash
# Run deepstream-test1 with TrafficCamNet on each junction video
# Check: does it detect vehicles? What confidence? Any misses?
deepstream-test1-app <path_to_741_73.mp4>
```
- If TrafficCamNet misses vehicles at your camera angle → try YOLOv8s ONNX via nvinfer
- If both miss → the camera angle may be too steep; need to evaluate viability

### Experiment 2: NvDCF tracker stability
```bash
# Run deepstream-test2 with NvDCF config on junction videos
# Check: do track IDs stay stable? How many ID switches?
```
- Log the ID switch rate. If > 10% of tracks get new IDs → tune `maxShadowTrackingAge`
- Try NvDeepSORT as alternative if NvDCF is poor

### Experiment 3: Frame buffer access from Python
```python
# In a probe function, try to access raw frame data:
frame = pyds.get_nvds_buf_surface(hash(info.get_buffer()), frame_meta.batch_id)
# Check: is it a valid numpy array? What shape/dtype? RGBA or NV12?
# This validates that best-photo cropping will work
```

### Experiment 4: Runtime interval change
```python
# Test if nvinfer interval can be changed at runtime:
pgie.set_property('interval', 15)
time.sleep(5)
pgie.set_property('interval', 0)
# Check: does it work without crashing? Does FPS actually change?
```

### Experiment 5: 60fps source handling
```bash
# Run a 60fps junction video through the pipeline with interval=1
# Check: does it effectively process at 30fps? Is decode still 60fps?
```

**Acceptance criteria:**
- [ ] TrafficCamNet detects vehicles in all 3 junction videos (document detection rates)
- [ ] NvDCF tracking maintains stable IDs (document ID switch rate)
- [ ] Raw frame buffer accessible from Python probe (document format: shape, dtype, color space)
- [ ] nvinfer interval is runtime-changeable (or document alternative approach)
- [ ] 60fps sources correctly skip frames for inference
- [ ] Findings documented in `experiments/` directory as markdown notes

**Estimated complexity:** M

**This phase is critical — if any experiment fails badly, it changes the plan.**

---

## Phase 1: Minimal DeepStream Pipeline — Decode + Detect + Display

**Goal:** A Python script that builds a GStreamer pipeline, decodes a junction video file, runs TrafficCamNet detection, and displays annotated frames (bboxes) — the thinnest possible DeepStream path.

**Slice:** Video file → nvstreammux → nvinfer (TrafficCamNet) → nvdsosd (draw boxes) → output to file/screen. No tracking, no analytics, no API. Just verify detection works on your junction footage.

**Files to create:**
```
backend/pipeline/deepstream/pipeline.py     # GStreamer pipeline construction
backend/pipeline/deepstream/config.py       # nvinfer config file generation
backend/models/                             # TrafficCamNet model files (downloaded in container)
config/deepstream/                          # nvinfer config, label file
```

**Acceptance criteria:**
- [ ] Pipeline loads a junction video file (e.g., `741_and_73.mp4`)
- [ ] TrafficCamNet detects vehicles — bboxes rendered on frames
- [ ] Output saved as annotated video file or displayed via fakesink + probe print
- [ ] Detection confidence threshold can be changed in config
- [ ] Frame rate normalization: 60fps source processes every other frame for inference (`interval=1`)
- [ ] Console prints detection count per frame

**Estimated complexity:** M

**Key unknowns:**
- TrafficCamNet model download and config setup inside the container
- nvinfer config file format for TrafficCamNet (cluster-mode, num-detected-classes, etc.)
- Whether TrafficCamNet performs acceptably on your junction camera angles (elevated corner pole view)

---

## Phase 2: Add NvDCF Tracker

**Goal:** Add persistent vehicle tracking with NvDCF. Each detected vehicle gets a stable ID across frames.

**Slice:** Same pipeline as Phase 1, but with `nvtracker` (NvDCF) inserted between `nvinfer` and `nvdsosd`. Bboxes now have track IDs. Console logs track lifecycle (new track, track lost).

**Files to modify:**
```
backend/pipeline/deepstream/pipeline.py     # Add nvtracker element
config/deepstream/tracker_config.yml        # NvDCF config (maxShadowTrackingAge=75, ReID features)
```

**Acceptance criteria:**
- [ ] Each vehicle gets a persistent track ID visible on the annotated output
- [ ] IDs remain stable across frames (vehicle doesn't get a new ID each frame)
- [ ] `maxShadowTrackingAge` set to 75 — occluded vehicles retain IDs for ~2.5 seconds
- [ ] Console logs: "New track #7 (car)", "Track #7 lost after 245 frames"
- [ ] Track count per frame printed to console

**Estimated complexity:** S

---

## Phase 3: Trajectory Ring Buffer + Track Lifecycle Events

**Goal:** Per-track centroid history maintained in a ring buffer. Track-ended events emitted with full trajectory data.

**Slice:** Probe function extracts per-frame centroids from tracker metadata, stores them in a ring buffer (300 entries = 10s). When a track is lost, the full trajectory is logged. This data will later feed direction detection and alerts.

**Files to create/modify:**
```
backend/pipeline/trajectory.py              # TrajectoryBuffer class (ring buffer, deque)
backend/pipeline/deepstream/probes.py       # Pad probe function: extract tracks, update trajectory
backend/pipeline/deepstream/pipeline.py     # Attach probe to nvtracker src pad
```

**Acceptance criteria:**
- [ ] `TrajectoryBuffer` stores last 300 centroids per track
- [ ] Probe function runs per-frame, extracts track_id + centroid + bbox from `NvDsObjectMeta`
- [ ] When a track is lost, console logs the full trajectory (list of centroids)
- [ ] Trajectory data includes frame_number for each entry
- [ ] Buffer correctly wraps (old entries evicted when full)

**Estimated complexity:** S

---

## Phase 4: ROI Polygon Filtering

**Goal:** Only vehicles inside a configured ROI polygon are promoted to "tracked for alerts." Vehicles outside ROI still get bboxes but are excluded from direction/alert processing.

**Slice:** Load ROI polygon from a site config JSON file. In the probe function, test each track's centroid against the polygon. Tag tracks as "in ROI" or "out of ROI." Console logs which tracks are in/out.

**Files to create/modify:**
```
backend/pipeline/roi.py                     # point_in_polygon() ray casting algorithm
backend/config/site_config.py               # Load/save site config JSON
backend/config/sites/741_73.json            # Hand-crafted initial site config for testing
backend/pipeline/deepstream/probes.py       # Add ROI check per track
```

**Acceptance criteria:**
- [ ] Site config loaded from `config/sites/741_73.json` with a manually defined ROI polygon
- [ ] `point_in_polygon()` correctly classifies centroids as inside/outside
- [ ] Probe function marks each track as roi_active=True/False
- [ ] Console logs: "Track #7: IN ROI", "Track #12: OUT OF ROI"
- [ ] Tracks outside ROI are NOT passed to downstream analytics (direction, alerts) — verified by log output
- [ ] Tracks outside ROI still get bboxes rendered by nvdsosd

**Estimated complexity:** S

---

## Phase 5: Entry/Exit Line Crossing Detection

**Goal:** Detect when a tracked vehicle's centroid crosses a configured entry or exit line. This is the foundation for direction detection.

**Slice:** Load 4 entry/exit lines from site config. In the probe, check consecutive centroids against each line using cross-product sign change. Log crossing events. No direction state machine yet — just raw crossing detection.

**Files to create/modify:**
```
backend/pipeline/direction.py               # LineSeg dataclass, check_line_crossing(), auto-calibration state
backend/pipeline/deepstream/probes.py       # Add line-crossing check per ROI-active track
backend/config/sites/741_73.json            # Add entry_exit_lines to config
```

**Acceptance criteria:**
- [ ] 4 entry/exit lines loaded from site config with labels (manually defined for 741 & 73)
- [ ] `check_line_crossing()` detects sign change of cross-product between consecutive centroids
- [ ] Console logs: "Track #7 crossed NORTH line (entry)" / "Track #7 crossed SOUTH line (exit)"
- [ ] Auto-calibration: first 5-10 crossings determine which cross-product sign means "entry" vs "exit"
- [ ] Calibration result logged: "Line NORTH calibrated: positive sign = entry"
- [ ] False crossings (noise) are minimal — centroid must actually cross the line, not oscillate near it

**Estimated complexity:** M

---

## Phase 6: Direction State Machine + Transit Events

**Goal:** Per-track state machine that transitions UNKNOWN → ENTERED(arm) → EXITED(arm) and emits transit events. First end-to-end detection of "vehicle went from North to South."

**Slice:** Wire the line-crossing events from Phase 5 into a per-track state machine. When a vehicle crosses an entry line then an exit line, emit a transit event to console. Handle edge cases: inferred direction for tracks that appear after entry line or disappear before exit line.

**Files to create/modify:**
```
backend/pipeline/direction.py               # DirectionStateMachine class, nearest_arm() for inference
backend/pipeline/deepstream/probes.py       # Update to use state machine per track
```

**Acceptance criteria:**
- [ ] Per-track state machine correctly transitions through UNKNOWN → ENTERED → EXITED
- [ ] Confirmed transit: "Track #7: NORTH → SOUTH (confirmed)" logged when both lines crossed
- [ ] Inferred transit: "Track #12: EAST → WEST (inferred)" logged when only one line crossed and direction inferred from trajectory
- [ ] `nearest_arm()` uses dot product of trajectory heading against arm direction vectors
- [ ] State machine handles edge cases: track lost before exit (infer exit), track appears after entry (infer entry)
- [ ] Vehicle state tracking: in-transit, waiting (stopped < 2.5min), stagnant (stopped > 2.5min)
- [ ] Stagnant detection: "Track #15: STAGNANT after 150s" logged

**Estimated complexity:** M

---

## Phase 7: Best-Photo Capture

**Goal:** Per-track best photo scoring and cropping. When a track ends, the highest-scoring frame crop is saved to disk.

**Slice:** In the probe, after detection, score each track's current frame (bbox_area × confidence). If better than the stored best, crop from the full-res frame and store. On track end, write JPEG to disk.

**Files to create/modify:**
```
backend/pipeline/snapshot.py                # BestPhotoTracker class (score, crop, finalize)
backend/pipeline/deepstream/probes.py       # Add best-photo scoring per ROI-active track
snapshots/                                  # Output directory for JPEG crops
```

**Acceptance criteria:**
- [ ] `BestPhotoTracker` scores each frame: `bbox_area * confidence`
- [ ] Only the highest-scoring crop is kept in memory per track (~50 KB)
- [ ] Crop is taken from full-resolution frame via `pyds.get_nvds_buf_surface()`
- [ ] On track end, JPEG written to `snapshots/{track_id}.jpg`
- [ ] Console logs: "Track #7: best photo saved (score=45230, 142x98 crop)"
- [ ] Photos are visually correct — show the vehicle, not garbage pixels

**Estimated complexity:** M

**Key unknown:**
- Accessing the raw frame buffer from DeepStream's `NvBufSurface` in Python. The `pyds.get_nvds_buf_surface()` API returns a numpy array, but the color format (RGBA vs NV12) and memory mapping need testing.

---

## Phase 8: Proximity-Based Track Stitching

**Goal:** When a new track appears near a recently lost track, merge them to preserve direction state across ID switches.

**Slice:** Maintain a buffer of recently lost tracks (last 2 seconds). When a new track appears, check proximity against all lost tracks. If match found, inherit the lost track's direction state and trajectory.

**Files to create/modify:**
```
backend/pipeline/stitch.py                  # stitch_tracks() function, lost track buffer
backend/pipeline/deepstream/probes.py       # Integrate stitching into track lifecycle
```

**Acceptance criteria:**
- [ ] Lost tracks retained for 2 seconds after loss
- [ ] New track within 100px of a lost track's predicted position triggers merge
- [ ] Merged track inherits entry_arm, direction_state, and trajectory history
- [ ] Console logs: "Track #15 merged with lost track #7 (dist=45px, gap=0.8s)"
- [ ] Direction state correctly preserved across merge (e.g., entry from NORTH still valid)

**Estimated complexity:** S

---

## Phase 9: Idle Optimization

**Goal:** Skip inference when the scene is empty to save GPU power and reduce thermal load.

**Slice:** Track the zero-detection counter. After 30 consecutive zero-detection frames, set nvinfer interval to 15. On any detection, restore full rate. Never skip while active tracks exist.

**Files to create/modify:**
```
backend/pipeline/idle.py                    # IdleOptimizer class (counter, state machine)
backend/pipeline/deepstream/probes.py       # Integrate idle optimizer, call pgie.set_property('interval', N)
backend/pipeline/deepstream/pipeline.py     # Pass pgie element reference to probe for interval control
```

**Acceptance criteria:**
- [ ] After 1 second of zero detections, inference drops to every 15th frame
- [ ] Any detection immediately restores full-rate inference
- [ ] Inference is never skipped while active tracks exist (even if one frame has 0 detections)
- [ ] Console logs: "IDLE MODE: inference interval set to 15" / "ACTIVE MODE: inference interval set to 0"
- [ ] GPU utilization visibly drops during idle periods (verify with `nvidia-smi`)

**Estimated complexity:** S

---

## Phase 10: Integration Test — Full M1 Pipeline

**Goal:** Run the complete M1 pipeline end-to-end on all 3 junction videos and verify correctness.

**Slice:** Single script that loads a video, runs the full pipeline (detect → track → ROI → line crossing → direction state machine → best photo → stitch → idle), and produces a summary report.

**Files to create/modify:**
```
backend/pipeline/deepstream/pipeline.py     # Ensure all components wired together
backend/tests/test_m1_integration.py        # Integration test script
```

**Acceptance criteria:**
- [ ] Pipeline runs on all 3 junction videos without crashes
- [ ] Transit events logged for vehicles passing through each junction
- [ ] Best photos saved and visually correct
- [ ] ROI filtering working — distant vehicles excluded
- [ ] Line polarity auto-calibrated correctly for each junction
- [ ] Stagnant detection fires for the repair truck at Drugmart & 73
- [ ] Idle mode engages during empty-road periods
- [ ] Summary report: total tracks, total transits (confirmed/inferred), total stagnant, processing FPS
- [ ] No memory leaks over a 10-minute video (monitor RSS)

**Estimated complexity:** M

---

## Risk Register

| Risk | Phase | Impact | Mitigation |
|---|---|---|---|
| DeepStream container doesn't work with host GPU driver | Phase 0 | Blocking | Check driver compat matrix first. May need driver update on host. |
| TrafficCamNet misses vehicles at elevated junction camera angle | Phase 1 | High | Test immediately. If poor, swap to YOLOv8s ONNX via nvinfer custom parser. |
| `pyds.get_nvds_buf_surface()` returns wrong format/crashes | Phase 7 | Medium | Test with small video first. Alternative: use nvvideoconvert to force RGBA before probe. |
| NvDCF tracker config not optimal — too many ID switches | Phase 2 | Medium | Tune `maxShadowTrackingAge`, `minIouDiff4NewTarget`. Log IDSW rate. |
| Auto-calibration fails (not enough vehicles cross during Phase 1) | Phase 5 | Low | Fall back to manual polarity config in site JSON. |
| nvinfer `interval` property can't be changed at runtime in this DS version | Phase 9 | Low | Verify in Phase 0. Fallback: skip inference in probe function instead. |

---

## Dependency Graph

```
Phase 0 (Dev Environment)
    |
    v
Phase 1 (Decode + Detect)
    |
    v
Phase 2 (NvDCF Tracker)
    |
    v
Phase 3 (Trajectory Buffer)
    |
    +--------+
    |        |
    v        v
Phase 4    Phase 7
(ROI)      (Best Photo)
    |
    v
Phase 5 (Line Crossing)
    |
    v
Phase 6 (Direction State Machine)
    |
    +--------+
    |        |
    v        v
Phase 8    Phase 9
(Stitch)   (Idle)
    |        |
    +--------+
    |
    v
Phase 10 (Integration Test)
```

Note: Phase 7 (Best Photo) and Phase 4 (ROI) can be developed in parallel after Phase 3. Phase 8 (Stitch) and Phase 9 (Idle) can be developed in parallel after Phase 6. The dependency graph allows some parallelism if needed.
