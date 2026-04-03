# M4 Implementation Plan: DeepStream-FastAPI Integration

**Milestone:** M4 — DeepStream-FastAPI integration
**Version target:** 0.4.0
**PRD requirements covered:** DeepStream pipeline adapter implementing PipelineBackend Protocol, per-channel lifecycle, MJPEG streaming, real-time alerts, clip extraction
**GitHub issues:** #48 (umbrella), #49-#53 (phases 1-5)

**Development notes:**
- All development and testing happens inside Docker containers
- Tests run via `make test`
- Version comes from git tags (`setuptools-scm`)
- Threading model: pipeline.start() is non-blocking, probe callbacks fire on GStreamer thread, bridge to asyncio via call_soon_threadsafe()

---

## Phase 1: DeepStreamPipeline Adapter + Drawing Offset Fix

**Goal:** Create the DeepStreamPipeline adapter class implementing PipelineBackend Protocol, wire it into FastAPI.

**Issue:** #49

**Key files:**
```
backend/pipeline/deepstream/adapter.py    (main adapter — PipelineBackend Protocol impl)
backend/pipeline/deepstream/pipeline.py   (GStreamer pipeline construction + lifecycle)
backend/pipeline/deepstream/batch_router.py (demux frames by channel)
backend/pipeline/deepstream/config.py     (pipeline configuration)
```

**Key decisions:**
- Adapter wraps per-channel GStreamer pipelines behind the Protocol interface
- Per-channel pipeline lifecycle: preview → analytics → review
- preview pipeline: detect + track + OSD + MJPEG, loops with sync=True, no ROI
- Analytics pipeline: adds ROI/lines + alert emission, plays once to EOS

**Acceptance criteria:**
- [x] DeepStreamPipeline implements PipelineBackend Protocol
- [x] add_channel() creates preview pipeline with looping playback
- [x] remove_channel() stops and destroys pipeline
- [x] Drawing offset fix applied for correct OSD annotation positioning

**Estimated complexity:** L

---

## Phase 2: MJPEG Extraction + Phase Transitions

**Goal:** GPU-to-JPEG frame extraction for live MJPEG streaming, and configure_channel Protocol method for phase transitions.

**Issue:** #50

**Key files:**
```
backend/pipeline/deepstream/adapter.py    (MjpegExtractor BufferOperator)
backend/pipeline/deepstream/pipeline.py   (pipeline rebuild on phase change)
```

**Key decisions:**
- MjpegExtractor as BufferOperator after nvdsosd: cupy.asnumpy() → cv2.imencode('.jpg') at ~15fps
- Skip every other frame from 30fps inference (~4ms per frame, negligible overhead)
- configure_channel() Protocol method stores ROI + lines before analytics transition
- Frontend sends ROI/lines with setup→analytics phase transition request

**Acceptance criteria:**
- [x] MjpegExtractor produces JPEG frames at ~15fps from GPU buffers
- [x] configure_channel() Protocol method added and wired
- [x] ROI and entry/exit lines sent with phase transition
- [x] Phase transitions rebuild pipeline with correct configuration

**Estimated complexity:** M

---

## Phase 3: Real-Time Alert Delivery + Stagnant Alerts

**Goal:** Wire alert callbacks from DeepStream probe to FastAPI WebSocket broadcaster.

**Issue:** #51

**Key files:**
```
backend/pipeline/deepstream/adapter.py    (alert callback registration)
backend/pipeline/alerts.py                (AlertStore with threading.Lock)
```

**Key decisions:**
- AlertStore uses threading.Lock for cross-thread safety (GStreamer thread → asyncio)
- Bridge via call_soon_threadsafe() — no ThreadPoolExecutor needed
- TrackingReporter stagnant alert emission added (M1/M2 gap fix)

**Acceptance criteria:**
- [x] Transit alerts delivered in real-time via callbacks
- [x] Stagnant alerts emitted (gap fix from M1/M2)
- [x] Thread-safe alert delivery from GStreamer thread to asyncio

**Estimated complexity:** M

---

## Phase 4: Clip Extraction + File Validation

**Goal:** Extract video clips for alert replay using ffmpeg, add file validation.

**Issue:** #52

**Key files:**
```
backend/pipeline/clip_extractor.py        (ffmpeg-based clip extraction)
```

**Key decisions:**
- ffmpeg subprocess for clip extraction (reliable, handles all codecs)
- File validation on input sources

**Acceptance criteria:**
- [x] Clip extraction via ffmpeg for alert replay
- [x] File validation with proper error handling
- [x] Loading spinner bug fixed

**Estimated complexity:** S

---

## Phase 5: EOS Auto-Transition + Cleanup

**Goal:** Automatic transition from analytics to review phase on end-of-stream, plus test suite cleanup.

**Issue:** #53

**Key files:**
```
backend/pipeline/deepstream/adapter.py    (EOS handler + phase callback)
backend/pipeline/deepstream/pipeline.py   (GStreamer EOS message handling)
```

**Key decisions:**
- GStreamer bus message callback detects EOS and triggers phase transition
- Phase callback notifies FastAPI layer of auto-transition
- Alert frame metadata attached for review phase context

**Acceptance criteria:**
- [x] EOS auto-transitions channel from analytics to review phase
- [x] Phase callback fires on auto-transition
- [x] Alert frame metadata preserved through transition
- [x] Pipeline stop crash fix (clean shutdown on EOS)
- [x] Test suite cleanup: removed duplicate, fixed comments, strengthened assertions

**Estimated complexity:** M

---

## Final State

- **5 phases** completed across 7 commits
- **312 tests** total (237 M2 + 75 M4)
- **All 6 GitHub issues** closed (#48-#53)
- **Key architectural pattern:** per-channel pipeline lifecycle (preview → analytics → review) behind Protocol interface
- PRD and design doc updated with M4 architecture decisions from grill session
