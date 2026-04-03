# Experiment 6: Async Pipeline (Non-blocking start/stop)

## Setup
- Pipeline: `nvurisrcbin` → `nvstreammux` → `nvinfer` → `nvtracker` → `fakesink` (manual Pipeline API)
- SimpleReporter `BatchMetadataOperator` counts frames/detections, records callback thread name
- Video: `741_73_10s.mp4` (10-second clip)
- Config: project's `PGIE_CONFIG`, `TRACKER_CONFIG` (imported from `backend.pipeline.deepstream.config`)

## Tests

| # | Test | What it verifies | Verdict |
|---|------|-----------------|---------|
| 1 | `pipeline.start()` returns immediately | Non-blocking — main thread is not held | PASS |
| 2 | Probe callbacks fire in background | Callback thread ≠ main thread (GStreamer streaming thread) | PASS |
| 3 | Main thread reads while pipeline runs | Frame count increases while main thread polls — no deadlock | PASS |
| 4 | `pipeline.stop()` graceful shutdown | No frames arrive after stop, returns quickly | PASS |
| 5 | Start new pipeline after stopping | Second pipeline produces frames — no resource leaks | PASS |

## Findings
1. `pipeline.start()` is non-blocking — returns in <1s. GStreamer spawns its own streaming thread(s).
2. Probe callbacks (`BatchMetadataOperator.handle_metadata`) fire on the GStreamer streaming thread, not the main thread. This means FastAPI's asyncio loop and GStreamer don't compete for the same thread.
3. Cross-thread reads are safe for simple counters (frame_count, detection_count). Production code uses `threading.Lock` for complex shared state (AlertStore).
4. `pipeline.stop()` is clean — no orphaned threads, no lingering callbacks.
5. Pipelines are reusable — can build and start a fresh pipeline after stopping a previous one. No GPU resource leaks.

## Key takeaway
The threading model confirms the M4 architecture: FastAPI runs on the asyncio event loop, GStreamer probes fire on streaming threads, and they communicate via `call_soon_threadsafe()`. No need for multiprocessing or subprocess isolation.
