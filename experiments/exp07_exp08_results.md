# Experiments 07-08: Multi-Source Shared Inference (M5 Viability)

Date: 2026-03-30/31

## Summary

**M5 shared-pipeline architecture is VIABLE.** All critical assumptions validated.

## Architecture Validated

```
src_0 (loop/once) ──┐                                                   ┌→ [ch0] OSD → caps → MjpegExtractor
                    ├→ nvstreammux → nvinfer → nvtracker → demux ────────┤
src_1 (loop/once) ──┘       (shared inference + tracking)                └→ [ch1] OSD → caps → MjpegExtractor
                                    │
                           [BatchMetadataRouter probe]
                           routes frame_meta by source_id
                           to per-channel TrackingReporter
```

## Experiment 07: Core Pipeline Tests

| Test | Question | Result |
|------|----------|--------|
| 1. Single source VRAM | Baseline | +369 MB |
| 2. Two sources shared | Batching + source_id routing | **PASS** — +459 MB (+90 MB marginal) |
| 3. Per-source EOS | One source ends, other continues? | **PASS** — src1 ran to 1800f after src0 stopped at 258f |
| 4. Mixed loop | Loop + non-loop in same pipeline | **PASS** — src0=1910f (looping), src1=600f (once) |

### VRAM Savings
- 2× separate pipelines: ~738 MB
- 1× shared pipeline: ~459 MB
- **Savings: ~280 MB (38%)**

### Key Findings
- `source_id` routing works natively in `frame_meta`
- NvDCF tracks independently per source (zero track ID overlap)
- `nvstreammux` handles per-source EOS correctly
- `nvurisrcbin` `file-loop` is per-source (can mix loop/non-loop)

## Experiment 08: Batch Routing + Frame Extraction

### Problem: BufferOperator stalls on batch_size > 1

Isolation tests (exp08b):
- No tee: 439 batches ✓
- Tee + fakesinks: 222 batches ✓
- Tee + OSD (no BufferOperator): 441 batches ✓
- Tee + OSD + BufferOperator: **3 batches ✗ (stall)**

BufferOperator (C++ pybind11) cannot handle batched GPU buffers. Even a no-op handle_buffer() stalls the pipeline.

### Solution: nvstreamdemux

Split the batched stream back to per-channel (batch_size=1) streams before BufferOperator probes:

```
tracker → nvstreamdemux → [ch0: queue → OSD → caps → MjpegExtractor (batch_size=1)]
                        → [ch1: queue → OSD → caps → MjpegExtractor (batch_size=1)]
```

### Validated with Actual Adapter Probes

Using the real `MjpegExtractor` and `FrameExtractor` classes from the adapter:

```
t=10s: batches=466, ch0=92 MJPEG frames, ch1=107 MJPEG frames — PASS
```

- Shared inference at ~46 batches/sec
- Per-channel MJPEG extraction at ~10fps each
- Both channels flowing continuously

## Architecture for M5

### Shared Part (single instance)
- `nvstreammux` (batch-size=N)
- `nvinfer` (shared TrafficCamNet engine)
- `nvtracker` (shared NvDCF, tracks per source_id)
- `BatchMetadataRouter` probe: routes `frame_meta` by `source_id` to per-channel `TrackingReporter`

### Per-Channel Part (after nvstreamdemux)
- `nvdsosd` → `nvvideoconvert` → `capsfilter` → `fakesink`
- `MjpegExtractor` probe (batch_size=1, existing code works unchanged)
- `FrameExtractor` probe for best-photo snapshots

### Per-Channel State (unchanged from M4)
- `TrackingReporter` (ROI, direction, alerts, per_frame_data)
- `BestPhotoTracker`
- `DirectionStateMachine` per track
- Sequential track ID mapping

### Phase Transitions
- **add_channel**: Add `nvurisrcbin(file-loop=True)` → mux. Add demux output → OSD chain.
- **Setup→Analytics**: Remove looping source, add same URI with `file-loop=False`. Reinit Reporter with ROI/lines.
- **Analytics EOS**: Per-source EOS detected. Remove source from mux. Finalize tracks.
- **remove_channel**: Remove source + demux output. Clean up state.
