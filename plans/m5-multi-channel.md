# M5 Implementation Plan: Multi-Channel Shared Pipeline

**Milestone:** M5 — Multi-channel shared pipeline
**Version target:** 0.5.0
**Tests at completion:** 336

**Development notes:**
- All development and testing happens inside Docker containers — nothing installed on host
- Tests run via `make test`
- Version comes from git tags (`setuptools-scm`) — no version field in any file to bump

---

## Phase 0: Infrastructure & Pre-work

**Goal:** Harden the dev workflow and fix outstanding E2E bugs before starting shared pipeline work.

**Key files:**
- `Makefile` (19 dev targets)
- `backend/Dockerfile` (baked-in deps)
- `README.md` (setup guide, dev workflow, project structure)
- `CHANGELOG.md`

**Acceptance criteria:**
- [x] Makefile with dev targets replaces raw `docker compose` calls
- [x] Backend Dockerfile bakes in all pip/apt deps (no install-on-start)
- [x] README rewritten with setup guide, dev workflow, project structure
- [x] CHANGELOG added
- [x] E2E bugs fixed: color swap, frame interleave, replay sync, direction inference, thumbnails, lines overlay
- [x] Replay overlay and best-photo snapshots fixed
- [x] `make start` uses detached exec so server persists after shell exit

**Estimated complexity:** M

---

## Phase 1: BatchMetadataRouter & Shared Pipeline Topology

**Goal:** Replace per-channel separate pipelines with a single shared pipeline: nvstreammux -> nvinfer -> nvtracker -> nvstreamdemux -> per-channel OSD/MJPEG. Route batch metadata back to the correct source channel.

**Key files:**
- `backend/pipeline/deepstream/batch_router.py` (BatchMetadataRouter — new)
- `backend/pipeline/deepstream/adapter.py` (refactored for shared pipeline)
- `backend/pipeline/deepstream/pipeline.py` (shared pipeline topology)

**Acceptance criteria:**
- [x] BatchMetadataRouter dispatches per-source metadata from batched nvinfer/nvtracker output
- [x] Single shared pipeline processes multiple sources through one nvinfer + nvtracker instance
- [x] nvstreamdemux fans out to per-channel OSD and MJPEG encoding
- [x] Independent phase transitions per channel (one channel can be in counting while another is in setup)
- [x] Per-source EOS detection via frame-gap watchdog (no global EOS)
- [x] 2-channel E2E validated on RTX 4050 (~459 MB shared vs ~738 MB separate)

**Estimated complexity:** L

---

## Phase 2: Dynamic Source Add/Remove

**Goal:** Allow channels to be added and removed at runtime without restarting the application.

**Key architectural decision:** nvurisrcbin cannot be dynamically added to a running GStreamer pipeline in a synchronized way. Solution: `_rebuild_shared_pipeline()` tears down and rebuilds the entire shared pipeline whenever the channel set changes. Existing channels resume from where they left off.

**Key files:**
- `backend/pipeline/deepstream/adapter.py` (`_rebuild_shared_pipeline()`)

**Acceptance criteria:**
- [x] Adding a new channel triggers pipeline rebuild with all active sources
- [x] Removing a channel triggers pipeline rebuild without that source
- [x] Existing channels continue processing after rebuild (state preserved)
- [x] Per-channel snapshot directories created on add, cleaned up on remove
- [x] Snapshots skipped during pipeline setup phase (avoid capturing garbage frames)

**Estimated complexity:** L

---

## Phase 3: Bug Fixes & Polish

**Goal:** Fix UI and backend issues found during multi-channel testing.

**Key files:**
- Various frontend and backend files (snapshots, replay, scroll, stop button, tab close)

**Acceptance criteria:**
- [x] Snapshot saving works correctly per-channel
- [x] Replay playback works across channels
- [x] UI scroll behavior correct with multiple channel tabs
- [x] Stop button works per-channel without affecting other channels
- [x] Tab close properly removes channel and triggers pipeline rebuild
- [x] Docs updated for v0.5.0 release

**Estimated complexity:** M

---

## Summary

| Phase | Description | Complexity |
|-------|-------------|------------|
| P0 | Infrastructure & pre-work | M |
| P1 | BatchMetadataRouter & shared pipeline | L |
| P2 | Dynamic source add/remove | L |
| P3 | Bug fixes & polish | M |

**Key result:** Multi-channel shared pipeline reduces VRAM from ~738 MB (separate pipelines) to ~459 MB (shared) for 2 channels on RTX 4050, with a single nvinfer + nvtracker instance serving all sources.
