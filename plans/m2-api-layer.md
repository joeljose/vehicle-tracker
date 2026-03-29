# M2 Implementation Plan: API Layer

**Milestone:** M2 — FastAPI REST + WebSocket + MJPEG endpoints
**Version target:** 0.2.0
**PRD requirements covered:** F-44 through F-74 (API endpoints, real-time streaming, alert management, site config)
**Design doc sections:** 3.3, 13.1, 18, 19, 20

**Development notes:**
- All development and testing inside Docker containers
- Tests run via `./dev.sh exec backend pytest backend/tests/`
- FakeBackend used for all tests — no GPU required for API layer testing
- Version from git tags (`setuptools-scm`)

---

## Phase 0: Square Crop Redesign

**Goal:** Redesign best-photo crop to be square with 50% vehicle fill, preparing for snapshot API serving.

**Issue:** #27

**Files modified:**
```
backend/pipeline/snapshot.py     (square crop logic)
backend/tests/test_snapshot.py   (updated tests)
```

---

## Phase 1: PipelineBackend Protocol + AlertStore

**Goal:** Define the pipeline-agnostic interface contract and in-memory alert storage.

**Issue:** #20

**Files created:**
```
backend/pipeline/protocol.py     (PipelineBackend Protocol, ChannelPhase, Detection, FrameResult)
backend/pipeline/alerts.py       (AlertStore — transit + stagnant alerts, summary/full access)
backend/tests/test_alerts.py     (17 tests for AlertStore)
```

**Key decisions:**
- Protocol with structural typing (no ABC) — both backends implement it independently
- AlertStore separates summary (for WS push/listing) from full metadata (for on-demand detail)
- Two access patterns: lightweight summaries exclude per_frame_data and full_trajectory

---

## Phase 2: FastAPI App + Pipeline Control Endpoints

**Goal:** FastAPI app factory with pipeline lifecycle control via REST.

**Issue:** #21

**Files created:**
```
backend/main.py                  (create_app factory with lifespan)
backend/api/models.py            (Pydantic request/response models)
backend/api/deps.py              (dependency injection: get_backend, get_alert_store)
backend/api/routes.py            (6 REST endpoints)
backend/pipeline/fake.py         (FakeBackend for testing)
backend/tests/test_api.py        (20 tests)
```

**Endpoints:**
- `POST /pipeline/start` — boot pipeline (409 if already started)
- `POST /pipeline/stop` — stop pipeline, reset state (409 if not started)
- `POST /channel/add` — add video source, returns auto-assigned channel_id
- `POST /channel/remove` — remove channel, clear its alerts
- `POST /channel/{id}/phase` — transition phase (setup/analytics/review)
- `PATCH /config` — update confidence threshold, inference interval

**Key decisions:**
- Server-side auto-increment channel IDs (reset on pipeline stop)
- Conditional imports: DeepStream only imported when backend="deepstream"
- FakeBackend raises same exceptions as real pipeline for consistent error handling

---

## Phase 3: MJPEG Streaming Endpoint

**Goal:** Live annotated video viewable in browser via `<img src="/stream/0">`.

**Issue:** #22

**Files created:**
```
backend/api/mjpeg.py             (MjpegBroadcaster + GET /stream/{channel_id})
```

**Files modified:**
```
backend/main.py                  (wire broadcaster, register frame callback)
backend/pipeline/fake.py         (add emit_frame() test helper)
backend/tests/test_api.py        (9 new tests — broadcaster unit tests + endpoint error cases)
```

**Key decisions:**
- Thread-safe `queue.Queue` per subscriber (not asyncio.Queue) for cross-thread safety
- `asyncio.to_thread` for non-blocking queue reads in async generator
- Fan-out pattern: one queue per HTTP client, frames copied to all
- Drop-oldest policy when queue full (maxsize=2) to keep stream current

---

## Phase 4: WebSocket Broadcaster

**Goal:** Real-time data push to browser clients with channel filtering.

**Issue:** #23

**Files created:**
```
backend/api/websocket.py         (WsBroadcaster + GET /ws endpoint)
```

**Files modified:**
```
backend/main.py                  (wire WS broadcaster, register alert/track_ended callbacks)
backend/api/routes.py            (emit pipeline_event on start/stop)
backend/pipeline/fake.py         (add emit_alert(), emit_track_ended() helpers)
backend/tests/test_api.py        (9 new tests — broadcaster unit tests + WS connection)
```

**Message types:**
- `frame_data` — per frame, per channel (~30/s)
- `pipeline_event` — started/stopped (emitted from route handlers)
- `transit_alert` / `stagnant_alert` — lightweight alert summaries
- `track_ended` — full track summary on track loss

**Key decisions:**
- Background asyncio task drains thread-safe queue and broadcasts
- Channel filtering via `?channels=0,1` query parameter
- Messages without a channel field (pipeline_event) go to all clients
- Dead clients auto-cleaned on send failure

---

## Phase 5: Alert + Snapshot REST Endpoints

**Goal:** REST endpoints for querying alerts and serving best-photo snapshots.

**Issue:** #24

**Files modified:**
```
backend/api/routes.py            (3 new endpoints)
```

**Files created:**
```
backend/tests/test_api_alerts.py (11 tests)
```

**Endpoints:**
- `GET /alerts?limit=50&type=transit_alert` — paginated alert summaries
- `GET /alert/{alert_id}` — full metadata with per_frame_data
- `GET /snapshot/{track_id}` — serve JPEG (image/jpeg content-type)

---

## Phase 6: Site Config REST Endpoints

**Goal:** CRUD for site configurations (ROI polygon, entry/exit lines).

**Issue:** #25

**Files modified:**
```
backend/api/routes.py            (3 new endpoints)
```

**Files created:**
```
backend/tests/test_api_site.py   (8 tests)
```

**Endpoints:**
- `POST /site/config` — save config (validates ROI >= 3 vertices)
- `GET /site/config?site_id=741_73` — load config
- `GET /site/configs` — list all saved site IDs

**Key decisions:**
- Access `SITES_DIR` via module attribute (not default arg) for monkeypatch-friendly testing
- Reuse existing `SiteConfig` dataclass and `load_site_config`/`save_site_config` functions

---

## Phase 7: Integration Test

**Goal:** End-to-end validation of the full API surface.

**Issue:** #26

**Files created:**
```
backend/tests/test_m2_integration.py  (3 tests)
```

**Tests:**
- `test_full_api_lifecycle` — start → add channels → analytics → emit frames/alerts/tracks → REST queries → snapshot → remove → stop
- `test_site_config_round_trip` — save → load → list
- `test_multi_channel_isolation` — frames and alerts don't leak across channels

---

## Final State

- **17 new files** created across backend/api/, backend/pipeline/, and backend/tests/
- **237 tests** total (177 M1 + 60 M2)
- **All 8 GitHub issues** closed (#20-#27)
- **Zero new lint errors** in new code
