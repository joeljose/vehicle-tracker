# M3 Implementation Plan: React UI

**Milestone:** M3 — React UI for pipeline control, monitoring, and alert review
**Version target:** 0.3.0
**PRD requirements covered:** F-75 through F-99 (home page, channel tab, drawing tools, alert feed, replay, stats)
**Design doc sections:** 14, 15, 16, 20

**Development notes:**
- All development and testing happens inside Docker containers — nothing installed on host
- Frontend dev container: `node:22-alpine` with Vite dev server on port 5173
- Tests run via `make test` (286 tests total at milestone completion)
- HTML mockups reviewed and approved before coding React components

---

## Phase 0: Backend Prerequisites

**Goal:** Add backend endpoints and WebSocket support needed by the React UI before writing any frontend code.
**Slice:** Channel phase transitions, MJPEG streaming endpoint, WebSocket with server-side type filtering — all testable via curl/wscat before any UI exists.
**Files created:**
- Backend endpoint additions for channel phase management and WebSocket filtering

**Acceptance criteria:**
- [x] `POST /channel/{id}/phase` endpoint transitions channels through Setup -> Analytics -> Review
- [x] `GET /stream/{channel_id}` serves MJPEG annotated video
- [x] WebSocket accepts `types` filter parameter to exclude `frame_data` on UI connections
- [x] Phase transition emits `phase_changed` WebSocket event
- [x] All new endpoints covered by tests

**Estimated complexity:** M

---

## Phase 1: Frontend Scaffold

**Goal:** Vite + React project with routing, context providers, and API layer — the skeleton that all UI phases build on.
**Slice:** `make dev` starts the frontend container, browser loads at localhost:5173, React Router navigates between Home and Channel pages (empty shells), API module talks to backend.
**Files created:**
- `frontend/src/main.jsx`, `frontend/src/App.jsx`
- `frontend/src/api/rest.js`, `frontend/src/api/ws.js`
- `frontend/src/contexts/HomeContext.jsx`, `frontend/src/contexts/ChannelContext.jsx`
- `frontend/src/lib/utils.js`
- Vite config with `/api` proxy to backend

**Acceptance criteria:**
- [x] `make dev` starts frontend container with hot-reload
- [x] React Router routes `/` to Home and `/channel/:id` to Channel page
- [x] `rest.js` wraps all backend API calls with error handling
- [x] `ws.js` manages WebSocket connection with auto-reconnect and type filtering
- [x] `HomeContext` and `ChannelContext` provide state to their respective page trees
- [x] Vite proxies `/api` requests to backend on port 8000

**Estimated complexity:** M

---

## Phase 1.5: UI Mockups for Design Approval

**Goal:** Static HTML mockups of every UI view for visual approval before writing React components.
**Slice:** Standalone HTML files with inline CSS demonstrating layout, color palette, and component structure for each phase of the channel workflow.
**Files created:**
- `frontend/mockups/home.html`
- `frontend/mockups/channel-setup.html`
- `frontend/mockups/channel-analytics.html`
- `frontend/mockups/channel-review.html`
- `frontend/mockups/alert-card.html`

**Acceptance criteria:**
- [x] Zinc color palette (dark theme) applied consistently across all mockups
- [x] Home page mockup shows pipeline controls and channel card grid
- [x] Channel mockups show phase indicator, video panel, and sidebar layout
- [x] Alert card mockup shows compact format: thumbnail, direction, track ID, timestamp
- [x] Stagnant alerts visually distinct with amber highlight vs blue for transit
- [x] Mockups reviewed and approved before proceeding to React implementation

**Estimated complexity:** S

---

## Phase 2: Home Page

**Goal:** Working home page with pipeline start/stop controls and channel management (add/remove sources).
**Slice:** Operator can start the pipeline, add a video source channel, see it appear as a card, click it to open the channel tab, and remove it.
**Files created:**
- `frontend/src/pages/Home.jsx`
- `frontend/src/components/ChannelCard.jsx`
- `frontend/src/components/ConfirmModal.jsx`
- `frontend/src/components/Toast.jsx`

**Acceptance criteria:**
- [x] Start/Stop pipeline buttons call backend and reflect pipeline state
- [x] "Add Channel" accepts YouTube Live URL or file path and calls `POST /channel/add`
- [x] Channel cards display source name, phase, and alert count
- [x] Clicking a channel card opens `/channel/:id` in a new browser tab
- [x] Remove channel button with confirmation modal calls `POST /channel/remove`
- [x] Home WebSocket subscribes to `pipeline_event` and `phase_changed` for live updates
- [x] Toast notifications for errors and confirmations

**Estimated complexity:** M

---

## Phase 3: Channel Tab — Setup Phase

**Goal:** Channel page with MJPEG video panel, phase indicator, and Setup phase UI where the operator views the live feed before drawing ROI/lines.
**Slice:** Operator opens a channel tab, sees the MJPEG stream in the video panel, phase indicator shows "Setup", confidence threshold slider adjusts detection in real time.
**Files created:**
- `frontend/src/pages/Channel.jsx`

**Acceptance criteria:**
- [x] MJPEG video rendered via `<img>` tag from `GET /stream/{channel_id}`
- [x] Phase indicator shows current phase (Setup / Analytics / Review) as top-level navigation
- [x] Confidence threshold slider sends `PATCH /config` on change
- [x] Channel WebSocket connects with `types` filter (no `frame_data`)
- [x] Video panel uses `object-fit: contain` for correct aspect ratio
- [x] Auto-reconnect on WebSocket drop

**Estimated complexity:** M

---

## Phase 4: Drawing Tools — ROI Polygon + Entry/Exit Lines

**Goal:** Transparent canvas overlay on the video panel for drawing ROI polygon and entry/exit lines during Setup phase.
**Slice:** Operator draws a polygon ROI and 4 entry/exit lines on the video feed, coordinates stored in original pixel space, then submits to transition to Analytics phase.
**Files created:**
- `frontend/src/components/DrawingCanvas.jsx`
- `frontend/src/components/DrawingTools.jsx`
- `frontend/src/components/LinesOverlay.jsx`
- `frontend/src/lib/coords.js`

**Acceptance criteria:**
- [x] Canvas overlay renders on top of video panel with correct coordinate mapping
- [x] ROI polygon drawn by clicking vertices, closed by clicking first point
- [x] Entry/exit lines drawn as labeled line segments with directional indicators
- [x] `coords.js` converts between display coordinates and original pixel space via `object-fit:contain` geometry
- [x] ResizeObserver redraws overlay on window resize
- [x] "Start Analytics" button sends ROI + lines via `POST /channel/{id}/phase` and transitions to Analytics
- [x] Lines displayed as read-only overlay during Analytics and Review phases

**Estimated complexity:** L

---

## Phase 5: Analytics Phase — Alert Feed + Stats Bar

**Goal:** Live monitoring UI during Analytics phase with real-time alert feed sidebar and pipeline stats bar.
**Slice:** Pipeline runs analytics, alerts stream in via WebSocket and populate the sidebar, stats bar shows fps/tracks/latency updated every second.
**Files created:**
- `frontend/src/components/AlertFeed.jsx`
- `frontend/src/components/StatsBar.jsx`
- `frontend/src/contexts/AlertContext.jsx`

**Acceptance criteria:**
- [x] Alert feed sidebar (~300px wide) displays compact alert cards as they arrive via WebSocket
- [x] Cards show: square thumbnail, direction labels, track ID, duration, method, relative timestamp
- [x] Transit alerts styled blue, stagnant alerts styled amber
- [x] Filter buttons (All / Transit / Stagnant) with client-side filtering
- [x] Auto-scroll to newest alert when at top; "New alerts" badge when scrolled away
- [x] Stats bar below video shows fps, active tracks, inference latency from `stats_update` messages
- [x] Alert cards are read-only (no click interaction) during Analytics phase

**Estimated complexity:** M

---

## Phase 6: Replay Backend — Clip Extraction + Replay Endpoint

**Goal:** Backend support for alert replay: extract video clips around transit events and serve them for the Review phase UI.
**Slice:** When Analytics -> Review transition happens, background task extracts short clips for each transit alert. `GET /alert/{id}/replay` returns the clip (or 202 if still extracting).
**Files created:**
- `backend/pipeline/clip_extractor.py`

**Acceptance criteria:**
- [x] Phase transition to Review triggers background clip extraction for all transit alerts
- [x] `clip_extractor.py` uses ffmpeg to cut clips from source video around each transit event
- [x] `GET /alert/{alert_id}/replay` returns extracted clip or 202 if not ready
- [x] Stagnant alert replay returns full frame with bbox as static JPEG
- [x] Clips cleaned up when channel is removed
- [x] Extraction progress does not block the API server

**Estimated complexity:** M

---

## Phase 7: Review Phase — Replay UI + Clickable Alert Feed

**Goal:** Review phase UI where the operator clicks alerts to replay them with bounding box overlays on the video panel.
**Slice:** Operator clicks a transit alert card, video panel switches to looping clip with canvas-drawn bbox overlay synced to playback. Stagnant alerts show static frame with bbox.
**Files created:**
- `frontend/src/components/ReplayView.jsx`

**Acceptance criteria:**
- [x] Alert cards become clickable when entering Review phase
- [x] Clicking a transit alert loads the pre-extracted clip in a `<video>` element with loop
- [x] `<canvas>` overlay draws bounding box synced to video playback via `requestVideoFrameCallback()`
- [x] Spinner shown while clip is still being extracted (202 response)
- [x] Clicking a stagnant alert shows full frame with bbox overlay as static image
- [x] Phase indicator reflects Review state
- [x] Entry/exit lines remain visible as read-only overlay during replay

**Estimated complexity:** L

---
