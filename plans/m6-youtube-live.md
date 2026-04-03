# M6 Implementation Plan: YouTube Live Streams

**Milestone:** M6 — YouTube Live streams
**Version target:** 0.6.0
**Tests at completion:** 371

**Development notes:**
- All development and testing happens inside Docker containers
- Tests run via `make test`

**Key design decisions (from grill session):**
- No quality selection — always pick best available stream
- Async URL resolution: POST returns 202, client gets `channel_added` via WebSocket
- Circuit breaker: 3 pipeline rebuilds within 10 min → eject channel
- Serialized yt-dlp calls via `asyncio.Lock` (yt-dlp is not concurrent-safe)
- Last frame buffer persisted at Phase 2→3 transition for frozen-frame replay

---

## Phase 0: Validation Spike

**Goal:** Confirm that DeepStream's `nvurisrcbin` can consume YouTube Live HLS streams natively, so we don't need a custom demuxer or relay proxy.

**Key files:** (exploratory — no production code)

**Acceptance criteria:**
- [x] `nvurisrcbin` accepts a YouTube HLS `.m3u8` URL and produces decoded frames
- [x] Pipeline runs stable for 60+ seconds on a live YouTube stream
- [x] Confirmed: yt-dlp can extract the HLS URL from a standard `youtube.com/watch?v=` link

**Estimated complexity:** S

---

## Phase 1: Source Resolver + Async Channel Add

**Goal:** Build the `source_resolver` module that extracts HLS URLs from YouTube links via yt-dlp, and wire up async channel addition (202 + WebSocket notification).

**Key files:**
- `backend/pipeline/source_resolver.py`
- `backend/Dockerfile` (yt-dlp + deno binaries added)
- `backend/pipeline/deepstream/adapter.py` (YouTube integration)

**Acceptance criteria:**
- [x] `source_resolver.py` accepts a YouTube URL and returns an HLS `.m3u8` URL via yt-dlp
- [x] yt-dlp and deno binaries baked into the Docker image
- [x] POST `/api/channels` with a YouTube URL returns 202 Accepted
- [x] Client receives `channel_added` WebSocket notification when the channel is ready
- [x] Non-YouTube URLs (RTSP, file) bypass the resolver and add synchronously as before

**Estimated complexity:** M

---

## Phase 2: Stream Recovery + Circuit Breaker

**Goal:** Make YouTube Live channels resilient to transient failures (stream interruptions, URL expiry) and protect the pipeline from channels that repeatedly fail.

**Key files:**
- `backend/pipeline/stream_recovery.py`
- `backend/pipeline/deepstream/adapter.py`
- `backend/pipeline/source_resolver.py` (serialized calls)

**Acceptance criteria:**
- [x] Stream recovery: 3 retries with exponential backoff → liveness check → URL re-extract → reconnect
- [x] Circuit breaker: 3 pipeline rebuilds within 10 min → eject channel with appropriate status/notification
- [x] yt-dlp calls serialized via `asyncio.Lock` to avoid concurrent extraction races
- [x] `MjpegExtractor` keeps a last-frame buffer (most recent decoded JPEG)
- [x] Last frame buffer persisted at Phase 2→3 boundary for frozen-frame replay

**Estimated complexity:** L

---

## Phase 3: Frontend — YouTube URL Input + Frozen-Frame Replay

**Goal:** Add YouTube URL support to the React UI with live-stream visual indicators and frozen-frame replay when a stream drops.

**Key files:**
- Frontend channel-add dialog (YouTube URL input)
- Channel tile components (live badge, frozen-frame overlay)

**Acceptance criteria:**
- [x] Channel-add dialog accepts YouTube URLs (validated client-side)
- [x] "Live" badge with red pulse animation displayed on YouTube Live channels
- [x] Frozen-frame replay: when a YouTube stream drops, the last good frame is shown instead of a black tile
- [x] Channel status transitions (connecting → live → recovering → ejected) reflected in UI

**Estimated complexity:** M
