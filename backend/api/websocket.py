"""WebSocket broadcaster — real-time data push to browser clients.

The fan-out has **per-client** queues + per-client send tasks so a single slow
client cannot stall delivery to other clients. Each enqueue is fanned out
synchronously to every matching client's bounded queue (drop-oldest on full),
and each client's background task pumps its own queue into ``ws.send_text``.

A bounded ``_queue`` tap is kept around for test introspection — every enqueued
message is recorded there so synchronous tests can ``_queue.get_nowait()``
after a request to verify the broadcaster received it. The tap is bounded
(drop-oldest) so it can't leak memory if production code never drains it.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.pipeline.protocol import FrameResult

router = APIRouter()
log = logging.getLogger(__name__)

# Bounded per-client queue. At 30 FPS frame_data on one channel that's ~8 s of
# buffering before drops, which is generous for normal browser hiccups but
# small enough that a frozen tab can't accumulate many seconds of stale state.
_CLIENT_QUEUE_MAXSIZE = 256

# Test-introspection tap. Production never reads this; bounded to prevent
# unbounded growth in long-running processes.
_TEST_TAP_MAXSIZE = 1000


@dataclass
class _ClientChannel:
    """Per-client send pipe: WebSocket + filters + queue + drain task."""

    ws: WebSocket
    channels: set[int] | None  # None = all channels
    types: set[str] | None     # None = all message types
    queue: asyncio.Queue = field(
        default_factory=lambda: asyncio.Queue(maxsize=_CLIENT_QUEUE_MAXSIZE),
    )
    task: asyncio.Task | None = None
    alive: bool = True


class WsBroadcaster:
    """Per-client fan-out of pipeline events to connected WebSocket clients.

    Pipeline callbacks (called from the pipeline thread) put messages into
    every matching client's queue via ``call_soon_threadsafe``. A background
    asyncio task per client pumps that client's queue into ``ws.send_text``.
    A slow client only loses its own backlog; everyone else is unaffected.
    """

    def __init__(self):
        self._clients: list[_ClientChannel] = []
        # Test-introspection tap: bounded with drop-oldest. Production never
        # reads from this; tests can ``_queue.get_nowait()`` / ``empty()``.
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=_TEST_TAP_MAXSIZE,
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        # Stats tracking for stats_update (per channel)
        self._stats_last_emit: dict[int, float] = {}  # channel_id -> last emit time
        self._stats_frame_count: dict[
            int, int
        ] = {}  # channel_id -> frames since last emit
        self._stats_inference_ms_sum: dict[int, float] = {}  # channel_id -> sum for avg

    async def start(self) -> None:
        """Capture the running event loop for cross-thread enqueue."""
        self._loop = asyncio.get_running_loop()

    async def stop(self) -> None:
        """Cancel all per-client tasks and disconnect."""
        for client in list(self._clients):
            client.alive = False
            if client.task is not None:
                client.task.cancel()
        # Drain the cancellations
        tasks = [c.task for c in self._clients if c.task is not None]
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                pass
        self._clients.clear()

    async def connect(
        self,
        ws: WebSocket,
        channels: set[int] | None,
        types: set[str] | None = None,
    ) -> None:
        """Accept a WebSocket and start its per-client send task."""
        await ws.accept()
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = None
        client = _ClientChannel(ws=ws, channels=channels, types=types)
        client.task = asyncio.create_task(self._client_loop(client))
        self._clients.append(client)

    def disconnect(self, ws: WebSocket) -> None:
        """Remove a WebSocket from the client list (idempotent).

        Cancels the client's send task. Safe to call from the websocket
        route handler on disconnect.
        """
        for client in list(self._clients):
            if client.ws is ws:
                client.alive = False
                if client.task is not None:
                    client.task.cancel()
                self._clients.remove(client)
                return

    def enqueue(self, message: dict) -> None:
        """Thread-safe: fan ``message`` out to every matching client.

        From a non-event-loop thread (pipeline thread, gst probe, etc) we
        hop onto the loop via ``call_soon_threadsafe``; from inside the
        loop or with no loop running (tests) we fan out synchronously.
        """
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._fanout, message)
        else:
            self._fanout(message)

    # -- Pipeline callbacks (called from pipeline thread) --

    def on_frame(self, result: FrameResult) -> None:
        """Convert FrameResult to frame_data WS message and emit stats_update."""
        msg = {
            "type": "frame_data",
            "channel": result.channel_id,
            "frame": result.frame_number,
            "timestamp_ms": result.timestamp_ms,
            "tracks": [
                {
                    "id": d.track_id,
                    "class": d.class_name,
                    "bbox": list(d.bbox),
                    "confidence": d.confidence,
                    "centroid": list(d.centroid),
                }
                for d in result.detections
            ],
        }
        self.enqueue(msg)
        self._maybe_emit_stats(result)

    def on_alert(self, alert: dict) -> None:
        """Forward an alert dict (already has 'type' field) to WS clients."""
        self.enqueue(alert)

    def on_track_ended(self, track: dict) -> None:
        """Forward a track_ended dict to WS clients."""
        self.enqueue(track)

    def _maybe_emit_stats(self, result: FrameResult) -> None:
        """Emit stats_update every 1 second per channel."""
        ch = result.channel_id
        now = time.monotonic()
        last = self._stats_last_emit.get(ch, 0.0)
        self._stats_frame_count[ch] = self._stats_frame_count.get(ch, 0) + 1
        self._stats_inference_ms_sum[ch] = (
            self._stats_inference_ms_sum.get(ch, 0.0) + result.inference_ms
        )
        if now - last >= 1.0:
            frames = self._stats_frame_count[ch]
            elapsed = now - last if last > 0 else 1.0
            fps = round(frames / elapsed, 1)
            avg_inference = (
                round(self._stats_inference_ms_sum[ch] / frames, 1)
                if frames > 0
                else 0.0
            )
            self.enqueue(
                {
                    "type": "stats_update",
                    "channel": ch,
                    "fps": fps,
                    "active_tracks": len(result.detections),
                    "inference_ms": avg_inference,
                    "phase": result.phase,
                    "idle_mode": result.idle_mode,
                }
            )
            self._stats_last_emit[ch] = now
            self._stats_frame_count[ch] = 0
            self._stats_inference_ms_sum[ch] = 0.0

    # -- Internal --

    def _fanout(self, message: dict) -> None:
        """Synchronous fan-out: record on tap, deliver to every matching client.

        Per-client delivery uses ``put_nowait`` with drop-oldest semantics so a
        slow client cannot back-pressure ``enqueue``.
        """
        # Test-tap (bounded, drop-oldest).
        self._put_with_drop(self._queue, message)

        msg_channel = message.get("channel")
        msg_type = message.get("type")
        for client in list(self._clients):
            if not client.alive:
                continue
            # Channel filter — messages with no channel (pipeline_event etc.)
            # always pass.
            if (
                client.channels is not None
                and msg_channel is not None
                and msg_channel not in client.channels
            ):
                continue
            # Type filter — messages with no type always pass.
            if (
                client.types is not None
                and msg_type is not None
                and msg_type not in client.types
            ):
                continue
            self._put_with_drop(client.queue, message)

    @staticmethod
    def _put_with_drop(q: asyncio.Queue, item: Any) -> None:
        """``put_nowait`` with drop-oldest fallback when the queue is full."""
        try:
            q.put_nowait(item)
            return
        except asyncio.QueueFull:
            pass
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            # Should not happen — the queue had room after the get above —
            # but stay silent if a concurrent producer beat us to it.
            pass

    async def _client_loop(self, client: _ClientChannel) -> None:
        """Drain a single client's queue and forward to its WebSocket.

        Survives ``send_text`` errors by marking the client dead and removing
        it from the broadcaster — one bad consumer cannot stall any other.
        """
        try:
            while client.alive:
                message = await client.queue.get()
                try:
                    await client.ws.send_text(json.dumps(message))
                except Exception:
                    client.alive = False
                    self._remove_client(client)
                    return
        except asyncio.CancelledError:
            return

    def _remove_client(self, client: _ClientChannel) -> None:
        """Remove a client (idempotent) without cancelling its task — caller
        is expected to be inside that task already."""
        try:
            self._clients.remove(client)
        except ValueError:
            pass


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint for real-time pipeline data."""
    broadcaster: WsBroadcaster = ws.app.state.ws

    # Parse optional channel filter from query params
    channels_param = ws.query_params.get("channels")
    channel_filter: set[int] | None = None
    if channels_param:
        channel_filter = {int(c) for c in channels_param.split(",") if c.strip()}

    # Parse optional type filter from query params
    types_param = ws.query_params.get("types")
    type_filter: set[str] | None = None
    if types_param:
        type_filter = {t.strip() for t in types_param.split(",") if t.strip()}

    await broadcaster.connect(ws, channel_filter, type_filter)
    try:
        while True:
            # Keep connection alive — client doesn't send meaningful data
            await ws.receive_text()
    except WebSocketDisconnect:
        broadcaster.disconnect(ws)
