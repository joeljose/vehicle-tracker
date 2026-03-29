"""WebSocket broadcaster — real-time data push to browser clients."""

import asyncio
import json
import logging
import queue
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.pipeline.protocol import FrameResult

router = APIRouter()
log = logging.getLogger(__name__)


class WsBroadcaster:
    """Broadcast pipeline events to connected WebSocket clients.

    Pipeline callbacks (called from the pipeline thread) put messages into
    a thread-safe queue. A background asyncio task drains the queue and
    sends to all connected clients with optional channel filtering.
    """

    def __init__(self):
        # (websocket, channel_filter) — None means all channels
        self._clients: list[tuple[WebSocket, set[int] | None]] = []
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._drain_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the background drain task."""
        self._drain_task = asyncio.create_task(self._drain_loop())

    async def stop(self) -> None:
        """Stop the background drain task."""
        if self._drain_task:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
            self._drain_task = None

    async def connect(self, ws: WebSocket, channels: set[int] | None) -> None:
        """Accept a WebSocket and register it for broadcasts."""
        await ws.accept()
        self._clients.append((ws, channels))

    def disconnect(self, ws: WebSocket) -> None:
        """Remove a WebSocket from the client list."""
        self._clients = [(w, c) for w, c in self._clients if w is not ws]

    def enqueue(self, message: dict) -> None:
        """Thread-safe: put a message on the broadcast queue."""
        self._queue.put_nowait(message)

    # -- Pipeline callbacks (called from pipeline thread) --

    def on_frame(self, result: FrameResult) -> None:
        """Convert FrameResult to frame_data WS message."""
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

    def on_alert(self, alert: dict) -> None:
        """Forward an alert dict (already has 'type' field) to WS clients."""
        self.enqueue(alert)

    def on_track_ended(self, track: dict) -> None:
        """Forward a track_ended dict to WS clients."""
        self.enqueue(track)

    # -- Internal --

    async def _drain_loop(self) -> None:
        """Background task: drain queue and broadcast to clients."""
        while True:
            try:
                msg = self._queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            await self._broadcast(msg)

    async def _broadcast(self, message: dict) -> None:
        """Send a message to all matching connected clients."""
        msg_channel = message.get("channel")
        text = json.dumps(message)
        dead: list[WebSocket] = []
        for ws, channel_filter in self._clients:
            # Channel filtering: skip if client subscribed to specific channels
            # and this message has a channel that doesn't match.
            # Messages without a channel (e.g. pipeline_event) go to everyone.
            if channel_filter is not None and msg_channel is not None:
                if msg_channel not in channel_filter:
                    continue
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint for real-time pipeline data."""
    broadcaster: WsBroadcaster = ws.app.state.ws

    # Parse optional channel filter from query params
    channels_param = ws.query_params.get("channels")
    channel_filter: set[int] | None = None
    if channels_param:
        channel_filter = {int(c) for c in channels_param.split(",") if c.strip()}

    await broadcaster.connect(ws, channel_filter)
    try:
        while True:
            # Keep connection alive — client doesn't send meaningful data
            await ws.receive_text()
    except WebSocketDisconnect:
        broadcaster.disconnect(ws)
