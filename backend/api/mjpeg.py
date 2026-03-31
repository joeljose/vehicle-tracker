"""MJPEG streaming — fan-out annotated frames to browser clients."""

import asyncio
import queue
from collections import defaultdict
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.pipeline.protocol import FrameResult

router = APIRouter()

BOUNDARY = "frame"


class MjpegBroadcaster:
    """Fan-out annotated JPEG frames to multiple HTTP clients per channel.

    The pipeline thread calls on_frame() which puts frames into per-subscriber
    thread-safe queues. Each /stream/{channel_id} client gets its own queue.
    """

    def __init__(self):
        # channel_id → list of subscriber queues
        self._subscribers: dict[int, list[queue.Queue]] = defaultdict(list)

    def on_frame(self, result: FrameResult) -> None:
        """Called from pipeline thread with each frame result."""
        if result.annotated_jpeg is None:
            return
        channel_id = result.channel_id
        jpeg = result.annotated_jpeg
        for q in self._subscribers.get(channel_id, []):
            try:
                q.put_nowait(jpeg)
            except queue.Full:
                # Drop oldest frame to keep stream current
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(jpeg)
                except queue.Full:
                    pass

    def subscribe(self, channel_id: int) -> queue.Queue:
        """Create a new subscriber queue for a channel."""
        q: queue.Queue = queue.Queue(maxsize=2)
        self._subscribers[channel_id].append(q)
        return q

    def unsubscribe(self, channel_id: int, q: queue.Queue) -> None:
        """Remove a subscriber queue."""
        subs = self._subscribers.get(channel_id)
        if subs:
            try:
                subs.remove(q)
            except ValueError:
                pass

    async def stream(self, channel_id: int) -> AsyncGenerator[bytes, None]:
        """Yield multipart MJPEG frames for one client."""
        q = self.subscribe(channel_id)
        try:
            while True:
                try:
                    jpeg = await asyncio.to_thread(q.get, timeout=1.0)
                except queue.Empty:
                    continue
                yield (
                    (
                        f"--{BOUNDARY}\r\n"
                        f"Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpeg)}\r\n"
                        f"\r\n"
                    ).encode()
                    + jpeg
                    + b"\r\n"
                )
        except asyncio.CancelledError:
            return
        finally:
            self.unsubscribe(channel_id, q)


@router.get("/stream/{channel_id}")
async def stream_channel(channel_id: int, request: Request):
    """MJPEG stream of annotated frames for a channel."""
    if not request.app.state.pipeline_started:
        raise HTTPException(status_code=409, detail="Pipeline not started")
    backend = request.app.state.backend
    try:
        backend.get_channel_phase(channel_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Channel not found")
    broadcaster: MjpegBroadcaster = request.app.state.mjpeg
    return StreamingResponse(
        broadcaster.stream(channel_id),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
    )
