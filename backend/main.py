"""FastAPI application factory."""

import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.mjpeg import MjpegBroadcaster
from backend.api.mjpeg import router as mjpeg_router
from backend.api.routes import router
from backend.api.websocket import WsBroadcaster
from backend.api.websocket import router as ws_router
from backend.pipeline.alerts import AlertStore
from backend.pipeline.clip_extractor import ClipExtractor
from backend.pipeline.protocol import ChannelPhase, FrameResult


log = logging.getLogger(__name__)


def _cleanup_stale_artifacts() -> None:
    """Remove leftover snapshots and clips from previous sessions."""
    snapshots = Path("snapshots")
    if snapshots.exists():
        for child in snapshots.iterdir():
            if child.name == ".gitkeep":
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        log.info("Cleaned up stale snapshots")

    clips = Path("/tmp/vt_clips")
    if clips.exists():
        shutil.rmtree(clips, ignore_errors=True)
        log.info("Cleaned up stale clips")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: clean stale artifacts, start WS broadcaster drain loop
    _cleanup_stale_artifacts()
    await app.state.ws.start()
    yield
    # Shutdown: stop WS broadcaster, clip extractor, and pipeline
    await app.state.ws.stop()
    app.state.clip_extractor.shutdown()
    app.state.clip_extractor.cleanup_all()
    if app.state.pipeline_started:
        app.state.backend.stop()


def create_app(backend: str = "deepstream") -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        backend: Pipeline backend to use. "deepstream" for production,
                 "fake" for testing.
    """
    if backend == "deepstream":
        from backend.pipeline.deepstream.adapter import DeepStreamPipeline

        pipeline_backend = DeepStreamPipeline()
    elif backend == "custom":
        from backend.pipeline.custom.adapter import CustomPipeline

        pipeline_backend = CustomPipeline()
    elif backend == "fake":
        from backend.pipeline.fake import FakeBackend

        pipeline_backend = FakeBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")

    app = FastAPI(title="Vehicle Tracker", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.backend = pipeline_backend
    app.state.alert_store = AlertStore()
    app.state.clip_extractor = ClipExtractor()
    app.state.pipeline_started = False
    app.state.next_channel_id = 0

    mjpeg = MjpegBroadcaster()
    app.state.mjpeg = mjpeg

    ws = WsBroadcaster()
    app.state.ws = ws

    alert_store = app.state.alert_store
    clip_extractor = app.state.clip_extractor

    # Fan-out frame callback to both MJPEG and WS broadcasters
    def on_frame(result: FrameResult) -> None:
        mjpeg.on_frame(result)
        ws.on_frame(result)

    # Alert callback: store in AlertStore, then push summary to WS.
    # If the channel is already in REVIEW phase (late arrival during transition),
    # extract the clip immediately instead of waiting for a batch that won't come.
    def on_alert(alert: dict) -> None:
        channel = alert.get("channel", 0)
        if alert.get("type") == "stagnant_alert":
            alert_id = alert_store.add_stagnant_alert(alert, channel)
        else:
            alert_id = alert_store.add_transit_alert(alert, channel)

        # Late alert: channel already transitioned to review — extract clip now
        try:
            phase = pipeline_backend.get_channel_phase(channel)
            if phase == ChannelPhase.REVIEW:
                source = pipeline_backend.channels.get(channel, "")
                full_alert = alert_store.get_alert(alert_id)
                if full_alert:
                    clip_extractor.extract_single(channel, source, full_alert)
        except KeyError:
            pass

        summary = alert_store.get_ws_summary(alert_id)
        if summary:
            ws.on_alert(summary)

    def on_phase_change(
        channel_id: int, new_phase: ChannelPhase, previous: ChannelPhase
    ) -> None:
        # Trigger clip extraction on analytics → review (EOS auto-transition)
        if previous == ChannelPhase.ANALYTICS and new_phase == ChannelPhase.REVIEW:
            source = pipeline_backend.channels.get(channel_id, "")
            alerts = alert_store.get_channel_alerts(channel_id)
            clip_extractor.extract_clips(channel_id, source, alerts)
        ws.enqueue(
            {
                "type": "phase_changed",
                "channel": channel_id,
                "phase": new_phase.value,
                "previous_phase": previous.value,
            }
        )

    pipeline_backend.register_frame_callback(on_frame)
    pipeline_backend.register_alert_callback(on_alert)
    pipeline_backend.register_track_ended_callback(ws.on_track_ended)
    pipeline_backend.register_phase_callback(on_phase_change)

    # Register WS broadcaster for stream recovery notifications (M6)
    if hasattr(pipeline_backend, "register_ws_broadcaster"):
        pipeline_backend.register_ws_broadcaster(ws)

    app.include_router(router)
    app.include_router(mjpeg_router)
    app.include_router(ws_router)
    return app


if __name__ == "__main__":
    import uvicorn

    app = create_app(backend=os.environ.get("VT_BACKEND", "custom"))
    uvicorn.run(app, host="0.0.0.0", port=8000)
