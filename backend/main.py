"""FastAPI application factory."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.mjpeg import MjpegBroadcaster
from backend.api.mjpeg import router as mjpeg_router
from backend.api.routes import router
from backend.api.websocket import WsBroadcaster
from backend.api.websocket import router as ws_router
from backend.pipeline.alerts import AlertStore
from backend.pipeline.clip_extractor import ClipExtractor
from backend.pipeline.protocol import FrameResult


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: start WS broadcaster drain loop
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

    # Fan-out frame callback to both MJPEG and WS broadcasters
    def on_frame(result: FrameResult) -> None:
        mjpeg.on_frame(result)
        ws.on_frame(result)

    # Alert callback: store in AlertStore, then push summary to WS
    def on_alert(alert: dict) -> None:
        channel = alert.get("channel", 0)
        if alert.get("type") == "stagnant_alert":
            alert_id = alert_store.add_stagnant_alert(alert, channel)
        else:
            alert_id = alert_store.add_transit_alert(alert, channel)
        summary = alert_store.get_ws_summary(alert_id)
        if summary:
            ws.on_alert(summary)

    pipeline_backend.register_frame_callback(on_frame)
    pipeline_backend.register_alert_callback(on_alert)
    pipeline_backend.register_track_ended_callback(ws.on_track_ended)

    app.include_router(router)
    app.include_router(mjpeg_router)
    app.include_router(ws_router)
    return app


if __name__ == "__main__":
    import uvicorn

    app = create_app(backend=os.environ.get("VT_BACKEND", "deepstream"))
    uvicorn.run(app, host="0.0.0.0", port=8000)
