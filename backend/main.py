"""FastAPI application factory."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.api.mjpeg import MjpegBroadcaster
from backend.api.mjpeg import router as mjpeg_router
from backend.api.routes import router
from backend.pipeline.alerts import AlertStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Shutdown: stop pipeline if still running
    if app.state.pipeline_started:
        app.state.backend.stop()


def create_app(backend: str = "deepstream") -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        backend: Pipeline backend to use. "deepstream" for production,
                 "fake" for testing.
    """
    if backend == "deepstream":
        from backend.pipeline.deepstream.pipeline import DeepStreamPipeline
        pipeline_backend = DeepStreamPipeline()
    elif backend == "fake":
        from backend.pipeline.fake import FakeBackend
        pipeline_backend = FakeBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")

    app = FastAPI(title="Vehicle Tracker", lifespan=lifespan)
    app.state.backend = pipeline_backend
    app.state.alert_store = AlertStore()
    app.state.pipeline_started = False
    app.state.next_channel_id = 0

    broadcaster = MjpegBroadcaster()
    app.state.mjpeg = broadcaster
    pipeline_backend.register_frame_callback(broadcaster.on_frame)

    app.include_router(router)
    app.include_router(mjpeg_router)
    return app


if __name__ == "__main__":
    import uvicorn

    app = create_app(backend=os.environ.get("VT_BACKEND", "deepstream"))
    uvicorn.run(app, host="0.0.0.0", port=8000)
