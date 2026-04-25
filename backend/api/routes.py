"""Pipeline control REST endpoints."""

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from backend.api.deps import get_alert_store, get_backend, get_clip_extractor
from backend.api.models import (
    AddChannelRequest,
    ChannelAddedResponse,
    PhaseResponse,
    RemoveChannelRequest,
    ResolvingResponse,
    SetPhaseRequest,
    SiteConfigRequest,
    StatusResponse,
    UpdateConfigRequest,
)
import backend.config.site_config as site_config_mod
from backend.config.site_config import SiteConfig
from backend.pipeline.alerts import AlertStore
from backend.pipeline.clip_extractor import ClipExtractor
from backend.pipeline.protocol import ChannelPhase, PipelineBackend
from backend.pipeline.source_resolver import is_youtube_url, resolve_source

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/channels")
def list_channels(
    request: Request,
    backend: PipelineBackend = Depends(get_backend),
    alert_store: AlertStore = Depends(get_alert_store),
):
    if not request.app.state.pipeline_started:
        return {"channels": [], "pipeline_started": False}
    channels = []
    for channel_id, source in backend.channels.items():
        phase = backend.get_channel_phase(channel_id)
        alert_count = alert_store.count_by_channel(channel_id)
        channels.append(
            {
                "channel_id": channel_id,
                "source": source,
                "phase": phase.value,
                "alert_count": alert_count,
            }
        )
    return {"channels": channels, "pipeline_started": True}


@router.post("/pipeline/start", response_model=StatusResponse)
def start_pipeline(
    request: Request,
    backend: PipelineBackend = Depends(get_backend),
):
    try:
        backend.start()
    except RuntimeError:
        raise HTTPException(status_code=409, detail="Pipeline already started")
    request.app.state.pipeline_started = True
    request.app.state.ws.enqueue(
        {
            "type": "pipeline_event",
            "event": "started",
            "detail": "Pipeline started",
        }
    )
    return StatusResponse(status="started")


@router.post("/pipeline/stop", response_model=StatusResponse)
def stop_pipeline(
    request: Request,
    backend: PipelineBackend = Depends(get_backend),
    alert_store: AlertStore = Depends(get_alert_store),
    clip_extractor: ClipExtractor = Depends(get_clip_extractor),
):
    try:
        backend.stop()
    except RuntimeError:
        raise HTTPException(status_code=409, detail="Pipeline not started")
    request.app.state.pipeline_started = False
    request.app.state.next_channel_id = 0
    clip_extractor.cleanup_all()
    alert_store.clear()
    request.app.state.ws.enqueue(
        {
            "type": "pipeline_event",
            "event": "stopped",
            "detail": "Pipeline stopped",
        }
    )
    return StatusResponse(status="stopped")


@router.get("/channel/{channel_id}")
def get_channel(
    channel_id: int,
    request: Request,
    backend: PipelineBackend = Depends(get_backend),
    alert_store: AlertStore = Depends(get_alert_store),
):
    if channel_id not in backend.channels:
        raise HTTPException(status_code=404, detail="Channel not found")
    phase = backend.get_channel_phase(channel_id)
    alert_count = alert_store.count_by_channel(channel_id)
    config = {}
    if hasattr(backend, "get_channel_config"):
        try:
            config = backend.get_channel_config(channel_id)
        except Exception:
            pass
    return {
        "channel_id": channel_id,
        "source": backend.channels[channel_id],
        "phase": phase.value,
        "alert_count": alert_count,
        "pipeline_started": request.app.state.pipeline_started,
        **config,
    }


@router.post("/channel/add")
async def add_channel(
    body: AddChannelRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    backend: PipelineBackend = Depends(get_backend),
):
    if not request.app.state.pipeline_started:
        raise HTTPException(status_code=409, detail="Pipeline not started")

    if is_youtube_url(body.source):
        # YouTube URLs resolve asynchronously — return 202 immediately
        request_id = str(uuid.uuid4())
        background_tasks.add_task(
            _resolve_and_add_channel, request.app, body.source, request_id
        )
        return JSONResponse(
            status_code=202,
            content=ResolvingResponse(
                status="resolving", request_id=request_id
            ).model_dump(),
        )

    # File sources: validate and add synchronously
    if not body.source.startswith(("http://", "https://", "rtsp://")):
        from pathlib import Path

        source_path = Path(body.source)
        if source_path.is_absolute() and not source_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Source file not found: {body.source}",
            )
    channel_id = request.app.state.next_channel_id
    request.app.state.next_channel_id += 1
    backend.add_channel(channel_id, body.source)
    return ChannelAddedResponse(channel_id=channel_id)


async def _resolve_and_add_channel(app, source: str, request_id: str) -> None:
    """Background task: resolve YouTube URL, then add channel."""
    ws = app.state.ws
    backend = app.state.backend
    try:
        resolved = await resolve_source(source)
        channel_id = app.state.next_channel_id
        app.state.next_channel_id += 1
        backend.add_channel(
            channel_id,
            resolved.stream_url,
            source_type="youtube_live",
            original_url=source,
        )
        ws.enqueue(
            {
                "type": "channel_added",
                "channel": channel_id,
                "source": source,
                "source_type": "youtube_live",
                "request_id": request_id,
            }
        )
        log.info("YouTube channel %d added: %s", channel_id, source)
    except (ValueError, TimeoutError) as exc:
        ws.enqueue(
            {
                "type": "resolution_failed",
                "source": source,
                "request_id": request_id,
                "error": str(exc),
            }
        )
        log.warning("YouTube resolution failed for %s: %s", source, exc)


@router.post("/channel/remove", response_model=StatusResponse)
def remove_channel(
    body: RemoveChannelRequest,
    backend: PipelineBackend = Depends(get_backend),
    alert_store: AlertStore = Depends(get_alert_store),
    clip_extractor: ClipExtractor = Depends(get_clip_extractor),
):
    try:
        backend.remove_channel(body.channel_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Channel not found")
    clip_extractor.cleanup_channel(body.channel_id)
    alert_store.clear_channel(body.channel_id)
    return StatusResponse(status="removed")


@router.post("/channel/{channel_id}/phase", response_model=PhaseResponse)
def set_channel_phase(
    channel_id: int,
    body: SetPhaseRequest,
    request: Request,
    backend: PipelineBackend = Depends(get_backend),
    alert_store: AlertStore = Depends(get_alert_store),
    clip_extractor: ClipExtractor = Depends(get_clip_extractor),
):
    try:
        phase = ChannelPhase(body.phase)
    except ValueError:
        valid = [p.value for p in ChannelPhase]
        raise HTTPException(
            status_code=422, detail=f"Invalid phase. Must be one of: {valid}"
        )
    try:
        previous = backend.get_channel_phase(channel_id)

        # Configure channel with ROI/lines before starting analytics
        if previous == ChannelPhase.SETUP and phase == ChannelPhase.ANALYTICS:
            if not body.roi_polygon or len(body.roi_polygon) < 3:
                raise HTTPException(
                    status_code=422,
                    detail="ROI polygon with at least 3 vertices required to start analytics",
                )
            entry_exit = {}
            if body.entry_exit_lines:
                # Check for duplicate labels across different keys
                labels = [v.label for v in body.entry_exit_lines.values()]
                if len(labels) != len(set(labels)):
                    raise HTTPException(
                        status_code=422,
                        detail="Duplicate entry/exit line labels are not allowed",
                    )
                entry_exit = {
                    k: {
                        "label": v.label,
                        "start": list(v.start),
                        "end": list(v.end),
                        "junction_side": v.junction_side,
                    }
                    for k, v in body.entry_exit_lines.items()
                }
            backend.configure_channel(
                channel_id,
                roi_polygon=[tuple(p) for p in body.roi_polygon],
                entry_exit_lines=entry_exit,
            )

        backend.set_channel_phase(channel_id, phase)
    except KeyError:
        raise HTTPException(status_code=404, detail="Channel not found")

    # Trigger clip extraction on analytics → review transition
    if previous == ChannelPhase.ANALYTICS and phase == ChannelPhase.REVIEW:
        source = backend.channels.get(channel_id, "")
        alerts = alert_store.get_channel_alerts(channel_id)
        clip_extractor.extract_clips(channel_id, source, alerts)

    request.app.state.ws.enqueue(
        {
            "type": "phase_changed",
            "channel": channel_id,
            "phase": phase.value,
            "previous_phase": previous.value,
        }
    )
    return PhaseResponse(status="ok", phase=phase.value)


@router.patch("/config", response_model=StatusResponse)
def update_config(
    body: UpdateConfigRequest,
    backend: PipelineBackend = Depends(get_backend),
):
    if body.confidence_threshold is not None:
        backend.set_confidence_threshold(body.confidence_threshold)
    return StatusResponse(status="updated")


@router.get("/alerts")
def list_alerts(
    limit: int = Query(default=50, ge=1, le=200),
    type: str | None = Query(default=None),
    channel: int | None = Query(default=None),
    alert_store: AlertStore = Depends(get_alert_store),
):
    return alert_store.get_alerts(limit=limit, alert_type=type, channel=channel)


@router.get("/alert/{alert_id}")
def get_alert(
    alert_id: str,
    alert_store: AlertStore = Depends(get_alert_store),
):
    alert = alert_store.get_alert(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@router.get("/alert/{alert_id}/replay")
def get_replay(
    alert_id: str,
    alert_store: AlertStore = Depends(get_alert_store),
    clip_extractor: ClipExtractor = Depends(get_clip_extractor),
):
    alert = alert_store.get_alert(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")

    status = clip_extractor.get_status(alert_id)

    # Clip not yet extracted (phase transition hasn't happened or not started)
    if status is None:
        return JSONResponse(
            status_code=202,
            content={"status": "not_started", "alert_id": alert_id},
        )

    if status == "pending":
        return JSONResponse(
            status_code=202,
            content={"status": "extracting", "alert_id": alert_id},
        )

    if status == "failed":
        raise HTTPException(status_code=500, detail="Clip extraction failed")

    # status == "ready"
    clip_path = clip_extractor.get_clip_path(alert_id)
    if clip_path is None or not clip_path.exists():
        raise HTTPException(status_code=500, detail="Clip file not found")

    if alert["type"] == "stagnant_alert":
        return FileResponse(clip_path, media_type="image/jpeg")

    # Transit alert — return video with range request support
    return FileResponse(clip_path, media_type="video/mp4")


@router.get("/channel/{channel_id}/last_frame")
def get_last_frame(channel_id: int):
    """Return the last captured frame for a channel (Phase 3 frozen-frame replay)."""
    from pathlib import Path

    frame_path = Path("snapshots") / str(channel_id) / "last_frame.jpg"
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Last frame not found")
    return FileResponse(frame_path, media_type="image/jpeg")


@router.get("/snapshot/{track_id}")
def get_snapshot(
    track_id: str,
    backend: PipelineBackend = Depends(get_backend),
):
    # track_id comes as string (to avoid JS precision loss); convert to int
    # for the backend lookup which uses DeepStream's integer IDs.
    try:
        tid = int(track_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid track ID")
    jpeg = backend.get_snapshot(tid)
    if jpeg is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return Response(content=jpeg, media_type="image/jpeg")


@router.post("/site/config", response_model=StatusResponse)
def save_site(body: SiteConfigRequest):
    entry_exit = {
        k: {"label": v.label, "start": list(v.start), "end": list(v.end), "junction_side": v.junction_side}
        for k, v in body.entry_exit_lines.items()
    }
    config = SiteConfig(
        site_id=body.site_id,
        roi_polygon=list(body.roi_polygon),
        entry_exit_lines=entry_exit,
    )
    try:
        site_config_mod.save_site_config(config, site_config_mod.SITES_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return StatusResponse(status="saved")


@router.get("/site/config")
def load_site(site_id: str = Query()):
    try:
        config = site_config_mod.load_site_config(site_id, site_config_mod.SITES_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Site config not found")
    return {
        "site_id": config.site_id,
        "roi_polygon": [list(p) for p in config.roi_polygon],
        "entry_exit_lines": config.entry_exit_lines,
    }


@router.get("/site/configs")
def list_sites():
    sites_dir = site_config_mod.SITES_DIR
    if not sites_dir.exists():
        return {"sites": []}
    sites = [f.stem for f in sorted(sites_dir.glob("*.json"))]
    return {"sites": sites}
