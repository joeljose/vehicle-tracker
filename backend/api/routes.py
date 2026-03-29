"""Pipeline control REST endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response

from backend.api.deps import get_alert_store, get_backend
from backend.api.models import (
    AddChannelRequest,
    ChannelAddedResponse,
    PhaseResponse,
    RemoveChannelRequest,
    SetPhaseRequest,
    SiteConfigRequest,
    StatusResponse,
    UpdateConfigRequest,
)
import backend.config.site_config as site_config_mod
from backend.config.site_config import SiteConfig
from backend.pipeline.alerts import AlertStore
from backend.pipeline.protocol import ChannelPhase, PipelineBackend

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
        channels.append({
            "channel_id": channel_id,
            "source": source,
            "phase": phase.value,
            "alert_count": alert_count,
        })
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
    request.app.state.ws.enqueue({
        "type": "pipeline_event",
        "event": "started",
        "detail": "Pipeline started",
    })
    return StatusResponse(status="started")


@router.post("/pipeline/stop", response_model=StatusResponse)
def stop_pipeline(
    request: Request,
    backend: PipelineBackend = Depends(get_backend),
    alert_store: AlertStore = Depends(get_alert_store),
):
    try:
        backend.stop()
    except RuntimeError:
        raise HTTPException(status_code=409, detail="Pipeline not started")
    request.app.state.pipeline_started = False
    request.app.state.next_channel_id = 0
    alert_store.clear()
    request.app.state.ws.enqueue({
        "type": "pipeline_event",
        "event": "stopped",
        "detail": "Pipeline stopped",
    })
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
    return {
        "channel_id": channel_id,
        "source": backend.channels[channel_id],
        "phase": phase.value,
        "alert_count": alert_count,
        "pipeline_started": request.app.state.pipeline_started,
    }


@router.post("/channel/add", response_model=ChannelAddedResponse)
def add_channel(
    body: AddChannelRequest,
    request: Request,
    backend: PipelineBackend = Depends(get_backend),
):
    if not request.app.state.pipeline_started:
        raise HTTPException(status_code=409, detail="Pipeline not started")
    channel_id = request.app.state.next_channel_id
    request.app.state.next_channel_id += 1
    backend.add_channel(channel_id, body.source)
    return ChannelAddedResponse(channel_id=channel_id)


@router.post("/channel/remove", response_model=StatusResponse)
def remove_channel(
    body: RemoveChannelRequest,
    backend: PipelineBackend = Depends(get_backend),
    alert_store: AlertStore = Depends(get_alert_store),
):
    try:
        backend.remove_channel(body.channel_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Channel not found")
    alert_store.clear_channel(body.channel_id)
    return StatusResponse(status="removed")


@router.post("/channel/{channel_id}/phase", response_model=PhaseResponse)
def set_channel_phase(
    channel_id: int,
    body: SetPhaseRequest,
    request: Request,
    backend: PipelineBackend = Depends(get_backend),
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
        backend.set_channel_phase(channel_id, phase)
    except KeyError:
        raise HTTPException(status_code=404, detail="Channel not found")
    request.app.state.ws.enqueue({
        "type": "phase_changed",
        "channel": channel_id,
        "phase": phase.value,
        "previous_phase": previous.value,
    })
    return PhaseResponse(status="ok", phase=phase.value)


@router.patch("/config", response_model=StatusResponse)
def update_config(
    body: UpdateConfigRequest,
    backend: PipelineBackend = Depends(get_backend),
):
    if body.confidence_threshold is not None:
        backend.set_confidence_threshold(body.confidence_threshold)
    if body.inference_interval is not None:
        backend.set_inference_interval(body.inference_interval)
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


@router.get("/snapshot/{track_id}")
def get_snapshot(
    track_id: int,
    backend: PipelineBackend = Depends(get_backend),
):
    jpeg = backend.get_snapshot(track_id)
    if jpeg is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return Response(content=jpeg, media_type="image/jpeg")


@router.post("/site/config", response_model=StatusResponse)
def save_site(body: SiteConfigRequest):
    entry_exit = {
        k: {"label": v.label, "start": list(v.start), "end": list(v.end)}
        for k, v in body.entry_exit_lines.items()
    }
    config = SiteConfig(
        site_id=body.site_id,
        roi_polygon=list(body.roi_polygon),
        entry_exit_lines=entry_exit,
    )
    site_config_mod.save_site_config(config, site_config_mod.SITES_DIR)
    return StatusResponse(status="saved")


@router.get("/site/config")
def load_site(site_id: str = Query()):
    try:
        config = site_config_mod.load_site_config(site_id, site_config_mod.SITES_DIR)
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
