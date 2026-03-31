"""M2 Integration test — full API lifecycle with FakeBackend.

Exercises the complete API surface end-to-end: pipeline control,
channel management, phase transitions, MJPEG broadcaster, WebSocket
broadcaster, alerts, snapshots, and site config.
"""

from backend.tests.helpers import ANALYTICS_PHASE_BODY


def test_full_api_lifecycle(client, app):
    """End-to-end: start → add channel → analytics → alerts → snapshot → stop."""
    backend = app.state.backend
    alert_store = app.state.alert_store
    ws_broadcaster = app.state.ws
    mjpeg_broadcaster = app.state.mjpeg

    # -- 1. Pipeline start --
    resp = client.post("/pipeline/start")
    assert resp.status_code == 200

    # WS broadcaster received pipeline_event
    msg = ws_broadcaster._queue.get_nowait()
    assert msg == {
        "type": "pipeline_event",
        "event": "started",
        "detail": "Pipeline started",
    }

    # -- 2. Add two channels --
    r0 = client.post("/channel/add", json={"source": "/data/lytle_south.mp4"})
    r1 = client.post("/channel/add", json={"source": "/data/741_73.mp4"})
    assert r0.json()["channel_id"] == 0
    assert r1.json()["channel_id"] == 1

    # -- 3. Transition to analytics --
    resp = client.post("/channel/0/phase", json=ANALYTICS_PHASE_BODY)
    assert resp.json()["phase"] == "analytics"
    resp = client.post("/channel/1/phase", json=ANALYTICS_PHASE_BODY)
    assert resp.json()["phase"] == "analytics"

    # Drain phase_changed events from WS queue
    while not ws_broadcaster._queue.empty():
        ws_broadcaster._queue.get_nowait()

    # -- 4. Update config --
    resp = client.patch("/config", json={"confidence_threshold": 0.6})
    assert resp.status_code == 200
    assert backend.confidence_threshold == 0.6

    # -- 5. Simulate frame emission → MJPEG + WS --
    mjpeg_q = mjpeg_broadcaster.subscribe(0)
    backend.emit_frame(channel_id=0, frame_number=1, jpeg=b"\xff\xd8frame1")

    # MJPEG subscriber got the frame
    assert mjpeg_q.get_nowait() == b"\xff\xd8frame1"
    # WS broadcaster got frame_data (may also get stats_update)
    ws_msgs = []
    while not ws_broadcaster._queue.empty():
        ws_msgs.append(ws_broadcaster._queue.get_nowait())
    frame_msg = next(m for m in ws_msgs if m["type"] == "frame_data")
    assert frame_msg["channel"] == 0
    assert frame_msg["frame"] == 1

    # -- 6. Simulate alert emission → AlertStore + WS --
    backend.emit_alert(
        {
            "track_id": 42,
            "label": "car",
            "entry_arm": "north",
            "entry_label": "741-North",
            "exit_arm": "south",
            "exit_label": "741-South",
            "method": "confirmed",
            "was_stagnant": False,
            "first_seen_frame": 100,
            "last_seen_frame": 400,
            "duration_frames": 300,
            "trajectory": [(100, 200), (200, 300)],
            "per_frame_data": [
                {
                    "frame": 100,
                    "bbox": (10, 20, 30, 40),
                    "centroid": (25, 40),
                    "confidence": 0.9,
                },
            ],
        }
    )

    # Alert stored
    alerts = alert_store.get_alerts()
    assert len(alerts) == 1
    assert alerts[0]["type"] == "transit_alert"
    assert alerts[0]["track_id"] == "42"

    # WS got the alert summary
    ws_alert = ws_broadcaster._queue.get_nowait()
    assert ws_alert["type"] == "transit_alert"

    # REST endpoint returns it
    resp = client.get("/alerts")
    assert len(resp.json()) == 1

    # Full metadata available
    alert_id = resp.json()[0]["alert_id"]
    full = client.get(f"/alert/{alert_id}").json()
    assert "per_frame_data" in full
    assert full["entry_direction"] == "north"

    # -- 7. Simulate track_ended → WS --
    backend.emit_track_ended(
        {
            "type": "track_ended",
            "channel": 0,
            "track_id": 42,
            "class": "car",
            "state": "completed",
        }
    )
    ws_track = ws_broadcaster._queue.get_nowait()
    assert ws_track["type"] == "track_ended"

    # -- 8. Snapshot serving --
    backend._snapshots[42] = b"\xff\xd8\xff\xe0snapshot"
    resp = client.get("/snapshot/42")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content == b"\xff\xd8\xff\xe0snapshot"

    # -- 9. Remove channel (clears alerts for that channel) --
    resp = client.post("/channel/remove", json={"channel_id": 0})
    assert resp.status_code == 200
    assert len(alert_store.get_alerts()) == 0  # alert was on channel 0

    # -- 10. Pipeline stop --
    resp = client.post("/pipeline/stop")
    assert resp.status_code == 200

    ws_stop = ws_broadcaster._queue.get_nowait()
    assert ws_stop == {
        "type": "pipeline_event",
        "event": "stopped",
        "detail": "Pipeline stopped",
    }

    # Pipeline state reset
    assert not app.state.pipeline_started
    assert app.state.next_channel_id == 0


def test_site_config_round_trip(client, monkeypatch, tmp_path):
    """Save → load → list site configs through the API."""
    import backend.config.site_config as sc

    monkeypatch.setattr(sc, "SITES_DIR", tmp_path)

    config = {
        "site_id": "741_73",
        "roi_polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
        "entry_exit_lines": {
            "north": {"label": "741-North", "start": [100, 50], "end": [200, 50]},
            "south": {"label": "741-South", "start": [100, 350], "end": [200, 350]},
        },
    }

    # Save
    resp = client.post("/site/config", json=config)
    assert resp.status_code == 200

    # Load
    resp = client.get("/site/config?site_id=741_73")
    assert resp.status_code == 200
    loaded = resp.json()
    assert loaded["site_id"] == "741_73"
    assert len(loaded["roi_polygon"]) == 4
    assert "north" in loaded["entry_exit_lines"]
    assert "south" in loaded["entry_exit_lines"]

    # List
    resp = client.get("/site/configs")
    assert "741_73" in resp.json()["sites"]


def test_multi_channel_isolation(client, app):
    """Alerts and frames for different channels don't leak."""
    backend = app.state.backend
    alert_store = app.state.alert_store
    mjpeg = app.state.mjpeg

    client.post("/pipeline/start")
    client.post("/channel/add", json={"source": "/data/a.mp4"})
    client.post("/channel/add", json={"source": "/data/b.mp4"})

    # Subscribe to channel 0 MJPEG only
    q0 = mjpeg.subscribe(0)
    q1 = mjpeg.subscribe(1)

    # Emit frame on channel 1 — only q1 should get it
    backend.emit_frame(channel_id=1, frame_number=1, jpeg=b"ch1")
    assert q0.empty()
    assert q1.get_nowait() == b"ch1"

    # Add alerts on different channels
    backend.emit_alert(
        {
            "track_id": 1,
            "label": "car",
            "entry_arm": "n",
            "entry_label": "N",
            "exit_arm": "s",
            "exit_label": "S",
            "method": "confirmed",
            "was_stagnant": False,
            "first_seen_frame": 0,
            "last_seen_frame": 100,
            "duration_frames": 100,
            "trajectory": [],
            "per_frame_data": [],
            "channel": 0,
        }
    )
    backend.emit_alert(
        {
            "track_id": 2,
            "label": "truck",
            "entry_arm": "e",
            "entry_label": "E",
            "exit_arm": "w",
            "exit_label": "W",
            "method": "confirmed",
            "was_stagnant": False,
            "first_seen_frame": 0,
            "last_seen_frame": 200,
            "duration_frames": 200,
            "trajectory": [],
            "per_frame_data": [],
            "channel": 1,
        }
    )

    # Remove channel 0 — only its alerts cleared
    client.post("/channel/remove", json={"channel_id": 0})
    remaining = alert_store.get_alerts()
    assert len(remaining) == 1
    assert remaining[0]["channel"] == 1

    client.post("/pipeline/stop")
