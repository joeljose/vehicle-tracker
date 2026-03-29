"""Tests for M3 Phase 6: Replay backend — clip extraction + replay endpoint."""

from pathlib import Path
from unittest.mock import patch

import pytest


TRANSIT_ALERT = {
    "track_id": 10,
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
    "trajectory": [(100, 200), (150, 250)],
    "per_frame_data": [
        {"frame": 100, "bbox": (10, 20, 30, 40), "centroid": (25, 40),
         "confidence": 0.9, "timestamp_ms": 3333},
        {"frame": 400, "bbox": (50, 60, 30, 40), "centroid": (65, 80),
         "confidence": 0.85, "timestamp_ms": 13333},
    ],
}

STAGNANT_ALERT = {
    "track_id": 20,
    "label": "truck",
    "position": (450, 320),
    "stationary_duration_frames": 4500,
    "first_seen_frame": 0,
    "last_seen_frame": 4500,
}


@pytest.fixture
def started_client(client):
    """Client with pipeline started and one channel added."""
    client.post("/pipeline/start")
    client.post("/channel/add", json={"source": "/data/video.mp4"})
    return client


@pytest.fixture
def seeded_alerts(app):
    """AlertStore with transit + stagnant alerts on channel 0."""
    store = app.state.alert_store
    t_id = store.add_transit_alert(TRANSIT_ALERT, channel=0)
    s_id = store.add_stagnant_alert(STAGNANT_ALERT, channel=0)
    return t_id, s_id


# -- ClipExtractor unit tests --

class TestClipExtractor:
    def test_initial_status_none(self, app):
        extractor = app.state.clip_extractor
        assert extractor.get_status("nonexistent") is None

    def test_get_clip_path_none_when_not_ready(self, app):
        extractor = app.state.clip_extractor
        assert extractor.get_clip_path("nonexistent") is None

    def test_cleanup_channel_no_error_when_empty(self, app):
        extractor = app.state.clip_extractor
        extractor.cleanup_channel(99)  # should not raise

    def test_cleanup_all(self, app):
        extractor = app.state.clip_extractor
        extractor._status["test"] = "ready"
        extractor._clip_paths["test"] = Path("/tmp/test.mp4")
        extractor.cleanup_all()
        assert extractor.get_status("test") is None
        assert extractor.get_clip_path("test") is None


# -- AlertStore.get_channel_alerts --

class TestGetChannelAlerts:
    def test_returns_full_alerts(self, app, seeded_alerts):
        store = app.state.alert_store
        alerts = store.get_channel_alerts(0)
        assert len(alerts) == 2
        # Full alerts include per_frame_data
        transit = next(a for a in alerts if a["type"] == "transit_alert")
        assert "per_frame_data" in transit

    def test_empty_channel(self, app):
        store = app.state.alert_store
        assert store.get_channel_alerts(99) == []


# -- Replay endpoint --

class TestReplayEndpoint:
    def test_alert_not_found(self, client):
        resp = client.get("/alert/nonexistent/replay")
        assert resp.status_code == 404

    def test_202_when_not_started(self, started_client, seeded_alerts):
        t_id, _ = seeded_alerts
        resp = started_client.get(f"/alert/{t_id}/replay")
        assert resp.status_code == 202
        assert resp.json()["status"] == "not_started"

    def test_202_when_pending(self, started_client, app, seeded_alerts):
        t_id, _ = seeded_alerts
        app.state.clip_extractor._status[t_id] = "pending"
        resp = started_client.get(f"/alert/{t_id}/replay")
        assert resp.status_code == 202
        assert resp.json()["status"] == "extracting"

    def test_500_when_failed(self, started_client, app, seeded_alerts):
        t_id, _ = seeded_alerts
        app.state.clip_extractor._status[t_id] = "failed"
        resp = started_client.get(f"/alert/{t_id}/replay")
        assert resp.status_code == 500

    def test_200_transit_clip_ready(self, started_client, app, seeded_alerts, tmp_path):
        t_id, _ = seeded_alerts
        # Create a fake clip file
        clip = tmp_path / f"clip_{t_id}.mp4"
        clip.write_bytes(b"\x00\x00\x00\x1cftyp")  # minimal mp4 header
        app.state.clip_extractor._status[t_id] = "ready"
        app.state.clip_extractor._clip_paths[t_id] = clip
        resp = started_client.get(f"/alert/{t_id}/replay")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "video/mp4"

    def test_200_stagnant_frame_ready(self, started_client, app, seeded_alerts, tmp_path):
        _, s_id = seeded_alerts
        frame = tmp_path / f"frame_{s_id}.jpg"
        frame.write_bytes(b"\xff\xd8\xff\xe0")  # minimal JPEG header
        app.state.clip_extractor._status[s_id] = "ready"
        app.state.clip_extractor._clip_paths[s_id] = frame
        resp = started_client.get(f"/alert/{s_id}/replay")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"


# -- Phase transition triggers clip extraction --

class TestPhaseTransitionClipExtraction:
    def test_analytics_to_review_triggers_extraction(self, started_client, app, seeded_alerts):
        # Move to analytics first
        started_client.post("/channel/0/phase", json={"phase": "analytics"})

        with patch.object(app.state.clip_extractor, "extract_clips") as mock:
            started_client.post("/channel/0/phase", json={"phase": "review"})
            mock.assert_called_once()
            args = mock.call_args
            assert args[0][0] == 0  # channel_id
            assert args[0][1] == "/data/video.mp4"  # source
            assert len(args[0][2]) == 2  # 2 alerts

    def test_setup_to_analytics_no_extraction(self, started_client, app):
        with patch.object(app.state.clip_extractor, "extract_clips") as mock:
            started_client.post("/channel/0/phase", json={"phase": "analytics"})
            mock.assert_not_called()


# -- Cleanup on channel remove --

class TestCleanupOnRemove:
    def test_channel_remove_cleans_clips(self, started_client, app):
        with patch.object(app.state.clip_extractor, "cleanup_channel") as mock:
            started_client.post("/channel/remove", json={"channel_id": 0})
            mock.assert_called_once_with(0)

    def test_pipeline_stop_cleans_all_clips(self, started_client, app):
        with patch.object(app.state.clip_extractor, "cleanup_all") as mock:
            started_client.post("/pipeline/stop")
            mock.assert_called_once()
