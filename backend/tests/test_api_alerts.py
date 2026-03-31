"""Tests for alert + snapshot REST endpoints."""

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
    "trajectory": [(100, 200), (150, 250), (200, 300)],
    "per_frame_data": [
        {
            "frame": 100,
            "bbox": (10, 20, 30, 40),
            "centroid": (25, 40),
            "confidence": 0.9,
        },
        {
            "frame": 200,
            "bbox": (50, 60, 30, 40),
            "centroid": (65, 80),
            "confidence": 0.85,
        },
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
def seeded_store(app):
    """Pre-populate AlertStore with one transit and one stagnant alert."""
    store = app.state.alert_store
    store.add_transit_alert(TRANSIT_ALERT, channel=0)
    store.add_stagnant_alert(STAGNANT_ALERT, channel=0)
    return store


class TestListAlerts:
    def test_list_default(self, client, seeded_store):
        resp = client.get("/alerts")
        assert resp.status_code == 200
        alerts = resp.json()
        assert len(alerts) == 2
        # Newest first
        assert alerts[0]["type"] == "stagnant_alert"
        assert alerts[1]["type"] == "transit_alert"

    def test_list_filter_transit(self, client, seeded_store):
        resp = client.get("/alerts?type=transit_alert")
        alerts = resp.json()
        assert len(alerts) == 1
        assert alerts[0]["type"] == "transit_alert"

    def test_list_filter_stagnant(self, client, seeded_store):
        resp = client.get("/alerts?type=stagnant_alert")
        alerts = resp.json()
        assert len(alerts) == 1
        assert alerts[0]["type"] == "stagnant_alert"

    def test_list_limit(self, client, seeded_store):
        resp = client.get("/alerts?limit=1")
        alerts = resp.json()
        assert len(alerts) == 1

    def test_list_empty(self, client):
        resp = client.get("/alerts")
        assert resp.json() == []

    def test_summaries_exclude_heavy_fields(self, client, seeded_store):
        resp = client.get("/alerts")
        transit = next(a for a in resp.json() if a["type"] == "transit_alert")
        assert "per_frame_data" not in transit
        assert "full_trajectory" not in transit


class TestGetAlert:
    def test_get_transit_full(self, client, seeded_store):
        # Get alert_id from listing
        alerts = client.get("/alerts?type=transit_alert").json()
        alert_id = alerts[0]["alert_id"]
        resp = client.get(f"/alert/{alert_id}")
        assert resp.status_code == 200
        full = resp.json()
        assert full["type"] == "transit_alert"
        assert full["track_id"] == "10"
        assert full["entry_direction"] == "north"
        assert full["exit_direction"] == "south"
        assert "per_frame_data" in full
        assert "full_trajectory" in full
        assert len(full["per_frame_data"]) == 2

    def test_get_stagnant_full(self, client, seeded_store):
        alerts = client.get("/alerts?type=stagnant_alert").json()
        alert_id = alerts[0]["alert_id"]
        resp = client.get(f"/alert/{alert_id}")
        assert resp.status_code == 200
        full = resp.json()
        assert full["type"] == "stagnant_alert"
        assert full["track_id"] == "20"

    def test_get_unknown_alert(self, client):
        resp = client.get("/alert/nonexistent")
        assert resp.status_code == 404


class TestSnapshot:
    def test_get_snapshot(self, client, app):
        # Pre-load a snapshot in the fake backend
        app.state.backend._snapshots[42] = b"\xff\xd8\xff\xe0jpeg-data"
        resp = client.get("/snapshot/42")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert resp.content == b"\xff\xd8\xff\xe0jpeg-data"

    def test_get_snapshot_not_found(self, client):
        resp = client.get("/snapshot/999")
        assert resp.status_code == 404
