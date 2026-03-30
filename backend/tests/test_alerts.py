"""Tests for AlertStore — in-memory alert storage and querying."""

from backend.pipeline.alerts import AlertStore


def _make_transit_alert(**overrides) -> dict:
    """Create a transit alert dict matching pipeline output."""
    alert = {
        "track_id": 567,
        "label": "car",
        "entry_arm": "north",
        "entry_label": "741-North",
        "exit_arm": "south",
        "exit_label": "741-South",
        "method": "confirmed",
        "was_stagnant": False,
        "first_seen_frame": 1001,
        "last_seen_frame": 1296,
        "duration_frames": 295,
        "trajectory": [(470, 338, 1001), (473, 340, 1002)],
        "per_frame_data": [
            {"frame": 1001, "bbox": (412, 305, 129, 73), "centroid": (476, 341), "confidence": 0.89},
            {"frame": 1002, "bbox": (414, 306, 130, 74), "centroid": (479, 343), "confidence": 0.95},
        ],
    }
    alert.update(overrides)
    return alert


def _make_stagnant_alert(**overrides) -> dict:
    """Create a stagnant alert dict."""
    alert = {
        "track_id": 571,
        "label": "truck",
        "position": (450, 320),
        "stationary_duration_frames": 4500,
        "first_seen_frame": 500,
        "last_seen_frame": 5000,
    }
    alert.update(overrides)
    return alert


class TestAddAlerts:
    """Test adding alerts to the store."""

    def test_add_transit_alert(self):
        store = AlertStore()
        alert_id = store.add_transit_alert(_make_transit_alert(), channel=0)
        assert alert_id is not None
        assert len(alert_id) == 8  # short hash

    def test_add_stagnant_alert(self):
        store = AlertStore()
        alert_id = store.add_stagnant_alert(_make_stagnant_alert(), channel=0)
        assert alert_id is not None

    def test_alert_ids_are_unique(self):
        store = AlertStore()
        id1 = store.add_transit_alert(_make_transit_alert(track_id=1), channel=0)
        id2 = store.add_transit_alert(_make_transit_alert(track_id=2), channel=0)
        assert id1 != id2


class TestGetAlert:
    """Test fetching full alert metadata."""

    def test_get_transit_alert_full(self):
        store = AlertStore()
        alert_id = store.add_transit_alert(_make_transit_alert(), channel=0)
        alert = store.get_alert(alert_id)

        assert alert is not None
        assert alert["alert_id"] == alert_id
        assert alert["type"] == "transit_alert"
        assert alert["channel"] == 0
        assert alert["track_id"] == "567"
        assert alert["class"] == "car"
        assert alert["entry_direction"] == "north"
        assert alert["entry_label"] == "741-North"
        assert alert["exit_direction"] == "south"
        assert alert["exit_label"] == "741-South"
        assert alert["method"] == "confirmed"
        assert alert["best_photo_url"] == "/snapshot/567"
        assert alert["duration_ms"] > 0
        assert alert["timestamp_ms"] > 0
        assert "per_frame_data" in alert
        assert "full_trajectory" in alert

    def test_get_stagnant_alert_full(self):
        store = AlertStore()
        alert_id = store.add_stagnant_alert(_make_stagnant_alert(), channel=0)
        alert = store.get_alert(alert_id)

        assert alert is not None
        assert alert["type"] == "stagnant_alert"
        assert alert["track_id"] == "571"
        assert alert["class"] == "truck"
        assert alert["best_photo_url"] == "/snapshot/571"
        assert alert["stationary_duration_ms"] > 0
        assert alert["position"] == [450, 320]

    def test_get_unknown_alert_returns_none(self):
        store = AlertStore()
        assert store.get_alert("nonexistent") is None


class TestGetAlerts:
    """Test paginated alert listing."""

    def test_list_alerts_default(self):
        store = AlertStore()
        for i in range(5):
            store.add_transit_alert(_make_transit_alert(track_id=i), channel=0)
        alerts = store.get_alerts()
        assert len(alerts) == 5

    def test_list_alerts_limit(self):
        store = AlertStore()
        for i in range(10):
            store.add_transit_alert(_make_transit_alert(track_id=i), channel=0)
        alerts = store.get_alerts(limit=3)
        assert len(alerts) == 3

    def test_list_alerts_newest_first(self):
        store = AlertStore()
        store.add_transit_alert(_make_transit_alert(track_id=1), channel=0)
        store.add_transit_alert(_make_transit_alert(track_id=2), channel=0)
        alerts = store.get_alerts()
        assert alerts[0]["track_id"] == "2"  # newest first

    def test_filter_by_type_transit(self):
        store = AlertStore()
        store.add_transit_alert(_make_transit_alert(track_id=1), channel=0)
        store.add_stagnant_alert(_make_stagnant_alert(track_id=2), channel=0)
        store.add_transit_alert(_make_transit_alert(track_id=3), channel=0)

        transit = store.get_alerts(alert_type="transit_alert")
        assert len(transit) == 2
        assert all(a["type"] == "transit_alert" for a in transit)

    def test_filter_by_type_stagnant(self):
        store = AlertStore()
        store.add_transit_alert(_make_transit_alert(track_id=1), channel=0)
        store.add_stagnant_alert(_make_stagnant_alert(track_id=2), channel=0)

        stagnant = store.get_alerts(alert_type="stagnant_alert")
        assert len(stagnant) == 1
        assert stagnant[0]["type"] == "stagnant_alert"

    def test_list_returns_summaries_not_full(self):
        """Listed alerts should not include per_frame_data (expensive)."""
        store = AlertStore()
        store.add_transit_alert(_make_transit_alert(), channel=0)
        alerts = store.get_alerts()
        assert "per_frame_data" not in alerts[0]
        assert "full_trajectory" not in alerts[0]

    def test_empty_store(self):
        store = AlertStore()
        assert store.get_alerts() == []


class TestWebSocketSummary:
    """Test that alert summaries match the WebSocket push schema."""

    def test_transit_summary_schema(self):
        store = AlertStore()
        alert_id = store.add_transit_alert(_make_transit_alert(), channel=0)
        summary = store.get_ws_summary(alert_id)

        assert summary["type"] == "transit_alert"
        assert summary["channel"] == 0
        assert summary["alert_id"] == alert_id
        assert summary["track_id"] == "567"
        assert summary["class"] == "car"
        assert summary["entry_direction"] == "north"
        assert summary["entry_label"] == "741-North"
        assert summary["exit_direction"] == "south"
        assert summary["exit_label"] == "741-South"
        assert summary["method"] == "confirmed"
        assert summary["best_photo_url"] == "/snapshot/567"
        assert summary["duration_ms"] > 0
        assert summary["timestamp_ms"] > 0
        # Should NOT have heavy fields
        assert "per_frame_data" not in summary
        assert "full_trajectory" not in summary

    def test_stagnant_summary_schema(self):
        store = AlertStore()
        alert_id = store.add_stagnant_alert(_make_stagnant_alert(), channel=0)
        summary = store.get_ws_summary(alert_id)

        assert summary["type"] == "stagnant_alert"
        assert summary["channel"] == 0
        assert summary["track_id"] == "571"
        assert summary["class"] == "truck"
        assert summary["best_photo_url"] == "/snapshot/571"
        assert summary["stationary_duration_ms"] > 0
        assert summary["position"] == [450, 320]


class TestClear:
    """Test clearing alerts for a channel."""

    def test_clear_channel(self):
        store = AlertStore()
        store.add_transit_alert(_make_transit_alert(track_id=1), channel=0)
        store.add_transit_alert(_make_transit_alert(track_id=2), channel=1)
        store.clear_channel(0)
        alerts = store.get_alerts()
        assert len(alerts) == 1
        assert alerts[0]["channel"] == 1

    def test_clear_all(self):
        store = AlertStore()
        store.add_transit_alert(_make_transit_alert(track_id=1), channel=0)
        store.add_transit_alert(_make_transit_alert(track_id=2), channel=1)
        store.clear()
        assert store.get_alerts() == []
