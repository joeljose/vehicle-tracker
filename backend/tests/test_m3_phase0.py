"""Tests for M3 Phase 0: Backend prerequisites for React UI."""

import time

import pytest

from backend.api.websocket import WsBroadcaster
from backend.pipeline.protocol import Detection, FrameResult
from backend.tests.helpers import ANALYTICS_PHASE_BODY


# -- Fixtures --

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
            "timestamp_ms": 3333,
        },
        {
            "frame": 200,
            "bbox": (50, 60, 30, 40),
            "centroid": (65, 80),
            "confidence": 0.85,
            "timestamp_ms": 6666,
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
def started_client(client):
    """Client with pipeline started and one channel added."""
    client.post("/pipeline/start")
    client.post("/channel/add", json={"source": "/data/video.mp4"})
    return client


@pytest.fixture
def seeded_store(app):
    """AlertStore with alerts on two channels."""
    store = app.state.alert_store
    store.add_transit_alert(TRANSIT_ALERT, channel=0)
    store.add_stagnant_alert(STAGNANT_ALERT, channel=0)
    store.add_transit_alert({**TRANSIT_ALERT, "track_id": 30}, channel=1)
    return store


# -- GET /channels --


class TestGetChannels:
    def test_no_pipeline(self, client):
        resp = client.get("/channels")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pipeline_started"] is False
        assert data["channels"] == []

    def test_one_channel(self, started_client, app):
        resp = started_client.get("/channels")
        data = resp.json()
        assert data["pipeline_started"] is True
        assert len(data["channels"]) == 1
        ch = data["channels"][0]
        assert ch["channel_id"] == 0
        assert ch["source"] == "/data/video.mp4"
        assert ch["phase"] == "setup"
        assert ch["alert_count"] == 0

    def test_two_channels_different_phases(self, started_client, app):
        started_client.post("/channel/add", json={"source": "/data/b.mp4"})
        started_client.post("/channel/0/phase", json=ANALYTICS_PHASE_BODY)
        # Seed an alert on channel 0
        app.state.alert_store.add_transit_alert(TRANSIT_ALERT, channel=0)

        resp = started_client.get("/channels")
        channels = resp.json()["channels"]
        assert len(channels) == 2
        ch0 = next(c for c in channels if c["channel_id"] == 0)
        ch1 = next(c for c in channels if c["channel_id"] == 1)
        assert ch0["phase"] == "analytics"
        assert ch0["alert_count"] == 1
        assert ch1["phase"] == "setup"
        assert ch1["alert_count"] == 0


# -- GET /channel/{id} --


class TestGetChannel:
    def test_returns_channel_state(self, started_client):
        resp = started_client.get("/channel/0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["channel_id"] == 0
        assert data["source"] == "/data/video.mp4"
        assert data["phase"] == "setup"
        assert data["alert_count"] == 0
        assert data["pipeline_started"] is True

    def test_not_found(self, started_client):
        resp = started_client.get("/channel/99")
        assert resp.status_code == 404

    def test_includes_alert_count(self, started_client, app):
        app.state.alert_store.add_transit_alert(TRANSIT_ALERT, channel=0)
        resp = started_client.get("/channel/0")
        assert resp.json()["alert_count"] == 1

    def test_reflects_phase_change(self, started_client):
        started_client.post("/channel/0/phase", json=ANALYTICS_PHASE_BODY)
        resp = started_client.get("/channel/0")
        assert resp.json()["phase"] == "analytics"


# -- Channel filter on GET /alerts --


class TestAlertChannelFilter:
    def test_filter_by_channel(self, client, seeded_store):
        resp = client.get("/alerts?channel=0")
        alerts = resp.json()
        assert len(alerts) == 2
        assert all(a["channel"] == 0 for a in alerts)

    def test_filter_channel_1(self, client, seeded_store):
        resp = client.get("/alerts?channel=1")
        alerts = resp.json()
        assert len(alerts) == 1
        assert alerts[0]["channel"] == 1

    def test_no_filter_returns_all(self, client, seeded_store):
        resp = client.get("/alerts?limit=200")
        assert len(resp.json()) == 3

    def test_combined_channel_and_type(self, client, seeded_store):
        resp = client.get("/alerts?channel=0&type=transit_alert")
        alerts = resp.json()
        assert len(alerts) == 1
        assert alerts[0]["type"] == "transit_alert"
        assert alerts[0]["channel"] == 0

    def test_empty_channel(self, client, seeded_store):
        resp = client.get("/alerts?channel=99")
        assert resp.json() == []


# -- WebSocket types filter --


class TestWsTypesFilter:
    @staticmethod
    def _client(broadcaster, channels=None, types=None):
        from unittest.mock import MagicMock
        from backend.api.websocket import _ClientChannel
        c = _ClientChannel(ws=MagicMock(), channels=channels, types=types)
        broadcaster._clients.append(c)
        return c

    def test_type_filter_allows_matching(self):
        broadcaster = WsBroadcaster()
        c = self._client(broadcaster, types={"transit_alert"})
        broadcaster._fanout({"type": "transit_alert", "channel": 0, "alert_id": "abc"})
        assert c.queue.qsize() == 1

    def test_type_filter_blocks_non_matching(self):
        broadcaster = WsBroadcaster()
        c = self._client(broadcaster, types={"transit_alert"})
        broadcaster._fanout({"type": "frame_data", "channel": 0, "frame": 1})
        assert c.queue.qsize() == 0

    def test_no_type_filter_receives_all(self):
        broadcaster = WsBroadcaster()
        c = self._client(broadcaster)
        for msg_type in ["frame_data", "transit_alert", "pipeline_event"]:
            broadcaster._fanout({"type": msg_type})
        assert c.queue.qsize() == 3

    def test_combined_channel_and_type_filter(self):
        broadcaster = WsBroadcaster()
        c = self._client(broadcaster, channels={0}, types={"transit_alert"})

        # Channel 0, transit_alert — should pass
        broadcaster._fanout({"type": "transit_alert", "channel": 0})
        assert c.queue.qsize() == 1

        # Channel 1, transit_alert — blocked by channel filter
        broadcaster._fanout({"type": "transit_alert", "channel": 1})
        assert c.queue.qsize() == 1

        # Channel 0, frame_data — blocked by type filter
        broadcaster._fanout({"type": "frame_data", "channel": 0})
        assert c.queue.qsize() == 1

    def test_ws_endpoint_parses_types_param(self, client):
        with client.websocket_connect("/ws?types=transit_alert,stagnant_alert"):
            pass


class TestWsHeadOfLineIsolation:
    """A slow WS client must not block delivery to fast clients (B8)."""

    def test_slow_client_does_not_starve_fast_client(self):
        """One client whose send_text is artificially slow should not delay
        the queue of another client. Each client owns its own queue + task,
        so the fast client's queue receives the message synchronously inside
        ``_fanout`` regardless of the slow one's progress."""
        from unittest.mock import MagicMock
        from backend.api.websocket import WsBroadcaster, _ClientChannel

        broadcaster = WsBroadcaster()
        slow = _ClientChannel(ws=MagicMock(), channels=None, types=None)
        fast = _ClientChannel(ws=MagicMock(), channels=None, types=None)
        broadcaster._clients = [slow, fast]

        # Don't start tasks — just verify the synchronous fan-out step puts
        # the message in BOTH queues even if the slow client never drains.
        for i in range(50):
            broadcaster._fanout({"type": "frame_data", "frame": i})

        assert slow.queue.qsize() == 50
        assert fast.queue.qsize() == 50

    def test_slow_client_drops_oldest_when_full(self):
        """When a single client's queue fills up, the broadcaster drops the
        oldest message from THAT client's queue only — without affecting any
        other client or back-pressuring producers."""
        from unittest.mock import MagicMock
        from backend.api.websocket import WsBroadcaster, _CLIENT_QUEUE_MAXSIZE, _ClientChannel

        broadcaster = WsBroadcaster()
        slow = _ClientChannel(ws=MagicMock(), channels=None, types=None)
        broadcaster._clients = [slow]

        # Push twice the queue's capacity; producers never block.
        n = _CLIENT_QUEUE_MAXSIZE * 2
        for i in range(n):
            broadcaster._fanout({"type": "frame_data", "frame": i})

        # Queue is full but bounded — no exception, no producer stall.
        assert slow.queue.qsize() == _CLIENT_QUEUE_MAXSIZE

    def test_dead_send_does_not_break_other_clients(self):
        """If one client's send_text raises (TCP close, etc.), the other
        client's task continues uninterrupted."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from backend.api.websocket import WsBroadcaster, _ClientChannel

        async def scenario():
            broadcaster = WsBroadcaster()
            await broadcaster.start()

            ws_dead = MagicMock()
            ws_dead.send_text = AsyncMock(side_effect=RuntimeError("connection closed"))
            ws_alive = MagicMock()
            ws_alive.send_text = AsyncMock()

            dead_client = _ClientChannel(ws=ws_dead, channels=None, types=None)
            alive_client = _ClientChannel(ws=ws_alive, channels=None, types=None)
            dead_client.task = asyncio.create_task(broadcaster._client_loop(dead_client))
            alive_client.task = asyncio.create_task(broadcaster._client_loop(alive_client))
            broadcaster._clients = [dead_client, alive_client]

            broadcaster._fanout({"type": "ping"})
            broadcaster._fanout({"type": "pong"})

            # Give both client tasks a turn to drain their queues.
            await asyncio.sleep(0.05)

            assert ws_alive.send_text.call_count == 2
            # Dead client errors out on first send, second message is then
            # left in its queue (task already exited).
            assert ws_dead.send_text.call_count == 1
            assert not dead_client.alive
            assert dead_client not in broadcaster._clients

            await broadcaster.stop()

        asyncio.run(scenario())


# -- phase_changed WS event --


class TestPhaseChangedEvent:
    def test_phase_change_emits_event(self, started_client, app):
        ws = app.state.ws
        # Drain queue
        while not ws._queue.empty():
            ws._queue.get_nowait()

        started_client.post("/channel/0/phase", json=ANALYTICS_PHASE_BODY)
        msg = ws._queue.get_nowait()
        assert msg["type"] == "phase_changed"
        assert msg["channel"] == 0
        assert msg["phase"] == "analytics"
        assert msg["previous_phase"] == "setup"

    def test_phase_change_review(self, started_client, app):
        ws = app.state.ws
        started_client.post("/channel/0/phase", json=ANALYTICS_PHASE_BODY)
        while not ws._queue.empty():
            ws._queue.get_nowait()

        started_client.post("/channel/0/phase", json={"phase": "review"})
        msg = ws._queue.get_nowait()
        assert msg["type"] == "phase_changed"
        assert msg["previous_phase"] == "analytics"
        assert msg["phase"] == "review"


# -- stats_update WS message --


class TestStatsUpdate:
    def test_stats_emitted_after_1_second(self, app):
        ws = app.state.ws
        # Drain
        while not ws._queue.empty():
            ws._queue.get_nowait()

        # Force the last emit to be >1s ago
        ws._stats_last_emit[0] = time.monotonic() - 2.0
        ws._stats_frame_count[0] = 0
        ws._stats_inference_ms_sum[0] = 0.0

        result = FrameResult(
            channel_id=0,
            frame_number=100,
            timestamp_ms=3333,
            detections=[
                Detection(
                    track_id=1,
                    class_name="car",
                    bbox=(10, 20, 30, 40),
                    confidence=0.9,
                    centroid=(25, 40),
                ),
            ],
            annotated_jpeg=b"\xff\xd8test",
            inference_ms=5.2,
            phase="analytics",
            idle_mode=False,
        )
        ws.on_frame(result)

        # Should have frame_data + stats_update
        messages = []
        while not ws._queue.empty():
            messages.append(ws._queue.get_nowait())
        types = [m["type"] for m in messages]
        assert "frame_data" in types
        assert "stats_update" in types

        stats = next(m for m in messages if m["type"] == "stats_update")
        assert stats["channel"] == 0
        assert stats["active_tracks"] == 1
        assert stats["phase"] == "analytics"
        assert stats["idle_mode"] is False
        assert "fps" in stats
        assert "inference_ms" in stats

    def test_stats_not_emitted_before_1_second(self, app):
        ws = app.state.ws
        while not ws._queue.empty():
            ws._queue.get_nowait()

        # Set last emit to now
        ws._stats_last_emit[0] = time.monotonic()
        ws._stats_frame_count[0] = 0
        ws._stats_inference_ms_sum[0] = 0.0

        result = FrameResult(
            channel_id=0,
            frame_number=1,
            timestamp_ms=33,
            detections=[],
            annotated_jpeg=b"\xff\xd8test",
            inference_ms=3.0,
            phase="setup",
            idle_mode=False,
        )
        ws.on_frame(result)

        messages = []
        while not ws._queue.empty():
            messages.append(ws._queue.get_nowait())
        types = [m["type"] for m in messages]
        assert "stats_update" not in types
        assert "frame_data" in types


# -- timestamp_ms in per_frame_data --


class TestTimestampMsInPerFrameData:
    def test_transit_alert_includes_timestamp_ms(self, app):
        store = app.state.alert_store
        alert_id = store.add_transit_alert(TRANSIT_ALERT, channel=0)
        full = store.get_alert(alert_id)
        for entry in full["per_frame_data"]:
            assert "timestamp_ms" in entry
        assert full["per_frame_data"][0]["timestamp_ms"] == 3333
        assert full["per_frame_data"][1]["timestamp_ms"] == 6666

    def test_timestamp_ms_defaults_to_zero(self, app):
        store = app.state.alert_store
        alert_no_ts = {
            **TRANSIT_ALERT,
            "per_frame_data": [
                {
                    "frame": 1,
                    "bbox": (0, 0, 10, 10),
                    "centroid": (5, 5),
                    "confidence": 0.8,
                },
            ],
        }
        alert_id = store.add_transit_alert(alert_no_ts, channel=0)
        full = store.get_alert(alert_id)
        assert full["per_frame_data"][0]["timestamp_ms"] == 0


# -- SiteConfigRequest Pydantic model --


class TestSiteConfigValidation:
    @pytest.fixture(autouse=True)
    def _redirect_sites(self, tmp_path, monkeypatch):
        import backend.config.site_config as sc

        monkeypatch.setattr(sc, "SITES_DIR", tmp_path)

    def test_valid_config(self, client):
        resp = client.post(
            "/site/config",
            json={
                "site_id": "junction_a",
                "roi_polygon": [[100, 200], [300, 400], [500, 600]],
                "entry_exit_lines": {
                    "north": {
                        "label": "741-North",
                        "start": [100, 50],
                        "end": [200, 50],
                    },
                },
            },
        )
        assert resp.status_code == 200
        assert resp.json() == {"status": "saved"}

    def test_missing_site_id(self, client):
        resp = client.post(
            "/site/config",
            json={
                "roi_polygon": [[0, 0], [1, 1], [2, 2]],
            },
        )
        assert resp.status_code == 422

    def test_too_few_vertices(self, client):
        resp = client.post(
            "/site/config",
            json={
                "site_id": "bad",
                "roi_polygon": [[0, 0], [1, 1]],
            },
        )
        assert resp.status_code == 422

    def test_empty_polygon(self, client):
        resp = client.post(
            "/site/config",
            json={
                "site_id": "bad",
                "roi_polygon": [],
            },
        )
        assert resp.status_code == 422

    def test_no_entry_exit_lines(self, client):
        resp = client.post(
            "/site/config",
            json={
                "site_id": "minimal",
                "roi_polygon": [[0, 0], [1, 1], [2, 2]],
            },
        )
        assert resp.status_code == 200

    def test_invalid_entry_exit_line(self, client):
        resp = client.post(
            "/site/config",
            json={
                "site_id": "bad",
                "roi_polygon": [[0, 0], [1, 1], [2, 2]],
                "entry_exit_lines": {
                    "north": {"start": [100, 50]},  # missing label and end
                },
            },
        )
        assert resp.status_code == 422


# -- CORS --


class TestCors:
    def test_cors_headers_present(self, client):
        resp = client.options(
            "/channels",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert (
            resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
        )

    def test_cors_rejects_other_origin(self, client):
        resp = client.options(
            "/channels",
            headers={
                "Origin": "http://evil.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") != "http://evil.com"


# -- AlertStore.count_by_channel --


class TestCountByChannel:
    def test_count_empty(self, app):
        assert app.state.alert_store.count_by_channel(0) == 0

    def test_count_with_alerts(self, app, seeded_store):
        assert seeded_store.count_by_channel(0) == 2
        assert seeded_store.count_by_channel(1) == 1
        assert seeded_store.count_by_channel(99) == 0
