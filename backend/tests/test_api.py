"""Integration tests for FastAPI pipeline control endpoints."""

import queue

from backend.tests.helpers import ANALYTICS_PHASE_BODY


class TestPipelineLifecycle:
    def test_start_pipeline(self, client):
        resp = client.post("/pipeline/start")
        assert resp.status_code == 200
        assert resp.json() == {"status": "started"}

    def test_start_already_started(self, client):
        client.post("/pipeline/start")
        resp = client.post("/pipeline/start")
        assert resp.status_code == 409

    def test_stop_pipeline(self, client):
        client.post("/pipeline/start")
        resp = client.post("/pipeline/stop")
        assert resp.status_code == 200
        assert resp.json() == {"status": "stopped"}

    def test_stop_not_started(self, client):
        resp = client.post("/pipeline/stop")
        assert resp.status_code == 409


class TestChannelManagement:
    def test_add_channel(self, client):
        client.post("/pipeline/start")
        resp = client.post("/channel/add", json={"source": "/data/video.mp4"})
        assert resp.status_code == 200
        assert resp.json() == {"channel_id": 0}

    def test_add_channel_auto_increment(self, client):
        client.post("/pipeline/start")
        r1 = client.post("/channel/add", json={"source": "/data/a.mp4"})
        r2 = client.post("/channel/add", json={"source": "/data/b.mp4"})
        assert r1.json()["channel_id"] == 0
        assert r2.json()["channel_id"] == 1

    def test_add_channel_pipeline_not_started(self, client):
        resp = client.post("/channel/add", json={"source": "/data/video.mp4"})
        assert resp.status_code == 409

    def test_add_channel_source_not_found(self, client):
        client.post("/pipeline/start")
        resp = client.post(
            "/channel/add",
            json={"source": "/nonexistent/path/video.mp4"},
        )
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"].lower()

    def test_add_channel_youtube_url_returns_202(self, client):
        """YouTube URLs trigger async resolution and return 202."""
        client.post("/pipeline/start")
        resp = client.post(
            "/channel/add",
            json={"source": "https://youtube.com/watch?v=abc"},
        )
        assert resp.status_code == 202
        assert resp.json()["status"] == "resolving"

    def test_add_channel_non_youtube_url_skips_validation(self, client):
        """Non-YouTube URLs (e.g. RTSP) skip file validation."""
        client.post("/pipeline/start")
        resp = client.post(
            "/channel/add",
            json={"source": "rtsp://camera.local/stream"},
        )
        assert resp.status_code == 200

    def test_remove_channel(self, client):
        client.post("/pipeline/start")
        client.post("/channel/add", json={"source": "/data/video.mp4"})
        resp = client.post("/channel/remove", json={"channel_id": 0})
        assert resp.status_code == 200
        assert resp.json() == {"status": "removed"}

    def test_remove_nonexistent_channel(self, client):
        client.post("/pipeline/start")
        resp = client.post("/channel/remove", json={"channel_id": 99})
        assert resp.status_code == 404

    def test_remove_clears_alerts(self, client, app):
        client.post("/pipeline/start")
        client.post("/channel/add", json={"source": "/data/video.mp4"})
        # Manually add an alert to the store
        alert_store = app.state.alert_store
        alert_store.add_transit_alert(
            {
                "track_id": 1,
                "label": "car",
                "entry_arm": "north",
                "entry_label": "N",
                "exit_arm": "south",
                "exit_label": "S",
                "method": "confirmed",
                "was_stagnant": False,
                "first_seen_frame": 0,
                "last_seen_frame": 100,
                "duration_frames": 100,
                "trajectory": [],
                "per_frame_data": [],
            },
            channel=0,
        )
        assert len(alert_store.get_alerts()) == 1
        client.post("/channel/remove", json={"channel_id": 0})
        assert len(alert_store.get_alerts()) == 0


class TestPhaseTransition:
    def _start_with_channel(self, client):
        client.post("/pipeline/start")
        client.post("/channel/add", json={"source": "/data/video.mp4"})

    def test_set_phase_analytics(self, client):
        self._start_with_channel(client)
        resp = client.post("/channel/0/phase", json=ANALYTICS_PHASE_BODY)
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "phase": "analytics"}

    def test_set_phase_review(self, client):
        self._start_with_channel(client)
        resp = client.post("/channel/0/phase", json={"phase": "review"})
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "phase": "review"}

    def test_set_phase_invalid(self, client):
        self._start_with_channel(client)
        resp = client.post("/channel/0/phase", json={"phase": "bogus"})
        assert resp.status_code == 422

    def test_set_phase_nonexistent_channel(self, client):
        client.post("/pipeline/start")
        resp = client.post("/channel/99/phase", json=ANALYTICS_PHASE_BODY)
        assert resp.status_code == 404


class TestConfig:
    def test_update_confidence(self, client, app):
        resp = client.patch("/config", json={"confidence_threshold": 0.7})
        assert resp.status_code == 200
        assert resp.json() == {"status": "updated"}
        assert app.state.backend.confidence_threshold == 0.7

    def test_update_inference_interval(self, client, app):
        resp = client.patch("/config", json={"inference_interval": 5})
        assert resp.status_code == 200
        assert app.state.backend.inference_interval == 5

    def test_update_both(self, client, app):
        resp = client.patch(
            "/config",
            json={"confidence_threshold": 0.3, "inference_interval": 10},
        )
        assert resp.status_code == 200
        assert app.state.backend.confidence_threshold == 0.3
        assert app.state.backend.inference_interval == 10

    def test_update_empty_body(self, client):
        resp = client.patch("/config", json={})
        assert resp.status_code == 200
        assert resp.json() == {"status": "updated"}


class TestIntegration:
    def test_stop_resets_channel_counter(self, client):
        client.post("/pipeline/start")
        client.post("/channel/add", json={"source": "/data/a.mp4"})
        client.post("/channel/add", json={"source": "/data/b.mp4"})
        client.post("/pipeline/stop")
        # Restart — channel IDs should reset to 0
        client.post("/pipeline/start")
        r = client.post("/channel/add", json={"source": "/data/c.mp4"})
        assert r.json()["channel_id"] == 0


class TestMjpegStream:
    def test_stream_nonexistent_channel(self, client):
        client.post("/pipeline/start")
        resp = client.get("/stream/99")
        assert resp.status_code == 404

    def test_stream_pipeline_not_started(self, client):
        resp = client.get("/stream/0")
        assert resp.status_code == 409


class TestMjpegBroadcaster:
    """Unit tests for MjpegBroadcaster — tests the fan-out logic directly."""

    def test_subscribe_creates_queue(self, app):
        broadcaster = app.state.mjpeg
        q = broadcaster.subscribe(0)
        assert isinstance(q, queue.Queue)

    def test_on_frame_delivers_to_subscriber(self, app):
        from backend.pipeline.protocol import FrameResult

        broadcaster = app.state.mjpeg
        q = broadcaster.subscribe(0)
        result = FrameResult(
            channel_id=0,
            frame_number=1,
            timestamp_ms=33,
            detections=[],
            annotated_jpeg=b"\xff\xd8test",
        )
        broadcaster.on_frame(result)
        assert q.get_nowait() == b"\xff\xd8test"

    def test_on_frame_skips_none_jpeg(self, app):
        from backend.pipeline.protocol import FrameResult

        broadcaster = app.state.mjpeg
        q = broadcaster.subscribe(0)
        result = FrameResult(
            channel_id=0,
            frame_number=1,
            timestamp_ms=33,
            detections=[],
            annotated_jpeg=None,
        )
        broadcaster.on_frame(result)
        assert q.empty()

    def test_on_frame_fan_out_to_multiple_subscribers(self, app):
        from backend.pipeline.protocol import FrameResult

        broadcaster = app.state.mjpeg
        q1 = broadcaster.subscribe(0)
        q2 = broadcaster.subscribe(0)
        result = FrameResult(
            channel_id=0,
            frame_number=1,
            timestamp_ms=33,
            detections=[],
            annotated_jpeg=b"\xff\xd8multi",
        )
        broadcaster.on_frame(result)
        assert q1.get_nowait() == b"\xff\xd8multi"
        assert q2.get_nowait() == b"\xff\xd8multi"

    def test_on_frame_drops_oldest_when_full(self, app):
        from backend.pipeline.protocol import FrameResult

        broadcaster = app.state.mjpeg
        q = broadcaster.subscribe(0)  # maxsize=2
        for i in range(3):
            result = FrameResult(
                channel_id=0,
                frame_number=i,
                timestamp_ms=i * 33,
                detections=[],
                annotated_jpeg=f"frame{i}".encode(),
            )
            broadcaster.on_frame(result)
        # Queue should have the 2 most recent frames
        assert q.get_nowait() == b"frame1"
        assert q.get_nowait() == b"frame2"
        assert q.empty()

    def test_unsubscribe_removes_queue(self, app):
        from backend.pipeline.protocol import FrameResult

        broadcaster = app.state.mjpeg
        q = broadcaster.subscribe(0)
        broadcaster.unsubscribe(0, q)
        result = FrameResult(
            channel_id=0,
            frame_number=1,
            timestamp_ms=33,
            detections=[],
            annotated_jpeg=b"\xff\xd8test",
        )
        broadcaster.on_frame(result)
        assert q.empty()

    def test_channels_are_independent(self, app):
        from backend.pipeline.protocol import FrameResult

        broadcaster = app.state.mjpeg
        q0 = broadcaster.subscribe(0)
        q1 = broadcaster.subscribe(1)
        result = FrameResult(
            channel_id=0,
            frame_number=1,
            timestamp_ms=33,
            detections=[],
            annotated_jpeg=b"\xff\xd8ch0",
        )
        broadcaster.on_frame(result)
        assert q0.get_nowait() == b"\xff\xd8ch0"
        assert q1.empty()


class TestWebSocket:
    def test_ws_connect(self, client):
        with client.websocket_connect("/ws"):
            pass

    def test_ws_connect_with_channel_filter(self, client):
        with client.websocket_connect("/ws?channels=0,1"):
            pass


class TestWsBroadcaster:
    """Unit tests for WsBroadcaster — tests message building and filtering."""

    def test_on_frame_builds_message(self, app):
        from backend.pipeline.protocol import Detection, FrameResult

        ws_broadcaster = app.state.ws
        result = FrameResult(
            channel_id=0,
            frame_number=42,
            timestamp_ms=1400,
            detections=[
                Detection(
                    track_id=7,
                    class_name="car",
                    bbox=(412, 305, 129, 73),
                    confidence=0.91,
                    centroid=(476, 341),
                ),
            ],
            annotated_jpeg=b"\xff\xd8test",
        )
        ws_broadcaster.on_frame(result)
        msg = ws_broadcaster._queue.get_nowait()
        assert msg["type"] == "frame_data"
        assert msg["channel"] == 0
        assert msg["frame"] == 42
        assert msg["timestamp_ms"] == 1400
        assert len(msg["tracks"]) == 1
        track = msg["tracks"][0]
        assert track["id"] == 7
        assert track["class"] == "car"
        assert track["bbox"] == [412, 305, 129, 73]
        assert track["confidence"] == 0.91
        assert track["centroid"] == [476, 341]

    def test_on_alert_enqueues(self, app):
        ws_broadcaster = app.state.ws
        alert = {
            "type": "transit_alert",
            "channel": 0,
            "alert_id": "abc123",
            "track_id": 5,
        }
        ws_broadcaster.on_alert(alert)
        msg = ws_broadcaster._queue.get_nowait()
        assert msg["type"] == "transit_alert"
        assert msg["alert_id"] == "abc123"

    def test_on_track_ended_enqueues(self, app):
        ws_broadcaster = app.state.ws
        track = {
            "type": "track_ended",
            "channel": 0,
            "track_id": 7,
            "class": "car",
            "state": "completed",
        }
        ws_broadcaster.on_track_ended(track)
        msg = ws_broadcaster._queue.get_nowait()
        assert msg["type"] == "track_ended"
        assert msg["track_id"] == 7

    def test_enqueue_from_route_handler(self, client, app):
        """Pipeline start/stop enqueues pipeline_event messages."""
        ws_broadcaster = app.state.ws
        # Drain any existing messages
        while not ws_broadcaster._queue.empty():
            ws_broadcaster._queue.get_nowait()
        client.post("/pipeline/start")
        msg = ws_broadcaster._queue.get_nowait()
        assert msg["type"] == "pipeline_event"
        assert msg["event"] == "started"

    def test_enqueue_stop_event(self, client, app):
        ws_broadcaster = app.state.ws
        client.post("/pipeline/start")
        # Drain the start event
        while not ws_broadcaster._queue.empty():
            ws_broadcaster._queue.get_nowait()
        client.post("/pipeline/stop")
        msg = ws_broadcaster._queue.get_nowait()
        assert msg["type"] == "pipeline_event"
        assert msg["event"] == "stopped"

    def test_broadcast_filters_by_channel(self):
        """Verify _broadcast respects channel filtering."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from backend.api.websocket import WsBroadcaster

        broadcaster = WsBroadcaster()

        # Create mock WS clients
        ws_all = MagicMock()
        ws_all.send_text = AsyncMock()
        ws_ch0 = MagicMock()
        ws_ch0.send_text = AsyncMock()

        broadcaster._clients = [
            (ws_all, None, None),  # no filter — receives all
            (ws_ch0, {0}, None),  # only channel 0
        ]

        # Broadcast a channel 1 message
        msg = {"type": "track_ended", "channel": 1, "track_id": 5}
        asyncio.run(broadcaster._broadcast(msg))

        # ws_all should receive it, ws_ch0 should NOT
        ws_all.send_text.assert_called_once()
        ws_ch0.send_text.assert_not_called()

    def test_broadcast_no_channel_goes_to_all(self):
        """Messages without a channel field go to all clients."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from backend.api.websocket import WsBroadcaster

        broadcaster = WsBroadcaster()

        ws_ch0 = MagicMock()
        ws_ch0.send_text = AsyncMock()
        broadcaster._clients = [(ws_ch0, {0}, None)]

        msg = {"type": "pipeline_event", "event": "started", "detail": "test"}
        asyncio.run(broadcaster._broadcast(msg))

        # pipeline_event has no channel — should be sent
        ws_ch0.send_text.assert_called_once()
