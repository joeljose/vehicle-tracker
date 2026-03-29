"""Integration tests for FastAPI pipeline control endpoints."""

import queue


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
        resp = client.post("/channel/0/phase", json={"phase": "analytics"})
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
        resp = client.post("/channel/99/phase", json={"phase": "analytics"})
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
    def test_full_lifecycle(self, client):
        # Start pipeline
        assert client.post("/pipeline/start").status_code == 200
        # Add channel
        r = client.post("/channel/add", json={"source": "/data/video.mp4"})
        assert r.json()["channel_id"] == 0
        # Set phase
        r = client.post("/channel/0/phase", json={"phase": "analytics"})
        assert r.json()["phase"] == "analytics"
        # Update config
        assert client.patch(
            "/config", json={"confidence_threshold": 0.8}
        ).status_code == 200
        # Remove channel
        assert client.post(
            "/channel/remove", json={"channel_id": 0}
        ).status_code == 200
        # Stop pipeline
        assert client.post("/pipeline/stop").status_code == 200

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
