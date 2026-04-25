"""Tests for DeepStreamPipeline adapter class.

Mocks pyservicemaker since GPU is unavailable on host.
Tests lifecycle, channel management, phase transitions, and callbacks.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from backend.pipeline.alerts import AlertStore
from backend.pipeline.protocol import ChannelPhase


@pytest.fixture
def mock_psm(monkeypatch):
    """Mock pyservicemaker module so adapter can be imported without GPU."""
    mock_pipeline = MagicMock()
    mock_pipeline.start = MagicMock()
    mock_pipeline.stop = MagicMock()
    mock_pipeline.get_bus.return_value.add_signal_watch = MagicMock()
    mock_pipeline.get_bus.return_value.connect = MagicMock()
    mock_pipeline.__getitem__ = MagicMock(return_value=MagicMock())

    with (
        patch(
            "backend.pipeline.deepstream.adapter.Pipeline", return_value=mock_pipeline
        ),
        patch("backend.pipeline.deepstream.adapter.Probe"),
        patch("backend.pipeline.deepstream.adapter.MjpegExtractor"),
        patch(
            "backend.pipeline.deepstream.adapter.TrackingReporter"
        ) as mock_reporter_cls,
        patch(
            "backend.pipeline.deepstream.adapter.load_labels",
            return_value={2: "car", 3: "motorcycle", 5: "bus", 7: "truck"},
        ),
        patch("backend.pipeline.deepstream.adapter.BestPhotoTracker"),
        patch("backend.pipeline.deepstream.batch_router.BatchMetadataRouter"),
    ):
        # Configure mock reporter
        mock_reporter = MagicMock()
        mock_reporter.stitcher.lost_tracks = {}
        mock_reporter.active_tracks = {}
        mock_reporter.best_photo = None
        mock_reporter_cls.return_value = mock_reporter

        from backend.pipeline.deepstream.adapter import DeepStreamPipeline

        yield DeepStreamPipeline


@pytest.fixture
def adapter(mock_psm):
    """Create a DeepStreamPipeline instance with mocked dependencies."""
    return mock_psm()


class TestSharedPipeline:
    def test_two_channels_share_one_pipeline(self, adapter, tmp_path):
        """Adding two channels creates one shared pipeline, not two separate ones."""
        adapter.start()
        v0 = tmp_path / "a.mp4"
        v0.write_bytes(b"fake")
        v1 = tmp_path / "b.mp4"
        v1.write_bytes(b"fake")

        adapter.add_channel(0, str(v0))
        adapter.add_channel(1, str(v1))

        assert adapter._shared_pipeline is not None
        # Both channels reference the same shared pipeline
        assert adapter._states[0].pipeline is adapter._shared_pipeline
        assert adapter._states[1].pipeline is adapter._shared_pipeline

    # Note: the previous "remove last channel destroys shared pipeline"
    # invariant was retired in #123. After every source has been
    # soft-removed (the post-#122 behaviour), calling pipeline.stop()
    # crashes pyservicemaker's C++ side. The dormant mux/infer/tracker
    # is harmless; final teardown happens at /pipeline/stop or process
    # exit. See TestSoftRemoveOnReview.test_remove_last_channel_keeps_
    # pipeline_alive for the current invariant.

    def test_removing_one_channel_keeps_pipeline_for_others(self, adapter, tmp_path):
        """Removing one channel doesn't affect the shared pipeline if others remain."""
        adapter.start()
        v0 = tmp_path / "a.mp4"
        v0.write_bytes(b"fake")
        v1 = tmp_path / "b.mp4"
        v1.write_bytes(b"fake")

        adapter.add_channel(0, str(v0))
        adapter.add_channel(1, str(v1))
        adapter.remove_channel(0)

        assert adapter._shared_pipeline is not None
        assert 1 in adapter.channels


class TestLifecycle:
    def test_start_sets_started(self, adapter):
        adapter.start()
        assert adapter._started is True

    def test_stop_clears_state(self, adapter):
        adapter.start()
        adapter.stop()
        assert adapter._started is False
        assert adapter.channels == {}

    def test_stop_clears_channels(self, adapter, tmp_path):
        adapter.start()
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        adapter.add_channel(0, str(video))
        adapter.stop()
        assert 0 not in adapter.channels


class TestChannelManagement:
    def test_add_channel(self, adapter, tmp_path):
        adapter.start()
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        adapter.add_channel(0, str(video))
        assert 0 in adapter.channels
        assert adapter.channels[0] == str(video)

    def test_add_channel_sets_setup_phase(self, adapter, tmp_path):
        adapter.start()
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        adapter.add_channel(0, str(video))
        assert adapter.get_channel_phase(0) == ChannelPhase.SETUP

    def test_add_duplicate_channel_raises(self, adapter, tmp_path):
        adapter.start()
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        adapter.add_channel(0, str(video))
        with pytest.raises(ValueError, match="already exists"):
            adapter.add_channel(0, str(video))

    def test_remove_channel(self, adapter, tmp_path):
        adapter.start()
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        adapter.add_channel(0, str(video))
        adapter.remove_channel(0)
        assert 0 not in adapter.channels

    def test_remove_nonexistent_raises(self, adapter):
        adapter.start()
        with pytest.raises(KeyError, match="not found"):
            adapter.remove_channel(99)

    def test_multiple_channels(self, adapter, tmp_path):
        adapter.start()
        for i in range(3):
            video = tmp_path / f"test{i}.mp4"
            video.write_bytes(b"fake")
            adapter.add_channel(i, str(video))
        assert len(adapter.channels) == 3


class TestConfigureChannel:
    def test_configure_stores_roi_and_lines(self, adapter, tmp_path):
        adapter.start()
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        adapter.add_channel(0, str(video))

        roi = [(0, 0), (100, 0), (100, 100), (0, 100)]
        lines = {"north": {"label": "North", "start": [0, 0], "end": [100, 0]}}
        adapter.configure_channel(0, roi, lines)

        state = adapter._states[0]
        assert state.roi_polygon == roi
        assert state.entry_exit_lines == lines

    def test_configure_nonexistent_raises(self, adapter):
        adapter.start()
        with pytest.raises(KeyError, match="not found"):
            adapter.configure_channel(99, [], {})


class TestPhaseTransitions:
    def test_setup_to_analytics(self, adapter, tmp_path):
        adapter.start()
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        adapter.add_channel(0, str(video))

        roi = [(0, 0), (100, 0), (100, 100), (0, 100)]
        lines = {"north": {"label": "North", "start": [0, 0], "end": [100, 0]}}
        adapter.configure_channel(0, roi, lines)
        adapter.set_channel_phase(0, ChannelPhase.ANALYTICS)

        assert adapter.get_channel_phase(0) == ChannelPhase.ANALYTICS

    def test_analytics_to_review(self, adapter, tmp_path):
        adapter.start()
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        adapter.add_channel(0, str(video))
        adapter.set_channel_phase(0, ChannelPhase.ANALYTICS)
        adapter.set_channel_phase(0, ChannelPhase.REVIEW)

        assert adapter.get_channel_phase(0) == ChannelPhase.REVIEW

    def test_eos_auto_transitions_to_review(self, adapter, tmp_path):
        adapter.start()
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        adapter.add_channel(0, str(video))
        adapter.set_channel_phase(0, ChannelPhase.ANALYTICS)

        # Simulate per-source EOS
        adapter._on_source_eos(0)
        assert adapter.get_channel_phase(0) == ChannelPhase.REVIEW

    def test_get_phase_nonexistent_raises(self, adapter):
        adapter.start()
        with pytest.raises(KeyError, match="not found"):
            adapter.get_channel_phase(99)

    def test_set_phase_nonexistent_raises(self, adapter):
        adapter.start()
        with pytest.raises(KeyError, match="not found"):
            adapter.set_channel_phase(99, ChannelPhase.ANALYTICS)


class TestSoftRemoveOnReview:
    """Analytics→Review and remove_channel must NOT rebuild the shared
    pipeline; that would tear down OSD/MJPEG branches for every other
    channel and produce a visible cross-channel stutter (B12)."""

    def test_review_does_not_rebuild_pipeline(self, adapter, tmp_path):
        adapter.start()
        v1 = tmp_path / "v1.mp4"
        v1.write_bytes(b"fake")
        v2 = tmp_path / "v2.mp4"
        v2.write_bytes(b"fake")
        adapter.add_channel(0, str(v1))
        adapter.add_channel(1, str(v2))
        adapter.set_channel_phase(0, ChannelPhase.ANALYTICS)
        adapter.set_channel_phase(1, ChannelPhase.ANALYTICS)
        # Snapshot the pipeline reference; soft-remove keeps it alive.
        pipeline_before = adapter._shared_pipeline
        assert pipeline_before is not None

        adapter.set_channel_phase(0, ChannelPhase.REVIEW)

        # Pipeline object identity must be unchanged: no rebuild happened.
        assert adapter._shared_pipeline is pipeline_before
        # Channel 1 still active (Analytics) and still attached to the
        # pipeline; channel 0 is in Review with a soft-removed source.
        assert adapter.get_channel_phase(0) == ChannelPhase.REVIEW
        assert adapter.get_channel_phase(1) == ChannelPhase.ANALYTICS
        assert adapter._states[1].pipeline is pipeline_before

    def test_eos_auto_review_does_not_rebuild_pipeline(self, adapter, tmp_path):
        adapter.start()
        v1 = tmp_path / "v1.mp4"
        v1.write_bytes(b"fake")
        v2 = tmp_path / "v2.mp4"
        v2.write_bytes(b"fake")
        adapter.add_channel(0, str(v1))
        adapter.add_channel(1, str(v2))
        adapter.set_channel_phase(0, ChannelPhase.ANALYTICS)
        adapter.set_channel_phase(1, ChannelPhase.ANALYTICS)
        pipeline_before = adapter._shared_pipeline

        adapter._on_source_eos(0)

        assert adapter._shared_pipeline is pipeline_before
        assert adapter.get_channel_phase(0) == ChannelPhase.REVIEW
        assert adapter.get_channel_phase(1) == ChannelPhase.ANALYTICS

    def test_remove_channel_keeps_others_running(self, adapter, tmp_path):
        adapter.start()
        v1 = tmp_path / "v1.mp4"
        v1.write_bytes(b"fake")
        v2 = tmp_path / "v2.mp4"
        v2.write_bytes(b"fake")
        adapter.add_channel(0, str(v1))
        adapter.add_channel(1, str(v2))
        pipeline_before = adapter._shared_pipeline

        adapter.remove_channel(0)

        # Channel 1's pipeline is the same instance as before.
        assert adapter._shared_pipeline is pipeline_before
        assert 1 in adapter.channels
        assert adapter._states[1].pipeline is pipeline_before

    def test_remove_last_channel_keeps_pipeline_alive(self, adapter, tmp_path):
        """When the last channel is removed, the shared pipeline stays up
        in dormant form (mux/infer/tracker waiting for data that won't
        come). Auto-destroying here crashes pyservicemaker after every
        source has been soft-removed (#123); the harmless dormant pipeline
        is cleaned up at /pipeline/stop or process exit instead.
        """
        adapter.start()
        v1 = tmp_path / "v1.mp4"
        v1.write_bytes(b"fake")
        adapter.add_channel(0, str(v1))
        pipeline_before = adapter._shared_pipeline
        assert pipeline_before is not None

        adapter.remove_channel(0)

        # Pipeline kept alive: the source has been soft-removed but the
        # broader pipeline is the same instance.
        assert adapter._shared_pipeline is pipeline_before
        # Channel state is gone from public-facing collections.
        assert 0 not in adapter.channels
        assert 0 not in adapter._states


class TestCallbacks:
    def test_register_frame_callback(self, adapter):
        cb = MagicMock()
        adapter.register_frame_callback(cb)
        assert adapter._frame_callback is cb

    def test_register_alert_callback(self, adapter):
        cb = MagicMock()
        adapter.register_alert_callback(cb)
        assert adapter._alert_callback is cb

    def test_register_track_ended_callback(self, adapter):
        cb = MagicMock()
        adapter.register_track_ended_callback(cb)
        assert adapter._track_ended_callback is cb


class TestConfiguration:
    def test_set_confidence_threshold(self, adapter):
        adapter.set_confidence_threshold(0.8)
        assert adapter._confidence_threshold == 0.8


class TestGetSnapshot:
    def test_returns_none_when_no_channels(self, adapter):
        adapter.start()
        assert adapter.get_snapshot(42) is None


class TestAlertStoreThreadSafety:
    def test_concurrent_add_and_read(self):
        store = AlertStore()
        errors = []

        def writer():
            for i in range(50):
                try:
                    store.add_transit_alert(
                        {
                            "track_id": i,
                            "entry_arm": "n",
                            "entry_label": "North",
                            "exit_arm": "s",
                            "exit_label": "South",
                            "method": "crossing",
                            "was_stagnant": False,
                            "first_seen_frame": 0,
                            "last_seen_frame": 100,
                            "duration_frames": 100,
                            "trajectory": [],
                            "per_frame_data": [],
                        },
                        channel=0,
                    )
                except Exception as e:
                    errors.append(e)

        def reader():
            for _ in range(50):
                try:
                    store.get_alerts(limit=10)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"


class TestTrackingReporterCallbacks:
    """Test TrackingReporter callback wiring.

    These tests use the real TrackingReporter (not mocked) since it
    needs the real pyservicemaker.BatchMetadataOperator base class.
    """

    def test_callbacks_default_none(self):
        from backend.pipeline.deepstream.pipeline import TrackingReporter

        reporter = TrackingReporter(labels={0: "car"})
        assert reporter.alert_callback is None
        assert reporter.track_ended_callback is None

    def test_callbacks_stored(self):
        from backend.pipeline.deepstream.pipeline import TrackingReporter

        alert_cb = MagicMock()
        track_cb = MagicMock()
        reporter = TrackingReporter(
            labels={0: "car"},
            channel_id=5,
            alert_callback=alert_cb,
            track_ended_callback=track_cb,
        )
        assert reporter.alert_callback is alert_cb
        assert reporter.track_ended_callback is track_cb
        assert reporter.channel_id == 5

    def test_transit_alert_callback_in_finalize(self):
        from backend.pipeline.deepstream.pipeline import TrackingReporter
        from backend.pipeline.direction import LineSeg
        from backend.pipeline.trajectory import TrajectoryBuffer

        alerts = []
        lines = {"north": LineSeg(label="North", start=(0, 0), end=(100, 0))}
        reporter = TrackingReporter(
            labels={0: "car"},
            lines=lines,
            channel_id=3,
            alert_callback=lambda a: alerts.append(a),
        )

        mock_dsm = MagicMock()
        mock_dsm.on_track_lost.return_value = {
            "entry_arm": "north",
            "entry_label": "North",
            "exit_arm": "south",
            "exit_label": "South",
            "method": "heading",
            "was_stagnant": False,
        }
        mock_dsm.state.value = "unknown"

        traj = TrajectoryBuffer()
        traj.append(50, 50, 0)

        reporter.finalize_lost_track(
            42,
            {
                "dsm": mock_dsm,
                "trajectory": traj,
                "label": "car",
                "lifetime": 100,
            },
        )

        assert len(alerts) == 1
        assert alerts[0]["track_id"] == reporter._seq_id(42)
        assert alerts[0]["channel"] == 3
        assert alerts[0]["label"] == "car"

    def test_track_ended_callback_in_finalize(self):
        from backend.pipeline.deepstream.pipeline import TrackingReporter
        from backend.pipeline.trajectory import TrajectoryBuffer

        ended = []
        reporter = TrackingReporter(
            labels={0: "car"},
            channel_id=1,
            track_ended_callback=lambda t: ended.append(t),
        )

        mock_dsm = MagicMock()
        mock_dsm.on_track_lost.return_value = None
        mock_dsm.state.value = "unknown"

        traj = TrajectoryBuffer()
        traj.append(50, 50, 0)

        reporter.finalize_lost_track(
            10,
            {
                "dsm": mock_dsm,
                "trajectory": traj,
                "label": "car",
                "lifetime": 50,
            },
        )

        assert len(ended) == 1
        assert ended[0]["track_id"] == reporter._seq_id(10)


class TestSoftRemoval:
    """P2: soft-removal of channels from the shared pipeline."""

    def test_soft_remove_disables_mjpeg(self, adapter, tmp_path):
        adapter.start()
        v = tmp_path / "a.mp4"
        v.write_bytes(b"fake")
        adapter.add_channel(0, str(v))
        state = adapter._states[0]

        # MjpegExtractor exists before removal
        assert state.mjpeg_extractor is not None

        adapter._soft_remove_source(0)
        assert state.mjpeg_extractor is None
        assert state.pipeline is None
        assert state.reporter is None

    def test_soft_remove_clears_source_idx(self, adapter, tmp_path):
        adapter.start()
        v = tmp_path / "a.mp4"
        v.write_bytes(b"fake")
        adapter.add_channel(0, str(v))

        adapter._soft_remove_source(0)
        state = adapter._states[0]
        assert state._source_idx is None
        assert state._src_name is None

    def test_soft_remove_idempotent(self, adapter, tmp_path):
        """Soft-removing twice should not raise."""
        adapter.start()
        v = tmp_path / "a.mp4"
        v.write_bytes(b"fake")
        adapter.add_channel(0, str(v))

        adapter._soft_remove_source(0)
        adapter._soft_remove_source(0)  # second call — no-op

    def test_remove_channel_keeps_shared_pipeline_for_others(self, adapter, tmp_path):
        adapter.start()
        v0 = tmp_path / "a.mp4"
        v0.write_bytes(b"fake")
        v1 = tmp_path / "b.mp4"
        v1.write_bytes(b"fake")

        adapter.add_channel(0, str(v0))
        adapter.add_channel(1, str(v1))
        adapter.remove_channel(0)

        assert adapter._shared_pipeline is not None
        assert 1 in adapter.channels
        assert adapter._states[1].pipeline is not None


class TestSourceIndexing:
    """P2: source_idx is monotonic and used as batch_router key."""

    def test_source_idx_assigned_on_add(self, adapter, tmp_path):
        adapter.start()
        v = tmp_path / "a.mp4"
        v.write_bytes(b"fake")

        adapter.add_channel(0, str(v))
        assert adapter._states[0]._source_idx is not None

        adapter.remove_channel(0)

        adapter.add_channel(0, str(v))
        assert adapter._states[0]._source_idx is not None

    def test_source_idx_independent_of_channel_id(self, adapter, tmp_path):
        adapter.start()
        v = tmp_path / "a.mp4"
        v.write_bytes(b"fake")

        # Add channel 5 first — should get source_idx 0
        adapter.add_channel(5, str(v))
        assert adapter._states[5]._source_idx == 0

        # Add channel 2 next — should get source_idx 1
        adapter.add_channel(2, str(v))
        assert adapter._states[2]._source_idx == 1

    def test_element_names_use_source_idx(self, adapter, tmp_path):
        adapter.start()
        v = tmp_path / "a.mp4"
        v.write_bytes(b"fake")
        adapter.add_channel(0, str(v))

        state = adapter._states[0]
        assert state._src_name == f"src_{state._source_idx}"


class TestPhaseTransitionSharedPipeline:
    """P2: phase transitions on the shared pipeline."""

    def test_analytics_rebuilds_pipeline(self, adapter, tmp_path):
        adapter.start()
        v = tmp_path / "a.mp4"
        v.write_bytes(b"fake")

        adapter.add_channel(0, str(v))
        assert adapter._states[0].pipeline is not None

        roi = [(0, 0), (100, 0), (100, 100), (0, 100)]
        lines = {"north": {"label": "North", "start": [0, 0], "end": [100, 0]}}
        adapter.configure_channel(0, roi, lines)
        adapter.set_channel_phase(0, ChannelPhase.ANALYTICS)

        # Pipeline rebuilt with analytics source
        assert adapter._states[0].pipeline is not None
        assert adapter._states[0]._source_idx is not None

    def test_analytics_preserves_roi_config(self, adapter, tmp_path):
        adapter.start()
        v = tmp_path / "a.mp4"
        v.write_bytes(b"fake")
        adapter.add_channel(0, str(v))

        roi = [(0, 0), (100, 0), (100, 100), (0, 100)]
        lines = {"north": {"label": "North", "start": [0, 0], "end": [100, 0]}}
        adapter.configure_channel(0, roi, lines)
        adapter.set_channel_phase(0, ChannelPhase.ANALYTICS)

        state = adapter._states[0]
        assert state.roi_polygon == roi
        assert state.entry_exit_lines == lines

    def test_eos_does_not_transition_non_analytics(self, adapter, tmp_path):
        adapter.start()
        v = tmp_path / "a.mp4"
        v.write_bytes(b"fake")
        adapter.add_channel(0, str(v))

        # Channel is in SETUP, EOS should not transition
        adapter._on_source_eos(0)
        assert adapter.get_channel_phase(0) == ChannelPhase.SETUP

    def test_eos_fires_phase_callback(self, adapter, tmp_path):
        adapter.start()
        v = tmp_path / "a.mp4"
        v.write_bytes(b"fake")
        adapter.add_channel(0, str(v))
        adapter.set_channel_phase(0, ChannelPhase.ANALYTICS)

        callbacks = []
        adapter.register_phase_callback(
            lambda ch, new, old: callbacks.append((ch, new, old))
        )

        adapter._on_source_eos(0)
        assert len(callbacks) == 1
        assert callbacks[0] == (0, ChannelPhase.REVIEW, ChannelPhase.ANALYTICS)

    def test_two_channels_independent_phase_transitions(self, adapter, tmp_path):
        adapter.start()
        v0 = tmp_path / "a.mp4"
        v0.write_bytes(b"fake")
        v1 = tmp_path / "b.mp4"
        v1.write_bytes(b"fake")

        adapter.add_channel(0, str(v0))
        adapter.add_channel(1, str(v1))

        roi = [(0, 0), (100, 0), (100, 100), (0, 100)]
        lines = {"n": {"label": "N", "start": [0, 0], "end": [100, 0]}}
        adapter.configure_channel(0, roi, lines)
        adapter.set_channel_phase(0, ChannelPhase.ANALYTICS)

        # Channel 0 is analytics, channel 1 still setup
        assert adapter.get_channel_phase(0) == ChannelPhase.ANALYTICS
        assert adapter.get_channel_phase(1) == ChannelPhase.SETUP
        # Both still have pipeline references
        assert adapter._states[0].pipeline is not None
        assert adapter._states[1].pipeline is not None
