"""Tests for IdleOptimizer — skip inference on empty scenes."""

from backend.pipeline.idle import IdleOptimizer


class TestIdleTransitions:
    """Test state transitions between active and idle modes."""

    def test_starts_active(self):
        opt = IdleOptimizer()
        assert opt.is_idle is False
        assert opt.recommended_interval == 0

    def test_stays_active_with_detections(self):
        opt = IdleOptimizer(idle_threshold_frames=30)
        for _ in range(100):
            opt.update(num_detections=1, num_active_tracks=1)
        assert opt.is_idle is False

    def test_goes_idle_after_threshold(self):
        opt = IdleOptimizer(idle_threshold_frames=30)
        for _ in range(30):
            opt.update(num_detections=0, num_active_tracks=0)
        assert opt.is_idle is False  # exactly 30 frames — not yet idle
        opt.update(num_detections=0, num_active_tracks=0)
        assert opt.is_idle is True
        assert opt.recommended_interval == 15

    def test_returns_to_active_on_detection(self):
        opt = IdleOptimizer(idle_threshold_frames=30)
        # Go idle
        for _ in range(31):
            opt.update(num_detections=0, num_active_tracks=0)
        assert opt.is_idle is True
        # One detection brings back active
        opt.update(num_detections=1, num_active_tracks=1)
        assert opt.is_idle is False
        assert opt.recommended_interval == 0

    def test_never_idle_with_active_tracks(self):
        opt = IdleOptimizer(idle_threshold_frames=30)
        # No detections but tracks still active (e.g. shadow tracking)
        for _ in range(50):
            opt.update(num_detections=0, num_active_tracks=1)
        assert opt.is_idle is False

    def test_resets_counter_on_detection(self):
        opt = IdleOptimizer(idle_threshold_frames=30)
        # 29 empty frames
        for _ in range(29):
            opt.update(num_detections=0, num_active_tracks=0)
        # One detection resets the counter
        opt.update(num_detections=1, num_active_tracks=1)
        # Need another 31 empty frames to go idle
        for _ in range(30):
            opt.update(num_detections=0, num_active_tracks=0)
        assert opt.is_idle is False
        opt.update(num_detections=0, num_active_tracks=0)
        assert opt.is_idle is True


class TestTransitionCallbacks:
    """Test that transitions are reported correctly."""

    def test_transition_to_idle_reported(self):
        opt = IdleOptimizer(idle_threshold_frames=30)
        transition = None
        for _ in range(31):
            transition = opt.update(num_detections=0, num_active_tracks=0)
        assert transition == "idle"

    def test_transition_to_active_reported(self):
        opt = IdleOptimizer(idle_threshold_frames=30)
        for _ in range(31):
            opt.update(num_detections=0, num_active_tracks=0)
        transition = opt.update(num_detections=1, num_active_tracks=1)
        assert transition == "active"

    def test_no_transition_when_stable(self):
        opt = IdleOptimizer(idle_threshold_frames=30)
        # Already active, stay active
        transition = opt.update(num_detections=1, num_active_tracks=1)
        assert transition is None
        # Go idle
        for _ in range(31):
            opt.update(num_detections=0, num_active_tracks=0)
        # Stay idle
        transition = opt.update(num_detections=0, num_active_tracks=0)
        assert transition is None


class TestCustomConfig:
    """Test configurable thresholds."""

    def test_custom_idle_threshold(self):
        opt = IdleOptimizer(idle_threshold_frames=10)
        for _ in range(11):
            opt.update(num_detections=0, num_active_tracks=0)
        assert opt.is_idle is True

    def test_custom_idle_interval(self):
        opt = IdleOptimizer(idle_interval=10)
        for _ in range(31):
            opt.update(num_detections=0, num_active_tracks=0)
        assert opt.recommended_interval == 10
