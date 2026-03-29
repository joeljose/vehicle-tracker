"""Idle optimization — skip inference when no vehicles are in the scene."""


class IdleOptimizer:
    """Tracks empty-frame streaks and recommends inference interval changes.

    After ``idle_threshold_frames`` consecutive frames with zero detections
    AND zero active tracks, recommends switching to a slower inference
    interval.  Any detection immediately restores full-rate inference.

    Args:
        idle_threshold_frames: Consecutive empty frames before idle (default 30 = 1s at 30fps).
        idle_interval: Inference interval to use in idle mode (default 15).
    """

    def __init__(
        self,
        idle_threshold_frames: int = 30,
        idle_interval: int = 15,
    ):
        self.idle_threshold_frames = idle_threshold_frames
        self.idle_interval = idle_interval
        self.is_idle = False
        self._empty_count = 0

    @property
    def recommended_interval(self) -> int:
        """Current recommended nvinfer interval."""
        return self.idle_interval if self.is_idle else 0

    def update(
        self,
        num_detections: int,
        num_active_tracks: int,
    ) -> str | None:
        """Process one frame and return any state transition.

        Args:
            num_detections: Number of detections in this frame.
            num_active_tracks: Number of currently active tracks.

        Returns:
            ``"idle"`` on transition to idle, ``"active"`` on transition
            to active, or ``None`` if no change.
        """
        if num_detections > 0 or num_active_tracks > 0:
            self._empty_count = 0
            if self.is_idle:
                self.is_idle = False
                return "active"
            return None

        self._empty_count += 1
        if not self.is_idle and self._empty_count > self.idle_threshold_frames:
            self.is_idle = True
            return "idle"

        return None
