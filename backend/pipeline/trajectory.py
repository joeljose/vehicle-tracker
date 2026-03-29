"""Per-track centroid trajectory ring buffer."""

import math
from collections import deque


class TrajectoryBuffer:
    """Ring buffer storing (x, y, frame_number) centroid entries per track.

    Default capacity is 300 entries (~10 seconds at 30fps).
    """

    def __init__(self, max_len: int = 300):
        self.buffer: deque[tuple[int, int, int]] = deque(maxlen=max_len)

    def append(self, x: int, y: int, frame: int):
        """Add a centroid entry."""
        self.buffer.append((x, y, frame))

    def __len__(self) -> int:
        return len(self.buffer)

    def get_trajectory(self) -> list[tuple[int, int]]:
        """Return (x, y) pairs for drawing."""
        return [(x, y) for x, y, _ in self.buffer]

    def get_full(self) -> list[tuple[int, int, int]]:
        """Return all (x, y, frame_number) entries."""
        return list(self.buffer)

    def get_displacement(self, window_frames: int) -> float:
        """Centroid displacement over the last N entries.

        Returns inf if not enough data yet.
        """
        if len(self.buffer) < window_frames:
            return float("inf")
        oldest = self.buffer[-window_frames]
        newest = self.buffer[-1]
        return math.sqrt((newest[0] - oldest[0]) ** 2 + (newest[1] - oldest[1]) ** 2)
