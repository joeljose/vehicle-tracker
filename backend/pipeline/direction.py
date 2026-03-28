"""Line-crossing detection and auto-calibration."""

from dataclasses import dataclass, field


@dataclass
class LineSeg:
    """A labeled entry/exit line segment."""

    label: str
    start: tuple[float, float]
    end: tuple[float, float]


def cross_product_sign(
    line: LineSeg, point: tuple[float, float]
) -> float:
    """2D cross product of (B-A) x (P-A). Sign indicates which side of the line."""
    ax, ay = line.start
    bx, by = line.end
    px, py = point
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def check_line_crossing(
    line: LineSeg,
    prev_centroid: tuple[float, float],
    curr_centroid: tuple[float, float],
) -> int | None:
    """Detect if a centroid crossed a line between two frames.

    Returns:
        +1 if crossed from negative to positive side,
        -1 if crossed from positive to negative side,
        None if no crossing.

    The raw sign is returned because entry/exit meaning depends on
    calibration (line drawing direction is arbitrary).
    """
    cross_prev = cross_product_sign(line, prev_centroid)
    cross_curr = cross_product_sign(line, curr_centroid)

    if cross_prev * cross_curr < 0:  # sign change
        return 1 if cross_prev < 0 else -1
    return None


class LineCalibration:
    """Auto-calibrates entry/exit polarity for a single line.

    Observes the first N crossings and uses the ROI centroid to determine
    which cross-product sign corresponds to "entry" (approaching from outside
    the junction) vs "exit" (leaving toward outside).

    The heuristic: if the centroid that was *closer to the ROI center* is on the
    positive side, then positive-to-negative = exit. But since we don't have
    ROI center easily, we use a simpler approach: the operator is expected to
    draw lines at junction borders. Vehicles entering the junction cross from
    outside to inside. We record crossing directions and use majority vote
    after observing `calibration_count` crossings.
    """

    def __init__(self, line: LineSeg, calibration_count: int = 10):
        self.line = line
        self.calibration_count = calibration_count
        self.observations: list[int] = []  # raw crossing directions (+1 or -1)
        self.entry_sign: int | None = None  # once calibrated: which sign means "entry"
        self._calibrated = False

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def observe(self, crossing_direction: int) -> None:
        """Record a crossing observation for calibration."""
        if self._calibrated:
            return
        self.observations.append(crossing_direction)
        if len(self.observations) >= self.calibration_count:
            self._calibrate()

    def _calibrate(self) -> None:
        """Majority vote: the more common direction is 'entry' (more vehicles enter than exit
        in a typical observation window, or at worst it's 50/50 and we pick one)."""
        pos_count = sum(1 for d in self.observations if d > 0)
        neg_count = len(self.observations) - pos_count
        # The more common crossing direction is assumed to be "entry"
        # (in a busy junction, roughly equal, so either assignment works —
        # the operator can override in config)
        self.entry_sign = 1 if pos_count >= neg_count else -1
        self._calibrated = True

    def classify(self, crossing_direction: int) -> str | None:
        """Classify a crossing as 'entry' or 'exit'. Returns None if not yet calibrated."""
        if not self._calibrated:
            return None
        return "entry" if crossing_direction == self.entry_sign else "exit"

    def force_calibrate(self, entry_sign: int) -> None:
        """Manually set calibration (e.g., from saved config)."""
        self.entry_sign = entry_sign
        self._calibrated = True


def load_lines_from_config(entry_exit_lines: dict) -> dict[str, LineSeg]:
    """Convert site config entry_exit_lines dict to LineSeg objects."""
    lines = {}
    for arm_id, line_data in entry_exit_lines.items():
        lines[arm_id] = LineSeg(
            label=line_data["label"],
            start=tuple(line_data["start"]),
            end=tuple(line_data["end"]),
        )
    return lines
