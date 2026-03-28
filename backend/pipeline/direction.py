"""Line-crossing detection, auto-calibration, and direction state machine."""

import math
from dataclasses import dataclass, field
from enum import Enum


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


# --- Direction state machine ---


class TrackState(Enum):
    UNKNOWN = "unknown"
    IN_TRANSIT = "in_transit"
    WAITING = "waiting"
    STAGNANT = "stagnant"


def nearest_arm(
    trajectory: list[tuple[int, int, int]],
    arms: dict[str, LineSeg],
    use_start: bool,
) -> str | None:
    """Infer direction from trajectory heading via dot product against arm vectors.

    Args:
        trajectory: List of (x, y, frame) entries.
        arms: Arm ID -> LineSeg mapping.
        use_start: True = use first centroids (infer entry), False = use last (infer exit).

    Returns:
        Best-matching arm ID, or None if trajectory is too short.
    """
    if len(trajectory) < 4:
        return None

    n = min(10, len(trajectory) // 2)
    if n < 2:
        return None

    segment = trajectory[:n] if use_start else trajectory[-n:]

    heading = (segment[-1][0] - segment[0][0], segment[-1][1] - segment[0][1])
    heading_mag = math.sqrt(heading[0] ** 2 + heading[1] ** 2)
    if heading_mag < 1e-6:
        return None
    hx, hy = heading[0] / heading_mag, heading[1] / heading_mag

    best_arm = None
    best_dot = -float("inf")
    for arm_id, line in arms.items():
        dx = line.end[0] - line.start[0]
        dy = line.end[1] - line.start[1]
        mag = math.sqrt(dx ** 2 + dy ** 2)
        if mag < 1e-6:
            continue
        dot = hx * (dx / mag) + hy * (dy / mag)
        if dot > best_dot:
            best_dot = dot
            best_arm = arm_id

    return best_arm


class DirectionStateMachine:
    """Per-track state machine: UNKNOWN -> IN_TRANSIT -> EXITED -> alert.

    Args:
        stagnant_threshold_sec: Time stopped before STAGNANT alert (default 150s).
        motion_threshold_px: Displacement below this = stopped (default 15px).
        fps: Source FPS for time-to-frame conversion.
    """

    def __init__(
        self,
        stagnant_threshold_sec: float = 150.0,
        motion_threshold_px: float = 15.0,
        fps: float = 30.0,
    ):
        self.stagnant_threshold_sec = stagnant_threshold_sec
        self.motion_threshold_px = motion_threshold_px
        self.fps = fps
        self.stagnant_window_frames = int(stagnant_threshold_sec * fps)

        self.state = TrackState.UNKNOWN
        self.entry_arm: str | None = None
        self.entry_label: str | None = None
        self.exit_arm: str | None = None
        self.exit_label: str | None = None
        self.was_stagnant = False

    def on_crossing(
        self,
        arm_id: str,
        label: str,
        crossing_type: str,
        trajectory: list[tuple[int, int, int]] | None = None,
        arms: dict[str, LineSeg] | None = None,
    ) -> dict | None:
        """Handle a line crossing event.

        Args:
            arm_id: Which arm was crossed (e.g. "north").
            label: Human label (e.g. "741-North").
            crossing_type: "entry" or "exit".
            trajectory: Current trajectory (needed for inferring entry on exit-first).
            arms: All arms (needed for inferring entry on exit-first).

        Returns:
            A transit alert dict if this crossing completes a transit, else None.
        """
        if crossing_type == "entry" and self.state in (
            TrackState.UNKNOWN, TrackState.WAITING, TrackState.STAGNANT,
        ):
            self.entry_arm = arm_id
            self.entry_label = label
            self.state = TrackState.IN_TRANSIT
            return None

        if crossing_type == "exit" and self.state in (
            TrackState.IN_TRANSIT,
            TrackState.WAITING,
            TrackState.STAGNANT,
        ):
            self.exit_arm = arm_id
            self.exit_label = label
            return {
                "entry_arm": self.entry_arm,
                "entry_label": self.entry_label,
                "exit_arm": self.exit_arm,
                "exit_label": self.exit_label,
                "method": "confirmed",
                "was_stagnant": self.was_stagnant,
            }

        if crossing_type == "exit" and self.state == TrackState.UNKNOWN:
            # Exit without entry — infer entry from trajectory heading
            self.exit_arm = arm_id
            self.exit_label = label
            if trajectory and arms:
                inferred_entry = nearest_arm(trajectory, arms, use_start=True)
                if inferred_entry and inferred_entry != arm_id:
                    self.entry_arm = inferred_entry
                    self.entry_label = arms[inferred_entry].label
                    return {
                        "entry_arm": self.entry_arm,
                        "entry_label": self.entry_label,
                        "exit_arm": self.exit_arm,
                        "exit_label": self.exit_label,
                        "method": "inferred",
                        "was_stagnant": self.was_stagnant,
                    }

        return None

    def on_track_lost(
        self,
        trajectory: list[tuple[int, int, int]],
        arms: dict[str, LineSeg],
    ) -> dict | None:
        """Handle track loss — infer missing direction if possible.

        Returns:
            A transit alert dict if direction can be inferred, else None.
        """
        if self.state == TrackState.UNKNOWN:
            # Never crossed any line — try to infer both entry and exit
            entry = nearest_arm(trajectory, arms, use_start=True)
            exit_ = nearest_arm(trajectory, arms, use_start=False)
            if entry and exit_ and entry != exit_:
                return {
                    "entry_arm": entry,
                    "entry_label": arms[entry].label,
                    "exit_arm": exit_,
                    "exit_label": arms[exit_].label,
                    "method": "inferred",
                    "was_stagnant": self.was_stagnant,
                }

        elif self.state in (TrackState.IN_TRANSIT, TrackState.WAITING, TrackState.STAGNANT):
            # Had entry but no exit — infer exit
            exit_ = nearest_arm(trajectory, arms, use_start=False)
            if exit_ and exit_ != self.entry_arm:
                return {
                    "entry_arm": self.entry_arm,
                    "entry_label": self.entry_label,
                    "exit_arm": exit_,
                    "exit_label": arms[exit_].label,
                    "method": "inferred",
                    "was_stagnant": self.was_stagnant,
                }

        return None

    def check_stagnant(self, trajectory: list[tuple[int, int, int]]) -> bool:
        """Check if vehicle is stagnant. Returns True on state transition to STAGNANT."""
        if len(trajectory) < self.stagnant_window_frames:
            return False

        oldest = trajectory[-self.stagnant_window_frames]
        newest = trajectory[-1]
        displacement = math.sqrt(
            (newest[0] - oldest[0]) ** 2 + (newest[1] - oldest[1]) ** 2
        )

        if displacement < self.motion_threshold_px:
            if self.state not in (TrackState.WAITING, TrackState.STAGNANT):
                self._state_before_wait = self.state
                self.state = TrackState.WAITING
            if self.state == TrackState.WAITING:
                self.state = TrackState.STAGNANT
                self.was_stagnant = True
                return True
            return False  # already STAGNANT, don't re-trigger
        else:
            # Moving again — return to previous state
            if self.state in (TrackState.WAITING, TrackState.STAGNANT):
                prev = getattr(self, "_state_before_wait", TrackState.UNKNOWN)
                self.state = TrackState.IN_TRANSIT if self.entry_arm else prev

        return False
