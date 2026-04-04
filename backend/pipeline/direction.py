"""Line-crossing detection and direction state machine.

Entry/exit polarity is determined by the user during Setup phase via
the junction_side field ("left" or "right"). The cross-product of the
line vector (start→end) with a point tells us which side the point is
on: positive = left side, negative = right side. The junction_side
field specifies which side is "inside" the junction. A vehicle crossing
from the junction side to the outside is "exiting"; outside to junction
side is "entering".
"""

import math
from dataclasses import dataclass
from enum import Enum


@dataclass
class LineSeg:
    """A labeled entry/exit line segment with junction side polarity."""

    label: str
    start: tuple[float, float]
    end: tuple[float, float]
    junction_side: str = "left"  # "left" or "right" of start→end vector


def cross_product_sign(line: LineSeg, point: tuple[float, float]) -> float:
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
    """
    cross_prev = cross_product_sign(line, prev_centroid)
    cross_curr = cross_product_sign(line, curr_centroid)

    if cross_prev * cross_curr < 0:  # sign change
        return 1 if cross_prev < 0 else -1
    return None


def classify_crossing(line: LineSeg, crossing_direction: int) -> str:
    """Classify a crossing as 'entry' or 'exit' using user-specified junction side.

    Convention:
        - Cross product positive = left side of start→end vector
        - Cross product negative = right side of start→end vector
        - junction_side tells us which side is "inside" the junction
        - Crossing from outside to inside = "entry"
        - Crossing from inside to outside = "exit"

    crossing_direction +1 means the vehicle moved from negative to positive side.
    crossing_direction -1 means the vehicle moved from positive to negative side.
    """
    # Which cross-product sign corresponds to the junction (inside) side?
    junction_sign_is_positive = line.junction_side == "left"

    if junction_sign_is_positive:
        # Junction is on the positive (left) side
        # +1 = moved from negative to positive = entering junction
        return "entry" if crossing_direction == 1 else "exit"
    else:
        # Junction is on the negative (right) side
        # -1 = moved from positive to negative = entering junction
        return "entry" if crossing_direction == -1 else "exit"


def load_lines_from_config(entry_exit_lines: dict) -> dict[str, LineSeg]:
    """Convert site config entry_exit_lines dict to LineSeg objects."""
    lines = {}
    for arm_id, line_data in entry_exit_lines.items():
        lines[arm_id] = LineSeg(
            label=line_data["label"],
            start=tuple(line_data["start"]),
            end=tuple(line_data["end"]),
            junction_side=line_data.get("junction_side", "left"),
        )
    return lines


# --- Direction state machine ---


class TrackState(Enum):
    UNKNOWN = "unknown"
    IN_TRANSIT = "in_transit"
    WAITING = "waiting"
    STAGNANT = "stagnant"


def _point_to_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """Minimum distance from point (px, py) to line segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def _trim_trajectory_jumps(
    trajectory: list[tuple[int, int, int]],
    jump_threshold_px: float = 80.0,
) -> list[tuple[int, int, int]]:
    """Trim sudden position jumps from a trajectory.

    When the tracker reassigns a track ID to a different vehicle during
    occlusion, there's a sudden large displacement between consecutive
    centroids.  This function scans from both ends inward and truncates
    at the first jump that exceeds the threshold.

    Returns the cleaned trajectory (may be shorter than input).
    """
    if len(trajectory) < 3:
        return trajectory

    # Scan forward: find first jump from the start
    start = 0
    for i in range(1, len(trajectory)):
        dx = trajectory[i][0] - trajectory[i - 1][0]
        dy = trajectory[i][1] - trajectory[i - 1][1]
        if math.sqrt(dx * dx + dy * dy) > jump_threshold_px:
            start = i
            break

    # Scan backward: find first jump from the end
    end = len(trajectory)
    for i in range(len(trajectory) - 1, 0, -1):
        dx = trajectory[i][0] - trajectory[i - 1][0]
        dy = trajectory[i][1] - trajectory[i - 1][1]
        if math.sqrt(dx * dx + dy * dy) > jump_threshold_px:
            end = i
            break

    trimmed = trajectory[start:end]
    return trimmed if len(trimmed) >= 4 else trajectory


def _heading(seg: list[tuple[int, int, int]]) -> tuple[float, float]:
    """Compute (hx, hy) heading vector from a trajectory segment."""
    return (seg[-1][0] - seg[0][0], seg[-1][1] - seg[0][1])


def _angle_between(h1: tuple[float, float], h2: tuple[float, float]) -> float:
    """Angle in degrees between two heading vectors. Returns 0-180."""
    dot = h1[0] * h2[0] + h1[1] * h2[1]
    m1 = math.sqrt(h1[0] ** 2 + h1[1] ** 2)
    m2 = math.sqrt(h2[0] ** 2 + h2[1] ** 2)
    if m1 < 1e-9 or m2 < 1e-9:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / (m1 * m2)))
    return math.degrees(math.acos(cos_a))


# Max heading divergence before we distrust the tail/head and use the
# stable mid-trajectory heading instead. 60° is generous — a real turn
# through a junction rarely exceeds this between the stable body and
# the endpoint segment.
_HEADING_DIVERGENCE_THRESHOLD = 60.0


def _ray_segment_intersection(
    ox: float, oy: float, dx: float, dy: float,
    ax: float, ay: float, bx: float, by: float,
) -> float | None:
    """Find where a ray intersects a line segment.

    Ray: origin (ox, oy) + t * direction (dx, dy), t >= 0.
    Segment: from (ax, ay) to (bx, by).

    Returns t (distance along ray) if intersection exists, else None.
    """
    sx, sy = bx - ax, by - ay
    denom = dx * sy - dy * sx
    if abs(denom) < 1e-12:
        return None  # parallel
    t = ((ax - ox) * sy - (ay - oy) * sx) / denom
    u = ((ax - ox) * dy - (ay - oy) * dx) / denom
    if t >= 0 and 0 <= u <= 1:
        return t
    return None


def nearest_arm(
    trajectory: list[tuple[int, int, int]],
    arms: dict[str, LineSeg],
    use_start: bool,
    roi_centroid: tuple[float, float] | None = None,
) -> str | None:
    """Infer entry/exit arm via ray intersection, with proximity fallback.

    Casts a ray from the trajectory endpoint along the heading direction
    (backwards for entry, forwards for exit) and finds which arm line the
    ray intersects first. This is geometrically exact — no arbitrary
    projection distance.

    If heading is too weak or the ray misses all lines, falls back to
    pure proximity (closest line to the anchor point).

    To guard against tracker ID drift during occlusion, the endpoint
    heading is cross-checked against the stable mid-trajectory heading.
    If they diverge by more than 60 degrees, the stable heading is used.

    Args:
        trajectory: List of (x, y, frame) entries.
        arms: Arm ID -> LineSeg mapping.
        use_start: True = use first centroids (infer entry), False = use last (infer exit).
        roi_centroid: Unused (kept for API compatibility).

    Returns:
        Best-matching arm ID, or None if trajectory is too short.
    """
    if len(trajectory) < 4:
        return None

    # Trim sudden position jumps (hard tracker ID switches)
    trajectory = _trim_trajectory_jumps(trajectory)

    n = min(5, len(trajectory) // 2)
    if n < 1:
        return None

    # Stable mid-trajectory heading (25%-75%), unlikely to be contaminated
    q1 = len(trajectory) // 4
    q3 = 3 * len(trajectory) // 4
    stable_seg = trajectory[q1 : q3 + 1] if q3 > q1 else trajectory
    stable_h = _heading(stable_seg)

    if use_start:
        seg = trajectory[:n]
        anchor_x = sum(p[0] for p in seg) / len(seg)
        anchor_y = sum(p[1] for p in seg) / len(seg)
        endpoint_h = _heading(seg)
        # Ray direction: backwards (opposite to travel)
        ray_dx, ray_dy = -endpoint_h[0], -endpoint_h[1]
    else:
        seg = trajectory[-n:]
        anchor_x = sum(p[0] for p in seg) / len(seg)
        anchor_y = sum(p[1] for p in seg) / len(seg)
        endpoint_h = _heading(seg)

        # For exit: if tail heading is contaminated (diverges from stable),
        # use stable heading and anchor from clean region
        if _angle_between(endpoint_h, stable_h) > _HEADING_DIVERGENCE_THRESHOLD:
            endpoint_h = stable_h
            clean_seg = trajectory[q3 - n : q3]
            if len(clean_seg) >= 2:
                anchor_x = sum(p[0] for p in clean_seg) / len(clean_seg)
                anchor_y = sum(p[1] for p in clean_seg) / len(clean_seg)

        # Ray direction: forwards (along travel)
        ray_dx, ray_dy = endpoint_h[0], endpoint_h[1]

    heading_len = math.sqrt(ray_dx * ray_dx + ray_dy * ray_dy)

    # Try ray intersection first (if heading is meaningful)
    if heading_len > 5.0:
        best_arm = None
        best_t = float("inf")
        for arm_id, line in arms.items():
            t = _ray_segment_intersection(
                anchor_x, anchor_y, ray_dx, ray_dy,
                line.start[0], line.start[1],
                line.end[0], line.end[1],
            )
            if t is not None and t < best_t:
                best_t = t
                best_arm = arm_id
        if best_arm is not None:
            return best_arm

    # Fallback: pure proximity (closest line to anchor)
    best_arm = None
    best_dist = float("inf")
    for arm_id, line in arms.items():
        dist = _point_to_segment_distance(
            anchor_x, anchor_y,
            line.start[0], line.start[1],
            line.end[0], line.end[1],
        )
        if dist < best_dist:
            best_dist = dist
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
        self.stopped_since_frame: int | None = None  # frame when vehicle first stopped
        self._state_before_wait: TrackState = TrackState.UNKNOWN
        self._motion_check_window = 30  # frames to check for movement (~1s at 30fps)

    def on_crossing(
        self,
        arm_id: str,
        label: str,
        crossing_type: str,
        trajectory: list[tuple[int, int, int]] | None = None,
        arms: dict[str, LineSeg] | None = None,
        roi_centroid: tuple[float, float] | None = None,
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
            TrackState.UNKNOWN,
            TrackState.WAITING,
            TrackState.STAGNANT,
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
            # Reject same-arm exit (bbox jitter near a line, not a real transit)
            if arm_id == self.entry_arm:
                # Vehicle oscillated back across the same line — reset to UNKNOWN
                self.entry_arm = None
                self.entry_label = None
                self.state = TrackState.UNKNOWN
                return None

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
                inferred_entry = nearest_arm(
                    trajectory,
                    arms,
                    use_start=True,
                    roi_centroid=roi_centroid,
                )
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
        roi_centroid: tuple[float, float] | None = None,
    ) -> dict | None:
        """Handle track loss — infer missing direction if possible.

        Returns:
            A transit alert dict if direction can be inferred, else None.
        """
        if self.state == TrackState.UNKNOWN:
            # Never crossed any line — try to infer both entry and exit
            entry = nearest_arm(
                trajectory, arms, use_start=True, roi_centroid=roi_centroid
            )
            exit_ = nearest_arm(
                trajectory, arms, use_start=False, roi_centroid=roi_centroid
            )
            if entry and exit_ and entry != exit_:
                return {
                    "entry_arm": entry,
                    "entry_label": arms[entry].label,
                    "exit_arm": exit_,
                    "exit_label": arms[exit_].label,
                    "method": "inferred",
                    "was_stagnant": self.was_stagnant,
                }

        elif self.state in (
            TrackState.IN_TRANSIT,
            TrackState.WAITING,
            TrackState.STAGNANT,
        ):
            # Had entry but no exit — infer exit
            exit_ = nearest_arm(
                trajectory, arms, use_start=False, roi_centroid=roi_centroid
            )
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
        """Check if vehicle is stagnant. Returns True on state transition to STAGNANT.

        Uses a short window to detect if vehicle is currently stopped, then a
        frame counter to measure how long it's been stopped. This avoids needing
        the trajectory buffer to hold minutes of data.
        """
        if len(trajectory) < self._motion_check_window:
            return False

        # Check displacement over last ~1 second
        oldest = trajectory[-self._motion_check_window]
        newest = trajectory[-1]
        displacement = math.sqrt(
            (newest[0] - oldest[0]) ** 2 + (newest[1] - oldest[1]) ** 2
        )
        current_frame = newest[2]

        if displacement < self.motion_threshold_px:
            # Vehicle is currently stopped
            if self.stopped_since_frame is None:
                self.stopped_since_frame = current_frame
                if self.state not in (TrackState.WAITING, TrackState.STAGNANT):
                    self._state_before_wait = self.state
                    self.state = TrackState.WAITING

            stopped_frames = current_frame - self.stopped_since_frame
            if (
                stopped_frames >= self.stagnant_window_frames
                and self.state != TrackState.STAGNANT
            ):
                self.state = TrackState.STAGNANT
                self.was_stagnant = True
                return True
        else:
            # Vehicle is moving
            self.stopped_since_frame = None
            if self.state in (TrackState.WAITING, TrackState.STAGNANT):
                self.state = TrackState.IN_TRANSIT if self.entry_arm else self._state_before_wait

        return False
