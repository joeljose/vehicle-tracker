"""Proximity-based track stitching — merge ID-switched tracks."""

import math

from backend.pipeline.direction import DirectionStateMachine
from backend.pipeline.trajectory import TrajectoryBuffer


# Cap velocity extrapolation to avoid wild predictions
_MAX_PREDICTION_FRAMES = 15


class TrackStitcher:
    """Buffers recently lost tracks and matches new tracks by proximity.

    When the NvDCF tracker loses a vehicle and re-detects it with a new ID
    (e.g. after brief occlusion), the stitcher merges the new track with the
    old one so direction state, trajectory, and best-photo are preserved.

    Args:
        max_age_frames: How long to retain lost tracks (default 60 = 2s at 30fps).
        max_distance: Maximum pixel distance for a match (default 100px).
    """

    def __init__(
        self,
        max_age_frames: int = 60,
        max_distance: float = 100.0,
    ):
        self.max_age_frames = max_age_frames
        self.max_distance = max_distance
        self.lost_tracks: dict[int, dict] = {}

    def on_track_lost(
        self,
        track_id: int,
        trajectory: TrajectoryBuffer,
        dsm: DirectionStateMachine,
        frame_number: int,
    ) -> None:
        """Buffer a lost track for potential stitching.

        Args:
            track_id: The lost track's ID.
            trajectory: The track's trajectory buffer.
            dsm: The track's direction state machine.
            frame_number: Frame when the track was lost.
        """
        full = trajectory.get_full()
        if not full:
            return

        last_x, last_y, _ = full[-1]

        # Estimate velocity from last two points
        vx, vy = 0.0, 0.0
        if len(full) >= 2:
            x1, y1, f1 = full[-2]
            x2, y2, f2 = full[-1]
            df = f2 - f1
            if df > 0:
                vx = (x2 - x1) / df
                vy = (y2 - y1) / df

        self.lost_tracks[track_id] = {
            "last_position": (last_x, last_y),
            "velocity": (vx, vy),
            "lost_frame": frame_number,
            "trajectory": trajectory,
            "dsm": dsm,
        }

    def expire(self, frame_number: int) -> list[tuple[int, dict]]:
        """Remove lost tracks older than max_age_frames.

        Returns:
            List of (track_id, state_dict) for expired tracks so the caller
            can do final processing (direction inference, best-photo save).
        """
        expired_ids = [
            tid for tid, state in self.lost_tracks.items()
            if frame_number - state["lost_frame"] > self.max_age_frames
        ]
        expired = [(tid, self.lost_tracks.pop(tid)) for tid in expired_ids]
        return expired

    def find_match(
        self,
        position: tuple[float, float],
        frame_number: int,
    ) -> dict | None:
        """Find the closest lost track near position.

        Expires old tracks first, then checks proximity against velocity-
        predicted positions.  The matched track is removed from the buffer.

        Args:
            position: (x, y) centroid of the new track.
            frame_number: Current frame number.

        Returns:
            Dict with lost_track_id, dsm, trajectory, distance, gap_frames
            or None if no match.
        """
        self.expire(frame_number)

        if not self.lost_tracks:
            return None

        best_tid = None
        best_dist = float("inf")

        px, py = position

        for tid, state in self.lost_tracks.items():
            gap = frame_number - state["lost_frame"]
            lx, ly = state["last_position"]
            vx, vy = state["velocity"]

            # Clamp prediction frames to avoid wild extrapolation
            pred_frames = min(gap, _MAX_PREDICTION_FRAMES)
            pred_x = lx + vx * pred_frames
            pred_y = ly + vy * pred_frames

            dist = math.hypot(px - pred_x, py - pred_y)
            if dist < best_dist:
                best_dist = dist
                best_tid = tid

        if best_tid is None or best_dist > self.max_distance:
            return None

        state = self.lost_tracks.pop(best_tid)
        gap_frames = frame_number - state["lost_frame"]

        return {
            "lost_track_id": best_tid,
            "dsm": state["dsm"],
            "trajectory": state["trajectory"],
            "distance": best_dist,
            "gap_frames": gap_frames,
        }
