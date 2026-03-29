"""In-memory alert storage — transit and stagnant alerts."""

import time
import uuid


class AlertStore:
    """Stores transit and stagnant alerts in memory.

    Alerts are kept until the channel is removed (Phase 4 teardown)
    or the pipeline stops. No disk persistence — v1 is memory-only.

    Two access patterns:
        1. Summary (lightweight): for WebSocket push and GET /alerts listing.
           Excludes per_frame_data and full_trajectory.
        2. Full metadata: for GET /alert/{id} on-demand fetch.
           Includes everything for replay.
    """

    def __init__(self, fps: float = 30.0):
        self._alerts: dict[str, dict] = {}  # alert_id → full metadata
        self._order: list[str] = []  # alert_ids in insertion order
        self._fps = fps

    def _generate_id(self) -> str:
        return uuid.uuid4().hex[:8]

    def add_transit_alert(self, pipeline_alert: dict, channel: int) -> str:
        """Store a transit alert from pipeline output.

        Args:
            pipeline_alert: Dict from DirectionStateMachine with keys:
                track_id, label, entry_arm, entry_label, exit_arm, exit_label,
                method, was_stagnant, first_seen_frame, last_seen_frame,
                duration_frames, trajectory, per_frame_data.
            channel: Channel ID.

        Returns:
            Generated alert_id.
        """
        alert_id = self._generate_id()
        track_id = pipeline_alert["track_id"]
        duration_frames = pipeline_alert.get("duration_frames", 0)
        trajectory = pipeline_alert.get("trajectory", [])
        per_frame_data = pipeline_alert.get("per_frame_data", [])

        # Compute confidence stats from per-frame data
        confidences = [f["confidence"] for f in per_frame_data if "confidence" in f]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        max_confidence = max(confidences) if confidences else 0.0

        full = {
            "alert_id": alert_id,
            "type": "transit_alert",
            "channel": channel,
            "track_id": track_id,
            "class": pipeline_alert.get("label", "unknown"),
            "entry_direction": pipeline_alert["entry_arm"],
            "entry_label": pipeline_alert["entry_label"],
            "exit_direction": pipeline_alert["exit_arm"],
            "exit_label": pipeline_alert["exit_label"],
            "method": pipeline_alert["method"],
            "best_photo_url": f"/snapshot/{track_id}",
            "full_trajectory": [(x, y) for x, y, *_ in trajectory],
            "per_frame_data": [
                {
                    "frame": f["frame"],
                    "bbox": list(f["bbox"]),
                    "centroid": list(f["centroid"]),
                    "timestamp_ms": f.get("timestamp_ms", 0),
                }
                for f in per_frame_data
            ],
            "first_seen_frame": pipeline_alert.get("first_seen_frame", 0),
            "last_seen_frame": pipeline_alert.get("last_seen_frame", 0),
            "duration_ms": int(duration_frames / self._fps * 1000),
            "timestamp_ms": int(time.time() * 1000),
            "avg_confidence": round(avg_confidence, 2),
            "max_confidence": round(max_confidence, 2),
            "frames_tracked": len(per_frame_data),
            "frames_stationary": 0,
        }

        self._alerts[alert_id] = full
        self._order.append(alert_id)
        return alert_id

    def add_stagnant_alert(self, pipeline_alert: dict, channel: int) -> str:
        """Store a stagnant alert.

        Args:
            pipeline_alert: Dict with keys:
                track_id, label, position, stationary_duration_frames,
                first_seen_frame, last_seen_frame.
            channel: Channel ID.

        Returns:
            Generated alert_id.
        """
        alert_id = self._generate_id()
        track_id = pipeline_alert["track_id"]
        duration_frames = pipeline_alert.get("stationary_duration_frames", 0)

        full = {
            "alert_id": alert_id,
            "type": "stagnant_alert",
            "channel": channel,
            "track_id": track_id,
            "class": pipeline_alert.get("label", "unknown"),
            "best_photo_url": f"/snapshot/{track_id}",
            "stationary_duration_ms": int(duration_frames / self._fps * 1000),
            "position": list(pipeline_alert.get("position", (0, 0))),
            "timestamp_ms": int(time.time() * 1000),
            "first_seen_frame": pipeline_alert.get("first_seen_frame", 0),
            "last_seen_frame": pipeline_alert.get("last_seen_frame", 0),
        }

        self._alerts[alert_id] = full
        self._order.append(alert_id)
        return alert_id

    def get_alert(self, alert_id: str) -> dict | None:
        """Get full alert metadata (for GET /alert/{id})."""
        return self._alerts.get(alert_id)

    def get_alerts(
        self,
        limit: int = 50,
        alert_type: str | None = None,
        channel: int | None = None,
    ) -> list[dict]:
        """Get alert summaries (newest first, paginated).

        Summaries exclude per_frame_data and full_trajectory to keep
        the response lightweight for listing and WebSocket push.
        """
        # Iterate in reverse (newest first)
        result = []
        for alert_id in reversed(self._order):
            alert = self._alerts.get(alert_id)
            if alert is None:
                continue
            if alert_type and alert["type"] != alert_type:
                continue
            if channel is not None and alert["channel"] != channel:
                continue
            result.append(self._to_summary(alert))
            if len(result) >= limit:
                break
        return result

    def get_ws_summary(self, alert_id: str) -> dict | None:
        """Get lightweight WebSocket push summary for an alert."""
        alert = self._alerts.get(alert_id)
        if alert is None:
            return None
        return self._to_summary(alert)

    def count_by_channel(self, channel: int) -> int:
        """Count alerts for a specific channel."""
        return sum(1 for a in self._alerts.values() if a["channel"] == channel)

    def clear_channel(self, channel: int) -> None:
        """Remove all alerts for a channel (Phase 4 teardown)."""
        to_remove = [
            aid for aid, a in self._alerts.items() if a["channel"] == channel
        ]
        for aid in to_remove:
            del self._alerts[aid]
        self._order = [aid for aid in self._order if aid not in set(to_remove)]

    def clear(self) -> None:
        """Remove all alerts."""
        self._alerts.clear()
        self._order.clear()

    @staticmethod
    def _to_summary(alert: dict) -> dict:
        """Strip heavy fields from an alert for summary use."""
        exclude = {"per_frame_data", "full_trajectory", "avg_confidence",
                    "max_confidence", "frames_tracked", "frames_stationary"}
        return {k: v for k, v in alert.items() if k not in exclude}
