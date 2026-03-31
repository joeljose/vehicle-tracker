"""Stream recovery — retry, liveness check, re-extract, circuit breaker.

Handles YouTube Live stream interruptions with a progressive recovery
strategy. A circuit breaker prevents one flaky stream from repeatedly
disrupting all channels in the shared pipeline.
"""

import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Recovery config
RETRY_DELAYS = [5.0, 10.0, 20.0]  # seconds between retries
CIRCUIT_BREAKER_WINDOW = 600.0  # 10 minutes
CIRCUIT_BREAKER_THRESHOLD = 3  # max rebuilds in window


@dataclass
class _RecoveryRecord:
    """Tracks rebuild timestamps for circuit breaker."""

    rebuild_times: list[float] = field(default_factory=list)

    def record_rebuild(self) -> None:
        self.rebuild_times.append(time.monotonic())

    def is_tripped(self) -> bool:
        """Check if circuit breaker threshold exceeded within window."""
        now = time.monotonic()
        cutoff = now - CIRCUIT_BREAKER_WINDOW
        # Prune old entries
        self.rebuild_times = [t for t in self.rebuild_times if t > cutoff]
        return len(self.rebuild_times) >= CIRCUIT_BREAKER_THRESHOLD


class CircuitBreaker:
    """Per-channel circuit breaker for stream recovery.

    Tracks pipeline rebuilds per channel. If a channel triggers too many
    rebuilds within the window, it is marked as tripped (should be ejected).
    """

    def __init__(self):
        self._records: dict[int, _RecoveryRecord] = {}

    def record_rebuild(self, channel_id: int) -> None:
        """Record a pipeline rebuild for a channel."""
        if channel_id not in self._records:
            self._records[channel_id] = _RecoveryRecord()
        self._records[channel_id].record_rebuild()

    def is_tripped(self, channel_id: int) -> bool:
        """Check if a channel's circuit breaker has tripped."""
        record = self._records.get(channel_id)
        if record is None:
            return False
        return record.is_tripped()

    def reset(self, channel_id: int) -> None:
        """Reset a channel's circuit breaker (e.g., on manual re-add)."""
        self._records.pop(channel_id, None)
