"""Tests for stream recovery — circuit breaker and recovery state machine."""

import time

from backend.pipeline.stream_recovery import (
    CIRCUIT_BREAKER_THRESHOLD,
    CIRCUIT_BREAKER_WINDOW,
    CircuitBreaker,
)


class TestCircuitBreaker:
    def test_not_tripped_initially(self):
        cb = CircuitBreaker()
        assert not cb.is_tripped(0)

    def test_not_tripped_below_threshold(self):
        cb = CircuitBreaker()
        for _ in range(CIRCUIT_BREAKER_THRESHOLD - 1):
            cb.record_rebuild(0)
        assert not cb.is_tripped(0)

    def test_tripped_at_threshold(self):
        cb = CircuitBreaker()
        for _ in range(CIRCUIT_BREAKER_THRESHOLD):
            cb.record_rebuild(0)
        assert cb.is_tripped(0)

    def test_independent_channels(self):
        cb = CircuitBreaker()
        for _ in range(CIRCUIT_BREAKER_THRESHOLD):
            cb.record_rebuild(0)
        assert cb.is_tripped(0)
        assert not cb.is_tripped(1)

    def test_reset_clears_state(self):
        cb = CircuitBreaker()
        for _ in range(CIRCUIT_BREAKER_THRESHOLD):
            cb.record_rebuild(0)
        assert cb.is_tripped(0)
        cb.reset(0)
        assert not cb.is_tripped(0)

    def test_old_entries_expire(self):
        cb = CircuitBreaker()
        # Record rebuilds with fake timestamps in the past
        record = cb._records.setdefault(
            0,
            cb._records.get(0)
            or __import__(
                "backend.pipeline.stream_recovery", fromlist=["_RecoveryRecord"]
            )._RecoveryRecord(),
        )
        now = time.monotonic()
        # Add old entries outside the window
        record.rebuild_times = [
            now - CIRCUIT_BREAKER_WINDOW - 10
        ] * CIRCUIT_BREAKER_THRESHOLD
        assert not cb.is_tripped(0)

    def test_mixed_old_and_new_entries(self):
        cb = CircuitBreaker()
        from backend.pipeline.stream_recovery import _RecoveryRecord

        record = _RecoveryRecord()
        cb._records[0] = record
        now = time.monotonic()
        # 2 old (expired) + 2 new (within window) = 2 active, below threshold of 3
        record.rebuild_times = [
            now - CIRCUIT_BREAKER_WINDOW - 10,
            now - CIRCUIT_BREAKER_WINDOW - 5,
            now - 1,
            now,
        ]
        assert not cb.is_tripped(0)

    def test_all_recent_entries_trip(self):
        cb = CircuitBreaker()
        from backend.pipeline.stream_recovery import _RecoveryRecord

        record = _RecoveryRecord()
        cb._records[0] = record
        now = time.monotonic()
        record.rebuild_times = [now - 2, now - 1, now]
        assert cb.is_tripped(0)
