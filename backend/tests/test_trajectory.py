"""Unit tests for TrajectoryBuffer (pure Python, no GPU needed)."""

from backend.pipeline.trajectory import TrajectoryBuffer


def test_append_and_len():
    """Buffer tracks entries correctly."""
    buf = TrajectoryBuffer(max_len=10)
    assert len(buf) == 0

    buf.append(100, 200, 1)
    assert len(buf) == 1

    buf.append(110, 210, 2)
    assert len(buf) == 2


def test_get_trajectory_returns_xy_pairs():
    """get_trajectory() returns (x, y) without frame numbers."""
    buf = TrajectoryBuffer()
    buf.append(10, 20, 1)
    buf.append(30, 40, 2)
    buf.append(50, 60, 3)

    traj = buf.get_trajectory()
    assert traj == [(10, 20), (30, 40), (50, 60)]


def test_get_full_includes_frame():
    """get_full() returns (x, y, frame_number) tuples."""
    buf = TrajectoryBuffer()
    buf.append(10, 20, 100)
    buf.append(30, 40, 101)

    full = buf.get_full()
    assert full == [(10, 20, 100), (30, 40, 101)]


def test_ring_buffer_eviction():
    """Old entries are evicted when buffer is full."""
    buf = TrajectoryBuffer(max_len=3)
    buf.append(1, 1, 1)
    buf.append(2, 2, 2)
    buf.append(3, 3, 3)
    assert len(buf) == 3

    buf.append(4, 4, 4)
    assert len(buf) == 3
    # First entry should be evicted
    full = buf.get_full()
    assert full == [(2, 2, 2), (3, 3, 3), (4, 4, 4)]


def test_ring_buffer_300_default():
    """Default max_len is 300."""
    buf = TrajectoryBuffer()
    for i in range(350):
        buf.append(i, i, i)
    assert len(buf) == 300
    # Oldest should be frame 50 (0-49 evicted)
    assert buf.get_full()[0] == (50, 50, 50)


def test_get_displacement():
    """Displacement measures distance between entries."""
    buf = TrajectoryBuffer()
    buf.append(0, 0, 1)
    buf.append(3, 4, 2)

    # Distance from (0,0) to (3,4) = 5.0
    assert buf.get_displacement(2) == 5.0


def test_get_displacement_insufficient_data():
    """Returns inf when not enough entries."""
    buf = TrajectoryBuffer()
    buf.append(0, 0, 1)

    assert buf.get_displacement(5) == float("inf")


def test_get_displacement_window():
    """Displacement uses last N entries, not all."""
    buf = TrajectoryBuffer()
    buf.append(0, 0, 1)
    buf.append(100, 100, 2)  # big jump
    buf.append(103, 104, 3)  # small move

    # Window of 2: distance from (100,100) to (103,104) = 5.0
    assert buf.get_displacement(2) == 5.0


def test_empty_trajectory():
    """Empty buffer returns empty lists."""
    buf = TrajectoryBuffer()
    assert buf.get_trajectory() == []
    assert buf.get_full() == []
