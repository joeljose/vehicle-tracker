"""Unit tests for line-crossing detection and calibration (pure Python)."""

from backend.pipeline.direction import (
    LineSeg,
    LineCalibration,
    check_line_crossing,
    load_lines_from_config,
)


# Horizontal line from (0, 50) to (100, 50)
HLINE = LineSeg(label="North", start=(0, 50), end=(100, 50))

# Vertical line from (50, 0) to (50, 100)
VLINE = LineSeg(label="East", start=(50, 0), end=(50, 100))


def test_crossing_detected():
    """Centroid moving across a horizontal line is detected."""
    result = check_line_crossing(HLINE, (50, 40), (50, 60))
    assert result is not None


def test_no_crossing_same_side():
    """Centroid staying on one side returns None."""
    assert check_line_crossing(HLINE, (50, 40), (50, 45)) is None


def test_crossing_returns_direction_sign():
    """Crossing returns +1 or -1 depending on direction."""
    # Above to below
    d1 = check_line_crossing(HLINE, (50, 40), (50, 60))
    # Below to above
    d2 = check_line_crossing(HLINE, (50, 60), (50, 40))
    assert d1 is not None
    assert d2 is not None
    assert d1 == -d2  # opposite directions


def test_vertical_line_crossing():
    """Crossing a vertical line works."""
    # Left to right
    result = check_line_crossing(VLINE, (40, 50), (60, 50))
    assert result is not None


def test_no_crossing_on_parallel_movement():
    """Movement parallel to line doesn't trigger crossing."""
    assert check_line_crossing(HLINE, (10, 40), (90, 40)) is None


def test_diagonal_line():
    """Crossing a diagonal line works."""
    diag = LineSeg(label="Diag", start=(0, 0), end=(100, 100))
    # Point moving from one side to the other
    result = check_line_crossing(diag, (10, 50), (50, 10))
    assert result is not None


# --- Calibration tests ---


def test_calibration_after_n_observations():
    """Calibration completes after observing enough crossings."""
    cal = LineCalibration(HLINE, calibration_count=5)
    assert not cal.calibrated

    for _ in range(5):
        cal.observe(1)

    assert cal.calibrated
    assert cal.entry_sign == 1


def test_calibration_majority_vote():
    """Entry sign is the majority direction."""
    cal = LineCalibration(HLINE, calibration_count=5)
    # 3 positive, 2 negative -> entry_sign = +1
    for d in [1, 1, -1, 1, -1]:
        cal.observe(d)

    assert cal.calibrated
    assert cal.entry_sign == 1


def test_classify_after_calibration():
    """classify() returns entry/exit after calibration."""
    cal = LineCalibration(HLINE, calibration_count=3)
    for _ in range(3):
        cal.observe(1)

    assert cal.classify(1) == "entry"
    assert cal.classify(-1) == "exit"


def test_classify_before_calibration():
    """classify() returns None before calibration."""
    cal = LineCalibration(HLINE, calibration_count=10)
    cal.observe(1)
    assert cal.classify(1) is None


def test_force_calibrate():
    """Manual calibration overrides auto."""
    cal = LineCalibration(HLINE, calibration_count=100)
    cal.force_calibrate(entry_sign=-1)
    assert cal.calibrated
    assert cal.classify(-1) == "entry"
    assert cal.classify(1) == "exit"


def test_observations_stop_after_calibration():
    """No more observations recorded after calibration."""
    cal = LineCalibration(HLINE, calibration_count=3)
    for _ in range(3):
        cal.observe(1)
    assert cal.calibrated

    cal.observe(-1)  # should be ignored
    assert len(cal.observations) == 3


# --- load_lines_from_config ---


def test_load_lines_from_config():
    config = {
        "north": {"label": "741-North", "start": [200, 90], "end": [450, 90]},
        "east": {"label": "73-East", "start": [580, 300], "end": [580, 800]},
    }
    lines = load_lines_from_config(config)
    assert len(lines) == 2
    assert lines["north"].label == "741-North"
    assert lines["north"].start == (200, 90)
    assert lines["east"].end == (580, 800)
