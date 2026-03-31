"""Unit tests for line-crossing detection and junction-side classification."""

from backend.pipeline.direction import (
    LineSeg,
    check_line_crossing,
    classify_crossing,
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


# --- classify_crossing tests ---


def test_classify_left_junction_side():
    """Left junction side: +1 crossing = entry (moving toward left/positive side)."""
    line = LineSeg(label="North", start=(0, 50), end=(100, 50), junction_side="left")
    assert classify_crossing(line, 1) == "entry"
    assert classify_crossing(line, -1) == "exit"


def test_classify_right_junction_side():
    """Right junction side: -1 crossing = entry (moving toward right/negative side)."""
    line = LineSeg(label="North", start=(0, 50), end=(100, 50), junction_side="right")
    assert classify_crossing(line, -1) == "entry"
    assert classify_crossing(line, 1) == "exit"


def test_classify_flip_reverses_polarity():
    """Flipping junction_side reverses entry/exit classification."""
    line_left = LineSeg(label="N", start=(0, 50), end=(100, 50), junction_side="left")
    line_right = LineSeg(label="N", start=(0, 50), end=(100, 50), junction_side="right")
    assert classify_crossing(line_left, 1) != classify_crossing(line_right, 1)
    assert classify_crossing(line_left, -1) != classify_crossing(line_right, -1)


def test_classify_vertical_line():
    """Junction side works correctly for vertical lines."""
    line = LineSeg(label="East", start=(50, 0), end=(50, 100), junction_side="left")
    # Left side of downward vector (50,0)->(50,100) is the left/west side
    d = check_line_crossing(line, (40, 50), (60, 50))  # left to right
    assert d is not None
    # Moving from left (junction side) to right = exit
    assert classify_crossing(line, d) == "exit"


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


def test_load_lines_from_config_with_junction_side():
    config = {
        "north": {
            "label": "North",
            "start": [200, 90],
            "end": [450, 90],
            "junction_side": "right",
        },
    }
    lines = load_lines_from_config(config)
    assert lines["north"].junction_side == "right"


def test_load_lines_from_config_default_junction_side():
    """Missing junction_side defaults to 'left'."""
    config = {
        "north": {"label": "North", "start": [200, 90], "end": [450, 90]},
    }
    lines = load_lines_from_config(config)
    assert lines["north"].junction_side == "left"
