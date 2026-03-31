"""Unit tests for ROI polygon filtering (pure Python, no GPU needed)."""

from backend.pipeline.roi import point_in_polygon


# Simple square polygon: (0,0), (10,0), (10,10), (0,10)
SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_point_inside():
    assert point_in_polygon((5, 5), SQUARE) is True


def test_point_outside():
    assert point_in_polygon((15, 5), SQUARE) is False


def test_point_outside_negative():
    assert point_in_polygon((-1, -1), SQUARE) is False


def test_point_on_edge():
    # Edge behavior is implementation-defined for ray casting;
    # just verify it doesn't crash
    result = point_in_polygon((0, 5), SQUARE)
    assert isinstance(result, bool)


def test_point_on_vertex():
    result = point_in_polygon((0, 0), SQUARE)
    assert isinstance(result, bool)


def test_concave_polygon():
    """L-shaped polygon."""
    # L-shape: bottom-left corner cut out
    l_shape = [(0, 0), (10, 0), (10, 10), (5, 10), (5, 5), (0, 5)]

    assert point_in_polygon((2, 2), l_shape) is True  # inside bottom
    assert point_in_polygon((7, 7), l_shape) is True  # inside top-right
    assert point_in_polygon((2, 7), l_shape) is False  # in the cut-out


def test_triangle():
    triangle = [(0, 0), (10, 0), (5, 10)]
    assert point_in_polygon((5, 3), triangle) is True
    assert point_in_polygon((0, 10), triangle) is False


def test_empty_polygon():
    """Empty or degenerate polygon returns False."""
    assert point_in_polygon((5, 5), []) is False


def test_float_coordinates():
    poly = [(0.0, 0.0), (10.5, 0.0), (10.5, 10.5), (0.0, 10.5)]
    assert point_in_polygon((5.25, 5.25), poly) is True
    assert point_in_polygon((11.0, 5.0), poly) is False
