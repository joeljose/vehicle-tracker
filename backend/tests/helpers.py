"""Shared test constants and helpers."""

ANALYTICS_PHASE_BODY = {
    "phase": "analytics",
    "roi_polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
    "entry_exit_lines": {
        "north": {"label": "North", "start": [0, 0], "end": [100, 0]},
    },
}
"""Valid phase transition body for setup → analytics (includes ROI + lines)."""
