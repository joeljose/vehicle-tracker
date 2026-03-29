"""Integration tests for idle optimization (Phase 9)."""

from pathlib import Path

CLIPS_DIR = Path("/data/test_clips")
CLIP_LYTLE = CLIPS_DIR / "lytle_south_10s.mp4"


def test_idle_mode_logged(capsys):
    """Pipeline logs idle/active mode transitions if they occur."""
    from backend.pipeline.deepstream.pipeline import run_pipeline

    summary = run_pipeline(str(CLIP_LYTLE))
    captured = capsys.readouterr()

    # On busy junction footage, idle mode may or may not trigger.
    # Just verify the pipeline runs without errors.
    assert summary["frames"] > 0
    assert summary["unique_tracks"] > 0

    # If idle mode was entered, verify the log format
    if "IDLE MODE" in captured.out:
        assert "interval=15" in captured.out
    if "ACTIVE MODE" in captured.out:
        assert "interval=0" in captured.out


def test_pipeline_detects_through_idle():
    """Detections still work after idle→active transitions."""
    from backend.pipeline.deepstream.pipeline import run_pipeline

    summary = run_pipeline(str(CLIP_LYTLE))

    # The pipeline should still detect vehicles regardless of idle mode
    assert summary["total_detections"] > 0
    assert summary["unique_tracks"] > 0
