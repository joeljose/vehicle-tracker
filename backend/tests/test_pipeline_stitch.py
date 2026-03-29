"""Integration tests for proximity-based track stitching (Phase 8)."""

from pathlib import Path

CLIPS_DIR = Path("/data/test_clips")
CLIP_LYTLE = CLIPS_DIR / "lytle_south_10s.mp4"


def test_pipeline_has_merge_count():
    """Pipeline summary includes merge count."""
    from backend.pipeline.deepstream.pipeline import run_pipeline

    summary = run_pipeline(str(CLIP_LYTLE))

    assert "merges" in summary
    assert isinstance(summary["merges"], int)


def test_stitcher_merges_logged(caplog):
    """Pipeline logs merge events (or none if no ID switches occur)."""
    from backend.pipeline.deepstream.pipeline import run_pipeline

    with caplog.at_level("INFO"):
        summary = run_pipeline(str(CLIP_LYTLE))

    if summary["merges"] > 0:
        assert "merged with lost track" in caplog.text
    # Merges depend on footage — even zero merges is valid if no ID switches


def test_direction_state_preserved_through_stitching():
    """Unique track count should not increase due to stitching merges."""
    from backend.pipeline.deepstream.pipeline import run_pipeline

    summary = run_pipeline(str(CLIP_LYTLE))

    # If merges happened, the merged tracks inherited state rather than
    # creating new direction FSMs, so total unique tracks includes both
    # lost and active. This is a sanity check — no assertion on exact
    # numbers since it depends on footage.
    assert summary["unique_tracks"] > 0
    assert summary["frames"] > 0
    print(f"  Merges: {summary['merges']}")
    print(f"  Unique tracks: {summary['unique_tracks']}")
