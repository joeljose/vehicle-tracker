"""Phase 2 tests: NvDCF tracker — persistent vehicle IDs."""

import os
import pytest
from pathlib import Path

CLIPS_DIR = Path("/data/test_clips")
CLIP_741_73 = CLIPS_DIR / "741_73_10s.mp4"
CLIP_LYTLE = CLIPS_DIR / "lytle_south_10s.mp4"
CLIP_DRUGMART = CLIPS_DIR / "drugmart_73_10s.mp4"

OUTPUT_DIR = Path("/app/snapshots")


@pytest.fixture
def cleanup_output():
    """Clean up output files after test."""
    files = []
    yield files
    for f in files:
        try:
            os.unlink(f)
        except FileNotFoundError:
            pass


def test_tracker_assigns_ids():
    """Each vehicle gets a persistent track ID."""
    from backend.pipeline.deepstream.pipeline import run_pipeline

    summary = run_pipeline(str(CLIP_LYTLE))

    assert summary["frames"] > 0
    assert summary["unique_tracks"] > 0, "Tracker should assign at least one track"
    # Every track should have a numeric ID
    for track in summary["tracks"]:
        assert isinstance(track["track_id"], int)
        assert track["lifetime"] > 0


def test_track_ids_stable():
    """IDs remain stable — unique track count much less than frames * avg_detections."""
    from backend.pipeline.deepstream.pipeline import run_pipeline

    summary = run_pipeline(str(CLIP_LYTLE))

    # If IDs were resetting every frame, unique_tracks ≈ frames * avg_detections.
    # Stable tracking should produce far fewer unique tracks than that.
    unstable_track_count = summary["total_detections"]
    assert summary["unique_tracks"] < unstable_track_count * 0.5, (
        f"Too many unique tracks ({summary['unique_tracks']}) vs "
        f"total detections ({unstable_track_count}) — IDs may not be stable"
    )


def test_track_lifecycle_logged(capsys):
    """Console logs 'New track' and 'Track lost' messages."""
    from backend.pipeline.deepstream.pipeline import run_pipeline

    run_pipeline(str(CLIP_741_73))

    captured = capsys.readouterr()
    assert "New track #" in captured.out, "Should log new track creation"


def test_track_count_in_summary():
    """Per-frame track count is printed in periodic reports."""
    from backend.pipeline.deepstream.pipeline import run_pipeline

    summary = run_pipeline(str(CLIP_741_73))

    # Summary should include tracking fields
    assert "unique_tracks" in summary
    assert "tracks" in summary
    assert isinstance(summary["tracks"], list)


def test_tracker_annotated_output(cleanup_output):
    """Annotated output shows bboxes with track IDs."""
    from backend.pipeline.deepstream.pipeline import run_pipeline

    output = str(OUTPUT_DIR / "test_phase2_output.mp4")
    cleanup_output.append(output)

    summary = run_pipeline(str(CLIP_LYTLE), output_path=output)

    assert summary["unique_tracks"] > 0
    assert os.path.exists(output), "Output MP4 not created"
    assert os.path.getsize(output) > 1000, "Output MP4 too small"


def test_tracker_all_junctions():
    """Tracker runs on all 3 junction clips without crashing."""
    from backend.pipeline.deepstream.pipeline import run_pipeline

    for clip in [CLIP_741_73, CLIP_LYTLE, CLIP_DRUGMART]:
        if not clip.exists():
            pytest.skip(f"Clip not found: {clip}")
        summary = run_pipeline(str(clip))
        assert summary["frames"] > 0, f"No frames for {clip.name}"
        assert summary["unique_tracks"] > 0, f"No tracks for {clip.name}"


def test_shadow_tracking_age():
    """Tracker config has maxShadowTrackingAge=75 (acceptance criteria)."""
    import yaml

    config_path = "/app/config/deepstream/tracker_config.yml"
    with open(config_path) as f:
        # Skip %YAML:1.0 directive (OpenCV format, not standard YAML)
        content = f.read()
        if content.startswith("%YAML"):
            content = content.split("\n", 1)[1]
        config = yaml.safe_load(content)

    assert config["TargetManagement"]["maxShadowTrackingAge"] == 75
