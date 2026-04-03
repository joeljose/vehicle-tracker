"""Detection pipeline tests."""

import os
import pytest
from pathlib import Path

# Short test clips (10s each, mounted at /data/test_clips/ in container)
CLIPS_DIR = Path("/data/test_clips")
CLIP_741_73 = CLIPS_DIR / "741_73_10s.mp4"
CLIP_LYTLE = CLIPS_DIR / "lytle_south_10s.mp4"

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


def test_config_loads_labels():
    """Labels file loads correctly."""
    from backend.pipeline.deepstream.config import load_labels

    labels = load_labels()
    assert labels[0] == "car"
    assert labels[1] == "bicycle"
    assert labels[2] == "person"
    assert labels[3] == "road_sign"
    assert len(labels) == 4


def test_pipeline_detects_vehicles():
    """Pipeline detects vehicles on junction footage."""
    from backend.pipeline.deepstream.pipeline import run_pipeline

    summary = run_pipeline(str(CLIP_741_73))

    assert summary["frames"] > 0, "No frames processed"
    assert summary["total_detections"] > 0, "No detections"
    assert summary["fps"] > 0, "FPS should be positive"
    assert "car" in summary["class_counts"], "Should detect cars"


def test_pipeline_annotated_output(cleanup_output):
    """Pipeline produces annotated MP4 with bboxes."""
    from backend.pipeline.deepstream.pipeline import run_pipeline

    output = str(OUTPUT_DIR / "test_phase1_output.mp4")
    cleanup_output.append(output)

    summary = run_pipeline(str(CLIP_LYTLE), output_path=output)

    assert summary["frames"] > 0
    assert summary["total_detections"] > 0
    assert os.path.exists(output), "Output MP4 not created"
    assert os.path.getsize(output) > 1000, "Output MP4 too small"


def test_pipeline_interval_1():
    """interval=1 infers every 2nd frame, producing fewer total detections."""
    from backend.pipeline.deepstream.pipeline import run_pipeline

    baseline = run_pipeline(str(CLIP_741_73), interval=0)
    skipped = run_pipeline(str(CLIP_741_73), interval=1)

    assert skipped["frames"] > 0
    # interval=1 skips every other frame for inference, so total detections
    # should be noticeably lower than interval=0 (every frame).
    assert skipped["total_detections"] < baseline["total_detections"]
