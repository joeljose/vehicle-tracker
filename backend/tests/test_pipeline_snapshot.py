"""Integration tests for best-photo capture (Phase 7)."""

import shutil

import pytest
from pathlib import Path

CLIPS_DIR = Path("/data/test_clips")
CLIP_LYTLE = CLIPS_DIR / "lytle_south_10s.mp4"

SNAPSHOT_DIR = Path("/tmp/test_snapshots")


@pytest.fixture(autouse=True)
def cleanup_snapshots():
    """Clean up snapshot directory before and after each test."""
    if SNAPSHOT_DIR.exists():
        shutil.rmtree(SNAPSHOT_DIR)
    yield
    if SNAPSHOT_DIR.exists():
        shutil.rmtree(SNAPSHOT_DIR)


def test_pipeline_saves_snapshots():
    """Pipeline produces JPEG snapshots for tracked vehicles."""
    from backend.pipeline.deepstream.pipeline import run_pipeline

    summary = run_pipeline(
        str(CLIP_LYTLE),
        snapshot_dir=str(SNAPSHOT_DIR),
    )

    assert summary["unique_tracks"] > 0
    jpegs = list(SNAPSHOT_DIR.glob("*.jpg"))
    assert len(jpegs) > 0, "Should save at least one best-photo JPEG"

    for jpeg in jpegs:
        assert jpeg.stat().st_size > 0, f"JPEG {jpeg.name} should not be empty"
        # Track ID filename should be numeric
        assert jpeg.stem.isdigit(), f"JPEG filename should be track ID: {jpeg.name}"


def test_snapshots_are_valid_images():
    """Saved snapshots are valid, decodable images."""
    import cv2

    from backend.pipeline.deepstream.pipeline import run_pipeline

    run_pipeline(str(CLIP_LYTLE), snapshot_dir=str(SNAPSHOT_DIR))

    jpegs = list(SNAPSHOT_DIR.glob("*.jpg"))
    assert len(jpegs) > 0

    for jpeg in jpegs:
        img = cv2.imread(str(jpeg))
        assert img is not None, f"Failed to decode {jpeg.name}"
        h, w = img.shape[:2]
        assert h > 0 and w > 0, f"Image {jpeg.name} has zero dimensions"
        print(f"  {jpeg.name}: {w}x{h}")


def test_snapshot_logs_printed(capsys):
    """Pipeline prints 'best photo saved' log messages."""
    from backend.pipeline.deepstream.pipeline import run_pipeline

    run_pipeline(str(CLIP_LYTLE), snapshot_dir=str(SNAPSHOT_DIR))

    captured = capsys.readouterr()
    assert "best photo saved" in captured.out, "Should log best-photo saves"
