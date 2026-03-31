"""Shared test fixtures."""

import logging
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure backend package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging for tests — output to stderr so capsys doesn't capture it,
# but capfd and pytest's log capture will pick it up.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s - %(levelname)s - %(message)s",
)


@pytest.fixture
def app():
    from backend.main import create_app

    return create_app(backend="fake")


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture(autouse=True)
def _fake_video_files(tmp_path, monkeypatch):
    """Create fake video files so source validation passes in tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for name in (
        "video.mp4",
        "a.mp4",
        "b.mp4",
        "c.mp4",
        "test.mp4",
        "lytle_south.mp4",
        "741_73.mp4",
    ):
        (data_dir / name).write_bytes(b"fake")
    # Patch the source paths used in tests to point to tmp_path
    original_exists = Path.exists

    def patched_exists(self):
        if str(self).startswith("/data/"):
            return (tmp_path / str(self).lstrip("/")).exists()
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", patched_exists)
