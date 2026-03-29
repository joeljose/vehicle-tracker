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
