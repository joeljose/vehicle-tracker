"""Shared test fixtures."""

import logging
import sys
from pathlib import Path

# Ensure backend package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging for tests — output to stderr so capsys doesn't capture it,
# but capfd and pytest's log capture will pick it up.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s - %(levelname)s - %(message)s",
)
