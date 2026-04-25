"""Auto-skip DeepStream tests unless running inside the deepstream container."""

import os

import pytest

if os.environ.get("VT_BACKEND") != "deepstream":
    pytest.skip(
        "DeepStream tests require VT_BACKEND=deepstream",
        allow_module_level=True,
    )
