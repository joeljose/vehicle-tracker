"""Auto-skip DeepStream tests when running in the custom container."""

import os

import pytest

if os.environ.get("VT_BACKEND") not in ("deepstream", None):
    pytest.skip(
        "DeepStream tests require VT_BACKEND=deepstream",
        allow_module_level=True,
    )
