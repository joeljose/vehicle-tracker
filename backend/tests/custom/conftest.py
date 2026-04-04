"""Auto-skip custom pipeline tests when running in the DeepStream container."""

import os

import pytest

if os.environ.get("VT_BACKEND") != "custom":
    pytest.skip(
        "Custom pipeline tests require VT_BACKEND=custom",
        allow_module_level=True,
    )
