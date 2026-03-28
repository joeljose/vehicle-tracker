"""Verify the dev environment is correctly set up."""


def test_python_version():
    import sys
    assert sys.version_info >= (3, 10), f"Python 3.10+ required, got {sys.version}"


def test_numpy_import():
    import numpy as np
    assert np.__version__


def test_opencv_import():
    import cv2
    assert cv2.__version__


def test_gstreamer_import():
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    Gst.init(None)
    assert Gst.version()


def test_deepstream_import():
    import pyservicemaker
    assert pyservicemaker
    # DS 8.0 uses pyservicemaker instead of pyds
    from pyservicemaker import Pipeline, Flow, Probe
    assert Pipeline and Flow and Probe


def test_gpu_available():
    """Verify GPU is accessible inside the container."""
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, "nvidia-smi failed"
    assert "RTX" in result.stdout or "GPU" in result.stdout, f"Unexpected GPU: {result.stdout}"


def test_project_structure():
    from pathlib import Path
    root = Path(__file__).parent.parent.parent
    expected_dirs = [
        "backend/pipeline/deepstream",
        "backend/pipeline/custom",
        "backend/api",
        "backend/config/sites",
        "backend/tests",
    ]
    for d in expected_dirs:
        assert (root / d).is_dir(), f"Missing directory: {d}"
