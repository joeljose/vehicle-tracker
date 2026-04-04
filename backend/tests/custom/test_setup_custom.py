"""Verify custom pipeline environment setup."""

import subprocess


def test_pynvvideocodec_import():
    from PyNvVideoCodec import CreateDecoder, CreateDemuxer

    assert CreateDecoder and CreateDemuxer


def test_tensorrt_import():
    import tensorrt as trt

    assert trt.__version__


def test_cupy_import():
    import cupy as cp

    assert cp.__version__
    # Verify GPU access
    a = cp.array([1, 2, 3])
    assert int(cp.sum(a)) == 6


def test_gst_launch_available():
    """gst-launch-1.0 is needed for HLS decode via FIFO."""
    result = subprocess.run(
        ["gst-launch-1.0", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "gst-launch-1.0 not found"


def test_gst_hls_plugins():
    """Verify GStreamer HLS plugins are available."""
    for plugin in ("souphttpsrc", "hlsdemux", "tsdemux", "h264parse", "mpegtsmux"):
        result = subprocess.run(
            ["gst-inspect-1.0", plugin],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"GStreamer plugin '{plugin}' not found"
