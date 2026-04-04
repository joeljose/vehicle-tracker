"""Verify DeepStream-specific environment setup."""


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
