"""Unit tests for NvDecoder._open_demuxer_with_timeout (B15).

The HLS init path used to block forever on FIFO open if gst-launch failed
to start. These tests pin the new bounded-wait behavior:

- gst-launch exits before any writer appears → RuntimeError within ~1 s
- demuxer never returns and gst stays alive → TimeoutError after the
  bounded wait
- happy path (demuxer returns quickly) → no error, returns the demuxer
"""

import os
import subprocess
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_decoder(tmp_path):
    """Bare object exposing the attrs _open_demuxer_with_timeout reaches for."""
    from backend.pipeline.custom.decoder import NvDecoder

    obj = MagicMock(spec=NvDecoder)
    obj._channel_id = 99
    obj._fifo_path = str(tmp_path / "vt_fifo")
    os.mkfifo(obj._fifo_path)
    # Bind the real methods to our stub so they manipulate obj's attrs.
    obj._open_demuxer_with_timeout = NvDecoder._open_demuxer_with_timeout.__get__(obj)
    obj._unblock_fifo_reader = NvDecoder._unblock_fifo_reader.__get__(obj)
    yield obj
    if os.path.exists(obj._fifo_path):
        os.remove(obj._fifo_path)


def _block_forever_demuxer(*args, **kwargs):
    """Stand-in for CreateDemuxer that opens the FIFO read-side and blocks."""
    fifo = args[0]
    fd = os.open(fifo, os.O_RDONLY)  # blocks until writer connects
    os.read(fd, 1)  # then immediately drained on EOF
    os.close(fd)
    raise RuntimeError("FIFO closed without producing a stream")


def test_gst_exits_early_raises_runtimeerror_fast(fake_decoder):
    """gst-launch that exits with non-zero RC before connecting → fail fast."""
    # `false` exits immediately with rc=1; using it as a stand-in for any
    # gst-launch that fails to start (binary missing, codec error).
    fake_decoder._gst_proc = subprocess.Popen(
        ["false"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    t0 = time.monotonic()
    with patch(
        "backend.pipeline.custom.decoder.CreateDemuxer",
        side_effect=_block_forever_demuxer,
    ):
        with pytest.raises(RuntimeError, match="gst-launch exited"):
            fake_decoder._open_demuxer_with_timeout()
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"raised in {elapsed:.2f}s; should be <2s"


def test_happy_path_returns_demuxer(fake_decoder):
    """When CreateDemuxer returns normally, _open_demuxer_with_timeout
    returns its result (no race-loss on a successful open)."""
    fake_decoder._gst_proc = subprocess.Popen(
        ["sleep", "5"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    sentinel = object()

    def fast_demuxer(*args, **kwargs):
        return sentinel

    try:
        with patch(
            "backend.pipeline.custom.decoder.CreateDemuxer",
            side_effect=fast_demuxer,
        ):
            assert fake_decoder._open_demuxer_with_timeout() is sentinel
    finally:
        fake_decoder._gst_proc.terminate()
        fake_decoder._gst_proc.wait(timeout=2)


def test_timeout_when_neither_side_progresses(fake_decoder, monkeypatch):
    """gst stays alive but never writes; demuxer also never returns →
    TimeoutError after the bounded wait. We monkeypatch the timeout to keep
    the test fast."""
    monkeypatch.setattr(
        "backend.pipeline.custom.decoder._HLS_INIT_TIMEOUT_S", 0.5,
    )
    fake_decoder._gst_proc = subprocess.Popen(
        ["sleep", "10"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    try:
        with patch(
            "backend.pipeline.custom.decoder.CreateDemuxer",
            side_effect=_block_forever_demuxer,
        ):
            with pytest.raises(TimeoutError, match="did not begin writing"):
                fake_decoder._open_demuxer_with_timeout()
    finally:
        fake_decoder._gst_proc.terminate()
        fake_decoder._gst_proc.wait(timeout=2)


def test_unblock_fifo_reader_releases_blocked_open(tmp_path):
    """_unblock_fifo_reader should release a thread blocked on FIFO open."""
    from backend.pipeline.custom.decoder import NvDecoder

    obj = MagicMock(spec=NvDecoder)
    obj._fifo_path = str(tmp_path / "vt_fifo_unblock")
    os.mkfifo(obj._fifo_path)
    obj._unblock_fifo_reader = NvDecoder._unblock_fifo_reader.__get__(obj)

    blocked_done = threading.Event()

    def block_on_open():
        try:
            fd = os.open(obj._fifo_path, os.O_RDONLY)
            os.close(fd)
        finally:
            blocked_done.set()

    t = threading.Thread(target=block_on_open, daemon=True)
    t.start()
    time.sleep(0.1)  # ensure the thread is parked in open()
    assert not blocked_done.is_set()

    obj._unblock_fifo_reader()

    assert blocked_done.wait(timeout=1.0), "reader should unblock within 1s"
    os.remove(obj._fifo_path)
