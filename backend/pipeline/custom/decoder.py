"""NVDEC hardware decoder wrapper.

Decodes video files to NV12 CUDA surfaces via PyNvVideoCodec.
One instance per channel. Supports looping for Setup phase.

For HLS streams (YouTube Live), a GStreamer subprocess handles
HTTPS download + HLS demux and pipes raw H.264 via a named FIFO.
PyNvVideoCodec reads the FIFO and decodes on NVDEC — same GPU-resident
output as local files.
"""

import logging
import os
import subprocess
import threading
import time

import cupy as cp

from PyNvVideoCodec import CreateDecoder, CreateDemuxer

logger = logging.getLogger(__name__)

# Bounded wait for gst-launch to begin writing to the FIFO during _init_hls.
# A real HLS source needs ~1-3 s for DNS + manifest + first segment; this gives
# generous slack while still failing fast on a misconfig (gst binary missing,
# codec not present, malformed URL).
_HLS_INIT_TIMEOUT_S = 15.0


class NvDecoder:
    """NVDEC hardware decoder — returns NV12 CuPy arrays on GPU.

    For local files: CreateDemuxer reads the file directly.
    For HLS URLs: gst-launch-1.0 subprocess demuxes HLS to a FIFO,
    CreateDemuxer reads from the FIFO.
    """

    def __init__(self, source: str, *, loop: bool = False, channel_id: int = 0):
        """
        Args:
            source: File path (mp4/mkv/avi) or HLS URL.
            loop: If True, seek to start on EOS (Setup phase).
            channel_id: Used for unique FIFO naming.
        """
        self._source = source
        self._loop = loop
        self._channel_id = channel_id
        self._is_hls = source.startswith(("http://", "https://"))
        self._gst_proc: subprocess.Popen | None = None
        self._fifo_path: str | None = None
        self._released = False

        if self._is_hls:
            self._init_hls(source)
        else:
            self._init_file(source)

        self._nv12_height = int(self._height * 3 // 2)
        self._frame_buffer: list[tuple[cp.ndarray, int]] = []
        self._eos = False
        # Seek epoch is bumped every time the demuxer is rewound (Setup loop).
        # Consumers can compare to detect "decoder seeked" without inspecting
        # PTS deltas — see CustomPipeline._pipeline_loop / _display_loop.
        self._seek_epoch = 0

        logger.info(
            "NvDecoder: %s (%dx%d @ %.1ffps, loop=%s, hls=%s)",
            source[:80], self._width, self._height, self._fps, loop, self._is_hls,
        )

    def _init_file(self, source: str):
        """Initialize decoder for local file sources."""
        self._demuxer = CreateDemuxer(source)
        self._decoder = CreateDecoder(
            gpuid=0,
            codec=self._demuxer.GetNvCodecId(),
            usedevicememory=True,
        )
        self._width = self._demuxer.Width()
        self._height = self._demuxer.Height()
        self._fps = self._demuxer.FrameRate()

    def _init_hls(self, source: str):
        """Initialize decoder for HLS URLs via GStreamer FIFO.

        GStreamer handles HTTPS + HLS demux → raw H.264 → mpegtsmux → FIFO.
        PyNvVideoCodec reads the FIFO as if it were a regular MPEG-TS file.
        """
        self._fifo_path = f"/tmp/vt_hls_{self._channel_id}.ts"

        # Clean up stale FIFO
        if os.path.exists(self._fifo_path):
            os.remove(self._fifo_path)
        os.mkfifo(self._fifo_path)

        # Start GStreamer subprocess — blocks on FIFO write until reader connects
        gst_cmd = [
            "gst-launch-1.0", "-q",
            "souphttpsrc", f"location={source}", "!",
            "hlsdemux", "!",
            "tsdemux", "!",
            "h264parse", "!",
            "mpegtsmux", "!",
            "filesink", f"location={self._fifo_path}",
        ]
        self._gst_proc = subprocess.Popen(
            gst_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        logger.info(
            "NvDecoder: GStreamer HLS subprocess started (pid=%d)",
            self._gst_proc.pid,
        )

        # CreateDemuxer's underlying open() on the FIFO blocks until GStreamer
        # opens the write end. If gst-launch fails to start (binary missing,
        # codec error, etc.) the writer never appears and CreateDemuxer would
        # block forever — halting the pipeline thread for every channel. Race
        # the demuxer-open against gst.poll() so a gst exit surfaces as a
        # clean RuntimeError instead of a multi-channel deadlock (B15).
        self._demuxer = self._open_demuxer_with_timeout()
        self._decoder = CreateDecoder(
            gpuid=0,
            codec=self._demuxer.GetNvCodecId(),
            usedevicememory=True,
        )
        self._width = self._demuxer.Width()
        self._height = self._demuxer.Height()
        self._fps = self._demuxer.FrameRate()

    def _open_demuxer_with_timeout(self):
        """Open the FIFO-backed demuxer, racing against gst-launch exit.

        Runs ``CreateDemuxer`` on a worker thread so we can poll
        ``self._gst_proc`` for early exit. If gst exits before any writer
        connects, briefly open the FIFO write end ourselves to unblock the
        demuxer thread (so it doesn't leak as a perpetually-stuck thread)
        and raise ``RuntimeError`` with the gst stderr tail.
        """
        result: dict = {}

        def _open():
            try:
                result["demuxer"] = CreateDemuxer(self._fifo_path)
            except Exception as exc:
                result["error"] = exc

        t = threading.Thread(
            target=_open,
            name=f"hls-demuxer-open-{self._channel_id}",
            daemon=True,
        )
        t.start()

        deadline = time.monotonic() + _HLS_INIT_TIMEOUT_S
        while time.monotonic() < deadline:
            if result:
                break
            rc = self._gst_proc.poll()
            if rc is not None:
                stderr = ""
                if self._gst_proc.stderr is not None:
                    try:
                        stderr = self._gst_proc.stderr.read(4096).decode(
                            "utf-8", "replace",
                        )
                    except Exception:
                        pass
                # Unblock the worker so it doesn't outlive us as a stuck
                # thread holding a FIFO fd.
                self._unblock_fifo_reader()
                raise RuntimeError(
                    "gst-launch exited before HLS source ready "
                    f"(rc={rc}): {stderr.strip()[-300:]}"
                )
            time.sleep(0.05)
        else:
            self._unblock_fifo_reader()
            raise TimeoutError(
                f"gst-launch did not begin writing within {_HLS_INIT_TIMEOUT_S}s"
            )

        if "error" in result:
            raise result["error"]
        return result["demuxer"]

    def _unblock_fifo_reader(self) -> None:
        """Briefly open the FIFO write end and close it.

        The blocking ``open(O_RDONLY)`` on the FIFO from CreateDemuxer's
        worker returns as soon as any writer connects; opening + closing the
        write side immediately gives it an EOF and lets the worker fall
        through to its error path instead of leaking forever.
        """
        if not self._fifo_path:
            return
        try:
            fd = os.open(self._fifo_path, os.O_WRONLY | os.O_NONBLOCK)
            os.close(fd)
        except OSError:
            # FIFO already gone, or kernel reports no readers — both fine.
            pass

    def read(self) -> tuple[cp.ndarray | None, int]:
        """Decode next frame.

        Returns:
            (nv12_gpu_frame, pts_ms) or (None, -1) on EOS.
        """
        if self._released:
            return None, -1

        while not self._frame_buffer:
            if self._eos:
                if self._loop and not self._is_hls:
                    self._restart()
                    continue
                return None, -1

            packet = self._demuxer.Demux()
            if packet.bsl == 0:
                # EOS — flush decoder
                self._eos = True
                frames = self._decoder.Decode(packet)
                for f in frames:
                    self._frame_buffer.append(self._frame_to_cupy(f))
                if not self._frame_buffer:
                    if self._loop and not self._is_hls:
                        self._restart()
                        continue
                    return None, -1
                break

            frames = self._decoder.Decode(packet)
            for f in frames:
                self._frame_buffer.append(self._frame_to_cupy(f))

        if self._frame_buffer:
            return self._frame_buffer.pop(0)
        return None, -1

    def _frame_to_cupy(self, frame) -> tuple[cp.ndarray, int]:
        """Convert DecodedFrame to (CuPy NV12 array, pts_ms)."""
        ptr = frame.GetPtrToPlane(0)
        # Keep reference to frame so GPU memory isn't freed
        mem = cp.cuda.UnownedMemory(ptr, self._nv12_height * self._width, frame)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        arr = cp.ndarray(
            (self._nv12_height, self._width), dtype=cp.uint8, memptr=memptr,
        )
        # Copy to owned memory — decoder may reuse the surface
        owned = arr.copy()
        # getPTS() returns 90kHz MPEG ticks — convert to milliseconds
        pts_ms = int(frame.getPTS() * 1000 / 90000) if hasattr(frame, "getPTS") else 0
        return owned, pts_ms

    def _restart(self):
        """Seek to beginning for loop mode. Bumps seek_epoch so consumers
        can re-anchor their PTS-pacing baselines."""
        self._demuxer.Seek(0)
        self._eos = False
        self._frame_buffer.clear()
        self._seek_epoch += 1
        logger.debug("NvDecoder: looping %s (epoch=%d)", self._source[:80], self._seek_epoch)

    def reconnect(self, new_source: str) -> None:
        """Reconnect to a new HLS URL (for stream recovery).

        Releases the current decoder and GStreamer subprocess,
        then re-initializes with the new URL.
        """
        self.release()
        self._source = new_source
        self._released = False
        self._eos = False
        self._frame_buffer.clear()
        self._init_hls(new_source)
        self._nv12_height = int(self._height * 3 // 2)
        logger.info(
            "NvDecoder: reconnected (%dx%d @ %.1ffps)",
            self._width, self._height, self._fps,
        )

    def release(self):
        """Release NVDEC resources and GStreamer subprocess."""
        if not self._released:
            self._released = True
            self._frame_buffer.clear()
            self._decoder = None
            self._demuxer = None

            # Kill GStreamer subprocess
            if self._gst_proc is not None:
                self._gst_proc.terminate()
                try:
                    self._gst_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._gst_proc.kill()
                    self._gst_proc.wait(timeout=2)
                self._gst_proc = None

            # Clean up FIFO
            if self._fifo_path and os.path.exists(self._fifo_path):
                os.remove(self._fifo_path)
                self._fifo_path = None

            logger.debug("NvDecoder: released %s", self._source[:80])

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_hls(self) -> bool:
        return self._is_hls

    @property
    def seek_epoch(self) -> int:
        """Monotonically increasing counter; incremented every time the
        decoder is rewound to the start of the source (Setup-phase loop).
        Consumers compare against a previous value to detect a seek."""
        return self._seek_epoch
