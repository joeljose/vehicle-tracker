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

import cupy as cp

from PyNvVideoCodec import CreateDecoder, CreateDemuxer

logger = logging.getLogger(__name__)


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

        # CreateDemuxer blocks on FIFO read until GStreamer starts writing.
        # Both sides synchronize automatically via the FIFO.
        self._demuxer = CreateDemuxer(self._fifo_path)
        self._decoder = CreateDecoder(
            gpuid=0,
            codec=self._demuxer.GetNvCodecId(),
            usedevicememory=True,
        )
        self._width = self._demuxer.Width()
        self._height = self._demuxer.Height()
        self._fps = self._demuxer.FrameRate()

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
        """Seek to beginning for loop mode."""
        self._demuxer.Seek(0)
        self._eos = False
        self._frame_buffer.clear()
        logger.debug("NvDecoder: looping %s", self._source[:80])

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
