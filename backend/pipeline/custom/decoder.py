"""NVDEC hardware decoder wrapper.

Decodes video files to NV12 CUDA surfaces via PyNvVideoCodec.
One instance per channel. Supports looping for Setup phase.
"""

import logging

import cupy as cp

from PyNvVideoCodec import CreateDecoder, CreateDemuxer

logger = logging.getLogger(__name__)


class NvDecoder:
    """NVDEC hardware decoder — returns NV12 CuPy arrays on GPU."""

    def __init__(self, source: str, *, loop: bool = False):
        """
        Args:
            source: File path (mp4/mkv/avi).
            loop: If True, seek to start on EOS (Setup phase).
        """
        self._source = source
        self._loop = loop
        self._demuxer = CreateDemuxer(source)
        self._decoder = CreateDecoder(
            gpuid=0,
            codec=self._demuxer.GetNvCodecId(),
            usedevicememory=True,
        )
        self._width = self._demuxer.Width()
        self._height = self._demuxer.Height()
        self._fps = self._demuxer.FrameRate()
        self._nv12_height = int(self._height * 3 // 2)
        self._frame_buffer: list[tuple[cp.ndarray, int]] = []
        self._eos = False
        self._released = False
        logger.info(
            "NvDecoder: %s (%dx%d @ %.1ffps, loop=%s)",
            source, self._width, self._height, self._fps, loop,
        )

    def read(self) -> tuple[cp.ndarray | None, int]:
        """Decode next frame.

        Returns:
            (nv12_gpu_frame, pts_ms) or (None, -1) on EOS.
        """
        if self._released:
            return None, -1

        while not self._frame_buffer:
            if self._eos:
                if self._loop:
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
                    if self._loop:
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
        pts_ms = int(frame.getPTS() / 1000) if hasattr(frame, "getPTS") else 0
        return owned, pts_ms

    def _restart(self):
        """Seek to beginning for loop mode."""
        self._demuxer.Seek(0)
        self._eos = False
        self._frame_buffer.clear()
        logger.debug("NvDecoder: looping %s", self._source)

    def release(self):
        """Release NVDEC resources."""
        if not self._released:
            self._released = True
            self._frame_buffer.clear()
            self._decoder = None
            self._demuxer = None
            logger.debug("NvDecoder: released %s", self._source)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return self._fps
