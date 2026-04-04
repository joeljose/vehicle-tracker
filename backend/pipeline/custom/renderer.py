"""Frame renderer — bbox drawing + JPEG encoding for MJPEG output.

Downloads NV12 GPU frame to CPU, draws bounding boxes and track IDs,
encodes to JPEG. ROI and entry/exit lines are rendered by the frontend
as HTML overlays (common to both backends).
"""

import cv2
import cupy as cp
import numpy as np

from backend.pipeline.custom.preprocess import nv12_to_bgr_gpu

# Colors (BGR)
_GREEN = (0, 255, 0)
_BOX_THICKNESS = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1
_JPEG_QUALITY = 80


class FrameRenderer:
    """Renders annotated frames for MJPEG output."""

    def decode_nv12(self, nv12_frame: cp.ndarray, height: int, width: int) -> np.ndarray:
        """NV12 → BGR on GPU, then download to CPU. Returns BGR numpy array."""
        bgr_gpu = nv12_to_bgr_gpu(nv12_frame, height, width)
        return cp.asnumpy(bgr_gpu)

    def render(
        self,
        nv12_frame: cp.ndarray,
        height: int,
        width: int,
        tracks: list[dict],
    ) -> bytes:
        """Download frame, draw bbox annotations, encode JPEG."""
        frame = self.decode_nv12(nv12_frame, height, width)
        return self.annotate_and_encode(frame, tracks)

    def annotate_and_encode(
        self,
        frame: np.ndarray,
        tracks: list[dict],
    ) -> bytes:
        """Draw bounding boxes on a BGR frame and encode to JPEG."""
        for track in tracks:
            x, y, w, h = track["bbox"]
            tid = track["track_id"]
            conf = track["confidence"]
            label = track.get("class_name", "")

            cv2.rectangle(frame, (x, y), (x + w, y + h), _GREEN, _BOX_THICKNESS)
            text = f"#{tid} {label} {conf:.2f}"
            cv2.putText(
                frame, text, (x, y - 8),
                _FONT, _FONT_SCALE, _GREEN, _FONT_THICKNESS,
            )

        _, jpeg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
        )
        return jpeg.tobytes()

    @staticmethod
    def encode_clean(frame: np.ndarray) -> bytes:
        """Encode a BGR frame to JPEG with no annotations."""
        _, jpeg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
        )
        return jpeg.tobytes()
