"""Frame renderer — bbox drawing + JPEG encoding for MJPEG output.

Downloads NV12 GPU frame to CPU, draws bounding boxes and track IDs,
optionally draws ROI polygon and entry/exit lines, encodes to JPEG.
"""

import cv2
import cupy as cp
import numpy as np

from backend.pipeline.custom.preprocess import nv12_to_bgr_gpu
from backend.pipeline.protocol import ChannelPhase

# Colors (BGR)
_GREEN = (0, 255, 0)
_WHITE = (255, 255, 255)
_CYAN = (255, 255, 0)
_YELLOW = (0, 255, 255)
_RED = (0, 0, 255)
_BOX_THICKNESS = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1
_JPEG_QUALITY = 80


class FrameRenderer:
    """Renders annotated frames for MJPEG output."""

    def render(
        self,
        nv12_frame: cp.ndarray,
        height: int,
        width: int,
        tracks: list[dict],
        phase: ChannelPhase,
        roi_polygon: list[tuple[float, float]] | None = None,
        entry_exit_lines: dict | None = None,
    ) -> bytes:
        """Download frame, draw annotations, encode JPEG.

        Returns: JPEG bytes.
        """
        # NV12 → BGR on GPU, then download to CPU
        bgr_gpu = nv12_to_bgr_gpu(nv12_frame, height, width)
        frame = cp.asnumpy(bgr_gpu)

        # Draw bounding boxes and track IDs
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

        # Draw ROI polygon (Setup + Analytics)
        if roi_polygon and len(roi_polygon) >= 3:
            pts = np.array(roi_polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=_CYAN, thickness=1)

        # Draw entry/exit lines
        if entry_exit_lines:
            for arm_id, line_data in entry_exit_lines.items():
                if isinstance(line_data, dict):
                    start = tuple(int(v) for v in line_data["start"])
                    end = tuple(int(v) for v in line_data["end"])
                    lbl = line_data.get("label", arm_id)
                else:
                    start = (int(line_data.start[0]), int(line_data.start[1]))
                    end = (int(line_data.end[0]), int(line_data.end[1]))
                    lbl = line_data.label
                cv2.line(frame, start, end, _YELLOW, 2)
                mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
                cv2.putText(
                    frame, lbl, (mid[0] - 20, mid[1] - 10),
                    _FONT, 0.4, _YELLOW, 1,
                )

        # Encode JPEG
        _, jpeg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
        )
        return jpeg.tobytes()

    def render_cpu_frame(
        self,
        frame: np.ndarray,
        tracks: list[dict],
    ) -> bytes:
        """Render on an already-downloaded CPU BGR frame. For best-photo crops."""
        for track in tracks:
            x, y, w, h = track["bbox"]
            tid = track["track_id"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), _GREEN, _BOX_THICKNESS)
            cv2.putText(
                frame, f"#{tid}", (x, y - 8),
                _FONT, _FONT_SCALE, _GREEN, _FONT_THICKNESS,
            )
        _, jpeg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
        )
        return jpeg.tobytes()
