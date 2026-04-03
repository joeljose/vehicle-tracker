"""GPU-resident preprocess: NV12→RGB, letterbox resize, normalize.

All operations stay on GPU via CuPy. The output is a (1, 3, 640, 640)
float32 tensor ready for TensorRT inference.
"""

from dataclasses import dataclass

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import zoom


@dataclass
class ScaleInfo:
    """Mapping info to convert detections back to original coordinates."""

    scale: float
    pad_top: int
    pad_left: int
    orig_h: int
    orig_w: int


# NV12 to RGB CUDA kernel (BT.601 color conversion)
_nv12_to_rgb_kernel = cp.RawKernel(
    r"""
extern "C" __global__
void nv12_to_rgb(const unsigned char* nv12, unsigned char* rgb,
                 int width, int y_height) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= y_height) return;

    int y_idx = y * width + x;
    unsigned char Y = nv12[y_idx];

    int uv_row = y / 2;
    int uv_col = (x / 2) * 2;
    int uv_idx = y_height * width + uv_row * width + uv_col;
    unsigned char U = nv12[uv_idx];
    unsigned char V = nv12[uv_idx + 1];

    float fY = (float)Y;
    float fU = (float)U - 128.0f;
    float fV = (float)V - 128.0f;

    float R = fY + 1.402f * fV;
    float G = fY - 0.344136f * fU - 0.714136f * fV;
    float B = fY + 1.772f * fU;

    R = fminf(fmaxf(R, 0.0f), 255.0f);
    G = fminf(fmaxf(G, 0.0f), 255.0f);
    B = fminf(fmaxf(B, 0.0f), 255.0f);

    int rgb_idx = (y * width + x) * 3;
    rgb[rgb_idx]     = (unsigned char)R;
    rgb[rgb_idx + 1] = (unsigned char)G;
    rgb[rgb_idx + 2] = (unsigned char)B;
}
""",
    "nv12_to_rgb",
)

# NV12 to BGR CUDA kernel (for OpenCV frame output)
_nv12_to_bgr_kernel = cp.RawKernel(
    r"""
extern "C" __global__
void nv12_to_bgr(const unsigned char* nv12, unsigned char* bgr,
                 int width, int y_height) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= y_height) return;

    int y_idx = y * width + x;
    unsigned char Y = nv12[y_idx];

    int uv_row = y / 2;
    int uv_col = (x / 2) * 2;
    int uv_idx = y_height * width + uv_row * width + uv_col;
    unsigned char U = nv12[uv_idx];
    unsigned char V = nv12[uv_idx + 1];

    float fY = (float)Y;
    float fU = (float)U - 128.0f;
    float fV = (float)V - 128.0f;

    float R = fY + 1.402f * fV;
    float G = fY - 0.344136f * fU - 0.714136f * fV;
    float B = fY + 1.772f * fU;

    R = fminf(fmaxf(R, 0.0f), 255.0f);
    G = fminf(fmaxf(G, 0.0f), 255.0f);
    B = fminf(fmaxf(B, 0.0f), 255.0f);

    int bgr_idx = (y * width + x) * 3;
    bgr[bgr_idx]     = (unsigned char)B;
    bgr[bgr_idx + 1] = (unsigned char)G;
    bgr[bgr_idx + 2] = (unsigned char)R;
}
""",
    "nv12_to_bgr",
)


def nv12_to_rgb_gpu(nv12_frame: cp.ndarray, height: int, width: int) -> cp.ndarray:
    """Convert NV12 CuPy array to RGB (h, w, 3) uint8 on GPU."""
    rgb = cp.empty((height, width, 3), dtype=cp.uint8)
    block = (32, 32, 1)
    grid = ((width + 31) // 32, (height + 31) // 32, 1)
    _nv12_to_rgb_kernel(
        grid, block, (nv12_frame, rgb, np.int32(width), np.int32(height)),
    )
    return rgb


def nv12_to_bgr_gpu(nv12_frame: cp.ndarray, height: int, width: int) -> cp.ndarray:
    """Convert NV12 CuPy array to BGR (h, w, 3) uint8 on GPU."""
    bgr = cp.empty((height, width, 3), dtype=cp.uint8)
    block = (32, 32, 1)
    grid = ((width + 31) // 32, (height + 31) // 32, 1)
    _nv12_to_bgr_kernel(
        grid, block, (nv12_frame, bgr, np.int32(width), np.int32(height)),
    )
    return bgr


class GpuPreprocessor:
    """GPU preprocess pipeline: NV12→RGB + letterbox resize + normalize."""

    def __init__(self, input_size: int = 640):
        self.input_size = input_size

    def __call__(
        self, nv12_frame: cp.ndarray, height: int, width: int,
    ) -> tuple[cp.ndarray, ScaleInfo]:
        """Preprocess NV12 frame for TRT inference.

        Returns:
            input_tensor: (1, 3, input_size, input_size) float32 on GPU.
            scale_info: For mapping detections to original coordinates.
        """
        # NV12 → RGB on GPU
        rgb = nv12_to_rgb_gpu(nv12_frame, height, width)

        # Letterbox resize
        s = self.input_size
        scale = min(s / height, s / width)
        new_h = int(height * scale)
        new_w = int(width * scale)

        rgb_float = rgb.astype(cp.float32) / 255.0
        scale_h = new_h / height
        scale_w = new_w / width
        resized = zoom(rgb_float, (scale_h, scale_w, 1.0), order=1)

        # Pad with 0.5 (gray, matching Ultralytics convention)
        padded = cp.full((s, s, 3), 0.5, dtype=cp.float32)
        pad_top = (s - new_h) // 2
        pad_left = (s - new_w) // 2
        padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w, :] = resized

        # HWC → CHW → NCHW
        nchw = padded.transpose(2, 0, 1)[cp.newaxis, ...]

        info = ScaleInfo(
            scale=scale,
            pad_top=pad_top,
            pad_left=pad_left,
            orig_h=height,
            orig_w=width,
        )
        return nchw, info
