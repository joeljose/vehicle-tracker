"""Experiment 12: GPU-resident pipeline — NVDEC decode + direct TensorRT inference

Tests the GPU-resident data flow: NVDEC decode to CUDA surface, GPU preprocess
(color convert + resize + normalize) via CuPy, direct TensorRT inference, and
CPU-side NMS + tracking.

Goal: Measure fps and per-stage latency for single and dual channel, compare
against exp11 baseline (62.9 fps single, 37.5 fps dual with Ultralytics wrapper).

Usage (inside backend container):
    python3 experiments/exp12_gpu_resident.py /data/test_clips/741_73_1min.mp4
    python3 experiments/exp12_gpu_resident.py /data/test_clips/741_73_1min.mp4 /data/test_clips/lytle_south_1min.mp4
"""

import os
import sys
import time

import cupy as cp
import numpy as np
import tensorrt as trt

# ---------------------------------------------------------------------------
# NVDEC Decoder
# ---------------------------------------------------------------------------

class NvDecoder:
    """NVDEC hardware decoder wrapper. Outputs NV12 CuPy arrays on GPU."""

    def __init__(self, path):
        from PyNvVideoCodec import CreateDecoder, CreateDemuxer
        self.demuxer = CreateDemuxer(path)
        self.decoder = CreateDecoder(
            gpuid=0,
            codec=self.demuxer.GetNvCodecId(),
            usedevicememory=True,
        )
        self.width = self.demuxer.Width()
        self.height = self.demuxer.Height()
        self.fps = self.demuxer.FrameRate()
        # NV12: height * 1.5 rows, width columns
        self.nv12_height = int(self.height * 3 // 2)
        self._frame_buffer = []  # buffered decoded frames
        self._eos = False

    def decode_frame(self):
        """Decode one frame. Returns CuPy NV12 array (h*1.5, w) on GPU, or None on EOS."""
        # Return buffered frame if available
        while not self._frame_buffer:
            if self._eos:
                return None

            packet = self.demuxer.Demux()
            if packet.bsl == 0:
                # EOS — flush decoder
                self._eos = True
                frames = self.decoder.Decode(packet)
                for f in frames:
                    self._frame_buffer.append(self._frame_to_cupy(f))
                if not self._frame_buffer:
                    return None
                break

            frames = self.decoder.Decode(packet)
            for f in frames:
                self._frame_buffer.append(self._frame_to_cupy(f))

        if self._frame_buffer:
            return self._frame_buffer.pop(0)
        return None

    def _frame_to_cupy(self, frame):
        """Convert DecodedFrame to CuPy array via raw GPU pointer."""
        ptr = frame.GetPtrToPlane(0)
        mem = cp.cuda.UnownedMemory(ptr, self.nv12_height * self.width, frame)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        return cp.ndarray((self.nv12_height, self.width), dtype=cp.uint8, memptr=memptr)


def create_decoder(path):
    """Create NVDEC decoder."""
    return NvDecoder(path)


def decode_frame_nvdec(decoder):
    """Decode one frame, return as CuPy NV12 array (GPU) or None on EOS."""
    return decoder.decode_frame()


# ---------------------------------------------------------------------------
# GPU Preprocess (CuPy)
# ---------------------------------------------------------------------------

# NV12 to RGB CUDA kernel via CuPy
_nv12_to_rgb_kernel = cp.RawKernel(r'''
extern "C" __global__
void nv12_to_rgb(const unsigned char* nv12, unsigned char* rgb,
                 int width, int y_height) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= y_height) return;

    int y_idx = y * width + x;
    unsigned char Y = nv12[y_idx];

    // UV plane starts at y_height * width
    int uv_row = y / 2;
    int uv_col = (x / 2) * 2;
    int uv_idx = y_height * width + uv_row * width + uv_col;
    unsigned char U = nv12[uv_idx];
    unsigned char V = nv12[uv_idx + 1];

    // YUV to RGB (BT.601)
    float fY = (float)Y;
    float fU = (float)U - 128.0f;
    float fV = (float)V - 128.0f;

    float R = fY + 1.402f * fV;
    float G = fY - 0.344136f * fU - 0.714136f * fV;
    float B = fY + 1.772f * fU;

    // Clamp
    R = fminf(fmaxf(R, 0.0f), 255.0f);
    G = fminf(fmaxf(G, 0.0f), 255.0f);
    B = fminf(fmaxf(B, 0.0f), 255.0f);

    int rgb_idx = (y * width + x) * 3;
    rgb[rgb_idx]     = (unsigned char)R;
    rgb[rgb_idx + 1] = (unsigned char)G;
    rgb[rgb_idx + 2] = (unsigned char)B;
}
''', 'nv12_to_rgb')


def _nv12_to_rgb_gpu(nv12_frame, height, width):
    """Convert NV12 CuPy array to RGB CuPy array on GPU."""
    rgb = cp.empty((height, width, 3), dtype=cp.uint8)
    block = (32, 32, 1)
    grid = ((width + 31) // 32, (height + 31) // 32, 1)
    _nv12_to_rgb_kernel(grid, block, (nv12_frame, rgb, np.int32(width), np.int32(height)))
    return rgb


def preprocess_gpu(gpu_frame, input_h=640, input_w=640):
    """GPU preprocess: NV12->RGB (if needed), resize via CuPy, normalize.

    Returns: CuPy array of shape (1, 3, input_h, input_w) float32, on GPU.
    """
    h, w = gpu_frame.shape[0], gpu_frame.shape[1]

    if gpu_frame.ndim == 2:
        # NV12 format: (h*1.5, w) — need color conversion
        y_h = int(h * 2 / 3)  # actual frame height
        rgb = _nv12_to_rgb_gpu(gpu_frame, y_h, w)
    elif gpu_frame.ndim == 3 and gpu_frame.shape[2] == 3:
        # Already RGB/BGR
        rgb = gpu_frame
        y_h = h
    else:
        raise ValueError(f"Unexpected frame shape: {gpu_frame.shape}")

    # Letterbox resize: scale to fit in input_h x input_w, pad with 0.5
    rgb_float = rgb.astype(cp.float32) / 255.0
    scale = min(input_h / y_h, input_w / w)
    new_h = int(y_h * scale)
    new_w = int(w * scale)

    from cupyx.scipy.ndimage import zoom
    scale_h = new_h / y_h
    scale_w = new_w / w
    resized = zoom(rgb_float, (scale_h, scale_w, 1.0), order=1)

    # Pad to input_h x input_w (letterbox with 0.5 fill)
    padded = cp.full((input_h, input_w, 3), 0.5, dtype=cp.float32)
    pad_top = (input_h - new_h) // 2
    pad_left = (input_w - new_w) // 2
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w, :] = resized

    # HWC -> CHW -> NCHW
    chw = padded.transpose(2, 0, 1)
    nchw = chw[cp.newaxis, ...]

    return nchw, (scale, pad_top, pad_left, y_h, w)


# ---------------------------------------------------------------------------
# Direct TensorRT Inference
# ---------------------------------------------------------------------------

class TRTInferencer:
    """Direct TensorRT inference — GPU tensor in, GPU tensor out."""

    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Get binding info
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)

        print(f"  TRT engine loaded: {engine_path}")
        print(f"  Input:  {self.input_name} {self.input_shape}")
        print(f"  Output: {self.output_name} {self.output_shape}")

        # Pre-allocate output buffer on GPU
        self.output_buffer = cp.empty(tuple(self.output_shape), dtype=cp.float32)

        # CUDA stream
        self.stream = cp.cuda.Stream()

    def infer(self, input_tensor):
        """Run inference. input_tensor is CuPy array (1,3,640,640) on GPU.
        Returns CuPy array (1,84,8400) on GPU."""
        input_tensor = cp.ascontiguousarray(input_tensor, dtype=cp.float32)

        self.context.set_tensor_address(self.input_name, input_tensor.data.ptr)
        self.context.set_tensor_address(self.output_name, self.output_buffer.data.ptr)

        self.context.execute_async_v3(self.stream.ptr)
        self.stream.synchronize()

        return self.output_buffer


# ---------------------------------------------------------------------------
# CPU NMS (postprocess)
# ---------------------------------------------------------------------------

def postprocess_cpu(output, conf_thresh=0.25, iou_thresh=0.45,
                    scale_info=None, classes=None):
    """NMS on CPU. output is (1, 84, 8400) numpy array.
    Returns (N, 6) array: [x1, y1, x2, y2, conf, class_id]."""
    preds = output[0].T  # (8400, 84)

    boxes_cxcywh = preds[:, :4]
    class_scores = preds[:, 4:]

    max_scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    mask = max_scores > conf_thresh
    if classes is not None:
        class_mask = np.isin(class_ids, classes)
        mask = mask & class_mask

    boxes_cxcywh = boxes_cxcywh[mask]
    scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(scores) == 0:
        return np.empty((0, 6), dtype=np.float32)

    # cxcywh -> xyxy
    boxes_xyxy = np.zeros_like(boxes_cxcywh)
    boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

    keep = _nms(boxes_xyxy, scores, iou_thresh)
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    # Scale back to original frame coordinates
    if scale_info:
        scale, pad_top, pad_left, orig_h, orig_w = scale_info
        boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - pad_left) / scale
        boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - pad_top) / scale
        boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - pad_left) / scale
        boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - pad_top) / scale

        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

    return np.column_stack([boxes_xyxy, scores, class_ids.astype(np.float32)])


def _nms(boxes, scores, iou_thresh):
    """Simple NMS."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        mask = iou <= iou_thresh
        order = order[1:][mask]

    return keep


# ---------------------------------------------------------------------------
# BoT-SORT standalone
# ---------------------------------------------------------------------------

class DetectionResults:
    """Minimal Results-like object for BoT-SORT tracker.
    Wraps a (N, 6) numpy array [x1, y1, x2, y2, conf, cls]."""

    def __init__(self, dets):
        """dets: (N, 6) array [x1, y1, x2, y2, conf, cls]"""
        self._dets = dets if len(dets) > 0 else np.empty((0, 6), dtype=np.float32)

    @property
    def conf(self):
        return self._dets[:, 4]

    @property
    def xyxy(self):
        return self._dets[:, :4]

    @property
    def cls(self):
        return self._dets[:, 5]

    @property
    def xywh(self):
        """Convert xyxy to xywh (center x, center y, width, height)."""
        xyxy = self._dets[:, :4]
        xywh = np.empty_like(xyxy)
        xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2  # cx
        xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2  # cy
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]  # w
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]  # h
        return xywh

    def __len__(self):
        return len(self._dets)

    def __getitem__(self, idx):
        return DetectionResults(self._dets[idx])


def create_tracker():
    """Create standalone BoT-SORT tracker (no ReID, no GMC)."""
    from ultralytics.trackers.bot_sort import BOTSORT
    import argparse

    args = argparse.Namespace(
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
        gmc_method="none",
        proximity_thresh=0.5,
        appearance_thresh=0.8,
        with_reid=False,
    )
    tracker = BOTSORT(args)
    return tracker


def update_tracker(tracker, detections, frame_shape):
    """Update tracker with detections.
    detections: (N, 6) array [x1,y1,x2,y2,conf,cls]
    Returns: (M, N) tracked array or empty array.
    """
    results = DetectionResults(detections)
    if len(results) == 0:
        tracker.multi_predict()
        return np.empty((0, 7), dtype=np.float32)

    tracks = tracker.update(results)
    if len(tracks) == 0:
        return np.empty((0, 7), dtype=np.float32)

    return np.array(tracks, dtype=np.float32)


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def test_decode_only(path, max_frames=300):
    """Benchmark NVDEC decode speed."""
    dec = create_decoder(path)
    timings = []
    count = 0
    for _ in range(max_frames):
        t0 = time.monotonic()
        frame = decode_frame_nvdec(dec)
        if frame is None:
            break
        cp.cuda.Device(0).synchronize()
        timings.append(time.monotonic() - t0)
        count += 1

    return count, timings


def test_single_channel(engine_path, source, max_inferences=300):
    """Single channel: NVDEC decode -> GPU preprocess -> TRT infer -> NMS -> track."""
    trt_infer = TRTInferencer(engine_path)
    tracker = create_tracker()
    decoder = create_decoder(source)

    vehicle_classes = [2, 3, 5, 7]

    timings = {"decode": [], "preprocess": [], "infer": [], "nms": [],
               "track": [], "total": []}
    frame_count = 0
    infer_count = 0
    all_ids = set()

    while infer_count < max_inferences:
        t_total = time.monotonic()

        # Decode (GPU)
        t0 = time.monotonic()
        gpu_frame = decode_frame_nvdec(decoder)
        if gpu_frame is None:
            break
        cp.cuda.Device(0).synchronize()
        timings["decode"].append(time.monotonic() - t0)

        # Skip every other frame (60fps -> 30fps)
        frame_count += 1
        if frame_count % 2 != 0:
            continue

        # GPU preprocess
        t0 = time.monotonic()
        input_tensor, scale_info = preprocess_gpu(gpu_frame)
        cp.cuda.Device(0).synchronize()
        timings["preprocess"].append(time.monotonic() - t0)

        # TRT inference (GPU)
        t0 = time.monotonic()
        output = trt_infer.infer(input_tensor)
        timings["infer"].append(time.monotonic() - t0)

        # NMS (CPU)
        t0 = time.monotonic()
        output_cpu = cp.asnumpy(output)
        dets = postprocess_cpu(output_cpu, conf_thresh=0.25, iou_thresh=0.45,
                               scale_info=scale_info, classes=vehicle_classes)
        timings["nms"].append(time.monotonic() - t0)

        # Track (CPU)
        t0 = time.monotonic()
        orig_h, orig_w = scale_info[3], scale_info[4]
        tracks = update_tracker(tracker, dets, (orig_h, orig_w, 3))
        timings["track"].append(time.monotonic() - t0)

        if len(tracks) > 0:
            for tid in tracks[:, 4].astype(int):
                all_ids.add(tid)

        timings["total"].append(time.monotonic() - t_total)
        infer_count += 1

    return infer_count, len(all_ids), timings


def test_two_channel(engine_path, sources, max_inferences=150):
    """Two channels: NVDEC decode both -> GPU preprocess -> TRT infer (sequential) -> NMS -> track."""
    trt_infer = TRTInferencer(engine_path)
    tracker_a = create_tracker()
    tracker_b = create_tracker()

    src_b = sources[1] if len(sources) > 1 else sources[0]
    decoder_a = create_decoder(sources[0])
    decoder_b = create_decoder(src_b)

    vehicle_classes = [2, 3, 5, 7]

    timings = {"decode": [], "preprocess": [], "infer": [], "nms": [],
               "track": [], "total": []}
    frame_count = 0
    infer_count = 0
    ids_a, ids_b = set(), set()

    while infer_count < max_inferences:
        t_total = time.monotonic()

        # Decode both (GPU)
        t0 = time.monotonic()
        frame_a = decode_frame_nvdec(decoder_a)
        frame_b = decode_frame_nvdec(decoder_b)
        if frame_a is None or frame_b is None:
            break
        cp.cuda.Device(0).synchronize()
        timings["decode"].append(time.monotonic() - t0)

        frame_count += 1
        if frame_count % 2 != 0:
            continue

        # GPU preprocess both
        t0 = time.monotonic()
        input_a, scale_a = preprocess_gpu(frame_a)
        input_b, scale_b = preprocess_gpu(frame_b)
        cp.cuda.Device(0).synchronize()
        timings["preprocess"].append(time.monotonic() - t0)

        # TRT inference both (sequential)
        t0 = time.monotonic()
        output_a = trt_infer.infer(input_a)
        output_a_cpu = cp.asnumpy(output_a.copy())  # copy before buffer reuse
        output_b = trt_infer.infer(input_b)
        output_b_cpu = cp.asnumpy(output_b)
        timings["infer"].append(time.monotonic() - t0)

        # NMS both (CPU)
        t0 = time.monotonic()
        dets_a = postprocess_cpu(output_a_cpu, conf_thresh=0.25, iou_thresh=0.45,
                                 scale_info=scale_a, classes=vehicle_classes)
        dets_b = postprocess_cpu(output_b_cpu, conf_thresh=0.25, iou_thresh=0.45,
                                 scale_info=scale_b, classes=vehicle_classes)
        timings["nms"].append(time.monotonic() - t0)

        # Track both (CPU)
        t0 = time.monotonic()
        tracks_a = update_tracker(tracker_a, dets_a, (scale_a[3], scale_a[4], 3))
        tracks_b = update_tracker(tracker_b, dets_b, (scale_b[3], scale_b[4], 3))
        timings["track"].append(time.monotonic() - t0)

        if len(tracks_a) > 0:
            for tid in tracks_a[:, 4].astype(int):
                ids_a.add(tid)
        if len(tracks_b) > 0:
            for tid in tracks_b[:, 4].astype(int):
                ids_b.add(tid)

        timings["total"].append(time.monotonic() - t_total)
        infer_count += 1

    return infer_count, (len(ids_a), len(ids_b)), timings


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_timings(timings, infer_count):
    """Print per-stage latency breakdown."""
    for stage in ["decode", "preprocess", "infer", "nms", "track", "total"]:
        vals = timings[stage]
        if not vals:
            continue
        avg = sum(vals) / len(vals) * 1000
        p50 = sorted(vals)[len(vals) // 2] * 1000
        p99 = sorted(vals)[int(len(vals) * 0.99)] * 1000
        print(f"  {stage:12s}: avg {avg:6.2f} ms  p50 {p50:6.2f} ms  p99 {p99:6.2f} ms")

    total_time = sum(timings["total"])
    fps = infer_count / total_time if total_time > 0 else 0
    print(f"  {'FPS':12s}: {fps:.1f}")
    print(f"  {'Budget':12s}: {'OK' if fps >= 30 else 'FAIL'} (need 30 fps per channel)")


def print_gpu_memory():
    """Print current GPU memory usage."""
    free, total = cp.cuda.Device(0).mem_info
    used = (total - free) / 1e6
    print(f"  GPU memory: {used:.0f} MB used / {total / 1e6:.0f} MB total")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sources = sys.argv[1:]
    if not sources:
        print(f"Usage: {sys.argv[0]} <video1.mp4> [video2.mp4]")
        sys.exit(1)

    ENGINE = "/app/models/yolov8s_direct.engine"

    print("=== Experiment 12: GPU-Resident Pipeline ===")
    print(f"Engine: {ENGINE} (direct TRT, pip TRT {trt.__version__})")
    print(f"Decode: NVDEC via PyNvVideoCodec")
    print(f"Preprocess: CuPy GPU (NV12->RGB + resize + normalize)")
    print(f"Tracking: standalone BoT-SORT (no ReID, no GMC)")
    print(f"Sources: {sources}")
    print()

    # Test 0: NVDEC decode only
    print("[TEST 0] NVDEC decode only (300 frames)")
    n, timings = test_decode_only(sources[0], 300)
    avg_ms = sum(timings) / len(timings) * 1000
    fps = n / sum(timings)
    print(f"  Frames: {n}, FPS: {fps:.1f}, avg: {avg_ms:.2f} ms/frame")
    print()

    # Test 1: Single channel full pipeline
    print("[TEST 1] Single channel — GPU-resident pipeline (300 inferences)")
    print_gpu_memory()
    n, ids, timings = test_single_channel(ENGINE, sources[0], 300)
    print(f"  Inferences: {n}, Unique IDs: {ids}")
    print_timings(timings, n)
    print_gpu_memory()
    print()

    # Test 2: Two channels
    if len(sources) >= 2:
        src_label = f"{sources[0]} + {sources[1]}"
    else:
        src_label = f"{sources[0]} x 2"
    print(f"[TEST 2] Two channels — sequential infer (150 inferences each)")
    print(f"  Sources: {src_label}")
    n, ids_tuple, timings = test_two_channel(ENGINE, sources, 150)
    print(f"  Inferences: {n} per channel, IDs: ch0={ids_tuple[0]}, ch1={ids_tuple[1]}")
    print_timings(timings, n)
    print_gpu_memory()
    print()

    # Comparison
    print("=== Comparison vs exp11 (Ultralytics wrapper) ===")
    print("  exp11 single:  62.9 fps, 15.9 ms/frame")
    print("  exp11 dual:    37.5 fps, 26.6 ms/frame")
    print("  Target: >= 30 fps per channel (60 fps total for 2 channels)")
    print()

    print("=== DONE ===")


if __name__ == "__main__":
    main()
