"""Direct TensorRT inference + CPU-side NMS.

Loads a TRT engine, runs inference with GPU tensors in/out,
then performs NMS on CPU. No Ultralytics wrapper.

M8-P1.5 v2: the deployed model is a single-class fine-tuned YOLOv8s
(project class 0 = vehicle). Output tensor shape is (1, 5, 8400) rather
than the COCO-pretrained (1, 84, 8400): 4 bbox coords + 1 class score
per anchor. All detections are vehicles by construction, so we no
longer filter on COCO vehicle ids 2/3/5/7.
"""

import logging

import cupy as cp
import numpy as np
import tensorrt as trt

from backend.pipeline.custom.preprocess import ScaleInfo

logger = logging.getLogger(__name__)

# Single-class model: the only project class id is 0 = "vehicle".
# Kept as a module-level constant for anything downstream that still
# wants a human-readable name.
PROJECT_CLASS_NAMES = {0: "vehicle"}


class TRTDetector:
    """Direct TensorRT inference with CPU-side NMS."""

    def __init__(self, engine_path: str, conf_thresh: float = 0.25):
        self._conf_thresh = conf_thresh
        self._iou_thresh = 0.45

        # Load engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to load TRT engine: {engine_path}")

        self._context = self._engine.create_execution_context()

        # Binding info
        self._input_name = self._engine.get_tensor_name(0)
        self._output_name = self._engine.get_tensor_name(1)
        self._output_shape = tuple(self._engine.get_tensor_shape(self._output_name))

        # Pre-allocate GPU output buffer
        self._output_buffer = cp.empty(self._output_shape, dtype=cp.float32)

        # CUDA stream for async execution
        self._stream = cp.cuda.Stream()

        logger.info(
            "TRTDetector: %s loaded (input=%s, output=%s)",
            engine_path, self._input_name, self._output_name,
        )

    def detect(
        self, input_tensor: cp.ndarray, scale_info: ScaleInfo,
    ) -> np.ndarray:
        """Run inference + NMS.

        Args:
            input_tensor: (1, 3, 640, 640) CuPy float32 on GPU.
            scale_info: For mapping boxes to original frame coords.

        Returns:
            (N, 6) numpy array: [x1, y1, x2, y2, conf, cls_id]
        """
        input_tensor = cp.ascontiguousarray(input_tensor, dtype=cp.float32)

        self._context.set_tensor_address(self._input_name, input_tensor.data.ptr)
        self._context.set_tensor_address(self._output_name, self._output_buffer.data.ptr)

        self._context.execute_async_v3(self._stream.ptr)
        self._stream.synchronize()

        # Download output to CPU for NMS
        output_cpu = cp.asnumpy(self._output_buffer)
        return self._postprocess(output_cpu, scale_info)

    def set_confidence_threshold(self, threshold: float) -> None:
        self._conf_thresh = threshold

    def _postprocess(self, output: np.ndarray, scale_info: ScaleInfo) -> np.ndarray:
        """NMS on CPU.

        Output tensor shape is (1, 5, 8400) for our 1-class fine-tuned
        model: 4 bbox coords + 1 class score per anchor. The transpose
        below yields (8400, 5); we slice the first 4 as cxcywh and the
        last 1 as the confidence directly — no argmax needed because
        there is only one class.
        """
        preds = output[0].T  # (8400, 5)

        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:]        # (8400, 1)

        # Single-class: the "max score" is just the one score per anchor,
        # and every detection has class id 0.
        max_scores = class_scores[:, 0]
        class_ids = np.zeros(len(preds), dtype=np.int32)

        # Filter by confidence only. No vehicle-class filter — everything
        # the model emits is already a vehicle by construction.
        mask = max_scores > self._conf_thresh

        boxes_cxcywh = boxes_cxcywh[mask]
        scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(scores) == 0:
            return np.empty((0, 6), dtype=np.float32)

        # cxcywh → xyxy
        boxes = np.empty_like(boxes_cxcywh)
        boxes[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        boxes[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        boxes[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        boxes[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

        # NMS
        keep = _nms(boxes, scores, self._iou_thresh)
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # Scale back to original coordinates
        boxes[:, 0] = (boxes[:, 0] - scale_info.pad_left) / scale_info.scale
        boxes[:, 1] = (boxes[:, 1] - scale_info.pad_top) / scale_info.scale
        boxes[:, 2] = (boxes[:, 2] - scale_info.pad_left) / scale_info.scale
        boxes[:, 3] = (boxes[:, 3] - scale_info.pad_top) / scale_info.scale

        # Clip to frame
        boxes[:, 0] = np.clip(boxes[:, 0], 0, scale_info.orig_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, scale_info.orig_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, scale_info.orig_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, scale_info.orig_h)

        return np.column_stack([boxes, scores, class_ids.astype(np.float32)])


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Greedy NMS."""
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
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        mask = iou <= iou_thresh
        order = order[1:][mask]

    return keep
