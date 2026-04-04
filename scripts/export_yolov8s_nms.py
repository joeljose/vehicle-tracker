#!/usr/bin/env python3
"""Export YOLOv8s ONNX with EfficientNMS_TRT plugin for DeepStream nvinfer.

Transforms the raw YOLOv8s output (1, 84, 8400) into 4 standard outputs
that nvinfer can consume with cluster-mode=4:
  - num_detections (1, 1)
  - detection_boxes (1, max_det, 4) in xyxy format
  - detection_scores (1, max_det)
  - detection_classes (1, max_det)

Usage (inside DeepStream container):
    python scripts/export_yolov8s_nms.py

Requires: onnx, onnx_graphsurgeon (ships with TensorRT in DS container)
"""

import numpy as np
import onnx
import onnx_graphsurgeon as gs

INPUT_ONNX = "/app/models/yolov8s.onnx"
OUTPUT_ONNX = "/app/models/yolov8s_nms.onnx"

# NMS parameters (match Custom backend)
SCORE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 100


def main():
    print(f"Loading {INPUT_ONNX}")
    graph = gs.import_onnx(onnx.load(INPUT_ONNX))

    # Find the raw output node — shape (1, 84, 8400)
    raw_output = graph.outputs[0]
    print(f"Raw output: {raw_output.name}, shape={raw_output.shape}")

    # Transpose: (1, 84, 8400) → (1, 8400, 84)
    transposed = gs.Variable("transposed", dtype=np.float32)
    transpose_node = gs.Node(
        op="Transpose",
        attrs={"perm": [0, 2, 1]},
        inputs=[raw_output],
        outputs=[transposed],
    )
    graph.nodes.append(transpose_node)

    # Split into boxes (1, 8400, 4) and scores (1, 8400, 80)
    boxes_raw = gs.Variable("boxes_raw", dtype=np.float32)
    scores = gs.Variable("scores", dtype=np.float32)
    split_node = gs.Node(
        op="Split",
        attrs={"axis": 2},
        inputs=[transposed, gs.Constant("split_sizes", values=np.array([4, 80], dtype=np.int64))],
        outputs=[boxes_raw, scores],
    )
    graph.nodes.append(split_node)

    # Convert cxcywh → xyxy
    # cx, cy, w, h → x1=cx-w/2, y1=cy-h/2, x2=cx+w/2, y2=cy+h/2
    half = gs.Constant("half", values=np.array([0.5], dtype=np.float32))

    # Split boxes into cx, cy, w, h — each (1, 8400, 1)
    cx = gs.Variable("cx", dtype=np.float32)
    cy = gs.Variable("cy", dtype=np.float32)
    w = gs.Variable("w", dtype=np.float32)
    h = gs.Variable("h", dtype=np.float32)
    split_box = gs.Node(
        op="Split",
        attrs={"axis": 2},
        inputs=[boxes_raw, gs.Constant("box_split", values=np.array([1, 1, 1, 1], dtype=np.int64))],
        outputs=[cx, cy, w, h],
    )
    graph.nodes.append(split_box)

    # half_w = w * 0.5, half_h = h * 0.5
    half_w = gs.Variable("half_w", dtype=np.float32)
    half_h = gs.Variable("half_h", dtype=np.float32)
    graph.nodes.append(gs.Node(op="Mul", inputs=[w, half], outputs=[half_w]))
    graph.nodes.append(gs.Node(op="Mul", inputs=[h, half], outputs=[half_h]))

    # x1 = cx - half_w, y1 = cy - half_h, x2 = cx + half_w, y2 = cy + half_h
    x1 = gs.Variable("x1", dtype=np.float32)
    y1 = gs.Variable("y1", dtype=np.float32)
    x2 = gs.Variable("x2", dtype=np.float32)
    y2 = gs.Variable("y2", dtype=np.float32)
    graph.nodes.append(gs.Node(op="Sub", inputs=[cx, half_w], outputs=[x1]))
    graph.nodes.append(gs.Node(op="Sub", inputs=[cy, half_h], outputs=[y1]))
    graph.nodes.append(gs.Node(op="Add", inputs=[cx, half_w], outputs=[x2]))
    graph.nodes.append(gs.Node(op="Add", inputs=[cy, half_h], outputs=[y2]))

    # Concat to (1, 8400, 4) in xyxy order
    boxes_xyxy = gs.Variable("boxes_xyxy", dtype=np.float32)
    graph.nodes.append(gs.Node(
        op="Concat",
        attrs={"axis": 2},
        inputs=[x1, y1, x2, y2],
        outputs=[boxes_xyxy],
    ))

    # EfficientNMS_TRT plugin
    num_detections = gs.Variable("num_detections", dtype=np.int32, shape=[1, 1])
    detection_boxes = gs.Variable("detection_boxes", dtype=np.float32, shape=[1, MAX_DETECTIONS, 4])
    detection_scores = gs.Variable("detection_scores", dtype=np.float32, shape=[1, MAX_DETECTIONS])
    detection_classes = gs.Variable("detection_classes", dtype=np.int32, shape=[1, MAX_DETECTIONS])

    nms_node = gs.Node(
        op="EfficientNMS_TRT",
        attrs={
            "score_threshold": SCORE_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "max_output_boxes": MAX_DETECTIONS,
            "background_class": -1,
            "score_activation": False,
            "box_coding": 0,  # 0 = xyxy (already converted)
            "plugin_version": "1",
        },
        inputs=[boxes_xyxy, scores],
        outputs=[num_detections, detection_boxes, detection_scores, detection_classes],
    )
    graph.nodes.append(nms_node)

    # Set graph outputs to NMS outputs
    graph.outputs = [num_detections, detection_boxes, detection_scores, detection_classes]

    # Clean up unused nodes
    graph.cleanup().toposort()

    # Save
    model = gs.export_onnx(graph)
    onnx.save(model, OUTPUT_ONNX)
    print(f"Saved {OUTPUT_ONNX}")
    print(f"  Outputs: {[o.name for o in model.graph.output]}")
    print(f"  NMS: score_thresh={SCORE_THRESHOLD}, iou_thresh={IOU_THRESHOLD}, max_det={MAX_DETECTIONS}")


if __name__ == "__main__":
    main()
