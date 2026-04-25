"""DeepStream pipeline configuration."""

# Paths inside the container
PGIE_CONFIG = "/app/config/deepstream/pgie_config.yml"
TRACKER_CONFIG = "/app/config/deepstream/tracker_config.yml"
TRACKER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
LABELS_FILE = "/app/config/deepstream/labels.txt"

# The ONNX model path is configured in pgie_config.yml — nvinfer reads it
# directly. Python doesn't load the model itself, so there's no Python-side
# constant for it. (Pre-M8-P1.5 v2 there was an ONNX_MODEL constant pointing
# at yolov8s_nms.onnx; both that constant and the export_yolov8s_nms.py
# script were unused after the migration to the fine-tuned single-class
# student.)

# Pipeline defaults
DEFAULT_STREAMMUX_WIDTH = 1920
DEFAULT_STREAMMUX_HEIGHT = 1080
DEFAULT_BATCH_PUSH_TIMEOUT = 33000  # microseconds


def load_labels(path: str = LABELS_FILE) -> dict[int, str]:
    """Load class labels from file. Returns {class_id: label_name}."""
    labels = {}
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                labels[i] = line
    return labels
