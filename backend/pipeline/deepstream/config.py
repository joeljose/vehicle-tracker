"""DeepStream pipeline configuration."""

from pathlib import Path

# Paths inside the container
PGIE_CONFIG = "/app/config/deepstream/pgie_config.yml"
TRACKER_CONFIG = "/app/config/deepstream/tracker_config.yml"
TRACKER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
LABELS_FILE = "/app/config/deepstream/labels.txt"

# Model paths (inside the container)
MODEL_DIR = Path("/app/models")
ONNX_MODEL = MODEL_DIR / "yolov8s_nms.onnx"

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
