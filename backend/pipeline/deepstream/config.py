"""DeepStream pipeline configuration."""

from pathlib import Path

# Paths inside the container
PGIE_CONFIG = "/app/config/deepstream/pgie_config.yml"
LABELS_FILE = "/app/config/deepstream/labels.txt"

# Model paths (inside the DS container)
MODEL_DIR = Path("/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector")
ONNX_MODEL = MODEL_DIR / "resnet18_trafficcamnet_pruned.onnx"

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
