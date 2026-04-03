"""Build TRT engine from ONNX using pip TensorRT Python API.

Run at container startup if the engine doesn't exist.
Takes ~150s on RTX 4050.
"""

import hashlib
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_ONNX_PATH = "/app/models/yolov8s.onnx"
DEFAULT_ENGINE_PATH = "/app/models/yolov8s_direct.engine"


def _file_hash(path: str) -> str:
    """MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_engine(
    onnx_path: str = DEFAULT_ONNX_PATH,
    engine_path: str = DEFAULT_ENGINE_PATH,
    fp16: bool = True,
) -> str:
    """Build TRT engine if missing or ONNX has changed.

    Returns the engine path.
    """
    hash_path = engine_path + ".onnx_hash"

    needs_build = False
    if not os.path.exists(engine_path):
        logger.info("Engine not found at %s — building", engine_path)
        needs_build = True
    elif os.path.exists(hash_path):
        stored_hash = open(hash_path).read().strip()
        current_hash = _file_hash(onnx_path)
        if stored_hash != current_hash:
            logger.info("ONNX changed — rebuilding engine")
            needs_build = True

    if needs_build:
        build_engine(onnx_path, engine_path, fp16=fp16)
        # Store hash for change detection
        with open(hash_path, "w") as f:
            f.write(_file_hash(onnx_path))

    return engine_path


def build_engine(
    onnx_path: str, engine_path: str, fp16: bool = True,
) -> None:
    """Build TRT engine from ONNX."""
    import tensorrt as trt

    logger.info(
        "Building TRT engine from %s (FP16=%s, TRT %s)...",
        onnx_path, fp16, trt.__version__,
    )

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH),
    )
    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error("ONNX parse error: %s", parser.get_error(i))
            raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Failed to build TRT engine")

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    size_mb = os.path.getsize(engine_path) / 1e6
    logger.info("Engine saved: %s (%.1f MB)", engine_path, size_mb)
