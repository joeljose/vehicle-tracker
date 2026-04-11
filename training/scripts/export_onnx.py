#!/usr/bin/env python3
"""Export a trained YOLOv8 checkpoint to ONNX.

The ONNX file is the portable intermediate that both backends consume:
  - Custom backend: backend/pipeline/custom/engine_builder.py reads it at
    startup and builds a TensorRT engine using its own TRT 10.16.0.72.
  - DeepStream backend: nvinfer reads `onnx-file` from pgie_config.yml at
    first launch and auto-builds the engine using DS-bundled TRT 10.9.0.34.

The training container itself does NOT need a working TRT install —
ultralytics produces ONNX via torch.onnx.export, with onnx + onnxslim
as the only required support packages.

Usage:
    python3 scripts/export_onnx.py --run yolov8s_rfdetr_v1_cont

Result:
    /models/yolov8s_rfdetr_v1_cont.onnx  (copied from the run dir)
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import REPO_ROOT, log  # noqa: E402

RUNS_ROOT = REPO_ROOT / "runs" / "detect"
# /models/ is bind-mounted into both backend containers at /app/models/
MODELS_ROOT = Path("/models")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run name under runs/detect/")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument(
        "--out-name",
        default=None,
        help="Filename for the copied ONNX in /models (default: <run>.onnx)",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Skip the copy to /models/ — just leave the ONNX alongside best.pt",
    )
    args = parser.parse_args()

    weights_pt = RUNS_ROOT / args.run / "weights" / "best.pt"
    if not weights_pt.exists():
        log(f"ERROR: {weights_pt} not found")
        sys.exit(1)

    log(f"Exporting {weights_pt} to ONNX...")
    log(f"  imgsz:    {args.imgsz}")
    log(f"  opset:    {args.opset}")
    log(f"  half:     False (FP32 ONNX; each backend quantizes to FP16 at engine build)")
    log(f"  dynamic:  False (static batch=1, 3x{args.imgsz}x{args.imgsz})")
    log(f"  simplify: True (onnxslim graph optimization)")

    from ultralytics import YOLO

    model = YOLO(str(weights_pt))

    t0 = time.time()
    onnx_path_str = model.export(
        format="onnx",
        imgsz=args.imgsz,
        half=False,          # keep ONNX FP32; let each backend pick its precision at build
        dynamic=False,       # static shape for deploy simplicity
        simplify=True,       # onnxslim graph fold/fuse
        opset=args.opset,
        batch=1,
        device="cpu",        # ONNX export doesn't need GPU — faster + more reliable
    )
    elapsed = time.time() - t0
    log(f"Export complete in {elapsed:.0f}s")

    onnx_path = Path(onnx_path_str)
    if not onnx_path.exists():
        log(f"ERROR: export returned {onnx_path_str} but file doesn't exist")
        sys.exit(1)
    log(f"ONNX file: {onnx_path}")
    log(f"  size: {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")

    if not args.no_copy:
        MODELS_ROOT.mkdir(parents=True, exist_ok=True)
        out_name = args.out_name or f"{args.run}.onnx"
        dst = MODELS_ROOT / out_name
        log(f"Copying to {dst}...")
        shutil.copy2(onnx_path, dst)
        log(f"  copied: {dst} ({dst.stat().st_size / 1024 / 1024:.1f} MB)")

    log("")
    log("Next steps:")
    log("  Custom backend: restart — engine_builder.py will auto-build on startup")
    log("  DeepStream backend: restart — nvinfer will auto-build on startup")
    log(f"  Both will write the resulting .engine next to the .onnx in /models/")


if __name__ == "__main__":
    main()
