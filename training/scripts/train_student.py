#!/usr/bin/env python3
"""Fine-tune YOLOv8s on RF-DETR auto-labels — M8-P1.5 v2.

Two modes:

  fresh     Start from COCO-pretrained yolov8s.pt weights. Standard
            fine-tuning with warmup and full LR schedule. This is the
            first training run.

  continue  Start from an existing run's last.pt checkpoint. Lower LR,
            no warmup (weights are near-converged). Writes to a NEW
            run directory named <from>_cont so the original run is
            preserved as a rollback point.

Hyperparameters default to the values in training/docs/design.md §8.
Any of them can be overridden via CLI flags.

Examples:

  # Fresh training from COCO pretrained weights
  python3 scripts/train_student.py

  # Continue an existing run for another 100 epochs
  python3 scripts/train_student.py --mode continue --from yolov8s_rfdetr_v1

  # Fresh training with a different name and smaller batch
  python3 scripts/train_student.py --name yolov8s_rfdetr_v2 --batch 4

  # Continuation with a custom learning rate override
  python3 scripts/train_student.py --mode continue --from yolov8s_rfdetr_v1 --lr0 0.0005
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from _common import REPO_ROOT, log  # noqa: E402

DATASET_YAML = REPO_ROOT / "data" / "dataset.yaml"
RUNS_ROOT = REPO_ROOT / "runs" / "detect"

DEFAULT_RUN_NAME = "yolov8s_rfdetr_v1"
DEFAULT_BASE_MODEL = "yolov8s.pt"  # ultralytics auto-download by name


# ---------------------------------------------------------------------------
#  Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All hyperparameters for a single training run.

    Defaults match training/docs/design.md §8 for fresh fine-tuning.
    For continuation training, build a TrainConfig via
    `make_continuation_config()` instead of constructing this directly.
    """

    # Starting point — either a pretrained-model name (COCO) or a .pt path.
    model: str = DEFAULT_BASE_MODEL
    # Target dataset and run name
    data: str = str(DATASET_YAML)
    name: str = DEFAULT_RUN_NAME
    # Core training knobs
    imgsz: int = 640
    batch: int = 8
    epochs: int = 100
    patience: int = 20
    workers: int = 4
    device: str = "0"
    # Optimizer + schedule
    optimizer: str = "SGD"
    lr0: float = 0.01
    lrf: float = 0.01         # final LR multiplier (cosine decay to lr0 * lrf)
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    # Augmentation (design doc §8: no flips for fixed camera angle)
    mosaic: float = 1.0
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    fliplr: float = 0.0
    flipud: float = 0.0
    scale: float = 0.5
    translate: float = 0.1
    # Precision / logging
    amp: bool = True
    plots: bool = True
    verbose: bool = True
    # Misc
    pretrained: bool = True
    exist_ok: bool = False    # fail loud if the run name already exists

    def as_train_kwargs(self) -> dict[str, Any]:
        """Extract just the keyword arguments ultralytics `model.train()` takes.

        The `model` field is NOT a train() kwarg — it's passed to YOLO() at
        instantiation. Everything else goes to train().
        """
        skip = {"model"}
        return {f.name: getattr(self, f.name) for f in fields_of(self) if f.name not in skip}


def fields_of(dc_instance: Any):
    """Return the dataclass fields for an instance (avoids importing dataclasses.fields at the call site)."""
    from dataclasses import fields
    return fields(dc_instance)


# ---------------------------------------------------------------------------
#  Mode builders
# ---------------------------------------------------------------------------

def make_fresh_config() -> TrainConfig:
    """Fresh fine-tune from COCO pretrained weights."""
    return TrainConfig()


def make_continuation_config(from_run: str) -> TrainConfig:
    """Continue training from an existing run's last.pt.

    - Loads weights from runs/detect/<from_run>/weights/last.pt
    - Writes to a NEW run named <from_run>_cont (does not overwrite the source)
    - Uses a 10x lower lr0 (0.001 vs 0.01) because the weights are already
      near-converged and a fresh high-LR kick would destabilize them.
    - Skips warmup (warmup_epochs=0) for the same reason.
    - `pretrained=False` because we're loading weights explicitly via `model=`
      and don't want ultralytics trying to overlay anything.

    Everything else matches the fresh config (same dataset, same augmentation,
    same batch size, same epoch count).
    """
    last_pt = RUNS_ROOT / from_run / "weights" / "last.pt"
    if not last_pt.exists():
        raise SystemExit(
            f"ERROR: cannot continue — {last_pt} does not exist. "
            f"Either run a fresh training first, or check the --from name."
        )
    return TrainConfig(
        model=str(last_pt),
        name=f"{from_run}_cont",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=0,
        pretrained=False,
    )


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["fresh", "continue"],
        default="fresh",
        help="fresh = start from COCO weights; continue = resume from an existing run",
    )
    parser.add_argument(
        "--from",
        dest="from_run",
        default=DEFAULT_RUN_NAME,
        help="(continue mode) name of the source run under runs/detect/",
    )
    # Common overrides. All optional — TrainConfig defaults apply if omitted.
    parser.add_argument("--name", default=None, help="Override run name")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size")
    parser.add_argument("--patience", type=int, default=None, help="Override early-stop patience")
    parser.add_argument("--lr0", type=float, default=None, help="Override initial learning rate")
    parser.add_argument("--lrf", type=float, default=None, help="Override final LR multiplier")
    parser.add_argument("--workers", type=int, default=None, help="Override dataloader worker count")
    parser.add_argument("--device", default=None, help="Override device (0, 0,1, cpu)")
    return parser.parse_args()


def apply_cli_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    """Apply any non-None CLI overrides to a TrainConfig. Returns the mutated config."""
    override_fields = {
        "name", "epochs", "batch", "imgsz", "patience",
        "lr0", "lrf", "workers", "device",
    }
    for name in override_fields:
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    return cfg


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.mode == "fresh":
        cfg = make_fresh_config()
        log(f"Mode: fresh fine-tune from {cfg.model}")
    else:
        cfg = make_continuation_config(args.from_run)
        log(f"Mode: continuation from runs/detect/{args.from_run}/weights/last.pt")
        log(f"       → new run name: {cfg.name}")

    apply_cli_overrides(cfg, args)

    if not Path(cfg.data).exists():
        log(f"ERROR: dataset config not found: {cfg.data}")
        log("       Run scripts/split_dataset.py first to create it.")
        sys.exit(1)

    log("Full training config:")
    log(f"  model:       {cfg.model}")
    for f in fields_of(cfg):
        if f.name == "model":
            continue
        log(f"  {f.name + ':':<16} {getattr(cfg, f.name)}")

    # Heavy imports after CLI sanity so --help is fast
    from ultralytics import YOLO

    log(f"Loading weights from: {cfg.model}")
    model = YOLO(cfg.model)

    log("Starting training...")
    model.train(**cfg.as_train_kwargs())

    log(f"Training complete. Outputs in runs/detect/{cfg.name}/")
    log("  - weights/best.pt : highest val mAP50-95 checkpoint")
    log("  - weights/last.pt : most recent epoch weights")
    log("  - results.csv     : per-epoch metrics")
    log("  - results.png     : auto-generated loss/metric plots")


if __name__ == "__main__":
    main()
