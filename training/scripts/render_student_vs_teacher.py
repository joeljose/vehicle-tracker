#!/usr/bin/env python3
"""Render visual overlays comparing the student vs the teacher on the
comparison set frames.

For each sampled frame, draws:
  - Matched boxes (student and teacher agree at IoU>=0.5):    CYAN
  - Teacher-only boxes (student missed):                      GREEN
  - Student-only boxes (student extras not in teacher):       MAGENTA

The color legend is baked into the top-left of each output image.
Fifteen frames per site, with the three known-fragmentation frames at
741_73 forced into the sample so we can always check the hardest cases.

Output: data/comparison/student_eval/{slug}/{stem}.jpg
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from _common import REPO_ROOT, log  # noqa: E402

COMPARISON_ROOT = REPO_ROOT / "data" / "comparison"
PLAN_PATH = COMPARISON_ROOT / "plan.json"
PREDICTIONS_ROOT = COMPARISON_ROOT / "predictions"
IMAGES_ROOT = COMPARISON_ROOT / "images"
OUT_ROOT = COMPARISON_ROOT / "student_eval"

TEACHER = "rfdetr_m"
STUDENT = "student_v1_cont"

# Colors in BGR (OpenCV convention)
COLOR_MATCHED = (200, 200, 0)    # cyan-ish — student+teacher agree
COLOR_TEACHER_ONLY = (60, 200, 60)   # green — teacher had it, student missed
COLOR_STUDENT_ONLY = (200, 0, 200)   # magenta — student extra, teacher didn't have

IOU_THRESHOLD = 0.5
PER_SITE_SAMPLE = 15
FORCED_741_73 = ["frame_004283", "frame_011477", "frame_005976"]
SEED = 0


def load_pred(teacher: str, slug: str, stem: str) -> dict | None:
    path = PREDICTIONS_ROOT / teacher / slug / f"{stem}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def xywhn_to_xyxy_pixels(xywhn: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    if len(xywhn) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    h, w = hw
    cx = xywhn[:, 0] * w
    cy = xywhn[:, 1] * h
    bw = xywhn[:, 2] * w
    bh = xywhn[:, 3] * h
    return np.stack(
        [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1
    ).astype(np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    aa = a[:, None, :]
    bb = b[None, :, :]
    inter_x1 = np.maximum(aa[..., 0], bb[..., 0])
    inter_y1 = np.maximum(aa[..., 1], bb[..., 1])
    inter_x2 = np.minimum(aa[..., 2], bb[..., 2])
    inter_y2 = np.minimum(aa[..., 3], bb[..., 3])
    iw = np.clip(inter_x2 - inter_x1, 0, None)
    ih = np.clip(inter_y2 - inter_y1, 0, None)
    inter = iw * ih
    area_a = (aa[..., 2] - aa[..., 0]) * (aa[..., 3] - aa[..., 1])
    area_b = (bb[..., 2] - bb[..., 0]) * (bb[..., 3] - bb[..., 1])
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-9)


def greedy_match(
    teacher_boxes: np.ndarray, student_boxes: np.ndarray
) -> tuple[list[int], list[int], set[int], set[int]]:
    """Return (matched_teacher_idxs, matched_student_idxs, unmatched_teacher, unmatched_student)."""
    if len(teacher_boxes) == 0 or len(student_boxes) == 0:
        return [], [], set(range(len(teacher_boxes))), set(range(len(student_boxes)))
    iou = iou_matrix(teacher_boxes, student_boxes)
    matched_t: list[int] = []
    matched_s: list[int] = []
    used_t: set[int] = set()
    used_s: set[int] = set()
    # Greedy: pick highest IoU pair, mark both used, repeat
    flat = [
        (float(iou[i, j]), i, j)
        for i in range(iou.shape[0])
        for j in range(iou.shape[1])
    ]
    flat.sort(reverse=True)
    for score, i, j in flat:
        if score < IOU_THRESHOLD:
            break
        if i in used_t or j in used_s:
            continue
        matched_t.append(i)
        matched_s.append(j)
        used_t.add(i)
        used_s.add(j)
    un_t = set(range(len(teacher_boxes))) - used_t
    un_s = set(range(len(student_boxes))) - used_s
    return matched_t, matched_s, un_t, un_s


def draw_box(img: np.ndarray, xyxy: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def draw_legend(img: np.ndarray, counts: dict[str, int]) -> None:
    """Draw a legend strip in the top-left corner."""
    lines = [
        (f"MATCHED (T=S): {counts['matched']}", COLOR_MATCHED),
        (f"TEACHER only (missed): {counts['teacher_only']}", COLOR_TEACHER_ONLY),
        (f"STUDENT only (extra): {counts['student_only']}", COLOR_STUDENT_ONLY),
    ]
    y = 30
    for text, color in lines:
        # white bg for text readability
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += 26


def render_one_frame(slug: str, stem: str) -> bool:
    image_path = IMAGES_ROOT / slug / f"{stem}.jpg"
    if not image_path.exists():
        log(f"  [warn] missing image: {image_path}")
        return False
    img = cv2.imread(str(image_path))
    if img is None:
        log(f"  [warn] could not read: {image_path}")
        return False
    h, w = img.shape[:2]

    t_pred = load_pred(TEACHER, slug, stem)
    s_pred = load_pred(STUDENT, slug, stem)
    if t_pred is None or s_pred is None:
        log(f"  [warn] missing prediction for {slug}/{stem}")
        return False

    t_boxes = xywhn_to_xyxy_pixels(np.asarray(t_pred["boxes_xywhn"]), (h, w))
    s_boxes = xywhn_to_xyxy_pixels(np.asarray(s_pred["boxes_xywhn"]), (h, w))

    matched_t, matched_s, un_t, un_s = greedy_match(t_boxes, s_boxes)

    # Draw in order so agreements are most prominent
    for i in matched_t:
        draw_box(img, t_boxes[i], COLOR_MATCHED, thickness=2)
    for i in matched_s:
        # Also outline the student match (same cyan). If the boxes are
        # perfectly coincident the second draw is a no-op; if they're
        # slightly different you'll see two thin cyan boxes — which is
        # informative about how tight the agreement actually is.
        draw_box(img, s_boxes[i], COLOR_MATCHED, thickness=1)
    for i in un_t:
        draw_box(img, t_boxes[i], COLOR_TEACHER_ONLY, thickness=2)
    for i in un_s:
        draw_box(img, s_boxes[i], COLOR_STUDENT_ONLY, thickness=2)

    counts = {
        "matched": len(matched_t),
        "teacher_only": len(un_t),
        "student_only": len(un_s),
    }
    draw_legend(img, counts)

    out_path = OUT_ROOT / slug / f"{stem}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


def sample_frames(plan_frames: list[dict]) -> list[tuple[str, str]]:
    """Pick 15 per site; force the 3 known fragmentation frames at 741_73."""
    rng = random.Random(SEED)
    by_site: dict[str, list[str]] = {}
    for entry in plan_frames:
        by_site.setdefault(entry["site"], []).append(entry["stem"])

    out: list[tuple[str, str]] = []
    for site, stems in by_site.items():
        stems = sorted(stems)
        if site == "junction_741_73":
            forced = [s for s in FORCED_741_73 if s in stems]
            remaining = [s for s in stems if s not in forced]
            n = max(0, PER_SITE_SAMPLE - len(forced))
            picked = forced + rng.sample(remaining, min(n, len(remaining)))
        else:
            picked = rng.sample(stems, min(PER_SITE_SAMPLE, len(stems)))
        for stem in picked:
            out.append((site, stem))
    return out


def main() -> None:
    if not PLAN_PATH.exists():
        log(f"ERROR: {PLAN_PATH} missing")
        sys.exit(1)
    plan = json.loads(PLAN_PATH.read_text())
    frames = sample_frames(plan["frames"])
    log(f"Rendering {len(frames)} overlay frames to {OUT_ROOT}")

    n_ok = 0
    for site, stem in frames:
        if render_one_frame(site, stem):
            n_ok += 1
    log(f"Wrote {n_ok}/{len(frames)} overlays")
    log(f"  Output: {OUT_ROOT}")


if __name__ == "__main__":
    main()
