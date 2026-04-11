#!/usr/bin/env python3
"""Export the M8-P1.5 Label Studio annotations into YOLO-format ground truth.

Pulls every task from the project, walks each task's annotations (NOT
predictions — annotations are the human-corrected labels), and writes a
YOLO-format `.txt` file to:

    data/benchmark/labels_gt/{slug}/{stem}.txt

`slug` and `stem` come from the task's `data` payload that ls_setup.py
embedded at import time.

Tasks with no annotation are skipped (and reported), so it's safe to run
mid-session — you'll get partial labels for whatever you've reviewed so far.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests

from _common import REPO_ROOT, log

LS_URL = os.environ.get("LS_URL", "http://host.docker.internal:8080")
LS_TOKEN = os.environ.get("LS_TOKEN", "vt_p1_5_token_2026")

PROJECT_TITLE = "M8-P1.5 Vehicle Benchmark"
LABELS_GT_ROOT = REPO_ROOT / "data" / "benchmark" / "labels_gt"

# Single-class collapse: "vehicle" is canonical.
# Any legacy labels (car/truck/bus/motorcycle) from earlier runs are
# remapped to vehicle so an unmigrated annotation still exports cleanly.
CLASS_NAME_TO_ID = {
    "vehicle": 0,
    "car": 0,
    "truck": 0,
    "bus": 0,
    "motorcycle": 0,
}


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Token {LS_TOKEN}"}


def find_project_id() -> int:
    r = requests.get(f"{LS_URL}/api/projects", headers=auth_headers(), timeout=10)
    r.raise_for_status()
    payload = r.json()
    items = payload.get("results", payload) if isinstance(payload, dict) else payload
    for proj in items:
        if proj.get("title") == PROJECT_TITLE:
            return int(proj["id"])
    raise SystemExit(f"Project '{PROJECT_TITLE}' not found")


def fetch_all_tasks(project_id: int) -> list[dict]:
    """Returns full task objects (with annotations)."""
    tasks: list[dict] = []
    page = 1
    while True:
        r = requests.get(
            f"{LS_URL}/api/projects/{project_id}/tasks",
            headers=auth_headers(),
            params={"page": page, "page_size": 200},
            timeout=30,
        )
        r.raise_for_status()
        items = r.json()
        if isinstance(items, dict):
            items = items.get("tasks") or items.get("results") or []
        if not items:
            break
        tasks.extend(items)
        if len(items) < 200:
            break
        page += 1
    return tasks


def fetch_task_detail(task_id: int) -> dict:
    r = requests.get(
        f"{LS_URL}/api/tasks/{task_id}",
        headers=auth_headers(),
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def annotation_to_yolo_lines(annotation: dict) -> list[str]:
    """Walk the annotation's `result` list and convert each rectangle to YOLO."""
    lines: list[str] = []
    for item in annotation.get("result", []):
        if item.get("type") != "rectanglelabels":
            continue
        v = item.get("value", {})
        labels = v.get("rectanglelabels", [])
        if not labels:
            continue
        cls_name = labels[0]
        cid = CLASS_NAME_TO_ID.get(cls_name)
        if cid is None:
            continue
        # LS gives x,y,width,height as percentages of image dimensions
        x_pct = float(v.get("x", 0))
        y_pct = float(v.get("y", 0))
        w_pct = float(v.get("width", 0))
        h_pct = float(v.get("height", 0))
        # Convert to YOLO normalized cx, cy, w, h
        cx = (x_pct + w_pct / 2) / 100.0
        cy = (y_pct + h_pct / 2) / 100.0
        bw = w_pct / 100.0
        bh = h_pct / 100.0
        # Clamp to [0, 1] in case the user dragged outside the image
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def main() -> None:
    project_id = find_project_id()
    log(f"Using project id={project_id}")

    tasks = fetch_all_tasks(project_id)
    log(f"Fetched {len(tasks)} tasks")

    annotated = 0
    skipped = 0
    written = 0
    empty_ann = 0
    for task in tasks:
        # The list endpoint sometimes returns annotations inline; fetch the
        # detail endpoint as a fallback to be safe.
        annotations = task.get("annotations") or []
        if not annotations:
            detail = fetch_task_detail(int(task["id"]))
            annotations = detail.get("annotations") or []
            task = detail
        if not annotations:
            skipped += 1
            continue
        annotated += 1

        data = task.get("data", {})
        slug = data.get("site")
        stem = data.get("stem")
        if not slug or not stem:
            log(f"  WARN task {task.get('id')} missing site/stem in data, skipping")
            continue

        # Use the most recent submitted annotation
        ann = sorted(
            annotations,
            key=lambda a: a.get("updated_at", a.get("created_at", "")),
            reverse=True,
        )[0]
        lines = annotation_to_yolo_lines(ann)

        out_path = LABELS_GT_ROOT / slug / f"{stem}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        written += 1
        if not lines:
            empty_ann += 1

    log(f"Wrote {written} label files (annotated tasks: {annotated})")
    log(f"  empty (zero-box) annotations: {empty_ann}")
    log(f"  unlabeled tasks skipped:      {skipped}")
    log(f"Output: {LABELS_GT_ROOT}")


if __name__ == "__main__":
    main()
