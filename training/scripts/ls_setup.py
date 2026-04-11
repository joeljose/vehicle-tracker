#!/usr/bin/env python3
"""Provision a Label Studio project for M8-P1.5.

Supports two modes:

  --mode comparison  (default)
    Creates a project titled "M8-P1.5 Teacher Comparison" backed by the
    1000-frame comparison set from `data/comparison/`. For each task, posts
    ONE PREDICTION PER TEACHER with a distinct `model_version` string so the
    user can flip between them in the LS UI and subjectively pick a winner.
    Reads per-teacher predictions from
    `data/comparison/predictions/{teacher}/{slug}/{stem}.json`.

  --mode review  (Phase F, not yet active)
    Reserved for post-bake-off review of the winning teacher's full auto-label
    output. Stubbed; raises NotImplementedError for now.

Idempotent: re-running finds the existing project by title, skips tasks
that are already fully configured. Safe to re-run after a partial Phase C
(only the teachers whose predictions exist at runtime will be posted).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

from _common import REPO_ROOT, log

# Training container → Label Studio container (separate compose project,
# talks to the host loopback via host.docker.internal).
LS_URL = os.environ.get("LS_URL", "http://host.docker.internal:8080")
LS_TOKEN = os.environ.get("LS_TOKEN", "vt_p1_5_token_2026")

# ---- comparison mode constants -------------------------------------------
COMPARISON_PROJECT_TITLE = "M8-P1.5 Teacher Comparison"
COMPARISON_LABEL_CONFIG = """
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="label" toName="image">
    <Label value="vehicle" background="#3CC83C"/>
  </RectangleLabels>
</View>
""".strip()

COMPARISON_ROOT = REPO_ROOT / "data" / "comparison"
COMPARISON_PLAN = COMPARISON_ROOT / "plan.json"
COMPARISON_PREDICTIONS = COMPARISON_ROOT / "predictions"

# Per-teacher model_version strings (used in LS prediction dropdown)
TEACHERS = ["yolov8x", "yolo26x", "yolo12x", "rfdetr_m", "grounding_dino_t"]

# LS-side storage config for comparison images
COMPARISON_STORAGE_TITLE = "comparison_images"
# LOCAL_FILES_DOCUMENT_ROOT is /label-studio/files, the storage is a
# subdirectory of that (required by LS for security reasons).
COMPARISON_STORAGE_PATH = "/label-studio/files/comparison/images"


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Token {LS_TOKEN}"}


# ---- LS API helpers ------------------------------------------------------

def wait_for_ls(timeout: int = 90) -> None:
    log(f"Waiting for Label Studio at {LS_URL}...")
    deadline = time.time() + timeout
    last_err: str | None = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{LS_URL}/health", timeout=3)
            if r.status_code == 200:
                log("  Label Studio is ready.")
                return
            last_err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(2)
    raise SystemExit(f"Label Studio never came up at {LS_URL}: {last_err}")


def verify_token_auth() -> None:
    r = requests.get(f"{LS_URL}/api/projects", headers=auth_headers(), timeout=10)
    if r.ok:
        return
    raise SystemExit(
        f"Token auth failed (HTTP {r.status_code}). Run `make train-ls-up` to "
        f"re-init the legacy API token, then re-run this script.\n"
        f"Body: {r.text[:300]}"
    )


def find_or_create_project(title: str, label_config: str, description: str) -> int:
    r = requests.get(f"{LS_URL}/api/projects", headers=auth_headers(), timeout=10)
    r.raise_for_status()
    payload = r.json()
    items = payload.get("results", payload) if isinstance(payload, dict) else payload
    for proj in items:
        if proj.get("title") == title:
            log(f"  Reusing existing project id={proj['id']} ({title!r})")
            return int(proj["id"])

    log(f"  Creating new project: {title!r}")
    r = requests.post(
        f"{LS_URL}/api/projects",
        headers=auth_headers(),
        json={
            "title": title,
            "description": description,
            "label_config": label_config,
        },
        timeout=10,
    )
    r.raise_for_status()
    project_id = int(r.json()["id"])
    log(f"  Created project id={project_id}")
    return project_id


def ensure_local_files_storage(project_id: int, title: str, path: str) -> None:
    """Register + sync a local-files storage on `path`. Idempotent."""
    r = requests.get(
        f"{LS_URL}/api/storages/localfiles",
        headers=auth_headers(),
        params={"project": project_id},
        timeout=10,
    )
    r.raise_for_status()
    existing = r.json() or []
    for s in existing:
        if s.get("title") == title:
            log(f"  Local-files storage {title!r} already registered (id={s['id']}).")
            return

    log(f"  Registering local-files storage {title!r} at {path}...")
    r = requests.post(
        f"{LS_URL}/api/storages/localfiles",
        headers=auth_headers(),
        json={
            "title": title,
            "path": path,
            "regex_filter": ".*\\.jpg$",
            "use_blob_urls": True,
            "recursive_scan": True,
            "project": project_id,
        },
        timeout=10,
    )
    if not r.ok:
        log(f"  storage register FAILED: HTTP {r.status_code} body={r.text[:500]}")
    r.raise_for_status()
    storage_id = int(r.json()["id"])
    log(f"  Local-files storage registered (id={storage_id}).")

    log("  Syncing storage (indexes files)...")
    r = requests.post(
        f"{LS_URL}/api/storages/localfiles/{storage_id}/sync",
        headers=auth_headers(),
        timeout=120,
    )
    r.raise_for_status()
    sync_info = r.json()
    log(f"  Sync complete: status={sync_info.get('status')}, count={sync_info.get('last_sync_count')}")


def fetch_all_tasks(project_id: int) -> list[dict]:
    out: list[dict] = []
    page = 1
    while True:
        r = requests.get(
            f"{LS_URL}/api/projects/{project_id}/tasks",
            headers=auth_headers(),
            params={"page": page, "page_size": 500},
            timeout=30,
        )
        if r.status_code == 404:
            break
        r.raise_for_status()
        items = r.json()
        if isinstance(items, dict):
            items = items.get("tasks") or items.get("results") or []
        if not items:
            break
        out.extend(items)
        if len(items) < 500:
            break
        page += 1
    return out


def index_tasks_by_stem(tasks: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for task in tasks:
        url = task.get("data", {}).get("image", "")
        if not url:
            continue
        leaf = url.split("/")[-1]
        if leaf.endswith(".jpg"):
            out[leaf[:-4]] = task
    return out


def patch_task_metadata(task_id: int, image_url: str, slug: str, stem: str) -> None:
    r = requests.patch(
        f"{LS_URL}/api/tasks/{task_id}",
        headers=auth_headers(),
        json={"data": {"image": image_url, "site": slug, "stem": stem}},
        timeout=10,
    )
    if not r.ok:
        log(f"  PATCH task {task_id} failed: HTTP {r.status_code} {r.text[:200]}")
    r.raise_for_status()


def post_prediction(task_id: int, result: list[dict], model_version: str, score: float = 0.9) -> None:
    r = requests.post(
        f"{LS_URL}/api/predictions",
        headers=auth_headers(),
        json={
            "task": task_id,
            "model_version": model_version,
            "score": score,
            "result": result,
        },
        timeout=10,
    )
    if not r.ok:
        log(f"  POST prediction for task {task_id} ({model_version}) failed: HTTP {r.status_code} {r.text[:200]}")
    r.raise_for_status()


def fetch_task_predictions(task_id: int) -> list[dict]:
    """Return the predictions already attached to a task (used for dedup)."""
    r = requests.get(
        f"{LS_URL}/api/predictions",
        headers=auth_headers(),
        params={"task": task_id},
        timeout=10,
    )
    r.raise_for_status()
    return r.json() or []


# ---- comparison-mode specific --------------------------------------------

def comparison_prediction_path(teacher: str, slug: str, stem: str) -> Path:
    return COMPARISON_PREDICTIONS / teacher / slug / f"{stem}.json"


def load_comparison_prediction(teacher: str, slug: str, stem: str) -> dict | None:
    path = comparison_prediction_path(teacher, slug, stem)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def comparison_pred_to_ls_result(pred: dict, teacher: str) -> list[dict]:
    """Convert a saved comparison prediction JSON into LS rectanglelabels."""
    boxes = pred.get("boxes_xywhn") or []
    results: list[dict] = []
    for i, box in enumerate(boxes):
        if len(box) != 4:
            continue
        cx, cy, bw, bh = box
        x_pct = (cx - bw / 2) * 100
        y_pct = (cy - bh / 2) * 100
        w_pct = bw * 100
        h_pct = bh * 100
        results.append(
            {
                "id": f"{teacher}_{i}",
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "image_rotation": 0,
                "value": {
                    "x": x_pct,
                    "y": y_pct,
                    "width": w_pct,
                    "height": h_pct,
                    "rotation": 0,
                    "rectanglelabels": ["vehicle"],
                },
            }
        )
    return results


def run_comparison_mode() -> None:
    if not COMPARISON_PLAN.exists():
        log(f"ERROR: {COMPARISON_PLAN} not found. Run scripts/build_comparison_set.py first.")
        sys.exit(1)

    wait_for_ls()
    verify_token_auth()
    project_id = find_or_create_project(
        COMPARISON_PROJECT_TITLE,
        COMPARISON_LABEL_CONFIG,
        description=(
            "M8-P1.5 teacher bake-off. 1000 frames stratified across 3 junctions "
            "with one prediction per teacher (yolov8x, yolo26x, yolo12x, rfdetr_m, "
            "grounding_dino_t). Flip between teachers via the prediction dropdown "
            "to pick a winner."
        ),
    )
    ensure_local_files_storage(project_id, COMPARISON_STORAGE_TITLE, COMPARISON_STORAGE_PATH)

    plan = json.loads(COMPARISON_PLAN.read_text())
    plan_frames = plan["frames"]
    log(f"Plan has {len(plan_frames)} frames")

    tasks = fetch_all_tasks(project_id)
    log(f"Found {len(tasks)} tasks in project (auto-synced from storage)")
    by_stem = index_tasks_by_stem(tasks)

    # Which teachers have saved predictions on disk?
    teachers_available = [
        t for t in TEACHERS
        if (COMPARISON_PREDICTIONS / t).exists()
    ]
    if not teachers_available:
        log("  WARN: no teacher predictions on disk — run scripts/run_comparison.py first")
    else:
        log(f"Teachers with predictions available: {teachers_available}")

    n_metadata = 0
    n_predictions_new: dict[str, int] = {t: 0 for t in teachers_available}
    n_predictions_skipped: dict[str, int] = {t: 0 for t in teachers_available}
    n_no_task = 0
    n_no_saved_pred: dict[str, int] = {t: 0 for t in teachers_available}

    for entry in plan_frames:
        slug = entry["site"]
        stem = entry["stem"]
        task = by_stem.get(stem)
        if task is None:
            n_no_task += 1
            continue
        task_id = int(task["id"])

        # Metadata on data payload (needed for downstream routing)
        existing_data = task.get("data") or {}
        if existing_data.get("site") != slug or existing_data.get("stem") != stem:
            patch_task_metadata(task_id, existing_data.get("image", ""), slug, stem)
            n_metadata += 1

        # Dedup: which teachers already have a prediction on this task?
        existing_preds = fetch_task_predictions(task_id)
        existing_versions = {
            p.get("model_version") for p in existing_preds
        }

        for teacher in teachers_available:
            if teacher in existing_versions:
                n_predictions_skipped[teacher] += 1
                continue
            saved = load_comparison_prediction(teacher, slug, stem)
            if saved is None:
                n_no_saved_pred[teacher] += 1
                continue
            result = comparison_pred_to_ls_result(saved, teacher)
            # We still POST even for empty result lists — LS accepts empty
            # predictions, and an empty prediction is a meaningful signal
            # ("this teacher saw nothing here"). But skip if result is empty
            # AND there's nothing to save, because LS rejects completely
            # empty predictions with a 400.
            if not result:
                continue
            post_prediction(task_id, result, model_version=teacher)
            n_predictions_new[teacher] += 1

    log("")
    log(f"  Metadata PATCHed: {n_metadata}")
    for t in teachers_available:
        log(
            f"  {t}: +{n_predictions_new[t]} new predictions "
            f"({n_predictions_skipped[t]} already posted, "
            f"{n_no_saved_pred[t]} missing on disk)"
        )
    if n_no_task:
        log(f"  WARN: {n_no_task} plan frames had no matching task")

    log("Done.")
    log(f"  Project URL: http://localhost:8080/projects/{project_id}/data")
    log("  Login: joel@vt.local / vtlabels2026")


def run_review_mode() -> None:
    raise NotImplementedError(
        "Review mode is deferred to Phase F (post-bake-off). "
        "Once a winning teacher is chosen and the full 6,434 frames are "
        "re-labeled via `make train-label TEACHER=<winner>`, extend this "
        "script to import those labels as a single prediction per task."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["comparison", "review"],
        default="comparison",
        help="LS project mode (default: comparison)",
    )
    args = parser.parse_args()

    if args.mode == "comparison":
        run_comparison_mode()
    elif args.mode == "review":
        run_review_mode()


if __name__ == "__main__":
    main()
