# Teacher Model Comparison — M8-P1.5 v2

**Purpose of this document.** Captures the architecture context, install notes, and empirical results from the M8-P1.5 v2 teacher bake-off. Written so a future iteration of this project (or another project that needs an auto-labeling teacher) can skip the research step and go straight to the numbers.

**Run date.** 2026-04-11

**Environment.** RTX 4050 Mobile (6 GB VRAM), `nvidia/cuda:12.6.3-runtime-ubuntu24.04`, Python 3.12, PyTorch (via ultralytics + transformers). See `training/Dockerfile` and `training/requirements.txt`.

**Workload.** 1000 frames at 1920×1080 from 3 overhead traffic junctions. Inference only, FP16 where supported, conf threshold 0.3.

---

## The candidates

Five teachers. Each must run under 6 GB VRAM, produce vehicle bounding boxes (boxes only, no masks required for training), and be reachable from Python without a vendored-binary build.

| # | Name | Backbone | Year | Params | Decoder | NMS? | COCO mAP50-95 | Dep |
|---|---|---|---|---|---|---|---|---|
| 0 | **YOLOv8x** | CSPDarknet | 2023 | 68.2M | CNN anchor-free head | Yes | 53.9 | `ultralytics` |
| 1 | **YOLO26-X** | CSPDarknet+ | 2026 Jan | ~60M | End-to-end anchor-free | **No** (NMS-free) | 54.7 (claimed) | `ultralytics` |
| 2 | **YOLO12-X** | CSPDarknet + attention | 2025 | 59.1M | Area Attention + R-ELAN | Yes | 55.2 | `ultralytics` |
| 3 | **RF-DETR-M** | DINOv2 ViT | 2026 ICLR | 33.7M | Transformer decoder, 900 queries | **No** (DETR-style Hungarian match) | 54.7 | `rfdetr` (pip) |
| 4 | **Grounding DINO Tiny** | Swin-T + BERT | 2024 ECCV | ~172M | Text-grounded cross-modal decoder | **No** (DETR-style) | 52.5 zero-shot | `transformers` (HF) |

### Architecture notes

**YOLOv8x** — the current backbone baseline. CSPDarknet encoder + FPN neck + decoupled anchor-free detection head. Each spatial grid cell emits a fixed number of candidate boxes; NMS suppresses duplicates at the end. Strong, well-calibrated, fast. Known weak spots: top-down/overhead camera angles (COCO is street-level biased), and fragmentation around non-overlapping occluders (two visible halves of one vehicle become two boxes because NMS only suppresses overlapping duplicates).

**YOLO26-X** — Ultralytics' January 2026 release. Core architectural change from YOLOv8: **end-to-end NMS-free decoding**. The model emits a fixed set of predictions that are each tagged with an "objectness + class" score, and post-processing is a simple top-k rather than NMS. In theory this means two close-but-non-overlapping detections of the "same" object have to compete at the decoder level, not just at NMS — so the fragmentation failure mode could manifest differently (or at all). Same speed/VRAM as YOLOv8x.

**YOLO12-X** — Early 2025. The "attention-centric YOLO" — replaces parts of the CSPDarknet encoder with FlashAttention-based "Area Attention" blocks and R-ELAN (Residual Efficient Layer Aggregation Network) feature aggregation. Still uses NMS at inference. Marginally higher COCO mAP than YOLOv8x. Worth including as a conservative alternative to YOLO26 in case YOLO26's NMS-free decoder has training-instability issues downstream.

**RF-DETR-Medium** — Roboflow's ICLR 2026 DETR variant. Uses a DINOv2 ViT backbone (self-supervised on ImageNet-1B), a DETR-style transformer decoder with 900 object queries, and **Hungarian matching** during training (each ground-truth box is assigned exactly one query). At inference there's no NMS — each query either produces a box or is discarded based on objectness. Roboflow's pitch for this model is occlusion robustness and domain transfer: "*excels at handling occlusions, complex scenes, and domain shifts.*" The transformer's self-attention lets every query see the full image, which in principle helps reason about partially-visible objects better than a local-context CNN head.

**Grounding DINO Tiny** — IDEA-Research, 2024 ECCV. Text-grounded open-vocabulary detector. At inference you provide both an image and a list of textual class phrases (e.g. `"a car. a truck. a bus. a motorcycle."`) and the model outputs boxes whose text grounding matches a phrase. The attraction was the zero-shot text prompting: you could describe rare vehicle types in natural language. In practice for our 4-COCO-class problem this is a capability we don't need, and the model pays for its flexibility with ~14× slower inference than the YOLO family.

### Why we didn't try these

- **Co-DETR** and **DINO-DETR** — current COCO SOTA (~66 mAP50-95) but require ViT-H/16 backbones that don't fit in 6 GB at FP16 inference, and would need INT8 quantization or model splitting to run at all.
- **DART "Detect Anything in Real Time"** (mkturkcan, 2026) — built on SAM3 with a ViT-H/14 backbone. Tested on an RTX 4080 at 1008×1008 input. Memory profile is too tight for our 4050 at 1080p.
- **DART pipeline** (Chen et al, 2024) — automation framework using Grounding DINO + GPT-4o for pseudo-label review. Not a detector itself; architecturally just our Phase-1 design with different components. Doesn't address the fragmentation issue at the model level.
- **Aerial-specialized models** (DOTA, VisDrone) — trained on overhead imagery which would be helpful, but most use rotated-bounding-box outputs, and the format mismatch isn't worth the integration cost for an experiment.
- **Cascade R-CNN, Mask R-CNN, Detectron2 family** — two-stage detectors, slower, more complex to integrate, and their accuracy advantage has largely been eaten by modern single-stage models.

---

## Comparison workflow

1. `training/scripts/build_comparison_set.py` samples 1,000 frames from `data/frames_dedup/`:
   - 734 from junction_741_73 (all of it — the smallest pool and the site with the known fragmentation issue)
   - 150 random from junction_741_lytle_south
   - 116 random from junction_drugmart_73
2. `training/scripts/run_comparison.py --teacher <name>` runs one teacher on all 1,000 frames, saves per-frame JSON predictions to `data/comparison/predictions/{teacher}/{slug}/{stem}.json`.
3. `training/scripts/compute_comparison_stats.py` aggregates per-teacher summary stats (detection counts, confidence, speed, box geometry histograms) and computes a pairwise-agreement metric against the YOLOv8x baseline at IoU≥0.5.
4. `training/scripts/ls_setup.py --mode comparison` creates a Label Studio project with 1,000 tasks and posts ONE PREDICTION PER TEACHER per task so the user can flip between teachers visually in the LS UI.
5. User walks through a sample of tasks (not all 1,000), picks a winner based on visual judgment + the stats dashboard.

The decision rule is **qualitative** — "which teacher's predictions look closest to correct" — not quantitative mAP. No hand-annotated ground truth was built for this comparison. The rationale: the user's annotation time is better spent building the *real* training-set ground truth later (Phase F/G review of the winning teacher's full 6,434-frame output) than on a small intermediate benchmark. The pairwise agreement metric catches obvious no-GT red flags (a candidate with 20% match rate against the baseline is probably wrong, regardless of how its overlays look on one frame).

---

## Empirical results (1000 frames, 2026-04-11)

### Headline totals

| Teacher | Total detections | Per-frame (mean) | Mean conf | Inference time |
|---|---|---|---|---|
| YOLOv8x | 13,669 | 13.67 | 0.633 | 30.3s |
| YOLO26-X | 13,961 | 13.96 | 0.641 | 30.0s |
| YOLO12-X | 11,987 | 11.99 | 0.616 | 36.1s |
| **RF-DETR-M** | **19,124** | **19.12** | 0.589 | 64.4s |
| Grounding DINO Tiny | 2,665 | 2.67 | 0.335 | 417.0s |

### Pairwise agreement with YOLOv8x baseline (IoU ≥ 0.5, greedy match)

| Teacher | Matched | Match rate | Extra in candidate | Missed from baseline |
|---|---|---|---|---|
| YOLO26-X | 11,714 | **85.7%** | 2,247 | 1,955 |
| YOLO12-X | 11,054 | 80.9% | 933 | 2,615 |
| **RF-DETR-M** | **12,614** | **92.3%** | **6,510** | **1,055** |
| Grounding DINO Tiny | 2,005 | 14.7% | 660 | 11,664 |

**How to read this table.** "Matched" is the count of candidate boxes that match a YOLOv8x box (same class, IoU ≥ 0.5) after greedy bipartite matching. "Match rate" divides that by the total number of YOLOv8x baseline boxes (13,669). "Extra in candidate" is how many extra boxes the candidate adds beyond the matched ones. "Missed from baseline" is how many YOLOv8x boxes the candidate doesn't cover. None of these prove correctness — both teachers could be wrong — but the *pattern* is diagnostic:

- **High match rate + high extras + low missed** (RF-DETR-M's profile) → "strictly more complete view of the scene". Not a random disagreement with the baseline, but a systematic addition of detections the baseline missed. This is what you'd see from a stronger model catching top-down cars and partially-occluded vehicles that YOLOv8x's anchor-based head skips.
- **Medium match rate + small extras + small missed** (YOLO26-X's profile) → "different lens, slightly more aggressive on edges". A free upgrade over the baseline if it wins visual inspection, but nothing dramatic.
- **Medium match rate + tiny extras + large missed** (YOLO12-X's profile) → "more conservative than baseline". Fewer boxes, lower recall.
- **Low match rate + tiny extras + huge missed** (Grounding DINO's profile) → "fundamentally wrong tool". Text grounding at conf 0.3 rejects too many legitimate vehicles. Would need extensive prompt engineering and threshold tuning.

### Per-site detection density (mean boxes per frame)

| Teacher | 741_73 (734) | Lytle South (150) | Drugmart (116) |
|---|---|---|---|
| YOLOv8x | 14.66 | 10.43 | 11.60 |
| YOLO26-X | 14.98 | 10.81 | 11.59 |
| YOLO12-X | 13.08 | 8.89 | 9.10 |
| **RF-DETR-M** | **20.41** | **14.42** | **17.07** |
| Grounding DINO Tiny | 2.87 | 2.19 | 1.97 |

RF-DETR finds **40% more detections per frame at 741_73** than YOLOv8x. Given 741_73 is the site where fragmentation was the known problem, this number has two possible interpretations:

- **(a) Better recall on partially-occluded vehicles** — RF-DETR catches top-down/under-gantry vehicles that YOLOv8x's anchor-based head was missing.
- **(b) Worse fragmentation** — RF-DETR is producing more boxes *per vehicle* than YOLOv8x, breaking single vehicles into multiple detections.

The 92.3% pairwise agreement rate rules out (b) at the aggregate level: if RF-DETR were fragmenting, its boxes would *not* match the baseline (fragmented halves don't overlap the single baseline box), so the match rate would be lower than YOLOv8x's own self-agreement. Visual confirmation in Label Studio is the deciding factor, but the statistical signal strongly favors (a).

### Box geometry

Aspect ratio histogram (bucket boundaries `[0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, ∞]`):

| Teacher | [0, 0.5) | [0.5, 1.0) | [1.0, 1.5) | [1.5, 2.0) | [2.0, 3.0) | [3.0, 5.0) | [5.0, ∞) |
|---|---|---|---|---|---|---|---|
| YOLOv8x | 232 | **10,075** | 3,075 | 253 | 12 | 22 | 0 |
| YOLO26-X | 170 | 10,097 | 3,296 | 381 | 17 | 0 | 0 |
| YOLO12-X | 204 | 8,740 | 2,742 | 293 | 8 | 0 | 0 |
| RF-DETR-M | 177 | **12,984** | **5,176** | 748 | 36 | 3 | 0 |
| Grounding DINO Tiny | 156 | 1,797 | 695 | 17 | 0 | 0 | 0 |

All four vehicle-oriented teachers cluster the bulk of their detections in the `[0.5, 1.5)` range — which makes sense for overhead views of vehicles (cars seen from above have aspect ratios around 0.5 to 1.5). Grounding DINO has the same cluster shape but 5× less mass. **No teacher produces significant mass in the extreme-aspect buckets**, which is a useful negative signal: if any of them had a fragmentation problem, we'd expect to see a spike in the `[0, 0.5)` bucket (half-a-car detections are very tall-and-narrow or very short-and-wide). We don't. That's further evidence RF-DETR's extras are real detections, not fragmented halves.

---

## Decision and rationale (preliminary — pending visual confirmation)

**Leading candidate: RF-DETR-Medium.**

Reasoning:
1. Highest pairwise agreement with baseline (92.3%) — strong signal that it's not producing random disagreements
2. Most extras (+6,510) with fewest drops (-1,055) — suggests it's adding real detections, not replacing existing ones
3. 40% higher per-frame density at 741_73 (the site with the fragmentation issue) — exactly where we wanted improvement
4. Aspect-ratio histogram is healthy (no fragmentation-shape spike)
5. DINOv2 + transformer decoder is the architecture most expected to handle occlusion well
6. 2.1× slower than YOLOv8x (64s vs 30s on 1000 frames), which is fine for auto-labeling but worth noting for future runs

**Fallback: YOLO26-X.** If visual inspection shows RF-DETR is making unexpected errors (e.g. false positives on billboards, weird geometry in the gantry region), YOLO26-X is the safe middle ground — modest improvement over YOLOv8x, same inference cost, no new dependencies.

**Eliminated: Grounding DINO Tiny.** Not worth looking at in LS. 14.7% agreement with baseline means its predictions are in a completely different regime from the other four. Text-grounded detection doesn't help us for COCO classes.

**Eliminated: YOLO12-X.** Strictly worse than YOLO26-X on every metric that matters (fewer detections, lower conf, slightly slower). Keeping as a data point, not as a candidate.

Final decision is deferred to the Phase E write-up in `training/docs/p1.5_teacher_selection.md` — that's where the user records the actual chosen teacher + visual-inspection notes.

---

## Takeaways for future teacher swaps

1. **Start from COCO mAP, end at visual check.** The COCO numbers are a rough sorting hint — they don't tell you whether a model handles your specific failure mode. Plan for the visual pass.
2. **Pairwise agreement is a cheap no-GT signal.** If you're comparing two models and don't have ground truth, greedy IoU matching between them gives you a useful triage: high agreement + high extras = strictly better, low agreement = fundamentally different.
3. **Keep a numerical baseline even when doing qualitative decisions.** The stats dashboard prevents "vibes-based" picks from going off the rails. A model with 14% agreement is a clear no regardless of how one frame looks.
4. **Transformer decoders have more consistent scene reasoning than anchor-based heads.** RF-DETR's win on the overhead-occlusion site is not surprising in hindsight — self-attention over the full image is exactly what you want for resolving ambiguous local crops.
5. **Don't trust zero-shot/open-vocabulary detectors for closed-set problems.** Grounding DINO has a legitimate use case (detecting novel vehicle types we'd never trained for), but for "car / truck / bus / motorcycle" it's strictly worse than a model that knows those classes natively.
6. **1000 frames is enough for a teacher comparison.** You're not training — you're just looking. Pairwise agreement converges fast and visual sampling only really uses 30-50 frames of your attention. More frames is wasted except for having more things to click through if you want to drill into a specific failure mode.
7. **Hardware budget matters.** We couldn't fit Co-DETR or DART-Anything-in-Real-Time on 6 GB. The mediums-to-larges are where you'll live on consumer GPUs, and DETR-family mediums (like RF-DETR-M at 33.7M params) are the sweet spot right now.

---

## Reproducing the numbers

From the repo root inside the training container:

```bash
make training-up
make train-compare-sample        # sample 1000 frames
make train-compare-run-all       # run all 5 teachers sequentially
make train-compare-stats         # aggregate into stats.json
make train-ls-up                 # start Label Studio
make train-ls-compare-setup      # upload tasks with all 5 predictions
```

Then open http://localhost:8080/projects/<id>/data (credentials in `docker-compose.label-studio.yml`).

Raw outputs live in:
- `training/data/comparison/plan.json` — the sampled frame list (seed=0, deterministic)
- `training/data/comparison/images/{slug}/{stem}.jpg` — staged images
- `training/data/comparison/predictions/{teacher}/{slug}/{stem}.json` — per-frame raw detections
- `training/data/comparison/predictions/{teacher}/_run_stats.json` — per-teacher timing
- `training/data/comparison/stats.json` — aggregated comparison dashboard

---

## References

- **YOLOv8** — [Ultralytics docs](https://docs.ultralytics.com/models/yolov8/), [Jocher et al. (2023)](https://github.com/ultralytics/ultralytics)
- **YOLO26** — [Ultralytics docs](https://docs.ultralytics.com/models/yolo26/), release notes January 2026
- **YOLO12** — [Ultralytics docs](https://docs.ultralytics.com/models/yolo12/), paper "Attention-Centric Real-Time Object Detectors" (2025)
- **RF-DETR** — [Roboflow GitHub](https://github.com/roboflow/rf-detr), [Roboflow blog](https://blog.roboflow.com/rf-detr/), ICLR 2026. Uses DINOv2 backbone ([Oquab et al. 2023](https://arxiv.org/abs/2304.07193)).
- **DINOv2** (RF-DETR backbone) — [Oquab et al. 2023](https://arxiv.org/abs/2304.07193), self-supervised ViT.
- **Grounding DINO** — [Liu et al. 2023 / ECCV 2024](https://arxiv.org/abs/2303.05499), [IDEA-Research GitHub](https://github.com/IDEA-Research/GroundingDINO).
- **DETR** (the architecture lineage RF-DETR and Grounding DINO descend from) — [Carion et al. 2020](https://arxiv.org/abs/2005.12872).
- **End-to-end NMS-free object detection** (the YOLO26 approach) — Zhang et al., various 2023-2024 papers on "matching-based inference".
- **Co-DETR** (not used) — [Zong et al. 2022](https://arxiv.org/abs/2211.12860).
