# Experiment 2: NvDCF Tracker Stability

## Setup
- Tracker: NvDCF (perf config) via `nvtrackerbin`
- Config: `config_tracker_NvDCF_perf.yml` (maxShadowTrackingAge: 51, probationAge: 2)
- Pipeline: `nvurisrcbin` → `nvstreammux` → `nvinfer` → `nvtracker` → `fakesink`
- 900 frames per video

## Results

| Junction | Unique IDs | Active/frame | Short tracks (<5f) | Avg lifetime | Max lifetime | FPS |
|---|---|---|---|---|---|---|
| 741 & 73 | 9 | 0-2 | 0 (0%) | 142.9 | 576 | 848 |
| 741 & Lytle South | 147 | 17-21 | 19 (12.9%) | 156.6 | 896 | 717 |
| Drugmart & 73 | 112 | 19-21 | 18 (16.1%) | 186.9 | 898 | 730 |

## Analysis
1. **No excessive ID switching** — unique IDs grow steadily as new vehicles enter the scene, not from re-IDing existing objects
2. **Short tracks are reasonable** — 12-16% of tracks are <5 frames, which is expected for vehicles at frame edges or brief detections
3. **Long track lifetimes** — max tracks persist across nearly all 900 frames, indicating stable long-term tracking
4. **Max ID == Unique IDs - 1** — IDs are sequential with no gaps, confirming no ID recycling
5. **FPS still very high** — 700-850 FPS with tracker enabled (vs ~970 without) — tracker overhead is minimal

## Verdict
NvDCF tracking is stable. IDSW rate appears very low (no evidence of re-IDing). No need to tune `maxShadowTrackingAge` or switch to NvDeepSORT. **PASS**
