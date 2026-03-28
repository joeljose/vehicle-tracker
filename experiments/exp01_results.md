# Experiment 1: TrafficCamNet Detection Quality

## Setup
- Model: TrafficCamNet (resnet18_trafficcamnet_pruned.onnx, FP16)
- Pipeline: `nvurisrcbin` → `nvstreammux` → `nvinfer` → `fakesink` (via Flow API)
- 900 frames per video, `interval=0` (every frame)
- Config: `dstest1_pgie_config.yml` (pre-cluster-threshold: 0.2, topk: 20, NMS IoU: 0.5)

## Results

| Junction | Avg det/frame | Frames w/ 0 dets | Avg confidence | FPS |
|---|---|---|---|---|
| 741 & 73 | 1.1 | 16.8% | 0.285 | 982 |
| 741 & Lytle South | 20.1 | 0.0% | 0.590 | 967 |
| Drugmart & 73 | 20.1 | 0.0% | 0.666 | 992 |

### Class distribution
- **741 & 73**: car=963 (100%)
- **741 & Lytle South**: car=17912 (98.8%), roadsign=217 (1.2%)
- **Drugmart & 73**: car=18000 (99.7%), roadsign=37, person=18, bicycle=2

## Findings
1. TrafficCamNet detects vehicles in all 3 junctions — **PASS**
2. 741 & 73 has low detection rate (1.1/frame) and low confidence (0.285) — likely wide-angle or distant shot. May need threshold tuning or alternative model for this junction.
3. Other two junctions have strong detection (~20/frame, >0.59 confidence)
4. Processing speed is ~970+ FPS on a single video — well above real-time requirements
5. All 4 classes detected (car, roadsign, person, bicycle) across the dataset
6. Audio stream warning is harmless — GStreamer skips audio tracks automatically

## Key API discovery
- Must use `Flow` API (not manual `Pipeline.add` + `Pipeline.link`) for `uridecodebin`-based sources
- `BatchMetadataOperator.handle_metadata` errors are silently swallowed — always wrap in try/except
- `os._exit()` doesn't flush stdout — use `flush=True` on all prints before calling it
