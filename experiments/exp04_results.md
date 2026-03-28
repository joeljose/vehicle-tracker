# Experiment 4: Runtime Interval Change

## Setup
- Pipeline: Flow API with TrafficCamNet, probe after nvinfer
- Tested interval changes at runtime: 0 → 5 → 15 → 0
- 200 frames per phase, 800 frames total
- API: `pipeline["infer-infer-0"].set({"interval": N})`

## Results

| Phase | Interval | FPS | Detections | Notes |
|---|---|---|---|---|
| 0 | 0 (every frame) | 952 | ~20/frame | Baseline |
| 1 | 5 (every 6th) | 1113 | 0 on skip frames | 17% faster |
| 2 | 15 (every 16th) | 1147 | 0 on skip frames | 20% faster |
| 3 | 0 (every frame) | 967 | ~21/frame | Back to baseline |

## Findings
1. **Runtime interval change works** — **PASS**
2. `pipeline["element_name"].set({"property": value})` — note dict syntax, not key-value
3. `pipeline["element_name"].get("property")` confirms readback
4. No crash, no stall, no pipeline restart needed
5. FPS increases proportionally with interval (less GPU inference work)
6. On skip frames, `object_items` is empty (no detections from previous frames carried over)
7. This validates idle-mode optimization: increase interval when no activity detected

## Implementation Notes
- Use `set({"interval": N})` not `set_property("interval", N)` — pyservicemaker Node API
- Interval=N means infer every (N+1)th frame: interval=0 = every frame, interval=5 = every 6th
- Skipped frames still flow through pipeline but have no detection metadata
