# Experiment 5: 60fps Source Handling

## Setup
- Source: "Live Traffic @ 741 & 73.mp4" (60fps, 1920x1080, H.264)
- Pipeline: Flow API with TrafficCamNet, interval=1 (infer every 2nd frame)
- 600 frames processed

## Results
- **Source FPS (from PTS)**: 60.0 — decode runs at full 60fps
- **Pipeline FPS**: 1072 — processing is faster than real-time
- **Frames with detections**: 278 (46.3%) — close to expected 50%
- **Frames without detections**: 322 (53.7%) — skipped inference frames

## Findings
1. **60fps sources correctly decoded at full rate** — **PASS**
2. `interval=1` correctly skips every other frame for inference, achieving ~30fps inference on 60fps source
3. All frames flow through the pipeline (PTS spacing confirms 60fps decode)
4. Detection ratio is slightly below 50% because this junction has sparser traffic (some inference frames also have 0 detections)
5. Pipeline can be set via `infer(config, interval=1)` or runtime `set({"interval": 1})`

## Video FPS Summary
| Video | Source FPS |
|---|---|
| Live Traffic @ 741 & 73 | 60 |
| Live Traffic @ 741 & Lytle South | 30 |
| Live Traffic @ Drugmart & 73 | 30 |

## Implementation Notes
- For 60fps sources: use `interval=1` to achieve 30fps inference
- For 30fps sources: use `interval=0` for every-frame inference
- Source FPS can be detected from PTS delta: `frame_meta.buffer_pts`
