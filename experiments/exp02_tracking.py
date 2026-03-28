"""
Experiment 2: NvDCF tracker stability on junction footage.

Runs DeepStream pipeline with TrafficCamNet + NvDCF tracker, measures
ID switch rate and tracking stability.

Usage (inside container):
    python3 experiments/exp02_tracking.py /data/site_videos/<video.mp4> [max_frames]
"""

from pyservicemaker import Pipeline, Flow, Probe, BatchMetadataOperator
from pyservicemaker.flow import RenderMode
import sys
import os
import time

PGIE_CONFIG = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test1/dstest1_pgie_config.yml"
TRACKER_CONFIG = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"

CLASS_LABELS = {0: "car", 1: "bicycle", 2: "person", 3: "roadsign"}


class TrackingAnalyzer(BatchMetadataOperator):
    """Analyze tracking stability: ID switches, track lifetimes."""

    def __init__(self, max_frames=900):
        super().__init__()
        self.frame_count = 0
        self.max_frames = max_frames
        self.start_time = None
        # Track state: object_id -> {first_frame, last_frame, class_id, positions}
        self.tracks = {}
        # Per-frame: set of active object IDs
        self.prev_frame_ids = set()
        self.id_appearances = 0  # total (id, frame) pairs
        self.max_id_seen = 0

    def handle_metadata(self, batch_meta):
        try:
            if self.start_time is None:
                self.start_time = time.time()
                print("Probe active, processing frames...", flush=True)

            for frame_meta in batch_meta.frame_items:
                self.frame_count += 1
                current_ids = set()

                for obj_meta in frame_meta.object_items:
                    oid = obj_meta.object_id
                    class_id = obj_meta.class_id
                    current_ids.add(oid)
                    self.id_appearances += 1

                    if oid > self.max_id_seen:
                        self.max_id_seen = oid

                    if oid not in self.tracks:
                        self.tracks[oid] = {
                            "first_frame": self.frame_count,
                            "last_frame": self.frame_count,
                            "class_id": class_id,
                            "frame_count": 1,
                        }
                    else:
                        self.tracks[oid]["last_frame"] = self.frame_count
                        self.tracks[oid]["frame_count"] += 1

                self.prev_frame_ids = current_ids

                if self.frame_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    print(f"[Frame {self.frame_count}] active={len(current_ids)}, "
                          f"unique_ids={len(self.tracks)}, fps={fps:.1f}", flush=True)

                if self.frame_count >= self.max_frames:
                    self.print_summary()
                    os._exit(0)
        except Exception as e:
            import traceback
            print(f"ERROR in handle_metadata: {e}", flush=True)
            traceback.print_exc()
            os._exit(1)

    def print_summary(self):
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        # Analyze track lifetimes
        lifetimes = []
        short_tracks = 0  # tracks < 5 frames
        class_track_counts = {}
        for oid, info in self.tracks.items():
            span = info["last_frame"] - info["first_frame"] + 1
            lifetimes.append(span)
            if span < 5:
                short_tracks += 1
            label = CLASS_LABELS.get(info["class_id"], f"unknown_{info['class_id']}")
            class_track_counts[label] = class_track_counts.get(label, 0) + 1

        avg_lifetime = sum(lifetimes) / len(lifetimes) if lifetimes else 0
        max_lifetime = max(lifetimes) if lifetimes else 0
        median_lifetime = sorted(lifetimes)[len(lifetimes) // 2] if lifetimes else 0

        # Estimate ID switch rate: total unique IDs vs expected
        # A high max_id relative to unique tracks suggests ID recycling/switching
        unique_tracks = len(self.tracks)

        lines = [
            "",
            "=" * 60,
            "EXPERIMENT 2: TRACKING RESULTS",
            "=" * 60,
            f"Frames processed:       {self.frame_count}",
            f"Processing FPS:         {fps:.1f}",
            f"Unique track IDs:       {unique_tracks}",
            f"Max ID seen:            {self.max_id_seen}",
            f"Total (id,frame) pairs: {self.id_appearances}",
            f"Short tracks (<5 fr):   {short_tracks} ({100*short_tracks/unique_tracks:.1f}%)" if unique_tracks else "Short tracks: 0",
            f"Avg track lifetime:     {avg_lifetime:.1f} frames",
            f"Median track lifetime:  {median_lifetime} frames",
            f"Max track lifetime:     {max_lifetime} frames",
            f"Tracks per class:       {class_track_counts}",
            f"Elapsed:                {elapsed:.1f}s",
            "=" * 60,
        ]
        print("\n".join(lines), flush=True)


def main(file_path, max_frames):
    analyzer = TrackingAnalyzer(max_frames=max_frames)
    file_uri = f"file://{os.path.abspath(file_path)}"

    pipeline = Pipeline("exp02-tracking")
    flow = Flow(pipeline, [file_uri])
    flow.batch_capture([file_uri]) \
        .infer(PGIE_CONFIG) \
        .track(ll_config_file=TRACKER_CONFIG, ll_lib_file="/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so") \
        .attach(Probe("analyzer", analyzer)) \
        .render(mode=RenderMode.DISCARD, enable_osd=False)
    flow()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <video.mp4> [max_frames=900]")
        sys.exit(1)

    file_path = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 900
    main(file_path, max_frames)
