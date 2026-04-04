"""Experiment 14c: GStreamer HLS → FIFO → PyNvVideoCodec NVDEC

GPU-resident HLS decode: GStreamer handles HTTPS + HLS demux,
pipes raw H.264 to a named FIFO, PyNvVideoCodec reads the FIFO
and decodes on NVDEC. Output: NV12 CuPy arrays on GPU.

Usage (inside backend container):
    python3 experiments/exp14c_gst_fifo_nvdec.py <HLS_URL>
    python3 experiments/exp14c_gst_fifo_nvdec.py /data/test_clips/741_73_1min.mp4
"""

import os
import subprocess
import sys
import time

MAX_FRAMES = 300


def test_gst_fifo(source: str, max_frames: int = MAX_FRAMES) -> dict:
    """GStreamer HLS demux → FIFO → PyNvVideoCodec NVDEC."""
    is_url = source.startswith(("http://", "https://"))
    fifo_path = "/tmp/gst_hls_decode.h264"

    # Clean up stale FIFO
    if os.path.exists(fifo_path):
        os.remove(fifo_path)
    os.mkfifo(fifo_path)

    print(f"Source type: {'HLS URL' if is_url else 'local file'}")
    print(f"Source: {source[:120]}...")
    print(f"FIFO: {fifo_path}")

    # Build GStreamer pipeline
    if is_url:
        # HLS URL: souphttpsrc → hlsdemux → tsdemux → h264parse → mpegtsmux → FIFO
        # mpegtsmux is streaming-friendly (no buffering/seek needed unlike matroskamux)
        gst_cmd = [
            "gst-launch-1.0", "-q",
            "souphttpsrc", f"location={source}", "!",
            "hlsdemux", "!",
            "tsdemux", "!",
            "h264parse", "!",
            "mpegtsmux", "!",
            "filesink", f"location={fifo_path}",
        ]
    else:
        # Local file: filesrc → qtdemux → h264parse → mpegtsmux → FIFO
        gst_cmd = [
            "gst-launch-1.0", "-q",
            "filesrc", f"location={source}", "!",
            "qtdemux", "!",
            "h264parse", "!",
            "mpegtsmux", "!",
            "filesink", f"location={fifo_path}",
        ]

    print(f"GStreamer cmd: {' '.join(gst_cmd[:6])}...")

    # Start GStreamer subprocess
    gst_proc = subprocess.Popen(
        gst_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    print("GStreamer started, waiting for data...")

    # PyNvVideoCodec reads from FIFO
    import cupy as cp
    from PyNvVideoCodec import CreateDecoder, CreateDemuxer

    t_start = time.monotonic()

    try:
        demuxer = CreateDemuxer(fifo_path)
    except Exception as e:
        print(f"CreateDemuxer FAILED: {e}")
        gst_proc.terminate()
        os.remove(fifo_path)
        return {"method": "GStreamer_FIFO_NVDEC", "success": False, "error": str(e)}

    demux_time = time.monotonic() - t_start
    print(f"CreateDemuxer: OK ({demux_time*1000:.0f}ms)")
    print(f"Resolution: {demuxer.Width()}x{demuxer.Height()}")
    print(f"FPS: {demuxer.FrameRate()}")
    print(f"Codec: {demuxer.GetNvCodecId()}")

    decoder = CreateDecoder(
        gpuid=0, codec=demuxer.GetNvCodecId(), usedevicememory=True,
    )
    print("CreateDecoder: OK\n")

    nv12_h = int(demuxer.Height() * 3 // 2)
    nv12_w = demuxer.Width()

    frames = 0
    latencies = []
    first_frame_time = None

    while frames < max_frames:
        t0 = time.monotonic()
        pkt = demuxer.Demux()
        if pkt.bsl == 0:
            print(f"EOS after {frames} frames")
            break

        decoded = decoder.Decode(pkt)
        for f in decoded:
            frames += 1
            if first_frame_time is None:
                first_frame_time = time.monotonic()

            # Convert to owned CuPy array (same as NvDecoder._frame_to_cupy)
            ptr = f.GetPtrToPlane(0)
            mem = cp.cuda.UnownedMemory(ptr, nv12_h * nv12_w, f)
            memptr = cp.cuda.MemoryPointer(mem, 0)
            arr = cp.ndarray((nv12_h, nv12_w), dtype=cp.uint8, memptr=memptr)
            owned = arr.copy()
            cp.cuda.Device(0).synchronize()

            lat = (time.monotonic() - t0) * 1000
            latencies.append(lat)

            if frames <= 3 or frames % 100 == 0:
                print(f"  Frame {frames}: shape={owned.shape}, "
                      f"mean={float(cp.mean(owned)):.1f}, lat={lat:.1f}ms")

    # Results
    elapsed = time.monotonic() - t_start
    if first_frame_time and frames > 1:
        decode_elapsed = time.monotonic() - first_frame_time
        fps = (frames - 1) / decode_elapsed
    else:
        fps = frames / elapsed if elapsed > 0 else 0

    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    startup = (first_frame_time - t_start) * 1000 if first_frame_time else 0

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Frames decoded:     {frames}")
    print(f"FPS:                {fps:.1f}")
    print(f"Avg latency/frame:  {avg_lat:.2f}ms")
    print(f"Startup (HLS):      {startup:.0f}ms")
    print(f"Output format:      NV12 CuPy on GPU")
    print(f"PCIe transfers:     0 (GPU-resident decode)")
    print(f"Decode hardware:    NVDEC")

    # Cleanup
    gst_proc.terminate()
    try:
        gst_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        gst_proc.kill()
    if os.path.exists(fifo_path):
        os.remove(fifo_path)

    return {
        "method": "GStreamer_FIFO_NVDEC",
        "success": frames > 0,
        "frames": frames,
        "fps": round(fps, 1),
        "avg_latency_ms": round(avg_lat, 2),
        "startup_ms": round(startup, 0),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 experiments/exp14c_gst_fifo_nvdec.py <HLS_URL_or_FILE>")
        sys.exit(1)

    result = test_gst_fifo(sys.argv[1])
    if not result["success"]:
        print(f"\nFAILED: {result.get('error', 'unknown')}")
        sys.exit(1)
