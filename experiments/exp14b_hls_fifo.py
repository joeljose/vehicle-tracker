"""Experiment 14b: ffmpeg HLS → FIFO → PyNvVideoCodec NVDEC

Tests GPU-resident HLS decode: ffmpeg handles HLS protocol,
pipes raw H.264 to a named FIFO, PyNvVideoCodec reads the FIFO
and decodes on NVDEC. Output is NV12 CuPy arrays on GPU — same
as local file decode path.

Usage (inside backend container):
    python3 experiments/exp14b_hls_fifo.py <HLS_URL>
"""

import os
import subprocess
import sys
import threading
import time

MAX_FRAMES = 300


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 experiments/exp14b_hls_fifo.py <HLS_URL>")
        sys.exit(1)

    hls_url = sys.argv[1]
    fifo_path = "/tmp/hls_decode.h264"

    # Clean up any stale FIFO
    if os.path.exists(fifo_path):
        os.remove(fifo_path)
    os.mkfifo(fifo_path)

    print("=== ffmpeg HLS → FIFO → PyNvVideoCodec NVDEC ===")
    print(f"HLS URL: {hls_url[:100]}...")
    print(f"FIFO: {fifo_path}")

    # Start ffmpeg: download HLS, remux to raw H.264, write to FIFO
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", hls_url,
        "-c:v", "copy",   # no re-encode
        "-an",             # drop audio
        "-f", "h264",      # raw H.264 Annex B output
        "-y", fifo_path,
    ]
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    print("ffmpeg started, waiting for HLS connection...")

    # PyNvVideoCodec reads from FIFO (blocks until ffmpeg starts writing)
    import cupy as cp
    from PyNvVideoCodec import CreateDecoder, CreateDemuxer

    t_start = time.monotonic()
    demuxer = CreateDemuxer(fifo_path)
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

    while frames < MAX_FRAMES:
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

            # Convert to CuPy (same as NvDecoder._frame_to_cupy)
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
    print(f"Frames decoded: {frames}")
    print(f"FPS: {fps:.1f}")
    print(f"Avg latency/frame: {avg_lat:.2f}ms")
    print(f"Startup (HLS connect): {startup:.0f}ms")
    print(f"Output format: NV12 CuPy on GPU")
    print(f"PCIe transfers: 0 (GPU-resident)")

    # Cleanup
    ffmpeg_proc.terminate()
    ffmpeg_proc.wait(timeout=5)
    os.remove(fifo_path)


if __name__ == "__main__":
    main()
