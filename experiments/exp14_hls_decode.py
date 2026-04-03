"""Experiment 14: HLS stream decode for custom pipeline

Tests whether PyNvVideoCodec CreateDemuxer can handle HLS URLs directly.
If not, tests OpenCV VideoCapture fallback (CPU decode + GPU upload).

Usage (inside backend container):
    # Step 1: Get an HLS URL from a YouTube Live stream
    yt-dlp -f bv* --get-url "https://www.youtube.com/watch?v=LIVE_VIDEO_ID"

    # Step 2: Run with the HLS URL
    python3 experiments/exp14_hls_decode.py <HLS_URL>

    # Or test with a local file first (baseline)
    python3 experiments/exp14_hls_decode.py /data/test_clips/741_73_1min.mp4
"""

import sys
import time
import traceback

MAX_FRAMES = 300  # ~10s at 30fps


def test_pynvvideocodec(source: str, max_frames: int = MAX_FRAMES) -> dict:
    """Test PyNvVideoCodec CreateDemuxer with the source."""
    print(f"\n{'='*60}")
    print("Test 1: PyNvVideoCodec CreateDemuxer")
    print(f"{'='*60}")
    print(f"Source: {source[:120]}...")

    try:
        from PyNvVideoCodec import CreateDecoder, CreateDemuxer

        t0 = time.monotonic()
        demuxer = CreateDemuxer(source)
        demux_time = time.monotonic() - t0
        print(f"  CreateDemuxer: OK ({demux_time*1000:.0f}ms)")
        print(f"  Resolution: {demuxer.Width()}x{demuxer.Height()}")
        print(f"  FPS: {demuxer.FrameRate()}")
        print(f"  Codec: {demuxer.GetNvCodecId()}")

        decoder = CreateDecoder(
            gpuid=0,
            codec=demuxer.GetNvCodecId(),
            usedevicememory=True,
        )
        print("  CreateDecoder: OK")

        import cupy as cp

        frames = 0
        errors = 0
        latencies = []
        t_start = time.monotonic()

        while frames < max_frames:
            t0 = time.monotonic()
            packet = demuxer.Demux()
            if packet.bsl == 0:
                print(f"  EOS after {frames} frames")
                break

            decoded = decoder.Decode(packet)
            for f in decoded:
                frames += 1
                lat = (time.monotonic() - t0) * 1000
                latencies.append(lat)
                if frames <= 3 or frames % 100 == 0:
                    # Verify we can access the GPU surface
                    ptr = f.GetPtrToPlane(0)
                    h = int(demuxer.Height() * 3 // 2)
                    w = demuxer.Width()
                    mem = cp.cuda.UnownedMemory(ptr, h * w, f)
                    memptr = cp.cuda.MemoryPointer(mem, 0)
                    arr = cp.ndarray((h, w), dtype=cp.uint8, memptr=memptr)
                    print(f"  Frame {frames}: shape={arr.shape}, "
                          f"mean={float(cp.mean(arr)):.1f}, lat={lat:.1f}ms")

        elapsed = time.monotonic() - t_start
        fps = frames / elapsed if elapsed > 0 else 0
        avg_lat = sum(latencies) / len(latencies) if latencies else 0

        result = {
            "method": "PyNvVideoCodec",
            "success": True,
            "frames": frames,
            "fps": round(fps, 1),
            "avg_latency_ms": round(avg_lat, 2),
            "errors": errors,
        }
        print(f"\n  Result: {frames} frames, {fps:.1f} fps, "
              f"avg latency {avg_lat:.1f}ms")
        return result

    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {
            "method": "PyNvVideoCodec",
            "success": False,
            "error": f"{type(e).__name__}: {e}",
        }


def test_opencv_cpu(source: str, max_frames: int = MAX_FRAMES) -> dict:
    """Test OpenCV VideoCapture (CPU decode) with GPU upload."""
    print(f"\n{'='*60}")
    print("Test 2: OpenCV VideoCapture (CPU decode + GPU upload)")
    print(f"{'='*60}")
    print(f"Source: {source[:120]}...")

    try:
        import cv2

        try:
            import cupy as cp
            has_cupy = True
        except ImportError:
            has_cupy = False

        t0 = time.monotonic()
        cap = cv2.VideoCapture(source)
        open_time = time.monotonic() - t0

        if not cap.isOpened():
            print(f"  FAILED: VideoCapture could not open source ({open_time*1000:.0f}ms)")
            return {
                "method": "OpenCV_CPU",
                "success": False,
                "error": "VideoCapture could not open source",
            }

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_source = cap.get(cv2.CAP_PROP_FPS)
        print(f"  VideoCapture opened: {w}x{h} @ {fps_source:.1f}fps ({open_time*1000:.0f}ms)")

        frames = 0
        latencies = []
        upload_times = []
        t_start = time.monotonic()

        while frames < max_frames:
            t0 = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                print(f"  EOS/error after {frames} frames")
                break

            read_time = time.monotonic() - t0
            frames += 1

            # GPU upload
            if has_cupy:
                t_up = time.monotonic()
                gpu_frame = cp.asarray(frame)
                cp.cuda.Device(0).synchronize()
                upload_ms = (time.monotonic() - t_up) * 1000
                upload_times.append(upload_ms)
            else:
                upload_ms = 0

            total_lat = (time.monotonic() - t0) * 1000
            latencies.append(total_lat)

            if frames <= 3 or frames % 100 == 0:
                print(f"  Frame {frames}: shape={frame.shape}, "
                      f"read={read_time*1000:.1f}ms, "
                      f"upload={upload_ms:.1f}ms, "
                      f"total={total_lat:.1f}ms")

        cap.release()
        elapsed = time.monotonic() - t_start
        fps = frames / elapsed if elapsed > 0 else 0
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        avg_upload = sum(upload_times) / len(upload_times) if upload_times else 0

        result = {
            "method": "OpenCV_CPU",
            "success": True,
            "frames": frames,
            "fps": round(fps, 1),
            "avg_latency_ms": round(avg_lat, 2),
            "avg_upload_ms": round(avg_upload, 2),
        }
        print(f"\n  Result: {frames} frames, {fps:.1f} fps, "
              f"avg latency {avg_lat:.1f}ms (upload {avg_upload:.1f}ms)")
        return result

    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {
            "method": "OpenCV_CPU",
            "success": False,
            "error": f"{type(e).__name__}: {e}",
        }


def test_gstreamer_nvdec(source: str, max_frames: int = MAX_FRAMES) -> dict:
    """Test GStreamer uridecodebin + NVDEC (same path as DeepStream backend)."""
    print(f"\n{'='*60}")
    print("Test 3: GStreamer uridecodebin + NVDEC (DeepStream path)")
    print(f"{'='*60}")
    print(f"Source: {source[:120]}...")

    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
    except (ImportError, ValueError) as e:
        print(f"  SKIPPED: GStreamer not available: {e}")
        return {
            "method": "GStreamer_NVDEC",
            "success": False,
            "error": f"GStreamer not available: {e}",
        }

    try:
        import cupy as cp

        # Build pipeline: uridecodebin → nvvideoconvert → appsink (raw GPU frames)
        # uridecodebin handles HLS natively via hlsdemux + souphttpsrc
        is_url = source.startswith(("http://", "https://"))
        uri = source if is_url else f"file://{source}"

        pipeline_str = (
            f'uridecodebin uri="{uri}" ! '
            "nvvideoconvert ! "
            'video/x-raw(memory:NVMM),format=NV12 ! '
            "appsink name=sink emit-signals=true max-buffers=2 drop=true"
        )
        pipeline = Gst.parse_launch(pipeline_str)
        sink = pipeline.get_by_name("sink")

        frames = 0
        latencies = []
        first_frame_time = None

        def on_new_sample(appsink):
            nonlocal frames, first_frame_time
            t0 = time.monotonic()
            sample = appsink.emit("pull-sample")
            if sample is None:
                return Gst.FlowReturn.ERROR

            buf = sample.get_buffer()
            frames += 1
            lat = (time.monotonic() - t0) * 1000
            latencies.append(lat)

            if first_frame_time is None:
                first_frame_time = time.monotonic()

            if frames <= 3 or frames % 100 == 0:
                caps = sample.get_caps()
                struct = caps.get_structure(0)
                w = struct.get_int("width")[1]
                h = struct.get_int("height")[1]
                print(f"  Frame {frames}: {w}x{h}, buf_size={buf.get_size()}, "
                      f"lat={lat:.1f}ms")

            if frames >= max_frames:
                return Gst.FlowReturn.EOS
            return Gst.FlowReturn.OK

        sink.connect("new-sample", on_new_sample)

        t_start = time.monotonic()
        pipeline.set_state(Gst.State.PLAYING)

        # Wait for frames or timeout
        bus = pipeline.get_bus()
        startup_timeout = 15  # seconds for HLS to start
        print(f"  Starting pipeline (timeout={startup_timeout}s)...")

        while frames < max_frames:
            msg = bus.timed_pop_filtered(
                1 * Gst.SECOND,
                Gst.MessageType.ERROR | Gst.MessageType.EOS,
            )
            if msg:
                if msg.type == Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                    print(f"  GStreamer error: {err.message}")
                    break
                elif msg.type == Gst.MessageType.EOS:
                    print(f"  EOS after {frames} frames")
                    break

            elapsed = time.monotonic() - t_start
            if frames == 0 and elapsed > startup_timeout:
                print(f"  Timeout: no frames after {startup_timeout}s")
                break

        pipeline.set_state(Gst.State.NULL)

        if first_frame_time and frames > 1:
            decode_elapsed = time.monotonic() - first_frame_time
            fps = (frames - 1) / decode_elapsed if decode_elapsed > 0 else 0
        else:
            elapsed = time.monotonic() - t_start
            fps = frames / elapsed if elapsed > 0 else 0

        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        startup_ms = ((first_frame_time - t_start) * 1000) if first_frame_time else 0

        result = {
            "method": "GStreamer_NVDEC",
            "success": frames > 0,
            "frames": frames,
            "fps": round(fps, 1),
            "avg_latency_ms": round(avg_lat, 2),
            "startup_ms": round(startup_ms, 0),
        }
        print(f"\n  Result: {frames} frames, {fps:.1f} fps, "
              f"avg latency {avg_lat:.1f}ms, startup {startup_ms:.0f}ms")
        return result

    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {
            "method": "GStreamer_NVDEC",
            "success": False,
            "error": f"{type(e).__name__}: {e}",
        }


def test_opencv_cuda(source: str, max_frames: int = MAX_FRAMES) -> dict:
    """Test OpenCV with CUDA-accelerated VideoReader (if available)."""
    print(f"\n{'='*60}")
    print("Test 3: OpenCV cudacodec.VideoReader (GPU decode)")
    print(f"{'='*60}")

    try:
        import cv2

        if not hasattr(cv2, "cudacodec"):
            print("  SKIPPED: cv2.cudacodec not available")
            return {
                "method": "OpenCV_CUDA",
                "success": False,
                "error": "cv2.cudacodec not available",
            }

        t0 = time.monotonic()
        reader = cv2.cudacodec.createVideoReader(source)
        open_time = time.monotonic() - t0
        print(f"  VideoReader opened ({open_time*1000:.0f}ms)")

        frames = 0
        latencies = []
        t_start = time.monotonic()

        while frames < max_frames:
            t0 = time.monotonic()
            ret, gpu_mat = reader.nextFrame()
            if not ret:
                print(f"  EOS/error after {frames} frames")
                break

            frames += 1
            lat = (time.monotonic() - t0) * 1000
            latencies.append(lat)

            if frames <= 3 or frames % 100 == 0:
                print(f"  Frame {frames}: lat={lat:.1f}ms")

        elapsed = time.monotonic() - t_start
        fps = frames / elapsed if elapsed > 0 else 0
        avg_lat = sum(latencies) / len(latencies) if latencies else 0

        result = {
            "method": "OpenCV_CUDA",
            "success": True,
            "frames": frames,
            "fps": round(fps, 1),
            "avg_latency_ms": round(avg_lat, 2),
        }
        print(f"\n  Result: {frames} frames, {fps:.1f} fps, "
              f"avg latency {avg_lat:.1f}ms")
        return result

    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {
            "method": "OpenCV_CUDA",
            "success": False,
            "error": f"{type(e).__name__}: {e}",
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 experiments/exp14_hls_decode.py <SOURCE>")
        print("  SOURCE: HLS URL or local file path")
        sys.exit(1)

    source = sys.argv[1]
    is_url = source.startswith(("http://", "https://"))
    print(f"Source type: {'URL (HLS)' if is_url else 'local file'}")

    results = []

    # Test 1: PyNvVideoCodec (the preferred path — GPU decode, no transfer)
    results.append(test_pynvvideocodec(source))

    # Test 2: OpenCV CPU decode + GPU upload (proven fallback)
    results.append(test_opencv_cpu(source))

    # Test 3: GStreamer uridecodebin + NVDEC (what DeepStream uses)
    results.append(test_gstreamer_nvdec(source))

    # Test 4: OpenCV CUDA decode (if available)
    results.append(test_opencv_cuda(source))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "OK" if r["success"] else "FAIL"
        fps = r.get("fps", "N/A")
        lat = r.get("avg_latency_ms", "N/A")
        err = r.get("error", "")
        print(f"  {r['method']:20s}  {status:4s}  fps={fps}  lat={lat}ms  {err}")


if __name__ == "__main__":
    main()
