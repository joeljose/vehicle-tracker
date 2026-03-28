"""
Experiment 3: Frame buffer access from Python.

Tests accessing raw frame data via BufferOperator probe to validate
that best-photo cropping is feasible.

Usage (inside container):
    python3 experiments/exp03_frame_buffer.py /data/site_videos/<video.mp4>
"""

from pyservicemaker import Pipeline, Flow, Probe, BufferOperator
from pyservicemaker.flow import RenderMode
import sys
import os
import numpy as np


class FrameInspector(BufferOperator):
    """Inspect raw frame buffer data."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def handle_buffer(self, buffer):
        try:
            self.count += 1
            if self.count > 3:
                return True

            print(f"\n--- Frame {self.count} ---", flush=True)
            print(f"Buffer batch_size: {buffer.batch_size}", flush=True)
            print(f"Buffer timestamp: {buffer.timestamp}", flush=True)

            for batch_id in range(buffer.batch_size):
                tensor = buffer.extract(batch_id)
                print(f"  Tensor shape:       {tensor.shape}", flush=True)
                print(f"  Tensor dtype:       {tensor.dtype}", flush=True)
                print(f"  Tensor strides:     {tensor.strides}", flush=True)
                print(f"  Tensor size:        {tensor.size()} bytes", flush=True)
                print(f"  Tensor device_type: {tensor.device_type}", flush=True)
                print(f"  DLPack device:      {tensor.__dlpack_device__()}", flush=True)

                # Convert GPU tensor to CuPy array, then to NumPy
                try:
                    import cupy as cp
                    gpu_arr = cp.from_dlpack(tensor)
                    print(f"  CuPy shape:         {gpu_arr.shape}", flush=True)
                    print(f"  CuPy dtype:         {gpu_arr.dtype}", flush=True)
                    cpu_arr = cp.asnumpy(gpu_arr)
                    print(f"  NumPy shape:        {cpu_arr.shape}", flush=True)
                    print(f"  NumPy dtype:        {cpu_arr.dtype}", flush=True)
                    print(f"  NumPy min/max:      {cpu_arr.min()}/{cpu_arr.max()}", flush=True)
                    print(f"  Non-zero pixels:    {np.count_nonzero(cpu_arr)}/{cpu_arr.size}", flush=True)
                except Exception as e:
                    import traceback
                    print(f"  CuPy/NumPy conversion failed: {e}", flush=True)
                    traceback.print_exc()

            if self.count == 3:
                print("\nSUCCESS: Frame buffer access works!", flush=True)
                os._exit(0)

            return True
        except Exception as e:
            import traceback
            print(f"ERROR: {e}", flush=True)
            traceback.print_exc()
            os._exit(1)


def main(file_path):
    inspector = FrameInspector()
    file_uri = f"file://{os.path.abspath(file_path)}"

    pgie_config = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test1/dstest1_pgie_config.yml"

    # Build pipeline manually to insert nvvideoconvert for RGB conversion
    pipeline = Pipeline("exp03-buffer")
    pipeline.add("nvurisrcbin", "src", {"uri": file_uri})
    pipeline.add("nvstreammux", "mux", {
        "batch-size": 1,
        "batched-push-timeout": 33000,
        "width": 1920,
        "height": 1080,
    })
    pipeline.add("nvinfer", "infer", {"config-file-path": pgie_config})
    pipeline.add("nvvideoconvert", "conv", {})
    pipeline.add("capsfilter", "caps", {
        "caps": "video/x-raw(memory:NVMM),format=RGB",
    })
    pipeline.add("fakesink", "sink", {"sync": False})

    pipeline.link(("src", "mux"), ("", "sink_%u"))
    pipeline.link("mux", "infer", "conv", "caps", "sink")
    pipeline.attach("caps", Probe("inspector", inspector))
    pipeline.start().wait()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <video.mp4>")
        sys.exit(1)

    main(sys.argv[1])
