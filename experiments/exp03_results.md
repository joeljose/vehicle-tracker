# Experiment 3: Frame Buffer Access from Python

## Setup
- Pipeline: `nvurisrcbin` → `nvstreammux` → `nvinfer` → `nvvideoconvert` → `capsfilter(RGB)` → `fakesink`
- Probe attached after `capsfilter` via `BufferOperator`
- CuPy used for GPU→CPU tensor transfer via DLPack

## Results
- **Tensor shape**: `(1080, 1920, 3)` — height, width, channels (RGB)
- **Tensor dtype**: `uint8`
- **Tensor strides**: `(6144, 3, 1)` — row stride padded to 6144 (vs 5760 = 1920*3) for GPU alignment
- **Tensor size**: 6,220,800 bytes (~6MB per frame)
- **Device**: GPU (DLPack device type 2 = CUDA, device 0)
- **GPU→CPU path**: `Buffer.extract(batch_id)` → `cupy.from_dlpack()` → `cupy.asnumpy()` → numpy array

## Key Findings
1. **Frame buffer IS accessible from Python** — **PASS**
2. `buffer.extract(batch_id)` requires RGB format — must insert `nvvideoconvert` + `capsfilter` before probe
3. `np.from_dlpack()` doesn't work directly (DLPack `__dlpack__` signature mismatch) — use CuPy as intermediary
4. `tensor.clone()` stays on GPU (doesn't copy to CPU)
5. Row stride padding (6144 vs 5760) — need to account for this when cropping ROIs
6. Best-photo cropping is feasible: access frame as numpy, crop bbox region, save to disk
7. For production: keep cropping on GPU with CuPy to avoid GPU→CPU→GPU overhead

## Requirements for Best-Photo Pipeline
- Add `nvvideoconvert` + RGB capsfilter branch after tracker
- Use `BufferOperator` probe (not `BatchMetadataOperator`) to access both frame data AND metadata
- CuPy required in container (already available: v13.4.1)
