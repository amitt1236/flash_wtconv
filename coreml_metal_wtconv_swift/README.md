# WTConv CoreML + Metal (Swift Example)

This folder contains a standalone Swift implementation that runs a one-level `WTConv2d` pipeline from `wtconv_model/wtconv.py` as:

1. Metal Haar forward (`haar2d_forward_kernel`)
2. Core ML depthwise conv + scale/bias
3. Metal Haar inverse (`haar2d_inverse_kernel`)
4. Sum with base branch

The Metal shaders are copied from your existing `metal_haar` kernels.

## Folder Layout

- `Package.swift`: SwiftPM executable package
- `Sources/WTConvCoreMLRunner/main.swift`: CLI entry point
- `Sources/WTConvCoreMLRunner/MetalHaarRunner.swift`: Metal kernel compile + dispatch
- `Sources/WTConvCoreMLRunner/CoreMLWTConvRunner.swift`: Core ML inference wrapper
- `Sources/WTConvCoreMLRunner/Resources/haar_single.metal`: forward Haar kernel source
- `Sources/WTConvCoreMLRunner/Resources/haar_inverse.metal`: inverse Haar kernel source
- `tools/export_wtconv_coreml.py`: export helper from `wtconv_model/wtconv.py`

## 1) Export Core ML model from `wtconv.py`

From repository root:

```bash
cd coreml_metal_wtconv_swift
python tools/export_wtconv_coreml.py --channels 32 --height 128 --width 128
```

This generates:

- `coreml_metal_wtconv_swift/Models/WTConvConvModules.mlpackage`

Notes:

- Export currently targets `wt_levels=1`.
- Export requires `coremltools` and Apple MPS.

## 2) Build and run Swift example

```bash
cd coreml_metal_wtconv_swift
swift build -c release
swift run -c release WTConvCoreMLRunner \
  --model Models/WTConvConvModules.mlpackage \
  --channels 32 --height 128 --width 128
```

Example output includes per-stage timing and a small output summary.

## 3) Check output vs PyTorch (`wtconv_model/wtconv.py`)

This runs a deterministic numerical comparison using the same input tensor and same WTConv weights:

```bash
cd coreml_metal_wtconv_swift
swift build -c release
python3 tools/check_outputs_vs_pytorch.py --channels 32 --height 128 --width 128
```

The script reports:

- `max_abs`
- `mean_abs`
- `rmse`
- `allclose(atol, rtol)`

## CLI options

- `--model`: path to `.mlpackage` (default: `Models/WTConvConvModules.mlpackage`)
- `--channels`: input channels C
- `--height`: input height H
- `--width`: input width W
- `--input-f32`: optional raw float32 input file (NCHW contiguous)
- `--output-f32`: optional raw float32 output file (NCHW contiguous)

## Important

- This is a runnable example pipeline for one-level WTConv (`wt_levels=1`).
- For multi-level WTConv (`wt_levels > 1`), you would extend the same pattern with your cascade kernels and extra Core ML wavelet conv branches.
