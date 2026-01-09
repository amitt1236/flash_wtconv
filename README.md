# Flash WTConv

A high-performance CUDA/Triton/Metal implementation of the Wavelet Convolution (WTConv) layer from *"Wavelet Convolutions for Large Receptive Fields"* by Finder et al.

## Overview

Flash WTConv optimizes the original WTConv implementation through:

- **Haar Transform Optimization**: Replaces convolution-based Haar filters with efficient addition/subtraction operations
- **Cascade Transform**: Fuses multi-level wavelet transforms to increase arithmetic intensity
- **Smart Scaling**: Bakes channel-wise scaling into convolution weights for zero overhead
- **Multi-precision Support**: FP32, FP16, and BF16

## Performance

Compared to the naive WTConv implementation:

| Platform | Speedup |
|----------|---------|
| CUDA (FP32) | ~2.9x |
| CUDA (FP16) | ~3.8x |
| Metal (M3) | ~2.3x |
| Triton | +10% over hand-written CUDA |

## Implementations

- **CUDA**: Custom kernels with fused Haar transform and inverse
- **Triton**: Fused Haar-Conv-Scale kernel with auto-tuning
- **Metal**: Apple Silicon support via custom Metal shaders
- **JAX**: XLA-compiled implementation for TPU compatibility

## Requirements

- PyTorch
- CUDA toolkit (for CUDA kernels)
- Triton (included with PyTorch)

## Citation

Based on the paper:
```bibtex
@article{finder2024wavelet,
  title={Wavelet Convolutions for Large Receptive Fields},
  author={Finder, et al.},
  year={2024}
}
```

## Authors

Amit Aflalo & Mohamad Essa
