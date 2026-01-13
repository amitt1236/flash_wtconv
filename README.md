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

## Usage

### CUDA / Metal (PyTorch)

The PyTorch implementation auto-detects CUDA or MPS (Metal) devices. You can also manually specify usage.

```python
import torch
from wtconv_model.wtconv import WTConv2d

# 1. Auto-detect device (CUDA or MPS)
model = WTConv2d(in_channels=64, out_channels=64, kernel_size=5, wt_levels=2)

# 2. Key functionality
x = torch.randn(2, 64, 128, 128).to(model.device)
y = model(x)  # Forward pass

# 3. Explicit device
model_cuda = WTConv2d(64, 64, device='cuda')
model_mps = WTConv2d(64, 64, device='mps')
```

### TPU (JAX/Flax)

The JAX implementation is optimized for TPUs using NHWC layout and Flax.

```python
import jax
import jax.numpy as jnp
from wtconv_model.wtconv_tpu import WTConv2d

# Initialize parameters
key = jax.random.PRNGKey(0)
model = WTConv2d(channels=64, kernel_size=5, depth=2)
x = jax.random.normal(key, (2, 128, 128, 64)) # NHWC

# Init and Apply
variables = model.init(key, x)
output = model.apply(variables, x)
```

### Triton (PyTorch)

A pure Triton implementation for CUDA/ROCm GPUs (requires no CUDA toolkit compilation).

```python
import torch
from wtconv_model.wtconv_triton import WTConv2d

# Usage matches the standard PyTorch module
model = WTConv2d(in_channels=64, out_channels=64, wt_levels=2).cuda()
x = torch.randn(2, 64, 128, 128).cuda()
y = model(x)
```

## Project Structure

```
├── wtconv_model/      # Flash WTConv implementations
├── cuda_haar/         # CUDA kernels
├── metal_haar/        # Metal shaders
├── tpu_haar/          # TPU ops
├── triton_haar/       # Triton kernels
├── tests/             # Test suites
└── WTConv/            # Naive reference implementation
```
