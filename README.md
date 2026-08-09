# Flash WTConv

A high-performance CUDA/Triton/Metal implementation of the Wavelet Convolution (WTConv) layer from *"Wavelet Convolutions for Large Receptive Fields"* by Finder et al.

## Overview

Flash WTConv optimizes the original WTConv implementation through:

- **Weight fusion with Haar**: each depthwise-conv tap reads exactly one Haar coefficient, which comes from exactly one 2x2 input block. So the transform is folded into the conv weights and the wavelet coefficients are never written to memory.
- **Cascade Transform**: the multi-level inverse runs in registers in one kernel, with the base-conv addition folded into its final store, so no intermediate low-pass or reconstruction tensor is ever written.
- **Fused weight gradient**: the weight gradient reduces the output gradient against Haar partial sums recomputed on the fly, replacing cuDNN's grouped depthwise weight gradient (K <= 5; larger kernels fall back to cuDNN). The base convolution's weight gradient gets the same treatment, minus the Haar step.
- **Smart Scaling**: Bakes channel-wise scaling into convolution weights for zero overhead
- **Multi-precision Support**: FP32, FP16, and BF16

### Fused wavelet level

For output position `(h2, w2)` and tap `(kh, kw)` with `R = K//2`:

```
                    load x[2*(h2+kh-R) .. +1, 2*(w2+kw-R) .. +1]   (one 2x2 block)
partial sums        S  = (a+b+c+d)/2   Dh = (a+b-c-d)/2
                    Dv = (a-b+c-d)/2   Dd = (a-b-c+d)/2
accumulate          acc[LL] += w[c,0,kh,kw] * S     acc[LH] += w[c,1,kh,kw] * Dh
                    acc[HL] += w[c,2,kh,kw] * Dv    acc[HH] += w[c,3,kh,kw] * Dd
store               coeffs[c, 0..3, h2, w2] = acc
                    ll[c, h2, w2] = S at the centre tap   (skipped at the deepest level)
```

`w` is `scale * weight`, folded on the host into a `(C, 4, K, K)` fp32 tensor — 4x less weight traffic than the `(C, 4, 2K, 2K)` "effective kernel" a naive fusion would build. Each CUDA block stages its output tile's partial sums (plus a `K-1` halo) in shared memory, so the butterfly runs once per 2x2 block and each input pixel is fetched once per tile that needs it, instead of the `K*K` times a thread-per-output formulation would read it; the halo positions are re-staged by neighbouring blocks, but they are small and local enough to hit L2 rather than HBM.

Every level runs this kernel, because the cascade has to consume the coefficients; deeper levels are a quarter of the work each. The whole branch is a single autograd node, which is what lets the raw-LL gradient fold into the same grad-input kernel as the coefficient gradients.

## Performance

Measured on an RTX A6000, fp32, K=3, forward+backward against the original implementation (`tests/cuda_metal_tests/test_wtconv.py`):

| wt_levels | 16x32x256x256 | 16x32x512x512 |
|-----------|---------------|---------------|
| 1 | 3.5x | 3.7x |
| 2 | 4.2x | 4.1x |
| 3 | 4.3x | 4.3x |
| 4 | 4.4x | 4.3x |
| 5 | 4.4x | 4.3x |

Activation memory drops ~2.7x against the original, since no coefficient tensor is ever materialised.

Fused weight gradients vs the cuDNN grouped weight gradient they replace (K=5, whole layer forward+backward):

| shape | wt_levels 1 | 3 | 5 |
|-------|-------------|---|---|
| 16x32x256x256 | 1.49x | 1.52x | 1.46x |
| 16x32x512x512 | 1.56x | 1.82x | 1.59x |
| 8x96x128x128 | 1.49x | 1.45x | 1.43x |
| 4x384x64x64 | 1.37x | 1.33x | 1.30x |

The wavelet kernel is 6.4x faster than cuDNN's (28.7 ms -> 4.4 ms at 16x32x512x512) and also removes the coefficient tensor the cuDNN path had to materialise. The *base* convolution's weight gradient — a plain depthwise gradient, unrelated to the wavelet branch — was then the largest kernel left, so it gets the same kernel minus the Haar step: 18.1 ms -> 3.7 ms. Together they take that training step from 52.8 ms to 27.8 ms, and no single kernel dominates the profile any more.

Gradient accumulation uses fp32 atomics, so the weight gradient is not bitwise reproducible run to run (neither is cuDNN's depthwise path).

`tests/cuda_metal_tests/test_wtconv_correctness.py` validates the layer against the original implementation (forward and all gradients, levels 1-5, K = 1..9, odd sizes, fp32/fp16/bf16).

## Implementations

- **CUDA**: fused Haar-conv-scale kernel + fused inverse cascade with fused add
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
│   ├── fused_haar_conv.cu   # fused Haar+conv+scale fwd/bwd, fused weight gradient
│   ├── ihaar_cascade.cu     # 1-5 level inverse cascade with optional fused add
│   ├── depthwise_grad.cu    # base-conv weight gradient
│   ├── haar_common.cuh      # dtype helpers, Haar / inverse-Haar primitives
│   └── haar_cuda.py         # autograd wrappers and public API
├── metal_haar/        # Metal shaders
├── tpu_haar/          # TPU ops
├── triton_haar/       # Triton kernels
├── tests/             # Test suites
└── WTConv/            # Naive reference implementation
```

## Tests

```bash
python tests/cuda_metal_tests/test_fused_kernels.py       # kernel-level vs reference ops
python tests/cuda_metal_tests/test_wtconv_correctness.py  # layer-level, fwd + all grads
python tests/cuda_metal_tests/test_wtconv.py              # correctness + benchmark vs original
```
