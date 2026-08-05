# Flash WTConv

A high-performance CUDA/Triton/Metal implementation of the Wavelet Convolution (WTConv) layer from *"Wavelet Convolutions for Large Receptive Fields"* by Finder et al.

## Overview

Flash WTConv optimizes the original WTConv implementation through:

- **Weight fusion with Haar**: each depthwise-conv tap reads exactly one Haar coefficient, which comes from exactly one 2x2 input block. So the transform is folded into the conv weights and the wavelet coefficients are never written to memory.
- **Both transforms fused**: at level 1 the inverse Haar, the deeper levels' reconstruction and the base-conv addition fold into the same kernel, so the full-resolution coefficient tensor never exists.
- **Cascade Transform**: the multi-level inverse runs in registers in one kernel
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
level 1 only        acc[LL] += deeper-level reconstruction
                    y[2h2..+1, 2w2..+1] = inverse_haar(acc) + base_conv
```

`w` is `scale * weight`, folded on the host into a `(C, 4, K, K)` fp32 tensor — 4x less weight traffic than the `(C, 4, 2K, 2K)` "effective kernel" a naive fusion would build. Each CUDA block stages its output tile's partial sums (plus a `K-1` halo) in shared memory, so every input pixel is read from HBM exactly once.

Composing the two transforms gives a polyphase-structured full-resolution convolution,

```
y[2h+qy, 2w+qx] = SUM_{kh,kw} SUM_{ry,rx} W[qy,qx][ry,rx][kh,kw] * x[2(h+kh-R)+ry, 2(w+kw-R)+rx]
W[qy,qx][ry,rx][kh,kw] = SUM_s H[s,qy,qx] * H[s,ry,rx] * w_s[kh,kw]     (H = 4x4 Haar matrix)
```

whose 16 phase pairs come from only `4*K*K` parameters. Keeping it factored (partial sums -> `K*K` taps -> butterfly) costs exactly what the unfused convolution did, while materialising `W` would mean 4x the multiply-accumulates.

Levels 2..L still produce coefficients, because the cascade has to consume them; they are a quarter of the work each. The whole branch is a single autograd node, which is what lets the raw-LL gradient fold into the same grad-input kernel as the coefficient gradients.

## Performance

Measured on an RTX A6000, fp32, K=3, forward+backward against the original implementation (`tests/cuda_metal_tests/test_wtconv.py`):

| wt_levels | 16x32x256x256 | 16x32x512x512 |
|-----------|---------------|---------------|
| 1 | 3.2x | 3.1x |
| 2 | 3.8x | 3.4x |
| 3 | 3.9x | 3.6x |
| 4 | 3.9x | 3.6x |
| 5 | 3.9x | 3.6x |

Activation memory drops ~2.7x against the original, since no coefficient tensor is ever materialised.

Fusing the inverse into level 1 (K=5, wavelet branch only, against the same kernels with a separate inverse pass):

| wt_levels | forward | fwd+bwd |
|-----------|---------|---------|
| 1 | 1.88x | 1.10x |
| 2-5 | 1.19x | 1.02x |

Level 1 saves the full-resolution coefficient round trip outright. Deeper levels pay one extra read of the input for the LL-only downsample that feeds them, so the win is smaller. Backward is unchanged by design — it is dominated by cuDNN's depthwise weight gradient (~55% of a training step), which is the next thing worth replacing.

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
│   ├── fused_haar_conv.cu   # fused Haar+conv+scale (+inverse) fwd/bwd, LL downsample
│   ├── ihaar_cascade.cu     # 1-5 level inverse cascade with optional fused add
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
