"""
Fused CUDA Haar kernels for WTConv.

Everything here is built around fusing the Haar transform into the depthwise
convolution weights, so the wavelet coefficients are never written to memory:

    fused_haar_conv_scale(x, weight, scale, K)   Haar -> conv -> scale, one kernel
    ihaar2d_*_fused(levels..., add)              1-5 level inverse cascade + add
    scaled_depthwise_conv(x, w, s, pad, bias)    base-conv path (scale folded, cuDNN)

The API mirrors `triton_haar` so the two backends are interchangeable.

Layout conventions:
    coefficients   (B, C, 4, H/2, W/2) contiguous, subbands [LL, LH, HL, HH]
    conv weights   (C*4, 1, K, K), channel c*4+s holds subband s of channel c
    scales         (1, C*4, 1, 1)
    fused weights  (C, 4, K, K) float32 = scale * weight, built by
                   compute_scaled_weight()
"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load


# =============================================================================
# Extension loading
# =============================================================================

def _setup_cuda_arch():
    """Auto-detect the compute capability so nvcc does not warn."""
    if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                arch = result.stdout.strip().split('\n')[0]
                os.environ['TORCH_CUDA_ARCH_LIST'] = arch
        except Exception:
            pass


_setup_cuda_arch()

_module = None


def _get_module():
    global _module
    if _module is None:
        src_dir = Path(__file__).parent
        print("Compiling fused Haar CUDA kernels...")
        _module = load(
            name='flash_wtconv_haar',
            sources=[
                str(src_dir / 'haar.cpp'),
                str(src_dir / 'fused_haar_conv.cu'),
                str(src_dir / 'ihaar_cascade.cu'),
            ],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False,
        )
        print("Done.")
    return _module


# =============================================================================
# Fused Haar -> conv -> scale
# =============================================================================

def compute_scaled_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    kernel_size: int = 3,
) -> torch.Tensor:
    """
    Fold the per-channel scale into the depthwise weights and regroup them by
    subband, giving the (C, 4, K, K) float32 tensor the fused kernel consumes.

    Args:
        weight: (C*4, 1, K, K) depthwise conv weights
        scale: (1, C*4, 1, 1) or (C*4,) per-channel scales
        kernel_size: K

    Returns:
        (C, 4, K, K) float32, contiguous
    """
    C4 = weight.shape[0]
    K = kernel_size
    scaled = weight.reshape(C4, K, K) * scale.reshape(C4, 1, 1)
    return scaled.reshape(C4 // 4, 4, K, K).to(torch.float32).contiguous()


def _haar_coeffs(x: torch.Tensor) -> torch.Tensor:
    """Single-level Haar coefficients: (B, C, H, W) -> (B, C, 4, ceil(H/2), ceil(W/2))."""
    B, C, H, W = x.shape
    out = torch.empty(B, C, 4, (H + 1) // 2, (W + 1) // 2,
                      device=x.device, dtype=x.dtype)
    _get_module().haar_coeffs(x, out)
    return out


def _grad_weight_scale(
    coeffs: torch.Tensor,      # (B, C, 4, H2, W2) Haar coefficients the conv saw
    grad_output: torch.Tensor,  # (B, C, 4, H2, W2)
    weight: torch.Tensor,      # (C*4, 1, K, K)
    scale: torch.Tensor,       # (1, C*4, 1, 1)
    kernel_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gradients of the *unfused* weight and scale.

    The forward convolves with w~ = scale * weight, so the chain rule gives
        dL/dweight = scale * dL/dw~        dL/dscale = sum(weight * dL/dw~).
    dL/dw~ itself is an ordinary grouped conv weight gradient (cuDNN).
    """
    B, C, _, H2, W2 = coeffs.shape
    C4 = C * 4
    padding = kernel_size // 2

    coeffs_flat = coeffs.reshape(B, C4, H2, W2).to(weight.dtype)
    grad_flat = grad_output.reshape(B, C4, H2, W2).to(weight.dtype)

    grad_fused = torch.nn.grad.conv2d_weight(
        coeffs_flat, weight.shape, grad_flat, padding=padding, groups=C4
    )

    grad_weight = grad_fused * scale.reshape(C4, 1, 1, 1)
    grad_scale = (grad_fused * weight).sum(dim=(1, 2, 3)).reshape_as(scale)
    return grad_weight, grad_scale


class FusedHaarConvScaleFunction(Function):
    """
    Autograd wrapper around the fused Haar -> conv -> scale kernel.

    forward:  x (B, C, H, W) -> coeffs (B, C, 4, H/2, W/2) [+ raw LL]
    backward: fused kernel for grad_input, cuDNN for grad_weight / grad_scale.
    """

    @staticmethod
    def forward(ctx, x, weight, scale, kernel_size, return_ll):
        assert x.is_cuda, "input must be on CUDA"
        assert x.dim() == 4, "input must be (B, C, H, W)"
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, \
            f"fused Haar conv needs even spatial dims, got {H}x{W} (pad first)"
        K = kernel_size
        assert K % 2 == 1, f"kernel_size must be odd, got {K}"

        x = x.contiguous()
        H2, W2 = H // 2, W // 2

        output = torch.empty(B, C, 4, H2, W2, device=x.device, dtype=x.dtype)
        ll_output = torch.empty(B, C, H2, W2, device=x.device, dtype=x.dtype) \
            if return_ll else None

        fused_weight = compute_scaled_weight(weight, scale, K)
        _get_module().fused_haar_conv_forward(x, fused_weight, output, ll_output)

        ctx.save_for_backward(x, weight, scale, fused_weight)
        ctx.kernel_size = K
        ctx.return_ll = return_ll

        if return_ll:
            return output, ll_output
        return output

    @staticmethod
    def backward(ctx, grad_output, grad_ll=None):
        x, weight, scale, fused_weight = ctx.saved_tensors
        K = ctx.kernel_size
        B, C, H, W = x.shape

        grad_input = grad_weight = grad_scale = None
        need_x, need_w, need_s = ctx.needs_input_grad[:3]

        grad_output = grad_output.contiguous()

        if need_x:
            grad_input = torch.empty_like(x)
            if grad_ll is not None:
                grad_ll = grad_ll.contiguous()
            _get_module().fused_haar_conv_backward(
                grad_output, fused_weight, grad_input, grad_ll
            )

        if need_w or need_s:
            coeffs = _haar_coeffs(x)
            grad_weight, grad_scale = _grad_weight_scale(
                coeffs, grad_output, weight, scale, K
            )
            if not need_w:
                grad_weight = None
            if not need_s:
                grad_scale = None

        return grad_input, grad_weight, grad_scale, None, None


def fused_haar_conv_scale(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    kernel_size: int = 3,
    return_ll: bool = False,
):
    """
    Fused Haar transform -> depthwise conv -> scale, in a single kernel.

    Args:
        x: (B, C, H, W) with even H, W
        weight: (C*4, 1, K, K) depthwise conv weights
        scale: (1, C*4, 1, 1) per-channel scales
        kernel_size: K (odd, <= 9)
        return_ll: also return the raw LL subband (B, C, H/2, W/2), i.e. the
                   input of the next decomposition level, computed for free.

    Returns:
        coeffs: (B, C, 4, H/2, W/2)
        ll_raw: (B, C, H/2, W/2) when return_ll=True
    """
    return FusedHaarConvScaleFunction.apply(x, weight, scale, kernel_size, return_ll)


# =============================================================================
# Inverse Haar cascade (1-5 levels, optional fused add)
# =============================================================================

def run_ihaar_cascade(
    levels: Sequence[torch.Tensor],
    output_size: Optional[Tuple[int, int]] = None,
    add_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused inverse Haar cascade.

    Args:
        levels: [(B, C, 4, H1, W1), (B, C, 4, H2, W2), ...] finest level first
        output_size: (H, W) of the reconstruction; defaults to (2*H1, 2*W1)
        add_tensor: optional (B, C, H, W) folded into the output store

    Returns:
        (B, C, H, W)
    """
    levels = [l.contiguous() for l in levels]
    assert 1 <= len(levels) <= 5, "cascade supports 1-5 levels"

    B, C = levels[0].shape[:2]
    H1, W1 = levels[0].shape[3], levels[0].shape[4]
    H, W = output_size if output_size is not None else (H1 * 2, W1 * 2)

    output = torch.empty(B, C, H, W, device=levels[0].device, dtype=levels[0].dtype)
    if add_tensor is not None:
        add_tensor = add_tensor.contiguous()
    _get_module().ihaar_cascade(levels, output, add_tensor)
    return output


def run_haar_cascade(x: torch.Tensor, num_levels: int) -> List[torch.Tensor]:
    """
    Forward Haar cascade -- the gradient of the inverse cascade.

    Each level transforms the previous level's LL subband, matching the
    ceil-halving shape chain of the forward pass.
    """
    levels = []
    curr = x
    for i in range(num_levels):
        coeffs = _haar_coeffs(curr)
        levels.append(coeffs)
        if i < num_levels - 1:
            curr = coeffs[:, :, 0, :, :]
    return levels


class IHaarCascadeFn(Function):
    """Inverse cascade; `add` may be None. Backward is the forward cascade."""

    @staticmethod
    def forward(ctx, output_size, add, *levels):
        ctx.num_levels = len(levels)
        ctx.has_add = add is not None
        ctx.level_shapes = [tuple(l.shape) for l in levels]
        return run_ihaar_cascade(list(levels), output_size, add)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        grads = run_haar_cascade(grad_output, ctx.num_levels)
        for g, shape in zip(grads, ctx.level_shapes):
            assert tuple(g.shape) == shape, \
                f"inverse-cascade gradient shape {tuple(g.shape)} != level shape {shape}"
        grad_add = grad_output if ctx.has_add else None
        return (None, grad_add, *grads)


def _ihaar(levels, add, output_size):
    if output_size is None:
        H2, W2 = levels[0].shape[3], levels[0].shape[4]
        output_size = (H2 * 2, W2 * 2)
    return IHaarCascadeFn.apply(output_size, add, *levels)


# Plain inverse cascade -------------------------------------------------------

def ihaar2d(x, output_size=None):
    return _ihaar([x], None, output_size)


def ihaar2d_double(l1, l2, output_size=None):
    return _ihaar([l1, l2], None, output_size)


def ihaar2d_triple(l1, l2, l3, output_size=None):
    return _ihaar([l1, l2, l3], None, output_size)


def ihaar2d_quad(l1, l2, l3, l4, output_size=None):
    return _ihaar([l1, l2, l3, l4], None, output_size)


def ihaar2d_quint(l1, l2, l3, l4, l5, output_size=None):
    return _ihaar([l1, l2, l3, l4, l5], None, output_size)


# Inverse cascade with the final add fused in ---------------------------------

def ihaar2d_fused(x, add_tensor, output_size=None):
    """ihaar(x) + add_tensor, in one kernel."""
    return _ihaar([x], add_tensor, output_size)


def ihaar2d_double_fused(l1, l2, add_tensor, output_size=None):
    return _ihaar([l1, l2], add_tensor, output_size)


def ihaar2d_triple_fused(l1, l2, l3, add_tensor, output_size=None):
    return _ihaar([l1, l2, l3], add_tensor, output_size)


def ihaar2d_quad_fused(l1, l2, l3, l4, add_tensor, output_size=None):
    return _ihaar([l1, l2, l3, l4], add_tensor, output_size)


def ihaar2d_quint_fused(l1, l2, l3, l4, l5, add_tensor, output_size=None):
    return _ihaar([l1, l2, l3, l4, l5], add_tensor, output_size)


# =============================================================================
# Plain forward Haar (utility / testing)
# =============================================================================

class HaarTransform(Function):
    """Single-level Haar transform. The transform is orthogonal, so its
    gradient is the inverse transform."""

    @staticmethod
    def forward(ctx, x):
        ctx.shape_hw = (x.shape[2], x.shape[3])
        return _haar_coeffs(x.contiguous())

    @staticmethod
    def backward(ctx, grad_output):
        return run_ihaar_cascade([grad_output.contiguous()], ctx.shape_hw)


def haar2d(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, C, 4, ceil(H/2), ceil(W/2)), subbands [LL, LH, HL, HH]."""
    return HaarTransform.apply(x)


# =============================================================================
# Scaled depthwise conv (base-conv path): scale folded into weight and bias
# =============================================================================

class ScaledDepthwiseConvFunction(Function):
    """
    y = scale * conv2d(x, weight, bias), computed as conv2d(x, scale*weight,
    scale*bias) so cuDNN handles both directions.
    """

    @staticmethod
    def forward(ctx, input, weight, scale, bias, padding, groups):
        scale_flat = scale.reshape(-1)
        fused_weight = scale_flat.view(-1, 1, 1, 1) * weight
        fused_bias = None if bias is None else scale_flat * bias
        output = F.conv2d(input, fused_weight, bias=fused_bias,
                          padding=padding, groups=groups)

        saved_bias = input.new_empty(0) if bias is None else bias
        ctx.save_for_backward(input, weight, scale, fused_weight, saved_bias)
        ctx.padding = padding
        ctx.groups = groups
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, scale, fused_weight, saved_bias = ctx.saved_tensors
        padding, groups = ctx.padding, ctx.groups

        grad_input = torch.nn.grad.conv2d_input(
            input.shape, fused_weight, grad_output, padding=padding, groups=groups
        )
        grad_fused_weight = torch.nn.grad.conv2d_weight(
            input, weight.shape, grad_output, padding=padding, groups=groups
        )

        # Unfuse. The forward folds the scale into the weight, W~ = s * W, so the
        # chain rule carries that factor back: dL/dW = s * dL/dW~. Dropping it
        # leaves grad_weight wrong by a per-channel factor of s (it only happens
        # to be right while s == 1, i.e. at initialisation).
        grad_weight = grad_fused_weight * scale.view(-1, 1, 1, 1)
        grad_scale = (grad_fused_weight * weight).sum(dim=(1, 2, 3))

        if ctx.has_bias:
            grad_fused_bias = grad_output.sum(dim=(0, 2, 3))
            grad_bias = scale.reshape(-1) * grad_fused_bias
            grad_scale = grad_scale + saved_bias * grad_fused_bias
        else:
            grad_bias = None

        grad_scale = grad_scale.reshape_as(scale)
        return grad_input, grad_weight, grad_scale, grad_bias, None, None


def scaled_depthwise_conv(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    padding: int = 1,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Scaled depthwise convolution: scale * depthwise_conv(input, weight, bias).

    Args:
        input: (B, C, H, W)
        weight: (C, 1, K, K)
        scale: (1, C, 1, 1)
        padding: usually kernel_size // 2
        bias: optional (C,), scaled along with the conv output
    """
    groups = input.size(1)
    return ScaledDepthwiseConvFunction.apply(input, weight, scale, bias, padding, groups)
