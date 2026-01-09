"""
WTConv2d - Wavelet Transform Convolution (JAX/Flax)

Pure JAX implementation using our optimized Haar transforms.
Uses Flax for nn.Module-like interface compatible with TPU.

NHWC format is native to Flax/TPU for optimal performance.

Performance optimized:
- Uses built-in nn.Conv with feature_group_count for depthwise convs
- XLA-friendly array update patterns
- Fused multi-level Haar forward for reduced memory bandwidth
"""

import jax
import jax.numpy as jnp
from jax import lax
from flax import linen as nn
from typing import Tuple, Sequence, Optional, List
import functools

from tpu_haar.haar import haar2d_multilevel_forward, haar2d_multilevel_inverse_add


class WTConv2d(nn.Module):
    """
    Wavelet Transform Convolution Layer (JAX/Flax).
    
    Uses fused multi-level Haar forward for efficient decomposition.
    Native NHWC format for optimal TPU performance.
    
    Args:
        channels: Number of input/output channels (must be equal)
        kernel_size: Convolution kernel size (default: 5)
        depth: Number of wavelet decomposition levels (1-5)
        use_bias: Include bias in base convolution (default: True)
        dtype: Data type for compute (default: None, inferred from input)
    
    Input: (B, H, W, C)
    Output: (B, H, W, C)
    """
    channels: int
    kernel_size: int = 5
    depth: int = 1
    use_bias: bool = True
    dtype: Optional[jnp.dtype] = None
    
    def setup(self):
        assert self.depth in [1, 2, 3, 4, 5], "depth must be 1-5"
        
        # Determine param dtype (use float32 for params, cast during forward)
        param_dtype = jnp.float32
        
        # Base convolution at full resolution - depthwise using feature_group_count
        self.base_conv = nn.Conv(
            features=self.channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding='SAME',
            feature_group_count=self.channels,  # Depthwise
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=param_dtype
        )
        
        # Wavelet-domain conv kernels defined directly as parameters
        # Shape for depthwise conv: (K, K, 1, C*4) in HWIO format
        # This allows fusing scale into kernel before convolution
        self.wavelet_kernels = [
            self.param(
                f'wavelet_kernel_{i}',
                nn.initializers.lecun_normal(),
                (self.kernel_size, self.kernel_size, 1, self.channels * 4)
            )
            for i in range(self.depth)
        ]
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, H, W, C = x.shape
        compute_dtype = self.dtype if self.dtype is not None else x.dtype
        
        # Learned scales - NHWC format: (1, 1, 1, C)
        base_scale = self.param('base_scale', nn.initializers.ones, (1, 1, 1, C))
        wavelet_scales = [
            self.param(f'wavelet_scale_{i}', 
                      lambda key, shape: jnp.ones(shape) * 0.1,
                      (1, 1, 1, C * 4))
            for i in range(self.depth)
        ]
        
        # Cast scales to compute dtype
        base_scale = base_scale.astype(compute_dtype)
        wavelet_scales = [s.astype(compute_dtype) for s in wavelet_scales]
        
        # Pad input if dimensions are odd
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = jnp.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        
        # Fused multi-level forward transform
        coeffs_all, dims = haar2d_multilevel_forward(x, self.depth)
        
        # Apply convolutions to all levels
        conv_coeffs = tuple(
            self._apply_conv(coeffs_all[level], level, wavelet_scales[level])
            for level in range(self.depth)
        )
        
        # Fused multi-level inverse with ADDITIVE reconstruction
        output_wt = haar2d_multilevel_inverse_add(conv_coeffs, dims, (H, W))
        
        # Base convolution (on original padded input, cropped to H, W)
        x_base = base_scale * self.base_conv(x)[:, :H, :W, :]
        
        return x_base + output_wt
    
    def _apply_conv(self, coeffs: jnp.ndarray, level: int, scale: jnp.ndarray) -> jnp.ndarray:
        """Apply scaled conv to coefficients with fused scale-weight multiplication.
        
        Fuses scale into convolution kernel before applying conv:
            y = conv(x, scale * kernel) instead of y = scale * conv(x, kernel)
        This is more XLA-friendly as it avoids a separate broadcast multiply.
        """
        # coeffs: (B, H2, W2, C, 4)
        B, h, w, C, _ = coeffs.shape
        flat = coeffs.reshape(B, h, w, C * 4)  # (B, H2, W2, C*4)
        
        # Get kernel and fuse with scale
        kernel = self.wavelet_kernels[level]  # (K, K, 1, C*4) in HWIO format
        # Cast kernel to match input dtype for lax.conv_general_dilated
        kernel = kernel.astype(flat.dtype)
        # scale is (1, 1, 1, C*4), kernel is (K, K, 1, C*4) - broadcast over spatial dims
        fused_kernel = scale * kernel
        
        # Apply depthwise conv with fused kernel
        out = lax.conv_general_dilated(
            flat,                                    # (B, H, W, C*4)
            fused_kernel,                            # (K, K, 1, C*4)
            window_strides=(1, 1),
            padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=C * 4                # Depthwise
        )
        
        return out.reshape(B, h, w, C, 4)