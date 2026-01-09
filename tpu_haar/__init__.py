"""
TPU-optimized Haar Wavelet Transform Package

High-performance JAX implementations of 2D Haar wavelet transforms for TPU.
"""

from .haar import (
    haar2d_forward,
    haar2d_backward,
    haar2d_transform,
    haar2d_inverse,
    haar2d_inverse_backward,
    haar2d_inverse_transform,
    haar2d_multilevel_forward,
    haar2d_multilevel_backward,
    haar2d_multilevel_inverse,
    haar2d_multilevel_inverse_add,
    haar2d_multilevel_inverse_backward,
    haar2d_multilevel_transform,
    haar2d_multilevel_inverse_transform,
    haar2d_forward_single_channel,
    haar2d_inverse_single_channel,
    create_haar2d_pmap,
    create_haar2d_inverse_pmap,
    get_output_shape,
    verify_reconstruction,
)

__all__ = [
    'haar2d_forward',
    'haar2d_backward',
    'haar2d_transform',
    'haar2d_inverse',
    'haar2d_inverse_backward',
    'haar2d_inverse_transform',
    'haar2d_multilevel_forward',
    'haar2d_multilevel_backward',
    'haar2d_multilevel_inverse',
    'haar2d_multilevel_inverse_add',
    'haar2d_multilevel_inverse_backward',
    'haar2d_multilevel_transform',
    'haar2d_multilevel_inverse_transform',
    'haar2d_forward_single_channel',
    'haar2d_inverse_single_channel',
    'create_haar2d_pmap',
    'create_haar2d_inverse_pmap',
    'get_output_shape',
    'verify_reconstruction',
    'WTConv2d',
]
