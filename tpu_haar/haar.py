"""
Haar 2D Wavelet Transform (JAX/TPU)

High-performance implementation optimized for TPU using JAX lax primitives.
- Forward: (B, H, W, C) -> (B, H/2, W/2, C, 4) where 4 = [LL, LH, HL, HH]
- Inverse: (B, H/2, W/2, C, 4) -> (B, H, W, C)
- Custom VJP for efficient gradients
- NHWC format (native to Flax/TPU)
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from typing import Tuple


# =============================================================================
# Forward Transform
# =============================================================================


@jax.jit
def haar2d_forward(x: jnp.ndarray) -> jnp.ndarray:
    """2D Haar forward: (B, H, W, C) -> (B, H//2, W//2, C, 4)"""
    B, H, W, C = x.shape
    H2, W2 = (H + 1) // 2, (W + 1) // 2
    
    # Pad if odd
    if H % 2 or W % 2:
        x = jnp.pad(x, ((0, 0), (0, H2 * 2 - H), (0, W2 * 2 - W), (0, 0)), mode='edge')
    
    # Split into 2x2 blocks
    rows = x.reshape(B, H2, 2, W2, 2, C)
    a, b = rows[:, :, 0, :, 0, :], rows[:, :, 0, :, 1, :]
    c, d = rows[:, :, 1, :, 0, :], rows[:, :, 1, :, 1, :]
    
    # Haar coefficients via lax primitives
    half = jnp.array(0.5, dtype=x.dtype)
    sum_ac, sum_bd = lax.add(a, c), lax.add(b, d)
    diff_ac, diff_bd = lax.sub(a, c), lax.sub(b, d)
    
    ll = lax.mul(half, lax.add(sum_ac, sum_bd))
    lh = lax.mul(half, lax.add(diff_ac, diff_bd))
    hl = lax.mul(half, lax.sub(sum_ac, sum_bd))
    hh = lax.mul(half, lax.sub(diff_ac, diff_bd))
    
    return jnp.stack([ll, lh, hl, hh], axis=-1)


@partial(jax.jit, static_argnums=(1, 2))
def haar2d_backward(g: jnp.ndarray, H: int, W: int) -> jnp.ndarray:
    """Backward for forward transform: (B, H2, W2, C, 4) -> (B, H, W, C)"""
    B, H2, W2, C, _ = g.shape
    g_ll, g_lh, g_hl, g_hh = g[..., 0], g[..., 1], g[..., 2], g[..., 3]
    
    half = jnp.array(0.5, dtype=g.dtype)
    grad_a = lax.mul(half, lax.add(lax.add(g_ll, g_lh), lax.add(g_hl, g_hh)))
    grad_b = lax.mul(half, lax.sub(lax.add(g_ll, g_lh), lax.add(g_hl, g_hh)))
    grad_c = lax.mul(half, lax.add(lax.sub(g_ll, g_lh), lax.sub(g_hl, g_hh)))
    grad_d = lax.mul(half, lax.sub(lax.sub(g_ll, g_lh), lax.sub(g_hl, g_hh)))
    
    row_even = jnp.stack([grad_a, grad_b], axis=3).reshape(B, H2, W2 * 2, C)
    row_odd = jnp.stack([grad_c, grad_d], axis=3).reshape(B, H2, W2 * 2, C)
    out = jnp.stack([row_even, row_odd], axis=2).reshape(B, H2 * 2, W2 * 2, C)
    
    return out[:, :H, :W, :]


# =============================================================================
# Inverse Transform
# =============================================================================


@partial(jax.jit, static_argnums=(1, 2))
def haar2d_inverse(coeffs: jnp.ndarray, H: int, W: int) -> jnp.ndarray:
    """2D Haar inverse: (B, H2, W2, C, 4) -> (B, H, W, C)"""
    B, H2, W2, C, _ = coeffs.shape
    ll, lh, hl, hh = coeffs[..., 0], coeffs[..., 1], coeffs[..., 2], coeffs[..., 3]
    
    half = jnp.array(0.5, dtype=coeffs.dtype)
    ll_lh_p, ll_lh_m = lax.add(ll, lh), lax.sub(ll, lh)
    hl_hh_p, hl_hh_m = lax.add(hl, hh), lax.sub(hl, hh)
    
    a = lax.mul(half, lax.add(ll_lh_p, hl_hh_p))
    b = lax.mul(half, lax.sub(ll_lh_p, hl_hh_p))
    c = lax.mul(half, lax.add(ll_lh_m, hl_hh_m))
    d = lax.mul(half, lax.sub(ll_lh_m, hl_hh_m))
    
    row_even = jnp.stack([a, b], axis=3).reshape(B, H2, W2 * 2, C)
    row_odd = jnp.stack([c, d], axis=3).reshape(B, H2, W2 * 2, C)
    out = jnp.stack([row_even, row_odd], axis=2).reshape(B, H2 * 2, W2 * 2, C)
    
    return lax.slice(out, (0, 0, 0, 0), (B, H, W, C)) if H2 * 2 > H or W2 * 2 > W else out


@jax.jit
def haar2d_inverse_backward(g: jnp.ndarray) -> jnp.ndarray:
    """Backward for inverse transform (essentially forward Haar)."""
    B, H, W, C = g.shape
    H2, W2 = (H + 1) // 2, (W + 1) // 2
    
    if H % 2 or W % 2:
        g = jnp.pad(g, ((0, 0), (0, H2 * 2 - H), (0, W2 * 2 - W), (0, 0)), mode='edge')
    
    blocks = g.reshape(B, H2, 2, W2, 2, C).transpose(0, 1, 3, 2, 4, 5)
    a, b = blocks[:, :, :, 0, 0, :], blocks[:, :, :, 0, 1, :]
    c, d = blocks[:, :, :, 1, 0, :], blocks[:, :, :, 1, 1, :]
    
    half = jnp.array(0.5, dtype=g.dtype)
    sum_ac, sum_bd = lax.add(a, c), lax.add(b, d)
    diff_ac, diff_bd = lax.sub(a, c), lax.sub(b, d)
    
    g_ll = lax.mul(half, lax.add(sum_ac, sum_bd))
    g_lh = lax.mul(half, lax.add(diff_ac, diff_bd))
    g_hl = lax.mul(half, lax.sub(sum_ac, sum_bd))
    g_hh = lax.mul(half, lax.sub(diff_ac, diff_bd))
    
    return jnp.stack([g_ll, g_lh, g_hl, g_hh], axis=-1)


# =============================================================================
# Custom VJP Wrappers
# =============================================================================


@jax.custom_vjp
def haar2d_transform(x: jnp.ndarray) -> jnp.ndarray:
    """Forward transform with custom VJP."""
    return haar2d_forward(x)

def _fwd(x):
    return haar2d_forward(x), (x.shape[1], x.shape[2])

def _bwd(res, g):
    return (haar2d_backward(g, res[0], res[1]),)

haar2d_transform.defvjp(_fwd, _bwd)


@jax.custom_vjp
def haar2d_inverse_transform(coeffs: jnp.ndarray, H: int, W: int) -> jnp.ndarray:
    """Inverse transform with custom VJP."""
    return haar2d_inverse(coeffs, H, W)

def _inv_fwd(coeffs, H, W):
    return haar2d_inverse(coeffs, H, W), (coeffs.shape, H, W)

def _inv_bwd(res, g):
    return (haar2d_inverse_backward(g), None, None)

haar2d_inverse_transform.defvjp(_inv_fwd, _inv_bwd)


# =============================================================================
# Multi-Level Forward
# =============================================================================


@partial(jax.jit, static_argnums=(1,))
def haar2d_multilevel_forward(x: jnp.ndarray, depth: int) -> Tuple[Tuple[jnp.ndarray, ...], Tuple[Tuple[int, int], ...]]:
    """
    Fused multi-level 2D Haar forward.
    
    Returns: (coeffs_tuple, dims_tuple) where dims_tuple has (h, w) per level.
    """
    B, H, W, C = x.shape
    
    if H % 2 or W % 2:
        x = jnp.pad(x, ((0, 0), (0, H % 2), (0, W % 2), (0, 0)), mode='edge')
    
    coeffs, dims = [], []
    current = x
    
    for _ in range(depth):
        h, w = current.shape[1], current.shape[2]
        dims.append((h, w))
        
        if h % 2 or w % 2:
            current = jnp.pad(current, ((0, 0), (0, h % 2), (0, w % 2), (0, 0)), mode='edge')
        
        # Inline core forward (lax primitives)
        B_, H_, W_, C_ = current.shape
        H2, W2 = H_ // 2, W_ // 2
        rows = current.reshape(B_, H2, 2, W2, 2, C_)
        a, b = rows[:, :, 0, :, 0, :], rows[:, :, 0, :, 1, :]
        c, d = rows[:, :, 1, :, 0, :], rows[:, :, 1, :, 1, :]
        
        half = jnp.array(0.5, dtype=current.dtype)
        sum_ac, sum_bd = lax.add(a, c), lax.add(b, d)
        diff_ac, diff_bd = lax.sub(a, c), lax.sub(b, d)
        
        level_coeffs = jnp.stack([
            lax.mul(half, lax.add(sum_ac, sum_bd)),
            lax.mul(half, lax.add(diff_ac, diff_bd)),
            lax.mul(half, lax.sub(sum_ac, sum_bd)),
            lax.mul(half, lax.sub(diff_ac, diff_bd))
        ], axis=-1)
        
        coeffs.append(level_coeffs)
        current = level_coeffs[:, :, :, :, 0]  # LL for next level
    
    return tuple(coeffs), tuple(dims)


@partial(jax.jit, static_argnums=(1,))
def haar2d_multilevel_backward(
    g_coeffs: Tuple[jnp.ndarray, ...], 
    original_shape: Tuple[int, int]
) -> jnp.ndarray:
    """
    Backward pass for haar2d_multilevel_forward.
    
    Args:
        g_coeffs: Tuple of gradients for each level's coefficients
        original_shape: (H, W) of the original input (must be static)
        
    Returns: Gradient w.r.t. input x of shape (B, H, W, C)
    """
    depth = len(g_coeffs)
    H_orig, W_orig = original_shape
    
    # Recompute dims from original shape (must match forward pass logic)
    dims = []
    h, w = H_orig, W_orig
    for _ in range(depth):
        dims.append((h, w))
        h = (h + 1) // 2
        w = (w + 1) // 2
    
    # Start from deepest level - gradient of LL at deepest level
    g_current = g_coeffs[depth - 1][..., 0]  # (B, H_d, W_d, C)
    
    # Cascade backwards from deepest to shallowest
    for level in range(depth - 1, -1, -1):
        g_level = g_coeffs[level]  # (B, H_l, W_l, C, 4)
        h, w = dims[level]
        
        # Add accumulated gradient to LL component
        if level < depth - 1:
            g_level = g_level.at[..., 0].add(g_current)
        
        # Apply single-level backward (inverse of forward)
        B, H2, W2, C, _ = g_level.shape
        g_ll, g_lh, g_hl, g_hh = g_level[..., 0], g_level[..., 1], g_level[..., 2], g_level[..., 3]
        
        half = jnp.array(0.5, dtype=g_level.dtype)
        grad_a = lax.mul(half, lax.add(lax.add(g_ll, g_lh), lax.add(g_hl, g_hh)))
        grad_b = lax.mul(half, lax.sub(lax.add(g_ll, g_lh), lax.add(g_hl, g_hh)))
        grad_c = lax.mul(half, lax.add(lax.sub(g_ll, g_lh), lax.sub(g_hl, g_hh)))
        grad_d = lax.mul(half, lax.sub(lax.sub(g_ll, g_lh), lax.sub(g_hl, g_hh)))
        
        # Interleave back to original resolution
        row_even = jnp.stack([grad_a, grad_b], axis=3).reshape(B, H2, W2 * 2, C)
        row_odd = jnp.stack([grad_c, grad_d], axis=3).reshape(B, H2, W2 * 2, C)
        g_current = jnp.stack([row_even, row_odd], axis=2).reshape(B, H2 * 2, W2 * 2, C)
        
        # Crop to original dimensions (static values work with regular slicing)
        g_current = g_current[:, :h, :w, :]
    
    return g_current


@partial(jax.jit, static_argnums=(2,))
def haar2d_multilevel_inverse(
    coeffs: Tuple[jnp.ndarray, ...], 
    dims: Tuple[Tuple[int, int], ...],
    output_shape: Tuple[int, int]
) -> jnp.ndarray:
    """
    Fused multi-level 2D Haar inverse (cascade reconstruction).
    
    Args:
        coeffs: Tuple of coefficients per level, each (B, H_l, W_l, C, 4)
        dims: Tuple of (h, w) original dimensions per level
        output_shape: (H, W) final output dimensions
        
    Returns: Reconstructed signal (B, H, W, C)
    """
    depth = len(coeffs)
    H, W = output_shape
    
    # Start from deepest level - apply inverse Haar
    level_coeffs = coeffs[depth - 1]
    B, H2, W2, C, _ = level_coeffs.shape
    ll, lh, hl, hh = level_coeffs[..., 0], level_coeffs[..., 1], level_coeffs[..., 2], level_coeffs[..., 3]
    
    half = jnp.array(0.5, dtype=level_coeffs.dtype)
    ll_lh_p, ll_lh_m = lax.add(ll, lh), lax.sub(ll, lh)
    hl_hh_p, hl_hh_m = lax.add(hl, hh), lax.sub(hl, hh)
    
    a = lax.mul(half, lax.add(ll_lh_p, hl_hh_p))
    b = lax.mul(half, lax.sub(ll_lh_p, hl_hh_p))
    c = lax.mul(half, lax.add(ll_lh_m, hl_hh_m))
    d = lax.mul(half, lax.sub(ll_lh_m, hl_hh_m))
    
    row_even = jnp.stack([a, b], axis=3).reshape(B, H2, W2 * 2, C)
    row_odd = jnp.stack([c, d], axis=3).reshape(B, H2, W2 * 2, C)
    current = jnp.stack([row_even, row_odd], axis=2).reshape(B, H2 * 2, W2 * 2, C)
    
    # Cascade from second deepest to shallowest
    for level in range(depth - 2, -1, -1):
        level_coeffs = coeffs[level]
        B, H2, W2, C, _ = level_coeffs.shape
        lh, hl, hh = level_coeffs[..., 1], level_coeffs[..., 2], level_coeffs[..., 3]
        
        # Use reconstructed signal from deeper level as LL (REPLACE, not add)
        # The forward transform used LL as input to the next level, so reconstruction gives it back
        ll = current[:, :H2, :W2, :]
        
        half = jnp.array(0.5, dtype=level_coeffs.dtype)
        ll_lh_p, ll_lh_m = lax.add(ll, lh), lax.sub(ll, lh)
        hl_hh_p, hl_hh_m = lax.add(hl, hh), lax.sub(hl, hh)
        
        a = lax.mul(half, lax.add(ll_lh_p, hl_hh_p))
        b = lax.mul(half, lax.sub(ll_lh_p, hl_hh_p))
        c = lax.mul(half, lax.add(ll_lh_m, hl_hh_m))
        d = lax.mul(half, lax.sub(ll_lh_m, hl_hh_m))
        
        row_even = jnp.stack([a, b], axis=3).reshape(B, H2, W2 * 2, C)
        row_odd = jnp.stack([c, d], axis=3).reshape(B, H2, W2 * 2, C)
        current = jnp.stack([row_even, row_odd], axis=2).reshape(B, H2 * 2, W2 * 2, C)
    
    # Final crop to output size
    return current[:, :H, :W, :]


@partial(jax.jit, static_argnums=(2,))
def haar2d_multilevel_inverse_add(
    coeffs: Tuple[jnp.ndarray, ...], 
    dims: Tuple[Tuple[int, int], ...],
    output_shape: Tuple[int, int]
) -> jnp.ndarray:
    """
    Multi-level 2D Haar inverse with ADDITIVE reconstruction.
    
    This variant ADDS the reconstruction from deeper levels to the LL component,
    which is required for WTConv architecture where contributions from all scales
    are accumulated.
    
    For perfect reconstruction (forward -> inverse = identity), use 
    haar2d_multilevel_inverse instead which REPLACES the LL.
    
    Args:
        coeffs: Tuple of coefficients per level, each (B, H_l, W_l, C, 4)
        dims: Tuple of (h, w) original dimensions per level
        output_shape: (H, W) final output dimensions
        
    Returns: Reconstructed signal (B, H, W, C)
    """
    depth = len(coeffs)
    H, W = output_shape
    
    # For depth=1, just return inverse of level 0
    if depth == 1:
        level_coeffs = coeffs[0]
        B, H2, W2, C, _ = level_coeffs.shape
        ll, lh, hl, hh = level_coeffs[..., 0], level_coeffs[..., 1], level_coeffs[..., 2], level_coeffs[..., 3]
        
        half = jnp.array(0.5, dtype=level_coeffs.dtype)
        ll_lh_p, ll_lh_m = lax.add(ll, lh), lax.sub(ll, lh)
        hl_hh_p, hl_hh_m = lax.add(hl, hh), lax.sub(hl, hh)
        
        a = lax.mul(half, lax.add(ll_lh_p, hl_hh_p))
        b = lax.mul(half, lax.sub(ll_lh_p, hl_hh_p))
        c = lax.mul(half, lax.add(ll_lh_m, hl_hh_m))
        d = lax.mul(half, lax.sub(ll_lh_m, hl_hh_m))
        
        row_even = jnp.stack([a, b], axis=3).reshape(B, H2, W2 * 2, C)
        row_odd = jnp.stack([c, d], axis=3).reshape(B, H2, W2 * 2, C)
        current = jnp.stack([row_even, row_odd], axis=2).reshape(B, H2 * 2, W2 * 2, C)
        return current[:, :H, :W, :]
    
    # Start from deepest level - apply inverse Haar
    level_coeffs = coeffs[depth - 1]
    B, H2, W2, C, _ = level_coeffs.shape
    ll, lh, hl, hh = level_coeffs[..., 0], level_coeffs[..., 1], level_coeffs[..., 2], level_coeffs[..., 3]
    
    half = jnp.array(0.5, dtype=level_coeffs.dtype)
    ll_lh_p, ll_lh_m = lax.add(ll, lh), lax.sub(ll, lh)
    hl_hh_p, hl_hh_m = lax.add(hl, hh), lax.sub(hl, hh)
    
    a = lax.mul(half, lax.add(ll_lh_p, hl_hh_p))
    b = lax.mul(half, lax.sub(ll_lh_p, hl_hh_p))
    c = lax.mul(half, lax.add(ll_lh_m, hl_hh_m))
    d = lax.mul(half, lax.sub(ll_lh_m, hl_hh_m))
    
    row_even = jnp.stack([a, b], axis=3).reshape(B, H2, W2 * 2, C)
    row_odd = jnp.stack([c, d], axis=3).reshape(B, H2, W2 * 2, C)
    current = jnp.stack([row_even, row_odd], axis=2).reshape(B, H2 * 2, W2 * 2, C)
    
    # Cascade from second deepest to shallowest
    for level in range(depth - 2, -1, -1):
        level_coeffs = coeffs[level]
        B, H2, W2, C, _ = level_coeffs.shape
        lh, hl, hh = level_coeffs[..., 1], level_coeffs[..., 2], level_coeffs[..., 3]
        
        # ADD reconstruction from deeper level to current level's LL
        # This accumulates contributions from all frequency bands
        ll = level_coeffs[..., 0] + current[:, :H2, :W2, :]
        
        half = jnp.array(0.5, dtype=level_coeffs.dtype)
        ll_lh_p, ll_lh_m = lax.add(ll, lh), lax.sub(ll, lh)
        hl_hh_p, hl_hh_m = lax.add(hl, hh), lax.sub(hl, hh)
        
        a = lax.mul(half, lax.add(ll_lh_p, hl_hh_p))
        b = lax.mul(half, lax.sub(ll_lh_p, hl_hh_p))
        c = lax.mul(half, lax.add(ll_lh_m, hl_hh_m))
        d = lax.mul(half, lax.sub(ll_lh_m, hl_hh_m))
        
        row_even = jnp.stack([a, b], axis=3).reshape(B, H2, W2 * 2, C)
        row_odd = jnp.stack([c, d], axis=3).reshape(B, H2, W2 * 2, C)
        current = jnp.stack([row_even, row_odd], axis=2).reshape(B, H2 * 2, W2 * 2, C)
    
    # Final crop to output size
    return current[:, :H, :W, :]


@partial(jax.jit, static_argnums=(1,))
def haar2d_multilevel_inverse_backward(
    g: jnp.ndarray, 
    depth: int
) -> Tuple[jnp.ndarray, ...]:
    """
    Backward pass for haar2d_multilevel_inverse.
    
    Args:
        g: Gradient w.r.t. output (B, H, W, C)
        depth: Number of decomposition levels
        
    Returns: Tuple of gradients for each level's coefficients
    """
    B, H, W, C = g.shape
    
    # Pad if odd
    if H % 2 or W % 2:
        g = jnp.pad(g, ((0, 0), (0, H % 2), (0, W % 2), (0, 0)), mode='edge')
    
    g_coeffs = []
    current = g
    
    # Forward through levels (like forward transform for gradient)
    for level in range(depth):
        h, w = current.shape[1], current.shape[2]
        
        if h % 2 or w % 2:
            current = jnp.pad(current, ((0, 0), (0, h % 2), (0, w % 2), (0, 0)), mode='edge')
        
        B_, H_, W_, C_ = current.shape
        H2, W2 = H_ // 2, W_ // 2
        rows = current.reshape(B_, H2, 2, W2, 2, C_)
        a, b = rows[:, :, 0, :, 0, :], rows[:, :, 0, :, 1, :]
        c, d = rows[:, :, 1, :, 0, :], rows[:, :, 1, :, 1, :]
        
        half = jnp.array(0.5, dtype=current.dtype)
        sum_ac, sum_bd = lax.add(a, c), lax.add(b, d)
        diff_ac, diff_bd = lax.sub(a, c), lax.sub(b, d)
        
        g_level = jnp.stack([
            lax.mul(half, lax.add(sum_ac, sum_bd)),
            lax.mul(half, lax.add(diff_ac, diff_bd)),
            lax.mul(half, lax.sub(sum_ac, sum_bd)),
            lax.mul(half, lax.sub(diff_ac, diff_bd))
        ], axis=-1)
        
        g_coeffs.append(g_level)
        current = g_level[:, :, :, :, 0]  # LL for next level
    
    return tuple(g_coeffs)


# =============================================================================
# Custom VJP for Multilevel Transforms
# =============================================================================


@jax.custom_vjp
def haar2d_multilevel_transform(x: jnp.ndarray, depth: int) -> Tuple[Tuple[jnp.ndarray, ...], Tuple[Tuple[int, int], ...]]:
    """Multilevel forward transform with custom VJP."""
    return haar2d_multilevel_forward(x, depth)


def _multilevel_fwd(x, depth):
    coeffs, dims = haar2d_multilevel_forward(x, depth)
    # Store original H, W as residuals (these are concrete Python ints)
    return (coeffs, dims), (x.shape[1], x.shape[2])


def _multilevel_bwd(res, g):
    H, W = res  # Static original dimensions
    g_coeffs, _ = g  # Unpack gradient tuple
    grad_x = haar2d_multilevel_backward(g_coeffs, (H, W))
    return (grad_x, None)


haar2d_multilevel_transform.defvjp(_multilevel_fwd, _multilevel_bwd)


@jax.custom_vjp
def haar2d_multilevel_inverse_transform(
    coeffs: Tuple[jnp.ndarray, ...], 
    dims: Tuple[Tuple[int, int], ...],
    output_shape: Tuple[int, int]
) -> jnp.ndarray:
    """Multilevel inverse transform with custom VJP."""
    return haar2d_multilevel_inverse(coeffs, dims, output_shape)


def _multilevel_inv_fwd(coeffs, dims, output_shape):
    out = haar2d_multilevel_inverse(coeffs, dims, output_shape)
    return out, (len(coeffs), dims)


def _multilevel_inv_bwd(res, g):
    depth, dims = res
    g_coeffs = haar2d_multilevel_inverse_backward(g, depth)
    return (g_coeffs, None, None)


haar2d_multilevel_inverse_transform.defvjp(_multilevel_inv_fwd, _multilevel_inv_bwd)


# =============================================================================
# Utilities
# =============================================================================


def get_output_shape(input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """(B, H, W, C) -> (B, H//2, W//2, C, 4)"""
    B, H, W, C = input_shape
    return (B, (H + 1) // 2, (W + 1) // 2, C, 4)


def haar2d_forward_single_channel(x: jnp.ndarray) -> jnp.ndarray:
    """(H, W) -> (H//2, W//2, 4)"""
    return haar2d_forward(x[None, :, :, None])[0, :, :, 0, :]


def haar2d_inverse_single_channel(coeffs: jnp.ndarray, H: int, W: int) -> jnp.ndarray:
    """(H2, W2, 4) -> (H, W)"""
    return haar2d_inverse(coeffs[None, :, :, None, :], H, W)[0, :, :, 0]


def create_haar2d_pmap(axis_name: str = 'device'):
    """Create pmapped forward transform."""
    return jax.pmap(haar2d_forward, axis_name=axis_name)


def create_haar2d_inverse_pmap(H: int, W: int, axis_name: str = 'device'):
    """Create pmapped inverse transform."""
    return jax.pmap(lambda c: haar2d_inverse(c, H, W), axis_name=axis_name)


def verify_reconstruction(x: jnp.ndarray, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
    """Verify forward -> inverse gives perfect reconstruction."""
    B, H, W, C = x.shape
    return jnp.allclose(x, haar2d_inverse(haar2d_forward(x), H, W), rtol=rtol, atol=atol)
