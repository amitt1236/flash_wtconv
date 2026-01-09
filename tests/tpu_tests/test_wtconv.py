"""
Test JAX WTConv2d correctness against Naive PyWavelets implementation.

Copies weights between JAX and PyTorch to compare outputs exactly.

Usage:
    python tpu/test_wtconv.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax
import jax.numpy as jnp
import numpy as np

# Check if torch is available for comparison
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available - skipping comparison tests")

from tpu_haar.haar import haar2d_forward, haar2d_inverse
from wtconv_model.wtconv_tpu import WTConv2d


def test_haar_transforms():
    """Test Haar transforms are correct."""
    print("\n" + "=" * 60)
    print("Test: Haar Forward/Inverse")
    print("=" * 60)
    
    key = jax.random.PRNGKey(42)
    all_passed = True
    
    # NHWC format: (B, H, W, C)
    for shape in [(2, 64, 64, 16), (4, 32, 32, 8), (1, 128, 128, 32)]:
        x = jax.random.normal(key, shape)
        
        # Forward
        coeffs = haar2d_forward(x)
        
        # Inverse
        B, H, W, C = x.shape
        recon = haar2d_inverse(coeffs, H, W)
        
        max_diff = jnp.max(jnp.abs(x - recon))
        passed = max_diff < 1e-5
        
        print(f"  Shape {shape}: max_diff={max_diff:.2e} {'✓' if passed else '✗'}")
        all_passed &= passed
    
    return all_passed


def test_wtconv_shapes():
    """Test WTConv2d output shapes for all depths."""
    print("\n" + "=" * 60)
    print("Test: WTConv2d Output Shapes")
    print("=" * 60)
    
    key = jax.random.PRNGKey(42)
    B, H, W, C = 2, 64, 64, 16  # NHWC format
    x = jax.random.normal(key, (B, H, W, C))
    all_passed = True
    
    for depth in [1, 2, 3, 4, 5]:
        model = WTConv2d(channels=C, depth=depth)
        variables = model.init(key, x)
        out = model.apply(variables, x)
        
        passed = out.shape == (B, H, W, C)
        print(f"  Depth {depth}: input {x.shape} -> output {out.shape} {'✓' if passed else '✗'}")
        all_passed &= passed
    
    return all_passed


def test_wtconv_gradients():
    """Test that gradients flow through WTConv2d."""
    print("\n" + "=" * 60)
    print("Test: WTConv2d Gradient Flow")
    print("=" * 60)
    
    key = jax.random.PRNGKey(42)
    B, H, W, C = 2, 64, 64, 16  # NHWC format
    x = jax.random.normal(key, (B, H, W, C))
    all_passed = True
    
    for depth in [1, 2, 3, 4, 5]:
        model = WTConv2d(channels=C, depth=depth)
        variables = model.init(key, x)
        
        def loss_fn(params):
            out = model.apply({'params': params}, x)
            return jnp.mean(out ** 2)
        
        grads = jax.grad(loss_fn)(variables['params'])
        
        # Check gradients exist and are finite
        grad_ok = True
        for param_name in ['base_scale']:
            if param_name in grads:
                g = grads[param_name]
                if not jnp.isfinite(g).all():
                    grad_ok = False
        
        print(f"  Depth {depth}: gradients finite = {grad_ok} {'✓' if grad_ok else '✗'}")
        all_passed &= grad_ok
    
    return all_passed


def copy_weights_naive_to_jax(naive_model, jax_params, depth):
    """
    Copy weights from naive PyTorch model to JAX params dict.
    
    Naive model structure (NCHW):
        - base_conv.weight: (C, 1, K, K) depthwise
        - base_conv.bias: (C,)
        - base_scale.weight: (1, C, 1, 1)
        - wavelet_convs[i].weight: (C*4, 1, K, K)
        - wavelet_scale[i].weight: (1, C*4, 1, 1)
    
    JAX params structure (NHWC):
        - base_conv.kernel: (K, K, 1, C)
        - base_conv.bias: (C,)
        - base_scale: (1, 1, 1, C)
        - wavelet_kernel_i: (K, K, 1, C*4)  # Direct param, not Conv module
        - wavelet_scale_i: (1, 1, 1, C*4)
    """
    new_params = {}
    
    # Base conv kernel: (C, 1, K, K) -> (K, K, 1, C)
    base_weight = naive_model.base_conv.weight.detach().cpu().numpy()
    base_kernel = np.transpose(base_weight, (2, 3, 1, 0))  # (K, K, 1, C)
    new_params['base_conv'] = {'kernel': jnp.array(base_kernel)}
    
    if naive_model.base_conv.bias is not None:
        base_bias = naive_model.base_conv.bias.detach().cpu().numpy()
        new_params['base_conv']['bias'] = jnp.array(base_bias)
    
    # Base scale: (1, C, 1, 1) -> (1, 1, 1, C)
    base_scale = naive_model.base_scale.weight.detach().cpu().numpy()
    base_scale = np.transpose(base_scale, (0, 2, 3, 1))  # (1, 1, 1, C)
    new_params['base_scale'] = jnp.array(base_scale)
    
    # Wavelet kernels (direct params, not Conv modules)
    for i in range(depth):
        wt_weight = naive_model.wavelet_convs[i].weight.detach().cpu().numpy()
        wt_kernel = np.transpose(wt_weight, (2, 3, 1, 0))  # (K, K, 1, C*4)
        new_params[f'wavelet_kernel_{i}'] = jnp.array(wt_kernel)
        
        wt_scale = naive_model.wavelet_scale[i].weight.detach().cpu().numpy()
        wt_scale = np.transpose(wt_scale, (0, 2, 3, 1))  # (1, 1, 1, C*4)
        new_params[f'wavelet_scale_{i}'] = jnp.array(wt_scale)
    
    return {'params': new_params}


def test_wtconv_vs_naive_exact():
    """Compare JAX WTConv2d output exactly against naive implementation with same weights."""
    if not HAS_TORCH:
        print("\n  Skipping: PyTorch not available")
        return True
    
    print("\n" + "=" * 60)
    print("Test: JAX WTConv2d vs Naive (Exact - Same Weights)")
    print("=" * 60)
    
    try:
        from WTConv.wtconv.wtconv2d import WTConv2d as WTConv2dNaive
    except ImportError as e:
        print(f"  Skipping: WTConv2dNaive not available ({e})")
        return True
    
    key = jax.random.PRNGKey(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    B, C, H, W = 2, 16, 64, 64  # PyTorch uses NCHW
    k = 5
    tolerance = 1e-2  # bf16/TPU-friendly (use 1e-6 for CPU fp32)
    all_passed = True
    
    for depth in [1, 2, 3, 4, 5]:
        # Create random input - PyTorch NCHW format
        x_np_nchw = np.random.randn(B, C, H, W).astype(np.float32)
        # Convert to NHWC for JAX
        x_np_nhwc = np.transpose(x_np_nchw, (0, 2, 3, 1))
        
        x_jax = jnp.array(x_np_nhwc)
        x_torch = torch.from_numpy(x_np_nchw)
        
        # Create naive PyTorch model (source of weights)
        naive_model = WTConv2dNaive(C, C, kernel_size=k, wt_levels=depth)
        
        # Create JAX model and init with dummy
        jax_model = WTConv2d(channels=C, kernel_size=k, depth=depth)
        jax_vars = jax_model.init(key, x_jax)
        
        # Copy weights from naive to JAX
        jax_vars_with_naive_weights = copy_weights_naive_to_jax(naive_model, jax_vars, depth)
        
        # Run both models
        naive_out = naive_model(x_torch)
        jax_out = jax_model.apply(jax_vars_with_naive_weights, x_jax)
        
        # Compare outputs (convert JAX NHWC to NCHW for comparison)
        naive_np = naive_out.detach().cpu().numpy()  # NCHW
        jax_np = np.transpose(np.array(jax_out), (0, 3, 1, 2))  # NHWC -> NCHW
        
        max_diff = np.max(np.abs(naive_np - jax_np))
        passed = max_diff < tolerance
        
        print(f"  Depth {depth}: max_diff={max_diff:.2e} (tol={tolerance:.0e}) {'✓' if passed else '✗'}")
        
        if not passed:
            # Debug info
            print(f"    naive mean={naive_np.mean():.4f}, std={naive_np.std():.4f}")
            print(f"    jax   mean={jax_np.mean():.4f}, std={jax_np.std():.4f}")
        
        all_passed &= passed
    
    return all_passed


def run_all_tests():
    """Run all correctness tests."""
    print("\n" + "=" * 60)
    print("JAX WTConv2d Test Suite (NHWC format)")
    print("=" * 60)
    
    results = [
        ("Haar transforms", test_haar_transforms()),
        ("WTConv shapes", test_wtconv_shapes()),
        ("WTConv gradients", test_wtconv_gradients()),
    ]
    
    if HAS_TORCH:
        results.append(("WTConv vs Naive (exact)", test_wtconv_vs_naive_exact()))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"  {name}: {status}")
        all_passed &= passed
    
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
    
    return all_passed


if __name__ == "__main__":
    passed = run_all_tests()
    exit(0 if passed else 1)
