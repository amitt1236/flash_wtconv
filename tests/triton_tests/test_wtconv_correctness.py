#!/usr/bin/env python3
"""
WTConv Test Suite - Benchmarks WTConv Implementations
=======================================================

Compares:
- WTConv2d: Triton-based WTConv2d
- Naive: PyWavelets-based reference
"""

from pathlib import Path
import argparse
import torch
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Lazy imports - avoid triggering CUDA compilation at module level
_WTConv2d = None
_WTConv2dNaive = None

def get_wtconv_classes():
    """Lazy import WTConv classes to avoid premature CUDA compilation."""
    global _WTConv2d, _WTConv2dNaive

    
    if _WTConv2d is None:
        from wtconv_model.wtconv_triton import WTConv2d
        _WTConv2d = WTConv2d

    if _WTConv2dNaive is None:
        from WTConv.wtconv.wtconv2d import WTConv2d as WTConv2dNaive
        _WTConv2dNaive = WTConv2dNaive
    
    return _WTConv2d, _WTConv2dNaive


DTYPE_MAP = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
TOLERANCE_MAP = {torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 2e-2}


def sync_device(device: str):
    """Synchronize the device."""
    if device == 'cuda':
        torch.cuda.synchronize()


def copy_weights_to_naive(model, naive, depth):
    """Copy weights from WTConv model to naive model.
    
    Handles WTConv2dHybrid's weight structure:
    - Level 1: wt_weight1 (C*4, 1, K, K), wt_scale1 (1, C*4, 1, 1)
    - Levels 2+: wavelet_convs[i-1].weight, wavelet_scales[i-1]
    """
    with torch.no_grad():
        naive.base_conv.weight.copy_(model.base_weight)
        if model.base_bias is not None and naive.base_conv.bias is not None:
            naive.base_conv.bias.copy_(model.base_bias)
        naive.base_scale.weight.copy_(model.base_scale)
        for level in range(depth):
            # WTConv2d now stores all levels in wt_weights/wt_scales ParameterLists
            naive.wavelet_convs[level].weight.copy_(model.wt_weights[level])
            naive.wavelet_scale[level].weight.copy_(model.wt_scales[level])


def test_correctness(device: str, dtype=torch.float32):
    """Test WTConv2d correctness against naive."""
    WTConv2d, WTConv2dNaive = get_wtconv_classes()
    
    print(f"\n{'='*70}")
    print(f"Correctness: WTConv2d vs Naive [{dtype}] on {device.upper()}")
    print(f"{'='*70}")
    
    tolerance = TOLERANCE_MAP[dtype]
    B, C, H, W, k = 2, 16, 64, 64, 3
    all_passed = True
    
    for depth in [1, 2, 3, 4, 5]:
        torch.manual_seed(42)
        x = torch.randn(B, C, H, W, device=device, dtype=dtype)
        
        # My WTConv model
        v2 = WTConv2d(C, C, kernel_size=k, wt_levels=depth).to(device).to(dtype)
        naive = WTConv2dNaive(C, C, kernel_size=k, wt_levels=depth).to(device).to(dtype)
        copy_weights_to_naive(v2, naive, depth)
        
        out_v2 = v2(x)
        out_naive = naive(x)
        
        max_diff = (out_v2 - out_naive).abs().max().item()
        passed = max_diff < tolerance
        status = "✓" if passed else "✗"
        print(f"  Depth {depth}: max_diff={max_diff:.2e} (tol={tolerance:.0e}) {status}")
        all_passed &= passed
    
    return all_passed


def test_backward(device: str, dtype=torch.float32):
    """Test WTConv2d backward pass against naive."""
    WTConv2d, WTConv2dNaive = get_wtconv_classes()
    
    print(f"\n{'='*70}")
    print(f"Backward: WTConv2d vs Naive [{dtype}] on {device.upper()}")
    print(f"{'='*70}")
    
    tolerance = TOLERANCE_MAP[dtype]
    B, C, H, W, k = 2, 16, 64, 64, 3
    all_passed = True
    
    for depth in [1, 2, 3, 4, 5]:
        torch.manual_seed(42)
        x = torch.randn(B, C, H, W, device=device, dtype=dtype, requires_grad=True)
        x_naive = x.clone().detach().requires_grad_(True)
        
        v2 = WTConv2d(C, C, kernel_size=k, wt_levels=depth).to(device).to(dtype)
        naive = WTConv2dNaive(C, C, kernel_size=k, wt_levels=depth).to(device).to(dtype)
        copy_weights_to_naive(v2, naive, depth)
        
        v2(x).sum().backward()
        naive(x_naive).sum().backward()
        
        input_grad_diff = (x.grad - x_naive.grad).abs().max().item()
        passed = input_grad_diff < tolerance
        status = "✓" if passed else "✗"
        print(f"  Depth {depth}: grad_diff={input_grad_diff:.2e} (tol={tolerance:.0e}) {status}")
        all_passed &= passed
    
    return all_passed

def main():
    parser = argparse.ArgumentParser(description="WTConv Test Suite")
    parser.add_argument("--dtype", choices=['fp32', 'fp16', 'bf16'], default='fp32')

    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile optimization for WTConv2d")
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        return
        
    device = 'cuda'
    dtype = DTYPE_MAP[args.dtype]
    
    print(f"Using device: {device.upper()}")
    
    passed_fwd = test_correctness(device, dtype)
    passed_bwd = test_backward(device, dtype)
    
    if not (passed_fwd and passed_bwd):
        print("\n✗ Some tests failed!")
        exit(1)
    else:
        print("\n✓ All correctness tests passed!")


if __name__ == "__main__":
    main()
