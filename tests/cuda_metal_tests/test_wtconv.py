#!/usr/bin/env python3
"""
WTConv Test Suite - Benchmarks WTConv Implementations
=======================================================

Compares:
- WTConv2d: Fused cascade kernels for BOTH forward and inverse
- Naive: PyWavelets-based reference

Usage:
    python test_wtconv.py                    # Run correctness + benchmark (auto-detect device)
    python test_wtconv.py --device mps       # Test on MPS (Metal)
    python test_wtconv.py --device cuda      # Test on CUDA
    python test_wtconv.py --dtype fp16       # Test with fp16
    python test_wtconv.py --bench-only       # Skip correctness, just benchmark
    python test_wtconv.py --compile          # Enable torch.compile optimization
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

# Suppress torch.compile warnings about graph breaks on custom CUDA ops
warnings.filterwarnings('ignore', message='.*Graph break.*')
warnings.filterwarnings('ignore', message='.*Unsupported builtin.*')

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Lazy imports - avoid triggering CUDA/Metal compilation at module level
_WTConv2d = None
_WTConv2dNaive = None


def get_wtconv_classes():
    """Lazy import WTConv classes to avoid premature CUDA/Metal compilation."""
    global _WTConv2d, _WTConv2dNaive
    
    if _WTConv2d is None:
        from wtconv_model.wtconv import WTConv2d
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
    elif device == 'mps':
        torch.mps.synchronize()


def copy_weights_to_naive(model, naive, depth):
    """Copy weights from WTConv model to naive model."""
    with torch.no_grad():
        naive.base_conv.weight.copy_(model.base_conv.weight)
        if model.base_conv.bias is not None and naive.base_conv.bias is not None:
            naive.base_conv.bias.copy_(model.base_conv.bias)
        naive.base_scale.weight.copy_(model.base_scale)
        for level in range(depth):
            naive.wavelet_convs[level].weight.copy_(model.wavelet_convs[level].weight)
            naive.wavelet_scale[level].weight.copy_(model.wavelet_scales[level])


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
        v2 = WTConv2d(C, C, kernel_size=k, wt_levels=depth, device=device).to(device).to(dtype)
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
        
        v2 = WTConv2d(C, C, kernel_size=k, wt_levels=depth, device=device).to(device).to(dtype)
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


def run_benchmark(device: str, dtype=torch.float32, iterations=50, use_compile=False):
    """Benchmark: My WTConv vs Naive."""
    print(f"\n{'='*90}")
    print(f"Benchmark: Forward + Backward [{dtype}] on {device.upper()}")
    
    if device == 'cuda':
        print(f"Device: {torch.cuda.get_device_name()}")
    else:
        print(f"Device: Apple MPS")
    
    if use_compile:
        print("torch.compile optimization: ENABLED")
    print(f"{'='*90}")
    
    configs = [(16, 32, 256, 256), (16, 32, 512, 512)]
    
    for depth in [1, 2, 3, 4, 5]:
        print(f"\n--- Depth = {depth} ---")
        print(f"{'Config':<22} {'My WTConv':<12} {'Naive':<12} {'My WTConv vs Naive':<12}")
        print("-" * 90)
        
        for B, C, H, W in configs:
            results = {}
            
            WTConv2d, WTConv2dNaive = get_wtconv_classes()
            for name, Model in [("v2", WTConv2d)]:
                x = torch.randn(B, C, H, W, device=device, dtype=dtype, requires_grad=True)
                model = Model(C, C, kernel_size=3, wt_levels=depth, device=device).to(device).to(dtype)
                
                # Enable torch.compile if requested
                if use_compile:
                    model = torch.compile(model)
                
                # Warmup
                for _ in range(5):
                    model(x).sum().backward()
                    x.grad = None
                    model.zero_grad()
                sync_device(device)
                
                # Benchmark
                if device == 'cuda':
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(iterations):
                        model(x).sum().backward()
                        x.grad = None
                        model.zero_grad()
                    end.record()
                    torch.cuda.synchronize()
                    results[name] = start.elapsed_time(end) / iterations
                else:  # mps
                    sync_device(device)
                    start_time = time.perf_counter()
                    for _ in range(iterations):
                        model(x).sum().backward()
                        x.grad = None
                        model.zero_grad()
                    sync_device(device)
                    results[name] = (time.perf_counter() - start_time) * 1000 / iterations
            
            # Naive
            x = torch.randn(B, C, H, W, device=device, dtype=dtype, requires_grad=True)
            naive = WTConv2dNaive(C, C, kernel_size=3, wt_levels=depth).to(device).to(dtype)
            if use_compile:
                naive = torch.compile(naive)
            for _ in range(5):
                naive(x).sum().backward()
                x.grad = None
                naive.zero_grad()
            sync_device(device)
            
            if device == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    naive(x).sum().backward()
                    x.grad = None
                    naive.zero_grad()
                end.record()
                torch.cuda.synchronize()
                results['naive'] = start.elapsed_time(end) / iterations
            else:  # mps
                sync_device(device)
                start_time = time.perf_counter()
                for _ in range(iterations):
                    naive(x).sum().backward()
                    x.grad = None
                    naive.zero_grad()
                sync_device(device)
                results['naive'] = (time.perf_counter() - start_time) * 1000 / iterations
            
            speedup_vs_naive = results['naive'] / results['v2']
            
            print(f"{B}x{C}x{H}x{W:<14} {results['v2']:>8.3f}ms "
                  f"{results['naive']:>8.3f}ms "
                  f"{speedup_vs_naive:>10.2f}x")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="WTConv Test Suite")
    parser.add_argument("--device", choices=['cuda', 'mps'], default=None,
                        help="Device to use: cuda or mps (auto-detected if not specified)")
    parser.add_argument("--dtype", choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument("--bench-only", action="store_true", help="Skip correctness tests")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile optimization for WTConv2d")
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            print("ERROR: No GPU available (need CUDA or MPS)")
            return
    else:
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            print("ERROR: CUDA is not available")
            return
        if device == 'mps' and not torch.backends.mps.is_available():
            print("ERROR: MPS is not available")
            return
    
    dtype = DTYPE_MAP[args.dtype]
    
    # bf16 not supported on MPS
    if device == 'mps' and dtype == torch.bfloat16:
        print("WARNING: bfloat16 not supported on MPS, using float16")
        dtype = torch.float16
    
    print(f"Using device: {device.upper()}")
    
    if not args.bench_only:
        passed_fwd = test_correctness(device, dtype)
        passed_bwd = test_backward(device, dtype)
        
        if not (passed_fwd and passed_bwd):
            print("\n✗ Some tests failed!")
            exit(1)
        else:
            print("\n✓ All correctness tests passed!")
    
    run_benchmark(device, dtype, args.iterations, args.compile)


if __name__ == "__main__":
    main()
