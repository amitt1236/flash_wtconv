"""
Benchmark: WTConv2d vs Regular Convolution vs Depthwise Convolution

Compares performance across different:
- Input sizes (spatial dimensions)
- Channel counts
- Wavelet levels (for WTConv2d)

Supports both CUDA and MPS (Metal) backends.
"""

import sys
import warnings
import time
from pathlib import Path

# Suppress torch.compile warnings about graph breaks and cache limits
warnings.filterwarnings('ignore', message='.*Graph break.*')
warnings.filterwarnings('ignore', message='.*Unsupported builtin.*')
warnings.filterwarnings('ignore', message='.*cache_size_limit.*')

import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64  # Increase cache for many model configurations
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict
import argparse

# Kernel size constants for each convolution type
DEPTHWISE_KERNEL_SIZE = 7
REGULAR_KERNEL_SIZE = 7
WTCONV_KERNEL_SIZE = 3

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from wtconv_model.wtconv import WTConv2d
from WTConv.wtconv.wtconv2d import WTConv2d as WTConv2dNaive


@dataclass
class BenchmarkConfig:
    batch_size: int = 8
    warmup_iters: int = 10
    benchmark_iters: int = 100
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    use_compile: bool = False


def sync_device(device: str):
    """Synchronize the device."""
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


def create_depthwise_conv(channels: int, kernel_size: int = DEPTHWISE_KERNEL_SIZE) -> nn.Module:
    """Create a depthwise convolution layer."""
    return nn.Conv2d(
        channels, channels, kernel_size,
        padding='same', stride=1, dilation=1,
        groups=channels, bias=True
    )


def create_regular_conv(channels: int, kernel_size: int = REGULAR_KERNEL_SIZE) -> nn.Module:
    """Create a regular (dense) convolution layer."""
    return nn.Conv2d(
        channels, channels, kernel_size,
        padding='same', stride=1, dilation=1,
        groups=1, bias=True
    )


def benchmark_layer(
    layer: nn.Module, 
    x: torch.Tensor, 
    config: BenchmarkConfig,
    include_backward: bool = True
) -> Dict[str, float]:
    """
    Benchmark a layer's forward and backward pass.
    
    Returns:
        Dict with 'forward_ms', 'backward_ms', 'total_ms'
    """
    layer = layer.to(config.device)
    x = x.to(config.device).requires_grad_(True)
    
    # Enable torch.compile if requested
    if config.use_compile:
        layer = torch.compile(layer)
    
    # Warmup
    for _ in range(config.warmup_iters):
        out = layer(x)
        if include_backward:
            loss = out.sum()
            loss.backward()
            x.grad = None
    
    sync_device(config.device)
    layer.zero_grad()
    
    # Benchmark forward
    forward_times = []
    for _ in range(config.benchmark_iters):
        x_bench = x.detach().requires_grad_(True)
        
        sync_device(config.device)
        start = time.perf_counter()
        
        out = layer(x_bench)
        
        sync_device(config.device)
        forward_times.append((time.perf_counter() - start) * 1000)
    
    # Benchmark backward
    backward_times = []
    if include_backward:
        for _ in range(config.benchmark_iters):
            x_bench = x.detach().requires_grad_(True)
            out = layer(x_bench)
            loss = out.sum()
            
            sync_device(config.device)
            start = time.perf_counter()
            
            loss.backward()
            
            sync_device(config.device)
            backward_times.append((time.perf_counter() - start) * 1000)
    
    fwd_avg = sum(forward_times) / len(forward_times)
    fwd_std = (sum((t - fwd_avg)**2 for t in forward_times) / len(forward_times)) ** 0.5
    
    if include_backward:
        bwd_avg = sum(backward_times) / len(backward_times)
        bwd_std = (sum((t - bwd_avg)**2 for t in backward_times) / len(backward_times)) ** 0.5
    else:
        bwd_avg = bwd_std = 0.0
    
    return {
        'forward_ms': fwd_avg,
        'forward_std': fwd_std,
        'backward_ms': bwd_avg,
        'backward_std': bwd_std,
        'total_ms': fwd_avg + bwd_avg,
    }


def format_result(name: str, result: Dict[str, float], baseline: float = None) -> str:
    """Format benchmark result with optional slowdown comparison."""
    fwd = f"{result['forward_ms']:7.3f} ± {result['forward_std']:5.3f}"
    bwd = f"{result['backward_ms']:7.3f} ± {result['backward_std']:5.3f}"
    total = f"{result['total_ms']:7.3f}"
    
    if baseline:
        slowdown = result['total_ms'] / baseline
        return f"  {name:30s} | Fwd: {fwd} ms | Bwd: {bwd} ms | Total: {total} ms | {slowdown:5.2f}x"
    else:
        return f"  {name:30s} | Fwd: {fwd} ms | Bwd: {bwd} ms | Total: {total} ms"


def run_benchmark_suite(config: BenchmarkConfig):
    """Run the full benchmark suite."""
    
    print("=" * 100)
    print("WTConv2d vs Regular/Depthwise Convolution Benchmark")
    print("=" * 100)
    
    if config.device == 'cuda':
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Device: Apple MPS")
    
    print(f"Dtype: {config.dtype}")
    print(f"Batch size: {config.batch_size}")
    print(f"Warmup iterations: {config.warmup_iters}")
    print(f"Benchmark iterations: {config.benchmark_iters}")
    print("=" * 100)
    
    # Test configurations
    spatial_sizes = [64, 128, 256]
    channel_counts = [32, 64, 128]
    wt_levels_to_test = [1, 2, 3, 4, 5]
    
    all_results = []
    
    for H in spatial_sizes:
        for C in channel_counts:
            print(f"\n{'─' * 100}")
            print(f"Input shape: ({config.batch_size}, {C}, {H}, {H})")
            print(f"{'─' * 100}")
            
            x = torch.randn(config.batch_size, C, H, H, device=config.device, dtype=config.dtype)
            
            # Benchmark depthwise convolution (baseline)
            dw_conv = create_depthwise_conv(C).to(dtype=config.dtype)
            dw_result = benchmark_layer(dw_conv, x, config)
            baseline = dw_result['total_ms']
            print(format_result("Depthwise Conv (baseline)", dw_result))
            
            # Benchmark regular convolution
            reg_conv = create_regular_conv(C).to(dtype=config.dtype)
            reg_result = benchmark_layer(reg_conv, x, config)
            print(format_result("Regular Conv", reg_result, baseline))
            
            # Benchmark WTConv2d at different levels
            for wt_level in wt_levels_to_test:
                try:
                    # Optimized WTConv2d
                    wt_conv = WTConv2d(C, C, kernel_size=WTCONV_KERNEL_SIZE, wt_levels=wt_level).to(dtype=config.dtype)
                    wt_result = benchmark_layer(wt_conv, x, config)
                    print(format_result(f"WTConv2d (levels={wt_level})", wt_result, baseline))
                    
                    # Naive WTConv2d (PyWavelets-based)
                    wt_naive = WTConv2dNaive(C, C, kernel_size=WTCONV_KERNEL_SIZE, wt_levels=wt_level).to(dtype=config.dtype)
                    naive_result = benchmark_layer(wt_naive, x, config)
                    print(format_result(f"WTConv2d Naive (levels={wt_level})", naive_result, baseline))
                    
                    # Print speedup of optimized vs naive
                    speedup = naive_result['total_ms'] / wt_result['total_ms']
                    print(f"    └─ Optimized is {speedup:.2f}x faster than Naive")
                    
                    all_results.append({
                        'H': H, 'C': C, 'wt_level': wt_level,
                        'dw_ms': dw_result['total_ms'],
                        'reg_ms': reg_result['total_ms'],
                        'wt_ms': wt_result['total_ms'],
                        'naive_ms': naive_result['total_ms'],
                        'slowdown_vs_dw': wt_result['total_ms'] / dw_result['total_ms'],
                        'slowdown_vs_reg': wt_result['total_ms'] / reg_result['total_ms'],
                        'speedup_vs_naive': speedup,
                    })
                except Exception as e:
                    print(f"  WTConv2d (levels={wt_level}): SKIPPED ({e})")
            
            # Clean up
            del dw_conv, reg_conv, x
            if config.device == 'cuda':
                torch.cuda.empty_cache()
    
    # Summary Table (Fused WTConv as base)
    print("\n" + "=" * 100)
    print(f"SUMMARY: Performance Relative to Fused WTConv (base = 1.00x) [dtype={config.dtype}]")
    print("=" * 100)
    
    # Table header
    header = f"{'WT Level':<10} | {'Depthwise':<12} | {'Regular Conv':<14} | {'Naive WTConv':<14} | {'Fused WTConv':<12}"
    print(header)
    print("-" * len(header))
    
    for wt_level in wt_levels_to_test:
        level_results = [r for r in all_results if r['wt_level'] == wt_level]
        if level_results:
            # Calculate ratios: other_time / fused_wt_time
            avg_dw_ratio = sum(r['dw_ms'] / r['wt_ms'] for r in level_results) / len(level_results)
            avg_reg_ratio = sum(r['reg_ms'] / r['wt_ms'] for r in level_results) / len(level_results)
            avg_naive_ratio = sum(r['naive_ms'] / r['wt_ms'] for r in level_results) / len(level_results)
            fused_ratio = 1.00  # base
            
            print(f"{wt_level:<10} | {avg_dw_ratio:<12.2f} | {avg_reg_ratio:<14.2f} | {avg_naive_ratio:<14.2f} | {fused_ratio:<12.2f}")

def _benchmark_fn(fn, config: BenchmarkConfig) -> float:
    """Benchmark a single function, return average time in ms."""
    # Warmup
    for _ in range(config.warmup_iters):
        fn()
    
    sync_device(config.device)
    
    times = []
    for _ in range(config.benchmark_iters):
        sync_device(config.device)
        start = time.perf_counter()
        fn()
        sync_device(config.device)
        times.append((time.perf_counter() - start) * 1000)
    
    return sum(times) / len(times)


def main():
    parser = argparse.ArgumentParser(description="Benchmark WTConv2d vs standard convolutions")
    parser.add_argument("--device", type=str, choices=['cuda', 'mps'], default=None,
                        help="Device to use: cuda or mps (auto-detected if not specified)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"],
                        help="Data type: fp16, bf16, or fp32 (default: fp32)")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile optimization")
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
    
    # Parse dtype
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    
    # bf16 not supported on MPS
    if device == 'mps' and dtype == torch.bfloat16:
        print("WARNING: bfloat16 not supported on MPS, using float16")
        dtype = torch.float16
    
    config = BenchmarkConfig(
        batch_size=args.batch_size,
        warmup_iters=args.warmup,
        benchmark_iters=args.iters,
        device=device,
        dtype=dtype,
        use_compile=args.compile,
    )
    
    print(f"Using device: {device.upper()}")
    
    if args.compile:
        print("torch.compile optimization enabled")
    
    run_benchmark_suite(config)
    
    print("\n" + "=" * 100)
    print("Benchmark complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()
