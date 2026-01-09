"""
Benchmark: WTConv2d vs Regular Convolution vs Depthwise Convolution
GPU Peak Memory Consumption

Compares peak memory usage across different:
- Input sizes (spatial dimensions)
- Channel counts
- Wavelet levels (for WTConv2d)
"""

import sys
import warnings
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
from typing import Dict
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
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    use_compile: bool = False


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


def measure_peak_memory(
    layer: nn.Module, 
    x: torch.Tensor, 
    config: BenchmarkConfig,
    include_backward: bool = True
) -> Dict[str, float]:
    """
    Measure peak GPU memory consumption for a layer's forward and backward pass.
    
    Returns:
        Dict with 'total_mb' (peak memory in MB for combined forward+backward)
    """
    layer = layer.to(config.device)
    
    # Enable torch.compile if requested
    if config.use_compile:
        layer = torch.compile(layer)
    
    # Measure total peak memory (forward + backward together)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    x_total = x.clone().detach().requires_grad_(True)
    out = layer(x_total)
    if include_backward:
        loss = out.sum()
        loss.backward()
    
    torch.cuda.synchronize()
    total_peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    
    return {
        'total_mb': total_peak_mb,
    }


def format_result(name: str, result: Dict[str, float], baseline: float = None) -> str:
    """Format benchmark result with optional comparison."""
    total = f"{result['total_mb']:8.2f}"
    
    if baseline:
        ratio = result['total_mb'] / baseline
        return f"  {name:30s} | Peak Memory: {total} MB | {ratio:5.2f}x"
    else:
        return f"  {name:30s} | Peak Memory: {total} MB"


def run_benchmark_suite(config: BenchmarkConfig):
    """Run the full memory benchmark suite."""
    
    print("=" * 110)
    print("WTConv2d vs Regular/Depthwise Convolution - GPU Peak Memory Benchmark")
    print("=" * 110)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.0f} MB")
    print(f"Dtype: {config.dtype}")
    print(f"Batch size: {config.batch_size}")
    print("=" * 110)
    
    # Test configurations
    spatial_sizes = [64, 128, 256]
    channel_counts = [32, 64, 128]
    wt_levels_to_test = [1, 2, 3, 4, 5]
    
    all_results = []
    
    for H in spatial_sizes:
        for C in channel_counts:
            print(f"\n{'─' * 110}")
            print(f"Input shape: ({config.batch_size}, {C}, {H}, {H})")
            print(f"{'─' * 110}")
            
            x = torch.randn(config.batch_size, C, H, H, device=config.device, dtype=config.dtype)
            
            # Benchmark depthwise convolution (baseline)
            dw_conv = create_depthwise_conv(C).to(dtype=config.dtype)
            dw_result = measure_peak_memory(dw_conv, x, config)
            baseline = dw_result['total_mb']
            print(format_result("Depthwise Conv (baseline)", dw_result))
            
            # Benchmark regular convolution
            reg_conv = create_regular_conv(C).to(dtype=config.dtype)
            reg_result = measure_peak_memory(reg_conv, x, config)
            print(format_result("Regular Conv", reg_result, baseline))
            
            # Benchmark WTConv2d at different levels
            for wt_level in wt_levels_to_test:
                try:
                    # Optimized WTConv2d
                    wt_conv = WTConv2d(C, C, kernel_size=WTCONV_KERNEL_SIZE, wt_levels=wt_level).to(dtype=config.dtype)
                    wt_result = measure_peak_memory(wt_conv, x, config)
                    print(format_result(f"WTConv2d (levels={wt_level})", wt_result, baseline))
                    
                    # Naive WTConv2d (PyWavelets-based)
                    wt_naive = WTConv2dNaive(C, C, kernel_size=WTCONV_KERNEL_SIZE, wt_levels=wt_level).to(dtype=config.dtype)
                    naive_result = measure_peak_memory(wt_naive, x, config)
                    print(format_result(f"WTConv2d Naive (levels={wt_level})", naive_result, baseline))
                    
                    # Print memory comparison of optimized vs naive
                    mem_ratio = naive_result['total_mb'] / wt_result['total_mb']
                    if mem_ratio > 1:
                        print(f"    └─ Optimized uses {mem_ratio:.2f}x less memory than Naive")
                    else:
                        print(f"    └─ Optimized uses {1/mem_ratio:.2f}x more memory than Naive")
                    
                    all_results.append({
                        'H': H, 'C': C, 'wt_level': wt_level,
                        'dw_mb': dw_result['total_mb'],
                        'reg_mb': reg_result['total_mb'],
                        'wt_mb': wt_result['total_mb'],
                        'naive_mb': naive_result['total_mb'],
                        'ratio_vs_dw': wt_result['total_mb'] / dw_result['total_mb'],
                        'ratio_vs_reg': wt_result['total_mb'] / reg_result['total_mb'],
                        'ratio_vs_naive': wt_result['total_mb'] / naive_result['total_mb'],
                    })
                except Exception as e:
                    print(f"  WTConv2d (levels={wt_level}): SKIPPED ({e})")
            
            # Clean up
            del dw_conv, reg_conv, x
            torch.cuda.empty_cache()
    
    # Summary Table (Fused WTConv as base)
    print("\n" + "=" * 110)
    print(f"SUMMARY: Peak Memory Relative to Fused WTConv (base = 1.00x) [dtype={config.dtype}]")
    print("=" * 110)
    
    # Table header
    header = f"{'WT Level':<10} | {'Depthwise':<12} | {'Regular Conv':<14} | {'Naive WTConv':<14} | {'Fused WTConv':<12}"
    print(header)
    print("-" * len(header))
    
    for wt_level in wt_levels_to_test:
        level_results = [r for r in all_results if r['wt_level'] == wt_level]
        if level_results:
            # Calculate ratios: other_memory / fused_wt_memory
            avg_dw_ratio = sum(r['dw_mb'] / r['wt_mb'] for r in level_results) / len(level_results)
            avg_reg_ratio = sum(r['reg_mb'] / r['wt_mb'] for r in level_results) / len(level_results)
            avg_naive_ratio = sum(r['naive_mb'] / r['wt_mb'] for r in level_results) / len(level_results)
            fused_ratio = 1.00  # base
            
            print(f"{wt_level:<10} | {avg_dw_ratio:<12.2f} | {avg_reg_ratio:<14.2f} | {avg_naive_ratio:<14.2f} | {fused_ratio:<12.2f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark WTConv2d vs standard convolutions - GPU Memory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"],
                        help="Data type: fp16, bf16, or fp32 (default: fp32)")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile optimization")
    args = parser.parse_args()
    
    # Parse dtype
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    
    config = BenchmarkConfig(
        batch_size=args.batch_size,
        dtype=dtype,
        use_compile=args.compile,
    )
    
    if args.compile:
        print("torch.compile optimization enabled")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for this benchmark")
        return
    
    run_benchmark_suite(config)
    
    print("\n" + "=" * 110)
    print("Memory Benchmark complete!")
    print("=" * 110)


if __name__ == "__main__":
    main()
