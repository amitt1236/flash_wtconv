"""
Benchmark WTConv2d Implementations

Tests WTConv2d variants (original, CUDA, Triton) with specific input sizes and wavelet levels:
- Size [1, 96, 128, 128] with level=5
- Size [1, 192, 64, 64] with level=4
- Size [1, 384, 32, 32] with level=3
- Size [1, 768, 16, 16] with level=2

Configuration:
- Warmup: 50 iterations
- Measurement: 300 iterations
"""

import sys
import os
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import io
import torch

# Add parent directory to path for custom wtconv implementations
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# =============================================================================
# Configuration flags
# =============================================================================
BENCHMARK_TRITON = False  # Set to False to skip Triton benchmarks

# Test configurations: (input_size, level)
TEST_CONFIGS = [
    # ([64, 96, 128, 128], 5),
    # ([64, 192, 64, 64], 4),
    # ([64, 384, 32, 32], 3),
    # ([64, 768, 16, 16], 2),

    ([64, 96, 56, 56], 5),
    ([64, 192, 28, 28], 4),
    ([64, 384, 14, 14], 3),
    ([64, 768, 7, 7], 2),
]

# Lazy-loaded WTConv classes
_WTConv2dOriginal = None
_WTConv2dCUDA = None
_WTConv2dTriton = None


def _get_wtconv_original():
    """Get original WTConv2d class."""
    global _WTConv2dOriginal
    if _WTConv2dOriginal is None:
        from wtconvnext.wtconvnext import WTConv2d
        _WTConv2dOriginal = WTConv2d
    return _WTConv2dOriginal


def _get_wtconv_cuda():
    """Get CUDA WTConv2d class (lazy load to avoid compile messages during import)."""
    global _WTConv2dCUDA
    if _WTConv2dCUDA is None:
        # Suppress compilation output
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            from wtconv_model.wtconv import WTConv2d
            _WTConv2dCUDA = WTConv2d
    return _WTConv2dCUDA


def _get_wtconv_triton():
    """Get Triton WTConv2d class."""
    global _WTConv2dTriton
    if _WTConv2dTriton is None:
        from wtconv_model.wtconv_triton import WTConv2d
        _WTConv2dTriton = WTConv2d
    return _WTConv2dTriton


def benchmark_wtconv(wtconv_layer, x, device, warmup_iters=50, measure_iters=300):
    """
    Benchmark WTConv2d layer throughput.
    
    Args:
        wtconv_layer: WTConv2d layer to benchmark
        x: Input tensor
        device: CUDA device
        warmup_iters: Number of warmup iterations (not timed)
        measure_iters: Number of iterations to measure
        
    Returns:
        float: Average time per forward pass in milliseconds
    """
    wtconv_layer.eval()
    wtconv_layer = wtconv_layer.to(device)
    x = x.to(device)
    
    # Warmup (suppress output during first forward which may trigger compilation)
    with torch.no_grad():
        # First forward may trigger CUDA compilation - suppress output
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            _ = wtconv_layer(x)
        # Rest of warmup
        for _ in range(warmup_iters - 1):
            _ = wtconv_layer(x)
    
    # Synchronize before timing
    torch.cuda.synchronize()
    
    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Measure
    with torch.no_grad():
        start_event.record()
        for _ in range(measure_iters):
            _ = wtconv_layer(x)
        end_event.record()
    
    # Wait for completion and get elapsed time
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / measure_iters
    
    return avg_time_ms


def main():
    device = torch.device('cuda')
    
    print("\n" + "=" * 80)
    print("WTConv2d Implementation Benchmark")
    print("=" * 80)
    
    # Get WTConv classes
    WTConv2dOriginal = _get_wtconv_original()
    WTConv2dCUDA = _get_wtconv_cuda()
    if BENCHMARK_TRITON:
        WTConv2dTriton = _get_wtconv_triton()
    
    for input_size, level in TEST_CONFIGS:
        print(f"\n{'─' * 80}")
        print(f"Input Size: {input_size}, Wavelet Level: {level}")
        print(f"{'─' * 80}")
        
        # Extract dimensions
        batch_size, in_channels, height, width = input_size
        
        # Create input tensor
        x = torch.randn(*input_size, device=device)
        
        # Build list of implementations to test
        implementations = [
            ('Original', WTConv2dOriginal(in_channels, in_channels, wt_levels=level, kernel_size=5)),
            ('CUDA', WTConv2dCUDA(in_channels, in_channels, wt_levels=level, kernel_size=5)),
        ]
        
        if BENCHMARK_TRITON:
            implementations.append(
                ('Triton', WTConv2dTriton(in_channels, in_channels, wt_levels=level, kernel_size=5))
            )
        
        print(f"\n{'Implementation':<20} {'Avg Time (ms)':<20} {'Speedup vs Original':<20}")
        print("-" * 60)
        
        results = []
        original_time = None
        
        for name, layer in implementations:
            avg_time = benchmark_wtconv(layer, x, device)
            results.append((name, avg_time))
            
            if name == 'Original':
                original_time = avg_time
                print(f"{name:<20} {avg_time:>15.4f} ms")
            else:
                speedup = original_time / avg_time if original_time else 0
                print(f"{name:<20} {avg_time:>15.4f} ms  {speedup:>15.2f}x")
            
            # Free memory
            del layer
            torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print("Benchmark Complete")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
