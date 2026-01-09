"""
Benchmark: JAX WTConv2d (CUDA) vs PyTorch Fused WTConv2d (CUDA)

Compares JAX implementation (running on CUDA via jax-cuda12-plugin) against 
the PyTorch CUDA fused implementation across:
- Input sizes (spatial dimensions)
- Channel counts
- Wavelet levels (1-5)

JAX implementation is the main benchmark target.
PyTorch CUDA fused implementation is used as the baseline for comparison.
"""

import os
import sys
import warnings
import time
from pathlib import Path

# Note: To enable JAX CUDA, install: pip install jax[cuda12]
# JAX will automatically detect and use available backends

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
import argparse

# Import JAX WTConv from tpu package
from wtconv_model.wtconv_tpu import WTConv2d as JAXWTConv2d

# Try to import PyTorch CUDA implementation
try:
    import torch
    from wtconv_model.wtconv import WTConv2d as PyTorchWTConv2d
    PYTORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    PYTORCH_CUDA_AVAILABLE = False
    PyTorchWTConv2d = None


# Kernel size constants
JAX_KERNEL_SIZE = 5
CUDA_KERNEL_SIZE = 5


@dataclass
class BenchmarkConfig:
    batch_size: int = 8
    warmup_iters: int = 10
    benchmark_iters: int = 100
    dtype: str = "float32"


def get_jax_dtype(dtype_str: str):
    """Convert dtype string to JAX dtype."""
    dtype_map = {
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
        "float32": jnp.float32,
    }
    return dtype_map.get(dtype_str, jnp.float32)


def get_torch_dtype(dtype_str: str):
    """Convert dtype string to torch dtype."""
    import torch
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.float32)


def benchmark_jax_layer(
    model: nn.Module,
    params,
    x: jnp.ndarray,
    config: BenchmarkConfig,
    include_backward: bool = True
) -> Dict[str, float]:
    """
    Benchmark a JAX/Flax layer's forward and backward pass.
    
    Returns:
        Dict with 'forward_ms', 'backward_ms', 'total_ms'
    """
    # JIT compile forward pass
    @jax.jit
    def forward_fn(params, x):
        return model.apply(params, x)
    
    # JIT compile forward + backward
    @jax.jit
    def forward_backward_fn(params, x):
        def loss_fn(params, x):
            out = model.apply(params, x)
            return out.sum()
        loss, grads = jax.value_and_grad(loss_fn)(params, x)
        return loss, grads
    
    # Warmup - this also triggers JIT compilation
    for _ in range(config.warmup_iters):
        out = forward_fn(params, x)
        if include_backward:
            loss, grads = forward_backward_fn(params, x)
    
    # Block until warmup complete
    jax.block_until_ready(out)
    if include_backward:
        jax.block_until_ready(grads)
    
    # Benchmark forward
    forward_times = []
    for _ in range(config.benchmark_iters):
        start = time.perf_counter()
        out = forward_fn(params, x)
        jax.block_until_ready(out)
        forward_times.append((time.perf_counter() - start) * 1000)
    
    # Benchmark backward
    backward_times = []
    if include_backward:
        for _ in range(config.benchmark_iters):
            # Run forward first
            out = forward_fn(params, x)
            jax.block_until_ready(out)
            
            start = time.perf_counter()
            loss, grads = forward_backward_fn(params, x)
            jax.block_until_ready(grads)
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


def benchmark_cuda_layer(
    layer,
    x,
    config: BenchmarkConfig,
    include_backward: bool = True
) -> Dict[str, float]:
    """
    Benchmark a CUDA PyTorch layer's forward and backward pass.
    
    Returns:
        Dict with 'forward_ms', 'backward_ms', 'total_ms'
    """
    import torch
    
    layer = layer.cuda()
    x = x.cuda().requires_grad_(True)
    
    # Warmup
    for _ in range(config.warmup_iters):
        out = layer(x)
        if include_backward:
            loss = out.sum()
            loss.backward()
            x.grad = None
    
    torch.cuda.synchronize()
    layer.zero_grad()
    
    # Benchmark forward
    forward_times = []
    for _ in range(config.benchmark_iters):
        x_bench = x.detach().requires_grad_(True)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        out = layer(x_bench)
        
        torch.cuda.synchronize()
        forward_times.append((time.perf_counter() - start) * 1000)
    
    # Benchmark backward
    backward_times = []
    if include_backward:
        for _ in range(config.benchmark_iters):
            x_bench = x.detach().requires_grad_(True)
            out = layer(x_bench)
            loss = out.sum()
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            loss.backward()
            
            torch.cuda.synchronize()
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
    """Format benchmark result with optional speedup comparison."""
    fwd = f"{result['forward_ms']:7.3f} ± {result['forward_std']:5.3f}"
    bwd = f"{result['backward_ms']:7.3f} ± {result['backward_std']:5.3f}"
    total = f"{result['total_ms']:7.3f}"
    
    if baseline:
        speedup = baseline / result['total_ms']
        return f"  {name:35s} | Fwd: {fwd} ms | Bwd: {bwd} ms | Total: {total} ms | {speedup:5.2f}x vs CUDA"
    else:
        return f"  {name:35s} | Fwd: {fwd} ms | Bwd: {bwd} ms | Total: {total} ms"


def run_benchmark_suite(config: BenchmarkConfig):
    """Run the full benchmark suite."""
    
    print("=" * 110)
    print("JAX WTConv2d (CUDA) vs PyTorch Fused WTConv2d (CUDA) Benchmark")
    print("=" * 110)
    
    # Print device info
    print(f"\nJAX Devices: {jax.devices()}")
    if PYTORCH_CUDA_AVAILABLE:
        import torch
        print(f"PyTorch CUDA Device: {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch CUDA: Not available (JAX-only benchmark)")
    
    print(f"\nDtype: {config.dtype}")
    print(f"Batch size: {config.batch_size}")
    print(f"Warmup iterations: {config.warmup_iters}")
    print(f"Benchmark iterations: {config.benchmark_iters}")
    print("=" * 110)
    
    jax_dtype = get_jax_dtype(config.dtype)
    
    # Test configurations
    spatial_sizes = [64, 128, 256]
    channel_counts = [32, 64, 128]
    wt_levels_to_test = [1, 2, 3, 4, 5]
    
    all_results = []
    
    for H in spatial_sizes:
        for C in channel_counts:
            print(f"\n{'─' * 110}")
            print(f"Input shape: ({config.batch_size}, {H}, {H}, {C}) [NHWC for JAX]")
            print(f"{'─' * 110}")
            
            # Create JAX input (NHWC format)
            rng = jax.random.PRNGKey(42)
            x_jax = jax.random.normal(rng, (config.batch_size, H, H, C), dtype=jax_dtype)
            
            # Create PyTorch CUDA input if available (NCHW format)
            if PYTORCH_CUDA_AVAILABLE:
                import torch
                torch_dtype = get_torch_dtype(config.dtype)
                x_cuda = torch.randn(config.batch_size, C, H, H, device='cuda', dtype=torch_dtype)
            
            for wt_level in wt_levels_to_test:
                try:
                    # Benchmark JAX WTConv2d
                    jax_model = JAXWTConv2d(
                        channels=C,
                        kernel_size=JAX_KERNEL_SIZE,
                        depth=wt_level,
                        use_bias=True,
                        dtype=jax_dtype
                    )
                    
                    # Initialize parameters
                    rng_init = jax.random.PRNGKey(0)
                    dummy_input = jnp.zeros((1, H, H, C), dtype=jax_dtype)
                    params = jax_model.init(rng_init, dummy_input)
                    
                    jax_result = benchmark_jax_layer(jax_model, params, x_jax, config)
                    
                    # Benchmark PyTorch CUDA if available
                    if PYTORCH_CUDA_AVAILABLE:
                        pytorch_model = PyTorchWTConv2d(C, C, kernel_size=CUDA_KERNEL_SIZE, wt_levels=wt_level).to(torch_dtype)
                        pytorch_result = benchmark_cuda_layer(pytorch_model, x_cuda, config)
                        baseline = pytorch_result['total_ms']
                        
                        print(format_result(f"PyTorch Fused WTConv (levels={wt_level})", pytorch_result))
                        print(format_result(f"JAX WTConv2d (levels={wt_level})", jax_result, baseline))
                        
                        # Calculate speedup
                        speedup = pytorch_result['total_ms'] / jax_result['total_ms']
                        if speedup >= 1.0:
                            print(f"    └─ JAX is {speedup:.2f}x faster than PyTorch Fused")
                        else:
                            print(f"    └─ PyTorch Fused is {1/speedup:.2f}x faster than JAX")
                        
                        all_results.append({
                            'H': H, 'C': C, 'wt_level': wt_level,
                            'jax_ms': jax_result['total_ms'],
                            'pytorch_ms': pytorch_result['total_ms'],
                            'jax_fwd_ms': jax_result['forward_ms'],
                            'pytorch_fwd_ms': pytorch_result['forward_ms'],
                            'speedup': speedup,
                        })
                        
                        del pytorch_model
                    else:
                        print(format_result(f"JAX WTConv2d (levels={wt_level})", jax_result))
                        all_results.append({
                            'H': H, 'C': C, 'wt_level': wt_level,
                            'jax_ms': jax_result['total_ms'],
                            'pytorch_ms': None,
                            'jax_fwd_ms': jax_result['forward_ms'],
                            'pytorch_fwd_ms': None,
                            'speedup': None,
                        })
                    
                except Exception as e:
                    print(f"  WTConv (levels={wt_level}): SKIPPED ({e})")
            
            # Clean up
            del x_jax
            if PYTORCH_CUDA_AVAILABLE:
                import torch
                del x_cuda
                torch.cuda.empty_cache()
    
    # Summary Table
    if PYTORCH_CUDA_AVAILABLE and all_results:
        print("\n" + "=" * 110)
        print(f"SUMMARY: JAX CUDA vs PyTorch Fused CUDA Performance [dtype={config.dtype}]")
        print("=" * 110)
        
        # Table header
        header = f"{'WT Level':<10} | {'PyTorch Fused (ms)':<18} | {'JAX CUDA (ms)':<18} | {'JAX Speedup':<12}"
        print(header)
        print("-" * len(header))
        
        for wt_level in wt_levels_to_test:
            level_results = [r for r in all_results if r['wt_level'] == wt_level]
            if level_results:
                avg_jax = sum(r['jax_ms'] for r in level_results) / len(level_results)
                avg_pytorch = sum(r['pytorch_ms'] for r in level_results) / len(level_results)
                avg_speedup = avg_pytorch / avg_jax
                
                if avg_speedup >= 1.0:
                    speedup_str = f"{avg_speedup:.2f}x faster"
                else:
                    speedup_str = f"{1/avg_speedup:.2f}x slower"
                
                print(f"{wt_level:<10} | {avg_pytorch:<18.2f} | {avg_jax:<18.2f} | {speedup_str:<12}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark JAX WTConv2d vs CUDA Fused WTConv2d")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--dtype", type=str, default="float32", 
                        choices=["float16", "bfloat16", "float32"],
                        help="Data type: float16, bfloat16, or float32 (default: float32)")
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        batch_size=args.batch_size,
        warmup_iters=args.warmup,
        benchmark_iters=args.iters,
        dtype=args.dtype,
    )
    
    print(f"JAX version: {jax.__version__}")
    print(f"Running on: {jax.devices()}")
    
    run_benchmark_suite(config)
    
    print("\n" + "=" * 110)
    print("Benchmark complete!")
    print("=" * 110)


if __name__ == "__main__":
    main()
