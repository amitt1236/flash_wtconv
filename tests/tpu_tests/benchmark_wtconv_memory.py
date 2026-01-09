"""
Benchmark: JAX WTConv2d (CUDA) vs PyTorch Fused WTConv2d (CUDA)
GPU Peak Memory Consumption

Compares peak memory usage across different:
- Input sizes (spatial dimensions)
- Channel counts
- Wavelet levels (1-5)
"""

import os
import sys
from pathlib import Path

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
PYTORCH_KERNEL_SIZE = 5


@dataclass
class BenchmarkConfig:
    batch_size: int = 8
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


def measure_jax_peak_memory(
    model: nn.Module,
    params,
    x: jnp.ndarray,
    include_backward: bool = True
) -> Dict[str, float]:
    """
    Measure peak GPU memory consumption for a JAX layer's forward and backward pass.
    
    Note: JAX doesn't have direct memory tracking like PyTorch.
    We use jax.local_devices()[0].memory_stats() for CUDA devices.
    
    Returns:
        Dict with 'total_mb' (peak memory in MB for combined forward+backward)
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
    
    # Get device
    device = jax.devices('gpu')[0]
    
    # Clear memory and run garbage collection
    jax.clear_caches()
    
    # Get memory stats before (if available)
    try:
        stats_before = device.memory_stats()
        peak_before = stats_before.get('peak_bytes_in_use', 0)
    except:
        peak_before = 0
    
    # Run forward (and backward if requested)
    if include_backward:
        loss, grads = forward_backward_fn(params, x)
        jax.block_until_ready(grads)
    else:
        out = forward_fn(params, x)
        jax.block_until_ready(out)
    
    # Get memory stats after
    try:
        stats_after = device.memory_stats()
        peak_after = stats_after.get('peak_bytes_in_use', 0)
        peak_mb = peak_after / 1024 / 1024
    except:
        # Fallback: calculate based on tensor sizes
        peak_mb = estimate_jax_memory(model, params, x, include_backward)
    
    return {
        'total_mb': peak_mb,
    }


def estimate_jax_memory(model, params, x, include_backward=True) -> float:
    """Estimate memory usage based on tensor sizes when direct measurement isn't available."""
    # Count parameter bytes
    param_bytes = sum(p.nbytes for p in jax.tree_util.tree_leaves(params))
    
    # Input size
    input_bytes = x.nbytes
    
    # Output size (approximate - same as input for most conv layers)
    output_bytes = input_bytes
    
    # Backward pass roughly doubles memory for gradients
    if include_backward:
        total_bytes = param_bytes * 2 + input_bytes * 3 + output_bytes
    else:
        total_bytes = param_bytes + input_bytes + output_bytes
    
    return total_bytes / 1024 / 1024


def measure_pytorch_peak_memory(
    layer,
    x,
    include_backward: bool = True
) -> Dict[str, float]:
    """
    Measure peak GPU memory consumption for a PyTorch layer's forward and backward pass.
    
    Returns:
        Dict with 'total_mb' (peak memory in MB)
    """
    import torch
    
    layer = layer.cuda()
    
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
        return f"  {name:35s} | Peak Memory: {total} MB | {ratio:5.2f}x"
    else:
        return f"  {name:35s} | Peak Memory: {total} MB"


def run_benchmark_suite(config: BenchmarkConfig):
    """Run the full memory benchmark suite."""
    
    print("=" * 110)
    print("JAX WTConv2d (CUDA) vs PyTorch Fused WTConv2d (CUDA) - GPU Peak Memory Benchmark")
    print("=" * 110)
    
    # Print device info
    print(f"\nJAX Devices: {jax.devices()}")
    if PYTORCH_CUDA_AVAILABLE:
        import torch
        print(f"PyTorch CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.0f} MB")
    else:
        print("PyTorch CUDA: Not available (JAX-only benchmark)")
    
    print(f"\nDtype: {config.dtype}")
    print(f"Batch size: {config.batch_size}")
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
                x_pytorch = torch.randn(config.batch_size, C, H, H, device='cuda', dtype=torch_dtype)
            
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
                    
                    jax_result = measure_jax_peak_memory(jax_model, params, x_jax)
                    
                    # Benchmark PyTorch CUDA if available
                    if PYTORCH_CUDA_AVAILABLE:
                        pytorch_model = PyTorchWTConv2d(C, C, kernel_size=PYTORCH_KERNEL_SIZE, wt_levels=wt_level).to(torch_dtype)
                        pytorch_result = measure_pytorch_peak_memory(pytorch_model, x_pytorch)
                        baseline = pytorch_result['total_mb']
                        
                        print(format_result(f"PyTorch Fused WTConv (levels={wt_level})", pytorch_result))
                        print(format_result(f"JAX WTConv2d (levels={wt_level})", jax_result, baseline))
                        
                        # Calculate memory comparison
                        mem_ratio = pytorch_result['total_mb'] / jax_result['total_mb']
                        if mem_ratio >= 1.0:
                            print(f"    └─ JAX uses {mem_ratio:.2f}x less memory than PyTorch")
                        else:
                            print(f"    └─ PyTorch uses {1/mem_ratio:.2f}x less memory than JAX")
                        
                        all_results.append({
                            'H': H, 'C': C, 'wt_level': wt_level,
                            'jax_mb': jax_result['total_mb'],
                            'pytorch_mb': pytorch_result['total_mb'],
                            'ratio': mem_ratio,
                        })
                        
                        del pytorch_model
                    else:
                        print(format_result(f"JAX WTConv2d (levels={wt_level})", jax_result))
                        all_results.append({
                            'H': H, 'C': C, 'wt_level': wt_level,
                            'jax_mb': jax_result['total_mb'],
                            'pytorch_mb': None,
                            'ratio': None,
                        })
                    
                except Exception as e:
                    print(f"  WTConv (levels={wt_level}): SKIPPED ({e})")
            
            # Clean up
            del x_jax
            jax.clear_caches()
            if PYTORCH_CUDA_AVAILABLE:
                import torch
                del x_pytorch
                torch.cuda.empty_cache()
    
    # Summary Table
    if PYTORCH_CUDA_AVAILABLE and all_results:
        print("\n" + "=" * 110)
        print(f"SUMMARY: Peak Memory - JAX CUDA vs PyTorch Fused CUDA [dtype={config.dtype}]")
        print("=" * 110)
        
        # Table header
        header = f"{'WT Level':<10} | {'PyTorch (MB)':<15} | {'JAX (MB)':<15} | {'JAX Memory Savings':<20}"
        print(header)
        print("-" * len(header))
        
        for wt_level in wt_levels_to_test:
            level_results = [r for r in all_results if r['wt_level'] == wt_level]
            if level_results:
                avg_jax = sum(r['jax_mb'] for r in level_results) / len(level_results)
                avg_pytorch = sum(r['pytorch_mb'] for r in level_results) / len(level_results)
                avg_ratio = avg_pytorch / avg_jax
                
                if avg_ratio >= 1.0:
                    ratio_str = f"{avg_ratio:.2f}x less"
                else:
                    ratio_str = f"{1/avg_ratio:.2f}x more"
                
                print(f"{wt_level:<10} | {avg_pytorch:<15.2f} | {avg_jax:<15.2f} | {ratio_str:<20}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark JAX WTConv2d vs PyTorch Fused WTConv2d - GPU Memory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--dtype", type=str, default="float32", 
                        choices=["float16", "bfloat16", "float32"],
                        help="Data type: float16, bfloat16, or float32 (default: float32)")
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        batch_size=args.batch_size,
        dtype=args.dtype,
    )
    
    print(f"JAX version: {jax.__version__}")
    print(f"Running on: {jax.devices()}")
    
    run_benchmark_suite(config)
    
    print("\n" + "=" * 110)
    print("Memory Benchmark complete!")
    print("=" * 110)


if __name__ == "__main__":
    main()
