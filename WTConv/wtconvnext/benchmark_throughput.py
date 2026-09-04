"""
Throughput and Peak Memory Benchmark: ConvNeXt vs WTConvNeXt

Measures throughput in images per second and peak allocated GPU memory over
the timed iterations for ConvNeXt-T, the released WTConvNeXt-T reference, and
WTConvNeXt-T with the fused CUDA kernels.

Defaults reproduce Table 7: batch size 64, 224x224 inputs, 20 warmup
batches, 50 measured batches, FP32, and torch.compile enabled. Run the
``inference`` and ``train`` modes separately with ``--out <path>`` to retain
the full protocol and per-batch latency summaries as JSON.
"""

import argparse
import json
import sys
import os
import statistics
import warnings
import logging

os.environ["TRITON_CACHE_DIR"] = os.path.expanduser("~/.triton/cache")

# Suppress torch.compile warnings
warnings.filterwarnings("ignore", message=".*_maybe_guard_rel.*")
warnings.filterwarnings("ignore", message=".*recompile_limit.*")
warnings.filterwarnings("ignore", message=".*pow_by_natural*")


from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import io
import torch
import timm

# Suppress torch dynamo and symbolic shapes logging (after torch import)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)

# Add parent directory to path for custom wtconv implementations
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wtconvnext import wtconvnext_tiny, wtconvnext_small, wtconvnext_base
from paper.scripts.common import environment as capture_environment
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# =============================================================================
# Configuration flags
# =============================================================================
BENCHMARK_TRITON = False  # Set to True to include Triton benchmarks
BENCHMARK_CUDA = True  # Set to True to include CUDA benchmarks
BENCHMARK_REGULAR = True  # Set to True to include regular/naive WTConvNeXt benchmarks
CONVNEXT_KERNEL_SIZE = 7  # Kernel size for ConvNeXt depthwise convolutions (default: 7)
WTCONVNEXT_KERNEL_SIZE = 5  # Kernel size for WTConvNeXt depthwise convolutions (default: 5)
USE_CONV_MLP = False  # Use 1x1 conv in MLP


# Lazy-loaded WTConv classes
_WTConv2dCUDA = None
_WTConv2dTriton = None


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


def create_wtconvnext_cuda(size='tiny'):
    """Create WTConvNeXt using CUDA Haar kernels."""
    WTConv2dCUDA = _get_wtconv_cuda()
    if size == 'tiny':
        return wtconvnext_tiny(pretrained=False, wtconv_class=WTConv2dCUDA, conv_mlp=USE_CONV_MLP, kernel_sizes=WTCONVNEXT_KERNEL_SIZE)
    elif size == 'small':
        return wtconvnext_small(pretrained=False, wtconv_class=WTConv2dCUDA, conv_mlp=USE_CONV_MLP, kernel_sizes=WTCONVNEXT_KERNEL_SIZE)
    else:
        return wtconvnext_base(pretrained=False, wtconv_class=WTConv2dCUDA, conv_mlp=USE_CONV_MLP, kernel_sizes=WTCONVNEXT_KERNEL_SIZE)


def create_wtconvnext_triton(size='tiny'):
    """Create WTConvNeXt using Triton kernels."""
    WTConv2dTriton = _get_wtconv_triton()
    if size == 'tiny':
        return wtconvnext_tiny(pretrained=False, wtconv_class=WTConv2dTriton, conv_mlp=USE_CONV_MLP, kernel_sizes=WTCONVNEXT_KERNEL_SIZE)
    elif size == 'small':
        return wtconvnext_small(pretrained=False, wtconv_class=WTConv2dTriton, conv_mlp=USE_CONV_MLP, kernel_sizes=WTCONVNEXT_KERNEL_SIZE)
    else:
        return wtconvnext_base(pretrained=False, wtconv_class=WTConv2dTriton, conv_mlp=USE_CONV_MLP, kernel_sizes=WTCONVNEXT_KERNEL_SIZE)


def benchmark_model(model, device, batch_size=64, warmup_batches=20,
                    measure_batches=50, training_step=False,
                    use_torch_compile=True):
    """
    Benchmark model throughput and peak GPU memory.

    Peak memory is measured over the timed iterations only: the counter is reset
    after warmup, so it reflects a steady-state iteration and excludes any
    transient allocations made while torch.compile traces the model. Weights and
    the input batch are already resident at that point and are therefore included.

    Args:
        model: PyTorch model to benchmark
        device: CUDA device
        batch_size: Number of images per batch
        warmup_batches: Number of warmup batches (not timed)
        measure_batches: Number of batches to measure
        training_step: If True, benchmark a full training step (forward + backward +
            optimizer step) instead of inference-only forward passes

    Returns:
        tuple[float, float]: Throughput in images per second, and peak allocated
            memory in MiB
    """
    model = model.to(device)

    # Optionally compile the model
    if use_torch_compile:
        model = torch.compile(model)

    # Create input tensor
    x = torch.randn(batch_size, 3, 224, 224, device=device)

    if training_step:
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        targets = torch.randint(0, 1000, (batch_size,), device=device)

        def step():
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), targets)
            loss.backward()
            optimizer.step()

        # Warmup (suppress output during first step which may trigger compilation)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            step()
        for _ in range(warmup_batches - 1):
            step()

        # Synchronize before timing
        torch.cuda.synchronize()

        # Track peak memory over the timed iterations only
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Measure every batch separately so dispersion is retained.
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_batches)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_batches)]
        for i in range(measure_batches):
            start_events[i].record()
            step()
            end_events[i].record()
    else:
        model.eval()

        # Warmup (suppress output during first forward which may trigger compilation)
        with torch.no_grad():
            # First forward may trigger CUDA compilation - suppress output
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                _ = model(x)
            # Rest of warmup
            for _ in range(warmup_batches - 1):
                _ = model(x)

        # Synchronize before timing
        torch.cuda.synchronize()

        # Track peak memory over the timed iterations only
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Measure every batch separately so dispersion is retained.
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_batches)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_batches)]
        with torch.no_grad():
            for i in range(measure_batches):
                start_events[i].record()
                _ = model(x)
                end_events[i].record()

    # Wait for completion and get elapsed time
    torch.cuda.synchronize()
    latencies_ms = [start_events[i].elapsed_time(end_events[i])
                    for i in range(measure_batches)]
    elapsed_ms = sum(latencies_ms)
    elapsed_sec = elapsed_ms / 1000.0

    # Peak allocated memory over the timed iterations
    peak_mem_mib = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Calculate throughput
    total_images = measure_batches * batch_size
    throughput = total_images / elapsed_sec

    ordered = sorted(latencies_ms)
    return {
        "throughput_images_s": throughput,
        "peak_memory_mib": peak_mem_mib,
        "latency_ms_mean": statistics.fmean(latencies_ms),
        "latency_ms_median": statistics.median(latencies_ms),
        "latency_ms_std": statistics.pstdev(latencies_ms)
        if len(latencies_ms) > 1 else 0.0,
        "latency_ms_p10": ordered[max(0, int(0.10 * len(ordered)) - 1)],
        "latency_ms_p90": ordered[min(len(ordered) - 1, int(0.90 * len(ordered)))],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inference", "train"),
                        default="inference")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--convnext-kernel-size", type=int, default=7)
    parser.add_argument("--wtconv-kernel-size", type=int, default=5)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction,
                        default=True, dest="use_torch_compile")
    parser.add_argument("--out", type=Path, help="optional JSON output path")
    return parser.parse_args()


def main():
    global CONVNEXT_KERNEL_SIZE, WTCONVNEXT_KERNEL_SIZE
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    if args.batch_size < 1 or args.warmup < 1 or args.iters < 1:
        raise ValueError("batch-size, warmup, and iters must be positive")
    CONVNEXT_KERNEL_SIZE = args.convnext_kernel_size
    WTCONVNEXT_KERNEL_SIZE = args.wtconv_kernel_size
    training_step = args.mode == "train"
    device = torch.device('cuda')
    
    # Build model list based on configuration
    models = [
        # Tiny variants
        ('ConvNeXt-T', lambda: timm.create_model('convnext_tiny', pretrained=False, kernel_sizes=CONVNEXT_KERNEL_SIZE, conv_mlp=USE_CONV_MLP), True),
    ]
    
    if BENCHMARK_REGULAR:
        models.append(('WTConvNeXt-T', lambda: wtconvnext_tiny(pretrained=False, conv_mlp=USE_CONV_MLP, kernel_sizes=WTCONVNEXT_KERNEL_SIZE), False))
    
    if BENCHMARK_CUDA:
        models.append(('WTConvNeXt-T (CUDA)', lambda: create_wtconvnext_cuda('tiny'), False))
    
    if BENCHMARK_TRITON:
        models.append(('WTConvNeXt-T (Triton)', lambda: create_wtconvnext_triton('tiny'), False))
    
    # models.extend([
    #     ('ConvNeXt-S', lambda: timm.create_model('convnext_small', pretrained=False, kernel_sizes=CONVNEXT_KERNEL_SIZE, conv_mlp=USE_CONV_MLP), True),
    # ])
    
    # if BENCHMARK_REGULAR:
    #     models.append(('WTConvNeXt-S', lambda: wtconvnext_small(pretrained=False, conv_mlp=USE_CONV_MLP, kernel_sizes=WTCONVNEXT_KERNEL_SIZE), False))
    
    # if BENCHMARK_CUDA:
    #     models.append(('WTConvNeXt-S (CUDA)', lambda: create_wtconvnext_cuda('small'), False))
    
    # if BENCHMARK_TRITON:
    #     models.append(('WTConvNeXt-S (Triton)', lambda: create_wtconvnext_triton('small'), False))
    
    # # Base variants  
    # models.extend([
    #     ('ConvNeXt-B', lambda: timm.create_model('convnext_base', pretrained=False, kernel_sizes=CONVNEXT_KERNEL_SIZE, conv_mlp=USE_CONV_MLP), True),
    # ])
    
    # if BENCHMARK_REGULAR:
    #     models.append(('WTConvNeXt-B', lambda: wtconvnext_base(pretrained=False, conv_mlp=USE_CONV_MLP, kernel_sizes=WTCONVNEXT_KERNEL_SIZE), False))
    
    # if BENCHMARK_CUDA:
    #     models.append(('WTConvNeXt-B (CUDA)', lambda: create_wtconvnext_cuda('base'), False))
    
    # if BENCHMARK_TRITON:
    #     models.append(('WTConvNeXt-B (Triton)', lambda: create_wtconvnext_triton('base'), False))
    
    mode = "training step" if training_step else "inference"
    print("\n" + "=" * 70)
    print(f"Throughput and Peak Memory Benchmark ({mode})")
    print("=" * 70)
    print(f"\n{'Model':<25} {'Images/sec':>20} {'Peak Mem (MiB)':>20}")
    print("-" * 67)

    results = []
    baseline_throughput = None
    baseline_memory = None
    current_size = None

    for name, model_factory, is_baseline in models:
        # Detect size change for grouping
        if 'ConvNeXt-T' in name and 'WT' not in name:
            current_size = 'T'
        elif 'ConvNeXt-S' in name and 'WT' not in name:
            current_size = 'S'
        elif 'ConvNeXt-B' in name and 'WT' not in name:
            current_size = 'B'

        model = model_factory()
        metrics = benchmark_model(
            model, device, batch_size=args.batch_size,
            warmup_batches=args.warmup, measure_batches=args.iters,
            training_step=training_step,
            use_torch_compile=args.use_torch_compile,
        )
        throughput = metrics["throughput_images_s"]
        peak_mem = metrics["peak_memory_mib"]
        results.append({"mode": args.mode, "model": name,
                        "is_baseline": is_baseline, **metrics})

        if is_baseline:
            baseline_throughput = throughput
            baseline_memory = peak_mem

        # Report absolute values, plus a ratio against the baseline where we have
        # one. Higher is better for throughput, lower is better for memory.
        thr_pct = mem_pct = ''
        if not is_baseline:
            if baseline_throughput:
                thr_pct = f"({throughput / baseline_throughput * 100:5.1f}%)"
            if baseline_memory:
                mem_pct = f"({peak_mem / baseline_memory * 100:5.1f}%)"
        print(f"{name:<25} {throughput:>10.2f} {thr_pct:>9} {peak_mem:>10.1f} {mem_pct:>9}")

        # Free memory
        del model
        torch.cuda.empty_cache()
        
        # Add separator at end of each size group
        is_last_in_group = (
            (not BENCHMARK_TRITON and 'CUDA' in name) or
            (BENCHMARK_TRITON and 'Triton' in name)
        )
        if is_last_in_group:
            print("-" * 67)

    print("\n" + "=" * 70)
    print("Summary: Implementation Comparison by Model Size")
    print("=" * 70)

    # Group results by size
    for size in ['T', 'S', 'B']:
        convnext = next((r for r in results if r["model"] == f'ConvNeXt-{size}'), None)
        if convnext:
            base_throughput = convnext["throughput_images_s"]
            base_memory = convnext["peak_memory_mib"]
            print(f"\n{size} variants (baseline: ConvNeXt-{size} = "
                  f"{base_throughput:.2f} img/sec, {base_memory:.1f} MiB)")

            for result in results:
                name = result["model"]
                throughput = result["throughput_images_s"]
                peak_mem = result["peak_memory_mib"]
                if f'WTConvNeXt-{size}' in name and throughput > 0:
                    thr_ratio = throughput / base_throughput * 100
                    mem_ratio = peak_mem / base_memory * 100
                    print(f"  {name:<25}: {throughput:>8.2f} img/sec ({thr_ratio:>5.1f}%)"
                          f" | {peak_mem:>8.1f} MiB ({mem_ratio:>5.1f}%)")

    payload = {
        "environment": capture_environment(),
        "protocol": {
            "mode": args.mode,
            "architecture": "ConvNeXt-T/WTConvNeXt-T",
            "depths": [3, 3, 9, 3],
            "widths": [96, 192, 384, 768],
            "wt_levels": [5, 4, 3, 2],
            "input_shape": [args.batch_size, 3, 224, 224],
            "precision": "fp32",
            "warmup": args.warmup,
            "iters": args.iters,
            "torch_compile": args.use_torch_compile,
            "convnext_kernel_size": args.convnext_kernel_size,
            "wtconv_kernel_size": args.wtconv_kernel_size,
            "data_loading_included": False,
            "host_to_device_transfer_included": False,
            "training_step": "zero_grad, forward cross-entropy, backward, SGD step",
        },
        "rows": results,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
