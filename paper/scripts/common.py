"""
Shared harness for the Fast-WTConv paper: model construction, weight
synchronisation between the reference and fused implementations, and
CUDA-event based timing / peak-memory measurement.

Every number in the paper is produced by `bench.py` and `correctness.py`,
both of which import this module. Nothing here is paper-specific beyond the
set of methods that get compared.
"""

from __future__ import annotations

import contextlib
import json
import os
import statistics
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn as nn

# --- make the project importable regardless of where the script is run from ---
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]                     # .../wt_tmlr
_WTCONV = _REPO / "WTConv"
for _p in (str(_WTCONV), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# =============================================================================
# Method registry
# =============================================================================

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _reference_wtconv(C: int, K: int, levels: int) -> nn.Module:
    """The implementation released by Finder et al. (2024)."""
    from wtconv.wtconv2d import WTConv2d as RefWTConv2d
    return RefWTConv2d(C, C, kernel_size=K, wt_levels=levels, bias=True)


def _fused_cuda_wtconv(C: int, K: int, levels: int) -> nn.Module:
    from fast_wtconv.wtconv import WTConv2d as CudaWTConv2d
    return CudaWTConv2d(C, C, kernel_size=K, wt_levels=levels, bias=True)


def _fused_triton_wtconv(C: int, K: int, levels: int) -> nn.Module:
    from fast_wtconv.wtconv_triton import WTConv2d as TritonWTConv2d
    return TritonWTConv2d(C, C, kernel_size=K, wt_levels=levels, bias=True)


def _depthwise(C: int, K: int, levels: int) -> nn.Module:
    return nn.Conv2d(C, C, K, padding="same", groups=C, bias=True)


def _dense(C: int, K: int, levels: int) -> nn.Module:
    return nn.Conv2d(C, C, K, padding="same", bias=True)


BUILDERS: dict[str, Callable[[int, int, int], nn.Module]] = {
    "reference": _reference_wtconv,
    "fused_cuda": _fused_cuda_wtconv,
    "fused_triton": _fused_triton_wtconv,
    "depthwise": _depthwise,
    "dense": _dense,
}

# Pretty names used in the LaTeX tables.
DISPLAY = {
    "depthwise": r"Depthwise $k$",
    "dense": r"Dense $k$",
    "reference": r"WTConv (reference)",
    "fused_cuda": r"Fused (CUDA)",
    "fused_triton": r"Fused (Triton)",
}


# =============================================================================
# Weight synchronisation
# =============================================================================

def sync_weights(ref: nn.Module, fast: nn.Module) -> None:
    """
    Copy every learnable tensor from the reference WTConv2d into a fused
    implementation, so that the two modules are the *same function* and any
    output difference is attributable to arithmetic reassociation alone.

    Reference parameter layout (wtconv/wtconv2d.py):
        base_conv.weight        (C, 1, K, K)
        base_conv.bias          (C,)
        base_scale.weight       (1, C, 1, 1)
        wavelet_convs[l].weight (4C, 1, K, K)
        wavelet_scale[l].weight (1, 4C, 1, 1)

    Both fused backends expose the same tensors under different names; the
    subband axis is packed identically (channel index = 4c + s with
    s in {LL, LH, HL, HH}), which we verify in `correctness.py`.
    """
    with torch.no_grad():
        # --- base path -------------------------------------------------------
        if hasattr(fast, "base_conv"):                     # CUDA backend reuses nn.Conv2d
            fast.base_conv.weight.copy_(ref.base_conv.weight)
            if ref.base_conv.bias is not None and fast.base_conv.bias is not None:
                fast.base_conv.bias.copy_(ref.base_conv.bias)
            fast.base_scale.weight.copy_(ref.base_scale.weight)
        else:                                              # Triton backend: raw Parameters
            fast.base_weight.copy_(ref.base_conv.weight)
            if ref.base_conv.bias is not None and fast.base_bias is not None:
                fast.base_bias.copy_(ref.base_conv.bias)
            fast.base_scale.copy_(ref.base_scale.weight)

        # --- per-level wavelet path -----------------------------------------
        n_levels = len(ref.wavelet_convs)
        if hasattr(fast, "wavelet_convs"):
            for l in range(n_levels):
                fast.wavelet_convs[l].weight.copy_(ref.wavelet_convs[l].weight)
                fast.wavelet_scale[l].weight.copy_(ref.wavelet_scale[l].weight)
        else:
            for l in range(n_levels):
                fast.wt_weights[l].copy_(ref.wavelet_convs[l].weight)
                fast.wt_scales[l].copy_(ref.wavelet_scale[l].weight)


def build_pair(method: str, C: int, K: int, levels: int, dtype: torch.dtype,
               device: str, seed: int = 0) -> nn.Module:
    """Build `method` with weights drawn from the same seed as the reference."""
    torch.manual_seed(seed)
    ref = _reference_wtconv(C, K, levels)
    if method == "reference":
        return ref.to(device=device, dtype=dtype)

    torch.manual_seed(seed)
    mod = BUILDERS[method](C, K, levels)
    if method in ("fused_cuda", "fused_triton"):
        sync_weights(ref, mod)
    return mod.to(device=device, dtype=dtype)


# =============================================================================
# Measurement
# =============================================================================

@dataclass
class Measurement:
    method: str
    dtype: str
    levels: int
    kernel_size: int
    batch: int
    channels: int
    spatial: int
    mode: str                      # "fwd" or "fwd_bwd"
    latency_ms_mean: float
    latency_ms_median: float
    latency_ms_std: float
    latency_ms_p10: float
    latency_ms_p90: float
    peak_mem_mib: float
    iters: int
    ok: bool = True
    error: str = ""


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _peak_mem_mib(device: str) -> float:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    if device == "mps":
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    return float("nan")


def _reset_peak(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def time_module(module: nn.Module, x: torch.Tensor, *, mode: str,
                warmup: int, iters: int, device: str) -> tuple[list[float], float]:
    """
    Return (per-iteration latencies in ms, peak memory in MiB).

    Timing uses CUDA events on NVIDIA (which measure device time and are immune
    to host-side launch skew) and a synchronised wall clock elsewhere. Every
    iteration is timed individually so the paper can report dispersion rather
    than a single mean.
    """
    import time

    needs_grad = mode == "fwd_bwd"
    x = x.detach().requires_grad_(needs_grad)

    def step():
        if needs_grad:
            out = module(x)
            out.sum().backward()
            module.zero_grad(set_to_none=True)
            if x.grad is not None:
                x.grad = None
        else:
            with torch.no_grad():
                module(x)

    # --- warmup: JIT compilation, Triton autotuning, cuDNN algorithm search ---
    for _ in range(warmup):
        step()
    _sync(device)

    # --- measure peak memory on a clean slate --------------------------------
    _reset_peak(device)
    step()
    _sync(device)
    peak = _peak_mem_mib(device)

    # --- timed loop ----------------------------------------------------------
    lat: list[float] = []
    if device == "cuda":
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            step()
            ends[i].record()
        torch.cuda.synchronize()
        lat = [starts[i].elapsed_time(ends[i]) for i in range(iters)]
    else:
        for _ in range(iters):
            _sync(device)
            t0 = time.perf_counter()
            step()
            _sync(device)
            lat.append((time.perf_counter() - t0) * 1e3)

    return lat, peak


def summarise(lat: list[float]) -> dict[str, float]:
    s = sorted(lat)
    n = len(s)
    return {
        "latency_ms_mean": statistics.fmean(s),
        "latency_ms_median": statistics.median(s),
        "latency_ms_std": statistics.pstdev(s) if n > 1 else 0.0,
        "latency_ms_p10": s[max(0, int(0.10 * n) - 1)],
        "latency_ms_p90": s[min(n - 1, int(0.90 * n))],
    }


# =============================================================================
# Environment capture (for the reproducibility appendix)
# =============================================================================

def environment() -> dict:
    env = {
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env.update({
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "hbm_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1),
        })
    with contextlib.suppress(Exception):
        import triton
        env["triton"] = triton.__version__
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        env["mps"] = True
    return env


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
    print(f"wrote {path}")
