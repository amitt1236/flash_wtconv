#!/usr/bin/env python3
"""
Latency and peak-memory benchmark for Fast-WTConv.

Reproduces every throughput/memory table in the paper. Results are written as
raw JSON; `make_tables.py` turns that JSON into the LaTeX the paper \\inputs,
so no number is ever transcribed by hand.

Two protocols, following the paper:

  homogeneous  every method uses the same kernel size k (default 5, the value
               WTConvNeXt uses). Isolates the cost of the wavelet machinery
               itself.

  dropin       WTConv at k=5 against dense/depthwise convolutions at k=7, i.e.
               the substitution WTConv is actually proposed for: the 7x7
               depthwise convolution in a ConvNeXt block. This is NOT a
               receptive-field match -- WTConv at k=5 with L levels reaches
               5*2^L >= 10 pixels, so the plain convolutions see strictly less
               context and the comparison is conservative in WTConv's favour.

The wavelet methods are measured under `homogeneous` only; the two protocols
differ solely in the kernel size given to the plain convolutions, so measuring
WTConv twice would be redundant. `make_tables.py` takes every ratio against the
homogeneous reference.

Usage
-----
    python bench.py --device cuda --out ../results/bench_cuda.json
    python bench.py --device cuda --mode fwd_bwd --out ../results/bench_cuda_bwd.json

Add --quick for a smoke test over a single small configuration.
"""

from __future__ import annotations

import argparse
import itertools
import traceback
from dataclasses import asdict
from pathlib import Path

import torch

from common import (
    BUILDERS, DTYPES, Measurement, WAVELET_METHODS, build_pair, environment,
    summarise, time_module, write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda", choices=["cuda"],
                   help="CUDA only; the paper reports no other device")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--protocol", default="both",
                   choices=["homogeneous", "dropin", "both"])
    p.add_argument("--mode", default="fwd", choices=["fwd", "fwd_bwd", "both"])
    p.add_argument("--methods", default="depthwise,dense,reference,fused_cuda",
                   help="comma-separated subset of " + ",".join(BUILDERS))
    p.add_argument("--dtypes", default="fp32,fp16")
    p.add_argument("--levels", default="1,2,3,4,5")
    p.add_argument("--spatial", default="128,256,512")
    p.add_argument("--channels", default="32,64,128")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--kernel-size", type=int, default=5,
                   help="k for WTConv and, under 'homogeneous', for the plain convs")
    p.add_argument("--dropin-kernel-size", type=int, default=7,
                   help="k for the plain convs under the 'dropin' protocol")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def _ints(s: str) -> list[int]:
    return [int(v) for v in s.split(",") if v]


def warm_device(seconds: float = 20.0) -> None:
    """
    Bring the GPU to a steady clock state before the first measurement.

    On an idle device the first configurations are timed while the clocks are
    still ramping, which inflates their means and their p10-p90 spread without
    inflating anything measured later; the effect is large enough (means up to
    2x the steady-state value on the smallest shapes) to distort a geometric
    mean taken over the sweep. A short dense load first costs one minute and
    removes the artifact.
    """
    import time
    a = torch.randn(4096, 4096, device="cuda")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(10):
            a = a @ a.T
            a = a / a.norm()
    torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()
    print(f"device warmed for {seconds:.0f}s")


def main() -> None:
    args = parse_args()
    methods = [m for m in args.methods.split(",") if m]
    dtypes = [d for d in args.dtypes.split(",") if d]
    levels = _ints(args.levels)
    spatials = _ints(args.spatial)
    channels = _ints(args.channels)
    protocols = ["homogeneous", "dropin"] if args.protocol == "both" else [args.protocol]
    modes = ["fwd", "fwd_bwd"] if args.mode == "both" else [args.mode]

    if args.quick:
        levels, spatials, channels = [1, 2], [128], [32]
        args.iters, args.warmup = 5, 3

    if not args.quick:
        warm_device()

    rows: list[dict] = []
    grid = list(itertools.product(protocols, modes, dtypes, levels, spatials, channels))
    print(f"up to {len(grid) * len(methods)} measurements to take")

    for protocol, mode, dt_name, L, S, C in grid:
        dtype = DTYPES[dt_name]
        x = torch.randn(args.batch, C, S, S, device=args.device, dtype=dtype)

        for method in methods:
            is_wt = method in WAVELET_METHODS

            # The protocols differ only in the plain convolutions' kernel size,
            # so the wavelet methods are measured once, under 'homogeneous'.
            if protocol != "homogeneous" and is_wt:
                continue

            K = args.kernel_size if (protocol == "homogeneous" or is_wt) \
                else args.dropin_kernel_size

            base = dict(method=method, dtype=dt_name, levels=L, kernel_size=K,
                        batch=args.batch, channels=C, spatial=S, mode=mode,
                        iters=args.iters)

            # A plain convolution has no notion of decomposition level; record
            # it once (at L=1) and let make_tables.py broadcast it.
            if not is_wt and L != levels[0]:
                continue

            try:
                mod = build_pair(method, C, K, L, dtype, args.device, seed=args.seed)
                lat, peak = time_module(mod, x, mode=mode, warmup=args.warmup,
                                        iters=args.iters, device=args.device)
                m = Measurement(**base, **summarise(lat), peak_mem_mib=peak)
            except Exception as exc:                       # OOM, unsupported dtype, ...
                traceback.print_exc()
                m = Measurement(**base, latency_ms_mean=float("nan"),
                                latency_ms_median=float("nan"), latency_ms_std=float("nan"),
                                latency_ms_p10=float("nan"), latency_ms_p90=float("nan"),
                                peak_mem_mib=float("nan"), ok=False,
                                error=f"{type(exc).__name__}: {exc}")
            finally:
                del_mod = locals().get("mod")
                if del_mod is not None:
                    del del_mod
                torch.cuda.empty_cache()

            rows.append({"protocol": protocol, **asdict(m)})
            status = f"{m.latency_ms_mean:8.3f} ms  {m.peak_mem_mib:8.1f} MiB" if m.ok else "FAILED"
            print(f"{protocol:11s} {mode:7s} {dt_name} L={L} {C:4d}c {S:4d}px "
                  f"{method:13s} k={K}  {status}")

        del x
        torch.cuda.empty_cache()

    write_json(args.out, {"env": environment(), "args": vars(args) | {"out": str(args.out)},
                          "rows": rows})


if __name__ == "__main__":
    main()
