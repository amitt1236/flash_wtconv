#!/usr/bin/env python3
"""
Latency and peak-memory benchmark for Fast-WTConv.

Reproduces every throughput/memory table in the paper. Results are written as
raw JSON; `make_tables.py` turns that JSON into the LaTeX the paper \\inputs,
so no number is ever transcribed by hand.

Two protocols, following the paper:

  homogeneous  every method uses the same kernel size k (default 3). Isolates
               the cost of the wavelet machinery itself.

  rfmatch      WTConv with k=3 against dense/depthwise convolutions with k=7,
               i.e. a comparison at roughly matched receptive field rather
               than matched kernel size.

Usage
-----
    python bench.py --device cuda --out ../results/bench_cuda.json
    python bench.py --device cuda --mode fwd_bwd --out ../results/bench_cuda_bwd.json
    python bench.py --device mps  --methods reference,fused_cuda \\
                    --dtypes fp32,fp16 --out ../results/bench_metal.json

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
    BUILDERS, DTYPES, Measurement, build_pair, environment, summarise,
    time_module, write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--protocol", default="both",
                   choices=["homogeneous", "rfmatch", "both"])
    p.add_argument("--mode", default="fwd", choices=["fwd", "fwd_bwd", "both"])
    p.add_argument("--methods", default="depthwise,dense,reference,fused_cuda,fused_triton",
                   help="comma-separated subset of " + ",".join(BUILDERS))
    p.add_argument("--dtypes", default="fp32,fp16")
    p.add_argument("--levels", default="1,2,3,4,5")
    p.add_argument("--spatial", default="128,256,512")
    p.add_argument("--channels", default="32,64,128")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--kernel-size", type=int, default=3,
                   help="k for WTConv and, under 'homogeneous', for the plain convs")
    p.add_argument("--rf-kernel-size", type=int, default=7,
                   help="k for the plain convs under the 'rfmatch' protocol")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def _ints(s: str) -> list[int]:
    return [int(v) for v in s.split(",") if v]


def main() -> None:
    args = parse_args()
    methods = [m for m in args.methods.split(",") if m]
    dtypes = [d for d in args.dtypes.split(",") if d]
    levels = _ints(args.levels)
    spatials = _ints(args.spatial)
    channels = _ints(args.channels)
    protocols = ["homogeneous", "rfmatch"] if args.protocol == "both" else [args.protocol]
    modes = ["fwd", "fwd_bwd"] if args.mode == "both" else [args.mode]

    if args.quick:
        levels, spatials, channels = [1, 2], [128], [32]
        args.iters, args.warmup = 5, 3

    # The Triton backend is CUDA-only; the Metal path is exposed through the
    # same `fast_wtconv.wtconv` entry point that dispatches on device.
    if args.device != "cuda" and "fused_triton" in methods:
        print("note: dropping fused_triton (requires CUDA)")
        methods = [m for m in methods if m != "fused_triton"]

    rows: list[dict] = []
    grid = list(itertools.product(protocols, modes, dtypes, levels, spatials, channels))
    print(f"{len(grid) * len(methods)} measurements to take")

    for protocol, mode, dt_name, L, S, C in grid:
        dtype = DTYPES[dt_name]
        x = torch.randn(args.batch, C, S, S, device=args.device, dtype=dtype)

        for method in methods:
            # kernel size depends on protocol and on whether the method is a
            # wavelet method or a plain convolution
            is_wt = method in ("reference", "fused_cuda", "fused_triton")
            if protocol == "rfmatch" and not is_wt:
                K = args.rf_kernel_size
            else:
                K = args.kernel_size

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
                if args.device == "cuda":
                    torch.cuda.empty_cache()

            rows.append({"protocol": protocol, **asdict(m)})
            status = f"{m.latency_ms_mean:8.3f} ms  {m.peak_mem_mib:8.1f} MiB" if m.ok else "FAILED"
            print(f"{protocol:11s} {mode:7s} {dt_name} L={L} {C:4d}c {S:4d}px "
                  f"{method:13s} k={K}  {status}")

        del x
        if args.device == "cuda":
            torch.cuda.empty_cache()

    write_json(args.out, {"env": environment(), "args": vars(args) | {"out": str(args.out)},
                          "rows": rows})


if __name__ == "__main__":
    main()
