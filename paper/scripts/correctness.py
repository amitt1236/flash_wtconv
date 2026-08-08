#!/usr/bin/env python3
"""
Numerical-equivalence check between the reference WTConv2d of Finder et al.
and the fused CUDA backend.

The fused kernels are not an approximation: they compute the same multilinear
map with a different association order. In exact arithmetic the outputs are
identical, so the only admissible discrepancy is floating-point reassociation
error. This script quantifies that error for the forward pass and for every
gradient the layer produces.

It also independently verifies the two structural properties the paper relies
on, rather than taking them on faith:

  (S1) subband packing: the reference packs the wavelet axis as channel
       index 4c+s with s in (LL, LH, HL, HH), matching the fused kernels;
  (S2) self-inverse orthogonality: the Haar analysis matrix H satisfies
       H = H^T = H^{-1}, which is what makes the backward pass of the
       forward transform equal to the forward transform itself.

Usage
-----
    python correctness.py --device cuda --out ../results/correctness.json
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import torch

from common import DTYPES, _reference_wtconv, build_pair, environment, write_json

# Tolerances. fp32 is held to near machine precision; the reduced-precision
# thresholds are set from the accumulated rounding of a K^2-term dot product
# in the respective format, not tuned to make the test pass.
TOL = {
    "fp32": dict(atol=1e-5, rtol=1e-4),
    "fp16": dict(atol=2e-2, rtol=1e-2),
    "bf16": dict(atol=1e-1, rtol=5e-2),
}


# =============================================================================
# Structural checks
# =============================================================================

def check_haar_matrix() -> dict:
    """(S2) H = H^T = H^{-1} for the normalised 4x4 Haar analysis matrix."""
    H = 0.5 * torch.tensor([
        [1.,  1.,  1.,  1.],     # LL
        [1.,  1., -1., -1.],     # LH
        [1., -1.,  1., -1.],     # HL
        [1., -1., -1.,  1.],     # HH
    ], dtype=torch.float64)
    return {
        "symmetric": torch.allclose(H, H.T),
        "self_inverse": torch.allclose(H @ H, torch.eye(4, dtype=torch.float64)),
        "orthogonal": torch.allclose(H @ H.T, torch.eye(4, dtype=torch.float64)),
        "max_dev_from_identity": float((H @ H - torch.eye(4, dtype=torch.float64)).abs().max()),
    }


def check_subband_packing(device: str) -> dict:
    """
    (S1) Apply the reference's own db1 filter bank to a one-hot 2x2 patch and
    confirm the resulting coefficient order is (LL, LH, HL, HH) with the sign
    pattern the fused kernels assume:
        LL = (a+b+c+d)/2   LH = (a+b-c-d)/2
        HL = (a-b+c-d)/2   HH = (a-b-c+d)/2   for patch [[a,b],[c,d]].
    """
    from wtconv.util import wavelet

    filt, _ = wavelet.create_2d_wavelet_filter("db1", 1, 1, torch.float64)
    patch = torch.tensor([[[[1., 2.], [3., 4.]]]], dtype=torch.float64)  # a,b,c,d
    got = wavelet.wavelet_2d_transform(patch, filt).reshape(4)
    a, b, c, d = 1., 2., 3., 4.
    want = torch.tensor([(a + b + c + d) / 2, (a + b - c - d) / 2,
                         (a - b + c - d) / 2, (a - b - c + d) / 2], dtype=torch.float64)
    return {
        "matches_fused_convention": torch.allclose(got, want),
        "reference_coeffs": got.tolist(),
        "expected_coeffs": want.tolist(),
        "max_abs_diff": float((got - want).abs().max()),
    }


# =============================================================================
# Forward / backward equivalence
# =============================================================================

def _err(a: torch.Tensor, b: torch.Tensor) -> dict:
    a32, b32 = a.detach().float(), b.detach().float()
    diff = (a32 - b32).abs()
    denom = b32.abs().clamp_min(1e-6)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float((diff / denom).max()),
        "rms": float(diff.pow(2).mean().sqrt()),
        "ref_scale": float(b32.abs().mean()),
    }


def compare(method: str, C: int, K: int, L: int, S: int, B: int,
            dt_name: str, device: str, seed: int) -> dict:
    dtype = DTYPES[dt_name]
    ref = build_pair("reference", C, K, L, dtype, device, seed=seed)
    fast = build_pair(method, C, K, L, dtype, device, seed=seed)

    torch.manual_seed(seed + 1)
    x = torch.randn(B, C, S, S, device=device, dtype=dtype)
    xr = x.clone().requires_grad_(True)
    xf = x.clone().requires_grad_(True)

    yr = ref(xr)
    yf = fast(xf)

    # A fixed random cotangent exercises every output element, unlike .sum()
    # which can mask sign-cancelling errors.
    torch.manual_seed(seed + 2)
    g = torch.randn_like(yr)
    yr.backward(g)
    yf.backward(g)

    out = {"forward": _err(yf, yr), "grad_input": _err(xf.grad, xr.grad)}

    # --- parameter gradients, matched across the two naming schemes ----------
    # The base branch is the only one carrying a bias, and the fused path folds
    # the scale into it as well as into the weight, so grad_base_bias is a
    # distinct check on the fold of Eq. (8) -- not implied by grad_base_weight.
    ref_params = {"base_weight": ref.base_conv.weight, "base_scale": ref.base_scale.weight}
    if ref.base_conv.bias is not None:
        ref_params["base_bias"] = ref.base_conv.bias
    for l in range(L):
        ref_params[f"wt_weight_{l}"] = ref.wavelet_convs[l].weight
        ref_params[f"wt_scale_{l}"] = ref.wavelet_scale[l].weight

    if hasattr(fast, "base_conv"):
        fast_params = {"base_weight": fast.base_conv.weight, "base_scale": fast.base_scale.weight}
        if fast.base_conv.bias is not None:
            fast_params["base_bias"] = fast.base_conv.bias
        for l in range(L):
            fast_params[f"wt_weight_{l}"] = fast.wavelet_convs[l].weight
            fast_params[f"wt_scale_{l}"] = fast.wavelet_scale[l].weight
    else:
        fast_params = {"base_weight": fast.base_weight, "base_scale": fast.base_scale}
        if getattr(fast, "base_bias", None) is not None:
            fast_params["base_bias"] = fast.base_bias
        for l in range(L):
            fast_params[f"wt_weight_{l}"] = fast.wt_weights[l]
            fast_params[f"wt_scale_{l}"] = fast.wt_scales[l]

    for name, rp in ref_params.items():
        fp = fast_params.get(name)
        if fp is None:
            out[f"grad_{name}"] = {"missing": True, "ref_none": False, "fast_none": True}
        elif rp.grad is None or fp.grad is None:
            out[f"grad_{name}"] = {"missing": True,
                                   "ref_none": rp.grad is None, "fast_none": fp.grad is None}
        else:
            out[f"grad_{name}"] = _err(fp.grad, rp.grad)

    tol = TOL[dt_name]
    out["pass"] = all(
        v["max_abs"] <= tol["atol"] + tol["rtol"] * v["ref_scale"]
        for v in out.values() if isinstance(v, dict) and "max_abs" in v
    )
    out["tolerance"] = tol
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda", choices=["cuda"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--methods", default="fused_cuda")
    p.add_argument("--dtypes", default="fp32,fp16,bf16")
    p.add_argument("--levels", default="1,2,3,4,5")
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--spatial", type=int, default=64)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    report = {
        "env": environment(),
        "structural": {
            "haar_matrix": check_haar_matrix(),
            "subband_packing": check_subband_packing(args.device),
        },
        "cases": [],
    }
    print("structural:", report["structural"]["haar_matrix"])
    print("packing   :", report["structural"]["subband_packing"]["matches_fused_convention"])

    for method in args.methods.split(","):
        for dt in args.dtypes.split(","):
            for L in (int(v) for v in args.levels.split(",")):
                case = dict(method=method, dtype=dt, levels=L,
                            channels=args.channels, spatial=args.spatial,
                            batch=args.batch, kernel_size=args.kernel_size)
                try:
                    case["result"] = compare(method, args.channels, args.kernel_size, L,
                                             args.spatial, args.batch, dt, args.device,
                                             args.seed)
                    verdict = "pass" if case["result"]["pass"] else "FAIL"
                    print(f"{method:13s} {dt} L={L}  fwd max_abs="
                          f"{case['result']['forward']['max_abs']:.3e}  "
                          f"dX max_abs={case['result']['grad_input']['max_abs']:.3e}  {verdict}")
                except Exception as exc:
                    traceback.print_exc()
                    case["error"] = f"{type(exc).__name__}: {exc}"
                report["cases"].append(case)

    write_json(args.out, report)


if __name__ == "__main__":
    main()
