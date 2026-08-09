"""
Golden-reference correctness test for the fused CUDA WTConv2d.

The reference is the original WTConv implementation (WTConv/wtconv/wtconv2d.py),
never the previous CUDA kernels. Covers forward outputs and gradients for every
parameter, across decomposition levels, kernel sizes, odd/even spatial sizes and
all supported dtypes.

Note on TF32: the reference builds the wavelet transform out of cuDNN
convolutions, which run in TF32 on Ampere+ by default (~1e-3 relative error).
The fused kernels accumulate in true fp32, so TF32 is disabled here to keep the
comparison meaningful.

Usage:
    python tests/cuda_metal_tests/test_wtconv_correctness.py
    python tests/cuda_metal_tests/test_wtconv_correctness.py --quick
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'WTConv'))

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

TOLERANCE = {
    torch.float32: dict(out=2e-5, grad=2e-4),
    torch.float16: dict(out=2e-2, grad=2e-1),
    torch.bfloat16: dict(out=1e-1, grad=1e0),
}


def get_classes():
    from wtconv_model.wtconv import WTConv2d
    from wtconv.wtconv2d import WTConv2d as WTConv2dNaive
    return WTConv2d, WTConv2dNaive


def copy_weights_to_naive(model, naive, depth):
    """Mirror the fused model's parameters onto the reference model."""
    with torch.no_grad():
        naive.base_conv.weight.copy_(model.base_weight)
        if model.base_bias is not None and naive.base_conv.bias is not None:
            naive.base_conv.bias.copy_(model.base_bias)
        naive.base_scale.weight.copy_(model.base_scale)
        for level in range(depth):
            naive.wavelet_convs[level].weight.copy_(model.wt_weights[level])
            naive.wavelet_scale[level].weight.copy_(model.wt_scales[level])


def param_pairs(model, naive, depth):
    """(name, fused param, reference param) triples, in a fixed order."""
    pairs = [('base_weight', model.base_weight, naive.base_conv.weight),
             ('base_scale', model.base_scale, naive.base_scale.weight)]
    if model.base_bias is not None:
        pairs.append(('base_bias', model.base_bias, naive.base_conv.bias))
    for i in range(depth):
        pairs.append((f'wt_weights[{i}]', model.wt_weights[i], naive.wavelet_convs[i].weight))
        pairs.append((f'wt_scales[{i}]', model.wt_scales[i], naive.wavelet_scale[i].weight))
    return pairs


def run_case(B, C, H, W, K, depth, dtype, stride=1, bias=True, verbose=True):
    """Compare forward + all gradients against the reference. Returns True on pass."""
    WTConv2d, WTConv2dNaive = get_classes()
    tol = TOLERANCE[dtype]

    torch.manual_seed(42)
    model = WTConv2d(C, C, kernel_size=K, wt_levels=depth, stride=stride,
                     bias=bias, device='cuda').cuda().to(dtype)
    naive = WTConv2dNaive(C, C, kernel_size=K, wt_levels=depth, stride=stride,
                          bias=bias).cuda().to(dtype)
    copy_weights_to_naive(model, naive, depth)

    x = torch.randn(B, C, H, W, device='cuda', dtype=dtype)
    x_f = x.clone().requires_grad_()
    x_n = x.clone().requires_grad_()

    out_f = model(x_f)
    out_n = naive(x_n)
    assert out_f.shape == out_n.shape, f"shape {out_f.shape} != {out_n.shape}"

    diffs = {}
    diffs['output'] = (out_f.float() - out_n.float()).abs().max().item()

    g = torch.randn_like(out_f)
    out_f.backward(g)
    out_n.backward(g)

    diffs['grad_input'] = (x_f.grad.float() - x_n.grad.float()).abs().max().item()
    for name, pf, pn in param_pairs(model, naive, depth):
        diffs[f'grad_{name}'] = (pf.grad.float() - pn.grad.float()).abs().max().item()

    ok = diffs['output'] < tol['out'] and all(
        v < tol['grad'] for k, v in diffs.items() if k != 'output')

    tag = (f"B{B} C{C} {H}x{W} K{K} L{depth} s{stride} "
           f"{'bias' if bias else 'nobias'} {str(dtype).split('.')[-1]}")
    if verbose:
        worst = max(diffs.items(), key=lambda kv: kv[1])
        print(f"  {'PASS' if ok else 'FAIL'}  {tag:52s} "
              f"out={diffs['output']:.2e} worst={worst[0]}:{worst[1]:.2e}")
        if not ok:
            for k, v in diffs.items():
                print(f"        {k:24s} {v:.3e}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='fp32, levels 1-3 only')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1

    print(f"Fused CUDA WTConv2d vs original WTConv reference on "
          f"{torch.cuda.get_device_name(0)}")

    results = []

    print("\n[levels x kernel sizes, fp32]")
    depths = [1, 2, 3] if args.quick else [1, 2, 3, 4, 5]
    kernels = [3, 5] if args.quick else [1, 3, 5, 7]
    for depth in depths:
        for K in kernels:
            results.append(run_case(2, 16, 64, 64, K, depth, torch.float32))

    print("\n[odd / non-square / small spatial sizes, fp32]")
    for (H, W) in [(63, 64), (64, 63), (33, 47), (48, 80), (7, 9), (2, 2), (17, 5)]:
        for depth in ([1, 3] if args.quick else [1, 2, 3, 5]):
            results.append(run_case(2, 8, H, W, 3, depth, torch.float32))

    print("\n[dtypes]")
    if not args.quick:
        for dtype in [torch.float16, torch.bfloat16]:
            for depth in [1, 3, 5]:
                results.append(run_case(2, 16, 64, 64, 5, depth, dtype))

    print("\n[stride, no-bias, channel counts]")
    results.append(run_case(2, 16, 64, 64, 5, 2, torch.float32, stride=2))
    results.append(run_case(2, 16, 64, 64, 5, 2, torch.float32, bias=False))
    results.append(run_case(1, 1, 32, 32, 3, 2, torch.float32))
    results.append(run_case(4, 96, 56, 56, 5, 3, torch.float32))

    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} cases passed")
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
