"""
Kernel-level tests for the fused CUDA Haar kernels.

Each kernel is checked against the operation it replaces, built from the original
WTConv wavelet filters (WTConv/wtconv/util/wavelet.py):

    haar2d                  vs wavelet_2d_transform
    fused_haar_conv_scale   vs haar -> grouped conv2d -> scale
    run_ihaar_cascade       vs the reference bottom-up reconstruction loop
    wavelet_branch          vs the whole reference wavelet branch
    fused weight gradient   vs the cuDNN fallback it replaces
    gradients               vs autograd through those reference compositions

TF32 is disabled: the reference paths are cuDNN convolutions, which would
otherwise run with ~1e-3 relative error while the fused kernels accumulate in
true fp32.

Usage:
    python tests/cuda_metal_tests/test_fused_kernels.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'WTConv'))

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

from wtconv.util import wavelet
from cuda_haar import haar_cuda as H

DEV = 'cuda'
_failures = []


def check(name, got, want, tol=2e-5):
    diff = (got - want).abs().max().item()
    ok = diff < tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:52s} maxdiff={diff:.3e}")
    if not ok:
        _failures.append(name)
    return ok


def ref_haar(x):
    f, _ = wavelet.create_2d_wavelet_filter('db1', x.shape[1], x.shape[1], torch.float32)
    return wavelet.wavelet_2d_transform(x, f.to(x.device, x.dtype))


def ref_ihaar(c):
    _, f = wavelet.create_2d_wavelet_filter('db1', c.shape[1], c.shape[1], torch.float32)
    return wavelet.inverse_2d_wavelet_transform(c, f.to(c.device, c.dtype))


def ref_fused(x, w, s, K):
    """Reference for one wavelet level: haar -> depthwise conv -> scale."""
    c = ref_haar(x)
    B, C, _, h, w_ = c.shape
    out = F.conv2d(c.reshape(B, C * 4, h, w_), w, padding=K // 2, groups=C * 4)
    return (out * s.reshape(1, C * 4, 1, 1)).reshape(B, C, 4, h, w_)


def ref_cascade(levels, out_hw):
    """Reference bottom-up reconstruction, as in WTConv2d.forward."""
    nxt = 0
    for i in range(len(levels) - 1, -1, -1):
        c = levels[i].clone()
        c[:, :, 0] = c[:, :, 0] + nxt
        rec = ref_ihaar(c)
        tgt = (levels[i - 1].shape[3], levels[i - 1].shape[4]) if i > 0 else out_hw
        nxt = rec[:, :, :tgt[0], :tgt[1]]
    return nxt


def level_shapes(H_, W_, levels):
    """Spatial shape of each level's coefficient grid (ceil-halving chain)."""
    out, h, w = [], H_, W_
    for _ in range(levels):
        h, w = (h + 1) // 2, (w + 1) // 2
        out.append((h, w))
    return out


def test_haar():
    print("\n[forward Haar]")
    for (B, C, h, w) in [(2, 3, 8, 8), (1, 1, 16, 32), (3, 8, 64, 64), (2, 5, 7, 9)]:
        x = torch.randn(B, C, h, w, device=DEV)
        xp = F.pad(x, (0, w % 2, 0, h % 2))
        check(f"haar2d {B}x{C}x{h}x{w}", H.haar2d(x), ref_haar(xp))


def test_fused_forward():
    print("\n[fused Haar -> conv -> scale]")
    for K in [1, 3, 5, 7, 9]:
        for (B, C, h, w) in [(2, 4, 16, 16), (1, 3, 32, 64), (2, 16, 64, 64)]:
            torch.manual_seed(0)
            x = torch.randn(B, C, h, w, device=DEV)
            weight = torch.randn(C * 4, 1, K, K, device=DEV) * 0.3
            scale = torch.rand(1, C * 4, 1, 1, device=DEV) + 0.2
            got, ll = H.fused_haar_conv_scale(x, weight, scale, K, return_ll=True)
            check(f"fused K={K} {B}x{C}x{h}x{w}", got, ref_fused(x, weight, scale, K))
            check(f"  raw LL K={K} {B}x{C}x{h}x{w}", ll, ref_haar(x)[:, :, 0])


def test_ihaar_cascade():
    print("\n[inverse cascade, with and without fused add]")
    for L in [1, 2, 3, 4, 5]:
        for (B, C, h, w) in [(2, 4, 64, 64), (1, 3, 48, 80), (2, 2, 33, 47)]:
            torch.manual_seed(1)
            levels = [torch.randn(B, C, 4, a, b, device=DEV)
                      for a, b in level_shapes(h, w, L)]
            add = torch.randn(B, C, h, w, device=DEV)
            want = ref_cascade(levels, (h, w))
            check(f"ihaar L={L} {B}x{C}x{h}x{w}",
                  H.run_ihaar_cascade(levels, (h, w)), want)
            check(f"  +fused add L={L} {B}x{C}x{h}x{w}",
                  H.run_ihaar_cascade(levels, (h, w), add_tensor=add), want + add)


def test_wavelet_branch():
    """Whole branch (one autograd node) vs haar -> conv -> cascade -> add."""
    print("\n[fused wavelet branch]")
    for L in [1, 2, 3, 5]:
        for K in [3, 5]:
            for (B, C, h, w) in [(2, 8, 64, 64), (2, 4, 33, 47)]:
                torch.manual_seed(4)
                x = torch.randn(B, C, h, w, device=DEV)
                base = torch.randn(B, C, h, w, device=DEV)
                ws = [torch.randn(C * 4, 1, K, K, device=DEV) * 0.2 for _ in range(L)]
                ss = [torch.rand(1, C * 4, 1, 1, device=DEV) * 0.3 for _ in range(L)]

                got = H.wavelet_branch(x, base, ws, ss, K)

                # reference: per-level pad + haar + conv + scale, then reconstruct
                levels, curr = [], x
                for i in range(L):
                    ch, cw = curr.shape[2], curr.shape[3]
                    if (ch & 1) or (cw & 1):
                        curr = F.pad(curr, (0, cw & 1, 0, ch & 1))
                    levels.append(ref_fused(curr, ws[i], ss[i], K))
                    curr = ref_haar(curr)[:, :, 0]
                want = ref_cascade(levels, (h, w)) + base

                check(f"branch L={L} K={K} {B}x{C}x{h}x{w}", got, want, 3e-5)


def test_branch_grads():
    print("\n[fused wavelet branch gradients]")
    for L in [1, 2, 3]:
        for (B, C, h, w) in [(2, 6, 32, 32), (2, 4, 31, 33)]:
            torch.manual_seed(5)
            K = 3
            x = torch.randn(B, C, h, w, device=DEV, requires_grad=True)
            base = torch.randn(B, C, h, w, device=DEV, requires_grad=True)
            ws = [(torch.randn(C * 4, 1, K, K, device=DEV) * 0.2).requires_grad_()
                  for _ in range(L)]
            ss = [(torch.rand(1, C * 4, 1, 1, device=DEV) * 0.3).requires_grad_()
                  for _ in range(L)]
            g = torch.randn(B, C, h, w, device=DEV)

            (H.wavelet_branch(x, base, ws, ss, K) * g).sum().backward()
            got = [x.grad.clone(), base.grad.clone()] + \
                  [p.grad.clone() for p in ws] + [p.grad.clone() for p in ss]

            rx = x.detach().clone().requires_grad_()
            rbase = base.detach().clone().requires_grad_()
            rws = [p.detach().clone().requires_grad_() for p in ws]
            rss = [p.detach().clone().requires_grad_() for p in ss]
            levels, curr = [], rx
            for i in range(L):
                ch, cw = curr.shape[2], curr.shape[3]
                if (ch & 1) or (cw & 1):
                    curr = F.pad(curr, (0, cw & 1, 0, ch & 1))
                levels.append(ref_fused(curr, rws[i], rss[i], K))
                curr = ref_haar(curr)[:, :, 0]
            ((ref_cascade(levels, (h, w)) + rbase) * g).sum().backward()
            want = [rx.grad, rbase.grad] + [p.grad for p in rws] + [p.grad for p in rss]

            names = ['grad_input', 'grad_base'] + \
                    [f'grad_w{i + 1}' for i in range(L)] + \
                    [f'grad_s{i + 1}' for i in range(L)]
            for name, a, b in zip(names, got, want):
                check(f"branch L={L} {h}x{w} {name}", a, b, 3e-4)


def test_fused_grads():
    print("\n[fused Haar -> conv -> scale gradients]")
    for K in [3, 5]:
        for with_ll in [False, True]:
            torch.manual_seed(2)
            B, C, h, w = 2, 6, 32, 32
            x = torch.randn(B, C, h, w, device=DEV, requires_grad=True)
            weight = (torch.randn(C * 4, 1, K, K, device=DEV) * 0.3).requires_grad_()
            scale = (torch.rand(1, C * 4, 1, 1, device=DEV) + 0.2).requires_grad_()
            g_out = torch.randn(B, C, 4, h // 2, w // 2, device=DEV)
            g_ll = torch.randn(B, C, h // 2, w // 2, device=DEV)

            if with_ll:
                out, ll = H.fused_haar_conv_scale(x, weight, scale, K, return_ll=True)
                ((out * g_out).sum() + (ll * g_ll).sum()).backward()
            else:
                out = H.fused_haar_conv_scale(x, weight, scale, K, return_ll=False)
                (out * g_out).sum().backward()
            got = [x.grad.clone(), weight.grad.clone(), scale.grad.clone()]

            refs = [t.detach().clone().requires_grad_() for t in (x, weight, scale)]
            loss = (ref_fused(refs[0], refs[1], refs[2], K) * g_out).sum()
            if with_ll:
                loss = loss + (ref_haar(refs[0])[:, :, 0] * g_ll).sum()
            loss.backward()

            tag = f"K={K} ll={with_ll}"
            for name, a, b, tol in zip(['grad_input', 'grad_weight', 'grad_scale'],
                                       got, [r.grad for r in refs],
                                       [3e-5, 3e-4, 3e-4]):
                check(f"{name} {tag}", a, b, tol)


def test_grad_weight_kernel():
    """Fused weight-gradient kernel vs the cuDNN fallback it replaces."""
    print("\n[fused weight gradient vs cuDNN fallback]")
    mod = H._get_module()
    real_max = mod.fused_haar_grad_weight_max_k
    try:
        for K in [1, 3, 5]:
            for (B, C, h, w) in [(2, 8, 32, 32), (3, 4, 64, 48), (1, 16, 16, 16)]:
                torch.manual_seed(6)
                x = torch.randn(B, C, h, w, device=DEV)
                weight = torch.randn(C * 4, 1, K, K, device=DEV) * 0.3
                scale = torch.rand(1, C * 4, 1, 1, device=DEV) + 0.2
                g = torch.randn(B, C, 4, h // 2, w // 2, device=DEV)

                gw_f, gs_f = H._grad_weight_scale(x, g, weight, scale, K)
                mod.fused_haar_grad_weight_max_k = lambda: 0   # force cuDNN path
                gw_c, gs_c = H._grad_weight_scale(x, g, weight, scale, K)
                mod.fused_haar_grad_weight_max_k = real_max

                check(f"grad_weight K={K} {B}x{C}x{h}x{w}", gw_f, gw_c, 2e-4)
                check(f"grad_scale  K={K} {B}x{C}x{h}x{w}", gs_f, gs_c, 2e-3)
    finally:
        mod.fused_haar_grad_weight_max_k = real_max


def test_cascade_grads():
    print("\n[inverse cascade gradients]")
    for L in [1, 2, 3, 5]:
        torch.manual_seed(3)
        B, C, h, w = 2, 4, 48, 48
        levels = [torch.randn(B, C, 4, a, b, device=DEV, requires_grad=True)
                  for a, b in level_shapes(h, w, L)]
        add = torch.randn(B, C, h, w, device=DEV, requires_grad=True)
        g = torch.randn(B, C, h, w, device=DEV)

        (H._ihaar(levels, add, (h, w)) * g).sum().backward()
        got = [l.grad.clone() for l in levels] + [add.grad.clone()]

        ref_levels = [l.detach().clone().requires_grad_() for l in levels]
        ref_add = add.detach().clone().requires_grad_()
        ((ref_cascade(ref_levels, (h, w)) + ref_add) * g).sum().backward()
        want = [l.grad for l in ref_levels] + [ref_add.grad]

        names = [f'level{i + 1}' for i in range(L)] + ['add']
        for name, a, b in zip(names, got, want):
            check(f"ihaar grad L={L} d/d {name}", a, b)


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1
    print(f"Fused Haar kernels vs WTConv reference ops on {torch.cuda.get_device_name(0)}")
    test_haar()
    test_fused_forward()
    test_ihaar_cascade()
    test_wavelet_branch()
    test_fused_grads()
    test_grad_weight_kernel()
    test_cascade_grads()
    test_branch_grads()
    print(f"\n{'ALL PASS' if not _failures else str(len(_failures)) + ' FAILURES: ' + str(_failures)}")
    return 1 if _failures else 0


if __name__ == '__main__':
    sys.exit(main())
