#!/usr/bin/env python3
"""
Numerical check: Swift(CoreML+Metal) output vs PyTorch wtconv_model.wtconv.WTConv2d.

This script:
1) Instantiates WTConv2d (wt_levels=1) in PyTorch on MPS.
2) Exports matching conv branches to Core ML.
3) Writes deterministic input tensor to float32 binary.
4) Runs Swift executable with that input.
5) Compares outputs with max/mean/rmse errors.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wtconv_model.wtconv import WTConv2d  # noqa: E402
from export_wtconv_coreml import WTConvConvModules  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Swift output against PyTorch WTConv")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--workdir", type=Path, default=Path("tmp_output_check"))
    parser.add_argument(
        "--swift-bin",
        type=Path,
        default=Path(".build/release/WTConvCoreMLRunner"),
        help="Path relative to coreml_metal_wtconv_swift/",
    )
    parser.add_argument(
        "--skip-swift-build",
        action="store_true",
        help="Skip rebuilding the Swift executable before running it.",
    )
    return parser.parse_args()


def export_coreml_model(
    wtconv: WTConv2d,
    channels: int,
    kernel_size: int,
    height: int,
    width: int,
    out_path: Path,
) -> None:
    try:
        import coremltools as ct
    except ImportError as exc:
        raise RuntimeError("coremltools is required. Install with: pip install coremltools") from exc

    module = WTConvConvModules(
        channels=channels,
        kernel_size=kernel_size,
        base_weight=wtconv.base_conv.weight,
        base_bias=wtconv.base_conv.bias,
        base_scale=wtconv.base_scale,
        wavelet_weight=wtconv.wavelet_convs[0].weight,
        wavelet_scale=wtconv.wavelet_scales[0],
    ).eval()

    b, c, h, w = 1, channels, height, width
    h2, w2 = (h + 1) // 2, (w + 1) // 2
    base_example = torch.randn(b, c, h, w, dtype=torch.float32)
    wavelet_example = torch.randn(b, c * 4, h2, w2, dtype=torch.float32)

    traced = torch.jit.trace(module, (base_example, wavelet_example))

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="base_input", shape=base_example.shape),
            ct.TensorType(name="wavelet_input", shape=wavelet_example.shape),
        ],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))


def main() -> int:
    args = parse_args()

    # metal_haar/haar.mm resolves shader sources via current working directory.
    # Force repo root so it can find ROOT/metal_haar/*.metal reliably.
    os.chdir(ROOT)

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for this check.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    coreml_dir = ROOT / "coreml_metal_wtconv_swift"
    workdir = (coreml_dir / args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    model_path = workdir / "WTConvConvModules.mlpackage"
    input_path = workdir / "input.f32"
    swift_output_path = workdir / "swift_output.f32"

    wtconv = WTConv2d(
        in_channels=args.channels,
        out_channels=args.channels,
        kernel_size=args.kernel_size,
        wt_levels=1,
        bias=True,
        device="mps",
    ).eval().to("mps")

    # Deterministic input in NCHW float32
    x_np = np.random.uniform(-1.0, 1.0, size=(1, args.channels, args.height, args.width)).astype(np.float32)
    x_pt = torch.from_numpy(x_np)

    # PyTorch reference output
    with torch.no_grad():
        y_pt = wtconv(x_pt.to("mps")).cpu().contiguous().numpy()

    # Save input as raw float32 for Swift
    x_np.tofile(input_path)

    # Export Core ML model from the exact same WTConv weights
    export_coreml_model(
        wtconv=wtconv,
        channels=args.channels,
        kernel_size=args.kernel_size,
        height=args.height,
        width=args.width,
        out_path=model_path,
    )

    if not args.skip_swift_build:
        print("Building Swift runner...")
        build_proc = subprocess.run(
            ["swift", "build", "-c", "release"],
            cwd=str(coreml_dir),
            capture_output=True,
            text=True,
        )
        print(build_proc.stdout)
        if build_proc.returncode != 0:
            print(build_proc.stderr, file=sys.stderr)
            raise RuntimeError(f"Swift build failed with code {build_proc.returncode}")

    swift_bin = (coreml_dir / args.swift_bin).resolve()
    if not swift_bin.exists():
        raise FileNotFoundError(f"Swift binary not found: {swift_bin}. Build first: swift build -c release")

    cmd = [
        str(swift_bin),
        "--model",
        str(model_path),
        "--channels",
        str(args.channels),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--input-f32",
        str(input_path),
        "--output-f32",
        str(swift_output_path),
    ]

    print("Running Swift pipeline...")
    proc = subprocess.run(cmd, cwd=str(coreml_dir), capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Swift runner failed with code {proc.returncode}")

    y_swift = np.fromfile(swift_output_path, dtype=np.float32)
    expected_count = y_pt.size
    if y_swift.size != expected_count:
        raise RuntimeError(f"Swift output size mismatch. expected={expected_count}, got={y_swift.size}")

    y_swift = y_swift.reshape(y_pt.shape)

    abs_err = np.abs(y_swift - y_pt)
    max_abs = float(abs_err.max())
    mean_abs = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean((y_swift - y_pt) ** 2)))
    passed = np.allclose(y_swift, y_pt, atol=args.atol, rtol=args.rtol)

    print("Comparison metrics")
    print(f"max_abs={max_abs:.8f}")
    print(f"mean_abs={mean_abs:.8f}")
    print(f"rmse={rmse:.8f}")
    print(f"allclose(atol={args.atol}, rtol={args.rtol})={passed}")

    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
