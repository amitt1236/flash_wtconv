#!/usr/bin/env python3
"""
Export a one-level WTConv Core ML module from wtconv_model/wtconv.py weights.

The exported model computes exactly the convolution+scale parts used by wtconv.py:
- base_output = depthwise_conv(base_input, base_weight) * base_scale + base_bias
- wavelet_output = depthwise_conv(wavelet_input, wavelet_weight) * wavelet_scale

You run Haar forward/inverse with Metal in Swift, and use this Core ML model for convs.
"""

import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export WTConv conv modules to Core ML")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("../Models/WTConvConvModules.mlpackage"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


class WTConvConvModules(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        base_weight: torch.Tensor,
        base_bias: torch.Tensor,
        base_scale: torch.Tensor,
        wavelet_weight: torch.Tensor,
        wavelet_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.padding = kernel_size // 2

        self.register_buffer("base_weight", base_weight.detach().cpu())
        self.register_buffer("base_bias", base_bias.detach().cpu())
        self.register_buffer("base_scale", base_scale.detach().cpu())
        self.register_buffer("wavelet_weight", wavelet_weight.detach().cpu())
        self.register_buffer("wavelet_scale", wavelet_scale.detach().cpu())

    def forward(self, base_input: torch.Tensor, wavelet_input: torch.Tensor):
        base = F.conv2d(
            base_input,
            self.base_weight,
            bias=None,
            stride=1,
            padding=self.padding,
            groups=self.channels,
        )
        base = base * self.base_scale + self.base_bias.view(1, -1, 1, 1)

        wavelet = F.conv2d(
            wavelet_input,
            self.wavelet_weight,
            bias=None,
            stride=1,
            padding=self.padding,
            groups=self.channels * 4,
        )
        wavelet = wavelet * self.wavelet_scale

        return base, wavelet


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from wtconv_model.wtconv import WTConv2d

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is required to instantiate wtconv_model/wtconv.py in this repo. "
            "Run on Apple Silicon/macOS with MPS enabled."
        )

    wtconv = WTConv2d(
        in_channels=args.channels,
        out_channels=args.channels,
        kernel_size=args.kernel_size,
        wt_levels=1,
        bias=True,
        device="mps",
    ).eval()

    module = WTConvConvModules(
        channels=args.channels,
        kernel_size=args.kernel_size,
        base_weight=wtconv.base_conv.weight,
        base_bias=wtconv.base_conv.bias,
        base_scale=wtconv.base_scale,
        wavelet_weight=wtconv.wavelet_convs[0].weight,
        wavelet_scale=wtconv.wavelet_scales[0],
    ).eval()

    b, c, h, w = 1, args.channels, args.height, args.width
    h2, w2 = (h + 1) // 2, (w + 1) // 2

    base_example = torch.randn(b, c, h, w, dtype=torch.float32)
    wavelet_example = torch.randn(b, c * 4, h2, w2, dtype=torch.float32)

    traced = torch.jit.trace(module, (base_example, wavelet_example))

    try:
        import coremltools as ct
    except ImportError as exc:
        raise RuntimeError("coremltools is required. Install with: pip install coremltools") from exc

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="base_input", shape=base_example.shape),
            ct.TensorType(name="wavelet_input", shape=wavelet_example.shape),
        ],
    )

    out_path = (Path(__file__).resolve().parent / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))

    print(f"Saved Core ML model: {out_path}")
    print(f"Base input shape: {tuple(base_example.shape)}")
    print(f"Wavelet input shape: {tuple(wavelet_example.shape)}")


if __name__ == "__main__":
    main()
