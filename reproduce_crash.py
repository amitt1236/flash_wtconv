import torch
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from cuda_haar.haar_cuda import haar2d_quint, haar2d_quad, haar2d_triple, haar2d_double, scaled_depthwise_conv, ihaar2d_quint, ihaar2d_quad, ihaar2d_triple, ihaar2d_double

def test_stage(B, C, H, W, levels_count):
    print(f"\nTesting Stage: B={B}, C={C}, H={H}, W={W}, Levels={levels_count}")
    device = torch.device('cuda')
    x = torch.randn(B, C, H, W, device=device)
    
    padding = 2 # kernel 5 // 2
    
    try:
        if levels_count == 5:
            levels = haar2d_quint(x)
            print("haar2d_quint OK")
            convs = []
            for i, level in enumerate(levels):
                b, c, _, h, w = level.shape
                flat = level.view(b, c * 4, h, w)
                weight = torch.randn(c * 4, 1, 5, 5, device=device)
                scale = torch.ones(1, c * 4, 1, 1, device=device) * 0.1
                out = scaled_depthwise_conv(flat, weight, scale, padding)
                convs.append(out.view(b, c, 4, h, w))
            print("convs OK")
            res = ihaar2d_quint(*convs, (H, W))
            print("ihaar OK")
            
        elif levels_count == 4:
            levels = haar2d_quad(x)
            print("haar2d_quad OK")
            convs = []
            for i, level in enumerate(levels):
                b, c, _, h, w = level.shape
                flat = level.view(b, c * 4, h, w)
                weight = torch.randn(c * 4, 1, 5, 5, device=device)
                scale = torch.ones(1, c * 4, 1, 1, device=device) * 0.1
                out = scaled_depthwise_conv(flat, weight, scale, padding)
                convs.append(out.view(b, c, 4, h, w))
            print("convs OK")
            res = ihaar2d_quad(*convs, (H, W))
            print("ihaar OK")

        elif levels_count == 3:
            levels = haar2d_triple(x)
            print("haar2d_triple OK")
            convs = []
            for i, level in enumerate(levels):
                b, c, _, h, w = level.shape
                flat = level.view(b, c * 4, h, w)
                weight = torch.randn(c * 4, 1, 5, 5, device=device)
                scale = torch.ones(1, c * 4, 1, 1, device=device) * 0.1
                out = scaled_depthwise_conv(flat, weight, scale, padding)
                convs.append(out.view(b, c, 4, h, w))
            print("convs OK")
            res = ihaar2d_triple(*convs, (H, W))
            print("ihaar OK")
            
        elif levels_count == 2:
            levels = haar2d_double(x)
            print("haar2d_double OK")
            convs = []
            for i, level in enumerate(levels):
                b, c, _, h, w = level.shape
                flat = level.view(b, c * 4, h, w)
                weight = torch.randn(c * 4, 1, 5, 5, device=device)
                scale = torch.ones(1, c * 4, 1, 1, device=device) * 0.1
                out = scaled_depthwise_conv(flat, weight, scale, padding)
                convs.append(out.view(b, c, 4, h, w))
            print("convs OK")
            res = ihaar2d_double(*convs, (H, W))
            print("ihaar OK")

    except Exception as e:
        print(f"FAILED: {e}")
        raise e

def main():
    # Stage 0
    test_stage(64, 128, 56, 56, 5)
    # Stage 1
    test_stage(64, 256, 28, 28, 4)
    # Stage 2
    test_stage(64, 512, 14, 14, 3)
    # Stage 3
    test_stage(64, 1024, 7, 7, 2)

if __name__ == "__main__":
    main()
