"""
Training Convergence Test
=========================
Trains both naive (WTConv2dNaive) and fused (WTConv2d) models on CIFAR-10
classification to verify they converge identically.

Uses a simple CNN with WTConv layer for CIFAR-10 classification (resized to 256x256).
Includes validation on a separate test set to verify both training and validation metrics.

Usage:
    python test_train_convergence.py              # Run with default settings
    python test_train_convergence.py --epochs 20  # Train for 20 epochs
    python test_train_convergence.py --depth 3    # Test specific depth
    python test_train_convergence.py --dtype fp16 # Test with fp16 (AMP)
    python test_train_convergence.py --dtype all  # Test all dtypes
    python test_train_convergence.py --compile    # Enable torch.compile optimization

"""

import argparse
import time
import sys
import warnings
from pathlib import Path

# Suppress torch.compile warnings about graph breaks on custom CUDA ops
warnings.filterwarnings('ignore', message='.*Graph break.*')
warnings.filterwarnings('ignore', message='.*Unsupported builtin.*')
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64  # Increase cache for many model configurations

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from contextlib import nullcontext
from tqdm import tqdm

import wandb

# Dtype configuration
DTYPE_MAP = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

DTYPE_NAMES = {
    torch.float32: 'fp32',
    torch.float16: 'fp16',
    torch.bfloat16: 'bf16',
}

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from wtconv_model.wtconv_triton import WTConv2d
from WTConv.wtconv.wtconv2d import WTConv2d as WTConv2dNaive


# ==============================================================================
# Simple CNN Model with WTConv layer
# ==============================================================================

class InvertedResidual(nn.Module):
    """MobileNetV2 Inverted Residual block using WTConv for depthwise convolution."""
    
    def __init__(self, in_channels, out_channels, wtconv_class, depth=2, 
                 expand_ratio=4, stride=1, kernel_size=3):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        
        # Expansion phase: 1x1 conv to expand channels
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        
        # WTConv requires in_channels == out_channels, so we use hidden_dim
        self.wtconv = wtconv_class(hidden_dim, hidden_dim, kernel_size=kernel_size, wt_levels=depth)
        
        # Projection phase: 1x1 conv to reduce channels (linear - no activation)
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        self.expand = nn.Sequential(*layers) if layers else nn.Identity()
        
        # Stride handling (applied after WTConv if stride > 1)
        self.downsample = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()
        
    def forward(self, x):
        identity = x
        
        # Expansion
        out = self.expand(x)
        
        # WTConv (depthwise)
        out = self.wtconv(out)
        out = nn.functional.relu6(out)
        
        # Downsample if needed
        out = self.downsample(out)
        
        # Projection (linear bottleneck - no activation)
        out = self.project(out)
        
        # Residual connection
        if self.use_residual:
            out = out + identity
        
        return out


class WTMobileNet(nn.Module):
    """MobileNetV2-style network using WTConv for depthwise convolutions.
    
    Architecture:
    - Stem: Conv2d -> BN -> ReLU6 (3 -> 32 channels)
    - Body: Inverted Residual blocks with WTConv
    - Head: AdaptiveAvgPool -> Linear
    """
    
    def __init__(self, wtconv_class, depth=2, num_classes=10, width_mult=1.0):
        super().__init__()
        
        # Configuration: (expand_ratio, out_channels, num_blocks, stride)
        # Progressive downsampling for 256x256 input: 256 -> 128 -> 64 -> 32 -> 16
        self.cfgs = [
            # expand, out, blocks, stride
            (1, 32, 1, 1),    # 128x128 (after stem stride=2)
            (4, 64, 2, 2),    # 64x64 (first block has stride=2)
            (4, 96, 3, 2),    # 32x32
            (4, 128, 2, 2),   # 16x16
            (4, 256, 1, 1),   # 16x16 (final)
        ]
        
        input_channels = int(32 * width_mult)
        
        # Stem (stride=2 for 256x256 input -> 128x128)
        self.stem = nn.Sequential(
            nn.Conv2d(3, input_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU6(inplace=True),
        )
        
        # Build inverted residual blocks
        layers = []
        for expand_ratio, out_c, num_blocks, stride in self.cfgs:
            out_channels = int(out_c * width_mult)
            for i in range(num_blocks):
                s = stride if i == 0 else 1
                layers.append(
                    InvertedResidual(
                        input_channels, out_channels, wtconv_class,
                        depth=depth, expand_ratio=expand_ratio, stride=s
                    )
                )
                input_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Store wtconv_class for weight copying
        self.wtconv_class = wtconv_class
        self.depth = depth
        
        # Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(input_channels, num_classes)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Alias for backward compatibility
SimpleCNN = WTMobileNet


def copy_wtconv_weights(src_wtconv, dst_wtconv, src_class, dst_class):
    """Copy WTConv weights from src (hybrid/fused) to dst (naive) model.
    
    Handles different naming conventions between WTConv2dHybrid and WTConv2dNaive.
    WTConv2dHybrid uses:
      - wt_weight1/wt_scale1 for level 1
      - wavelet_convs/wavelet_scales for levels 2+
    WTConv2dNaive uses:
      - wavelet_convs/wavelet_scale for all levels
    """
    with torch.no_grad():
        # Copy base conv
        dst_wtconv.base_conv.weight.copy_(src_wtconv.base_weight)
        if hasattr(src_wtconv, 'base_bias') and src_wtconv.base_bias is not None:
            if hasattr(dst_wtconv.base_conv, 'bias') and dst_wtconv.base_conv.bias is not None:
                dst_wtconv.base_conv.bias.copy_(src_wtconv.base_bias)
        
        # Copy base scale
        # src uses base_scale (Parameter), dst uses base_scale (Scale module) or Parameter
        src_scale = src_wtconv.base_scale
            
        if hasattr(dst_wtconv, 'base_scale'):
            if isinstance(dst_wtconv.base_scale, nn.Parameter):
                dst_wtconv.base_scale.copy_(src_scale)
            else:
                dst_wtconv.base_scale.weight.copy_(src_scale)
        
        # Get depth from source model
        depth = src_wtconv.wt_levels
        
        # Copy wavelet convs and scales for each level
        for level in range(depth):
            # Source uses ParameterLists: wt_weights and wt_scales
            src_weight = src_wtconv.wt_weights[level]
            src_wscale = src_wtconv.wt_scales[level]
            
            # Copy to destination (WTConv2dNaive uses wavelet_convs and wavelet_scale)
            dst_wtconv.wavelet_convs[level].weight.copy_(src_weight)
            if hasattr(dst_wtconv, 'wavelet_scales'):
                dst_wtconv.wavelet_scales[level].copy_(src_wscale)
            else:
                dst_wtconv.wavelet_scale[level].weight.copy_(src_wscale)


def copy_full_model_weights(src_model, dst_model, src_wtconv_class, dst_wtconv_class):
    """Copy all weights from source WTMobileNet to destination WTMobileNet."""
    with torch.no_grad():
        # Copy stem (Conv + BN)
        for i, (src_layer, dst_layer) in enumerate(zip(src_model.stem, dst_model.stem)):
            if hasattr(src_layer, 'weight'):
                dst_layer.weight.copy_(src_layer.weight)
            if hasattr(src_layer, 'bias') and src_layer.bias is not None:
                if hasattr(dst_layer, 'bias') and dst_layer.bias is not None:
                    dst_layer.bias.copy_(src_layer.bias)
            if hasattr(src_layer, 'running_mean'):
                dst_layer.running_mean.copy_(src_layer.running_mean)
                dst_layer.running_var.copy_(src_layer.running_var)
        
        # Copy inverted residual blocks
        for src_block, dst_block in zip(src_model.features, dst_model.features):
            # Copy expansion layers
            if hasattr(src_block.expand, 'weight'):  # nn.Identity doesn't have weight
                for src_layer, dst_layer in zip(src_block.expand, dst_block.expand):
                    if hasattr(src_layer, 'weight'):
                        dst_layer.weight.copy_(src_layer.weight)
                    if hasattr(src_layer, 'bias') and src_layer.bias is not None:
                        if hasattr(dst_layer, 'bias') and dst_layer.bias is not None:
                            dst_layer.bias.copy_(src_layer.bias)
                    if hasattr(src_layer, 'running_mean'):
                        dst_layer.running_mean.copy_(src_layer.running_mean)
                        dst_layer.running_var.copy_(src_layer.running_var)
            
            # Copy WTConv
            copy_wtconv_weights(src_block.wtconv, dst_block.wtconv, src_wtconv_class, dst_wtconv_class)
            
            # Copy projection layers
            for src_layer, dst_layer in zip(src_block.project, dst_block.project):
                if hasattr(src_layer, 'weight'):
                    dst_layer.weight.copy_(src_layer.weight)
                if hasattr(src_layer, 'bias') and src_layer.bias is not None:
                    if hasattr(dst_layer, 'bias') and dst_layer.bias is not None:
                        dst_layer.bias.copy_(src_layer.bias)
                if hasattr(src_layer, 'running_mean'):
                    dst_layer.running_mean.copy_(src_layer.running_mean)
                    dst_layer.running_var.copy_(src_layer.running_var)
        
        # Copy classifier
        dst_model.classifier.weight.copy_(src_model.classifier.weight)
        dst_model.classifier.bias.copy_(src_model.classifier.bias)


# ==============================================================================
# Data Loading
# ==============================================================================

def get_cifar10_loaders(batch_size=32, img_size=256, num_workers=4, device=None, seed=42, drop_last=False, generator=None):
    """Load CIFAR-10 dataset resized to specified size. Uses subset for training, full test set for validation.
    
    Args:
        generator: Optional torch.Generator to use for shuffling. If None, creates one with the given seed.
                   Pass the same generator to multiple loaders for identical shuffle order.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Download CIFAR-10
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    pin_memory = True
    
    # Use provided generator or create a new seeded one
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory,
        generator=generator, drop_last=drop_last
    )
    
    val_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=drop_last
    )
    
    return train_loader, val_loader


# ==============================================================================
# Training Functions
# ==============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, amp_dtype=None):
    """Train for one epoch, return average loss, accuracy, and throughput."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Setup autocast context
    if amp_dtype is not None and device.type == 'cuda':
        autocast_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype)
    else:
        autocast_ctx = nullcontext()
    
    # Sync before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for inputs, targets in tqdm(loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        with autocast_ctx:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    # Sync after training
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    throughput = total / elapsed  # images per second
    return avg_loss, accuracy, throughput


def validate_one_epoch(model, loader, criterion, device, amp_dtype=None):
    """Validate for one epoch, return average loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Setup autocast context
    if amp_dtype is not None and device.type == 'cuda':
        autocast_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype)
    else:
        autocast_ctx = nullcontext()
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Validation", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            with autocast_ctx:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, epochs, lr, device, verbose=False, name="Model", 
                amp_dtype=None, use_wandb=False):
    """Train model and return loss/accuracy/throughput history for both train and val."""
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Setup gradient scaler for fp16 (CUDA only)
    scaler = None
    if amp_dtype == torch.float16 and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
    
    train_loss_history = []
    train_acc_history = []
    throughput_history = []
    val_loss_history = []
    val_acc_history = []
    
    for epoch in range(epochs):
        # Training
        train_loss, train_acc, throughput = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, amp_dtype
        )
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        throughput_history.append(throughput)
        
        # Validation
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device, amp_dtype
        )
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        if verbose:
            print(f"  [{name}] Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%, {throughput:.1f} img/s")
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                f"{name}/train_loss": train_loss,
                f"{name}/train_acc": train_acc,
                f"{name}/val_loss": val_loss,
                f"{name}/val_acc": val_acc,
                f"{name}/throughput": throughput,
                "epoch": epoch + 1,
            })
    
    return {
        'train_loss': train_loss_history,
        'train_acc': train_acc_history,
        'val_loss': val_loss_history,
        'val_acc': val_acc_history,
        'throughput': throughput_history
    }


# ==============================================================================
# Convergence Test
# ==============================================================================

def run_convergence_test(depth=2, epochs=10, batch_size=32, 
                         img_size=256, lr=0.01, seed=42, 
                         verbose=True, dtype=torch.float32, device=None, use_wandb=False,
                         use_compile=False):
    """Run convergence test comparing fused vs naive models on CIFAR-10. Uses full test set for validation."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype_name = DTYPE_NAMES[dtype]
    
    # Determine AMP dtype (None for fp32, use dtype for fp16/bf16)
    amp_dtype = None if dtype == torch.float32 else dtype
    
    print(f"\n{'='*70}")
    print(f"Training Convergence Test: CIFAR-10 @ {img_size}x{img_size} [{dtype_name}]")
    print(f"depth={depth}, epochs={epochs}, batch_size={batch_size}")
    print(f"{'='*70}")
    
    # Get data loaders - create shared generator for identical shuffle order
    print("\n--- Loading CIFAR-10 ---")
    shared_generator = torch.Generator()
    shared_generator.manual_seed(seed)
    
    train_loader_fused, val_loader_fused = get_cifar10_loaders(
        batch_size, img_size, device=device, seed=seed, drop_last=use_compile,
        generator=shared_generator
    )
    train_loader_naive, val_loader_naive = get_cifar10_loaders(
        batch_size, img_size, device=device, seed=seed, drop_last=use_compile,
        generator=shared_generator  # Same generator = same shuffle order
    )
        
    # Create models with same initial weights
    torch.manual_seed(seed)
    
    # Fused model (uses WTConv2d)
    model_fused = WTMobileNet(WTConv2d, depth=depth).to(device)
    
    # Naive model with copied weights
    torch.manual_seed(seed)  # Same seed for same initial weights
    model_naive = WTMobileNet(WTConv2dNaive, depth=depth).to(device)
    
    # Copy all weights to ensure identical starting point
    copy_full_model_weights(model_fused, model_naive, WTConv2d, WTConv2dNaive)
    
    # Verify initial outputs match
    print("\n--- Verifying Initial State ---")
    model_fused.eval()
    model_naive.eval()
    with torch.no_grad():
        test_input = torch.randn(2, 3, img_size, img_size, device=device)
        out_fused = model_fused(test_input)
        out_naive = model_naive(test_input)
        init_diff = (out_fused - out_naive).abs().max().item()
        print(f"  Initial output diff: {init_diff:.2e}")
        if init_diff < 1e-4:
            print("  ✓ Models start with identical outputs")
        else:
            print("  ⚠ Warning: Initial outputs differ!")
    
    # Enable torch.compile on fused model if requested
    if use_compile and device.type == 'cuda':
        print("\n--- Enabling torch.compile ---")
        model_fused = torch.compile(model_fused)
        model_naive = torch.compile(model_naive)
        print("  ✓ Model compiled with torch.compile()")
        print("  Note: First iteration will be slower due to compilation")
    
    # Train fused model
    print(f"\n--- Training Fused Model ---")
    # Reset the shared generator to ensure deterministic order
    shared_generator.manual_seed(seed)
    results_fused = train_model(
        model_fused, train_loader_fused, val_loader_fused, epochs, lr, device, verbose, "Fused", amp_dtype, use_wandb
    )
    
    # Train naive model - reset generator to get same data order as fused
    print(f"\n--- Training Naive Model ---")
    # Reset the shared generator to same seed for identical data order
    shared_generator.manual_seed(seed)
    results_naive = train_model(
        model_naive, train_loader_naive, val_loader_naive, epochs, lr, device, verbose, "Naive", amp_dtype, use_wandb
    )
    
    # Compare results
    print(f"\n{'='*70}")
    print("Results Comparison - Training")
    print(f"{'='*70}")
    
    print(f"\n  {'Epoch':<6} {'Fused TrLoss':<13} {'Naive TrLoss':<13} {'Fused TrAcc':<12} {'Naive TrAcc':<12} {'Fused img/s':<12} {'Naive img/s':<12}")
    print("  " + "-" * 95)
    
    for i in range(epochs):
        print(f"  {i+1:<6} {results_fused['train_loss'][i]:<13.4f} {results_naive['train_loss'][i]:<13.4f} {results_fused['train_acc'][i]:<12.2f} {results_naive['train_acc'][i]:<12.2f} {results_fused['throughput'][i]:<12.1f} {results_naive['throughput'][i]:<12.1f}")
    
    print(f"\n{'='*70}")
    print("Results Comparison - Validation")
    print(f"{'='*70}")
    
    print(f"\n  {'Epoch':<6} {'Fused ValLoss':<14} {'Naive ValLoss':<14} {'Fused ValAcc':<13} {'Naive ValAcc':<13}")
    print("  " + "-" * 60)
    
    for i in range(epochs):
        print(f"  {i+1:<6} {results_fused['val_loss'][i]:<14.4f} {results_naive['val_loss'][i]:<14.4f} {results_fused['val_acc'][i]:<13.2f} {results_naive['val_acc'][i]:<13.2f}")
    
    # Summary
    avg_tp_fused = sum(results_fused['throughput']) / len(results_fused['throughput'])
    avg_tp_naive = sum(results_naive['throughput']) / len(results_naive['throughput'])
    speedup = avg_tp_fused / avg_tp_naive if avg_tp_naive > 0 else 0
    
    print(f"\n--- Summary ---")
    print(f"  Training:")
    print(f"    Final loss (Fused):     {results_fused['train_loss'][-1]:.4f}")
    print(f"    Final loss (Naive):     {results_naive['train_loss'][-1]:.4f}")
    print(f"    Final acc (Fused):      {results_fused['train_acc'][-1]:.2f}%")
    print(f"    Final acc (Naive):      {results_naive['train_acc'][-1]:.2f}%")
    print(f"  Validation:")
    print(f"    Final loss (Fused):     {results_fused['val_loss'][-1]:.4f}")
    print(f"    Final loss (Naive):     {results_naive['val_loss'][-1]:.4f}")
    print(f"    Final acc (Fused):      {results_fused['val_acc'][-1]:.2f}%")
    print(f"    Final acc (Naive):      {results_naive['val_acc'][-1]:.2f}%")
    print(f"  Throughput:")
    print(f"    Avg throughput (Fused): {avg_tp_fused:.1f} img/s")
    print(f"    Avg throughput (Naive): {avg_tp_naive:.1f} img/s")
    print(f"    Fused speedup:          {speedup:.2f}x")
    
    print(f"\n  ✓ Training comparison complete")
    
    return results_fused, results_naive


def main():
    parser = argparse.ArgumentParser(description="Training Convergence Test on CIFAR-10")
    parser.add_argument("--depth", type=int, default=2, choices=[1, 2, 3, 4, 5],
                        help="WTConv depth (default: 2)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (default: 0.01)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")

    parser.add_argument("--img-size", type=int, default=256,
                        help="Image size (default: 256)")
    parser.add_argument("--all-depths", action="store_true",
                        help="Test all depths (1-4)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print loss/acc each epoch")
    parser.add_argument("--dtype", choices=['fp32', 'fp16', 'bf16', 'all'], default='fp32',
                        help="Data type for training (default: fp32, use 'all' for all types)")

    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile optimization for fused model")
    args = parser.parse_args()
    
    # Set device
    # Check for CUDA
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    device = torch.device('cuda')
    
    print(f"Using device: {device}")
    
    # Determine which dtypes to test
    if args.dtype == 'all':
        dtypes_to_test = [torch.float32, torch.float16, torch.bfloat16]
    else:
        dtypes_to_test = [DTYPE_MAP[args.dtype]]
    
    if args.all_depths:
        depths = [1, 2, 3, 4, 5]
    else:
        depths = [args.depth]
    

    # Initialize wandb if enabled
    use_wandb = args.wandb

    if use_wandb:
        wandb.init(
            project="wtconv-convergence_compile",
            config={
                "depth": args.depth,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "img_size": args.img_size,
                "lr": args.lr,
                "dtype": args.dtype,
                "device": str(device),
            }
        )
    
    for dtype in dtypes_to_test:
        for depth in depths:
            run_convergence_test(
                depth=depth,
                epochs=args.epochs,
                batch_size=args.batch_size,
                img_size=args.img_size,
                lr=args.lr,
                verbose=args.verbose,
                dtype=dtype,
                device=device,
                use_wandb=use_wandb,
                use_compile=args.compile
            )
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
