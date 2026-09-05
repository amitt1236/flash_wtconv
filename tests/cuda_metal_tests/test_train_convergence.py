"""
Training Convergence Test
=========================
Trains both naive (WTConv2dNaive) and fused (WTConv2d) models on Tiny ImageNet
classification to verify they converge identically.

Uses WTConvNeXt Tiny with a 200-class head for Tiny ImageNet (resized to 128x128).
Defaults to stage wavelet levels (5, 4, 3, 2); --depth overrides all four stages.
Uses the official labeled validation split; the test split has no public labels.
Downloads tiny-imagenet-200 into --data-root when it is missing.
Uses AdamW (lr=3e-4, weight decay=0.05), 3 warmup epochs, and cosine decay
to 1e-6. Training uses random resized crops and horizontal flips.

Usage:
    python test_train_convergence.py              # Run with default settings
    python test_train_convergence.py --epochs 20  # Train for 20 epochs
    python test_train_convergence.py --depth 3    # Test specific depth
    python test_train_convergence.py --dtype fp16 # Test with fp16 (AMP)
    python test_train_convergence.py --dtype all  # Test all dtypes
    python test_train_convergence.py --compile    # Enable torch.compile optimization

"""

import argparse
import json
import math
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
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_and_extract_archive
from contextlib import nullcontext

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
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "WTConv"))

from wtconv_model.wtconv import WTConv2d
from wtconv import WTConv2d as WTConv2dNaive
from wtconvnext import wtconvnext_tiny


# ==============================================================================
# WTConvNeXt Tiny models and weight synchronization
# ==============================================================================

DEFAULT_WT_LEVELS = (5, 4, 3, 2)


def build_model(wtconv_class, depth=None, num_classes=200):
    """Build WTConvNeXt Tiny, optionally overriding wavelet levels in every stage."""
    wt_levels = DEFAULT_WT_LEVELS if depth is None else (depth,) * 4
    return wtconvnext_tiny(
        pretrained=False,
        num_classes=num_classes,
        wtconv_class=wtconv_class,
        wt_levels=wt_levels,
        conv_mlp=True,
    )


def _wtconv_tensors(wtconv):
    """Return (base_weight, base_bias, base_scale, wt_weights, wt_scales).

    The fused WTConv2d keeps plain parameter tensors (base_weight / wt_weights),
    since the scale is folded into them before every convolution; the naive one
    keeps nn.Conv2d and _ScaleModule submodules.
    """
    if hasattr(wtconv, 'base_weight'):  # fused
        return (
            wtconv.base_weight,
            wtconv.base_bias,
            wtconv.base_scale,
            list(wtconv.wt_weights),
            list(wtconv.wt_scales),
        )
    return (  # naive
        wtconv.base_conv.weight,
        wtconv.base_conv.bias,
        wtconv.base_scale.weight,
        [conv.weight for conv in wtconv.wavelet_convs],
        [scale.weight for scale in wtconv.wavelet_scale],
    )


def copy_wtconv_weights(src_wtconv, dst_wtconv, src_class=None, dst_class=None):
    """Copy WTConv weights from src to dst, in either naming convention."""
    src_w, src_b, src_scale, src_wt_w, src_wt_s = _wtconv_tensors(src_wtconv)
    dst_w, dst_b, dst_scale, dst_wt_w, dst_wt_s = _wtconv_tensors(dst_wtconv)

    assert len(src_wt_w) == len(dst_wt_w), \
        f"wt_levels mismatch: {len(src_wt_w)} vs {len(dst_wt_w)}"

    with torch.no_grad():
        dst_w.copy_(src_w)
        if src_b is not None and dst_b is not None:
            dst_b.copy_(src_b)
        dst_scale.copy_(src_scale)
        for dst_lw, src_lw in zip(dst_wt_w, src_wt_w):
            dst_lw.copy_(src_lw)
        for dst_ls, src_ls in zip(dst_wt_s, src_wt_s):
            dst_ls.copy_(src_ls)


def copy_full_model_weights(src_model, dst_model, src_wtconv_class, dst_wtconv_class):
    """Copy shared model state and translate the two WTConv parameter layouts."""
    src_wtconvs = {
        name: module for name, module in src_model.named_modules()
        if isinstance(module, src_wtconv_class)
    }
    dst_wtconvs = {
        name: module for name, module in dst_model.named_modules()
        if isinstance(module, dst_wtconv_class)
    }
    if src_wtconvs.keys() != dst_wtconvs.keys():
        raise ValueError("Models have different WTConv module layouts")

    wtconv_prefixes = tuple(f"{name}." for name in src_wtconvs)
    src_state = {
        name: tensor for name, tensor in src_model.state_dict().items()
        if not name.startswith(wtconv_prefixes)
    }
    dst_state = {
        name: tensor for name, tensor in dst_model.state_dict().items()
        if not name.startswith(wtconv_prefixes)
    }
    if src_state.keys() != dst_state.keys():
        raise ValueError("Models have different non-WTConv state layouts")

    with torch.no_grad():
        for name, tensor in src_state.items():
            dst_state[name].copy_(tensor)
        for name, src_wtconv in src_wtconvs.items():
            copy_wtconv_weights(src_wtconv, dst_wtconvs[name])


# ==============================================================================
# Data Loading
# ==============================================================================

TINY_IMAGENET_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"


class TinyImageNetValidation(Dataset):
    """Read the official flat validation directory using the training class mapping."""

    def __init__(self, root, class_to_idx, transform=None):
        self.root = Path(root)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples = []
        annotations = self.root / "val_annotations.txt"
        for line in annotations.read_text().splitlines():
            if not line.strip():
                continue
            filename, wnid, *_ = line.split()
            self.samples.append((self.root / "images" / filename, class_to_idx[wnid]))
        if not self.samples:
            raise ValueError(f"No validation samples found in {annotations}")
        self.targets = [target for _, target in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = default_loader(str(path))  # Convert grayscale images to RGB too.
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def get_tiny_imagenet_loaders(batch_size=128, img_size=128, num_workers=4,
                              device=None, seed=42, drop_last=False,
                              generator=None, data_root='./data'):
    """Load Tiny ImageNet training and labeled validation images.

    data_root can be the extracted tiny-imagenet-200 directory or its parent.
    Reset the shared generator before each model's training for the same shuffle.
    """
    data_root = Path(data_root).expanduser()
    dataset_root = (data_root if data_root.name == "tiny-imagenet-200"
                    or (data_root / "train").is_dir()
                    else data_root / "tiny-imagenet-200")
    if not dataset_root.exists():
        download_and_extract_archive(
            TINY_IMAGENET_URL, download_root=str(dataset_root.parent),
            filename="tiny-imagenet-200.zip",
        )
    if (not (dataset_root / "train").is_dir()
            or not (dataset_root / "val" / "val_annotations.txt").is_file()):
        raise FileNotFoundError(
            f"Expected the official Tiny ImageNet layout in {dataset_root}: "
            "train/<class>/images and val/val_annotations.txt. "
            "Set --data-root to the extracted dataset or its parent."
        )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_dataset = torchvision.datasets.ImageFolder(
        dataset_root / "train", transform=train_transform,
    )
    val_dataset = TinyImageNetValidation(
        dataset_root / "val", train_dataset.class_to_idx, transform=val_transform,
    )
    pin_memory = device is not None and device.type == 'cuda'
    if generator is None:
        generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        generator=generator, drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,  # Always evaluate the entire labeled validation split.
    )
    return train_loader, val_loader


# ==============================================================================
# Training Functions
# ==============================================================================

def build_optimizer(model, lr=3e-4, weight_decay=0.05):
    """AdamW with no decay on biases, norm parameters, or either WTConv scale layout."""
    decay, no_decay = [], []
    scale_names = {'base_scale', 'wt_scales', 'wavelet_scale'}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (param.ndim <= 1 or name.endswith('bias')
                or scale_names.intersection(name.split('.'))):
            no_decay.append(param)
        else:
            decay.append(param)
    return optim.AdamW([
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=lr, betas=(0.9, 0.999))


def learning_rate_for_epoch(epoch, epochs, lr, warmup_epochs=3, min_lr=1e-6):
    """Linear epoch-wise warmup, then cosine decay to min_lr in the final epoch.

    Short pilots cap warmup at epochs - 1; a single-epoch run uses the peak LR.
    """
    if epochs < 1 or not 0 <= epoch < epochs:
        raise ValueError("epoch must be in [0, epochs), with epochs >= 1")
    if warmup_epochs < 0 or lr <= 0 or not 0 <= min_lr <= lr:
        raise ValueError("Require warmup_epochs >= 0 and 0 <= min_lr <= lr, with lr > 0")
    warmup = min(warmup_epochs, epochs - 1)
    if epoch < warmup:
        return lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(epochs - warmup - 1, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, amp_dtype=None):
    """Train for one epoch and retain dynamic-loss-scaling behavior."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    skipped_updates = 0
    initial_grad_scale = scaler.get_scale() if scaler is not None else None
    
    # Setup autocast context
    if amp_dtype is not None and device.type == 'cuda':
        autocast_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype)
    else:
        autocast_ctx = nullcontext()
    
    # Sync before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        with autocast_ctx:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        if scaler is not None:
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skipped_updates += 1
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
    final_grad_scale = scaler.get_scale() if scaler is not None else None
    return (avg_loss, accuracy, throughput, initial_grad_scale,
            final_grad_scale, skipped_updates)


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
        for inputs, targets in loader:
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
                amp_dtype=None, use_wandb=False, weight_decay=0.05,
                warmup_epochs=3, min_lr=1e-6, seed=42):
    """Train model and return loss/accuracy/throughput history for both train and val."""
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    # Reset CPU/CUDA RNGs too: transforms run in the main process with workers=0.
    # With workers>0, DataLoader seeds each worker from its dedicated generator.
    torch.manual_seed(seed)
    if train_loader.generator is not None:
        train_loader.generator.manual_seed(seed)
    if val_loader.generator is not None:
        val_loader.generator.manual_seed(seed)
    criterion = nn.CrossEntropyLoss()
    
    # Setup gradient scaler for fp16 (CUDA only)
    scaler = None
    if amp_dtype == torch.float16 and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
    
    train_loss_history = []
    train_acc_history = []
    throughput_history = []
    val_loss_history = []
    val_acc_history = []
    grad_scale_start_history = []
    grad_scale_end_history = []
    skipped_updates_history = []
    lr_history = []
    
    for epoch in range(epochs):
        epoch_lr = learning_rate_for_epoch(epoch, epochs, lr, warmup_epochs, min_lr)
        for group in optimizer.param_groups:
            group['lr'] = epoch_lr
        lr_history.append(epoch_lr)
        # Training
        (train_loss, train_acc, throughput, grad_scale_start, grad_scale_end,
         skipped_updates) = train_one_epoch(
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
        grad_scale_start_history.append(grad_scale_start)
        grad_scale_end_history.append(grad_scale_end)
        skipped_updates_history.append(skipped_updates)
        
        if verbose:
            scale_text = (f", grad_scale={grad_scale_start:.0f}->{grad_scale_end:.0f}, "
                          f"skipped_updates={skipped_updates}"
                          if grad_scale_start is not None else "")
            print(f"  [{name}] Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                  f"train_acc={train_acc:.2f}%, val_loss={val_loss:.4f}, "
                  f"val_acc={val_acc:.2f}%, lr={epoch_lr:.2e}, {throughput:.1f} img/s{scale_text}")
        
        # Log to wandb
        if use_wandb:
            log_row = {
                f"{name}/lr": epoch_lr,
                f"{name}/train_loss": train_loss,
                f"{name}/train_acc": train_acc,
                f"{name}/val_loss": val_loss,
                f"{name}/val_acc": val_acc,
                f"{name}/throughput": throughput,
                f"{name}/skipped_updates": skipped_updates,
                "epoch": epoch + 1,
            }
            if grad_scale_end is not None:
                log_row[f"{name}/grad_scale"] = grad_scale_end
            wandb.log(log_row)
    
    return {
        'lr': lr_history,
        'train_loss': train_loss_history,
        'train_acc': train_acc_history,
        'val_loss': val_loss_history,
        'val_acc': val_acc_history,
        'throughput': throughput_history,
        'grad_scale_start': grad_scale_start_history,
        'grad_scale_end': grad_scale_end_history,
        'skipped_updates': skipped_updates_history,
    }


# ==============================================================================
# Convergence Test
# ==============================================================================

def run_convergence_test(depth=None, epochs=50, batch_size=128,
                         img_size=128, lr=3e-4, seed=42,
                         verbose=True, dtype=torch.float16, device=None, use_wandb=False,
                         use_compile=False, data_root='./data', num_workers=4,
                         weight_decay=0.05, warmup_epochs=3, min_lr=1e-6):
    """Run convergence test comparing fused vs naive models on Tiny ImageNet. Uses the full labeled validation split."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate_for_epoch(0, epochs, lr, warmup_epochs, min_lr)
    if weight_decay < 0:
        raise ValueError("weight_decay must be nonnegative")
    dtype_name = DTYPE_NAMES[dtype]
    
    # Determine AMP dtype (None for fp32, use dtype for fp16/bf16)
    amp_dtype = None if dtype == torch.float32 else dtype
    
    print(f"\n{'='*70}")
    print(f"Training Convergence Test: WTConvNeXt Tiny on Tiny ImageNet @ {img_size}x{img_size} [{dtype_name}]")
    wt_levels = DEFAULT_WT_LEVELS if depth is None else (depth,) * 4
    print(f"wt_levels={wt_levels}, epochs={epochs}, batch_size={batch_size}")
    print(f"AdamW: lr={lr:g}, weight_decay={weight_decay:g}, "
          f"warmup_epochs={min(warmup_epochs, epochs - 1)}, cosine min_lr={min_lr:g}")
    print(f"{'='*70}")
    
    # Get data loaders - create shared generator for identical shuffle order
    print("\n--- Loading Tiny ImageNet ---")
    shared_generator = torch.Generator()
    shared_generator.manual_seed(seed)
    
    train_loader_fused, val_loader_fused = get_tiny_imagenet_loaders(
        batch_size, img_size, device=device, seed=seed, drop_last=use_compile,
        generator=shared_generator, data_root=data_root, num_workers=num_workers,
    )
    train_loader_naive, val_loader_naive = get_tiny_imagenet_loaders(
        batch_size, img_size, device=device, seed=seed, drop_last=use_compile,
        generator=shared_generator, data_root=data_root, num_workers=num_workers,
    )
        
    # Create models with same initial weights
    torch.manual_seed(seed)
    
    # Use the reference architecture's initialization for both implementations.
    model_naive = build_model(WTConv2dNaive, depth=depth).to(device)
    torch.manual_seed(seed)
    model_fused = build_model(WTConv2d, depth=depth).to(device)
    copy_full_model_weights(model_naive, model_fused, WTConv2dNaive, WTConv2d)

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
        model_fused, train_loader_fused, val_loader_fused, epochs, lr, device, verbose, "Fused", amp_dtype, use_wandb,
        weight_decay=weight_decay, warmup_epochs=warmup_epochs, min_lr=min_lr, seed=seed,
    )
    
    # Train naive model - reset generator to get same data order as fused
    print(f"\n--- Training Naive Model ---")
    # Reset the shared generator to same seed for identical data order
    shared_generator.manual_seed(seed)
    results_naive = train_model(
        model_naive, train_loader_naive, val_loader_naive, epochs, lr, device, verbose, "Naive", amp_dtype, use_wandb,
        weight_decay=weight_decay, warmup_epochs=warmup_epochs, min_lr=min_lr, seed=seed,
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
    if results_fused['grad_scale_end'][-1] is not None:
        print(f"  Dynamic gradient scaling:")
        print(f"    Final scale (Fused):    {results_fused['grad_scale_end'][-1]:.0f}")
        print(f"    Final scale (Naive):    {results_naive['grad_scale_end'][-1]:.0f}")
        print(f"    Skipped updates (Fused): {sum(results_fused['skipped_updates'])}")
        print(f"    Skipped updates (Naive): {sum(results_naive['skipped_updates'])}")
    
    print(f"\n  ✓ Training comparison complete")
    
    return results_fused, results_naive


def main():
    parser = argparse.ArgumentParser(description="WTConvNeXt Tiny Training Convergence Test on Tiny ImageNet")
    parser.add_argument("--depth", type=int, default=None, choices=[1, 2, 3, 4, 5],
                        help="Override WTConv levels in all stages (default: 5, 4, 3, 2)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak AdamW learning rate (default: 3e-4)")
    parser.add_argument("--weight-decay", type=float, default=0.05,
                        help="AdamW weight decay on convolution/linear weights (default: 0.05)")
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="Linear warmup epochs, capped at epochs - 1 (default: 3)")
    parser.add_argument("--min-lr", type=float, default=1e-6,
                        help="Final cosine learning rate (default: 1e-6)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Shared initialization and data-order seed (default: 42)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size (default: 128)")

    parser.add_argument("--img-size", type=int, default=128,
                        help="Image size (default: 128)")
    parser.add_argument("--data-root", type=Path, default=Path('./data'),
                        help="Tiny ImageNet directory or its parent (default: ./data; downloads if missing)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Data loader workers (default: 4)")
    parser.add_argument("--all-depths", action="store_true",
                        help="Test all depths (1-5)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print loss/acc each epoch")
    parser.add_argument("--dtype", choices=['fp32', 'fp16', 'bf16', 'all'], default='fp16',
                        help="Data type for training (default: fp16, use 'all' for all types)")
    parser.add_argument("--device", choices=['cuda', 'cpu'], default='cuda',
                        help="Device to test on (default: cuda)")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile optimization for fused model")
    parser.add_argument("--out", type=Path,
                        help="optional JSON output path for all epoch histories")
    args = parser.parse_args()
    try:
        learning_rate_for_epoch(0, args.epochs, args.lr, args.warmup_epochs, args.min_lr)
        if args.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
    except ValueError as error:
        parser.error(str(error))
    recipe = {
        "optimizer": "AdamW",
        "betas": [0.9, 0.999],
        "weight_decay": args.weight_decay,
        "weight_decay_exclusions": "biases, normalization parameters, learned scales",
        "scheduler": "linear_warmup_cosine_epoch",
        "warmup_epochs": min(args.warmup_epochs, args.epochs - 1),
        "min_lr": args.min_lr,
        "train_augmentation": "RandomResizedCrop(scale=(0.8, 1.0)), RandomHorizontalFlip(p=0.5)",
    }
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
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
                **recipe,
                "model": "wtconvnext_tiny",
                "dataset": "tiny-imagenet-200",
                "num_classes": 200,
                "data_root": str(args.data_root),
                "wt_levels": DEFAULT_WT_LEVELS if args.depth is None else (args.depth,) * 4,
                "depth": args.depth,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "img_size": args.img_size,
                "lr": args.lr,
                "seed": args.seed,
                "dtype": args.dtype,
                "device": str(device),
            }
        )
    
    all_results = []
    for dtype in dtypes_to_test:
        for depth in depths:
            results_fused, results_naive = run_convergence_test(
                depth=depth,
                epochs=args.epochs,
                batch_size=args.batch_size,
                img_size=args.img_size,
                lr=args.lr,
                seed=args.seed,
                verbose=args.verbose,
                dtype=dtype,
                device=device,
                use_wandb=use_wandb,
                use_compile=args.compile,
                data_root=args.data_root,
                num_workers=args.num_workers,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                min_lr=args.min_lr,
            )
            all_results.append({
                **recipe,
                "dtype": DTYPE_NAMES[dtype],
                "model": "wtconvnext_tiny",
                "dataset": "tiny-imagenet-200",
                "num_classes": 200,
                "data_root": str(args.data_root),
                "wt_levels": DEFAULT_WT_LEVELS if depth is None else (depth,) * 4,
                "depth": depth,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "img_size": args.img_size,
                "lr": args.lr,
                "seed": args.seed,
                "amp": dtype != torch.float32 and device.type == 'cuda',
                "fused": results_fused,
                "reference": results_naive,
            })
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"runs": all_results}, indent=2) + "\n")
        print(f"wrote {args.out}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
