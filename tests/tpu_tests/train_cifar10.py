"""
Training Convergence Test (JAX/Flax) - Performance Optimized
=============================================================
Trains JAX WTConv2d model on CIFAR-10 classification.
Run separately from PyTorch version and compare results.

Performance optimizations:
- Pre-loads data to device with jax.device_put()
- Uses jax.lax.scan for training loop (better XLA fusion)
- Pre-batches data for minimal per-epoch overhead
- Increased default batch size (64)
- Supports bf16/fp16 mixed precision for TPU

Usage:
    python tpu/train_cifar10.py                  # Run with default settings
    python tpu/train_cifar10.py --epochs 20      # Train for 20 epochs
    python tpu/train_cifar10.py --depth 3        # Test specific depth
    python tpu/train_cifar10.py --dtype bf16     # Use bfloat16 for TPU
"""

import argparse
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import train_state
import optax

# Check for TensorFlow datasets (for CIFAR-10)
try:
    import tensorflow_datasets as tfds
    HAS_TFDS = True
except ImportError:
    HAS_TFDS = False

from tpu import WTConv2d


# ==============================================================================
# Simple CNN Model with WTConv layer
# ==============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN with WTConv layer for CIFAR-10 classification."""
    channels: int = 32
    depth: int = 2
    kernel_size: int = 5
    num_classes: int = 10
    dtype: jnp.dtype = None
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # Initial conv: 3 -> channels
        x = nn.Conv(features=self.channels, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        # WTConv layer (native NHWC format)
        x = WTConv2d(channels=self.channels, kernel_size=self.kernel_size, depth=self.depth, dtype=self.dtype)(x)
        
        # Adaptive average pooling to 8x8 (matching PyTorch version)
        B, H, W, C = x.shape
        pool_h, pool_w = H // 8, W // 8
        x = nn.avg_pool(x, window_shape=(pool_h, pool_w), strides=(pool_h, pool_w))  # (B, 8, 8, C)
        x = x.reshape((x.shape[0], -1))  # (B, 32*8*8) = (B, 2048)
        
        # Classifier
        x = nn.Dense(features=self.num_classes)(x)
        return x


def create_train_state(rng, model, learning_rate, img_size=256, channels=32, dtype=None):
    """Create initial training state."""
    # Initialize model
    dummy_input = jnp.ones((1, img_size, img_size, 3), dtype=dtype if dtype else jnp.float32)
    variables = model.init(rng, dummy_input, train=True)
    
    # Create optimizer
    tx = optax.sgd(learning_rate, momentum=0.9)
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    ), variables.get('batch_stats', {})


# ==============================================================================
# Data Loading
# ==============================================================================

def load_cifar10_numpy(num_samples=512, img_size=256, seed=42):
    """Load CIFAR-10 as numpy arrays (resized)."""
    if HAS_TFDS:
        return load_cifar10_tfds(num_samples, img_size, seed)
    else:
        return load_cifar10_torch(num_samples, img_size, seed)


def load_cifar10_tfds(num_samples=512, img_size=256, seed=42):
    """Load CIFAR-10 using TensorFlow Datasets."""
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')  # Use CPU for data loading
    
    ds = tfds.load('cifar10', split='train', as_supervised=True)
    ds = ds.take(num_samples)
    
    images = []
    labels = []
    
    for img, label in ds:
        # Resize
        img = tf.image.resize(img, (img_size, img_size))
        # Normalize
        img = tf.cast(img, tf.float32) / 255.0
        mean = tf.constant([0.4914, 0.4822, 0.4465])
        std = tf.constant([0.2470, 0.2435, 0.2616])
        img = (img - mean) / std
        
        images.append(img.numpy())
        labels.append(label.numpy())
    
    images = np.stack(images, axis=0)
    labels = np.array(labels)
    
    return images, labels


def load_cifar10_torch(num_samples=512, img_size=256, seed=42):
    """Load CIFAR-10 using torchvision."""
    import torch
    import torchvision
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    images = []
    labels = []
    
    for i in range(num_samples):
        img, label = dataset[i]
        # (C, H, W) -> (H, W, C)
        img = img.numpy().transpose(1, 2, 0)
        images.append(img)
        labels.append(label)
    
    images = np.stack(images, axis=0)
    labels = np.array(labels)
    
    return images, labels


def prepare_batches(images, labels, batch_size, rng):
    """
    Pre-create all batches for an epoch as a single array.
    Returns (batched_images, batched_labels, num_complete_batches).
    
    Drops incomplete last batch for scan compatibility.
    """
    n = len(images)
    
    # Shuffle indices
    perm = jax.random.permutation(rng, n)
    images = images[perm]
    labels = labels[perm]
    
    # Calculate number of complete batches
    num_batches = n // batch_size
    
    # Truncate to complete batches only
    images = images[:num_batches * batch_size]
    labels = labels[:num_batches * batch_size]
    
    # Reshape into batches: (num_batches, batch_size, ...)
    batched_images = images.reshape(num_batches, batch_size, *images.shape[1:])
    batched_labels = labels.reshape(num_batches, batch_size)
    
    return batched_images, batched_labels, num_batches


# ==============================================================================
# Training Functions
# ==============================================================================

def train_step(state, batch_stats, images, labels):
    """Single training step (not jitted here - jitted in scan)."""
    def loss_fn(params):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            images, train=True,
            mutable=['batch_stats']
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss, (logits, updates)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    new_batch_stats = updates['batch_stats']
    
    # Compute accuracy
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)
    
    return state, new_batch_stats, loss, accuracy


def train_one_epoch_scan(state, batch_stats, batched_images, batched_labels, num_batches):
    """Train for one epoch using jax.lax.scan for better XLA fusion."""
    
    def scan_body(carry, batch):
        state, batch_stats, total_loss, total_acc = carry
        images, labels = batch
        
        state, batch_stats, loss, acc = train_step(state, batch_stats, images, labels)
        
        new_total_loss = total_loss + loss
        new_total_acc = total_acc + acc
        
        return (state, batch_stats, new_total_loss, new_total_acc), None
    
    # Initial carry
    init_carry = (state, batch_stats, jnp.array(0.0), jnp.array(0.0))
    
    # Stack batches for scan: (num_batches, batch_size, ...) already done
    batches = (batched_images, batched_labels)
    
    # Run scan - JIT the entire epoch
    (state, batch_stats, total_loss, total_acc), _ = jax.lax.scan(
        scan_body, init_carry, batches
    )
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches * 100.0
    
    return state, batch_stats, avg_loss, avg_acc


# JIT the entire epoch function for maximum performance
@jax.jit
def train_one_epoch_jit(state, batch_stats, batched_images, batched_labels, num_batches):
    """JIT-compiled epoch training using scan."""
    return train_one_epoch_scan(state, batch_stats, batched_images, batched_labels, num_batches)


def train_model(state, batch_stats, images, labels, epochs, batch_size, seed=42, verbose=False, dtype=None):
    """Train model and return history."""
    rng = jax.random.PRNGKey(seed)
    
    # Pre-transfer data to device ONCE
    images = jax.device_put(jnp.array(images))
    labels = jax.device_put(jnp.array(labels))
    
    # Convert to specified dtype for mixed precision
    if dtype is not None:
        images = images.astype(dtype)
    
    loss_history = []
    acc_history = []
    throughput_history = []
    
    # Calculate samples per epoch (after dropping incomplete batches)
    num_batches = len(images) // batch_size
    samples_per_epoch = num_batches * batch_size
    
    for epoch in range(epochs):
        rng, epoch_rng = jax.random.split(rng)
        
        # Prepare batches for this epoch (shuffling)
        batched_images, batched_labels, num_batches = prepare_batches(
            images, labels, batch_size, epoch_rng
        )
        
        start_time = time.perf_counter()
        
        # Train epoch with scan
        state, batch_stats, loss, acc = train_one_epoch_jit(
            state, batch_stats, batched_images, batched_labels, num_batches
        )
        
        # Wait for computation to complete
        jax.block_until_ready(state.params)
        elapsed = time.perf_counter() - start_time
        
        loss_val = float(loss)
        acc_val = float(acc)
        throughput = samples_per_epoch / elapsed
        
        loss_history.append(loss_val)
        acc_history.append(acc_val)
        throughput_history.append(throughput)
        
        if verbose:
            print(f"  Epoch {epoch+1:3d}: loss={loss_val:.4f}, acc={acc_val:.2f}%, {throughput:.1f} img/s")
    
    return state, batch_stats, loss_history, acc_history, throughput_history


# ==============================================================================
# Main
# ==============================================================================

def run_training(depth=2, epochs=10, batch_size=64, num_samples=512, 
                 img_size=256, lr=0.01, seed=42, verbose=True, dtype_str='fp32'):
    """Run training on CIFAR-10."""
    # Parse dtype
    dtype_map = {
        'fp32': jnp.float32,
        'fp16': jnp.float16,
        'bf16': jnp.bfloat16,
    }
    dtype = dtype_map.get(dtype_str, jnp.float32)
    
    print(f"\n{'='*70}")
    print(f"JAX WTConv2d Training: CIFAR-10 @ {img_size}x{img_size}")
    print(f"depth={depth}, epochs={epochs}, batch_size={batch_size}, samples={num_samples}, dtype={dtype_str}")
    print(f"{'='*70}")
    
    # Load data
    print("\n--- Loading CIFAR-10 ---")
    images, labels = load_cifar10_numpy(num_samples, img_size, seed)
    print(f"  Loaded {len(images)} training samples, shape: {images.shape}")
    
    # Create model
    print("\n--- Creating Model ---")
    rng = jax.random.PRNGKey(seed)
    model = SimpleCNN(channels=32, depth=depth, kernel_size=5, dtype=dtype)
    state, batch_stats = create_train_state(rng, model, lr, img_size, dtype=dtype)
    
    num_params = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    print(f"  Model parameters: {num_params:,}")
    
    # Train
    print("\n--- Training ---")
    state, batch_stats, loss_history, acc_history, throughput_history = train_model(
        state, batch_stats, images, labels, epochs, batch_size, seed, verbose, dtype
    )
    
    # Summary
    print(f"\n{'='*70}")
    print("Results")
    print(f"{'='*70}")
    
    print(f"\n  {'Epoch':<6} {'Loss':<11} {'Accuracy':<10} {'Throughput':<12}")
    print("  " + "-" * 45)
    
    for i in range(epochs):
        print(f"  {i+1:<6} {loss_history[i]:<11.4f} {acc_history[i]:<10.2f} {throughput_history[i]:<12.1f}")
    
    avg_throughput = sum(throughput_history) / len(throughput_history)
    print(f"\n--- Summary ---")
    print(f"  Final loss:         {loss_history[-1]:.4f}")
    print(f"  Final accuracy:     {acc_history[-1]:.2f}%")
    print(f"  Avg throughput:     {avg_throughput:.1f} img/s")
    
    return loss_history, acc_history, throughput_history


def main():
    parser = argparse.ArgumentParser(description="JAX WTConv2d Training on CIFAR-10")
    parser.add_argument("--depth", type=int, default=2, choices=[1, 2, 3, 4, 5],
                        help="WTConv depth (default: 2)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (default: 0.01)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--samples", type=int, default=512,
                        help="Number of training samples (default: 512)")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Image size (default: 256)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print loss/acc each epoch")
    parser.add_argument("--all-depths", action="store_true",
                        help="Test all depths (1-4)")
    parser.add_argument("--dtype", type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'],
                        help="Data type for training: fp32, fp16, or bf16 (default: fp32)")
    args = parser.parse_args()
    
    print(f"Using JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    
    if args.all_depths:
        depths = [1, 2, 3, 4]
    else:
        depths = [args.depth]
    
    for depth in depths:
        run_training(
            depth=depth,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_samples=args.samples,
            img_size=args.img_size,
            lr=args.lr,
            verbose=args.verbose,
            dtype_str=args.dtype
        )
    
    print("\n" + "=" * 70)
    print("✓ Training complete!")


if __name__ == "__main__":
    main()
