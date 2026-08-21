#!/usr/bin/env python3
"""
This implements MNIST classifier using Slang/SlangPy
Architecture: 784 -> 128 -> 64 -> 10
"""

import slangpy as spy
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Global module variable (loaded once in main())
module = None

# Network architecture
INPUT_SIZE = 784
HIDDEN1_SIZE = 128
HIDDEN2_SIZE = 64
OUTPUT_SIZE = 10

# ============================================================================
# Helper Functions
# ============================================================================

def get_network_shape():
    """Get network shape as list."""
    return [INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE]

def get_total_params():
    """Calculate total number of parameters (weights + biases)."""
    shape = get_network_shape()
    total = 0
    for i in range(len(shape) - 1):
        # Weights: input_size * output_size
        total += shape[i] * shape[i + 1]
        # Biases: output_size
        total += shape[i + 1]
    return total

def initialize_params(device):

    total_params = get_total_params()
    
    shape = get_network_shape()
    params_data = []
    
    for i in range(len(shape) - 1):
        input_dim = shape[i]
        output_dim = shape[i + 1]
        
        # Initialize weights
        std = np.sqrt(2.0 / input_dim) # He initialization: std = sqrt(2 / fan_in) since we use ReLU activation function
        weights = np.random.randn(output_dim, input_dim).astype(np.float32) * std
        params_data.append(weights.flatten())
        
        # Initialize biases to zero
        biases = np.zeros(output_dim, dtype=np.float32)
        params_data.append(biases)
    
    # Concatenate all parameters and ensure contiguous memory layout
    # SlangPy requires contiguous arrays for proper GPU transfer
    params_data = np.concatenate(params_data)
    params_data = np.ascontiguousarray(params_data, dtype=np.float32)
    
    # Create buffers
    params = spy.Tensor.from_numpy(device, params_data)
    params_grad_data = np.zeros_like(params_data)
    params_grad = spy.Tensor.from_numpy(device, params_grad_data)
    
    print(f"Initialized {total_params:,} parameters")
    return params, params_grad

def create_mlp_params(device, params, params_grad):
    """Create MNIST_MLP_Params structure."""

    global module
    if module is None:
        module = spy.Module.load_from_file(device, "mnist_mlp.slang")
    
    mlp_params = spy.Tensor.empty(
        device,
        shape=(1,),
        dtype=module.MNIST_MLP_Params
    )
    
    cursor = mlp_params.cursor()
    cursor[0].write(
        {
            "m_params": params.storage.device_address,
            "m_grads": params_grad.storage.device_address,
        }
    )
    cursor.apply()
    
    return mlp_params

def load_mnist_data(batch_size=64):
    """Load MNIST dataset."""
    # .1307 is the mean of the entire MNIST dataset, .3081 is the standard deviation 
    # (these are computed over the entire dataset and commonly used when dealing with MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader

# ============================================================================
# Training Functions
# ============================================================================

def train_batch(mlp_params, adam_state, images, labels, device):
    """
    Train on a batch of images.
    
    Args:
        mlp_params: MNIST_MLP_Params buffer
        adam_state: Adam optimizer state buffer
        images: Batch of images (batch_size, 1, 28, 28) # channels, height, width
        labels: Batch of labels (batch_size,)
        device: Slang device
    
    Returns:
        Average loss for the batch
    """
    # Use global module
    global module
    
    batch_size = images.shape[0]
    
    # Flatten images: (batch_size, 1, 28, 28) -> (batch_size, 784)
    images_flat = images.view(batch_size, -1).numpy().astype(np.float32)
    
    # Convert labels to float for Tensor batching (converted back to int in Slang)
    # This allows SlangPy to batch both inputs together
    labels_np = labels.numpy().astype(np.float32)
    
    # Create batched Tensors - SlangPy will dispatch trainSample once per sample
    images_tensor = spy.Tensor.from_numpy(device, images_flat)
    labels_tensor = spy.Tensor.from_numpy(device, labels_np)
    
    # Batch train ALL samples at once (single dispatch, GPU handles parallelism)
    losses = module.trainSample(
        mlp_params.storage.device_address,
        images_tensor,
        labels_tensor
    )
    
    # Update parameters using Adam optimizer
    module.updateParams(
        mlp_params.storage.device_address,
        adam_state,
        batch_size,
        0.9,
        0.999
    )
    
    # Average loss over batch
    if hasattr(losses, 'to_numpy'):
        avg_loss = np.mean(losses.to_numpy())
    else:
        avg_loss = float(losses) if batch_size == 1 else np.mean(losses)
    
    return avg_loss

def evaluate(mlp_params, test_loader, device):
    """
    Evaluate model on test set.
    
    Returns:
        accuracy: Test accuracy (%)
    """
    # Use global module
    global module
    
    correct = 0
    total = 0
    
    for images, labels in test_loader:
        batch_size = images.shape[0]
        
        # Flatten images: (batch_size, 1, 28, 28) -> (batch_size, 784)
        images_flat = images.view(batch_size, -1).numpy().astype(np.float32)
        
        # Convert to Slang tensor (SlangPy dispatches once per batch element)
        images_tensor = spy.Tensor.from_numpy(device, images_flat)
        
        # Predict entire batch
        predictions = module.predictSample(
            mlp_params.storage.device_address,
            images_tensor
        )
        
        # Get predictions as numpy array
        if hasattr(predictions, 'to_numpy'):
            pred_np = predictions.to_numpy()
        else:
            pred_np = np.array([predictions])
        
        # Compare with labels
        labels_np = labels.numpy()
        correct += np.sum(pred_np == labels_np)
        total += batch_size
    
    accuracy = 100.0 * correct / total
    return accuracy

# ============================================================================
# Visualization
# ============================================================================

def plot_training_history(train_losses, test_accs, output_path="mnist_slang_history.png"):
    """Plot training progress."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(train_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(test_accs)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Test Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved training history to {output_path}")
    plt.close()

# ============================================================================
# Main Training Loop
# ============================================================================

def save_model(params, filepath='mnist_slang_model.npz'):
    """
    Save trained model parameters to disk.
    
    Args:
        params: Slang parameter buffer
        filepath: Path to save the model
    """
    # Convert GPU buffer to numpy array
    params_np = params.to_numpy()
    
    # Save with network shape metadata
    np.savez(
        filepath,
        params=params_np,
        shape=[INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE]
    )
    print(f"Model saved to {filepath}")


def load_model(device, filepath='mnist_slang_model.npz'):
    """
    Load trained model parameters from disk.
    
    Args:
        device: Slang device
        filepath: Path to the saved model
        
    Returns:
        params: Slang parameter buffer
    """
    # Load numpy array
    data = np.load(filepath)
    params_np = data['params']
    shape = data['shape']
    
    # Verify shape matches
    expected_shape = [INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE]
    if not np.array_equal(shape, expected_shape):
        raise ValueError(f"Model shape mismatch! Expected {expected_shape}, got {shape}")
    
    # Create Slang buffer from numpy array
    params = spy.Tensor.from_numpy(device, params_np)
    
    print(f"Model loaded from {filepath}")
    return params


def visualize_predictions(mlp_params, test_loader, device, num_images=10):
    """
    Visualize model predictions on test images.
    
    Args:
        mlp_params: MLP parameters structure
        test_loader: PyTorch DataLoader for test data
        device: Slang device
        num_images: Number of images to visualize
    """
    # Use global module if available, otherwise load it
    global module
    if module is None:
        module = spy.Module.load_from_file(device, "mnist_mlp.slang")
    
    # Get a batch of test images
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # Prepare images for Slang
    batch_size = min(num_images, images.shape[0])
    images = images[:batch_size]
    labels = labels[:batch_size]
    
    # Flatten images: (batch_size, 1, 28, 28) -> (batch_size, 784)
    images_flat = images.view(batch_size, -1).numpy().astype(np.float32)
    
    # Create Slang tensor
    images_tensor = spy.Tensor.from_numpy(device, images_flat)
    
    # Get predictions
    predictions = module.predictSample(
        mlp_params.storage.device_address,
        images_tensor
    )
    
    # Convert to numpy
    if hasattr(predictions, 'to_numpy'):
        pred_np = predictions.to_numpy()
    else:
        pred_np = np.array([predictions])
    
    labels_np = labels.numpy()
    
    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for idx in range(batch_size):
        ax = axes[idx]
        
        # Denormalize image for display (reverse the normalization)
        img = images[idx].squeeze().numpy()
        img = img * 0.3081 + 0.1307  # Reverse: (x - mean) / std
        img = np.clip(img, 0, 1)
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Pred: {pred_np[idx]}, True: {labels_np[idx]}')
        ax.axis('off')
        
        # Color code: green if correct, red if wrong
        color = 'green' if pred_np[idx] == labels_np[idx] else 'red'
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(3)
            spine.set_visible(True)
    
    plt.tight_layout()
    plt.savefig('mnist_slang_predictions.png', dpi=150)
    print(f"Saved predictions to mnist_slang_predictions.png")
    plt.close()


def main():
    
    # Hyperparameters
    batch_size = 64
    num_epochs = 10
    
    print("="*60)
    print("MNIST MLP Training with Slang/SlangPy")
    print("="*60)
    
    # Initialize Slang device
    print("\nInitializing Slang device...")
    device = spy.create_device(
        spy.DeviceType.cuda,
        include_paths=[Path(__file__).parent]
    )
    
    # Load Slang module
    print("Loading Slang module...")
    
    global module
    module = spy.Module.load_from_file(device, "mnist_mlp.slang")
    
    # Load MNIST data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Initialize network parameters
    print("\nInitializing network...")
    params, params_grad = initialize_params(device)
    mlp_params = create_mlp_params(device, params, params_grad)
    
    # Initialize Adam optimizer state
    print("Initializing Adam optimizer...")
    total_params = get_total_params()
    adam_state = spy.Tensor.empty(
        device,
        shape=(total_params,),
        dtype=module.AdamState
    )
    module.clearAdamState(adam_state)
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    train_losses = []
    test_accs = []
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 60)
        
        # Training
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Train on batch
            batch_loss = train_batch(
                mlp_params, 
                adam_state, 
                images, 
                labels, 
                device
            )
            
            epoch_loss += batch_loss
            num_batches += 1
            
            # Print progress frequently
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {batch_loss:.4f}", flush=True)
        
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        # Evaluation
        print(f"\n  Evaluating on test set...")
        test_acc = evaluate(mlp_params, test_loader, device)
        test_accs.append(test_acc)
        
        print(f"\n  Epoch {epoch} Summary:")
        print(f"    Train Loss: {avg_loss:.4f}")
        print(f"    Test Accuracy: {test_acc:.2f}%")
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    
    # Final evaluation
    print(f"\nFinal Test Accuracy: {test_accs[-1]:.2f}%")
    
    # Plot results
    print("\nGenerating plots...")
    plot_training_history(train_losses, test_accs)
    
    # Visualize predictions
    print("\nVisualizing predictions...")
    visualize_predictions(mlp_params, test_loader, device)
    
    # Save model
    print("\nSaving model...")
    save_model(params)
    
    print("\n" + "="*60)
    print("All done!")
    print("="*60)
    print("\nGenerated files:")
    print("  - mnist_slang_history.png (training curves)")
    print("  - mnist_slang_predictions.png (sample predictions)")
    print("  - mnist_slang_model.npz (saved model)")
    print("\nTo load the model later, use: load_model(device, 'mnist_slang_model.npz')")

if __name__ == "__main__":
    main()

