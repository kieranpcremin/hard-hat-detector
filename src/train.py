"""
train.py - Model Training Loop

This module handles the complete training pipeline:
1. Load data
2. Create model
3. Train for N epochs
4. Evaluate on validation set
5. Save best model

KEY CONCEPTS:
-------------
- Epoch: One complete pass through the entire training dataset
- Batch: A subset of data processed together (e.g., 32 images)
- Loss Function: Measures how wrong our predictions are
- Optimizer: Algorithm that updates model weights to reduce loss
- Learning Rate: How big of a step to take when updating weights
- Backpropagation: Algorithm to calculate gradients (how to adjust weights)

TRAINING LOOP FLOW:
-------------------
For each epoch:
    For each batch:
        1. Forward pass: Feed images through model, get predictions
        2. Calculate loss: Compare predictions to true labels
        3. Backward pass: Calculate gradients (how to adjust weights)
        4. Update weights: Apply gradients using optimizer
    Evaluate on validation set
    Save model if it's the best so far
"""

import os
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import our modules
from dataset import create_dataloaders
from model import create_model, get_device, unfreeze_layers


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: The neural network
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to train on (CPU/GPU)

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()  # Set model to training mode (enables dropout, etc.)

    running_loss = 0.0
    correct = 0
    total = 0

    # Progress bar for this epoch
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for images, labels in progress_bar:
        # =================================================================
        # STEP 1: Move data to device (GPU if available)
        # =================================================================
        images = images.to(device)
        labels = labels.to(device)

        # =================================================================
        # STEP 2: Zero the gradients
        # =================================================================
        # Gradients accumulate by default, so we reset them each batch
        optimizer.zero_grad()

        # =================================================================
        # STEP 3: Forward pass
        # =================================================================
        # Feed images through the model to get predictions (logits)
        outputs = model(images)

        # =================================================================
        # STEP 4: Calculate loss
        # =================================================================
        # CrossEntropyLoss compares predicted logits to true labels
        loss = criterion(outputs, labels)

        # =================================================================
        # STEP 5: Backward pass (backpropagation)
        # =================================================================
        # Calculate gradients: how should each weight change to reduce loss?
        loss.backward()

        # =================================================================
        # STEP 6: Update weights
        # =================================================================
        # Apply the gradients to update model weights
        optimizer.step()

        # =================================================================
        # Track metrics
        # =================================================================
        running_loss += loss.item() * images.size(0)

        # Get predictions (class with highest probability)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.1f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate the model on validation data.

    Args:
        model: The neural network
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)

    running_loss = 0.0
    correct = 0
    total = 0

    # No gradient calculation needed during validation (saves memory)
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def train(
    data_dir: str,
    output_dir: str = "models",
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    fine_tune_epochs: int = 5,
    fine_tune_lr: float = 0.0001
) -> Dict:
    """
    Complete training pipeline.

    Training strategy:
    1. Train with frozen backbone (fast, trains only classifier head)
    2. Fine-tune with unfrozen layers (slower, improves accuracy)

    Args:
        data_dir: Path to data directory with train/val folders
        output_dir: Where to save model checkpoints
        num_epochs: Number of epochs with frozen backbone
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        fine_tune_epochs: Additional epochs for fine-tuning
        fine_tune_lr: Learning rate for fine-tuning (should be lower)

    Returns:
        Dictionary with training history
    """
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Data directory:    {data_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"Batch size:        {batch_size}")
    print(f"Initial LR:        {learning_rate}")
    print(f"Epochs (frozen):   {num_epochs}")
    print(f"Epochs (finetune): {fine_tune_epochs}")
    print(f"Finetune LR:       {fine_tune_lr}")
    print("=" * 60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Get device
    device = get_device()

    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size
    )

    # Create model with frozen backbone
    print("\nCreating model...")
    model = create_model(num_classes=2, pretrained=True, freeze_backbone=True)
    model = model.to(device)

    # Loss function
    # CrossEntropyLoss is standard for classification
    # It combines softmax + negative log likelihood
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    # Adam is a good default optimizer that adapts learning rates per-parameter
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  # Only trainable params
        lr=learning_rate
    )

    # Learning rate scheduler
    # Reduces LR when validation loss plateaus (stops improving)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',        # Minimize loss
        factor=0.5,        # Reduce LR by half
        patience=2         # Wait 2 epochs before reducing
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    best_model_path = output_path / "best_model.pth"

    # =========================================================================
    # PHASE 1: Train with frozen backbone
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Training with frozen backbone")
    print("=" * 60)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc * 100:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, best_model_path)
            print(f"Saved new best model with val_acc: {val_acc * 100:.2f}%")

    # =========================================================================
    # PHASE 2: Fine-tune with unfrozen layers
    # =========================================================================
    if fine_tune_epochs > 0:
        print("\n" + "=" * 60)
        print("PHASE 2: Fine-tuning with unfrozen layers")
        print("=" * 60)

        # Unfreeze last 2 layer groups
        unfreeze_layers(model, num_layers=2)

        # New optimizer with lower learning rate
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=fine_tune_lr
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )

        for epoch in range(fine_tune_epochs):
            print(f"\nFine-tune Epoch {epoch + 1}/{fine_tune_epochs}")
            print("-" * 30)

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc * 100:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': num_epochs + epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, best_model_path)
                print(f"Saved new best model with val_acc: {val_acc * 100:.2f}%")

    # =========================================================================
    # Training complete
    # =========================================================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc * 100:.2f}%")
    print(f"Model saved to: {best_model_path}")

    return history


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train hard hat detector")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Path to save models")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")

    args = parser.parse_args()

    history = train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
