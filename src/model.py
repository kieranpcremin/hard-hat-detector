"""
model.py - Neural Network Architecture

This module defines our image classification model using TRANSFER LEARNING.

KEY CONCEPTS:
-------------
- Transfer Learning: Using a model pre-trained on a large dataset (ImageNet)
  and adapting it for our specific task. This works because:
  1. Early layers learn generic features (edges, textures, shapes)
  2. These features are useful for many vision tasks
  3. We only need to train the final layers for our specific task

- ResNet (Residual Network): A popular CNN architecture that uses "skip connections"
  to allow training of very deep networks (50, 101, 152 layers).

- Fine-tuning: Unfreezing some pre-trained layers and training them with a low
  learning rate to adapt them to our data.

WHY ResNet18?
-------------
- Good balance of accuracy and speed for learning
- Small enough to train on modest hardware
- Pre-trained weights readily available
- Great for understanding before moving to larger models
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


def create_model(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True
) -> nn.Module:
    """
    Create a ResNet18 model for binary classification.

    Architecture overview:
    ┌─────────────────────────────────────────────────────────────┐
    │  Input Image (3 x 224 x 224)                                │
    │         ↓                                                   │
    │  ┌─────────────────────────────────────────┐                │
    │  │  ResNet18 Backbone (Pre-trained)        │  ← FROZEN      │
    │  │  - Conv layers                          │    (or fine-   │
    │  │  - BatchNorm layers                     │     tuned)     │
    │  │  - Residual blocks                      │                │
    │  │  Output: 512 features                   │                │
    │  └─────────────────────────────────────────┘                │
    │         ↓                                                   │
    │  ┌─────────────────────────────────────────┐                │
    │  │  Custom Classifier Head (Trainable)     │  ← TRAINED     │
    │  │  - Dropout (prevent overfitting)        │                │
    │  │  - Linear layer (512 → num_classes)     │                │
    │  └─────────────────────────────────────────┘                │
    │         ↓                                                   │
    │  Output: Class probabilities                                │
    └─────────────────────────────────────────────────────────────┘

    Args:
        num_classes: Number of output classes (2 for hard_hat/no_hard_hat)
        pretrained: If True, use ImageNet pre-trained weights
        freeze_backbone: If True, freeze the pre-trained layers

    Returns:
        PyTorch model ready for training
    """
    # Load pre-trained ResNet18
    # weights=IMAGENET1K_V1 gives us the model trained on 1.2M ImageNet images
    if pretrained:
        print("Loading pre-trained ResNet18 weights (ImageNet)...")
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        print("Creating ResNet18 with random weights...")
        model = models.resnet18(weights=None)

    # Freeze backbone layers if requested
    # "Freezing" means setting requires_grad=False so weights don't update during training
    if freeze_backbone:
        print("Freezing backbone layers...")
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully connected layer
    # Original ResNet18 outputs 1000 classes (ImageNet categories)
    # We need only 2 classes (hard_hat / no_hard_hat)
    num_features = model.fc.in_features  # Get input size of original fc layer (512)
    print(f"Original classifier input features: {num_features}")

    # Create new classifier head
    # This is the part we WILL train
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),           # Dropout: randomly zero 50% of inputs (prevents overfitting)
        nn.Linear(num_features, num_classes)  # Linear: 512 inputs → 2 outputs
    )

    # Count trainable vs frozen parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\nModel summary:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters:    {frozen_params:,}")

    return model


def unfreeze_layers(model: nn.Module, num_layers: int = 2) -> None:
    """
    Unfreeze the last N layers of the backbone for fine-tuning.

    After initial training with frozen backbone, we can "fine-tune" by
    unfreezing some layers and training with a lower learning rate.

    ResNet18 layer structure:
    - layer1: First residual block
    - layer2: Second residual block
    - layer3: Third residual block
    - layer4: Fourth residual block (closest to output)
    - fc: Classifier head

    Args:
        model: The ResNet model
        num_layers: Number of layer groups to unfreeze (from the end)
    """
    # Get the layer groups (excluding fc which is already trainable)
    layer_groups = ['layer4', 'layer3', 'layer2', 'layer1']

    layers_to_unfreeze = layer_groups[:num_layers]
    print(f"Unfreezing layers: {layers_to_unfreeze}")

    for name, param in model.named_parameters():
        for layer_name in layers_to_unfreeze:
            if layer_name in name:
                param.requires_grad = True
                break

    # Recount trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after unfreezing: {trainable_params:,}")


def get_device() -> torch.device:
    """
    Get the best available device for training.

    Priority:
    1. CUDA (NVIDIA GPU) - fastest
    2. MPS (Apple Silicon) - fast on Mac
    3. CPU - slowest but always available

    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


# =============================================================================
# MAIN - Test the model
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing model creation...")
    print("=" * 60)

    # Create model
    model = create_model(num_classes=2, pretrained=True, freeze_backbone=True)

    # Get device
    device = get_device()

    # Move model to device
    model = model.to(device)

    # Test forward pass with dummy input
    print("\nTesting forward pass...")
    dummy_input = torch.randn(4, 3, 224, 224).to(device)  # Batch of 4 images
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (raw logits):\n{output}")

    # Apply softmax to get probabilities
    probs = torch.softmax(output, dim=1)
    print(f"\nOutput (probabilities):\n{probs}")
