"""
dataset.py - Dataset Loading and Preprocessing

This module handles:
1. Loading images from disk
2. Applying transforms (resize, normalize, augment)
3. Creating PyTorch DataLoaders for training

KEY CONCEPTS:
-------------
- Dataset: A class that knows how to load individual samples (image + label)
- DataLoader: Wraps a Dataset and provides batching, shuffling, parallel loading
- Transforms: Operations applied to images (resize, flip, normalize, etc.)
- Augmentation: Artificially increasing dataset size by applying random transforms
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# =============================================================================
# TRANSFORMS
# =============================================================================
# Transforms are operations we apply to images before feeding them to the model.
# We use different transforms for training vs validation/testing.

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get transforms for training data.

    Training transforms include DATA AUGMENTATION - random modifications that
    help the model generalize better by seeing varied versions of images.

    Args:
        image_size: Target size for images (ResNet expects 224x224)

    Returns:
        Composed transform pipeline
    """
    return transforms.Compose([
        # Resize to slightly larger than target, then crop
        # This adds some variation in what part of the image we see
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),

        # Randomly flip images horizontally (50% chance)
        # A hard hat is a hard hat whether the person faces left or right!
        transforms.RandomHorizontalFlip(p=0.5),

        # Randomly adjust brightness, contrast, saturation
        # Helps model handle different lighting conditions
        transforms.ColorJitter(
            brightness=0.2,  # +/- 20% brightness
            contrast=0.2,    # +/- 20% contrast
            saturation=0.2,  # +/- 20% saturation
        ),

        # Small random rotation (+/- 10 degrees)
        transforms.RandomRotation(degrees=10),

        # Convert PIL Image to PyTorch Tensor
        # This changes shape from (H, W, C) to (C, H, W)
        # and scales pixel values from [0, 255] to [0.0, 1.0]
        transforms.ToTensor(),

        # Normalize using ImageNet statistics
        # Pre-trained models were trained on ImageNet, so we use the same normalization
        # This helps transfer learning work better
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means (RGB)
            std=[0.229, 0.224, 0.225]    # ImageNet standard deviations (RGB)
        )
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get transforms for validation/test data.

    NO augmentation here! We want to evaluate on consistent, unmodified images.

    Args:
        image_size: Target size for images

    Returns:
        Composed transform pipeline
    """
    return transforms.Compose([
        # Simple resize to target size
        transforms.Resize((image_size, image_size)),

        # Convert to tensor
        transforms.ToTensor(),

        # Same normalization as training
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# =============================================================================
# DATASET CLASS
# =============================================================================

class HardHatDataset(Dataset):
    """
    Custom Dataset for Hard Hat detection images.

    PyTorch Dataset classes must implement:
    - __len__(): Returns the total number of samples
    - __getitem__(idx): Returns one sample (image, label) at index idx

    Folder structure expected:
        data/
        ├── train/
        │   ├── hard_hat/
        │   │   ├── image1.jpg
        │   │   └── ...
        │   └── no_hard_hat/
        │       ├── image1.jpg
        │       └── ...
        └── val/
            ├── hard_hat/
            └── no_hard_hat/
    """

    # Class labels - these map folder names to numeric labels
    CLASSES = ['no_hard_hat', 'hard_hat']

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize the dataset.

        Args:
            root_dir: Path to data folder (e.g., 'data/train')
            transform: Transform pipeline to apply to images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Build list of (image_path, label) tuples
        self.samples = []

        for label_idx, class_name in enumerate(self.CLASSES):
            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist")
                continue

            # Find all images in this class folder
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((img_path, label_idx))

        print(f"Loaded {len(self.samples)} images from {root_dir}")

        # Print class distribution
        for label_idx, class_name in enumerate(self.CLASSES):
            count = sum(1 for _, label in self.samples if label == label_idx)
            print(f"  - {class_name}: {count} images")

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get one sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            tuple: (image_tensor, label)
                - image_tensor: Shape (3, 224, 224) after transforms
                - label: 0 for no_hard_hat, 1 for hard_hat
        """
        img_path, label = self.samples[idx]

        # Load image using PIL
        # PIL loads images in RGB format, which is what we want
        image = Image.open(img_path).convert('RGB')

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, label


# =============================================================================
# DATA LOADERS
# =============================================================================

def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0  # Set to 0 for Windows compatibility
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.

    DataLoaders handle:
    - Batching: Grouping samples together (e.g., 32 images at a time)
    - Shuffling: Randomizing order each epoch (training only)
    - Parallel loading: Using multiple CPU cores to load data faster

    Args:
        data_dir: Root data directory containing 'train' and 'val' folders
        batch_size: Number of samples per batch
        image_size: Target image size
        num_workers: Number of parallel data loading workers

    Returns:
        tuple: (train_loader, val_loader)
    """
    data_path = Path(data_dir)

    # Create datasets with appropriate transforms
    train_dataset = HardHatDataset(
        root_dir=data_path / 'train',
        transform=get_train_transforms(image_size)
    )

    val_dataset = HardHatDataset(
        root_dir=data_path / 'val',
        transform=get_val_transforms(image_size)
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data each epoch
        num_workers=num_workers,
        pin_memory=True  # Speeds up CPU to GPU transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse the ImageNet normalization for visualization.

    When we want to display an image that's been normalized,
    we need to undo the normalization to get back to [0, 1] range.

    Args:
        tensor: Normalized image tensor of shape (C, H, W)

    Returns:
        Denormalized tensor with values in [0, 1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Reverse: original = (normalized * std) + mean
    return tensor * std + mean


# =============================================================================
# MAIN - Test the dataset
# =============================================================================

if __name__ == "__main__":
    # Quick test to verify everything works
    print("Testing dataset module...")
    print("\nTrain transforms:")
    print(get_train_transforms())
    print("\nVal transforms:")
    print(get_val_transforms())
