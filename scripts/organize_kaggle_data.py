"""
organize_kaggle_data.py - Organize the Kaggle Hard Hat Detection dataset

The Kaggle dataset (andrewmvd/hard-hat-detection) has:
- images/ folder with all images
- annotations/ folder with XML files containing bounding boxes

This script reads the annotations to determine if an image has a hardhat,
then copies images to the appropriate classification folders.

Usage:
    python organize_kaggle_data.py <path_to_extracted_kaggle_folder>

Example:
    python organize_kaggle_data.py "D:/Downloads/archive"
"""

import os
import sys
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

def parse_annotation(xml_path):
    """
    Parse a Pascal VOC annotation XML file.
    Returns list of object labels found in the image.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        labels = []
        for obj in root.findall('object'):
            name = obj.find('name')
            if name is not None:
                labels.append(name.text.lower())

        return labels
    except Exception as e:
        return []

def main():
    if len(sys.argv) < 2:
        print("Usage: python organize_kaggle_data.py <path_to_kaggle_folder>")
        print("Example: python organize_kaggle_data.py D:/Downloads/archive")
        sys.exit(1)

    kaggle_path = Path(sys.argv[1])

    if not kaggle_path.exists():
        print(f"Error: Path does not exist: {kaggle_path}")
        sys.exit(1)

    print("=" * 60)
    print("Kaggle Dataset Organizer")
    print("=" * 60)

    # Find images and annotations folders
    images_dir = None
    annotations_dir = None

    for subdir in kaggle_path.rglob('*'):
        if subdir.is_dir():
            name = subdir.name.lower()
            if 'image' in name:
                images_dir = subdir
            elif 'annotation' in name:
                annotations_dir = subdir

    # Also check root level
    if (kaggle_path / 'images').exists():
        images_dir = kaggle_path / 'images'
    if (kaggle_path / 'annotations').exists():
        annotations_dir = kaggle_path / 'annotations'

    print(f"\nKaggle folder: {kaggle_path}")
    print(f"Images folder: {images_dir}")
    print(f"Annotations folder: {annotations_dir}")

    if not images_dir or not images_dir.exists():
        print("\nError: Could not find images folder!")
        print("Expected structure:")
        print("  archive/")
        print("    images/")
        print("    annotations/")
        sys.exit(1)

    # Setup destination folders
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"

    dest_folders = {
        "train_hat": data_dir / "train" / "hard_hat",
        "train_no": data_dir / "train" / "no_hard_hat",
        "val_hat": data_dir / "val" / "hard_hat",
        "val_no": data_dir / "val" / "no_hard_hat",
    }

    for folder in dest_folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    # Get all images
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    print(f"\nFound {len(image_files)} images")

    # Classify and organize images
    hardhat_images = []
    no_hardhat_images = []

    print("\nAnalyzing images...")
    for img_path in image_files:
        # Try to find annotation
        if annotations_dir:
            xml_name = img_path.stem + '.xml'
            xml_path = annotations_dir / xml_name

            if xml_path.exists():
                labels = parse_annotation(xml_path)

                # Check for hardhat/helmet labels
                has_hardhat = any(
                    'helmet' in label or 'hat' in label
                    for label in labels
                )

                if has_hardhat:
                    hardhat_images.append(img_path)
                else:
                    no_hardhat_images.append(img_path)
            else:
                # No annotation - skip or assign randomly
                no_hardhat_images.append(img_path)
        else:
            # No annotations folder - assign based on filename or randomly
            if 'helmet' in img_path.name.lower() or 'hat' in img_path.name.lower():
                hardhat_images.append(img_path)
            else:
                no_hardhat_images.append(img_path)

    print(f"\nClassification results:")
    print(f"  With hard hat: {len(hardhat_images)}")
    print(f"  Without hard hat: {len(no_hardhat_images)}")

    # Shuffle for random train/val split
    random.shuffle(hardhat_images)
    random.shuffle(no_hardhat_images)

    # Split 80/20 for train/val
    def split_and_copy(images, train_folder, val_folder, max_train=400, max_val=100):
        """Split images into train and val sets."""
        n_train = min(int(len(images) * 0.8), max_train)
        n_val = min(len(images) - n_train, max_val)

        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]

        for img in train_images:
            shutil.copy2(img, train_folder / img.name)

        for img in val_images:
            shutil.copy2(img, val_folder / img.name)

        return len(train_images), len(val_images)

    print("\nCopying images...")

    train_hat, val_hat = split_and_copy(
        hardhat_images,
        dest_folders['train_hat'],
        dest_folders['val_hat']
    )

    train_no, val_no = split_and_copy(
        no_hardhat_images,
        dest_folders['train_no'],
        dest_folders['val_no']
    )

    # Summary
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nDataset organized in: {data_dir}")
    print(f"\nTraining set:")
    print(f"  hard_hat:    {train_hat} images")
    print(f"  no_hard_hat: {train_no} images")
    print(f"\nValidation set:")
    print(f"  hard_hat:    {val_hat} images")
    print(f"  no_hard_hat: {val_no} images")
    print(f"\nTotal: {train_hat + train_no + val_hat + val_no} images")

    print("\nReady to train! Run:")
    print("  cd D:\\AIO\\safety-detector\\src")
    print("  python train.py --data_dir ../data --epochs 10")

if __name__ == "__main__":
    main()
