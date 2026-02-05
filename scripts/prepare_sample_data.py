"""
prepare_sample_data.py - Create a sample dataset for testing

This script downloads sample images to create a small test dataset.
For a real project, you'd use a larger dataset from Kaggle or Roboflow.

This uses free images from the web to create a minimal working dataset.
"""

import os
import urllib.request
import ssl
from pathlib import Path

# Disable SSL verification for downloading (some sites have cert issues)
ssl._create_default_https_context = ssl._create_unverified_context

# Sample image URLs (free/public domain images of workers with/without hard hats)
# These are placeholder URLs - we'll use a different approach

def create_folder_structure(base_path):
    """Create the required folder structure."""
    folders = [
        base_path / "train" / "hard_hat",
        base_path / "train" / "no_hard_hat",
        base_path / "val" / "hard_hat",
        base_path / "val" / "no_hard_hat",
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Created: {folder}")

def main():
    # Get the data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"

    print("=" * 50)
    print("Sample Dataset Preparation")
    print("=" * 50)

    # Create folder structure
    print("\nCreating folder structure...")
    create_folder_structure(data_dir)

    print("\n" + "=" * 50)
    print("NEXT STEPS")
    print("=" * 50)
    print("""
The folder structure is ready. Now you need to add images:

OPTION 1: Manual Collection (Quick Start)
-----------------------------------------
1. Google Image search for "construction worker hard hat"
2. Save 20-30 images to: data/train/hard_hat/
3. Save 5-10 images to: data/val/hard_hat/
4. Google Image search for "construction worker no hard hat" or "person head"
5. Save 20-30 images to: data/train/no_hard_hat/
6. Save 5-10 images to: data/val/no_hard_hat/

OPTION 2: Kaggle Dataset (Recommended for real training)
--------------------------------------------------------
1. Create account at kaggle.com
2. Download: kaggle.com/datasets/andrewmvd/hard-hat-detection
3. The images need to be sorted based on annotations

OPTION 3: Roboflow (Easiest)
----------------------------
1. Go to: universe.roboflow.com
2. Search for "hard hat classification"
3. Download in "folder" format
4. Copy images to the appropriate folders

Minimum images needed:
- Training: 20+ per class (more is better)
- Validation: 5+ per class
""")

if __name__ == "__main__":
    main()
