"""
download_dataset.py - Setup folders and download instructions

Since most ML datasets require authentication, this script:
1. Creates the required folder structure
2. Provides clear instructions for getting data
"""

import os
from pathlib import Path

def main():
    print("=" * 60)
    print("Hard Hat Dataset Setup")
    print("=" * 60)

    # Paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data"

    # Create folder structure
    folders = {
        "train/hard_hat": data_dir / "train" / "hard_hat",
        "train/no_hard_hat": data_dir / "train" / "no_hard_hat",
        "val/hard_hat": data_dir / "val" / "hard_hat",
        "val/no_hard_hat": data_dir / "val" / "no_hard_hat",
    }

    print("\nCreating folder structure...")
    for name, folder in folders.items():
        folder.mkdir(parents=True, exist_ok=True)
        print(f"  Created: data/{name}/")

    print("\n" + "=" * 60)
    print("NEXT STEP: Add Images")
    print("=" * 60)

    print("""
EASIEST OPTION - Kaggle (Recommended):
--------------------------------------
1. Create free account: https://www.kaggle.com/account/login?phase=startRegisterTab
2. Go to: https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection
3. Click "Download" button (top right)
4. Extract the zip file
5. Run: python scripts/organize_kaggle_data.py <path_to_extracted_folder>

QUICK OPTION - Manual Collection:
---------------------------------
For a quick test with ~50 images:

1. Google Images: "worker wearing hard hat"
   - Download 20 images to: data/train/hard_hat/
   - Download 5 images to:  data/val/hard_hat/

2. Google Images: "worker without helmet" or "construction worker head"
   - Download 20 images to: data/train/no_hard_hat/
   - Download 5 images to:  data/val/no_hard_hat/

That's enough to test the pipeline!

Folder locations:
""")
    for name, path in folders.items():
        print(f"  {path}")

    print("""
After adding images, verify with:
  python scripts/download_dataset.py --check
""")

    # Check if --check argument
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        print("\n" + "=" * 60)
        print("DATASET CHECK")
        print("=" * 60)
        total = 0
        for name, path in folders.items():
            images = list(path.glob('*.jpg')) + list(path.glob('*.jpeg')) + list(path.glob('*.png'))
            count = len(images)
            total += count
            status = "OK" if count >= 5 else "NEED MORE"
            print(f"  {name}: {count} images [{status}]")

        print(f"\n  Total: {total} images")

        if total >= 50:
            print("\n  Dataset ready! Run training with:")
            print("    cd src")
            print("    python train.py --data_dir ../data --epochs 10")
        else:
            print("\n  Need at least 50 images total (25+ per class) for meaningful training.")

if __name__ == "__main__":
    main()
