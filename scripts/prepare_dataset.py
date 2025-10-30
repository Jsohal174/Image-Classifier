"""
Dataset Preparation Script
Organize and split dataset into train/val/test sets
"""

import argparse
import os
import shutil
from pathlib import Path
import random
from typing import List, Tuple


def get_image_files(directory: str, extensions: List[str] = None) -> List[Path]:
    """
    Get all image files in directory

    Args:
        directory: Path to directory
        extensions: List of valid extensions (default: ['.jpg', '.jpeg', '.png'])

    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

    directory = Path(directory)
    image_files = []

    for ext in extensions:
        image_files.extend(directory.glob(f'*{ext}'))

    return sorted(image_files)


def split_dataset(
    files: List[Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split files into train/val/test sets

    Args:
        files: List of file paths
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    random.seed(seed)
    files = list(files)
    random.shuffle(files)

    n_total = len(files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    return train_files, val_files, test_files


def copy_files_with_labels(
    image_files: List[Path],
    output_dir: Path,
    split_name: str,
    copy_labels: bool = True
):
    """
    Copy image files and corresponding label files to output directory

    Args:
        image_files: List of image paths
        output_dir: Output directory
        split_name: Name of split (train/val/test)
        copy_labels: Whether to copy label files too
    """
    # Create directories
    img_dir = output_dir / split_name / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)

    if copy_labels:
        label_dir = output_dir / split_name / 'labels'
        label_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCopying {len(image_files)} files to {split_name}...")

    for img_path in image_files:
        # Copy image
        shutil.copy2(img_path, img_dir / img_path.name)

        # Copy label if exists
        if copy_labels:
            label_path = img_path.parent / (img_path.stem + '.txt')
            if label_path.exists():
                shutil.copy2(label_path, label_dir / label_path.name)
            else:
                print(f"  Warning: No label file for {img_path.name}")

    print(f"  ✓ Copied {len(image_files)} images to {img_dir}")
    if copy_labels:
        print(f"  ✓ Copied labels to {label_dir}")


def create_dataset_yaml(output_dir: Path, class_names: List[str] = None):
    """
    Create dataset configuration YAML file

    Args:
        output_dir: Output directory
        class_names: List of class names
    """
    if class_names is None:
        class_names = ['car', 'person', 'truck']  # Example

    yaml_content = f"""# Dataset configuration for YOLO training

# Paths (relative to this file)
train: train/images
val: val/images
test: test/images

# Number of classes
nc: {len(class_names)}

# Class names
names: {class_names}
"""

    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n✓ Created dataset configuration: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare and split dataset for YOLO training'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing images (and labels)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for organized dataset'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Ratio for training set (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Ratio for validation set (default: 0.2)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Ratio for test set (default: 0.1)'
    )
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='Dataset has no label files (inference only)'
    )
    parser.add_argument(
        '--classes',
        type=str,
        nargs='+',
        default=None,
        help='Class names for dataset.yaml'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    # Validate paths
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("YOLO Dataset Preparation")
    print("="*60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Split:  Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}")
    print("="*60)

    # Get all image files
    print("\nScanning for images...")
    image_files = get_image_files(input_dir)
    print(f"Found {len(image_files)} images")

    if len(image_files) == 0:
        print("Error: No images found!")
        return

    # Split dataset
    print("\nSplitting dataset...")
    train_files, val_files, test_files = split_dataset(
        image_files,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    print(f"  Train: {len(train_files)} images ({args.train_ratio*100:.1f}%)")
    print(f"  Val:   {len(val_files)} images ({args.val_ratio*100:.1f}%)")
    print(f"  Test:  {len(test_files)} images ({args.test_ratio*100:.1f}%)")

    # Copy files
    copy_files_with_labels(train_files, output_dir, 'train', not args.no_labels)
    copy_files_with_labels(val_files, output_dir, 'val', not args.no_labels)
    copy_files_with_labels(test_files, output_dir, 'test', not args.no_labels)

    # Create dataset config
    if args.classes:
        create_dataset_yaml(output_dir, args.classes)

    print("\n" + "="*60)
    print("✓ Dataset preparation complete!")
    print("="*60)
    print(f"\nDataset structure:")
    print(f"{output_dir}/")
    print(f"├── train/")
    print(f"│   ├── images/")
    print(f"│   └── labels/")
    print(f"├── val/")
    print(f"│   ├── images/")
    print(f"│   └── labels/")
    print(f"├── test/")
    print(f"│   ├── images/")
    print(f"│   └── labels/")
    print(f"└── dataset.yaml")
    print("\nYou can now use this dataset for training!")


if __name__ == '__main__':
    main()
