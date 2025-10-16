"""
Dataset utilities for Car Detection Project
"""

from pathlib import Path
from typing import Dict, Tuple
import yaml


def count_dataset_files(data_dir: Path) -> Dict[str, int]:
    """
    Count number of images in train/valid/test sets
    
    Args:
        data_dir: Path to dataset root directory
    
    Returns:
        Dict with counts for each split
    """
    counts = {}
    
    for split in ['train', 'valid', 'test']:
        images_dir = data_dir / split / 'images'
        if images_dir.exists():
            counts[split] = len(list(images_dir.glob('*.jpg'))) + \
                           len(list(images_dir.glob('*.png')))
        else:
            counts[split] = 0
    
    return counts


def verify_dataset_structure(data_dir: Path) -> Tuple[bool, str]:
    """
    Verify dataset has correct structure
    
    Args:
        data_dir: Path to dataset root directory
    
    Returns:
        Tuple of (is_valid, message)
    """
    required_dirs = [
        'train/images', 'train/labels',
        'valid/images', 'valid/labels',
        'test/images', 'test/labels'
    ]
    
    for dir_path in required_dirs:
        full_path = data_dir / dir_path
        if not full_path.exists():
            return False, f"Missing directory: {dir_path}"
    
    # Check if there are any images
    counts = count_dataset_files(data_dir)
    if counts['train'] == 0:
        return False, "No training images found"
    if counts['valid'] == 0:
        return False, "No validation images found"
    
    return True, "Dataset structure is valid"


def create_dataset_yaml(data_dir: Path, yaml_path: Path) -> Path:
    """
    Create YAML configuration file for dataset
    
    Args:
        data_dir: Path to dataset root directory
        yaml_path: Path where to save YAML file
    
    Returns:
        Path to created YAML file
    """
    yaml_content = {
        'path': str(data_dir),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'names': {
            0: 'car'
        },
        'nc': 1
    }
    
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    return yaml_path


def get_dataset_stats(data_dir: Path) -> Dict:
    """
    Get comprehensive dataset statistics
    
    Args:
        data_dir: Path to dataset root directory
    
    Returns:
        Dict with dataset statistics
    """
    counts = count_dataset_files(data_dir)
    
    stats = {
        'total_images': sum(counts.values()),
        'train_images': counts.get('train', 0),
        'valid_images': counts.get('valid', 0),
        'test_images': counts.get('test', 0),
        'train_ratio': counts.get('train', 0) / max(sum(counts.values()), 1),
        'valid_ratio': counts.get('valid', 0) / max(sum(counts.values()), 1),
        'test_ratio': counts.get('test', 0) / max(sum(counts.values()), 1),
    }
    
    return stats


def print_dataset_info(data_dir: Path):
    """
    Print dataset information
    
    Args:
        data_dir: Path to dataset root directory
    """
    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    
    # Verify structure
    is_valid, message = verify_dataset_structure(data_dir)
    if not is_valid:
        print(f"❌ {message}")
        return
    
    print(f"✓ {message}")
    print(f"\n📁 Dataset Path: {data_dir}")
    
    # Get statistics
    stats = get_dataset_stats(data_dir)
    
    print(f"\n📊 Statistics:")
    print(f"  Total Images: {stats['total_images']:,}")
    print(f"  Training: {stats['train_images']:,} ({stats['train_ratio']:.1%})")
    print(f"  Validation: {stats['valid_images']:,} ({stats['valid_ratio']:.1%})")
    print(f"  Test: {stats['test_images']:,} ({stats['test_ratio']:.1%})")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    from config.config import DATA_DIR
    print_dataset_info(DATA_DIR)
