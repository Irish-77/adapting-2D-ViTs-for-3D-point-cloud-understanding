import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader
from src.data.scanobjectnn import ScanObjectNN

def get_scanobjectnn_dataset(root_dir, train=True, variant='main_split', 
                           use_background=True, augmentation_level='base',
                           num_points=1024):
    augmentation_map = {
        'base': 'base',
        'light': 'augmented25_norot',
        'medium': 'augmented25rot',
        'heavy': 'augmentedrot',
        'heavy_scaled': 'augmentedrot_scale75'
    }
    
    return ScanObjectNN(
        root_dir=root_dir,
        split='training' if train else 'test',
        variant=variant,
        augmentation=augmentation_map.get(augmentation_level, 'base'),
        background=use_background,
        num_points=num_points
    )

def visualize_point_cloud(points, title="Point Cloud", ax=None):
    """Visualize a single point cloud"""
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Points is [N, 3]
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=points[:, 2], cmap='viridis', s=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Equal aspect ratio
    max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                         points[:, 1].max()-points[:, 1].min(),
                         points[:, 2].max()-points[:, 2].min()]).max() / 2.0
    
    mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    return ax


def test_dataset_loading(root_dir):
    """Test different dataset configurations"""
    print("Testing ScanObjectNN Dataset Loading...\n")
    
    # Test 1: Basic loading
    print("1. Testing basic dataset loading...")
    try:
        dataset = ScanObjectNN(
            root_dir=root_dir,
            split='training',
            variant='main_split',
            augmentation='base',
            background=True,
            num_points=1024
        )
        print(f"✓ Successfully loaded dataset with {len(dataset)} samples")
        print(f"  Number of classes: {dataset.num_classes}")
        print(f"  Data shape: {dataset.data.shape}")
        print(f"  Labels shape: {dataset.labels.shape}\n")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}\n")
        return
    
    # Test 2: Different variants
    print("2. Testing different variants...")
    variants_to_test = ['main_split', 'split1']
    for variant in variants_to_test:
        try:
            dataset = ScanObjectNN(root_dir=root_dir, variant=variant)
            print(f"✓ Variant '{variant}': {len(dataset)} samples")
        except Exception as e:
            print(f"✗ Variant '{variant}' failed: {e}")
    print()
    
    # Test 3: Different augmentations
    print("3. Testing different augmentation levels...")
    augmentations = ['base', 'augmented25_norot', 'augmented25rot', 'augmentedrot']
    for aug in augmentations:
        try:
            dataset = ScanObjectNN(root_dir=root_dir, augmentation=aug)
            print(f"✓ Augmentation '{aug}': loaded successfully")
        except Exception as e:
            print(f"✗ Augmentation '{aug}' failed: {e}")
    print()
    
    # Test 4: Background vs no background
    print("4. Testing background variants...")
    for bg in [True, False]:
        try:
            dataset = ScanObjectNN(root_dir=root_dir, background=bg)
            print(f"✓ Background={bg}: loaded successfully")
        except Exception as e:
            print(f"✗ Background={bg} failed: {e}")
    print()
    
    return dataset


def visualize_samples(dataset, num_samples=8, save_path=None):
    """Visualize multiple samples from the dataset"""
    print(f"\n5. Visualizing {num_samples} random samples...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 8))
    rows = 2
    cols = num_samples // 2
    
    # Get random indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        # Get sample
        points, label = dataset[idx]
        points = points.numpy()
        
        # Create subplot
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        visualize_point_cloud(points, title=f"Sample {idx}, Label: {label.item()}", ax=ax)
        
        # Adjust view angle for better visualization
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    
    plt.show()


def test_dataloader(dataset, batch_size=32):
    """Test DataLoader functionality"""
    print(f"\n6. Testing DataLoader with batch_size={batch_size}...")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for testing, increase for actual training
    )
    
    # Get one batch
    for batch_idx, (points, labels) in enumerate(dataloader):
        print(f"✓ Batch {batch_idx}: points shape: {points.shape}, labels shape: {labels.shape}")
        print(f"  Points dtype: {points.dtype}, Labels dtype: {labels.dtype}")
        print(f"  Points range: [{points.min():.3f}, {points.max():.3f}]")
        print(f"  Unique labels in batch: {torch.unique(labels).tolist()}")
        
        if batch_idx >= 2:  # Just test first 3 batches
            break
    
    return dataloader


def compare_augmentations(root_dir, sample_idx=0):
    """Compare the same sample across different augmentation levels"""
    print("\n7. Comparing augmentation effects on the same sample...")
    
    fig = plt.figure(figsize=(20, 4))
    augmentations = [
        ('base', 'Original'),
        ('augmented25_norot', '25% Aug (No Rot)'),
        ('augmented25rot', '25% Aug (With Rot)'),
        ('augmentedrot', 'Full Aug (With Rot)'),
        ('augmentedrot_scale75', 'Full Aug + Scale 75%')
    ]
    
    for i, (aug, title) in enumerate(augmentations):
        try:
            dataset = ScanObjectNN(
                root_dir=root_dir,
                split='test',  # Use test to avoid runtime augmentation
                augmentation=aug,
                num_points=1024
            )
            
            points, label = dataset[sample_idx]
            points = points.numpy()
            
            ax = fig.add_subplot(1, 5, i+1, projection='3d')
            visualize_point_cloud(points, title=title, ax=ax)
            ax.view_init(elev=20, azim=45)
            
        except Exception as e:
            print(f"✗ Failed to load {aug}: {e}")
    
    plt.tight_layout()
    plt.show()


def main():
    # Configure paths
    root_dir = "/home/basti/Development/University/3DVision/adapting-2D-ViTs-for-3D-point-cloud-understanding/.data/h5_files"  # Update this path!
    
    # Run tests
    print("="*60)
    print("ScanObjectNN Dataset Test Suite")
    print("="*60)
    
    # Test loading
    dataset = test_dataset_loading(root_dir)
    
    if dataset is not None:
        # Visualize samples
        visualize_samples(dataset, num_samples=8, save_path="scanobjectnn_samples.png")
        
        # Test dataloader
        test_dataloader(dataset, batch_size=32)
        
        # Compare augmentations
        compare_augmentations(root_dir)
        
        # Test convenience function
        print("\n8. Testing convenience function...")
        for aug_level in ['base', 'light', 'medium', 'heavy']:
            try:
                dataset = get_scanobjectnn_dataset(
                    root_dir=root_dir,
                    train=True,
                    augmentation_level=aug_level
                )
                print(f"✓ Augmentation level '{aug_level}': {len(dataset)} samples")
            except Exception as e:
                print(f"✗ Augmentation level '{aug_level}' failed: {e}")
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
