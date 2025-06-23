# prepare_data.py

import sys
import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from utils.data_utils import WaterBodiesDataset, compute_dataset_ndwi_stats
# Paths
SRC_IMAGES = "Water Bodies Dataset/Images"
SRC_MASKS = "Water Bodies Dataset/Masks"

DATASET_OUT_DIR = "datasets"

# NDWI Configuration
USE_NDWI = True
NDWI_THRESHOLD = 0.0  # Will be optimized based on data
COMPUTE_NDWI_STATS = True

# Fixed transform wrapper for NDWI
class NDWIAwareTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform
    
    def __call__(self, *args):
        if len(args) == 1:
            return self.base_transform(args[0])
        else:
            seed = torch.seed()
            results = []
            for arg in args:
                torch.manual_seed(seed)
                results.append(self.base_transform(arg))
            return tuple(results)

def get_data_loaders(batch_size=16, num_workers=None, use_ndwi=False, compute_stats=True):
    global USE_NDWI, NDWI_THRESHOLD
    USE_NDWI = use_ndwi
    
    print("Loading dataset paths...")
    image_paths = sorted(glob.glob(os.path.join(SRC_IMAGES, "*")))
    mask_paths = sorted(glob.glob(os.path.join(SRC_MASKS, "*")))
    print(f"Found {len(image_paths)} images and {len(mask_paths)} masks")
    
    if len(mask_paths) == 0:
        print("No masks found. Will generate masks from NDWI.")
        mask_paths = [None] * len(image_paths)
        USE_NDWI = True
    else:
        if len(image_paths) != len(mask_paths):
            print(f"Warning: Mismatch in number of images ({len(image_paths)}) and masks ({len(mask_paths)})")
            min_count = min(len(image_paths), len(mask_paths))
            image_paths = image_paths[:min_count]
            mask_paths = mask_paths[:min_count]
            print(f"Aligned to {min_count} samples")
    
    if compute_stats and USE_NDWI:
        print("\nComputing NDWI statistics from sample data...")
        try:
            ndwi_stats = compute_dataset_ndwi_stats(image_paths, sample_size=min(50, len(image_paths)))
            print(f"NDWI Statistics:")
            print(f"  Mean: {ndwi_stats['mean']:.3f}")
            print(f"  Std: {ndwi_stats['std']:.3f}")
            print(f"  Range: [{ndwi_stats['min']:.3f}, {ndwi_stats['max']:.3f}]")
            print(f"  Percentiles: {ndwi_stats['percentiles']}")
            NDWI_THRESHOLD = ndwi_stats['mean'] + 0.5 * ndwi_stats['std']
            NDWI_THRESHOLD = np.clip(NDWI_THRESHOLD, -0.5, 0.5)
            print(f"Using NDWI threshold: {NDWI_THRESHOLD:.3f}")
        except Exception as e:
            print(f"Error computing NDWI stats: {e}")
            print("Using default NDWI threshold: 0.1")
            NDWI_THRESHOLD = 0.1

    print(f"\nSplitting dataset...")
    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        temp_imgs, temp_masks, test_size=0.5, random_state=42
    )
    print(f"Dataset split:")
    print(f"  Train: {len(train_imgs)} samples")
    print(f"  Validation: {len(val_imgs)} samples")
    print(f"  Test: {len(test_imgs)} samples")

    # Define transformations
    base_train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15, expand=False, fill=0),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
    ])
    base_eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_transform = NDWIAwareTransform(base_train_transform)
    eval_transform = NDWIAwareTransform(base_eval_transform)

    # Create datasets
    train_dataset = WaterBodiesDataset(
        train_imgs, train_masks,
        transform=train_transform,
        use_ndwi=USE_NDWI,
        ndwi_threshold=NDWI_THRESHOLD
    )
    val_dataset = WaterBodiesDataset(
        val_imgs, val_masks,
        transform=eval_transform,
        use_ndwi=USE_NDWI,
        ndwi_threshold=NDWI_THRESHOLD
    )
    test_dataset = WaterBodiesDataset(
        test_imgs, test_masks,
        transform=eval_transform,
        use_ndwi=USE_NDWI,
        ndwi_threshold=NDWI_THRESHOLD
    )
    print("Datasets created successfully!")

    if num_workers is None:
        num_workers = max(1, os.cpu_count() // 4)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    print(f"Data loaders created with {num_workers} workers and batch size {batch_size}")

    # Test loaders once
    print("\nTesting data loaders...")
    test_loader_safely(train_loader, "Train")
    print(f"\nData loading complete!")
    return train_loader, val_loader, test_loader

def test_loader_safely(loader, name):
    try:
        for i, batch in enumerate(loader):
            if USE_NDWI and len(batch) == 3:
                images, masks, ndwi = batch
                print(f"{name} Batch {i+1}: Image {images.shape}, Mask {masks.shape}, NDWI {ndwi.shape}")
                print(f"  Image range: [{images.min().item():.3f}, {images.max().item():.3f}]")
                print(f"  Mask unique values: {torch.unique(masks).tolist()}")
                print(f"  Water pixels (mask=1): {(masks > 0.5).sum().item()}/{masks.numel()}")
            else:
                images, masks = batch
                print(f"{name} Batch {i+1}: Image {images.shape}, Mask {masks.shape}")
                print(f"  Image range: [{images.min().item():.3f}, {images.max().item():.3f}]")
                print(f"  Mask unique values: {torch.unique(masks).tolist()}")
                print(f"  Water pixels (mask=1): {(masks > 0.5).sum().item()}/{masks.numel()}")
            break
    except Exception as e:
        print(f"Error testing {name} loader: {e}")

def visualize_sample(dataset, idx=0, save_path='sample_visualization.png'):
    try:
        if USE_NDWI:
            image, mask, ndwi = dataset[idx]
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        else:
            image, mask = dataset[idx]
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        img_np = image.permute(1, 2, 0).numpy() if isinstance(image, torch.Tensor) else np.array(image)
        mask_np = mask.squeeze().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
        img_np = np.clip(img_np / 255.0, 0, 1) if img_np.max() > 1 else np.clip(img_np, 0, 1)

        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(mask_np, cmap='Blues')
        axes[1].set_title('Water Mask')
        axes[1].axis('off')

        if USE_NDWI:
            ndwi_np = ndwi.squeeze().numpy() if isinstance(ndwi, torch.Tensor) else np.array(ndwi)
            im = axes[2].imshow(ndwi_np, cmap='RdYlBu', vmin=0, vmax=1)
            axes[2].set_title('NDWI (Normalized)')
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Sample visualization saved as '{save_path}'")
    except Exception as e:
        print(f"Error creating visualization: {e}")

if __name__ == "__main__":
    # Run main pipeline using above functions
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=8, use_ndwi=True, compute_stats=COMPUTE_NDWI_STATS)
    
    print("\nVisualizing a sample from validation set...")
    val_dataset = val_loader.dataset
    visualize_sample(val_dataset, idx=0)
    
    print("\nSetup complete! Data loaders and datasets are ready for training.")
