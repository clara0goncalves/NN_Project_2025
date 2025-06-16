import os
import glob
import sys
from torch.utils.data import DataLoader
from .data_utils import WaterBodiesDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_data_loaders(batch_size, num_workers=4, use_augmentation=True): # CHANGED: Added use_augmentation parameter
    """
    Finds image/mask files, creates Datasets, and returns DataLoaders.
    Conditionally applies augmentations to the training set based on the
    use_augmentation flag.
    """
    IMAGE_SIZE = 256
    DATA_DIR = 'datasets/'

    # --- Augmentation Pipelines ---
    # This pipeline is for training data when augmentation is enabled.
    # In utils/prepare_data.py

    train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # Cleaned up Affine transform definition
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(-0.05, 0.05),
            rotate=(-15, 15),
            p=0.5,
            cval=0, # Fill value for empty areas
            mode=cv2.BORDER_CONSTANT
        ),
        
        # Color Augmentations...
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.8),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    )
    
    # This minimal pipeline is for validation, testing, AND for training when augmentation is disabled.
    val_transform = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # --- File Path Gathering ---
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')
    test_dir = os.path.join(DATA_DIR, 'test')

    train_images = sorted(glob.glob(os.path.join(train_dir, 'images', '*.*')))
    train_masks = sorted(glob.glob(os.path.join(train_dir, 'masks', '*.*')))
    val_images = sorted(glob.glob(os.path.join(val_dir, 'images', '*.*')))
    val_masks = sorted(glob.glob(os.path.join(val_dir, 'masks', '*.*')))
    test_images = sorted(glob.glob(os.path.join(test_dir, 'images', '*.*')))
    test_masks = sorted(glob.glob(os.path.join(test_dir, 'masks', '*.*')))

    if not train_images or not train_masks:
        raise FileNotFoundError("Error: Training data not found. Please run the dataset split operation or check the 'datasets/train' directory.")

    # --- CHANGED: Conditionally select the transform for the training set ---
    if use_augmentation:
        print("Data augmentation is ENABLED for the training set.")
        active_train_transform = train_transform
    else:
        print("Data augmentation is DISABLED for the training set.")
        active_train_transform = val_transform # Use the basic transform without augmentations

    print(f"Loading data from '{DATA_DIR}':")
    print(f"- Training set:   {len(train_images)} images")
    print(f"- Validation set: {len(val_images)} images")
    print(f"- Test set:       {len(test_images)} images")

    # --- Dataset & DataLoader Creation ---
    train_dataset = WaterBodiesDataset(train_images, train_masks, transform=active_train_transform)
    val_dataset = WaterBodiesDataset(val_images, val_masks, transform=val_transform)
    test_dataset = WaterBodiesDataset(test_images, test_masks, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader