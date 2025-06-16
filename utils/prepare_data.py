import os
import glob
import sys
from torch.utils.data import DataLoader
from utils.data_utils import WaterBodiesDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# --- Configuration ---
DATA_DIR = 'datasets/'
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256

# --- Augmentation Pipelines ---
train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(p=0.2),
        
        # REMOVED: A.CoarseDropout was removed as it was also cutting out parts of the mask,
        # which is a form of label corruption. Other augmentations are sufficient.
        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

def get_data_loaders(batch_size):
    """
    Finds image/mask files, creates Datasets, and returns DataLoaders.
    """
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')
    test_dir = os.path.join(DATA_DIR, 'test')

    # CHANGED: Now collects both .jpg and .png files.
    train_images = sorted(glob.glob(os.path.join(train_dir, 'images', '*.jpg')) + glob.glob(os.path.join(train_dir, 'images', '*.png')))
    train_masks = sorted(glob.glob(os.path.join(train_dir, 'masks', '*.jpg')) + glob.glob(os.path.join(train_dir, 'masks', '*.png')))
    
    val_images = sorted(glob.glob(os.path.join(val_dir, 'images', '*.jpg')) + glob.glob(os.path.join(val_dir, 'images', '*.png')))
    val_masks = sorted(glob.glob(os.path.join(val_dir, 'masks', '*.jpg')) + glob.glob(os.path.join(val_dir, 'masks', '*.png')))
    
    test_images = sorted(glob.glob(os.path.join(test_dir, 'images', '*.jpg')) + glob.glob(os.path.join(test_dir, 'images', '*.png')))
    test_masks = sorted(glob.glob(os.path.join(test_dir, 'masks', '*.jpg')) + glob.glob(os.path.join(test_dir, 'masks', '*.png')))

    # CHANGED: Fails fast if data is missing, preventing a crash later.
    if not train_images or not train_masks:
        sys.exit("Error: Training data not found. Please run the dataset split operation or check the 'datasets/train' directory.")

    print(f"Loading data from '{DATA_DIR}':")
    print(f"- Training set:   {len(train_images)} images")
    print(f"- Validation set: {len(val_images)} images")
    print(f"- Test set:       {len(test_images)} images")

    train_dataset = WaterBodiesDataset(train_images, train_masks, transform=train_transform)
    val_dataset = WaterBodiesDataset(val_images, val_masks, transform=val_transform)
    test_dataset = WaterBodiesDataset(test_images, test_masks, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, test_loader

# Create the data loaders to be imported by other scripts
train_loader, val_loader, test_loader = get_data_loaders(BATCH_SIZE)