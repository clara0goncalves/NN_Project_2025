# utils/prepare_data.py
import os
import glob
from torch.utils.data import DataLoader
from utils.data_utils import WaterBodiesDataset

# --- Configuration ---
DATA_DIR = 'datasets/'
BATCH_SIZE = 16
NUM_WORKERS = 2  # Number of CPU cores for data loading

# --- Data Path Setup ---
train_dir = os.path.join(DATA_DIR, 'train')
val_dir = os.path.join(DATA_DIR, 'val')
test_dir = os.path.join(DATA_DIR, 'test')

def get_data_loaders(batch_size):
    """
    Prepares and returns data loaders for the pre-split train, val, and test sets.
    """
    # Get file paths for each set
    train_images = sorted(glob.glob(os.path.join(train_dir, 'images', '*.jpg')))
    # --- CORRECTED LINE BELOW ---
    train_masks = sorted(glob.glob(os.path.join(train_dir, 'masks', '*.jpg')))

    val_images = sorted(glob.glob(os.path.join(val_dir, 'images', '*.jpg')))
    # --- CORRECTED LINE BELOW ---
    val_masks = sorted(glob.glob(os.path.join(val_dir, 'masks', '*.jpg')))

    test_images = sorted(glob.glob(os.path.join(test_dir, 'images', '*.jpg')))
    # --- CORRECTED LINE BELOW ---
    test_masks = sorted(glob.glob(os.path.join(test_dir, 'masks', '*.jpg')))

    # Check if data exists
    if not train_images or not train_masks:
        print("Error: Training data directories are not populated or mask files not found.")
        print("Please ensure you have run the dataset split and that masks have the '.jpg' extension.")
        return None, None, None

    print(f"Loading data from '{DATA_DIR}':")
    print(f"- Training set:   {len(train_images)} images")
    print(f"- Validation set: {len(val_images)} images")
    print(f"- Test set:       {len(test_images)} images")

    # Create datasets
    train_dataset = WaterBodiesDataset(train_images, train_masks, augment=True)
    val_dataset = WaterBodiesDataset(val_images, val_masks, augment=False)
    test_dataset = WaterBodiesDataset(test_images, test_masks, augment=False)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    return train_loader, val_loader, test_loader

# Create the data loaders to be imported by other scripts
train_loader, val_loader, test_loader = get_data_loaders(BATCH_SIZE)