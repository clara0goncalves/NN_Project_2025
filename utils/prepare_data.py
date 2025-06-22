# utils/prepare_data.py
import os
import glob
import shutil
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import sys

# Add project root to path to solve import issues in some environments
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_utils import WaterBodiesDataset

def get_data_loaders(batch_size=16):
    """
    Finds data, splits it, and returns the corresponding DataLoaders.
    This function is intended to be imported by other scripts like evaluate.py.
    """
    # Define paths relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_images_path = os.path.join(project_root, "Water Bodies Dataset/Images")
    src_masks_path = os.path.join(project_root, "Water Bodies Dataset/Masks")

    # Load and sort image and mask paths
    image_paths = sorted(glob.glob(os.path.join(src_images_path, "*")))
    mask_paths = sorted(glob.glob(os.path.join(src_masks_path, "*")))
    
    if not image_paths or not mask_paths:
        raise FileNotFoundError("Could not find images or masks. Check dataset paths.")
    assert len(image_paths) == len(mask_paths), "Mismatch in number of images and masks."

    # Split the data into training, validation, and test sets
    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
        image_paths, mask_paths, test_size=0.3, random_state=42)
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        temp_imgs, temp_masks, test_size=0.5, random_state=42)

    # Create PyTorch Datasets
    train_dataset = WaterBodiesDataset(train_imgs, train_masks, augment=True)
    val_dataset = WaterBodiesDataset(val_imgs, val_masks, augment=False)
    test_dataset = WaterBodiesDataset(test_imgs, test_masks, augment=False)

    # Create PyTorch DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader

def save_split_files(name, image_paths, mask_paths):
    """Utility function to copy split files to the datasets directory."""
    # Define paths relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, "datasets", name)
    
    os.makedirs(os.path.join(dataset_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "masks"), exist_ok=True)
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        shutil.copy(img_path, os.path.join(dataset_dir, "images"))
        shutil.copy(mask_path, os.path.join(dataset_dir, "masks"))
    print(f"Saved {len(image_paths)} images and masks to '{name}' split.")

# --- Main execution block ---
# This code will only run when the script is executed directly (e.g., `python utils/prepare_data.py`)
# It will NOT run when the script is imported by train.py or evaluate.py
if __name__ == "__main__":
    print("Preparing data splits...")
    
    # Get DataLoaders and implicitly the underlying data splits
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # Save the file splits to the 'datasets' directory for inspection
    print("\nSaving file splits to the 'datasets' directory...")
    save_split_files("train", train_loader.dataset.image_paths, train_loader.dataset.mask_paths)
    save_split_files("val", val_loader.dataset.image_paths, val_loader.dataset.mask_paths)
    save_split_files("test", test_loader.dataset.image_paths, test_loader.dataset.mask_paths)
    
    print("\n--- Data Preparation and Splitting Complete ---")
    
    # Quick test of the train_loader
    print("\nTesting the train_loader...")
    for images, masks in train_loader:
        print("Image batch shape:", images.shape)
        print("Mask batch shape:", masks.shape)
        break
    print("Test complete. Data loaders are working.")