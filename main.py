# prepare_data.py
import os, glob, shutil
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.data_utils import WaterBodiesDataset

# Paths
SRC_IMAGES = "Water Bodies Dataset/Images"
SRC_MASKS = "Water Bodies Dataset/Masks"

# Load and sort
image_paths = sorted(glob.glob(os.path.join(SRC_IMAGES, "*")))
mask_paths = sorted(glob.glob(os.path.join(SRC_MASKS, "*")))
assert len(image_paths) == len(mask_paths), "Mismatch in number of images/masks"

# Split data
train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
    image_paths, mask_paths, test_size=0.3, random_state=42)
val_imgs, test_imgs, val_masks, test_masks = train_test_split(
    temp_imgs, temp_masks, test_size=0.5, random_state=42)

# Optional: Save splits
def save_subset(name, imgs, masks):
    for d in ["images", "masks"]:
        os.makedirs(os.path.join("datasets", name, d), exist_ok=True)
    for img_p, mask_p in zip(imgs, masks):
        shutil.copy(img_p, os.path.join("datasets", name, "images"))
        shutil.copy(mask_p, os.path.join("datasets", name, "masks"))

save_subset("train", train_imgs, train_masks)
save_subset("val", val_imgs, val_masks)
save_subset("test", test_imgs, test_masks)

# Create datasets and loaders
train_dataset = WaterBodiesDataset(train_imgs, train_masks, augment=True)
val_dataset = WaterBodiesDataset(val_imgs, val_masks)
test_dataset = WaterBodiesDataset(test_imgs, test_masks)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Quick test
for images, masks in train_loader:
    print("Image batch shape:", images.shape)
    print("Mask batch shape:", masks.shape)
    break
