# utils/data_utils.py
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class WaterBodiesDataset(Dataset):
    """
    Custom PyTorch Dataset for loading water body images and their masks.
    """
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Read image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # --- THE FIX ---
        # Binarize the mask to ensure pixel values are 0 or 1.
        # Any pixel value above 127 will become 1, all others will become 0.
        mask = (mask > 127).astype(np.float32)

        # Apply the transformations from the Albumentations pipeline
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Add a channel dimension for the loss function
        mask = mask.float().unsqueeze(0)

        return image, mask