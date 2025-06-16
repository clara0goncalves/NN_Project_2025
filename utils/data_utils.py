# utils/data_utils.py
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class WaterBodiesDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Create the ignore_mask from the original image to identify black "no-data" areas.
        ignore_mask = (image.sum(axis=-1) > 0)
        
        # Binarize the ground truth mask to ensure values are 0 or 1.
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # We also need to resize the ignore_mask to match the final output size.
        # This is done after the main transform to ensure correct dimensions.
        ignore_mask = cv2.resize(ignore_mask.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Convert all to PyTorch tensors
        mask = mask.float().unsqueeze(0)
        ignore_mask = torch.from_numpy(ignore_mask).bool().unsqueeze(0)

        ## --- THIS LINE IS CORRECTED --- ##
        return image, mask, ignore_mask