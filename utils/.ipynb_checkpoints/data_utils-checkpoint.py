# utils/data_utils.py
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F

class WaterBodiesDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(img_path)[:, :, ::-1]  # BGR to RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        if self.augment:
            if np.random.rand() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            if np.random.rand() > 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()


        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        mask = torch.from_numpy(mask).unsqueeze(0)        # Add channel dim

        return image, mask
