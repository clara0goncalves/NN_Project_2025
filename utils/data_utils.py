import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import rasterio
from rasterio.plot import reshape_as_image

class WaterBodiesDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, use_ndwi=True, ndwi_threshold=0.0,
                 image_size=(256, 256), transform=None):
        """
        Dataset supporting masks or NDWI-generated water masks.

        Args:
            image_paths (list): List of image file paths.
            mask_paths (list or None): List of mask file paths, or None to generate masks from NDWI.
            use_ndwi (bool): Whether to compute NDWI and generate mask if masks are not provided.
            ndwi_threshold (float): Threshold for NDWI-based water mask generation.
            image_size (tuple): Target spatial size for images, masks, and NDWI tensors.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.use_ndwi = use_ndwi
        self.ndwi_threshold = ndwi_threshold
        self.image_size = image_size
        self.transform = transform 

        # Define separate transforms for image and mask
        self.image_transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            # Add normalization or augmentation here if needed
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=Image.NEAREST),
            transforms.ToTensor(),  # Converts to [0,1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def compute_ndwi(self, image_array):
        """
        Compute NDWI using RGB proxy: (Green - Red) / (Green + Red).

        Args:
            image_array (np.ndarray): H x W x 3 RGB image normalized [0,1] or uint8.

        Returns:
            np.ndarray: NDWI values clipped between -1 and 1.
        """
        try:
            if image_array.ndim == 3 and image_array.shape[2] >= 3:
                if image_array.dtype == np.uint8:
                    image_array = image_array.astype(np.float32) / 255.0

                green = image_array[:, :, 1]
                red = image_array[:, :, 0]

                numerator = green - red
                denominator = green + red + 1e-8

                ndwi = numerator / denominator
                ndwi = np.clip(ndwi, -1.0, 1.0)
                return ndwi
            else:
                raise ValueError(f"Image must have >= 3 channels, got shape {image_array.shape}")
        except Exception as e:
            print(f"Error computing NDWI: {e}")
            return np.zeros(image_array.shape[:2], dtype=np.float32)

    def generate_water_mask_from_ndwi(self, ndwi, threshold=0.0):
        """
        Generate binary water mask from NDWI.

        Args:
            ndwi (np.ndarray): NDWI array.
            threshold (float): Threshold to classify water.

        Returns:
            np.ndarray: Binary mask (float32).
        """
        return (ndwi > threshold).astype(np.float32)

    def load_image_safely(self, img_path):
        """
        Load image, trying GeoTIFF first then PIL fallback.

        Args:
            img_path (str): Image path.

        Returns:
            PIL.Image.Image, np.ndarray: Loaded image and its numpy array.
        """
        try:
            if img_path.lower().endswith(('.tif', '.tiff')):
                try:
                    with rasterio.open(img_path) as src:
                        image_array = src.read()
                        image_array = reshape_as_image(image_array)

                        # Take first 3 bands if more available
                        if image_array.ndim == 3 and image_array.shape[2] > 3:
                            image_array = image_array[:, :, :3]

                        # Normalize to uint8 if needed
                        if image_array.dtype != np.uint8:
                            image_array = ((image_array - image_array.min()) /
                                           (image_array.max() - image_array.min()) * 255).astype(np.uint8)

                        image = Image.fromarray(image_array)
                        return image, image_array
                except Exception as e:
                    print(f"Failed to load GeoTIFF {img_path}: {e}")

            # Fallback: PIL RGB load
            image = Image.open(img_path).convert("RGB")
            image_array = np.array(image)
            return image, image_array

        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")
            dummy = Image.new('RGB', self.image_size, 'black')
            return dummy, np.array(dummy)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image and numpy array for NDWI
        image, image_array = self.load_image_safely(img_path)

        # Compute NDWI if enabled
        ndwi = None
        if self.use_ndwi:
            ndwi = self.compute_ndwi(image_array)

        # Load or generate mask
        if self.mask_paths and self.mask_paths[idx] is not None:
            try:
                mask = Image.open(self.mask_paths[idx]).convert("L")
            except Exception as e:
                print(f"Failed to load mask {self.mask_paths[idx]}: {e}")
                mask = Image.new('L', image.size, 0)
        elif self.use_ndwi and ndwi is not None:
            water_mask = self.generate_water_mask_from_ndwi(ndwi, self.ndwi_threshold)
            mask = Image.fromarray((water_mask * 255).astype(np.uint8), mode='L')
        else:
            raise ValueError("Either mask_paths must be provided or use_ndwi must be True")

        # Apply transforms
        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        # Binarize mask
        mask = (mask > 0.5).float()

        if self.use_ndwi and ndwi is not None:
            # Normalize NDWI to [0,1]
            ndwi_normalized = (ndwi + 1) / 2

            # Resize NDWI to match image size
            ndwi_img = Image.fromarray((ndwi_normalized * 255).astype(np.uint8))
            ndwi_img = ndwi_img.resize(self.image_size, resample=Image.BILINEAR)
            ndwi_resized = np.array(ndwi_img).astype(np.float32) / 255.0

            ndwi_tensor = torch.from_numpy(ndwi_resized).unsqueeze(0)

            return image, mask, ndwi_tensor

        return image, mask


def compute_dataset_ndwi_stats(image_paths, sample_size=100):
    """
    Compute NDWI statistics from a sample of the dataset to determine optimal threshold
    """
    print(f"Computing NDWI stats from {min(sample_size, len(image_paths))} samples...")

    if len(image_paths) > sample_size:
        indices = np.random.choice(len(image_paths), sample_size, replace=False)
        sample_paths = [image_paths[i] for i in indices]
    else:
        sample_paths = image_paths

    ndwi_values = []
    successful_samples = 0

    temp_dataset = WaterBodiesDataset(
        sample_paths,
        mask_paths=[None] * len(sample_paths),
        use_ndwi=True,
        ndwi_threshold=0.0,
        image_size=(256, 256),
    )

    for i, img_path in enumerate(sample_paths):
        try:
            image, image_array = temp_dataset.load_image_safely(img_path)
            ndwi = temp_dataset.compute_ndwi(image_array)

            if ndwi is not None and ndwi.size > 0:
                valid_ndwi = ndwi[np.isfinite(ndwi)]
                if len(valid_ndwi) > 0:
                    ndwi_values.extend(valid_ndwi.flatten())
                    successful_samples += 1
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    print(f"Successfully processed {successful_samples}/{len(sample_paths)} samples")

    if len(ndwi_values) == 0:
        print("Warning: No valid NDWI values found. Using default stats.")
        return {
            'mean': 0.0,
            'std': 0.5,
            'min': -1.0,
            'max': 1.0,
            'percentiles': {
                '25': -0.2,
                '50': 0.0,
                '75': 0.2,
                '90': 0.4,
                '95': 0.6
            }
        }

    ndwi_values = np.array(ndwi_values)

    # Remove outliers beyond 3 std dev
    mean_val = np.mean(ndwi_values)
    std_val = np.std(ndwi_values)
    ndwi_values = ndwi_values[np.abs(ndwi_values - mean_val) <= 3 * std_val]

    stats = {
        'mean': np.mean(ndwi_values),
        'std': np.std(ndwi_values),
        'min': np.min(ndwi_values),
        'max': np.max(ndwi_values),
        'percentiles': {
            '25': np.percentile(ndwi_values, 25),
            '50': np.percentile(ndwi_values, 50),
            '75': np.percentile(ndwi_values, 75),
            '90': np.percentile(ndwi_values, 90),
            '95': np.percentile(ndwi_values, 95),
        }
    }

    return stats
