import os
import glob
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import sys
import numpy as np

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import WaterBodiesDataset

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

def visualize_samples(samples, save_path="outputs/visualized_sample.png"):
    fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4 * len(samples)))

    for i, (img, mask, ndwi) in enumerate(samples):
        img_np = img.permute(1, 2, 0).numpy()
        mask_np = mask.squeeze().numpy()
        ndwi_np = ndwi.squeeze().numpy()

        # Clip image if in [0, 255]
        if img_np.max() > 1:
            img_np = img_np / 255.0

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask_np, cmap="Blues")
        axes[i, 1].set_title("Mask")
        axes[i, 1].axis("off")

        im = axes[i, 2].imshow(ndwi_np, cmap="RdYlBu", vmin=0, vmax=1)
        axes[i, 2].set_title("NDWI")
        axes[i, 2].axis("off")
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved visualization to: {save_path}")

def main():
    try:
        print("--- Visualizing Dataset ---")

        # Load sample images/masks
        image_paths = sorted(glob.glob("datasets/train/images/*"))
        mask_paths = sorted(glob.glob("datasets/train/masks/*"))

        if not image_paths or not mask_paths:
            print("❌ No images or masks found in dataset folder.")
            return

        img_subset, _, mask_subset, _ = train_test_split(
            image_paths, mask_paths, test_size=0.98, random_state=42
        )

        dataset = WaterBodiesDataset(img_subset, mask_subset)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        samples = []
        for i, batch in enumerate(dataloader):
            if len(batch) == 3:
                img, mask, ndwi = batch
            else:
                print("❌ Unexpected batch structure.")
                return

            # Resize mismatch fix: skip variable sized items
            if img.shape[2:] != mask.shape[2:] or img.shape[2:] != ndwi.shape[2:]:
                continue

            samples.append((img[0], mask[0], ndwi[0]))
            if len(samples) == 4:
                break

        if not samples:
            print("❌ No valid samples found to visualize.")
            return

        visualize_samples(samples)

    except Exception as e:
        print(f"⚠️ Error during visualization: {e}")

if __name__ == "__main__":
    main()
