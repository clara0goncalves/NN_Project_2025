# utils/inference_utils.py
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from models.unet import get_model


class WaterBodySegmentor:
    """
    Easy-to-use wrapper for water body segmentation inference
    """
    
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Loading model on {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        # Initialize model
        self.model = get_model(
            n_channels=config.get('n_channels', 3),
            n_classes=config.get('n_classes', 1),
            bilinear=config.get('bilinear', False)
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
        
    def preprocess_image(self, image_path):
        """Preprocess image for inference"""
        if isinstance(image_path, str):
            image = cv2.imread(image_path)[:, :, ::-1]  # BGR to RGB
        else:
            image = image_path  # Assume it's already a numpy array
            
        original_size = image.shape[:2]
        
        # Resize to model input size (256x256)
        image_resized = cv2.resize(image, (256, 256))
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device), original_size, image
    
    def predict(self, image_path, threshold=0.5):
        """
        Predict water bodies in an image
        
        Args:
            image_path: Path to image or numpy array
            threshold: Probability threshold for binary classification
            
        Returns:
            probability_map: Probability map of water bodies
            binary_mask: Binary mask of water bodies
            original_image: Original input image
        """
        image_tensor, original_size, original_image = self.preprocess_image(image_path)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            probability = torch.sigmoid(output).cpu().squeeze().numpy()
            
        # Resize back to original size
        probability_resized = cv2.resize(probability, (original_size[1], original_size[0]))
        binary_mask = (probability_resized > threshold).astype(np.uint8)
        
        return probability_resized, binary_mask, original_image
    
    def visualize_prediction(self, image_path, threshold=0.5, save_path=None, show_plot=True):
        """
        Visualize prediction results
        """
        prob_map, binary_mask, original_image = self.predict(image_path, threshold)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontsize=14)
        axes[0, 0].axis('off')
        
        # Probability map
        im1 = axes[0, 1].imshow(prob_map, cmap='viridis', vmin=0, vmax=1)
        axes[0, 1].set_title('Water Probability Map', fontsize=14)
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Binary mask
        axes[1, 0].imshow(binary_mask, cmap='Blues')
        axes[1, 0].set_title(f'Binary Mask (threshold={threshold})', fontsize=14)
        axes[1, 0].axis('off')
        
        # Overlay
        overlay = self.create_overlay(original_image, binary_mask)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Overlay on Original', fontsize=14)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
            
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def create_overlay(self, original_image, binary_mask, color=[0, 255, 255], alpha=0.4):
        """Create overlay of mask on original image"""
        overlay = original_image.copy()
        overlay[binary_mask > 0] = color
        result = cv2.addWeighted(original_image.astype(np.uint8), 1-alpha, 
                               overlay.astype(np.uint8), alpha, 0)
        return result
    
    def batch_predict(self, image_folder, output_folder, threshold=0.5):
        """
        Predict on a batch of images
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(image_folder) if f.lower().endswith(ext)])
        
        print(f"Found {len(image_files)} images to process")
        
        for i, image_file in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_file}")
            
            image_path = os.path.join(image_folder, image_file)
            prob_map, binary_mask, original_image = self.predict(image_path, threshold)
            
            # Save results
            base_name = os.path.splitext(image_file)[0]
            
            # Save probability map
            prob_vis = (prob_map * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_probability.png"), prob_vis)
            
            # Save binary mask
            mask_vis = binary_mask * 255
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_mask.png"), mask_vis)
            
            # Save overlay
            overlay = self.create_overlay(original_image, binary_mask)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_overlay.png"), 
                       cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        print(f"Batch processing completed. Results saved in {output_folder}")
    
    def calculate_water_area(self, image_path, threshold=0.5, pixel_size_m2=None):
        """
        Calculate water area in the image
        
        Args:
            image_path: Path to image
            threshold: Probability threshold
            pixel_size_m2: Area represented by each pixel in square meters
            
        Returns:
            Dictionary with area statistics
        """
        prob_map, binary_mask, original_image = self.predict(image_path, threshold)
        
        total_pixels = binary_mask.size
        water_pixels = np.sum(binary_mask)
        water_percentage = (water_pixels / total_pixels) * 100
        
        results = {
            'total_pixels': total_pixels,
            'water_pixels': int(water_pixels),
            'land_pixels': int(total_pixels - water_pixels),
            'water_percentage': water_percentage,
            'land_percentage': 100 - water_percentage
        }
        
        if pixel_size_m2:
            results['total_area_m2'] = total_pixels * pixel_size_m2
            results['water_area_m2'] = water_pixels * pixel_size_m2
            results['land_area_m2'] = (total_pixels - water_pixels) * pixel_size_m2
        
        return results


# Example usage script
def main():
    """Example usage of the WaterBodySegmentor"""
    
    # Initialize segmentor
    model_path = 'checkpoints/unet_water_segmentation_20241201_123456/best.pth'  # Update path
    segmentor = WaterBodySegmentor(model_path)
    
    # Single image prediction
    image_path = 'path/to/your/test/image.jpg'  # Update path
    
    if os.path.exists(image_path):
        print("Running single image prediction...")
        
        # Visualize prediction
        segmentor.visualize_prediction(
            image_path, 
            threshold=0.5, 
            save_path='prediction_result.png',
            show_plot=False
        )
        
        # Calculate water area
        area_stats = segmentor.calculate_water_area(image_path)
        print("\nWater Area Statistics:")
        print(f"Total pixels: {area_stats['total_pixels']:,}")
        print(f"Water pixels: {area_stats['water_pixels']:,}")
        print(f"Water percentage: {area_stats['water_percentage']:.2f}%")
    
    # Batch processing example
    input_folder = 'test_images'  # Update path
    output_folder = 'predictions'
    
    if os.path.exists(input_folder):
        print(f"\nRunning batch prediction on {input_folder}...")
        segmentor.batch_predict(input_folder, output_folder, threshold=0.5)


if __name__ == "__main__":
    main()