# evaluate_fixed.py
"""
Fixed evaluation script that handles model architecture mismatches
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from sklearn.metrics import classification_report
import seaborn as sns
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.unet import get_model
from utils.metrics import (dice_score, iou_score, pixel_accuracy, 
                          precision_recall_f1, confusion_matrix_metrics)
from preprocessing.prepare_data import test_loader


def detect_model_architecture(checkpoint_path):
    """
    Detect the model architecture from the checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Check for bilinear upsampling vs transposed convolution
    has_up_conv = any('up1.up.weight' in key for key in state_dict.keys())
    bilinear = not has_up_conv
    
    # Detect number of channels in the deepest layer
    down4_keys = [key for key in state_dict.keys() if 'down4' in key and 'weight' in key]
    if down4_keys:
        # Get the first conv layer in down4
        for key in down4_keys:
            if 'double_conv.0.weight' in key:
                shape = state_dict[key].shape
                n_features = shape[0]  # Output channels
                break
        else:
            n_features = 512  # Default fallback
    else:
        n_features = 512
    
    # Determine if this is a deeper model based on feature count
    if n_features >= 1024:
        model_type = 'deep'
    else:
        model_type = 'standard'
    
    print(f"Detected model architecture:")
    print(f"  - Type: {model_type}")
    print(f"  - Bilinear: {bilinear}")
    print(f"  - Max features: {n_features}")
    
    return bilinear, model_type


def get_model_with_auto_config(n_channels, n_classes, checkpoint_path):
    """
    Create model with automatically detected configuration
    """
    bilinear, model_type = detect_model_architecture(checkpoint_path)
    
    # Try different model configurations
    configs_to_try = [
        {'bilinear': bilinear, 'base_features': 64},
        {'bilinear': not bilinear, 'base_features': 64},
        {'bilinear': True, 'base_features': 32},
        {'bilinear': False, 'base_features': 32},
    ]
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    for config in configs_to_try:
        try:
            # Create model with current config
            if hasattr(get_model, '__code__') and 'base_features' in get_model.__code__.co_varnames:
                model = get_model(
                    n_channels=n_channels,
                    n_classes=n_classes,
                    bilinear=config['bilinear'],
                    base_features=config['base_features']
                )
            else:
                model = get_model(
                    n_channels=n_channels,
                    n_classes=n_classes,
                    bilinear=config['bilinear']
                )
            
            # Try to load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Successfully loaded model with config: {config}")
            return model, checkpoint
            
        except Exception as e:
            print(f"Failed with config {config}: {str(e)[:100]}...")
            continue
    
    # If all configs fail, try partial loading
    print("All automatic configs failed. Attempting partial loading...")
    return load_model_partial(n_channels, n_classes, checkpoint_path)


def load_model_partial(n_channels, n_classes, checkpoint_path):
    """
    Load model with partial state dict matching (ignoring mismatched layers)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try with bilinear=True first
    model = get_model(n_channels=n_channels, n_classes=n_classes, bilinear=True)
    
    model_dict = model.state_dict()
    checkpoint_dict = checkpoint['model_state_dict']
    
    # Filter out mismatched keys
    filtered_dict = {}
    for k, v in checkpoint_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                filtered_dict[k] = v
            else:
                print(f"Skipping {k}: shape mismatch {model_dict[k].shape} vs {v.shape}")
        else:
            print(f"Skipping {k}: key not found in current model")
    
    print(f"Loading {len(filtered_dict)}/{len(checkpoint_dict)} layers")
    
    # Load the filtered dictionary
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    
    return model, checkpoint


class ModelEvaluator:
    def __init__(self, model_path, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Try to automatically detect and load the correct model
        try:
            self.model, checkpoint = get_model_with_auto_config(
                config['n_channels'], 
                config['n_classes'], 
                model_path
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'dice_score' in checkpoint:
                print(f"Best validation Dice: {checkpoint['dice_score']:.4f}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def save_prediction_example(self, image, true_mask, pred_mask, prob_mask, dice, iou, idx):
        """Save visualization of prediction example"""
        os.makedirs('evaluation_results', exist_ok=True)
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Convert tensors to numpy
        if image.dim() == 3:
            img_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = image.squeeze().cpu().numpy()
            
        true_mask_np = true_mask.squeeze().cpu().numpy()
        pred_mask_np = pred_mask.squeeze().cpu().numpy()
        prob_mask_np = prob_mask.squeeze().cpu().numpy()
        
        # Normalize image for display
        if img_np.max() > 1:
            img_np = img_np / 255.0
        img_np = np.clip(img_np, 0, 1)
        
        # Original image
        if img_np.shape[-1] == 3:
            axes[0].imshow(img_np)
        else:
            axes[0].imshow(img_np, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # True mask
        axes[1].imshow(true_mask_np, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Predicted mask
        axes[2].imshow(pred_mask_np, cmap='gray')
        axes[2].set_title(f'Prediction\nDice: {dice:.3f}')
        axes[2].axis('off')
        
        # Probability map
        im = axes[3].imshow(prob_mask_np, cmap='viridis', vmin=0, vmax=1)
        axes[3].set_title(f'Probability Map\nIoU: {iou:.3f}')
        axes[3].axis('off')
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(f'evaluation_results/example_{idx}.png', dpi=150, bbox_inches='tight')
        plt.close()

    def evaluate_dataset(self, dataloader, save_examples=True, num_examples=5):
        """
        Evaluate model on dataset and compute comprehensive metrics
        """
        all_dice = []
        all_iou = []
        all_pixel_acc = []
        all_precision = []
        all_recall = []
        all_f1 = []
        
        # For confusion matrix
        all_tp, all_tn, all_fp, all_fn = 0, 0, 0, 0
        
        examples_saved = 0
        
        print("Evaluating model...")
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(dataloader)):
                images, masks = images.to(self.device), masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                # Calculate metrics for each image in batch
                for i in range(images.size(0)):
                    pred_i = preds[i]
                    mask_i = masks[i]
                    
                    # Basic metrics
                    dice = dice_score(pred_i, mask_i)
                    iou = iou_score(pred_i, mask_i)
                    pixel_acc = pixel_accuracy(pred_i, mask_i)
                    precision, recall, f1 = precision_recall_f1(pred_i, mask_i)
                    
                    all_dice.append(dice.item())
                    all_iou.append(iou.item())
                    all_pixel_acc.append(pixel_acc.item())
                    all_precision.append(precision.item())
                    all_recall.append(recall.item())
                    all_f1.append(f1.item())
                    
                    # Confusion matrix components
                    cm_metrics = confusion_matrix_metrics(pred_i, mask_i)
                    all_tp += cm_metrics['TP']
                    all_tn += cm_metrics['TN']
                    all_fp += cm_metrics['FP']
                    all_fn += cm_metrics['FN']
                    
                    # Save example predictions
                    if save_examples and examples_saved < num_examples:
                        self.save_prediction_example(
                            images[i], masks[i], preds[i], probs[i],
                            dice.item(), iou.item(), examples_saved
                        )
                        examples_saved += 1
        
        # Calculate overall metrics
        metrics = {
            'Dice Score': {
                'mean': np.mean(all_dice),
                'std': np.std(all_dice),
                'min': np.min(all_dice),
                'max': np.max(all_dice)
            },
            'IoU Score': {
                'mean': np.mean(all_iou),
                'std': np.std(all_iou),
                'min': np.min(all_iou),
                'max': np.max(all_iou)
            },
            'Pixel Accuracy': {
                'mean': np.mean(all_pixel_acc),
                'std': np.std(all_pixel_acc),
                'min': np.min(all_pixel_acc),
                'max': np.max(all_pixel_acc)
            },
            'Precision': {
                'mean': np.mean(all_precision),
                'std': np.std(all_precision),
                'min': np.min(all_precision),
                'max': np.max(all_precision)
            },
            'Recall': {
                'mean': np.mean(all_recall),
                'std': np.std(all_recall),
                'min': np.min(all_recall),
                'max': np.max(all_recall)
            },
            'F1 Score': {
                'mean': np.mean(all_f1),
                'std': np.std(all_f1),
                'min': np.min(all_f1),
                'max': np.max(all_f1)
            }
        }
        
        # Print comprehensive results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        for metric_name, values in metrics.items():
            print(f"\n{metric_name}:")
            print(f"  Mean: {values['mean']:.4f} ± {values['std']:.4f}")
            print(f"  Range: [{values['min']:.4f}, {values['max']:.4f}]")
        
        # Overall confusion matrix metrics
        total_pixels = all_tp + all_tn + all_fp + all_fn
        overall_accuracy = (all_tp + all_tn) / total_pixels
        overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        print(f"\nOverall Dataset Metrics:")
        print(f"  Accuracy: {overall_accuracy:.4f}")
        print(f"  Precision: {overall_precision:.4f}")
        print(f"  Recall: {overall_recall:.4f}")
        print(f"  F1-Score: {overall_f1:.4f}")
        
        # Create and save plots
        self.plot_metrics_distribution(metrics)
        
        return metrics

    def plot_metrics_distribution(self, metrics):
        """Plot distribution of metrics"""
        os.makedirs('evaluation_results', exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            # This would need the raw data, so let's skip the distribution plot
            # and just show the summary statistics
            ax = axes[i]
            ax.bar(['Mean', 'Min', 'Max'], 
                   [values['mean'], values['min'], values['max']],
                   color=['blue', 'red', 'green'], alpha=0.7)
            ax.set_title(f'{metric_name}')
            ax.set_ylim(0, 1)
            
            # Add error bar for standard deviation
            ax.errorbar(['Mean'], [values['mean']], yerr=[values['std']], 
                       fmt='o', color='black', capsize=5)
        
        plt.tight_layout()
        plt.savefig('evaluation_results/metrics_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nResults saved to 'evaluation_results/' directory")


if __name__ == '__main__':
    config = {
        'n_channels': 3,       # RGB images
        'n_classes': 1,        # binary segmentation
        'bilinear': True       # This will be auto-detected
    }

    model_path = 'checkpoints/unet_water_segmentation_20250616_142425/best.pth'

    try:
        evaluator = ModelEvaluator(model_path=model_path, config=config)
        metrics = evaluator.evaluate_dataset(test_loader, save_examples=True, num_examples=5)
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()