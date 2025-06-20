#!/usr/bin/env python3
"""
Unified evaluation script that handles all model architectures.
Loads model configurations directly from saved checkpoints for reliability.
Allows for interactive selection of model checkpoints to evaluate.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import argparse
import sys
from datetime import datetime
import seaborn as sns
import inspect

# Add project root to path to allow for clean imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.metrics import (dice_score, iou_score, pixel_accuracy, 
                          precision_recall_f1, confusion_matrix_metrics)
from src.preprocessing.prepare_data import test_loader

# Import all model factory functions
from models.unet import get_model as get_unet_model
from models.unet_enhanced import get_enhanced_model
from models.attention import get_attention_model
from models.unet_plus_plus import get_unet_plus_plus_model
from models.aer_unet import get_aer_unet_model

def get_model_from_name(model_name):
    """Returns the correct model factory function based on its name."""
    model_map = {
        'unet': get_unet_model,
        'enhanced': get_enhanced_model,
        'attention': get_attention_model,
        'unet++': get_unet_plus_plus_model,
        'aer-unet': get_aer_unet_model
    }
    # Handle legacy naming conventions for backward compatibility
    if model_name == "unet_enhanced": model_name = "enhanced"

    if model_name not in model_map:
        raise ValueError(f"Unknown model name: {model_name}. Supported models are {list(model_map.keys())}")
    return model_map[model_name]

def select_checkpoint_interactive():
    """Scans the checkpoints directory and lets the user choose which one to evaluate."""
    checkpoint_dir = 'checkpoints'
    if not os.path.isdir(checkpoint_dir):
        print(f"Error: Checkpoints directory '{checkpoint_dir}' not found.")
        return None, None

    available_checkpoints = [d for d in sorted(os.listdir(checkpoint_dir)) if os.path.isdir(os.path.join(checkpoint_dir, d)) and 'best.pth' in os.listdir(os.path.join(checkpoint_dir, d))]

    if not available_checkpoints:
        print("No valid checkpoints found. (A valid checkpoint folder must contain a 'best.pth' file).")
        return None, None

    print("\nPlease select a model checkpoint to evaluate:")
    for i, name in enumerate(available_checkpoints):
        print(f"  {i + 1}: {name}")
    
    try:
        choice = int(input(f"\nEnter number (1-{len(available_checkpoints)}): ")) - 1
        if not 0 <= choice < len(available_checkpoints):
            raise ValueError
    except (ValueError, IndexError):
        print("Invalid selection.")
        return None, None

    selected_dir = available_checkpoints[choice]
    checkpoint_path = os.path.join(checkpoint_dir, selected_dir, 'best.pth')

    return checkpoint_path

class ModelEvaluator:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # --- Simplified Model Loading ---
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'config' not in checkpoint:
            raise ValueError("Cannot evaluate model: Checkpoint is missing the 'config' dictionary.")

        config = checkpoint['config']
        model_type = config.get('model_type')
        
        print(f"\nLoading model '{model_type}' with configuration from checkpoint...")

        get_model_func = get_model_from_name(model_type)
        
        # Filter config to only pass relevant args to the model function
        sig = inspect.signature(get_model_func)
        model_args = {k: v for k, v in config.items() if k in sig.parameters}
        
        self.model = get_model_func(**model_args).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Successfully loaded model from epoch {checkpoint.get('epoch', 'N/A')}.")
        if 'iou_score' in checkpoint:
            print(f"Checkpoint achieved best validation IoU: {checkpoint['iou_score']:.4f}")

    def save_prediction_example(self, image, true_mask, pred_mask, prob_mask, dice, iou, idx, save_dir):
        """Save visualization of a prediction example."""
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        img_np = image.permute(1, 2, 0).cpu().numpy()
        true_mask_np = true_mask.squeeze().cpu().numpy()
        pred_mask_np = pred_mask.squeeze().cpu().numpy()
        prob_mask_np = prob_mask.squeeze().cpu().numpy()
        
        img_np = np.clip(img_np, 0, 1)
        
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(true_mask_np, cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        axes[2].imshow(pred_mask_np, cmap='gray')
        axes[2].set_title(f'Predicted Mask\nDice: {dice:.4f}')
        axes[2].axis('off')
        
        im = axes[3].imshow(prob_mask_np, cmap='viridis', vmin=0, vmax=1)
        axes[3].set_title(f'Probability Map\nIoU: {iou:.4f}')
        axes[3].axis('off')
        
        fig.colorbar(im, ax=axes[3])
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'example_{idx}.png'), dpi=150)
        plt.close()

    def evaluate_dataset(self, dataloader, num_examples=5):
        """Evaluate model on a dataset and compute comprehensive metrics."""
        all_metrics = {'dice': [], 'iou': [], 'pixel_acc': [], 'precision': [], 'recall': [], 'f1': []}
        cm_totals = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        
        eval_dir = f"evaluation_results/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(eval_dir, exist_ok=True)
        
        print(f"\nEvaluating model... Results will be saved to '{eval_dir}'")
        with torch.no_grad():
            for i, (images, masks) in enumerate(tqdm(dataloader, desc="Evaluating")):
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                if isinstance(outputs, list):
                    outputs = outputs[-1]
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                for j in range(images.size(0)):
                    pred_j, mask_j = preds[j], masks[j]
                    
                    all_metrics['dice'].append(dice_score(pred_j, mask_j).item())
                    all_metrics['iou'].append(iou_score(pred_j, mask_j).item())
                    all_metrics['pixel_acc'].append(pixel_accuracy(pred_j, mask_j).item())
                    
                    precision, recall, f1 = precision_recall_f1(pred_j, mask_j)
                    all_metrics['precision'].append(precision.item())
                    all_metrics['recall'].append(recall.item())
                    all_metrics['f1'].append(f1.item())
                    
                    cm = confusion_matrix_metrics(pred_j, mask_j)
                    cm_totals['tp'] += cm['TP']
                    cm_totals['tn'] += cm['TN']
                    cm_totals['fp'] += cm['FP']
                    cm_totals['fn'] += cm['FN']

                    if i * dataloader.batch_size + j < num_examples:
                        self.save_prediction_example(images[j], masks[j], pred_j, probs[j], all_metrics['dice'][-1], all_metrics['iou'][-1], i * dataloader.batch_size + j, eval_dir)
        
        self.log_and_plot_results(all_metrics, cm_totals, eval_dir)
        
    def log_and_plot_results(self, metrics_dict, cm_totals, save_dir):
        """Prints, plots, and saves all evaluation results."""
        print("\n" + "="*60 + "\nEVALUATION RESULTS\n" + "="*60)
        
        for name, values in metrics_dict.items():
            print(f"\n{name.replace('_', ' ').capitalize()} Score:")
            print(f"  Mean: {np.mean(values):.4f} ± {np.std(values):.4f}")
            print(f"  Range: [{np.min(values):.4f}, {np.max(values):.4f}]")
        
        overall_precision = cm_totals['tp'] / (cm_totals['tp'] + cm_totals['fp']) if (cm_totals['tp'] + cm_totals['fp']) > 0 else 0
        overall_recall = cm_totals['tp'] / (cm_totals['tp'] + cm_totals['fn']) if (cm_totals['tp'] + cm_totals['fn']) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        print("\nOverall Dataset Metrics (from total counts):")
        print(f"  Precision: {overall_precision:.4f}")
        print(f"  Recall (Sensitivity): {overall_recall:.4f}")
        print(f"  F1-Score: {overall_f1:.4f}")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        sns.set_theme(style="whitegrid")
        for i, (name, values) in enumerate(metrics_dict.items()):
            sns.histplot(values, kde=True, ax=axes[i], bins=20)
            axes[i].set_title(f"Distribution of {name.replace('_', ' ').capitalize()}")
            axes[i].set_xlabel("Score")
            axes[i].set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'metrics_distribution.png'), dpi=150)
        plt.close()

        print(f"\nEvaluation complete. All results saved in '{save_dir}'")

def main():
    parser = argparse.ArgumentParser(description='Unified Segmentation Model Evaluation')
    parser.add_argument('--path', type=str, default=None, help='(Optional) Direct path to a specific model checkpoint (.pth file).')
    parser.add_argument('--examples', type=int, default=10, help='Number of visual examples to save.')
    
    args = parser.parse_args()

    model_path = args.path
    
    if model_path is None:
        model_path = select_checkpoint_interactive()
        if model_path is None:
            return 

    if not os.path.exists(model_path):
        print(f"Error: Model path does not exist: {model_path}")
        return

    try:
        evaluator = ModelEvaluator(model_path=model_path)
        evaluator.evaluate_dataset(test_loader, num_examples=args.examples)
    except Exception as e:
        print(f"\nAn error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
