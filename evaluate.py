#!/usr/bin/env python3
"""
Unified evaluation script for segmentation models.

This script provides a framework to evaluate trained models interactively.
It supports:
- Evaluating single models or an ensemble of two models.
- Applying Test-Time Augmentation (TTA) for potentially better results.
- Calculating a comprehensive set of metrics (IoU, Dice, Precision, etc.).
- Saving visual examples and metric distribution plots.
"""
import os
import sys
import json
import glob
import inspect
import traceback
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import ttach as tta
import segmentation_models_pytorch as smp

# Add project root to path to allow for clean imports from other directories
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Local Imports ---
# Corrected import path for get_data_loaders
from utils.prepare_data import get_data_loaders
from utils.metrics import (dice_score, iou_score, pixel_accuracy,
                           precision_recall_f1, confusion_matrix_metrics)
from models.aer_unet import get_aer_unet_model
from models.segformer import get_segformer_model


# --- MODEL FACTORY ---

def get_smp_unet_plus_plus_effnetb4(n_channels=3, n_classes=1, **kwargs):
    """Factory for the segmentation_models_pytorch U-Net++ model."""
    return smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=n_channels,
        classes=n_classes
    )

def get_model_from_name(model_name):
    """
    Returns the correct model-building function based on its name.

    Args:
        model_name (str): The name of the model architecture.

    Returns:
        function: The corresponding model factory function.
    """
    model_map = {
        'aer-unet': get_aer_unet_model,
        'unet++-pretrained-encoder': get_smp_unet_plus_plus_effnetb4,
        'segformer-b4': get_segformer_model,
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown model name: {model_name}. Supported models are {list(model_map.keys())}")
    return model_map[model_name]


# --- HELPER FUNCTIONS ---

def select_checkpoint_interactive():
    """Interactively prompts the user to select a trained model checkpoint."""
    checkpoint_dir = 'checkpoints'
    if not os.path.isdir(checkpoint_dir):
        print(f"Error: Checkpoints directory '{checkpoint_dir}' not found.")
        return None

    available_experiments = sorted([d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))])
    if not available_experiments:
        print("No trained models found in the 'checkpoints' directory.")
        return None

    print("\nPlease select a model experiment:")
    for i, name in enumerate(available_experiments):
        print(f"  {i + 1}: {name}")

    try:
        exp_choice = int(input(f"Enter experiment number (1-{len(available_experiments)}): ")) - 1
        if not 0 <= exp_choice < len(available_experiments):
            raise ValueError
    except (ValueError, IndexError):
        print("Invalid selection.")
        return None

    selected_experiment_path = os.path.join(checkpoint_dir, available_experiments[exp_choice])
    checkpoint_files = sorted(glob.glob(os.path.join(selected_experiment_path, '*.pth')))

    if not checkpoint_files:
        print(f"Error: No .pth checkpoint files found in '{available_experiments[exp_choice]}'.")
        return None

    print("\nPlease select a specific checkpoint file to load:")
    for i, path in enumerate(checkpoint_files):
        print(f"  {i + 1}: {os.path.basename(path)}")

    try:
        ckpt_choice = int(input(f"Enter checkpoint number (1-{len(checkpoint_files)}): ")) - 1
        if not 0 <= ckpt_choice < len(checkpoint_files):
            raise ValueError
    except (ValueError, IndexError):
        print("Invalid selection.")
        return None

    return checkpoint_files[ckpt_choice]

def save_prediction_example(image, true_mask, pred_mask, prob_mask, dice, iou, idx, save_dir):
    """Saves a visual comparison of the original image, ground truth, and prediction."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # De-normalize image for visualization (assuming standard ImageNet stats)
    img_np = image.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)

    # Prepare masks for display
    true_mask_np = true_mask.squeeze().cpu().numpy()
    pred_mask_np = pred_mask.squeeze().cpu().numpy()
    prob_mask_np = prob_mask.squeeze().cpu().numpy()

    # Plotting
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(true_mask_np, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    axes[2].imshow(pred_mask_np, cmap='gray')
    axes[2].set_title(f'Predicted Mask\nIoU: {iou:.4f} | Dice: {dice:.4f}')
    axes[2].axis('off')

    im = axes[3].imshow(prob_mask_np, cmap='viridis', vmin=0, vmax=1)
    axes[3].set_title('Probability Map')
    axes[3].axis('off')
    fig.colorbar(im, ax=axes[3])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'example_{idx}.png'), dpi=150)
    plt.close()

def log_and_plot_results(metrics_dict, cm_totals, save_dir):
    """Logs metrics to the console and saves distribution plots."""
    print("\n" + "="*60 + "\nEVALUATION RESULTS\n" + "="*60)

    # Print summary for each metric
    for name, values in metrics_dict.items():
        if not values:
            print(f"\n{name.replace('_', ' ').capitalize()} Score: No values to calculate.")
            continue
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        print(f"\n{name.replace('_', ' ').capitalize()} Score:")
        print(f"  Mean: {mean_val:.4f} ± {std_val:.4f}")
        print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")

    # Calculate overall metrics from total True Positives, False Positives, etc.
    tp = cm_totals['tp']
    fp = cm_totals['fp']
    fn = cm_totals['fn']

    # Corrected and more readable calculation for Precision, Recall, and F1
    overall_precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
    overall_recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
    f1_denominator = overall_precision + overall_recall
    overall_f1 = (2 * overall_precision * overall_recall / f1_denominator) if f1_denominator > 0 else 0

    print("\nOverall Dataset Metrics (from total counts):")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall (Sensitivity): {overall_recall:.4f}")
    print(f"  F1-Score: {overall_f1:.4f}")

    # Plot and save histograms of metric distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    sns.set_theme(style="whitegrid")
    for i, (name, values) in enumerate(metrics_dict.items()):
        if values:
            sns.histplot(values, kde=True, ax=axes[i], bins=20)
            axes[i].set_title(f"Distribution of {name.replace('_', ' ').capitalize()}")
            axes[i].set_xlabel("Score")
            axes[i].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_distribution.png'), dpi=150)
    plt.close()

    print(f"\nEvaluation complete. All results saved in '{save_dir}'")

def visualize_for_verification(image, true_mask):
    """Displays an image and its ground truth mask for user verification."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Manual Verification: Is the mask on the right correct?", fontsize=14)

    # De-normalize image for display
    img_np = image.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)

    true_mask_np = true_mask.squeeze().cpu().numpy()

    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(true_mask_np, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    # plt.show() will pause the script until the user closes the plot window
    plt.show()


# --- EVALUATOR CLASSES ---

class ModelEvaluator:
    """Handles the evaluation process for a single model."""
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model_path = model_path
        checkpoint = torch.load(model_path, map_location=self.device)
        
        config = checkpoint.get('config')
        if not config:
            raise ValueError("Checkpoint is missing the 'config' dictionary.")
            
        model_type = config.get('model_type')
        print(f"\nLoading model '{model_type}' with configuration from checkpoint...")

        # Re-create model using the factory and the config saved in the checkpoint
        get_model_func = get_model_from_name(model_type)
        sig = inspect.signature(get_model_func)
        model_args = {k: v for k, v in config.items() if k in sig.parameters}
        
        self.model = get_model_func(**model_args).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Successfully loaded model from epoch {checkpoint.get('epoch', 'N/A')}.")

    def evaluate_dataset(self, dataloader, num_examples=5, use_tta=False):
        """Runs the evaluation loop on a given dataloader."""
        all_metrics = {'dice': [], 'iou': [], 'pixel_acc': [], 'precision': [], 'recall': [], 'f1': []}
        cm_totals = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

        # Setup results directory
        model_name_for_dir = os.path.basename(os.path.dirname(self.model_path))
        eval_dir_suffix = "TTA" if use_tta else "Base"
        eval_dir = f"evaluation_results/{model_name_for_dir}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{eval_dir_suffix}"
        os.makedirs(eval_dir, exist_ok=True)
        low_iou_dir = os.path.join(eval_dir, 'low_iou_predictions')
        os.makedirs(low_iou_dir, exist_ok=True)

        print(f"\nEvaluating model... (TTA: {'Enabled' if use_tta else 'Disabled'})")
        print(f"Results will be saved to '{eval_dir}'")
        
        # Wrap model with TTA if enabled
        eval_model = self.model
        if use_tta:
            tta_transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
            eval_model = tta.SegmentationTTAWrapper(self.model, tta_transforms, merge_mode='mean')

        with torch.no_grad():
            for i, (images, masks) in enumerate(tqdm(dataloader, desc="Evaluating")):
                images_gpu, masks_gpu = images.to(self.device), masks.to(self.device)
                
                # Forward pass
                outputs = eval_model(images_gpu)
                # Handle SegFormer's different output format
                if hasattr(outputs, 'logits'):
                    outputs = F.interpolate(outputs.logits, size=images_gpu.shape[2:], mode='bilinear', align_corners=False)
                
                # Get probabilities and process each image in the batch
                probs = torch.sigmoid(outputs).cpu()
                for j in range(images.size(0)):
                    prob_j = probs[j].squeeze()
                    mask_j = masks[j].squeeze()
                    pred_j = (prob_j > 0.5).float()
                    
                    # --- Calculate and store all metrics for this sample ---
                    current_dice = dice_score(pred_j, mask_j).item()
                    current_iou = iou_score(pred_j, mask_j).item()
                    all_metrics['dice'].append(current_dice)
                    all_metrics['iou'].append(current_iou)
                    all_metrics['pixel_acc'].append(pixel_accuracy(pred_j, mask_j).item())
                    
                    precision, recall, f1 = precision_recall_f1(pred_j, mask_j)
                    all_metrics['precision'].append(precision.item())
                    all_metrics['recall'].append(recall.item())
                    all_metrics['f1'].append(f1.item())
                    
                    # Update confusion matrix totals
                    cm = confusion_matrix_metrics(pred_j, mask_j)
                    cm_totals['tp'] += cm['TP']
                    cm_totals['tn'] += cm['TN']
                    cm_totals['fp'] += cm['FP']
                    cm_totals['fn'] += cm['FN']
                    
                    # Save examples of poor predictions for analysis
                    if current_iou < 0.8:
                        save_prediction_example(images[j], masks[j], pred_j, prob_j, current_dice, current_iou, f"low_iou_{i*dataloader.batch_size+j}", low_iou_dir)
                    
                    # Save a few initial examples regardless of score
                    if i * dataloader.batch_size + j < num_examples:
                        save_prediction_example(images[j], masks[j], pred_j, prob_j, current_dice, current_iou, f"example_{i*dataloader.batch_size+j}", eval_dir)

        log_and_plot_results(all_metrics, cm_totals, eval_dir)

class EnsembleEvaluator:
    """Handles the evaluation process for an ensemble of two models."""
    def __init__(self, model_path_a, model_path_b):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f"Using device: {self.device}")
        print("\n--- Loading Model A for Ensemble ---"); self.model_a, self.config_a = self._load_model_from_path(model_path_a)
        print("\n--- Loading Model B for Ensemble ---"); self.model_b, self.config_b = self._load_model_from_path(model_path_b)
    def _load_model_from_path(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device); config = checkpoint.get('config')
        if not config: raise ValueError(f"Checkpoint '{model_path}' is missing the 'config' dictionary.")
        model_type = config.get('model_type'); print(f"Loading model '{model_type}' from {os.path.basename(model_path)}...")
        get_model_func = get_model_from_name(model_type); sig = inspect.signature(get_model_func)
        model_args = {k: v for k, v in config.items() if k in sig.parameters}
        model = get_model_func(**model_args).to(self.device); model.load_state_dict(checkpoint['model_state_dict']); model.eval()
        print(f"Successfully loaded model from epoch {checkpoint.get('epoch', 'N/A')}."); return model, config

    def evaluate_dataset(self, dataloader, num_examples=5, use_tta=False, low_iou_threshold=0.42):
        """Runs the ensemble evaluation loop with interactive verification."""
        all_metrics = {'dice': [], 'iou': [], 'pixel_acc': [], 'precision': [], 'recall': [], 'f1': []}
        cm_totals = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        
        model_a_name = self.config_a.get('model_type', 'modelA'); model_b_name = self.config_b.get('model_type', 'modelB')
        eval_dir_suffix = "TTA" if use_tta else "Base"
        eval_dir = f"evaluation_results/Ensemble_{model_a_name}+{model_b_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{eval_dir_suffix}"
        os.makedirs(eval_dir, exist_ok=True); low_iou_dir = os.path.join(eval_dir, 'low_iou_predictions'); os.makedirs(low_iou_dir, exist_ok=True)
        
        print(f"\nEvaluating Ensemble... (TTA: {'Enabled' if use_tta else 'Disabled'})")
        print(f"Results will be saved to '{eval_dir}'")
        
        model_a_eval, model_b_eval = self.model_a, self.model_b
        if use_tta:
            tta_transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()]); model_a_eval = tta.SegmentationTTAWrapper(self.model_a, tta_transforms, merge_mode='mean'); model_b_eval = tta.SegmentationTTAWrapper(self.model_b, tta_transforms, merge_mode='mean')

        with torch.no_grad():
            for i, (images, masks) in enumerate(tqdm(dataloader, desc="Evaluating Ensemble")):
                images_gpu = images.to(self.device)
                outputs_a = model_a_eval(images_gpu); outputs_b = model_b_eval(images_gpu)
                if hasattr(outputs_a, 'logits'): outputs_a = F.interpolate(outputs_a.logits, size=images_gpu.shape[2:], mode='bilinear', align_corners=False)
                if hasattr(outputs_b, 'logits'): outputs_b = F.interpolate(outputs_b.logits, size=images_gpu.shape[2:], mode='bilinear', align_corners=False)
                probs_a, probs_b = torch.sigmoid(outputs_a), torch.sigmoid(outputs_b)
                ensemble_probs = ((probs_a + probs_b) / 2.0).cpu()
                
                for j in range(images.size(0)):
                    prob_j, mask_j, pred_j = ensemble_probs[j].squeeze(), masks[j].squeeze(), (ensemble_probs[j].squeeze() > 0.5).float()
                    
                    # Calculate metrics for the current sample
                    current_iou = iou_score(pred_j, mask_j).item()
                    
                    # --- Interactive Verification Logic ---
                    is_broken = False
                    if current_iou < low_iou_threshold:
                        print(f"\nSample {i*dataloader.batch_size+j} has low IoU ({current_iou:.4f}). Displaying for verification.")
                        visualize_for_verification(images[j], masks[j])
                        
                        while True:
                            user_input = input("Is the ground truth mask 'right' or 'broken'? [r/b]: ").lower()
                            if user_input == 'b':
                                is_broken = True
                                print("Sample marked as 'broken' and will be EXCLUDED from metrics.")
                                break
                            elif user_input == 'r':
                                print("Sample marked as 'right' and will be INCLUDED in metrics.")
                                break
                            else:
                                print("Invalid input. Please enter 'r' for right or 'b' for broken.")

                    # Only add metrics to the final tally if the sample is not broken
                    if not is_broken:
                        current_dice = dice_score(pred_j, mask_j).item()
                        all_metrics['dice'].append(current_dice); all_metrics['iou'].append(current_iou); all_metrics['pixel_acc'].append(pixel_accuracy(pred_j, mask_j).item())
                        precision, recall, f1 = precision_recall_f1(pred_j, mask_j)
                        all_metrics['precision'].append(precision.item()); all_metrics['recall'].append(recall.item()); all_metrics['f1'].append(f1.item())
                        cm = confusion_matrix_metrics(pred_j, mask_j)
                        cm_totals['tp'] += cm['TP']; cm_totals['tn'] += cm['TN']; cm_totals['fp'] += cm['FP']; cm_totals['fn'] += cm['FN']
                        
                        # Save visual examples of included low-IoU predictions
                        if current_iou < low_iou_threshold:
                            save_prediction_example(images[j], masks[j], pred_j, prob_j, current_dice, current_iou, f"verified_low_iou_{i*dataloader.batch_size+j}", low_iou_dir)
                    
                    if i * dataloader.batch_size + j < num_examples and not is_broken:
                        save_prediction_example(images[j], masks[j], pred_j, prob_j, dice_score(pred_j, mask_j).item(), current_iou, f"example_{i*dataloader.batch_size+j}", eval_dir)

        log_and_plot_results(all_metrics, cm_totals, eval_dir)

def main():
    """Main function to run the interactive evaluation menu."""
    parser = argparse.ArgumentParser(description='Unified Segmentation Model Evaluation')
    parser.add_argument('--path', type=str, default=None, help='(Optional) Direct path to a model checkpoint for non-interactive scripting.')
    parser.add_argument('--examples', type=int, default=10, help='Number of visual examples to save.')
    args = parser.parse_args()

    # Non-interactive mode
    if args.path:
        if not os.path.exists(args.path):
            print(f"Error: Model path does not exist: {args.path}")
            return
        try:
            evaluator = ModelEvaluator(model_path=args.path)
            _, _, test_loader = get_data_loaders(batch_size=16)
            if test_loader:
                evaluator.evaluate_dataset(test_loader, num_examples=args.examples, use_tta=False)
        except Exception as e:
            print(f"\nAn error occurred during non-interactive evaluation: {e}")
            traceback.print_exc()
        return

    # --- Interactive Menu ---
    print("\n--- Evaluation Menu ---")
    print("1. Evaluate a Single Model")
    print("2. Evaluate an Ensemble of Two Models")
    
    try:
        choice = input("Enter your choice (1-2): ")
        evaluator = None

        if choice == '1':
            model_path = select_checkpoint_interactive()
            if model_path:
                evaluator = ModelEvaluator(model_path=model_path)
        elif choice == '2':
            print("\n--- Select Model A for the Ensemble ---")
            model_path_a = select_checkpoint_interactive()
            if not model_path_a: return
            
            print("\n--- Select Model B for the Ensemble ---")
            model_path_b = select_checkpoint_interactive()
            if not model_path_b: return

            if model_path_a == model_path_b:
                print("Warning: You selected the same model twice for the ensemble.")
            
            # --- FIX: This instantiates the real EnsembleEvaluator ---
            # It no longer prints a placeholder and exits.
            evaluator = EnsembleEvaluator(model_path_a, model_path_b)

        else:
            print("Invalid choice.")
            return

        # If an evaluator was successfully created, proceed with evaluation
        if evaluator:
            low_iou_threshold = 0.42
            if isinstance(evaluator, EnsembleEvaluator):
                try:
                    low_iou_threshold = float(input("\nEnter IoU threshold for manual verification (default: 0.42): ") or "0.42")
                except ValueError:
                    print("Invalid input. Using default threshold of 0.42.")
            
            use_tta = (input("\nEnable Test-Time Augmentation (TTA)? (y/N): ").lower() == 'y')
            
            print("Loading test data...")
            _, _, test_loader = get_data_loaders(batch_size=8)
            if test_loader is None: print("Failed to load test data."); return

            # Pass the threshold to the evaluate_dataset method
            if isinstance(evaluator, EnsembleEvaluator):
                evaluator.evaluate_dataset(test_loader, num_examples=args.examples, use_tta=use_tta, low_iou_threshold=low_iou_threshold)
            else:
                # Single model evaluator does not have this feature
                evaluator.evaluate_dataset(test_loader, num_examples=args.examples, use_tta=use_tta)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()