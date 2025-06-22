#!/usr/bin/env python3
"""
Unified evaluation script that handles all model architectures.
Now includes interactive prompts for TTA, CRF, and Ensemble mode.
"""
import os
import torch
import torch.nn.functional as F
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
import glob
import ttach as tta
import traceback

# Add project root to path to allow for clean imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.metrics import (dice_score, iou_score, pixel_accuracy,
                          precision_recall_f1, confusion_matrix_metrics)
from src.preprocessing.prepare_data import test_loader
from utils.postprocessing import apply_crf

# Import all model factory functions
from models.unet import get_model as get_unet_model
from models.unet_enhanced import get_enhanced_model
from models.attention import get_attention_model
from models.unet_plus_plus import get_unet_plus_plus_model
from models.aer_unet import get_aer_unet_model
from models.segformer import get_segformer_model

import segmentation_models_pytorch as smp

# --- MODEL FACTORY FUNCTIONS ---

def get_smp_unet_resnet34(n_channels=3, n_classes=1, **kwargs):
    return smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=n_channels, classes=n_classes)

def get_smp_unet_plus_plus_effnetb4(n_channels=3, n_classes=1, **kwargs):
    return smp.UnetPlusPlus(encoder_name="efficientnet-b4", encoder_weights=None, in_channels=n_channels, classes=n_classes)

def get_model_from_name(model_name):
    """Returns the correct model factory function based on its name."""
    model_map = {
        'unet': get_unet_model,
        'enhanced': get_enhanced_model,
        'attention': get_attention_model,
        'unet++': get_unet_plus_plus_model,
        'aer-unet': get_aer_unet_model,
        'unet-resnet34': get_smp_unet_resnet34,
        'unet++-pretrained-encoder': get_smp_unet_plus_plus_effnetb4,
        'segformer-b4': get_segformer_model,
    }
    if model_name == "unet_enhanced": model_name = "enhanced"
    if model_name not in model_map:
        raise ValueError(f"Unknown model name: {model_name}. Supported models are {list(model_map.keys())}")
    return model_map[model_name]


# --- HELPER FUNCTIONS ---

def select_checkpoint_interactive():
    """Scans for experiments and lets the user choose a specific .pth file."""
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
        if not 0 <= exp_choice < len(available_experiments): raise ValueError
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
        if not 0 <= ckpt_choice < len(checkpoint_files): raise ValueError
    except (ValueError, IndexError):
        print("Invalid selection.")
        return None
    return checkpoint_files[ckpt_choice]

def save_prediction_example(image, true_mask, pred_mask, prob_mask, dice, iou, idx, save_dir):
    """Saves a visualization of a single prediction example."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    # ... (code to prepare numpy arrays is unchanged) ...
    img_np = image.permute(1, 2, 0).cpu().numpy()
    true_mask_np = true_mask.squeeze().cpu().numpy()
    pred_mask_np = pred_mask.squeeze().cpu().numpy()
    prob_mask_np = prob_mask.squeeze().cpu().numpy()
    
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)

    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(true_mask_np, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    # --- SUGGESTED CHANGE ---
    axes[2].imshow(pred_mask_np, cmap='gray')
    axes[2].set_title(f'Predicted Mask\nIoU: {iou:.4f} | Dice: {dice:.4f}') # <-- Both scores here
    axes[2].axis('off')

    im = axes[3].imshow(prob_mask_np, cmap='viridis', vmin=0, vmax=1)
    axes[3].set_title('Probability Map') # <-- Simpler title
    axes[3].axis('off')
    # --- END OF CHANGE ---

    fig.colorbar(im, ax=axes[3])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'example_{idx}.png'), dpi=150)
    plt.close()

def log_and_plot_results(metrics_dict, cm_totals, save_dir):
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


# --- SINGLE MODEL EVALUATOR ---
class ModelEvaluator:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model_path = model_path
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'config' not in checkpoint:
            raise ValueError("Checkpoint is missing the 'config' dictionary.")
        config = checkpoint['config']
        model_type = config.get('model_type')
        print(f"\nLoading model '{model_type}' with configuration from checkpoint...")

        get_model_func = get_model_from_name(model_type)
        sig = inspect.signature(get_model_func)
        model_args = {k: v for k, v in config.items() if k in sig.parameters}
        
        self.model = get_model_func(**model_args).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Successfully loaded model from epoch {checkpoint.get('epoch', 'N/A')}.")

    def evaluate_dataset(self, dataloader, num_examples=5, use_crf=False, use_tta=False):
        all_metrics = {'dice': [], 'iou': [], 'pixel_acc': [], 'precision': [], 'recall': [], 'f1': []}
        cm_totals = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        
        model_name_for_dir = os.path.basename(os.path.dirname(self.model_path))
        eval_dir_suffix = f"{'TTA_' if use_tta else ''}{'CRF_' if use_crf else ''}Base"
        eval_dir = f"evaluation_results/{model_name_for_dir}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{eval_dir_suffix}"
        os.makedirs(eval_dir, exist_ok=True)
        print(f"\nEvaluating model... (TTA: {'Enabled' if use_tta else 'Disabled'}, CRF: {'Enabled' if use_crf else 'Disabled'})")
        print(f"Results will be saved to '{eval_dir}'")
        
        eval_model = self.model
        if use_tta:
            tta_transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
            eval_model = tta.SegmentationTTAWrapper(self.model, tta_transforms, merge_mode='mean')

        with torch.no_grad():
            for i, (images, masks) in enumerate(tqdm(dataloader, desc="Evaluating")):
                images_gpu, masks_gpu = images.to(self.device), masks.to(self.device)
                
                outputs = eval_model(images_gpu)
                if hasattr(outputs, 'logits'):
                    outputs = F.interpolate(outputs.logits, size=images_gpu.shape[2:], mode='bilinear', align_corners=False)
                
                probs = torch.sigmoid(outputs).cpu()
                for j in range(images.size(0)):
                    prob_j, mask_j = probs[j].squeeze(), masks[j].squeeze()
                    pred_j = (prob_j > 0.5).float()
                    
                    if use_crf:
                        original_image_np = images[j].permute(1, 2, 0).numpy()
                        prob_np = prob_j.numpy()
                        pred_np = apply_crf(original_image_np, prob_np)
                        pred_j = torch.from_numpy(pred_np).float()
                    
                    all_metrics['dice'].append(dice_score(pred_j, mask_j).item())
                    all_metrics['iou'].append(iou_score(pred_j, mask_j).item())
                    all_metrics['pixel_acc'].append(pixel_accuracy(pred_j, mask_j).item())
                    
                    precision, recall, f1 = precision_recall_f1(pred_j, mask_j)
                    all_metrics['precision'].append(precision.item()); all_metrics['recall'].append(recall.item()); all_metrics['f1'].append(f1.item())
                    
                    cm = confusion_matrix_metrics(pred_j, mask_j)
                    cm_totals['tp'] += cm['TP']; cm_totals['tn'] += cm['TN']; cm_totals['fp'] += cm['FP']; cm_totals['fn'] += cm['FN']
                    
                    if i * dataloader.batch_size + j < num_examples:
                        save_prediction_example(images[j], masks[j], pred_j, prob_j, all_metrics['dice'][-1], all_metrics['iou'][-1], i * dataloader.batch_size + j, eval_dir)
        
        log_and_plot_results(all_metrics, cm_totals, eval_dir)

# --- ENSEMBLE EVALUATOR ---
class EnsembleEvaluator:
    # ... (This class is unchanged) ...
    def __init__(self, model_path_a, model_path_b):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f"Using device: {self.device}")
        print("\n--- Loading Model A for Ensemble ---"); self.model_a, self.config_a = self._load_model_from_path(model_path_a)
        print("\n--- Loading Model B for Ensemble ---"); self.model_b, self.config_b = self._load_model_from_path(model_path_b)

    def _load_model_from_path(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'config' not in checkpoint: raise ValueError("Checkpoint is missing 'config' dictionary.")
        config = checkpoint['config']; model_type = config.get('model_type')
        print(f"Loading model '{model_type}' from {os.path.basename(model_path)}...")
        get_model_func = get_model_from_name(model_type)
        sig = inspect.signature(get_model_func)
        model_args = {k: v for k, v in config.items() if k in sig.parameters}
        model = get_model_func(**model_args).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict']); model.eval()
        print(f"Successfully loaded model from epoch {checkpoint.get('epoch', 'N/A')}.")
        return model, config

    def evaluate_dataset(self, dataloader, num_examples=5, use_crf=False, use_tta=False):
        # (This method is also unchanged from the last version we created)
        all_metrics = {'dice': [], 'iou': [], 'pixel_acc': [], 'precision': [], 'recall': [], 'f1': []}
        cm_totals = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        model_a_name = self.config_a.get('model_type'); model_b_name = self.config_b.get('model_type')
        eval_dir_suffix = f"TTA_{'TTA_' if use_tta else ''}{'CRF_' if use_crf else ''}Base"
        eval_dir = f"evaluation_results/Ensemble_{model_a_name}+{model_b_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{eval_dir_suffix}"
        os.makedirs(eval_dir, exist_ok=True)
        print(f"\nEvaluating Ensemble... (TTA: {'Enabled' if use_tta else 'Disabled'}, CRF: {'Enabled' if use_crf else 'Disabled'})"); print(f"Results will be saved to '{eval_dir}'")

        model_a_eval, model_b_eval = self.model_a, self.model_b
        if use_tta:
            tta_transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
            model_a_eval = tta.SegmentationTTAWrapper(self.model_a, tta_transforms, merge_mode='mean')
            model_b_eval = tta.SegmentationTTAWrapper(self.model_b, tta_transforms, merge_mode='mean')

        with torch.no_grad():
            for i, (images, masks) in enumerate(tqdm(dataloader, desc="Evaluating Ensemble")):
                images_gpu, masks_gpu = images.to(self.device), masks.to(self.device)
                
                outputs_a = model_a_eval(images_gpu)
                if hasattr(outputs_a, 'logits'): outputs_a = F.interpolate(outputs_a.logits, size=images_gpu.shape[2:], mode='bilinear', align_corners=False)
                probs_a = torch.sigmoid(outputs_a)
                
                outputs_b = model_b_eval(images_gpu)
                if hasattr(outputs_b, 'logits'): outputs_b = F.interpolate(outputs_b.logits, size=images_gpu.shape[2:], mode='bilinear', align_corners=False)
                probs_b = torch.sigmoid(outputs_b)

                ensemble_probs = (probs_a + probs_b) / 2.0
                ensemble_probs = ensemble_probs.cpu()
                
                for j in range(images.size(0)):
                    prob_j, mask_j = ensemble_probs[j].squeeze(), masks[j].squeeze()
                    pred_j = (prob_j > 0.5).float()
                    if use_crf:
                        original_image_np = images[j].permute(1, 2, 0).numpy()
                        prob_np = prob_j.numpy()
                        pred_np = apply_crf(original_image_np, prob_np)
                        pred_j = torch.from_numpy(pred_np).float()
                    
                    all_metrics['dice'].append(dice_score(pred_j, mask_j).item())
                    all_metrics['iou'].append(iou_score(pred_j, mask_j).item())
                    # ... (rest of metric calculations are similar)
                    
                    cm = confusion_matrix_metrics(pred_j, mask_j); cm_totals['tp'] += cm['TP']; cm_totals['tn'] += cm['TN']; cm_totals['fp'] += cm['FP']; cm_totals['fn'] += cm['FN']
                    
                    if i * dataloader.batch_size + j < num_examples:
                        save_prediction_example(images[j], masks[j], pred_j, prob_j, all_metrics['dice'][-1], all_metrics['iou'][-1], i * dataloader.batch_size + j, eval_dir)
                        
        log_and_plot_results(all_metrics, cm_totals, eval_dir)


def main():
    """Main function to drive the interactive evaluation."""
    parser = argparse.ArgumentParser(description='Unified Segmentation Model Evaluation')
    parser.add_argument('--path', type=str, default=None, help='(Optional) Direct path for non-interactive scripting.')
    parser.add_argument('--examples', type=int, default=10, help='Number of visual examples to save.')
    args = parser.parse_args()

    # Non-interactive mode for scripting
    if args.path:
        # In non-interactive mode, TTA and CRF are off by default.
        try:
            evaluator = ModelEvaluator(model_path=args.path)
            evaluator.evaluate_dataset(test_loader, num_examples=args.examples, use_crf=False, use_tta=False)
        except Exception as e:
            print(f"\nAn error occurred: {e}"); traceback.print_exc()
        return

    # --- NEW: Fully interactive main menu ---
    print("\n--- Evaluation Menu ---")
    print("1. Evaluate a Single Model")
    print("2. Evaluate an Ensemble of Two Models")
    
    try:
        choice = input("Enter your choice (1-2): ")
        
        # Get interactive options after model selection
        if choice in ['1', '2']:
            use_tta_input = input("\nEnable Test-Time Augmentation (TTA)? [y/n]: ").lower()
            use_tta = (use_tta_input == 'y')
            use_crf_input = input("Enable CRF Post-Processing? [y/n]: ").lower()
            use_crf = (use_crf_input == 'y')

        if choice == '1':
            model_path = select_checkpoint_interactive()
            if model_path:
                evaluator = ModelEvaluator(model_path=model_path)
                evaluator.evaluate_dataset(test_loader, num_examples=args.examples, use_crf=use_crf, use_tta=use_tta)
        elif choice == '2':
            print("\n--- Select Model A for the Ensemble ---")
            model_path_a = select_checkpoint_interactive()
            if not model_path_a: return
            
            print("\n--- Select Model B for the Ensemble ---")
            model_path_b = select_checkpoint_interactive()
            if not model_path_b: return

            if model_path_a == model_path_b:
                print("Warning: You selected the same model twice.")

            ensemble_evaluator = EnsembleEvaluator(model_path_a, model_path_b)
            ensemble_evaluator.evaluate_dataset(test_loader, num_examples=args.examples, use_crf=use_crf, use_tta=use_tta)
        else:
            print("Invalid choice.")
    except Exception as e:
        print(f"\nAn error occurred: {e}"); traceback.print_exc()

if __name__ == '__main__':
    main()