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
import traceback # <-- FIX: Import traceback at the top level

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

# --- MODEL FACTORY and HELPER FUNCTIONS (Unchanged) ---
def get_model_from_name(model_name):
    # ... (function is unchanged)
    model_map = {
        'unet': get_unet_model, 'enhanced': get_enhanced_model, 'attention': get_attention_model,
        'unet++': get_unet_plus_plus_model, 'aer-unet': get_aer_unet_model,
        'unet++-pretrained-encoder': get_smp_unet_plus_plus_effnetb4,
        'segformer-b4': get_segformer_model,
    }
    if model_name == "unet_enhanced": model_name = "enhanced"
    if model_name not in model_map:
        raise ValueError(f"Unknown model name: {model_name}. Supported models are {list(model_map.keys())}")
    return model_map[model_name]

def get_smp_unet_plus_plus_effnetb4(n_channels=3, n_classes=1, **kwargs):
    # ... (function is unchanged)
    return smp.UnetPlusPlus(encoder_name="efficientnet-b4", encoder_weights=None, in_channels=n_channels, classes=n_classes)

def select_checkpoint_interactive():
    # ... (function is unchanged)
    checkpoint_dir = 'checkpoints'
    if not os.path.isdir(checkpoint_dir): print(f"Error: Checkpoints directory '{checkpoint_dir}' not found."); return None
    available_experiments = sorted([d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))])
    if not available_experiments: print("No trained models found in the 'checkpoints' directory."); return None
    print("\nPlease select a model experiment:")
    for i, name in enumerate(available_experiments): print(f"  {i + 1}: {name}")
    try:
        exp_choice = int(input(f"Enter experiment number (1-{len(available_experiments)}): ")) - 1
        if not 0 <= exp_choice < len(available_experiments): raise ValueError
    except (ValueError, IndexError): print("Invalid selection."); return None
    selected_experiment_path = os.path.join(checkpoint_dir, available_experiments[exp_choice])
    checkpoint_files = sorted(glob.glob(os.path.join(selected_experiment_path, '*.pth')))
    if not checkpoint_files: print(f"Error: No .pth checkpoint files found in '{available_experiments[exp_choice]}'."); return None
    print("\nPlease select a specific checkpoint file to load:")
    for i, path in enumerate(checkpoint_files): print(f"  {i + 1}: {os.path.basename(path)}")
    try:
        ckpt_choice = int(input(f"Enter checkpoint number (1-{len(checkpoint_files)}): ")) - 1
        if not 0 <= ckpt_choice < len(checkpoint_files): raise ValueError
    except (ValueError, IndexError): print("Invalid selection."); return None
    return checkpoint_files[ckpt_choice]

def save_prediction_example(image, true_mask, pred_mask, prob_mask, dice, iou, idx, save_dir):
    # ... (function is unchanged)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    img_np = image.permute(1, 2, 0).cpu().numpy()
    true_mask_np = true_mask.squeeze().cpu().numpy()
    pred_mask_np = pred_mask.squeeze().cpu().numpy()
    prob_mask_np = prob_mask.squeeze().cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean; img_np = np.clip(img_np, 0, 1)
    axes[0].imshow(img_np); axes[0].set_title('Original Image'); axes[0].axis('off')
    axes[1].imshow(true_mask_np, cmap='gray'); axes[1].set_title('Ground Truth Mask'); axes[1].axis('off')
    axes[2].imshow(pred_mask_np, cmap='gray'); axes[2].set_title(f'Predicted Mask\nDice: {dice:.4f}'); axes[2].axis('off')
    im = axes[3].imshow(prob_mask_np, cmap='viridis', vmin=0, vmax=1); axes[3].set_title(f'Probability Map\nIoU: {iou:.4f}'); axes[3].axis('off')
    fig.colorbar(im, ax=axes[3]); plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'example_{idx}.png'), dpi=150); plt.close()

def log_and_plot_results(metrics_dict, cm_totals, save_dir):
    # ... (function is unchanged)
    print("\n" + "="*60 + "\nEVALUATION RESULTS\n" + "="*60)
    for name, values in metrics_dict.items(): print(f"\n{name.replace('_', ' ').capitalize()} Score:\n  Mean: {np.mean(values):.4f} ± {np.std(values):.4f}\n  Range: [{np.min(values):.4f}, {np.max(values):.4f}]")
    overall_precision = cm_totals['tp'] / (cm_totals['tp'] + cm_totals['fp']) if (cm_totals['tp'] + cm_totals['fp']) > 0 else 0
    overall_recall = cm_totals['tp'] / (cm_totals['tp'] + cm_totals['fn']) if (cm_totals['tp'] + cm_totals['fn']) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    print(f"\nOverall Dataset Metrics (from total counts):\n  Precision: {overall_precision:.4f}\n  Recall (Sensitivity): {overall_recall:.4f}\n  F1-Score: {overall_f1:.4f}")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10)); axes = axes.flatten(); sns.set_theme(style="whitegrid")
    for i, (name, values) in enumerate(metrics_dict.items()): sns.histplot(values, kde=True, ax=axes[i], bins=20); axes[i].set_title(f"Distribution of {name.replace('_', ' ').capitalize()}"); axes[i].set_xlabel("Score"); axes[i].set_ylabel("Frequency")
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'metrics_distribution.png'), dpi=150); plt.close()
    print(f"\nEvaluation complete. All results saved in '{save_dir}'")

# --- Classes are unchanged, but they now accept use_tta as a parameter ---
class ModelEvaluator:
    # ... (class is unchanged)
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f"Using device: {self.device}")
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'config' not in checkpoint: raise ValueError("Checkpoint is missing the 'config' dictionary.")
        config = checkpoint['config']; model_type = config.get('model_type')
        print(f"\nLoading model '{model_type}' with configuration from checkpoint...")
        get_model_func = get_model_from_name(model_type)
        sig = inspect.signature(get_model_func)
        model_args = {k: v for k, v in config.items() if k in sig.parameters}
        self.model = get_model_func(**model_args).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict']); self.model.eval()
        print(f"Successfully loaded model from epoch {checkpoint.get('epoch', 'N/A')}.")
        if 'iou_score' in checkpoint: print(f"Checkpoint achieved best validation IoU: {checkpoint['iou_score']:.4f}")

    def evaluate_dataset(self, dataloader, num_examples=5, use_crf=False, use_tta=False):
        all_metrics = {'dice': [], 'iou': [], 'pixel_acc': [], 'precision': [], 'recall': [], 'f1': []}
        cm_totals = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        model_name_for_dir = os.path.basename(os.path.dirname(model_path))
        eval_dir_suffix = f"{'TTA_' if use_tta else ''}{'CRF' if use_crf else 'Base'}"
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
                if hasattr(outputs, 'logits'): outputs = F.interpolate(outputs.logits, size=images_gpu.shape[2:], mode='bilinear', align_corners=False)
                elif isinstance(outputs, list): outputs = outputs[-1]
                probs = torch.sigmoid(outputs).cpu()
                for j in range(images.size(0)):
                    prob_j = probs[j].squeeze(); mask_j_gpu = masks_gpu[j]
                    if use_crf:
                        original_image_np = images[j].permute(1, 2, 0).numpy(); prob_np = prob_j.numpy()
                        pred_np = apply_crf(original_image_np, prob_np); pred_j = torch.from_numpy(pred_np).float().to(self.device)
                    else:
                        pred_j = (prob_j.to(self.device) > 0.5).float()
                    all_metrics['dice'].append(dice_score(pred_j, mask_j_gpu).item()); all_metrics['iou'].append(iou_score(pred_j, mask_j_gpu).item()); all_metrics['pixel_acc'].append(pixel_accuracy(pred_j, mask_j_gpu).item())
                    precision, recall, f1 = precision_recall_f1(pred_j, mask_j_gpu); all_metrics['precision'].append(precision.item()); all_metrics['recall'].append(recall.item()); all_metrics['f1'].append(f1.item())
                    cm = confusion_matrix_metrics(pred_j, mask_j_gpu); cm_totals['tp'] += cm['TP']; cm_totals['tn'] += cm['TN']; cm_totals['fp'] += cm['FP']; cm_totals['fn'] += cm['FN']
                    if i * dataloader.batch_size + j < num_examples: save_prediction_example(images[j], masks[j], pred_j.cpu(), prob_j, all_metrics['dice'][-1], all_metrics['iou'][-1], i * dataloader.batch_size + j, eval_dir)
        log_and_plot_results(all_metrics, cm_totals, eval_dir)

class EnsembleEvaluator:
    # ... (class is unchanged)
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
        all_metrics = {'dice': [], 'iou': [], 'pixel_acc': [], 'precision': [], 'recall': [], 'f1': []}
        cm_totals = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        model_a_name = self.config_a.get('model_type'); model_b_name = self.config_b.get('model_type')
        eval_dir_suffix = f"{'TTA_' if use_tta else ''}{'CRF' if use_crf else 'Base'}"
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
                elif isinstance(outputs_a, list): outputs_a = outputs_a[-1]
                probs_a = torch.sigmoid(outputs_a)
                outputs_b = model_b_eval(images_gpu)
                if hasattr(outputs_b, 'logits'): outputs_b = F.interpolate(outputs_b.logits, size=images_gpu.shape[2:], mode='bilinear', align_corners=False)
                elif isinstance(outputs_b, list): outputs_b = outputs_b[-1]
                probs_b = torch.sigmoid(outputs_b)
                ensemble_probs = (probs_a + probs_b) / 2.0; ensemble_probs = ensemble_probs.cpu()
                for j in range(images.size(0)):
                    prob_j = ensemble_probs[j].squeeze(); mask_j_gpu = masks_gpu[j]
                    if use_crf:
                        original_image_np = images[j].permute(1, 2, 0).numpy(); prob_np = prob_j.numpy()
                        pred_np = apply_crf(original_image_np, prob_np); pred_j = torch.from_numpy(pred_np).float().to(self.device)
                    else:
                        pred_j = (prob_j.to(self.device) > 0.5).float()
                    all_metrics['dice'].append(dice_score(pred_j, mask_j_gpu).item()); all_metrics['iou'].append(iou_score(pred_j, mask_j_gpu).item()); all_metrics['pixel_acc'].append(pixel_accuracy(pred_j, mask_j_gpu).item())
                    precision, recall, f1 = precision_recall_f1(pred_j, mask_j_gpu); all_metrics['precision'].append(precision.item()); all_metrics['recall'].append(recall.item()); all_metrics['f1'].append(f1.item())
                    cm = confusion_matrix_metrics(pred_j, mask_j_gpu); cm_totals['tp'] += cm['TP']; cm_totals['tn'] += cm['TN']; cm_totals['fp'] += cm['FP']; cm_totals['fn'] += cm['FN']
                    if i * dataloader.batch_size + j < num_examples: save_prediction_example(images[j], masks[j], pred_j.cpu(), prob_j, all_metrics['dice'][-1], all_metrics['iou'][-1], i * dataloader.batch_size + j, eval_dir)
        log_and_plot_results(all_metrics, cm_totals, eval_dir)

# --- NEW: main function with fully interactive controls ---
def main():
    global model_path
    parser = argparse.ArgumentParser(description='Unified Segmentation Model Evaluation')
    parser.add_argument('--path', type=str, default=None, help='(Optional) Direct path to a model checkpoint for non-interactive scripting.')
    parser.add_argument('--examples', type=int, default=10, help='Number of visual examples to save.')
    args = parser.parse_args()

    # Non-interactive mode for scripting
    if args.path:
        model_path = args.path
        if not os.path.exists(model_path): print(f"Error: Model path does not exist: {model_path}"); return
        try:
            # In non-interactive mode, TTA and CRF are off by default.
            # Could add more flags like --script_use_tta if needed.
            evaluator = ModelEvaluator(model_path=model_path)
            evaluator.evaluate_dataset(test_loader, num_examples=args.examples, use_crf=False, use_tta=False)
        except Exception as e:
            print(f"\nAn error occurred: {e}"); traceback.print_exc()
        return

    # --- Interactive Menu ---
    print("\n--- Evaluation Menu ---")
    print("1. Evaluate a Single Model")
    print("2. Evaluate an Ensemble of Two Models")
    
    try:
        choice = input("Enter your choice (1-2): ")
        model_paths, is_ensemble = [], False
        
        if choice == '1':
            path = select_checkpoint_interactive()
            if path: model_paths.append(path)
        elif choice == '2':
            is_ensemble = True
            print("\n--- Select Model A for the Ensemble ---"); path_a = select_checkpoint_interactive()
            if path_a: model_paths.append(path_a)
            else: return
            print("\n--- Select Model B for the Ensemble ---"); path_b = select_checkpoint_interactive()
            if path_b: model_paths.append(path_b)
            else: return
        else:
            print("Invalid choice."); return

        if not model_paths: print("No model was selected for evaluation."); return
        
        # --- NEW: Interactive prompts for TTA and CRF ---
        use_tta_input = input("\nEnable Test-Time Augmentation (TTA)? [y/n]: ").lower()
        use_tta = True if use_tta_input == 'y' else False
        use_crf_input = input("Enable CRF Post-Processing? [y/n]: ").lower()
        use_crf = True if use_crf_input == 'y' else False

        if is_ensemble:
            if model_paths[0] == model_paths[1]: print("Warning: You selected the same model twice.")
            ensemble_evaluator = EnsembleEvaluator(model_paths[0], model_paths[1])
            ensemble_evaluator.evaluate_dataset(test_loader, num_examples=args.examples, use_crf=use_crf, use_tta=use_tta)
        else:
            model_path = model_paths[0]
            evaluator = ModelEvaluator(model_path=model_path)
            evaluator.evaluate_dataset(test_loader, num_examples=args.examples, use_crf=use_crf, use_tta=use_tta)

    except Exception as e:
        print(f"\nAn error occurred: {e}"); traceback.print_exc()

if __name__ == '__main__':
    main()