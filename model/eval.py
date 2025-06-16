# model/eval.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .unet import get_model
from utils.metrics import dice_score, iou_score, pixel_accuracy, precision_recall_f1, confusion_matrix_metrics

class ModelEvaluator:
    def __init__(self, model_path, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = get_model(
            n_channels=config.get('n_channels', 3),
            n_classes=config.get('n_classes', 1),
            base_filters=config.get('base_filters', 64),
            norm_type=config.get('norm_type', 'batch')
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'N/A')} with validation Dice: {checkpoint.get('dice_score', 0.0):.4f}")

        exp_name = os.path.basename(os.path.dirname(model_path))
        self.results_dir = os.path.join("evaluation_results", exp_name)
        os.makedirs(self.results_dir, exist_ok=True)

    def evaluate_dataset(self, dataloader, num_examples=5):
        all_metrics = {'dice':[], 'iou':[], 'pixel_acc':[], 'precision':[], 'recall':[], 'f1':[]}
        all_cm = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        
        print("Evaluating model on the test set...")
        with torch.no_grad():
            for i, (images, masks) in enumerate(tqdm(dataloader, desc="Evaluating")):
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                
                for j in range(images.size(0)):
                    pred_i, mask_i = preds[j], masks[j]
                    
                    all_metrics['dice'].append(dice_score(pred_i, mask_i).item())
                    all_metrics['iou'].append(iou_score(pred_i, mask_i).item())
                    all_metrics['pixel_acc'].append(pixel_accuracy(pred_i, mask_i).item())
                    precision, recall, f1 = precision_recall_f1(pred_i, mask_i)
                    all_metrics['precision'].append(precision.item()); all_metrics['recall'].append(recall.item()); all_metrics['f1'].append(f1.item())
                    
                    cm = confusion_matrix_metrics(pred_i, mask_i)
                    for key in all_cm: all_cm[key] += cm[key]
                    
                    if i * dataloader.batch_size + j < num_examples:
                        self.save_prediction_example(images[j], masks[j], pred_i, i * dataloader.batch_size + j)
        
        self.log_results(all_metrics, all_cm)

    def save_prediction_example(self, image_tensor, mask_tensor, pred_tensor, idx):
        # Denormalize image from PyTorch tensor format for correct visualization
        image = image_tensor.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

        mask = mask_tensor.squeeze().cpu().numpy()
        pred = pred_tensor.squeeze().cpu().numpy()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        ax1.imshow(image)
        ax1.set_title("Input Image")
        ax1.axis('off')
        
        ax2.imshow(mask, cmap='gray')
        ax2.set_title("Ground Truth Mask")
        ax2.axis('off')
        
        ax3.imshow(pred, cmap='gray')
        ax3.set_title("Model Prediction")
        ax3.axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.results_dir, f"prediction_example_{idx}.png")
        plt.savefig(save_path)
        plt.close(fig)

    def log_results(self, metrics_dict, cm_dict):
        print("\n" + "="*30)
        print("--- Final Evaluation Metrics ---")
        print("="*30)
        for name, values in metrics_dict.items():
            mean_val, std_val = np.mean(values), np.std(values)
            print(f"{name.replace('_', ' ').title():<15}: Mean={mean_val:.4f}, Std={std_val:.4f}")
        
        print("\n--- Overall Confusion Matrix ---")
        for key, value in cm_dict.items():
            print(f"{key:<15}: {int(value)}")

        cm_array = np.array([[cm_dict['TN'], cm_dict['FP']], [cm_dict['FN'], cm_dict['TP']]])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_array, annot=True, fmt='.0f', cmap='Blues',
                    xticklabels=['Not Water', 'Water'], yticklabels=['Not Water', 'Water'])
        plt.xlabel("Predicted Label"); plt.ylabel("True Label"); plt.title("Confusion Matrix")
        save_path = os.path.join(self.results_dir, "confusion_matrix.png")
        plt.savefig(save_path)
        plt.close()
        print(f"\nConfusion matrix plot saved to '{save_path}'")