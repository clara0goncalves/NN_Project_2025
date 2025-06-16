# model/eval.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import sys

# Corrected: Add path modification to find utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .unet import get_model # Corrected: Relative import
from utils.metrics import (dice_score, iou_score, pixel_accuracy, 
                          precision_recall_f1, confusion_matrix_metrics)
from utils.prepare_data import test_loader

class ModelEvaluator:
    def __init__(self, model_path, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = get_model(
            n_channels=config['n_channels'],
            n_classes=config['n_classes']
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")
        print(f"Best validation Dice: {checkpoint.get('dice_score', 0.0):.4f}")

        # Create evaluation results directory
        self.results_dir = "evaluation_results"
        os.makedirs(self.results_dir, exist_ok=True)

    def evaluate_dataset(self, dataloader, num_examples=5):
        """Evaluate model and compute comprehensive metrics."""
        all_metrics = {
            'dice': [], 'iou': [], 'pixel_acc': [], 
            'precision': [], 'recall': [], 'f1': []
        }
        all_cm = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        
        print("Evaluating model...")
        with torch.no_grad():
            for i, (images, masks) in enumerate(tqdm(dataloader)):
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                for j in range(images.size(0)):
                    pred_i, mask_i = preds[j], masks[j]
                    
                    all_metrics['dice'].append(dice_score(pred_i, mask_i).item())
                    all_metrics['iou'].append(iou_score(pred_i, mask_i).item())
                    all_metrics['pixel_acc'].append(pixel_accuracy(pred_i, mask_i).item())
                    
                    precision, recall, f1 = precision_recall_f1(pred_i, mask_i)
                    all_metrics['precision'].append(precision.item())
                    all_metrics['recall'].append(recall.item())
                    all_metrics['f1'].append(f1.item())
                    
                    cm = confusion_matrix_metrics(pred_i, mask_i)
                    for key in all_cm: all_cm[key] += cm[key]
                    
                    if i * dataloader.batch_size + j < num_examples:
                        self.save_prediction_example(
                            images[j], masks[j], pred_i, probs[j],
                            all_metrics['dice'][-1], i * dataloader.batch_size + j
                        )
        
        self.log_results(all_metrics, all_cm)

    def save_prediction_example(self, image, mask, pred, prob, dice, idx):
        """Saves a visual example of a prediction."""
        image = image.permute(1, 2, 0).cpu().numpy()
        mask = mask.squeeze().cpu().numpy()
        pred = pred.squeeze().cpu().numpy()
        prob = prob.squeeze().cpu().numpy()

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
        ax1.imshow(image)
        ax1.set_title("Input Image")
        ax1.axis('off')

        ax2.imshow(mask, cmap='gray')
        ax2.set_title("Ground Truth Mask")
        ax2.axis('off')

        ax3.imshow(pred, cmap='gray')
        ax3.set_title(f"Prediction (Dice: {dice:.3f})")
        ax3.axis('off')

        ax4.imshow(prob, cmap='viridis')
        ax4.set_title("Probability Map")
        ax4.axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, f"prediction_example_{idx}.png")
        plt.savefig(save_path)
        plt.close(fig)

    def log_results(self, metrics_dict, cm_dict):
        """Prints and logs the final evaluation metrics."""
        print("\n--- Evaluation Results ---")
        for name, values in metrics_dict.items():
            mean, std = np.mean(values), np.std(values)
            print(f"{name.replace('_', ' ').title():<15}: Mean={mean:.4f}, Std={std:.4f}")
        
        print("\n--- Overall Confusion Matrix ---")
        print(f"True Positives:  {cm_dict['TP']}")
        print(f"True Negatives:  {cm_dict['TN']}")
        print(f"False Positives: {cm_dict['FP']}")
        print(f"False Negatives: {cm_dict['FN']}")
        
        # Plot confusion matrix
        cm_array = np.array([[cm_dict['TN'], cm_dict['FP']], [cm_dict['FN'], cm_dict['TP']]])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_array, annot=True, fmt='.0f', cmap='Blues', 
                    xticklabels=['Not Water', 'Water'], yticklabels=['Not Water', 'Water'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Overall Confusion Matrix")
        save_path = os.path.join(self.results_dir, "confusion_matrix.png")
        plt.savefig(save_path)
        plt.close()
        print(f"\nConfusion matrix saved to {save_path}")