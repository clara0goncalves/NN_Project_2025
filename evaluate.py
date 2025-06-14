# evaluate.py
"""

Extensive evaluation metrics (Dice, IoU, Precision, Recall, F1)
Confusion matrix analysis
Visualization of predictions
Statistical analysis with confidence intervals

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

from models.unet import get_model
from utils.metrics import (dice_score, iou_score, pixel_accuracy, 
                          precision_recall_f1, confusion_matrix_metrics)
from prepare_data import test_loader


class ModelEvaluator:
    def __init__(self, model_path, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = get_model(
            n_channels=config['n_channels'],
            n_classes=config['n_classes'],
            bilinear=config['bilinear']
        ).to(self.device)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from epoch {checkpoint['epoch']}")
        print(f"Best validation Dice: {checkpoint['dice_score']:.4f}")

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