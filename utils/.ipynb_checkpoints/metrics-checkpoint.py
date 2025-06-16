# utils/metrics.py
"""
Dice Loss, Focal Loss, Tversky Loss
Combined loss functions
Comprehensive evaluation metrics
Confusion matrix utilities

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def dice_score(pred, target, smooth=1e-6):
    """
    Calculate Dice Score (F1 Score for binary segmentation)
    
    Args:
        pred: Predicted tensor (0 or 1)
        target: Ground truth tensor (0 or 1)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice score
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    total = pred.sum() + target.sum()
    
    dice = (2. * intersection + smooth) / (total + smooth)
    return dice

def iou_score(pred, target, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU)
    
    Args:
        pred: Predicted tensor (0 or 1)
        target: Ground truth tensor (0 or 1)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        IoU score
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou

def pixel_accuracy(pred, target):
    """
    Calculate pixel accuracy
    
    Args:
        pred: Predicted tensor (0 or 1)
        target: Ground truth tensor (0 or 1)
    
    Returns:
        Pixel accuracy
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    correct = (pred == target).sum().float()
    total = target.numel()
    
    return correct / total

def precision_recall_f1(pred, target, smooth=1e-6):
    """
    Calculate precision, recall, and F1 score
    
    Args:
        pred: Predicted tensor (0 or 1)
        target: Ground truth tensor (0 or 1)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Tuple of (precision, recall, f1)
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    tp = (pred * target).sum().float()
    fp = (pred * (1 - target)).sum().float()
    fn = ((1 - pred) * target).sum().float()
    
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    
    return precision, recall, f1

def confusion_matrix_metrics(pred, target):
    """
    Calculate confusion matrix based metrics
    
    Args:
        pred: Predicted tensor (0 or 1)
        target: Ground truth tensor (0 or 1)
    
    Returns:
        Dictionary with TP, TN, FP, FN, Sensitivity, Specificity
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    tp = (pred * target).sum().float()
    tn = ((1 - pred) * (1 - target)).sum().float()
    fp = (pred * (1 - target)).sum().float()
    fn = ((1 - pred) * target).sum().float()
    
    sensitivity = tp / (tp + fn + 1e-6)  # Recall/True Positive Rate
    specificity = tn / (tn + fp + 1e-6)  # True Negative Rate
    
    return {
        'TP': tp.item(),
        'TN': tn.item(),
        'FP': fp.item(),
        'FN': fn.item(),
        'Sensitivity': sensitivity.item(),
        'Specificity': specificity.item()
    }

