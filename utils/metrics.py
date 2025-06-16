# utils/metrics.py
import torch
import numpy as np

def dice_score(preds, targets, smooth=1e-6):
    """Calculates the Dice Score."""
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    total = preds.sum() + targets.sum()
    dice = (2. * intersection + smooth) / (total + smooth)
    return dice

def iou_score(preds, targets, smooth=1e-6):
    """Calculates the Intersection over Union (IoU) score."""
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def pixel_accuracy(preds, targets):
    """Calculates pixel accuracy."""
    correct = (preds == targets).sum()
    total = targets.numel()
    accuracy = correct.float() / total
    return accuracy

def precision_recall_f1(preds, targets, smooth=1e-6):
    """Calculates precision, recall, and F1 score."""
    preds = preds.view(-1)
    targets = targets.view(-1)

    tp = (preds * targets).sum()
    fp = ((1 - targets) * preds).sum()
    fn = (targets * (1 - preds)).sum()

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = 2 * (precision * recall) / (precision + recall + smooth)

    return precision, recall, f1

def confusion_matrix_metrics(preds, targets):
    """Calculates TP, TN, FP, FN."""
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    tp = (preds * targets).sum().item()
    fp = ((1 - targets) * preds).sum().item()
    fn = (targets * (1 - preds)).sum().item()
    tn = ((1 - targets) * (1 - preds)).sum().item()
    
    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}