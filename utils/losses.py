# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from segmentation_models_pytorch.losses import LovaszLoss

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        total = pred.sum() + target.sum()
        
        dice = (2. * intersection + self.smooth) / (total + self.smooth)
        return 1 - dice

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    def __init__(self, alpha=1, gamma=2, smooth=1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred, target):
        # NOTE: The 'pred' input is the raw logits from the model, NOT the sigmoid output
        
        # We use the numerically stable 'binary_cross_entropy_with_logits'
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # To get pt, we still need the probabilities, so we apply sigmoid here just for that
        pred_prob = torch.sigmoid(pred)
        pt = torch.where(target == 1, pred_prob, 1 - pred_prob)
        
        # Calculate focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        
        return focal_loss.mean()

class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice Loss
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky

class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice Loss
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

class IoULoss(nn.Module):
    """
    IoU Loss for binary segmentation
    """
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou
    
# --- NEW: Combined Focal + Lovasz Loss ---
class FocalLovaszLoss(nn.Module):
    def __init__(self, focal_weight=0.5, lovasz_weight=0.5, focal_alpha=1, focal_gamma=2):
        super(FocalLovaszLoss, self).__init__()
        self.focal_weight = focal_weight
        self.lovasz_weight = lovasz_weight
        
        # Instantiate the component losses with their respective parameters
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.lovasz = LovaszLoss(mode='binary')

    def forward(self, pred, target):
        # Calculate the individual losses
        focal_loss = self.focal(pred, target)
        lovasz_loss = self.lovasz(pred, target)
        
        # Calculate the weighted sum
        combined_loss = (self.focal_weight * focal_loss) + (self.lovasz_weight * lovasz_loss)
        
        return combined_loss
