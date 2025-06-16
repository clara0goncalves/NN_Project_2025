# model/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets, ignore_mask): # CHANGED: Added ignore_mask
        # --- CHANGED: Apply the ignore mask ---
        logits = logits.view(-1)[ignore_mask.view(-1)]
        targets = targets.view(-1)[ignore_mask.view(-1)]

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        total = probs.sum() + targets.sum()
        dice = (2. * intersection + self.smooth) / (total + self.smooth)
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, ignore_mask): # CHANGED: Added ignore_mask
        # --- CHANGED: Apply the ignore mask ---
        logits = logits.view(-1)[ignore_mask.view(-1)]
        targets = targets.view(-1)[ignore_mask.view(-1)]
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, base_loss_type='bce', focal_alpha=0.25, focal_gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.base_loss_weight = 1 - dice_weight
        self.dice_loss = DiceLoss()

        if base_loss_type == 'bce':
            self.base_loss = nn.BCEWithLogitsLoss()
        elif base_loss_type == 'focal':
            self.base_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            raise ValueError(f"Unknown base_loss_type: {base_loss_type}")

    def forward(self, logits, targets, ignore_mask): # CHANGED: Added ignore_mask
        base_component_loss = self.base_loss(logits, targets, ignore_mask)
        dice_component_loss = self.dice_loss(logits, targets, ignore_mask)
        return self.base_loss_weight * base_component_loss + self.dice_weight * dice_component_loss