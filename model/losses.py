# model/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Calculates the Dice Loss. This part is unchanged.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        total = probs.sum() + targets.sum()
        dice = (2. * intersection + self.smooth) / (total + self.smooth)
        return 1 - dice

## ADDED: The new Focal Loss implementation.
class FocalLoss(nn.Module):
    """
    Implements Focal Loss, a powerful alternative to BCE Loss for handling class imbalance.
    It focuses training on hard-to-classify examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Calculate probabilities pt for the ground truth class
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        
        # Calculate the modulating factor (1 - pt)^gamma
        modulating_factor = (1.0 - pt).pow(self.gamma)
        
        # Calculate the alpha-weighting factor
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # The final focal loss
        focal_loss = alpha_weight * modulating_factor * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

## CHANGED: CombinedLoss is now more flexible.
class CombinedLoss(nn.Module):
    """
    A combined loss function that can use either BCE or Focal Loss along with Dice Loss.
    """
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

    def forward(self, logits, targets):
        base = self.base_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        
        # Return the weighted sum of the two losses
        return self.base_loss_weight * base + self.dice_weight * dice