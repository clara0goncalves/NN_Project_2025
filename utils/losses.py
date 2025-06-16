# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Calculates the Dice Loss, which is a common metric for evaluating
    segmentation models. It is calculated as 1 - Dice Score.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: The raw, unnormalized output from the model (before sigmoid).
            targets: The ground truth masks.
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten both the probabilities and the targets
        probs = probs.view(-1)
        targets = targets.view(-1)

        # Calculate the intersection and the total number of pixels
        intersection = (probs * targets).sum()
        total = probs.sum() + targets.sum()

        # Calculate the Dice coefficient
        dice = (2. * intersection + self.smooth) / (total + self.smooth)

        # Return the Dice Loss
        return 1 - dice

class CombinedLoss(nn.Module):
    """
    A combined loss function that is a weighted sum of Binary Cross-Entropy (BCE)
    and Dice Loss. This often leads to better performance in segmentation tasks.
    """
    def __init__(self, alpha=0.5, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth)

    def forward(self, logits, targets):
        """
        Args:
            logits: The raw, unnormalized output from the model.
            targets: The ground truth masks.
        """
        # Calculate BCE and Dice losses
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        # Return the weighted sum of the two losses
        return self.alpha * bce + (1 - self.alpha) * dice