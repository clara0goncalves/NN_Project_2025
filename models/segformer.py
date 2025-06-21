# models/segformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

class SegformerFinetuner(nn.Module):
    """
    A wrapper class to make the Hugging Face SegformerForSemanticSegmentation model
    compatible with our existing training pipeline.
    """
    def __init__(self, n_classes=1, pretrained_model_name="nvidia/segformer-b4-finetuned-ade-512-512"):
        super().__init__()
        
        self.n_classes = n_classes
        
        # Load the pre-trained SegFormer model.
        # We replace the final classification head with a new one adapted to our number of classes.
        # `ignore_mismatched_sizes=True` is crucial for this transfer learning step.
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name,
            num_labels=self.n_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        # The base model returns a dictionary-like object. We need the 'logits'.
        outputs = self.segformer(pixel_values=x)
        logits = outputs.logits # Raw, unnormalized scores

        # SegFormer's output logits are smaller than the input image (e.g., H/4, W/4).
        # We must upsample them to match the ground truth mask's size for loss calculation.
        # We use bilinear interpolation to do this.
        upsampled_logits = F.interpolate(
            logits,
            size=x.shape[2:],  # Upsample to the original H, W of the input tensor `x`
            mode='bilinear',
            align_corners=False
        )
        
        return upsampled_logits

def get_segformer_model(n_classes=1, **kwargs):
    """
    Factory function to create our SegFormer finetuning model.
    """
    # Using 'nvidia/segformer-b4-finetuned-ade-512-512' is a great, powerful choice.
    # It was pre-trained on the ADE20K dataset.
    model = SegformerFinetuner(n_classes=n_classes)
    return model