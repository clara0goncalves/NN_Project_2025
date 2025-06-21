# utils/postprocessing.py
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
import cv2

def apply_crf(image, prob_map):
    """
    Applies Conditional Random Field (CRF) post-processing to a probability map,
    inspired by the working example provided.

    Args:
        image (np.ndarray): The original input image (H, W, C), expected to be uint8 [0, 255].
        prob_map (np.ndarray): The model's probability map (H, W).

    Returns:
        np.ndarray: The refined segmentation mask.
    """
    # Ensure image is in the correct format (uint8)
    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    
    # Ensure image is 3-channel RGB. This is still good practice.
    if len(image.shape) < 3 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    h, w = image.shape[:2]

    # 1. Create unary potentials from the model's prediction
    # The user's method is clean and effective.
    unary = np.stack([1 - prob_map, prob_map], axis=0)
    unary = unary.reshape((2, -1))  # Reshape to (n_classes, H*W)
    unary = unary_from_softmax(unary)

    # 2. Create the DenseCRF object
    d = dcrf.DenseCRF2D(w, h, 2)  # (width, height, n_classes)
    d.setUnaryEnergy(unary)

    # 3. Add a pairwise bilateral term
    # This considers both color and position, using the corrected `chdim` argument.
    pairwise_bilateral = create_pairwise_bilateral(
        sdims=(10, 10),
        schan=(13, 13, 13),
        img=image,
        chdim=2  # The key fix from your example
    )
    d.addPairwiseEnergy(pairwise_bilateral, compat=3)

    # 4. Add a pairwise Gaussian term
    # This acts as a smoothing filter, considering only position.
    pairwise_gaussian = create_pairwise_gaussian(
        sdims=(3, 3),
        shape=(w, h)
    )
    d.addPairwiseEnergy(pairwise_gaussian, compat=1)

    # 5. Run CRF inference
    Q = d.inference(5)  # Run 5 inference steps

    # 6. Get the final refined mask
    refined_mask = np.argmax(Q, axis=0).reshape((h, w))

    return refined_mask