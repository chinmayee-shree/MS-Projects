
import numpy as np


def calculate_iou(mask1, mask2):
    """
    Calculate Intersection over Union (IoU).
    """

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    return intersection / union if union != 0 else 0


def calculate_dice(mask1, mask2):
    """
    Calculate Dice Similarity Coefficient.
    """

    intersection = np.logical_and(mask1, mask2).sum()

    denominator = mask1.sum() + mask2.sum()

    return (
        (2 * intersection) / denominator
        if denominator != 0 else 0
    )
