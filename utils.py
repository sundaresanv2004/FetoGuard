import torch
import torch.nn.functional as F


def get_device():
    """Selects the best available device (CUDA, MPS, or CPU) and prints it."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    return device


def dice_score(preds, targets, num_classes, smooth=1e-6):
    """
    Calculates the Dice score for multi-class segmentation.

    Args:
        preds (torch.Tensor): Predicted class indices. Shape (N, H, W).
        targets (torch.Tensor): Ground truth class indices. Shape (N, H, W).
        num_classes (int): The total number of classes.
        smooth (float): A small value to prevent division by zero.

    Returns:
        float: The average Dice score, excluding the background class.
    """
    # Convert predictions and targets to one-hot encoding
    # Shape: (N, C, H, W)
    preds_one_hot = F.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2)
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2)

    # Calculate intersection and union for each class
    # The dimensions to sum over are the spatial ones (height and width)
    intersection = torch.sum(preds_one_hot * targets_one_hot, dim=(2, 3))
    union = torch.sum(preds_one_hot, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))

    # Calculate Dice coefficient for each class
    # Shape: (N, C)
    dice_coeffs = (2. * intersection + smooth) / (union + smooth)

    # We typically ignore the background class (class 0) in evaluation
    # Average over the batch (dim=0) and then over the foreground classes (dim=1)
    # [:, 1:] selects all classes from index 1 onwards
    avg_dice = torch.mean(dice_coeffs[:, 1:])

    return avg_dice.item()

