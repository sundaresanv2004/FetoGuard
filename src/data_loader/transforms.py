import torch
import torchvision.transforms.functional as TF
import random
from typing import Tuple
from PIL import Image

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

class Resize:
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = TF.resize(image, self.size, interpolation=Image.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=Image.NEAREST)
        return image, mask

class ToTensor:
    def __call__(self, image: Image.Image, mask: Image.Image):
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

class Normalize:
    def __init__(self, mean: Tuple[float, ...], std: Tuple[float, ...]):
        self.mean = mean
        self.std = std
        
    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        image = TF.normalize(image, self.mean, self.std)
        # Masks are typically not normalized in the same way (they stay 0-1)
        return image, mask

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image: Image.Image, mask: Image.Image):
        if random.random() < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask

class RandomRotation:
    def __init__(self, degrees=10):
        self.degrees = degrees

    def __call__(self, image: Image.Image, mask: Image.Image):
        angle = random.uniform(-self.degrees, self.degrees)
        image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)
        return image, mask

def get_transforms(mode: str = 'train', img_size=(256, 256)):
    """
    Returns the transform pipeline for a given mode.
    
    Args:
        mode (str): 'train' or 'val'/'test'.
        img_size (tuple): Target resize dimension.
    """
    transforms = []
    
    # 1. Resize
    transforms.append(Resize(img_size))
    
    # 2. Augmentations (Train only)
    if mode == 'train':
        transforms.append(RandomHorizontalFlip(p=0.5))
        transforms.append(RandomRotation(degrees=15))
        
    # 3. ToTensor
    transforms.append(ToTensor())
    
    # 4. Normalize (Using approx ImageNet stats or 0.5/0.5, let's use 0.5 for now for simplicity in medical usually)
    # Using 0.5 mean and std maps [0,1] to [-1,1] which is standard for many GANs/UNets
    transforms.append(Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    
    return Compose(transforms)
