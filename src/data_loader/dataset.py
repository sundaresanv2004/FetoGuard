import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from typing import Optional, Callable, Tuple, List


class FetalHeadDataset(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 files: List[str], 
                 transform: Optional[Callable] = None):
        """
        Args:
            root_dir (str): Directory with all the images.
            files (List[str]): List of filenames for this split (images only, not annotations).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # Construct annotation path based on image name
        # Convention: XXX_HC.png -> XXX_HC_Annotation.png
        # Some files might have different conventions (e.g., 2HC), assuming consistent pair exists
        base_name = os.path.splitext(img_name)[0]
        annotation_name = f"{base_name}_Annotation.png"
        annotation_path = os.path.join(self.root_dir, annotation_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(annotation_path).convert("L")  # Grayscale for mask

        if self.transform:
            # We need to apply the SAME random transform to both image and mask
            # This is usually handled by functional transforms or libraries like albumentations
            # For simplicity with torchvision, we handle the seed locally or pass PIL images if transform supports it
            # However, standard torchvision transforms don't support joint transform easily.
            # We will handle this in the transform callable or dataset logic.
            # Ideally the transform callable takes (image, mask) or we use a library that supports it.
            # For now, let's assume the transform handles (image, mask) tuple or we adapt.
            
            # NOTE: Standard torchvision transforms like Resize/ToTensor work on single images.
            # We will implement a custom interface in transforms.py.
            image, mask = self.transform(image, mask)

        return image, mask


def get_dataloaders(data_dir: str, 
                    batch_size: int = 8, 
                    train_transform: Optional[Callable] = None,
                    val_transform: Optional[Callable] = None,
                    test_transform: Optional[Callable] = None,
                    val_split: float = 0.2,
                    num_workers: int = 4,
                    seed: int = 42):
    """
    Creates DataLoaders for train, validation, and test sets.
    
    Args:
        data_dir (str): Path to 'data/dataset' containing 'training_set' and 'test_set'.
        batch_size (int): Batch size.
        train_transform: Transform for training set.
        val_transform: Transform for validation set.
        test_transform: Transform for test set.
        val_split (float): Fraction of training set to use for validation.
        num_workers (int): Number of subprocesses for data loading.
        seed (int): Random seed for splitting.
        
    Returns:
        dict: {'train': loader, 'val': loader, 'test': loader}
    """
    
    train_dir = os.path.join(data_dir, 'training_set')
    test_dir = os.path.join(data_dir, 'test_set')
    
    # helper to filter only image files (exclude annotations)
    def is_image_file(filename):
        return filename.endswith('_HC.png') or \
               filename.endswith('_2HC.png') or \
               filename.endswith('_3HC.png') or \
               filename.endswith('_4HC.png') 
               
    # Get all potential image files
    all_train_files = [f for f in os.listdir(train_dir) if f.lower().endswith('.png') and 'Annotation' not in f]
    all_test_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.png') and 'Annotation' not in f]
    
    # Split training set
    train_files, val_files = train_test_split(all_train_files, test_size=val_split, random_state=seed)
    
    # Create Datasets
    train_dataset = FetalHeadDataset(train_dir, train_files, transform=train_transform)
    val_dataset = FetalHeadDataset(train_dir, val_files, transform=val_transform)
    test_dataset = FetalHeadDataset(test_dir, all_test_files, transform=test_transform)
    
    # Create DataLoaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    
    return dataloaders
