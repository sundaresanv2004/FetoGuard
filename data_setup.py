import torch
import os
from xml.etree import ElementTree as ET
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from torch.utils.data import Subset, random_split, DataLoader


class OD_US_Dataset_Segmentation(torch.utils.data.Dataset):
    """Custom Dataset for fetal ultrasound image segmentation."""

    def __init__(self, anno_path, img_path, transforms=None):
        self.classes = ['bkg', 'Head', 'Abdomen', 'Arms', 'Legs']
        self.img_path = img_path
        self.transforms = transforms
        self.X = []
        self.Y = []

        self.label_map = {'head': 1, 'abdomen': 2, 'arm': 3, 'legs': 4}

        for obj in os.listdir(anno_path):
            if obj == '.DS_Store': continue

            # <-- FIX: Changed the filename to 'box_annotations.xml'
            xml_file_path = os.path.join(anno_path, obj, 'annotations.xml')

            if not os.path.exists(xml_file_path):
                # This check is useful if some folders in 'boxes' don't have annotations
                print(f"Warning: Annotation file not found at {xml_file_path}, skipping.")
                continue

            dom = ET.parse(xml_file_path)
            for n in dom.findall('image'):
                bbox, labels = [], []
                name = n.attrib.get('name')

                for l in n.findall('box'):
                    label_str = l.attrib.get('label')
                    if label_str:
                        label_str = label_str.lower()

                    if label_str in self.label_map:
                        labels.append(self.label_map[label_str])

                        xtl = float(l.attrib.get('xtl'))
                        ytl = float(l.attrib.get('ytl'))
                        xbr = float(l.attrib.get('xbr'))
                        ybr = float(l.attrib.get('ybr'))
                        bbox.append([xtl, ytl, xbr, ybr])

                self.X.append(os.path.join(self.img_path, obj, name))
                self.Y.append({'boxes': bbox, 'labels': labels})

    def __getitem__(self, idx):
        img_name = self.X[idx]
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ht, wt = img.shape[:2]
        mask = np.zeros((ht, wt), dtype=np.int64)

        for box, label in zip(self.Y[idx]['boxes'], self.Y[idx]['labels']):
            xtl, ytl, xbr, ybr = map(int, box)
            xtl, ytl = max(0, xtl), max(0, ytl)
            xbr, ybr = min(wt, xbr), min(ht, ybr)
            mask[ytl:ybr, xtl:xbr] = label

        if self.transforms:
            transformed = self.transforms(image=img, mask=mask)
            img, target = transformed['image'], transformed['mask']
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1))
            target = torch.as_tensor(mask, dtype=torch.int64)

        return img, target.long()

    def __len__(self):
        return len(self.X)


def get_transform(train, image_size):
    """Defines the augmentation pipeline for training and validation."""
    transforms = [A.Resize(image_size, image_size)]
    if train:
        transforms.append(A.HorizontalFlip(p=0.5))
    transforms.append(ToTensorV2(p=1.0))
    return A.Compose(transforms)


def collate_fn(batch):
    """Custom collate function to stack images and targets into a batch."""
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)
    return images, targets


def create_dataloaders(anno_path, img_path, batch_size, train_split_ratio, image_size, random_seed, num_workers=0):
    """Creates and returns the training and testing DataLoaders."""
    dataset_train_source = OD_US_Dataset_Segmentation(anno_path, img_path,
                                                      transforms=get_transform(train=True, image_size=image_size))
    dataset_test_source = OD_US_Dataset_Segmentation(anno_path, img_path,
                                                     transforms=get_transform(train=False, image_size=image_size))

    if len(dataset_train_source) == 0:
        raise ValueError("Dataset is empty. Check paths and annotation files.")

    dataset_size = len(dataset_train_source)
    train_size = int(train_split_ratio * dataset_size)
    test_size = dataset_size - train_size

    generator = torch.Generator().manual_seed(random_seed)
    train_indices, test_indices = random_split(range(dataset_size), [train_size, test_size], generator=generator)

    train_dataset = Subset(dataset_train_source, train_indices.indices)
    test_dataset = Subset(dataset_test_source, test_indices.indices)

    print(f"Total images found: {dataset_size}")
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             collate_fn=collate_fn)

    return train_loader, test_loader