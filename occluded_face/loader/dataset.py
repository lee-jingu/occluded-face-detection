from __future__ import annotations
import os
from typing import Callable

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, random_split

class MultiLabelDataset(Dataset):
    def __init__(self, image_ids: list[str], transform: Callable | None = None):
        self.image_ids = image_ids
        self.transform = transform

        self.labels = self._get_labels(image_ids)
        self.class_names = ['not_occluded', 'top_occluded', 'bottom_occluded']
        self.class_count = [0, 0, 0]

    def _get_labels(self, image_ids: os.PathLike) -> dict:
        labels = {}
        mislabels = []
        for idx, path in enumerate(image_ids):
            file = os.path.basename(path)
            txt_label = file.split('_')[:3]
            txt_label[2] = txt_label[2][0]
            not_occluded = txt_label[0] == '0'
            top_occluded = txt_label[1] == '1'
            bottom_occluded = txt_label[2] == '1'
            if top_occluded or bottom_occluded:
                if not_occluded:
                    mislabels.append(idx)
                    continue
            if not_occluded:
                self.class_count[0] += 1
            if top_occluded:
                self.class_count[1] += 1
            if bottom_occluded:
                self.class_count[2] += 1
            labels[path] = [not_occluded, top_occluded, bottom_occluded]
        
        if len(mislabels) > 0:
            self.image_ids = [self.image_ids[i] for i in range(len(self.image_ids)) if i not in mislabels]
        return labels
    
    @property
    def class_weights(self) -> torch.Tensor:
        num_data = sum(self.class_count)
        class_weights = [1 - i / num_data for i in range(len(self.class_count))]
        return torch.Tensor(class_weights)
    
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, ids: int) -> tuple:
        img_path = self.image_ids[ids]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image) if self.transform else image
        target = np.array(self.labels[img_path]).astype(np.float32)
        return image, target

class MultiClassDataset(Dataset):
    def __init__(self, image_ids: list[str], transform: Callable | None = None):
        self.image_ids = image_ids
        self.transform = transform

        self.labels = self._get_labels(image_ids)
        self.class_names = ['not_occluded', 'top_only_occluded', 'bottom_only_occluded', 'both_occluded']
        self.class_count = [0, 0, 0, 0]
    
    def _get_labels(self, image_ids: os.PathLike) -> dict:
        labels = {}
        mislabels = []
        for idx, path in enumerate(image_ids):
            file = os.path.basename(path)
            txt_label = file.split('_')[:3]
            txt_label[2] = txt_label[2][0]
            not_occluded = txt_label[0] == '0'
            top_occluded = txt_label[1] == '1'
            bottom_occluded = txt_label[2] == '1'

            if top_occluded or bottom_occluded:
                if not_occluded:
                    mislabels.append(idx)
                    continue
            
            if not_occluded:
                self.class_count[0] += 1
                label = 0
            if top_occluded and not bottom_occluded:
                self.class_count[1] += 1
                label = 1
            if bottom_occluded and not top_occluded:
                self.class_count[2] += 1
                label = 2
            if top_occluded and bottom_occluded:
                self.class_count[3] += 1
                label = 3
            
            labels[path] = label
        
        return labels

    @property
    def class_weights(self) -> torch.Tensor:
        num_data = sum(self.class_count)
        class_weights = [1 - i / num_data for i in range(len(self.class_count))]
        return torch.Tensor(class_weights)
    
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, ids: int) -> tuple:
        img_path = self.image_ids[ids]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image) if self.transform else image
        target = torch.FloatTensor(self.labels[img_path])
        return image, target

def get_train_val_dataset(img_dir: os.PathLike,
                          split_ratio: float,
                          transforms: dict[str, Callable]
                          )-> tuple[MultiLabelDataset, MultiLabelDataset, MultiClassDataset, MultiClassDataset]:
    if 'train' not in transforms:
        raise ValueError('transforms must contain train')
    
    if 'val' not in transforms:
        raise ValueError('transforms must contain val')
    
    dataset = os.listdir(img_dir)
    dataset = [os.path.join(img_dir, x) for x in dataset if x.endswith(".jpg")]
    
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    ml_train_set = MultiLabelDataset(train_set, transforms.get('train'))
    ml_val_set = MultiLabelDataset(val_set, transforms.get('val'))

    mc_train_set = MultiClassDataset(train_set, transforms.get('train'))
    mc_val_set = MultiClassDataset(val_set, transforms.get('val'))

    return ml_train_set, ml_val_set, mc_train_set, mc_val_set