from __future__ import annotations
import os
from typing import Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, random_split

class CustomDataset(Dataset):
    def __init__(self, image_ids: list[str], transform: Callable | None = None):
        self.image_ids = image_ids
        self.transform = transform

        self.labels = self._get_labels(image_ids)

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
            labels[path] = [not_occluded, top_occluded, bottom_occluded]
        
        if len(mislabels) > 0:
            self.image_ids = [self.image_ids[i] for i in range(len(self.image_ids)) if i not in mislabels]
        return labels
    
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, ids: int) -> tuple:
        img_path = self.image_ids[ids]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image) if self.transform else image
        target = np.array(self.labels[img_path]).astype(np.float32)
        return image, target

def get_train_val_dataset(img_dir: os.PathLike,
                          split_ratio: float,
                          transforms: dict[str, Callable]
                          )-> tuple[CustomDataset, CustomDataset]:
    if 'train' not in transforms:
        raise ValueError('transforms must contain train')
    
    if 'val' not in transforms:
        raise ValueError('transforms must contain val')
    
    dataset = os.listdir(img_dir)
    dataset = [os.path.join(img_dir, x) for x in dataset if x.endswith(".jpg")]
    
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_set = CustomDataset(train_set, transforms.get('train'))
    val_set = CustomDataset(val_set, transforms.get('val'))

    return train_set, val_set