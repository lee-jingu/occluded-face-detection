from __future__ import annotations
import os
from typing import Callable

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, random_split


class MultiLabelDataset(Dataset):

    def __init__(self,
                 image_ids: list[str],
                 transform: Callable | None = None):
        self.image_ids = image_ids
        self.transform = transform

        self.class_names = ['non_occluded', 'top_occluded', 'bottom_occluded']
        self.class_count = [0, 0, 0]
        
        self.labels = self._get_labels(image_ids)
        self.class_weights = self._get_class_weights()

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
            self.image_ids = [
                self.image_ids[i] for i in range(len(self.image_ids))
                if i not in mislabels
            ]
        return labels

    def _get_class_weights(self) -> torch.Tensor:
        num_data = sum(self.class_count)
        num_classes = len(self.class_count)
        class_weights = [
            num_data / (cnt * num_classes) for cnt in self.class_count
        ]
        return torch.Tensor([1, 1, 4])
    
    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, ids: int) -> tuple:
        img_path = self.image_ids[ids]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image) if self.transform else image
        target = np.array(self.labels[img_path]).astype(np.float32)
        return image, target


class MultiClassDataset(Dataset):

    def __init__(self,
                 image_ids: list[str],
                 transform: Callable | None = None):
        self.image_ids = image_ids
        self.transform = transform

        self.class_names = [
            'not_occluded',
            'top_only_occluded',
            'bottom_only_occluded',
            'both_occluded'
        ]
        self.class_count = [0, 0, 0, 0]
        
        self.labels = self._get_labels(image_ids)
        self.class_weights = self._get_class_weights()

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
                label = [1, 0, 0, 0]
            if top_occluded and not bottom_occluded:
                self.class_count[1] += 1
                label = [0, 1, 0, 0]
            if bottom_occluded and not top_occluded:
                self.class_count[2] += 1
                label = [0, 0, 1, 0]
            if top_occluded and bottom_occluded:
                self.class_count[3] += 1
                label = [0, 0, 0, 1]

            labels[path] = label

        if len(mislabels) > 0:
            self.image_ids = [
                self.image_ids[i] for i in range(len(self.image_ids))
                if i not in mislabels
            ]

        return labels

    def _get_class_weights(self) -> torch.Tensor:
        num_data = sum(self.class_count)
        num_classes = len(self.class_count)
        class_weights = [
            num_data / (cnt * num_classes) for cnt in self.class_count
        ]
        return torch.Tensor([1, 1, 4, 10])
    
    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, ids: int) -> tuple:
        img_path = self.image_ids[ids]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image) if self.transform else image
        target = np.array(self.labels[img_path]).astype(np.float32)
        return image, target


def get_train_val_dataset(
        img_dir: os.PathLike, split_ratio: float,
        transforms: dict[str, Callable]) -> dict[str, dict[str, Dataset]]:
    if 'train' not in transforms:
        raise ValueError('transforms must contain train')

    if 'val' not in transforms:
        raise ValueError('transforms must contain val')

    image_extensions = ['jpg', 'png', 'jpeg']
    dataset = os.listdir(img_dir)
    dataset = [os.path.join(img_dir, x) for x in dataset if x.split('.')[-1].lower() in image_extensions]

    train_size = int(len(dataset) * split_ratio)
    val_size = (len(dataset) - train_size) // 2
    test_size = (len(dataset) - train_size) - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    output = {
        'multi-label': {
            'train': MultiLabelDataset(train_set, transforms['train']),
            'val': MultiLabelDataset(val_set, transforms['val']) if val_size else None,
            'test': MultiLabelDataset(test_set, transforms['val']) if test_size else None
        },
        'multi-class': {
            'train': MultiClassDataset(train_set, transforms['train']),
            'val': MultiClassDataset(val_set, transforms['val']) if val_size else None,
            'test': MultiClassDataset(test_set, transforms['val']) if test_size else None
        }
    }

    return output