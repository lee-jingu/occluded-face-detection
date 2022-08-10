from __future__ import annotations
import os

from torch.utils.data import random_split
from torch.utils.data.dataset import Subset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomHorizontalFlip

from trainer import MultiClassDataset, MultiLabelDataset


def split_dataset(
        img_dir: os.PathLike,
        split_ratio: tuple[float, float, float] | None = None) -> tuple[list, list, list]:

    if split_ratio is None:
        split_ratio = [1.0, 0.0, 0.0]

    if sum(split_ratio) != 1:
        raise ValueError('split_ratio must sum to 1')

    if not os.path.exists(img_dir):
        raise ValueError(f'{os.path.abspath(img_dir)} does not exist')

    image_extensions = ['jpg', 'png', 'jpeg']
    dataset = os.listdir(img_dir)

    dataset = [
        os.path.join(img_dir, x) for x in dataset
        if x.split('.')[-1].lower() in image_extensions
    ]

    train_size = int(len(dataset) * split_ratio[0])

    val_test_ratio = (split_ratio[1] + split_ratio[2]) / 2
    val_size = int((len(dataset) - train_size) * val_test_ratio)

    test_size = int((len(dataset) - train_size) - val_size)

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size])

    return train_set, val_set, test_set


def transform_dataset(
        dataset: Subset,
        task: str,
        is_train: bool = False,
        img_size: int = 112) -> MultiClassDataset | MultiLabelDataset:
    if is_train:
        transforms = Compose([
            Resize(img_size),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transforms = Compose([
            Resize(img_size),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    if task == 'multi-label':
        return MultiLabelDataset(dataset, transforms)
    elif task == 'multi-class':
        return MultiClassDataset(dataset, transforms)
    else:
        raise ValueError(f'{task} is not a valid task')