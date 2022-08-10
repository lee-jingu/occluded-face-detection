import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.models import MobileNetV2

from utils import split_dataset, transform_dataset, get_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        type=str,
                        default='./checkpoints/multi-label/none/best.pth',
                        help='path to checkpoint')
    parser.add_argument('--task',
                        type=str,
                        default='multi-label',
                        help='task type')
    parser.add_argument('--img-dir', type=str, default='./dataset/')

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = transform_dataset(split_dataset(args.img_dir)[0], args.task)
    dataloader = DataLoader(dataset,
                            batch_size=256,
                            shuffle=False,
                            num_workers=8)

    model = MobileNetV2(num_classes=dataset.num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    scores = get_scores(model, dataloader, device)

    for name, score in scores.items():
        print(f'{name}: {score}')


if __name__ == '__main__':
    main()