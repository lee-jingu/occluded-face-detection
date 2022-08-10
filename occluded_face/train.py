import os
import argparse

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.models import MobileNetV2

from utils import split_dataset, transform_dataset, get_scores
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--task',
                        type=str,
                        default='multi-label',
                        help='multi-label or multi-class')
    parser.add_argument('--img-dir', type=str, default='./dataset/')
    parser.add_argument('--output-dir', type=str, default='./checkpoints/')
    parser.add_argument('--use-class-weight', action='store_true')
    parser.add_argument('--metric', action='store_true')
    parser.add_argument('--split-ratio',
                        nargs=3,
                        type=float,
                        default=[0.8, 0.1, 0.1])

    return parser.parse_args()


def main():
    args = parse_args()
    args.metric = True
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_set, valid_set, test_set = split_dataset(args.img_dir,
                                                   args.split_ratio)
    train_set = transform_dataset(train_set, args.task, is_train=True)
    valid_set = transform_dataset(valid_set, args.task)
    test_set = transform_dataset(test_set, args.task)

    train_loader = DataLoader(train_set,
                              batch_size=256,
                              shuffle=True,
                              num_workers=8)
    valid_loader = DataLoader(valid_set,
                              batch_size=16,
                              shuffle=False,
                              num_workers=2)
    test_loader = DataLoader(test_set,
                             batch_size=16,
                             shuffle=False,
                             num_workers=2)

    checkpoint_dir = os.path.join(
        args.output_dir, args.task,
        'weight' if args.use_class_weight else 'none')

    model = MobileNetV2(num_classes=train_set.num_classes).to(device)

    weight = train_set.class_weights.to(
        device) if args.use_class_weight else None

    if args.task == 'multi-label':
        criterion = torch.nn.BCEWithLogitsLoss(weight=weight)
    elif args.task == 'multi-class':
        criterion = torch.nn.CrossEntropyLoss(weight=weight)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer,
                                  'min',
                                  patience=args.epochs // 5,
                                  verbose=True)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()
    except ImportError:
        writer = None

    trainer = Trainer(model, train_loader, valid_loader, checkpoint_dir,
                      criterion, scheduler, optimizer, writer)

    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.validation(epoch)

    if args.metric:
        results = {}
        with torch.no_grad():
            results['train'] = get_scores(model, train_loader, device)
            results['valid'] = get_scores(model, valid_loader, device)
            results['test'] = get_scores(model, test_loader, device)

        with open('results.txt', 'a+') as f:
            f.write(
                f'{args.task}-{args.use_class_weight} Epeochs: {args.epochs} \n'
            )
            for _name, res in results.items():
                f.write(f'Dataset: {_name}\n')
                for k, v in res.items():
                    f.write(f'{k}: {v}\n')


if __name__ == '__main__':
    main()
