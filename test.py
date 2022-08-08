from __future__ import annotations
import argparse

import torch
from torchvision import transforms
from torchvision.models import MobileNetV2
from torchmetrics import F1Score, Recall, Precision, Accuracy

from occluded_face.loader.dataset import MultiClassDataset, MultiLabelDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='./new_train_3/', help='path to image directory')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/model_best.pth', help='path to checkpoint')
    parser.add_argument('--task', type=str, default='class', help='task')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    return parser.parse_args()


def get_f1_score_class(device: torch.device,
                       model: torch.nn.Module,
                       dataset: MultiClassDataset | MultiClassDataset,
                       num_classes: int):
    f1_score = 0
    precision_score = 0
    recall_score = 0
    accuracy_score = 0

    f1 = F1Score(num_classes=num_classes)
    weighted_f1 = F1Score(num_classes=num_classes, average='weighted')
    precision = Precision(num_classes=num_classes)
    recall = Recall(num_classes=num_classes)
    accuracy = Accuracy(num_classes=num_classes)

    model.eval()
    model.to(device)
    with torch.no_grad():
        for image, target in dataset:
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            f1_score += f1(output, target)
            weighted_f1 += weighted_f1(output, target)
            precision_score += precision(output, target)
            recall_score += recall(output, target)
            accuracy_score += accuracy(output, target)
    
    print(f'f1_score: {f1_score / len(dataset)}')
    print(f'weighted_f1_score: {weighted_f1 / len(dataset)}')
    print(f'precision_score: {precision_score / len(dataset)}')
    print(f'recall_score: {recall_score / len(dataset)}')
    print(f'accuracy_score: {accuracy_score / len(dataset)}')

def main():
    args = parse_args()
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')

    transform = transforms.Compose([
        transforms.Resize(112),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    if args.task == 'class':
        dataset = MultiClassDataset(args.img_dir, transform)
        model = MobileNetV2(num_classes=4, pretrained=False)
        model.load_state_dict(torch.load(args.checkpoint))
    
        get_f1_score_class(device, model, dataset, 4)
    
    elif args.task == 'label':
        dataset = MultiLabelDataset(args.img_dir, transform)
        model = MobileNetV2(num_classes=3, pretrained=False)
        model.load_state_dict(torch.load(args.checkpoint))
    
        get_f1_score_class(device, model, dataset, 3)     


if __name__ == '__main__':
    main()