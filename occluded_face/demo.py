import argparse

import os

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.models import MobileNetV2

class Inference:
    def __init__(self, model, device):
        self.model = model
        self.device = device

        self.transforms = Compose([
            Resize(112),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        img = self.transforms(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)

        return np.argmax(self.model(img)[0].detach().cpu().numpy())

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='./dataset/')
    parser.add_argument('--checkpoint', type=str, default='14.pth', help='path to checkpoint')
    parser.add_argument('--task', type=str, default='multi-label', help='multi-label or multi-class')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--interval', type=int, default=800, help='interval [ms]')

    return parser.parse_args()

def main():
    args = parse_args()

    if args.task == 'multi-label':
        num_classes = 3
    elif args.task == 'multi-class':
        num_classes = 4
    else:
        raise ValueError(f'Invalid task: {args.task}')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = MobileNetV2(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    inference = Inference(model, device)

    for file in os.listdir(args.dir):
        path = os.path.join(args.dir, file)

        output = inference(Image.open(path))

        image_for_view = cv2.imread(path)
        image_for_view = cv2.resize(image_for_view, (600, 600))
        if output == 0:
            cv2.putText(image_for_view, 'NON-OCCLUDE', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)
            if args.verbose:
                print('NON-OCCLUDE')
        else:
            cv2.putText(image_for_view, 'OCCLUDE', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
            if args.verbose:
                print('OCCLUDE')

        cv2.imshow('image', image_for_view)
        if cv2.waitKey(args.interval) == ord('q'):
            break

if __name__ == '__main__':
    with torch.inference_mode():
        main()