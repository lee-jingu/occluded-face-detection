import argparse

import onnx
import torch
from torchvision.models import MobileNetV2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint',
                        type=str,
                        default='./checkpoints/multi-class/none/last.pth',
                        help='path to checkpoint')
    parser.add_argument('--output',
                        type=str,
                        default='output.onnx',
                        help='output path')
    parser.add_argument('--task',
                        type=str,
                        default='multi-class',
                        help='task type')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.task not in ['multi-label', 'multi-class']:
        raise ValueError(f'Invalid task: {args.task}')

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 3 if args.task == 'multi-label' else 4

    model = MobileNetV2(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    dummy_data = torch.randn(1, 3, 112, 112, device=device)

    input_names = ['input']
    output_names = ['output']

    torch.onnx.export(model,
                      dummy_data,
                      args.output,
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=12)

    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    onnx.helper.printable_graph(onnx_model.graph)


if __name__ == '__main__':
    main()