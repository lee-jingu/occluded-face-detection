import argparse

import cv2

from occluded_face.loader import FileLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./dataset/')
    return parser.parse_args()

def main():
    args = parse_args()
    loader = FileLoader(args.image_dir)

    for file in loader:
        file.show()
        print(file.info)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()