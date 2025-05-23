import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn

from pytorch_grad_cam.model_visualizer import ModelVisualizer
from pytorch_grad_cam.utils.image import preprocess_image


ALLOWED_CAMS = [
    'gradcam', 'scorecam', 'gradcamplusplus', 'ablationcam', 'xgradcam',
    'eigencam', 'eigengradcam', 'layercam', 'fullgrad', 'gradcamelementwise',
    'randomcam', 'fem', 'shapleycam', 'kpcacam', 'finercam'
]


class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cam_names', nargs='+', default=['gradcam'], choices=ALLOWED_CAMS
    )
    parser.add_argument(
        '--image_path', type=str, default='./examples/both.png'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = SimpleConvNet().eval()

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (128, 128))
    input_tensor = preprocess_image(rgb_img.astype(np.float32) / 255)

    visualizer = ModelVisualizer(model, args.cam_names)
    visualizer.visualize(input_tensor, rgb_img.astype(np.float32) / 255)


if __name__ == '__main__':
    main()
