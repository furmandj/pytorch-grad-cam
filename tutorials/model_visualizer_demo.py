import cv2
import numpy as np
import torch
from torchvision import models

from pytorch_grad_cam import GradCAM, LayerCAM, RecursiveCAM
from pytorch_grad_cam.utils.image import preprocess_image


if __name__ == "__main__":
    model = models.resnet18(pretrained=True)
    rgb_img = cv2.imread("examples/both.png", 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255

    input_tensor = preprocess_image(rgb_img)

    visualizer = RecursiveCAM(model, [GradCAM, LayerCAM])
    cams = visualizer.compute_importances(input_tensor)
    visualizer.visualize(cams)
