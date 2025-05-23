import cv2
import numpy as np
import torch
from torchvision import models

from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.model_visualizer import ModelVisualizer
from pytorch_grad_cam.utils.image import preprocess_image


if __name__ == "__main__":
    model = models.resnet18(pretrained=True)
    visualizer = ModelVisualizer(model, [GradCAM, ScoreCAM])

    img = cv2.imread("examples/both.png")
    img = np.float32(img) / 255
    input_tensor = preprocess_image(img)

    visualizer.visualize(input_tensor)
