"""Demo script for ModelVisualizer."""
import cv2
import numpy as np
import torch
from torchvision import models

from pytorch_grad_cam.model_visualizer import ModelVisualizer, CAM_METHODS
from pytorch_grad_cam.utils.image import preprocess_image


if __name__ == "__main__":
    model = models.resnet18(pretrained=True).eval()

    visualizer = ModelVisualizer(
        model=model,
        cam_algorithms=list(CAM_METHODS.values()),
    )

    img = cv2.imread("./examples/both.png")[:, :, ::-1]
    img = np.float32(img) / 255
    input_tensor = preprocess_image(img)

    visualizer.visualize_importances(input_tensor)


