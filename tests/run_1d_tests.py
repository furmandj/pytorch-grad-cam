import torch
import pytest

from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    EigenGradCAM,
    LayerCAM,
    FullGrad,
    KPCA_CAM,
    ShapleyCAM,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class Simple1DCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(3, 4, kernel_size=3, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(4, 8, kernel_size=3, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(8, 2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@pytest.mark.parametrize(
    "cam_method",
    [
        ScoreCAM,
        AblationCAM,
        GradCAM,
        GradCAMPlusPlus,
        XGradCAM,
        EigenCAM,
        EigenGradCAM,
        LayerCAM,
        FullGrad,
        KPCA_CAM,
        ShapleyCAM,
    ],
)
def test_cam_with_conv1d(cam_method):
    model = Simple1DCNN()
    target_layers = [model.conv2]
    cam = cam_method(model=model, target_layers=target_layers)
    cam.batch_size = 4
    input_tensor = torch.randn(2, 3, 32)
    targets = [ClassifierOutputTarget(1) for _ in range(input_tensor.size(0))]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    assert grayscale_cam.shape[0] == input_tensor.shape[0]
    assert grayscale_cam.shape[1] == input_tensor.shape[2]
