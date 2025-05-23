import torch
from pytorch_grad_cam import GradCAM

class Simple1D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 6, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(6, 2)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = x.mean(dim=-1)
        return self.fc(x)


def test_gradcam_1d():
    model = Simple1D()
    input_tensor = torch.randn(2, 3, 20)
    cam = GradCAM(model=model, target_layers=[model.conv])
    result = cam(input_tensor=input_tensor)
    assert result.shape == (2, 20)

