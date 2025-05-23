import math
from typing import List, Type, Iterable

import numpy as np
import torch
import matplotlib.pyplot as plt

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
    GradCAMElementWise,
    RandomCAM,
    FEM,
    ShapleyCAM,
    KPCA_CAM,
    FinerCAM,
)
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.find_layers import find_layer_predicate_recursive


CAM_ALGORITHMS = {
    "gradcam": GradCAM,
    "scorecam": ScoreCAM,
    "gradcam++": GradCAMPlusPlus,
    "ablationcam": AblationCAM,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "layercam": LayerCAM,
    "fullgrad": FullGrad,
    "gradcamelementwise": GradCAMElementWise,
    "randomcam": RandomCAM,
    "fem": FEM,
    "shapleycam": ShapleyCAM,
    "kpcacam": KPCA_CAM,
    "finercam": FinerCAM,
}

CAM_METHODS = CAM_ALGORITHMS


class ModelVisualizer:
    """Recursively compute layer importances using other CAMs."""

    CAM_METHODS = CAM_ALGORITHMS

    def __init__(
        self,
        model: torch.nn.Module,
        cam_algorithms: Iterable[Type[BaseCAM]] | Type[BaseCAM] | str,
        reshape_transform=None,
        batch_size: int = 32,
    ) -> None:
        self.model = model
        if isinstance(cam_algorithms, (str,)):
            cam_algorithms = [CAM_ALGORITHMS[cam_algorithms]]
        elif isinstance(cam_algorithms, type) and issubclass(cam_algorithms, BaseCAM):
            cam_algorithms = [cam_algorithms]
        self.cam_algorithms: List[Type[BaseCAM]] = list(cam_algorithms)
        self.reshape_transform = reshape_transform
        self.batch_size = batch_size
        self.device = next(model.parameters()).device
        self.conv_layers = find_layer_predicate_recursive(
            model, lambda l: isinstance(l, torch.nn.Conv2d)
        )

    def compute_importances(self, input_tensor: torch.Tensor, targets=None) -> List[np.ndarray]:
        importances = []
        for layer in self.conv_layers:
            cams = []
            for algo in self.cam_algorithms:
                with algo(
                    model=self.model,
                    target_layers=[layer],
                    reshape_transform=self.reshape_transform,
                ) as cam:
                    cam.batch_size = self.batch_size
                    grayscale = cam(input_tensor=input_tensor, targets=targets)
                    cams.append(grayscale)
            avg = np.mean(cams, axis=0)
            importances.append(avg)
        return importances

    def visualize_importances(self, input_tensor: torch.Tensor, targets=None, cols: int = 4) -> None:
        cams = self.compute_importances(input_tensor, targets)
        num_layers = len(cams)
        rows = math.ceil(num_layers / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).reshape(rows, cols)
        for idx, cam in enumerate(cams):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            ax.imshow(cam[0], cmap="jet")
            ax.set_title(f"Layer {idx}")
            ax.axis("off")
        for idx in range(num_layers, rows * cols):
            r, c = divmod(idx, cols)
            axes[r, c].axis("off")
        plt.tight_layout()
        plt.show()

