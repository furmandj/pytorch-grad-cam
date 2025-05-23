import matplotlib.pyplot as plt
import numpy as np
import torch

from typing import List, Type, Dict, Union

from .base_cam import BaseCAM


class ModelVisualizer:
    """Compute and visualize layer importances using CAMs."""

    def __init__(
        self,
        model: torch.nn.Module,
        cam_methods: Union[Type[BaseCAM], List[Type[BaseCAM]]],
        reshape_transform=None,
    ) -> None:
        if not isinstance(cam_methods, list):
            cam_methods = [cam_methods]
        self.cam_methods = cam_methods
        self.model = model.eval()
        self.reshape_transform = reshape_transform
        self.device = next(self.model.parameters()).device
        self.conv_layers = [
            (name, module)
            for name, module in self.model.named_modules()
            if isinstance(module, torch.nn.Conv2d)
        ]

    def _compute_cam(
        self, cam_cls: Type[BaseCAM], layer: torch.nn.Module, input_tensor, targets
    ) -> np.ndarray:
        with cam_cls(self.model, [layer], reshape_transform=self.reshape_transform) as cam:
            return cam(input_tensor=input_tensor, targets=targets)

    def compute_importances(
        self, input_tensor: torch.Tensor, targets=None
    ) -> Dict[str, np.ndarray]:
        input_tensor = input_tensor.to(self.device)
        importances: Dict[str, np.ndarray] = {}
        for name, layer in self.conv_layers:
            cams = [
                self._compute_cam(cam_cls, layer, input_tensor, targets)
                for cam_cls in self.cam_methods
            ]
            cam_avg = np.mean(np.stack(cams, axis=0), axis=0)
            importances[name] = cam_avg
        return importances

    def visualize(self, input_tensor: torch.Tensor, targets=None, cmap: str = "jet") -> None:
        importances = self.compute_importances(input_tensor, targets)
        num_layers = len(importances)
        fig, axes = plt.subplots(1, num_layers, figsize=(4 * num_layers, 4))
        if num_layers == 1:
            axes = [axes]
        for ax, (name, cam) in zip(axes, importances.items()):
            ax.imshow(cam[0], cmap=cmap)
            ax.set_title(name)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

