import numpy as np
import torch
from typing import Dict, List
import matplotlib.pyplot as plt

from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.find_layers import find_layer_predicate_recursive


class RecursiveCAM:
    """Compute CAMs for all layers in a model using other CAM algorithms."""

    def __init__(self, model: torch.nn.Module, cam_algorithms, reshape_transform=None) -> None:
        self.model = model.eval()
        if not isinstance(cam_algorithms, list):
            cam_algorithms = [cam_algorithms]
        self.cam_algorithms = cam_algorithms
        self.reshape_transform = reshape_transform
        self.device = next(self.model.parameters()).device
        self.layers = self._get_leaf_layers(self.model)

    def _get_leaf_layers(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        return find_layer_predicate_recursive(model, lambda l: len(list(l.children())) == 0)

    def compute_importances(self, input_tensor: torch.Tensor, targets=None, eigen_smooth: bool = False) -> Dict[torch.nn.Module, np.ndarray]:
        input_tensor = input_tensor.to(self.device)
        cams: Dict[torch.nn.Module, np.ndarray] = {}
        for layer in self.layers:
            cam_results = []
            for cam_method in self.cam_algorithms:
                with cam_method(self.model, [layer], self.reshape_transform) as cam:
                    cam_result = cam(input_tensor=input_tensor, targets=targets, eigen_smooth=eigen_smooth)
                    cam_results.append(cam_result)
            cam_avg = np.mean(np.stack(cam_results, axis=0), axis=0)
            cams[layer] = cam_avg
        return cams

    def visualize(self, cam_dict: Dict[torch.nn.Module, np.ndarray]) -> None:
        num_layers = len(cam_dict)
        fig, axes = plt.subplots(num_layers, 1, figsize=(6, 3 * num_layers))
        if num_layers == 1:
            axes = [axes]
        for ax, (layer, cam) in zip(axes, cam_dict.items()):
            img = cam[0]
            im = ax.imshow(img, cmap='jet')
            ax.set_title(str(layer))
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()
