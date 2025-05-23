import os
from typing import List, Union

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn as nn

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
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    scale_cam_image,
)


CAM_MAP = {
    "gradcam": GradCAM,
    "scorecam": ScoreCAM,
    "gradcamplusplus": GradCAMPlusPlus,
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


class ModelVisualizer:
    """Compute CAM visualizations for every convolutional layer."""

    def __init__(
        self,
        model: nn.Module,
        cam_names: Union[str, List[str]],
        reshape_transform=None,
    ) -> None:
        if isinstance(cam_names, str):
            cam_names = [cam_names]
        self.cam_names = cam_names
        self.model = model
        self.reshape_transform = reshape_transform
        self.device = next(model.parameters()).device

    def _get_conv_layers(self) -> List[nn.Module]:
        layers: List[nn.Module] = []
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                layers.append(module)
        return layers

    def _compute_layer_cam(
        self,
        layer: nn.Module,
        input_tensor: torch.Tensor,
        targets=None,
        eigen_smooth: bool = False,
        aug_smooth: bool = False,
    ) -> np.ndarray:
        cams = []
        for cam_name in self.cam_names:
            cam_class = CAM_MAP[cam_name.lower()]
            cam = cam_class(
                model=self.model,
                target_layers=[layer],
                reshape_transform=self.reshape_transform,
            )
            cam.batch_size = getattr(cam, "batch_size", 32)
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets,
                eigen_smooth=eigen_smooth,
                aug_smooth=aug_smooth,
            )[0]
            cams.append(grayscale_cam)
        cam_array = np.stack(cams, axis=0)
        return np.mean(cam_array, axis=0)

    def compute_importances(
        self,
        input_tensor: torch.Tensor,
        targets=None,
        eigen_smooth: bool = False,
        aug_smooth: bool = False,
    ) -> List[np.ndarray]:
        conv_layers = self._get_conv_layers()
        results: List[np.ndarray] = []
        for layer in conv_layers:
            cam = self._compute_layer_cam(
                layer,
                input_tensor,
                targets=targets,
                eigen_smooth=eigen_smooth,
                aug_smooth=aug_smooth,
            )
            results.append(cam)
        return results

    def visualize(
        self,
        input_tensor: torch.Tensor,
        input_image: np.ndarray,
        targets=None,
        eigen_smooth: bool = False,
        aug_smooth: bool = False,
    ) -> List[np.ndarray]:
        cams = self.compute_importances(
            input_tensor,
            targets=targets,
            eigen_smooth=eigen_smooth,
            aug_smooth=aug_smooth,
        )
        overlay_images = []
        for cam in cams:
            cam_img = scale_cam_image(np.expand_dims(cam, axis=0))[0]
            overlay = show_cam_on_image(
                cv2.resize(input_image, cam_img.shape[::-1]), cam_img, use_rgb=True
            )
            overlay_images.append(overlay)
        fig, axes = plt.subplots(1, len(overlay_images), figsize=(3 * len(overlay_images), 3))
        if len(overlay_images) == 1:
            axes = [axes]
        for ax, img in zip(axes, overlay_images):
            ax.imshow(img)
            ax.axis("off")
        plt.tight_layout()
        os.makedirs("visualization_outputs", exist_ok=True)
        cam_identifier = "_".join(self.cam_names)
        out_path = os.path.join(
            "visualization_outputs", f"{cam_identifier}_full_model.png"
        )
        fig.savefig(out_path)
        plt.close(fig)
        return cams
