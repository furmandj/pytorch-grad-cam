import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM


class XGradCAM(BaseCAM):
    def __init__(
            self,
            model,
            target_layers,
            reshape_transform=None):
        super(
            XGradCAM,
            self).__init__(
            model,
            target_layers,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        if len(activations.shape) == 3:
            axes = (2,)
        elif len(activations.shape) == 4:
            axes = (2, 3)
        elif len(activations.shape) == 5:
            axes = (2, 3, 4)
        else:
            raise ValueError(
                "Invalid activations shape."
                "Shape should be 3 (1D), 4 (2D) or 5 (3D).")

        sum_activations = np.sum(activations, axis=axes)
        eps = 1e-7
        weights = grads * activations / (
            sum_activations[(...,) + (None,) * (grads.ndim - 2)] + eps)
        weights = weights.sum(axis=axes)
        return weights
