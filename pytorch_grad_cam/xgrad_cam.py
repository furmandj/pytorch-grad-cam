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
        dims = len(activations.shape) - 2
        eps = 1e-7
        if dims == 1:
            sum_activations = np.sum(activations, axis=2)
            weights = grads * activations / (sum_activations[:, :, None] + eps)
            weights = weights.sum(axis=2)
        elif dims == 2:
            sum_activations = np.sum(activations, axis=(2, 3))
            weights = grads * activations / (
                sum_activations[:, :, None, None] + eps)
            weights = weights.sum(axis=(2, 3))
        elif dims == 3:
            sum_activations = np.sum(activations, axis=(2, 3, 4))
            weights = grads * activations / (
                sum_activations[:, :, None, None, None] + eps)
            weights = weights.sum(axis=(2, 3, 4))
        else:
            raise ValueError("Unsupported activation dimension")
        return weights
