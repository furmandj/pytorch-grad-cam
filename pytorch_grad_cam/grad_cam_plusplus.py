import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM

# https://arxiv.org/abs/1710.11063


class GradCAMPlusPlus(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(GradCAMPlusPlus, self).__init__(model, target_layers,
                                              reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layers,
                        target_category,
                        activations,
                        grads):
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        dims = len(activations.shape) - 2
        if dims == 1:
            sum_activations = np.sum(activations, axis=2)
            eps = 1e-6
            aij = grads_power_2 / (
                2 * grads_power_2
                + sum_activations[:, :, None] * grads_power_3
                + eps
            )
            aij = np.where(grads != 0, aij, 0)
            weights = np.maximum(grads, 0) * aij
            weights = np.sum(weights, axis=2)
        elif dims == 2:
            sum_activations = np.sum(activations, axis=(2, 3))
            eps = 1e-6
            aij = grads_power_2 / (
                2 * grads_power_2
                + sum_activations[:, :, None, None] * grads_power_3
                + eps
            )
            aij = np.where(grads != 0, aij, 0)
            weights = np.maximum(grads, 0) * aij
            weights = np.sum(weights, axis=(2, 3))
        elif dims == 3:
            sum_activations = np.sum(activations, axis=(2, 3, 4))
            eps = 1e-6
            aij = grads_power_2 / (
                2 * grads_power_2
                + sum_activations[:, :, None, None, None] * grads_power_3
                + eps
            )
            aij = np.where(grads != 0, aij, 0)
            weights = np.maximum(grads, 0) * aij
            weights = np.sum(weights, axis=(2, 3, 4))
        else:
            raise ValueError("Unsupported activation dimension")
        return weights
