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
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
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
        eps = 0.000001
        aij = grads_power_2 / (
            2 * grads_power_2 +
            sum_activations[(...,) + (None,) * (grads.ndim - 2)] * grads_power_3 +
            eps
        )
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)
        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=axes)
        return weights
