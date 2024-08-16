from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
import torch

class MaxMinNormalization(BaseTransform):
    def __init__(self, dataset=None):
        """
        Initialize the normalization transform with optional max and min values.
        If not provided, they should be computed from the dataset.
        """
        if dataset == "HCP_105" or dataset == "Tractoinferno" or dataset == "HCP_105_without_CC":
        # Normalizacion para datos MNI-152

            self.max_values = torch.tensor([74.99879455566406, 82.36431884765625, 97.47947692871094], dtype=torch.float)
            self.min_values = torch.tensor([-76.92510986328125, -120.4773941040039, -81.27867126464844], dtype=torch.float)

        elif dataset == "FiberCup":
            self.max_values = torch.tensor([486.051, 454.08902, 25.558952], dtype=torch.float)
            self.min_values = torch.tensor([72.0, 47.815502, -6.408], dtype=torch.float)
        else:
            # Lanzar error
            raise ValueError("Dataset no reconocido")

    def __call__(self, data: Data) -> Data:
        """
        Apply min-max normalization to the node features.
        data.x is expected to be a tensor of shape [num_nodes, 3].
        """
        data.x = (data.x - self.min_values) / (self.max_values - self.min_values)
        return data




class Cartesian2SphericalCoords(BaseTransform):
    """
    Data(x = [x, y, z]) -> Data(x = [r, theta, phi])
    """
    def __call__(self, data: Data) -> Data:
        """
        Convert Cartesian coordinates to spherical coordinates.
        data.x is expected to be a tensor of shape [num_nodes, 3].
        """
        x, y, z = data.x[:, 0], data.x[:, 1], data.x[:, 2]
        r = torch.sqrt(x**2 + y**2 + z**2)
        theta = torch.acos(z / r)
        phi = torch.atan2(y, x)
        data.x = torch.stack([r, theta, phi], dim=1)
        return data

