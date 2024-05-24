from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
import torch

class MaxMinNormalization(BaseTransform):
    def __init__(self, 
                 max_values = [75, 82.5, 97.5], 
                 min_values = [-77, -120.5, -81.5]):
        """
        Inicializa la transformación de normalización con valores máximos y mínimos opcionales.
        Si no se proporcionan, deben calcularse a partir del conjunto de datos.
        
        [74.99879455566406, 82.36431884765625, 97.47947692871094], 
        [-76.92510986328125, -120.4773941040039, -81.27867126464844], 

        """
        self.max_values = torch.tensor(max_values, dtype=torch.float)
        self.min_values = torch.tensor(min_values, dtype=torch.float)

    def __call__(self, data: Data) -> Data:
        """
        Aplica normalización min-max a las características del nodo.
        """
        data.x = (data.x - self.min_values) / (self.max_values - self.min_values)
        return data
