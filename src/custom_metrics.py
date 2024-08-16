import torch
from dipy.tracking.utils import density_map
from dipy.io.streamline import load_trk

@torch.jit.script
def get_weighted_dice_coefficient(density_map_gt: torch.Tensor, 
                                  density_map_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculates the weighted Dice coefficient between two fiber density maps.
    """

    # Ensure tensors are on the same device (GPU)
    density_map_gt = density_map_gt.view(-1).float()
    density_map_pred = density_map_pred.view(-1).float()

    # Identify the intersection
    intersection_indices = (density_map_gt > 0) & (density_map_pred > 0)

    # Sum the intersection values
    sum_intersection = torch.sum(density_map_gt[intersection_indices]) + torch.sum(density_map_pred[intersection_indices])

    # Total sum of all voxels
    sum_total = torch.sum(density_map_gt) + torch.sum(density_map_pred)

    # Calculate the weighted Dice coefficient
    wdice = sum_intersection / sum_total

    # Handle NaN values
    if wdice.isnan():
        wdice = torch.tensor(0.0, device=density_map_gt.device)

    return wdice

@torch.jit.script
def get_dice_coefficient(density_map_gt: torch.Tensor,
                            density_map_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Dice coefficient between two fiber density maps.
    """
    # Ensure tensors are on the same device (GPU)
    density_map_gt = (density_map_gt > 0).view(-1).float()
    density_map_pred = (density_map_pred > 0).view(-1).float()

    # Calculate intersection and union
    intersection = 2 * torch.sum(torch.minimum(density_map_gt, density_map_pred))
    union = torch.sum(density_map_gt) + torch.sum(density_map_pred)

    # Calculate the Dice coefficient
    dice = intersection / union

    # Handle NaN values
    if dice.isnan():
        dice = torch.tensor(0.0, device=density_map_gt.device)

    return dice


def get_dice_metrics(file:str, 
                     correct_streamlines_idx:list[int]) -> tuple[float, float]:
    
    
    # Cargar el archivo trk y obtener el tractograma
    tractogram = load_trk(str(file), 
                          reference='same', 
                          bbox_valid_check=False)
    
    # tractogram.remove_invalid_streamlines()
    
    streamlines = tractogram.streamlines
    affine = tractogram.affine
    dimensions = tractogram.dimensions
    
    print(f'Fibras clasificadas correctamente: {len(correct_streamlines_idx)}')
    print(f'Fibras totales: {len(streamlines)}')
    
    # Calculo del weighted dice coefficient
    density_map_gt = density_map(streamlines, affine, dimensions)
    
    # Calcular el mapa de densidad de las fibras clasificadas correctamente
    correct_streamlines = [streamlines[i] for i in correct_streamlines_idx]
    density_map_pred = density_map(correct_streamlines, affine, dimensions)
    
    # Convertir a tensores
    density_map_gt = torch.tensor(density_map_gt)
    density_map_pred = torch.tensor(density_map_pred)
    
    # Calcular el weighted dice coefficient
    wdice = get_weighted_dice_coefficient(density_map_gt, 
                                          density_map_pred)
    # Calcular el dice coefficient
    dice = get_dice_coefficient(density_map_gt, density_map_pred)
   

    return wdice, dice