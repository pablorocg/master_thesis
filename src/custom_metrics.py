import torch
from dipy.tracking.utils import density_map
from dipy.io.streamline import load_trk



@torch.jit.script
def get_weighted_dice_coefficient(density_map_gt: torch.Tensor, 
                                  density_map_pred: torch.Tensor,
                                  gt_len:int, 
                                  pred_len:int) -> torch.Tensor:
    """
    Calcula el weighted dice coefficient entre dos mapas de densidad de fibras.
    """

    # Convertir a tensor plano con view(-1)
    density_map_gt = density_map_gt.view(-1).float()
    density_map_pred = density_map_pred.view(-1).float()

    # Dividir n de cada mapa de densidad por el numero de fibras en el tracto
    weighted_density_map_gt = density_map_gt / gt_len
    weighted_density_map_pred = density_map_pred / pred_len

    # Obtener la interseccion entre conjuntos
    intersection = 2 * torch.sum(torch.minimum(weighted_density_map_gt, weighted_density_map_pred))
    union = torch.sum(weighted_density_map_gt) + torch.sum(weighted_density_map_pred)

    wdice = intersection / union

    if wdice.isnan():
        wdice = torch.tensor(0.0)

    return wdice


@torch.jit.script
def get_dice_coefficient(density_map_gt: torch.Tensor,
                            density_map_pred: torch.Tensor) -> torch.Tensor:
    """
    Calcula el dice coefficient entre dos mapas de densidad de fibras.
    """
   
    density_map_gt = density_map_gt > 0
    density_map_pred = density_map_pred > 0

    # Convertir a tensor plano con view(-1)
    density_map_gt = density_map_gt.view(-1).float()
    density_map_pred = density_map_pred.view(-1).float()

    # Obtener la interseccion entre conjuntos
    intersection = 2 * torch.sum(torch.minimum(density_map_gt, density_map_pred))
    union = torch.sum(density_map_gt) + torch.sum(density_map_pred)

    dice = (intersection) / union
    
    if dice.isnan():
        dice = torch.tensor(0.0)

    return dice



def get_dice_metrics(file:str, 
                     correct_streamlines_idx:list[int]) -> tuple[float, float]:
    
    
    # Cargar el archivo trk y obtener el tractograma
    tractogram = load_trk(str(file), 
                          reference='same', 
                          bbox_valid_check=False)
    
    tractogram.remove_invalid_streamlines()
    
    streamlines = tractogram.streamlines
    affine = tractogram.affine
    dimensions = tractogram.dimensions

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
                                          density_map_pred,
                                          len(streamlines), 
                                          len(correct_streamlines))
    # Calcular el dice coefficient
    dice = get_dice_coefficient(density_map_gt, density_map_pred)
   

    return wdice, dice