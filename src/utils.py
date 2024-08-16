import os
import random
import numpy as np
import torch
import pathlib2 as pathlib
from collections import defaultdict
from torch_geometric.data import Data
from dataset_handlers import HCPHandler, HCP_Without_CC_Handler, TractoinfernoHandler, FiberCupHandler



def create_graph(streamline, tract_key):
    nodes = torch.from_numpy(streamline).float()
    edges = torch.tensor(
        [[i, i+1] for i in range(nodes.size(0)-1)], dtype=torch.long
    ).t().contiguous()
    return Data(x=nodes, edge_index=edges, y=tract_key.clone().detach().long())


def create_tracts_dict(subjects):
    """
    Crear un diccionario donde las claves son los nombres de los tractos y los valores son listas de archivos .trk.
    
    Args:
        subjects (list): Lista de diccionarios, cada uno representando un sujeto y sus tractos.
    
    Returns:
        dict: Diccionario con los tractos como claves y listas de rutas de archivos .trk como valores.
    """
    tracts_dict = defaultdict(list)
    
    for subject in subjects:
        for tract_path in subject['tracts']:
            # Obtener el nombre del tracto a partir del nombre del archivo (sin la extensión)
            tract_name = tract_path.stem
            # Agregar la ruta del archivo a la lista correspondiente en el diccionario
            tracts_dict[tract_name].append(str(tract_path))
    
    return tracts_dict

# Crear una lista de todos los tractos disponibles
def get_all_tracts(tracts_dict):
    return list(tracts_dict.keys())

def fill_missing_tracts(subjects, tracts_dict):
    """
    Completar los tractos faltantes en cada sujeto seleccionando tractos aleatorios de otros sujetos.
    
    Args:
        subjects (list): Lista de diccionarios, cada uno representando un sujeto y sus tractos.
        tracts_dict (dict): Diccionario con los tractos como claves y listas de rutas de archivos .trk como valores.
    
    Returns:
        list: Lista de diccionarios de sujetos con tractos completos.
    """
    all_tracts = get_all_tracts(tracts_dict)
    
    for subject in subjects:
        subject_tract_names = {tract.stem for tract in subject['tracts']}
        missing_tracts = set(all_tracts) - subject_tract_names
        
        for missing_tract in missing_tracts:
            random_tract_path = random.choice(tracts_dict[missing_tract])
            subject['tracts'].append(pathlib.Path(random_tract_path))
    
    return subjects

def fill_tracts_ds(subjects):
    tracts_dict = create_tracts_dict(subjects)
    filled_subjects = fill_missing_tracts(subjects, tracts_dict)
    return filled_subjects

# Seed setting function
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Función para guardar un checkpoint
def save_checkpoint(epoch, model, optimizer, loss, filename='checkpoint.pth'):
    checkpoint_dir = '/app/trained_models'
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, filename))



def get_dataset(dataset_name:str, path:str):
    # Cargar las rutas de los sujetos de entrenamiento, validación y test
    if dataset_name == "HCP_105":
        handler = HCPHandler(path = path, scope = "trainset")
        train_data = handler.get_data()

        handler = HCPHandler(path = path, scope = "validset")
        valid_data = handler.get_data()

        handler = HCPHandler(path = path, scope = "testset")
        test_data = handler.get_data()
        n_classes = 72

    elif dataset_name == "HCP_105_without_CC":
        handler = HCP_Without_CC_Handler(path = path, scope = "trainset")
        train_data = handler.get_data()

        handler = HCP_Without_CC_Handler(path = path, scope = "validset")
        valid_data = handler.get_data()

        handler = HCP_Without_CC_Handler(path = path, scope = "testset")
        test_data = handler.get_data()
        n_classes = 71

    elif dataset_name == "Tractoinferno":
        handler = TractoinfernoHandler(path = path, scope = "trainset")
        train_data = handler.get_data()
        train_data = fill_tracts_ds(train_data)

        handler = TractoinfernoHandler(path = path, scope = "validset")
        valid_data = handler.get_data()

        handler = TractoinfernoHandler(path = path, scope = "testset")
        test_data = handler.get_data()
        n_classes = 32

    elif dataset_name == "FiberCup":
        handler = FiberCupHandler(path = path, scope = "trainset")
        train_data = handler.get_data()

        handler = FiberCupHandler(path = path, scope = "validset")
        valid_data = handler.get_data()

        handler = FiberCupHandler(path = path, scope = "testset")
        test_data = handler.get_data()

    d = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data,
        "handler": handler,
        "n_classes": n_classes
    }
    return d
