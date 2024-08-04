"""
Fichero con clases y funciones para manejar datasets de tractografías adaptados a las diferentes arquitecturas.
""" 

from collections import defaultdict
from dipy.io.streamline import load_trk
import pathlib2 as pathlib
import random
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform





#================================================TRIPLET DATASET=====================================================
class StreamlineTripletDataset(Dataset):
    """
    Dataset para entrenar una red siamesa con triplet loss. Se generan tripletas de la forma (anchor, positive, negative).

    Args:
        datadict (dict): Diccionario con la información de los tractos de un sujeto.
        ds_handler (DatasetHandler): Objeto que maneja la información del dataset.
        transform (callable, optional): Transformación a aplicar a cada grafo.
    """

    def __init__(self, datadict: dict, ds_handler, transform=None):
        self.subject = datadict # Subject data dictionary

        self.transform = transform # Transform to apply to each graph

        self.streamlines_by_tract = {# Dictionary with the streamlines of each tract
            ds_handler.get_label_from_tract(tract.stem): load_trk(str(tract), 'same', bbox_valid_check=False).streamlines
            for tract in self.subject["tracts"]
        }

        # Filter out tracts with no streamlines
        self.streamlines_by_tract = {k: v for k, v in self.streamlines_by_tract.items() if len(v) > 0}


    def __len__(self):
        return sum(len(streamlines) for streamlines in self.streamlines_by_tract.values())

    def __getitem__(self, idx):
        # Select a random tract key
        tract_key = random.choice(list(self.streamlines_by_tract.keys()))
        
        # Create graphs for anchor, positive and negative examples
        anchor_graph = create_graph(random.choice(self.streamlines_by_tract[tract_key]), tract_key)

        positive_graph = create_graph(random.choice(self.streamlines_by_tract[tract_key]), tract_key)

        negative_tract_key = random.choice([key for key in self.streamlines_by_tract.keys() if key != tract_key])
        negative_graph = create_graph(random.choice(self.streamlines_by_tract[negative_tract_key]), negative_tract_key)


        if self.transform:
            anchor_graph = self.transform(anchor_graph)
            positive_graph = self.transform(positive_graph)
            negative_graph = self.transform(negative_graph)

        return anchor_graph, positive_graph, negative_graph

def collate_triplet_ds(data_list):
    graphs_anchor, graphs_pos, graphs_neg = zip(*data_list)
    return Batch.from_data_list(graphs_anchor), Batch.from_data_list(graphs_pos), Batch.from_data_list(graphs_neg)


#================================================PAIR DATASET=========================================================
class StreamlinePairDataset(Dataset):
    """
    Dataset para entrenar una red siamesa con pares de datos.
    Genera pares de streamlines que pueden ser del mismo tracto (positivos) o de tractos diferentes (negativos).

    Args:
        datadict (dict): Diccionario con la información de los tractos de un sujeto.
        ds_handler (DatasetHandler): Objeto que maneja la información del dataset.
        transform (callable, optional): Transformación a aplicar a cada grafo.
    """

    def __init__(self, datadict: dict, ds_handler, pos_neg_ratio:float = 2/5, transform=None):
        self.subject = datadict
        self.transform = transform
        self.streamlines_by_tract = {
            ds_handler.get_label_from_tract(tract.stem): load_trk(str(tract), 'same', bbox_valid_check=False).streamlines
            for tract in self.subject["tracts"]
        }
        self.pos_neg_ratio = pos_neg_ratio

    def __len__(self):
        return sum(len(streamlines) for streamlines in self.streamlines_by_tract.values())

    def __getitem__(self, idx):
        tract_key = random.choice(list(self.streamlines_by_tract.keys()))
        graph_1 = create_graph(random.choice(self.streamlines_by_tract[tract_key]), tract_key)

        if random.random() < self.pos_neg_ratio:
            streamline = random.choice(self.streamlines_by_tract[tract_key])
        else:
            tract_key = random.choice([key for key in self.streamlines_by_tract.keys() if key != tract_key])
            streamline = random.choice(self.streamlines_by_tract[tract_key])

        graph_2 = create_graph(streamline, tract_key)

        if self.transform:
            graph_1 = self.transform(graph_1)
            graph_2 = self.transform(graph_2)

        return graph_1, graph_2

def collate_pairs_ds(data_list):
    type_of_pair = torch.tensor(
        [graph_1.y == graph_2.y for graph_1, graph_2 in data_list], 
        dtype=torch.long
    )
    graphs_1, graphs_2 = zip(*data_list)
    return Batch.from_data_list(graphs_1), Batch.from_data_list(graphs_2), type_of_pair

#===========================================SIMCLR DATASET=========================================================
class StreamlineSimCLRDataset(Dataset):
    def __init__(self, datadict: dict, ds_handler, transform=None):
        self.subject = datadict
        self.transform = transform
        self.streamlines_by_tract = {
            ds_handler.get_label_from_tract(tract.stem): load_trk(str(tract), 'same', bbox_valid_check=False).streamlines
            for tract in self.subject["tracts"]
        }

    def __len__(self):
        return sum(len(streamlines) for streamlines in self.streamlines_by_tract.values())
    
    def __getitem__(self, idx):
        # Select a random tract key
        tract_key = random.choice(list(self.streamlines_by_tract.keys()))
        graph_1 = create_graph(random.choice(self.streamlines_by_tract[tract_key]), tract_key)
        graph_2 = create_graph(random.choice(self.streamlines_by_tract[tract_key]), tract_key)

        # Select a random tract key different from the previous one
        while True:
            tract_key_2 = random.choice(list(self.streamlines_by_tract.keys()))
            if tract_key_2 != tract_key:
                break
        
        graph_3 = create_graph(random.choice(self.streamlines_by_tract[tract_key_2]), tract_key_2)
        graph_4 = create_graph(random.choice(self.streamlines_by_tract[tract_key_2]), tract_key_2)

        if self.transform:
            graph_1 = self.transform(graph_1)
            graph_2 = self.transform(graph_2)
            graph_3 = self.transform(graph_3)
            graph_4 = self.transform(graph_4)

        # graph_1 and graph_2 are positive examples, graph_3 and graph_4 are negative examples
        return graph_1, graph_2, graph_3, graph_4
    
def collate_simclr_ds(data_list):
    graphs_1, graphs_2, graphs_3, graphs_4 = zip(*data_list)
    return Batch.from_data_list(graphs_1), Batch.from_data_list(graphs_2), Batch.from_data_list(graphs_3), Batch.from_data_list(graphs_4)


#================================================INFERENCE DATASET=====================================================
class StreamlineTestDataset(Dataset):
    """
    Dataset para validación y test. No se necesita balancear las clases ni generar pares positivos y negativos.
    Retorna un solo grafo con una sola streamline basado en un índice global.
    """

    def __init__(self, datadict: dict, ds_handler, transform=None):
        self.subject = datadict
        self.transform = transform
        self.streamlines_by_tract = {
            ds_handler.get_label_from_tract(tract.stem): load_trk(str(tract), 'same', bbox_valid_check=False).streamlines
            for tract in self.subject["tracts"]
        }

        # Ordenar el diccionario por clave
        self.streamlines_by_tract = dict(sorted(self.streamlines_by_tract.items()))

    def __len__(self):
        return sum(len(streamlines) for streamlines in self.streamlines_by_tract.values())

    def __getitem__(self, idx):
        cumulative_count = 0
        for tract, streamlines in self.streamlines_by_tract.items():
            if cumulative_count + len(streamlines) > idx:
                streamline_idx = idx - cumulative_count
                graph = create_graph(streamlines[streamline_idx], tract)
                if self.transform:
                    graph = self.transform(graph)
                return graph
            cumulative_count += len(streamlines)
        raise IndexError("Index out of range")

def collate_test_ds(data_list):
    return Batch.from_data_list(data_list)
#=======================================TEST DATASET=========================================================
class TestDataset(Dataset):
    def __init__(self, trk_file, ds_handler, transform=None):

        self.trk_file = load_trk(str(trk_file), 'same', bbox_valid_check=False)
        # Compute the bounding box of the tract
        self.trk_file.remove_invalid_streamlines()
        
        self.streamlines = self.trk_file.streamlines
        self.label = ds_handler.get_label_from_tract(trk_file.stem)
        
        self.transform = transform

    def __len__(self):
        return len(self.streamlines)
    
    def __getitem__(self, idx):

        graph = create_graph(self.streamlines[idx], self.label)

        if self.transform:
            graph = self.transform(graph)
        return graph
    

#================================================TRANSFORMS=====================================================
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
        """
        data.x = (data.x - self.min_values) / (self.max_values - self.min_values)
        return data

#================================================UTILS=====================================================

def create_graph(streamline, tract_key):
    nodes = torch.from_numpy(streamline).float()
    edges = torch.tensor(
        [[i, i+1] for i in range(nodes.size(0)-1)] + [[i+1, i] for i in range(nodes.size(0)-1)], dtype=torch.long
    ).t().contiguous()
    return Data(x=nodes, edge_index=edges, y=torch.tensor(tract_key, dtype=torch.long))


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






# class TrainSubjectStreamlineTripletDataset(Dataset):
#     """
#     Dataset para entrenar una red siamesa con triplet loss. Se generan tripletas de la forma (anchor, positive, negative).

#     Args:
#         datadict (dict): Diccionario con la información de los tractos de un sujeto.
#         ds_handler (DatasetHandler): Objeto que maneja la información del dataset.

#     Inputs:
#         idx (int): Índice del ítem a obtener.

#     Outputs:
#         anchor_graph (Data): Grafo con la streamline de la clase seleccionada.
#         positive_graph (Data): Grafo con una streamline del mismo tracto que la de anchor.
#         negative_graph (Data): Grafo con una streamline de un tracto distinto al de anchor.
#     """

#     def __init__(self, datadict:dict, ds_handler, transform=None):
        
#         self.subject = datadict
#         self.transform = transform 
#         self.streamlines_by_tract = {}

#         for tract in self.subject["tracts"]:
#             tractogram = load_trk(str(tract), 'same', bbox_valid_check=False)
#             self.streamlines_by_tract[ds_handler.get_label_from_tract(tract.stem)] = tractogram.streamlines

#     def __len__(self):
#         return np.sum([len(streamlines) for streamlines in self.streamlines_by_tract.values()])
    
#     def __getitem__(self, idx):
#         # Seleccionar una key del diccionario de streamlines al azar (es un entero que representa una clase)
#         tract_key = random.choice(list(self.streamlines_by_tract.keys()))

#         # Seleccionar una streamline al azar del tracto seleccionado y crear un grafo
#         streamline = random.choice(self.streamlines_by_tract[tract_key])
#         nodes = torch.from_numpy(streamline).float()
#         edges = torch.tensor([[i, i+1] for i in range(nodes.size(0)-1)] + [[i+1, i] for i in range(nodes.size(0)-1)], dtype=torch.long).T
#         anchor_graph = Data(x = nodes, edge_index = edges, y = torch.tensor(tract_key, dtype = torch.long))

#         # Seleccionar una streamline al azar del mismo tracto y crear un grafo
#         positive_streamline = random.choice(self.streamlines_by_tract[tract_key])
#         nodes = torch.from_numpy(positive_streamline).float()
#         edges = torch.tensor([[i, i+1] for i in range(nodes.size(0)-1)] + [[i+1, i] for i in range(nodes.size(0)-1)], dtype=torch.long).T
#         positive_graph = Data(x = nodes, edge_index = edges, y = torch.tensor(tract_key, dtype = torch.long))

#         # Seleccionar un tracto distinto al azar
#         while True:
#             negative_tract_key = random.choice(list(self.streamlines_by_tract.keys()))
#             if negative_tract_key != tract_key:
#                 break

#         # Seleccionar una streamline al azar del tracto distinto y crear un grafo
#         negative_streamline = random.choice(self.streamlines_by_tract[negative_tract_key])
#         nodes = torch.from_numpy(negative_streamline).float()
#         edges = torch.tensor([[i, i+1] for i in range(nodes.size(0)-1)] + [[i+1, i] for i in range(nodes.size(0)-1)], dtype=torch.long).T
#         negative_graph = Data(x = nodes, edge_index = edges, y = torch.tensor(negative_tract_key, dtype = torch.long))

#         if self.transform:
#             anchor_graph = self.transform(anchor_graph)
#             positive_graph = self.transform(positive_graph)
#             negative_graph = self.transform(negative_graph)

#         return anchor_graph, positive_graph, negative_graph

# def collate_pairs_triplet(data_list):
#     graphs_anchor, graphs_pos, graphs_neg = zip(*data_list)
#     return Batch.from_data_list(graphs_anchor), Batch.from_data_list(graphs_pos), Batch.from_data_list(graphs_neg)

#================================================INFERENCE DATASET=====================================================
# class TestSubjectStreamlineDataset(Dataset):
#     """
#     Este dataset es para validación y test y no se necesita balancear las clases ni generar pares positivos y negativos.
#     Además, cuando se solicita un ítem, se debe retornar un solo grafo con una sola streamline que no sea al azar.
#     Por ejemplo: si hay 3 tractos con 10, 10 y 5 streamlines respectivamente, y se solicita la fibra 12, se debe retornar la fibra 2 del segundo tracto.
#     """
#     def __init__(self, datadict: dict, ds_handler, transform=None):
#         # Cargar todos los tractos de un sujeto en un diccionario de la forma {tracto: [streamlines], ...}
#         self.subject = datadict
#         self.transform = transform
#         # Read trk files and store them in a dictionary where the key is the tract name and the value is a list of streamlines
#         self.streamlines_by_tract = {}

#         for tract in self.subject["tracts"]:
#             tractogram = load_trk(str(tract), 'same', bbox_valid_check=False)
#             # La key debe ser un entero descrito por ds_handler.get_label_from_tract
#             self.streamlines_by_tract[ds_handler.get_label_from_tract(tract.stem)] = tractogram.streamlines  # Esto es un ArrayList de numpy arrays

#     def __len__(self):
#         return np.sum([len(streamlines) for streamlines in self.streamlines_by_tract.values()])

#     def __getitem__(self, idx):
#         # Encontrar el tracto y la streamline correspondiente dado un índice global
#         cumulative_count = 0
#         for tract, streamlines in self.streamlines_by_tract.items():
#             if cumulative_count + len(streamlines) > idx:
#                 streamline_idx = idx - cumulative_count
#                 # Crear un grafo con la streamline seleccionada
#                 nodes = torch.from_numpy(streamlines[streamline_idx]).float()
#                 edges = torch.tensor([[i, i + 1] for i in range(nodes.size(0) - 1)] + [[i + 1, i] for i in range(nodes.size(0) - 1)], dtype=torch.long).T
#                 graph = Data(x = nodes, edge_index = edges, y = torch.tensor(tract, dtype = torch.long))
#                 if self.transform:
#                     graph = self.transform(graph)
#                 return graph
#             cumulative_count += len(streamlines)
#         raise IndexError("Index out of range")