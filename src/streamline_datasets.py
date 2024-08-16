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
import numpy as np
import os
from dipy.tracking.streamline import select_random_set_of_streamlines
from utils import create_graph

# SINGLE DATASET CLASS

# class StreamlineSingleDataset(Dataset):
#     def __init__(self, datadict: dict, ds_handler, transform=None, select_n_streamlines=1000):
        
#         self.subject = datadict # Subject data dictionary
#         self.transform = transform # Transform to apply to each graph
#         self.streamlines = ArraySequence()
#         self.labels = []

#         # Cargar los tractos, eliminar streamlines inválidas y construir las estructuras en un solo bucle
#         for tract in self.subject["tracts"]:
#             tract_label = ds_handler.get_label_from_tract(tract.stem)
#             if select_n_streamlines is not None and select_n_streamlines >= 0:
#                 tractogram = load_trk(str(tract), 'same', bbox_valid_check=False)
#                 tractogram.remove_invalid_streamlines()
#                 tractogram.streamlines = select_random_set_of_streamlines(tractogram.streamlines, select_n_streamlines)
#             else:

#                 # Cargar el tracto y remover las streamlines inválidas
#                 tractogram = load_trk(str(tract), 'same', bbox_valid_check=False)
#                 tractogram.remove_invalid_streamlines()  # Remover streamlines inválidas
            
#             # Extender las streamlines y etiquetas en una sola pasada
#             self.streamlines.extend(tractogram.streamlines)
#             self.labels.extend([tract_label] * len(tractogram.streamlines))

#         # Convertir self.labels a un array de numpy
#         self.labels = np.array(self.labels)  

#     def __len__(self):
#         return len(self.streamlines)

#     def __getitem__(self, idx):
#         # Select streamline in idx and its label
#         streamline = self.streamlines[idx]
#         label = self.labels[idx]

#         # Create graph
#         graph = create_graph(streamline, label)

#         if self.transform:
#             graph = self.transform(graph)
#         return graph






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
    
    graphs_anchor = Batch.from_data_list(graphs_anchor)
    graphs_pos = Batch.from_data_list(graphs_pos)
    graphs_neg = Batch.from_data_list(graphs_neg)

    return graphs_anchor, graphs_pos, graphs_neg

















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

        # Unir todas las streamlines en un solo array de x 
        self.streamlines = ArraySequence()
        for tract, streamlines in self.streamlines_by_tract.items():
            self.streamlines.extend(streamlines)
            
        self.labels = np.concatenate([[label]*len(streamlines) for label, streamlines in self.streamlines_by_tract.items()])



        # Ordenar el diccionario por clave
        # self.streamlines_by_tract = dict(sorted(self.streamlines_by_tract.items()))

    def __len__(self):
        return len(self.streamlines)
    #sum(len(streamlines) for streamlines in self.streamlines_by_tract.values())

    def __getitem__(self, idx):
        # Select streamline in idx and its label
        streamline = self.streamlines[idx]
        label = self.labels[idx]

        # Create graph for the anchor example
        graph = create_graph(streamline, label)

        if self.transform:
            graph = self.transform(graph)
        return graph
        # cumulative_count = 0
        # for tract, streamlines in self.streamlines_by_tract.items():
        #     if cumulative_count + len(streamlines) > idx:
        #         streamline_idx = idx - cumulative_count
        #         graph = create_graph(streamlines[streamline_idx], tract)
        #         if self.transform:
        #             graph = self.transform(graph)
        #         return graph
        #     cumulative_count += len(streamlines)
        # raise IndexError("Index out of range")

def collate_test_ds(data_list):
    return Batch.from_data_list(data_list)


#==================================================================================================
# class StreamlineSingleDataset(Dataset):
#     """
#     Dataset para entrenar una red siamesa con triplet loss. Se generan tripletas de la forma (anchor, positive, negative).

#     Args:
#         datadict (dict): Diccionario con la información de los tractos de un sujeto.
#         ds_handler (DatasetHandler): Objeto que maneja la información del dataset.
#         transform (callable, optional): Transformación a aplicar a cada grafo.
#     """

#     def __init__(self, datadict: dict, ds_handler, transform=None):
#         self.subject = datadict # Subject data dictionary

#         self.transform = transform # Transform to apply to each graph

#         self.streamlines_by_tract = {# Dictionary with the streamlines of each tract
#             ds_handler.get_label_from_tract(tract.stem): load_trk(str(tract), 'same', bbox_valid_check=False).streamlines
#             for tract in self.subject["tracts"]
#         }

#         # Filter out tracts with no streamlines
#         self.streamlines_by_tract = {k: v for k, v in self.streamlines_by_tract.items() if len(v) > 0}


#     def __len__(self):
#         return sum(len(streamlines) for streamlines in self.streamlines_by_tract.values())

#     def __getitem__(self, idx):
#         # Select a random tract key
#         tract_key = random.choice(list(self.streamlines_by_tract.keys()))
        
#         # Create graphs for anchor, positive and negative examples
#         anchor_graph = create_graph(random.choice(self.streamlines_by_tract[tract_key]), tract_key)


#         if self.transform:
#             anchor_graph = self.transform(anchor_graph)


#         return anchor_graph

# def collate_triplet_ds(data_list):
#     graphs_anchor = data_list
#     return Batch.from_data_list(graphs_anchor)





#=======================================TEST DATASET=========================================================
class TestDataset(Dataset):
    def __init__(self, trk_file, ds_handler, transform=None):

        self.trk_file = load_trk(str(trk_file), 'same', bbox_valid_check=False)
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
    




from nibabel.streamlines import ArraySequence
class StreamlineTripletDataset_v2(Dataset):
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
        self.streamlines = ArraySequence()
        self.labels = []

        # Cargar los tractos, eliminar streamlines inválidas y construir las estructuras en un solo bucle
        for tract in self.subject["tracts"]:
            tract_label = ds_handler.get_label_from_tract(tract.stem)
            
            # Cargar el tracto y remover las streamlines inválidas
            tractogram = load_trk(str(tract), 'same', bbox_valid_check=False)
            tractogram.remove_invalid_streamlines()  # Remover streamlines inválidas
            
            # Extender las streamlines y etiquetas en una sola pasada
            self.streamlines.extend(tractogram.streamlines)
            self.labels.extend([tract_label] * len(tractogram.streamlines))

        # Convertir self.labels a un array de numpy
        self.labels = np.array(self.labels)
        # self.streamlines_by_tract = {# Dictionary with the streamlines of each tract
        #     ds_handler.get_label_from_tract(tract.stem): load_trk(str(tract), 'same', bbox_valid_check=False).streamlines
        #     for tract in self.subject["tracts"]
        # }
        

        # # Unir todas las streamlines en un solo objeto ArraySequence
        # # Unir todas las streamlines en un ArrayList()

        # self.streamlines = ArraySequence()
        # for tract, streamlines in self.streamlines_by_tract.items():
        #     self.streamlines.extend(streamlines)

        # self.labels = np.concatenate([[label]*len(streamlines) for label, streamlines in self.streamlines_by_tract.items()])


    def __len__(self):
        return len(self.streamlines)
    
    def __getitem__(self, idx):
        # Select streamline in idx and its label
        streamline = self.streamlines[idx]
        label = self.labels[idx]

        # Create graph for the anchor example
        anchor_graph = create_graph(streamline, label)

        # Select a positive example from the same tract
        positive_streamline = random.choice(self.streamlines_by_tract[label])
        positive_graph = create_graph(positive_streamline, label)

        # Select a negative example from a different tract
        negative_label = random.choice([key for key in self.streamlines_by_tract.keys() if key != label])
        negative_streamline = random.choice(self.streamlines_by_tract[negative_label])
        negative_graph = create_graph(negative_streamline, negative_label)

        if self.transform:
            anchor_graph = self.transform(anchor_graph)
            positive_graph = self.transform(positive_graph)
            negative_graph = self.transform(negative_graph)

        return anchor_graph, positive_graph, negative_graph


def collate_triplet_ds(data_list):
    graphs_anchor, graphs_pos, graphs_neg = zip(*data_list)
    
    graphs_anchor = Batch.from_data_list(graphs_anchor)
    graphs_pos = Batch.from_data_list(graphs_pos)
    graphs_neg = Batch.from_data_list(graphs_neg)

    return graphs_anchor, graphs_pos, graphs_neg