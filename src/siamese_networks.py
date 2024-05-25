from dataset_handlers import Tractoinferno_handler, HCP_handler
import numpy as np
import random
from dipy.io.streamline import load_trk
import pathlib2 as pathlib
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from torch_geometric.data import Data
from torch.nn import ModuleList
import torch.nn.functional as F

#================================================TRAIN DATASET======================================================
class TrainSubjectStreamlinePairDataset(Dataset):
    def __init__(self, datadict:dict, ds_handler):
        # Cargar todos los tractos de un sujeto en un diccionario de la forma {tracto: [streamlines], ...}
        self.subject = datadict 
        # Read trk files and store them in a dictionary where the key is the tract name and the value is a list of streamlines
        self.streamlines_by_tract = {}

        for tract in self.subject["tracts"]:
            tractogram = load_trk(str(tract), 'same', bbox_valid_check=False)
            # la key debe ser un entero descrito por ds_handler.get_label_from_tract
            self.streamlines_by_tract[ds_handler.get_label_from_tract(tract.stem)] = tractogram.streamlines

        self.pos_neg_ratio = 1/5

    def __len__(self):
        return np.sum([len(streamlines) for streamlines in self.streamlines_by_tract.values()])
    
    def __getitem__(self, idx):
        # Seleccionar una key del diccionario de streamlines al azar (es un entero que representa una clase)
        tract_key = random.choice(list(self.streamlines_by_tract.keys()))

        # Seleccionar una streamline al azar del tracto seleccionado
        streamline = random.choice(self.streamlines_by_tract[tract_key])

        # Crear un grafo con la streamline seleccionada y otro con una streamline del mismo tracto o de otro tracto
        nodes = torch.from_numpy(streamline).float()
        edges = torch.tensor([[i, i+1] for i in range(nodes.size(0)-1)] + [[i+1, i] for i in range(nodes.size(0)-1)], dtype=torch.long).T
        graph_1 = Data(x = nodes, edge_index = edges, y = torch.tensor(tract_key, dtype = torch.long))
        
        # Seleccionar un par positivo o negativo segun la proporción pos_neg_ratio
        same_class = random.random() < self.pos_neg_ratio
        if same_class:
            # Seleccionar una streamline del mismo tracto
            streamline = random.choice(self.streamlines_by_tract[tract_key])
        else:
            while True:
                # Seleccionar un tracto distinto
                other_tract_key = random.choice(list(self.streamlines_by_tract.keys()))
                if other_tract_key != tract_key:
                    break
            
            tract_key = other_tract_key# 
            # Seleccionar una streamline del tracto seleccionado
            streamline = random.choice(self.streamlines_by_tract[other_tract_key])

        # Crear un grafo con la streamline seleccionada
        nodes = torch.from_numpy(streamline).float()
        edges = torch.tensor([[i, i+1] for i in range(nodes.size(0)-1)] + [[i+1, i] for i in range(nodes.size(0)-1)], dtype=torch.long).T
        graph_2 = Data(x = nodes, edge_index = edges, y = torch.tensor(tract_key, dtype = torch.long))

        return graph_1, graph_2

def collate_pairs_contrastive(data_list):
    # data_list es una lista de tuplas (graph_1, graph_2)
    # graph_1 y graph_2 son objetos Data
    type_of_pair = torch.tensor(
        [
            graph_1.y == graph_2.y for graph_1, graph_2 in data_list
        ], 
        dtype=torch.long)
    graphs_1, graphs_2 = zip(*data_list)
    return Batch.from_data_list(graphs_1), Batch.from_data_list(graphs_2), type_of_pair





  

class TrainSubjectStreamlineTripletDataset(Dataset):
    """
    Dataset para entrenar una red siamesa con triplet loss. Se generan tripletas de la forma (anchor, positive, negative).

    Args:
        datadict (dict): Diccionario con la información de los tractos de un sujeto.
        ds_handler (DatasetHandler): Objeto que maneja la información del dataset.

    Inputs:
        idx (int): Índice del ítem a obtener.

    Outputs:
        anchor_graph (Data): Grafo con la streamline de la clase seleccionada.
        positive_graph (Data): Grafo con una streamline del mismo tracto que la de anchor.
        negative_graph (Data): Grafo con una streamline de un tracto distinto al de anchor.
    """

    def __init__(self, datadict:dict, ds_handler):
        # Cargar todos los tractos de un sujeto en un diccionario de la forma {tracto: [streamlines], ...}
        self.subject = datadict 
        # Read trk files and store them in a dictionary where the key is the tract name and the value is a list of streamlines
        self.streamlines_by_tract = {}

        for tract in self.subject["tracts"]:
            tractogram = load_trk(str(tract), 'same', bbox_valid_check=False)
            self.streamlines_by_tract[ds_handler.get_label_from_tract(tract.stem)] = tractogram.streamlines

    def __len__(self):
        return np.sum([len(streamlines) for streamlines in self.streamlines_by_tract.values()])
    
    def __getitem__(self, idx):
        # Seleccionar una key del diccionario de streamlines al azar (es un entero que representa una clase)
        tract_key = random.choice(list(self.streamlines_by_tract.keys()))

        # Seleccionar una streamline al azar del tracto seleccionado y crear un grafo
        streamline = random.choice(self.streamlines_by_tract[tract_key])
        nodes = torch.from_numpy(streamline).float()
        edges = torch.tensor([[i, i+1] for i in range(nodes.size(0)-1)] + [[i+1, i] for i in range(nodes.size(0)-1)], dtype=torch.long).T
        anchor_graph = Data(x = nodes, edge_index = edges, y = torch.tensor(tract_key, dtype = torch.long))

        # Seleccionar una streamline al azar del mismo tracto y crear un grafo
        positive_streamline = random.choice(self.streamlines_by_tract[tract_key])
        nodes = torch.from_numpy(positive_streamline).float()
        edges = torch.tensor([[i, i+1] for i in range(nodes.size(0)-1)] + [[i+1, i] for i in range(nodes.size(0)-1)], dtype=torch.long).T
        positive_graph = Data(x = nodes, edge_index = edges, y = torch.tensor(tract_key, dtype = torch.long))

        # Seleccionar un tracto distinto al azar
        while True:
            negative_tract_key = random.choice(list(self.streamlines_by_tract.keys()))
            if negative_tract_key != tract_key:
                break

        # Seleccionar una streamline al azar del tracto distinto y crear un grafo
        negative_streamline = random.choice(self.streamlines_by_tract[negative_tract_key])
        nodes = torch.from_numpy(negative_streamline).float()
        edges = torch.tensor([[i, i+1] for i in range(nodes.size(0)-1)] + [[i+1, i] for i in range(nodes.size(0)-1)], dtype=torch.long).T
        negative_graph = Data(x = nodes, edge_index = edges, y = torch.tensor(negative_tract_key, dtype = torch.long))

        return anchor_graph, positive_graph, negative_graph
        




#================================================INFERENCE DATASET=====================================================
class TestSubjectStreamlineDataset(Dataset):
    """
    Este dataset es para validación y test y no se necesita balancear las clases ni generar pares positivos y negativos.
    Además, cuando se solicita un ítem, se debe retornar un solo grafo con una sola streamline que no sea al azar.
    Por ejemplo: si hay 3 tractos con 10, 10 y 5 streamlines respectivamente, y se solicita la fibra 12, se debe retornar la fibra 2 del segundo tracto.
    """
    def __init__(self, datadict: dict, ds_handler):
        # Cargar todos los tractos de un sujeto en un diccionario de la forma {tracto: [streamlines], ...}
        self.subject = datadict
        # Read trk files and store them in a dictionary where the key is the tract name and the value is a list of streamlines
        self.streamlines_by_tract = {}

        for tract in self.subject["tracts"]:
            tractogram = load_trk(str(tract), 'same', bbox_valid_check=False)
            # La key debe ser un entero descrito por ds_handler.get_label_from_tract
            self.streamlines_by_tract[ds_handler.get_label_from_tract(tract.stem)] = tractogram.streamlines  # Esto es un ArrayList de numpy arrays

    def __len__(self):
        return np.sum([len(streamlines) for streamlines in self.streamlines_by_tract.values()])

    def __getitem__(self, idx):
        # Encontrar el tracto y la streamline correspondiente dado un índice global
        cumulative_count = 0
        for tract, streamlines in self.streamlines_by_tract.items():
            if cumulative_count + len(streamlines) > idx:
                streamline_idx = idx - cumulative_count
                # Crear un grafo con la streamline seleccionada
                nodes = torch.from_numpy(streamlines[streamline_idx]).float()
                edges = torch.tensor([[i, i + 1] for i in range(nodes.size(0) - 1)] + [[i + 1, i] for i in range(nodes.size(0) - 1)], dtype=torch.long).T
                return Data(x = nodes, edge_index = edges, y = torch.tensor(tract, dtype = torch.long))
            cumulative_count += len(streamlines)
        raise IndexError("Index out of range")

#================================================MODELS=====================================================



class Graph_Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(Graph_Conv_Block, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.LeakyReLU()
        if dropout:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.relu(x)
        x = self.bn(x)
        if self.dropout:
            x = self.dropout(x)
        return x
    
class GCN_Encoder(nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_dim, 
                 out_channels,
                 dropout,
                 n_hidden_blocks
                 ):
        
        super(GCN_Encoder, self).__init__()
        self.input_block = Graph_Conv_Block(in_channels, hidden_dim, dropout)
        self.hidden_blocks = nn.ModuleList([Graph_Conv_Block(hidden_dim, hidden_dim, dropout) for _ in range(n_hidden_blocks - 1)])
        self.output_block = Graph_Conv_Block(hidden_dim, out_channels, dropout)
        

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_block(x, edge_index)
        for layer in self.hidden_blocks:
            x = layer(x, edge_index)
        x = self.output_block(x, edge_index)
        return global_mean_pool(x, batch) # (batch_size, out_channels)


class ClassifierHead(nn.Module):
    """
    Capa FC con activación softmax para que clasifique la clase.
    """
    def __init__(
        self,
        projection_dim, # Dimensión de la proyección (512)
        n_classes # Número de clases a clasificar (32)
    ):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Linear(projection_dim, n_classes)
        # self.softmax = nn.Softmax(dim=1)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.fc(x)
        return self.softmax(x)


class SiameseGraphNNetwork(nn.Module):
    def __init__(self, encoder, classifier):
        super(SiameseGraphNNetwork, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, graph_1, graph_2):
        x_1 = self.encoder(graph_1)
        x_2 = self.encoder(graph_2)
        return self.classifier(x_1), self.classifier(x_2)


#================================================LOSS FUNCTIONS=====================================================



#================================================TRAINING=====================================================
# trinf_handler = Tractoinferno_handler(path = "/app/dataset/Tractoinferno/derivatives", scope = "testset")
# trinf_paths = trinf_handler.get_data()



if __name__ == '__main__':
    #================================================CONFIGURATION=====================================================
    class CFG:
        dataset = "HCP"
        hcp_path = "/app/dataset/HCP_105"
        tractoinferno_path = "/app/dataset/Tractoinferno/tractoinferno_preprocessed_mni"
        graph_encoder_input_channels = 3
        graph_encoder_hidden_channels = 64
        graph_encoder_graph_embedding = 64
        graph_encoder_dropout = 0.5
        graph_encoder_n_hidden_blocks = 2
        projection_head_output_dim = 512
        n_classes = 72
    
    # Seleccionar el dataset
    if CFG.dataset == "Tractoinferno":
        
        handler = Tractoinferno_handler(path = "/app/dataset/Tractoinferno/derivatives", 
                                        scope = "trainset")
        train_data = handler.get_data()

        handler = Tractoinferno_handler(path = "/app/dataset/Tractoinferno/derivatives", 
                                        scope = "validset")
        valid_data = handler.get_data()

        handler = Tractoinferno_handler(path = "/app/dataset/Tractoinferno/derivatives", 
                                        scope = "testset")
        test_data = handler.get_data()

    elif CFG.dataset == "HCP":
        handler = HCP_handler(path = "/app/dataset/HCP_105", 
                              scope = "trainset")
        train_data = handler.get_data()

        handler = HCP_handler(path = "/app/dataset/HCP_105", 
                              scope = "validset")
        valid_data = handler.get_data()

        handler = HCP_handler(path = "/app/dataset/HCP_105", 
                              scope = "testset")
        test_data = handler.get_data()

    else:
        raise ValueError("Invalid dataset")
    
    


    # Seleccionar el loss function
    if CFG.loss_function == "Contrastive":
        loss_function = ContrastiveLoss(margin=CFG.margin)
    elif CFG.loss_function == "Triplet":
        loss_function = TripletLoss(margin=CFG.margin)
    elif CFG.loss_function == "MultiTaskSiamese":
        loss_function = MultiTaskSiameseLoss(classification_weight=CFG.classification_weight)
    elif CFG.loss_function == "InfoNCE":
        loss_function = InfoNCELoss(temperature=CFG.temperature)
    else:
        raise ValueError("Invalid loss function")
    
    # 
    






    ds = TestSubjectStreamlineDataset(train_data[0], handler)

    for i in range(len(ds)):
        print(f'Index {i}')
        print(ds[i])
        
# dl = DataLoader(ds, batch_size = 32, shuffle = True, collate_fn = collate)

# for c, batch in enumerate(dl):
#     print(c)
#     print(batch)
#     if c == 10:
#         break
# for i in range(5):
#     print(f'Sujeto {i}')
#     ds = SubjectDataset(data[i], handler)
#     dl = DataLoader(ds, batch_size = 32, shuffle = True, collate_fn = collate)

#     for c, batch in enumerate(dl):
#         print(c)
#         print(batch)
#         if c == 10:
#             break