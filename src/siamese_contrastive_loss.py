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
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
import torch.optim as optim
from tqdm import tqdm
# Importar BaseTransform para normalización
from torch_geometric.transforms import BaseTransform

#================================================TRAIN DATASET======================================================
class TrainSubjectStreamlinePairDataset(Dataset):
    def __init__(self, datadict:dict, ds_handler, transform=None):
        # Cargar todos los tractos de un sujeto en un diccionario de la forma {tracto: [streamlines], ...}
        self.subject = datadict
        self.transform = transform 
        # Read trk files and store them in a dictionary where the key is the tract name and the value is a list of streamlines
        self.streamlines_by_tract = {}

        for tract in self.subject["tracts"]:
            tractogram = load_trk(str(tract), 'same', bbox_valid_check=False)
            # la key debe ser un entero descrito por ds_handler.get_label_from_tract
            self.streamlines_by_tract[ds_handler.get_label_from_tract(tract.stem)] = tractogram.streamlines

        self.pos_neg_ratio = 2/5

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
        
        if self.transform:
            graph_1 = self.transform(graph_1)
            graph_2 = self.transform(graph_2)

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

#================================================INFERENCE DATASET=====================================================
class TestSubjectStreamlineDataset(Dataset):
    """
    Este dataset es para validación y test y no se necesita balancear las clases ni generar pares positivos y negativos.
    Además, cuando se solicita un ítem, se debe retornar un solo grafo con una sola streamline que no sea al azar.
    Por ejemplo: si hay 3 tractos con 10, 10 y 5 streamlines respectivamente, y se solicita la fibra 12, se debe retornar la fibra 2 del segundo tracto.
    """
    def __init__(self, datadict: dict, ds_handler, transform=None):
        # Cargar todos los tractos de un sujeto en un diccionario de la forma {tracto: [streamlines], ...}
        self.subject = datadict
        self.transform = transform
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
                graph = Data(x = nodes, edge_index = edges, y = torch.tensor(tract, dtype = torch.long))
                if self.transform:
                    graph = self.transform(graph)
                return graph
            cumulative_count += len(streamlines)
        raise IndexError("Index out of range")

#================================================MODEL======================================================
class GraphConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(GraphConvBlock, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.relu(x)
        x = self.bn(x)
        if self.dropout:
            x = self.dropout(x)
        return x
    
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dropout, n_hidden_blocks):
        super(GCNEncoder, self).__init__()
        self.input_block = GraphConvBlock(in_channels, hidden_dim, dropout)
        self.hidden_blocks = nn.ModuleList([GraphConvBlock(hidden_dim, hidden_dim, dropout) for _ in range(n_hidden_blocks - 1)])
        self.output_block = GraphConvBlock(hidden_dim, out_channels, dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_block(x, edge_index)
        for layer in self.hidden_blocks:
            x = layer(x, edge_index)
        x = self.output_block(x, edge_index)
        return global_mean_pool(x, batch) # (batch_size, out_channels)
    
class ProjectionHead(nn.Module):
    """
    Proyección de las embeddings de texto a un espacio de dimensión reducida.
    """
    def __init__(
        self,
        embedding_dim,# Salida del modelo de lenguaje (768)
        projection_dim, # Dimensión de la proyección (256)
        dropout=0.1
    ):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class ClassifierHead(nn.Module):
    """
    Capa FC con activación softmax para que clasifique la clase.
    """
    def __init__(self, projection_dim, n_classes):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Linear(projection_dim, n_classes)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.fc(x)
        return self.softmax(x)

class SiameseGraphNetwork(nn.Module):
    def __init__(self, encoder, projection_head, classifier):
        super(SiameseGraphNetwork, self).__init__()
        self.encoder = encoder
        self.projection_head = projection_head
        self.classifier = classifier

    def forward(self, graph):
        x_1 = self.encoder(graph)
        x_1 = self.projection_head(x_1)
        c1 = self.classifier(x_1)
        return x_1, c1

 
#================================================LOSS FUNCTIONS=====================================================
class ContrastiveLoss(nn.Module):
    """
    Implementación de la pérdida contrastiva como se describe en la imagen proporcionada.
    
    La pérdida contrastiva se define como:
    
    L(w, (y, I_a, I_b)) = (1 - y) * L_S(D_w) + y * L_D(D_w)
    
    donde:
    - y ∈ {0, 1} es un indicador binario que representa si el par de entradas (I_a, I_b) es similar (y = 0) o disimilar (y = 1).
    - L_S(·) es la pérdida para pares similares.
    - L_D(·) es la pérdida para pares disimilares.
    - D_w es la distancia (en este caso, la distancia euclidiana) entre las salidas de la red para las entradas I_a e I_b.
    - La pérdida L_D(·) se define con un margen m tal que L_D = max(m - D_w, 0)^2.
    - La pérdida total para un conjunto P de pares de entrada se define como la suma de las pérdidas para cada par en P.

    L_S(w) = Σ_{i=1}^{|P|} L(w, (y, I_a, I_b)^(i))
    
    Args:
        margin (float): El margen para la pérdida de disimilaridad. Default: 1.0.
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calcula la distancia euclidiana
        euclidean_distance = F.pairwise_distance(output1, output2)

        # Calcula la pérdida de similaridad (LS) y disimilaridad (LD)
        LS = torch.mean((1 - label) * torch.pow(euclidean_distance, 2))
        LD = torch.mean(label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        # Combina LS y LD para obtener la pérdida final
        loss = LS + LD
        return loss



class MultiTaskSiameseLoss(nn.Module):
    """
    Multi-Task Siamese Loss Function.

    La función de pérdida se define como:

        L_SC(w) = sum_{i=1}^{|P|} L(w, (y, I_a, I_b)^(i)) + θ [ L_C(ĉ_a, ζ(I_a))^(i) + L_C(ĉ_b, ζ(I_b))^(i) ]

    donde:
        L(w, (y, I_a, I_b)) es la pérdida de disimilitud,
        L_C es la pérdida de clasificación,
        D_w es la puntuación de similitud,
        θ es un peso para la contribución de la pérdida de clasificación.

    Args:
        classification_weight (float): El peso θ para la pérdida de clasificación. Default: 1.0

    Inputs:
        x_a (Tensor): Embeddings del ejemplo A.
        x_b (Tensor): Embeddings del ejemplo B.
        y (Tensor): Etiquetas indicando si los ejemplos son similares o no.
        class_a (Tensor): Clases predichas del ejemplo A.
        class_b (Tensor): Clases predichas del ejemplo B.
        target_a (Tensor): Clases reales del ejemplo A.
        target_b (Tensor): Clases reales del ejemplo B.

    Output:
        loss (Tensor): El valor del multi-task siamese loss.
    """

    def __init__(self, classification_weight=1.0):
        super(MultiTaskSiameseLoss, self).__init__()
        self.classification_weight = classification_weight
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, x_a, x_b, y, class_a, class_b, target_a, target_b):
        # Calcular la disimilitud (distancia euclidiana)
        D_w = F.pairwise_distance(x_a, x_b, p=2)

        # Calcular la pérdida de disimilitud (loss de margen contrastivo)
        margin_loss = torch.mean((1 - y) * torch.pow(D_w, 2) + y * torch.pow(torch.clamp(1 - D_w, min=0.0), 2))

        # Calcular la pérdida de clasificación (Cross Entropy Loss)
        classification_loss_a = self.cross_entropy_loss(class_a, target_a)
        classification_loss_b = self.cross_entropy_loss(class_b, target_b)

        # Combinar las pérdidas
        loss = margin_loss + self.classification_weight * (classification_loss_a + classification_loss_b)
        return loss

#================================================NORMALIZATION=================================================
class MaxMinNormalization(BaseTransform):
        def __init__(self, max_values=None, min_values=None):
            """
            Initialize the normalization transform with optional max and min values.
            If not provided, they should be computed from the dataset.
            """
            # 76.03170776367188, 77.9359130859375, 88.72427368164062
            # -73.90082550048828, -112.23554992675781, -79.38320922851562
            self.max_values = max_values if max_values is not None else torch.tensor([74.99879455566406, 82.36431884765625, 97.47947692871094], 
                                                                                      dtype=torch.float)
            self.min_values = min_values if min_values is not None else torch.tensor([-76.92510986328125, -120.4773941040039, -81.27867126464844], 
                                                                                      dtype=torch.float)

        def __call__(self, data: Data) -> Data:
            """
            Apply min-max normalization to the node features.
            """
            data.x = (data.x - self.min_values) / (self.max_values - self.min_values)
            return data
        

#================================================TRAINING======================================================
# Bucle de entrenamiento:
MAX_EPOCHS = 1
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
MAX_BATCHES_PER_SUBJECT = 5000

# Cargar las rutas de los sujetos de entrenamiento, validación y test
handler = HCP_handler(path = "/app/dataset/HCP_105", scope = "trainset")
train_data = handler.get_data()

handler = HCP_handler(path = "/app/dataset/HCP_105", scope = "validset")
valid_data = handler.get_data()

handler = HCP_handler(path = "/app/dataset/HCP_105", scope = "testset")
test_data = handler.get_data()


# Crear el modelo, la función de pérdida y el optimizador
model = SiameseGraphNetwork(
    encoder=GCNEncoder(in_channels=3, 
                       hidden_dim=64, 
                       out_channels=512, 
                       dropout=0.5, 
                       n_hidden_blocks=3),
    projection_head=ProjectionHead(embedding_dim=512, 
                                   projection_dim=256),
    classifier=ClassifierHead(projection_dim=256, 
                              n_classes=72)
)

criterion = MultiTaskSiameseLoss(classification_weight=0.7)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

model = model.cuda()
criterion = criterion.cuda()


# Crear las métricas con torchmetrics
accuracy_train = MulticlassAccuracy(num_classes=72, average='macro').cuda()
accuracy_val = MulticlassAccuracy(num_classes=72, average='macro').cuda()

f1_train = MulticlassF1Score(num_classes=72, average='macro').cuda()
f1_val = MulticlassF1Score(num_classes=72, average='macro').cuda()

# Entrenar el modelo
for epoch in range(MAX_EPOCHS):
    model.train()
    for subject in train_data:
        train_dataset = TrainSubjectStreamlinePairDataset(subject, handler, transform=MaxMinNormalization())
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pairs_contrastive)

        for i, (graph_1, graph_2, labels) in enumerate(train_loader):
            graph_1, graph_2, labels = graph_1.to('cuda'), graph_2.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            embedding_1, pred_1 = model(graph_1)
            embedding_2, pred_2 = model(graph_2)
            # loss = criterion(embedding_1, embedding_2, labels)# Si se usa contrastive loss
            loss = criterion(embedding_1, embedding_2, labels, pred_1, pred_2, graph_1.y, graph_2.y)# Si se usa multi-task siamese loss
            loss.backward()
            optimizer.step()

            # Actualizar métricas de entrenamiento
            # Concatenar las predicciones de los dos grafos
            preds = torch.cat((pred_1, pred_2))
            # Concatenar las etiquetas de los dos grafos
            targets = torch.cat((graph_1.y, graph_2.y))


            # Mostar métricas de entrenamiento cada 100 batches
            if i % 50 == 0:
                train_acc = accuracy_train(preds, targets).item()
                train_f1 = f1_train(preds, targets).item()
                print(f"Epoch {epoch+1}/{MAX_EPOCHS} - Loss: {loss.item():.4f} - Batch {i} - Train Accuracy: {train_acc:.4f}, Train F1 Score: {train_f1:.4f}")
                

            if i == MAX_BATCHES_PER_SUBJECT:
                break

    # Imprimir métricas de entrenamiento
    train_acc = accuracy_train.compute().item()
    train_f1 = f1_train.compute().item()
    print(f"Epoch {epoch+1}/{MAX_EPOCHS} - Train Accuracy: {train_acc:.4f}, Train F1 Score: {train_f1:.4f}")
    accuracy_train.reset()
    f1_train.reset()

    # Bucle de validación del modelo
    model.eval()
    for subject in valid_data:
        val_dataset = TestSubjectStreamlineDataset(subject, handler, transform=MaxMinNormalization())
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        with torch.no_grad():
            for i, (graph, labels) in enumerate(val_loader):
                graph, labels = graph.to('cuda'), labels.to('cuda')
                embedding, pred = model(graph)

                # Calcular métricas de validación
                accuracy_val.update(pred, labels)
                f1_val.update(pred, labels)

                print(f"Epoch {epoch+1}/{MAX_EPOCHS} - Batch {i} - Val Accuracy: {accuracy_val.compute().item():.4f}, Val F1 Score: {f1_val.compute().item():.4f}")

    # Imprimir métricas de validación
    val_acc = accuracy_val.compute().item()
    val_f1 = f1_val.compute().item()
    print(f"Epoch {epoch+1}/{MAX_EPOCHS} - Val Accuracy: {val_acc:.4f}, Val F1 Score: {val_f1:.4f}")
    accuracy_val.reset()
    f1_val.reset()
