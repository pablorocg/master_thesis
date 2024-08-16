# Autor: Pablo Rocamora

import matplotlib.pyplot as plt
import neptune
import numpy as np
import random
import seaborn as sns
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from dataset_handlers import (FiberCupHandler, 
                              HCPHandler, 
                              TractoinfernoHandler, 
                              HCP_Without_CC_Handler)
from encoders import (
    GCNEncoder,
    ProjectionHead
)
from graph_transformer import GraphTransformerEncoder
from loss_functions import MultiTaskTripletLoss
from streamline_datasets import (
    MaxMinNormalization,
    StreamlineTestDataset,
    StreamlineTripletDataset,
    collate_test_ds,
    collate_triplet_ds,
    fill_tracts_ds
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score
)
from tqdm import tqdm
import os
from gcn_encoder_model_v2 import SiameseGraphNetworkGCN_v2
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv,GraphConv, global_mean_pool, BatchNorm
from torch.nn import ModuleList
import torch.nn.functional as F
# Comando para lanzar tensorboard en el navegador local a través del puerto 8888 reenviado por ssh:
# tensorboard --logdir=/app/runs/HCP_105_without_CC_embedding_visualization --host 0.0.0.0 --port 8888


# Enable TensorFloat32 for better performance in matrix multiplication
torch.set_float32_matmul_precision('high')

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




log = True
if log:
    run = neptune.init_run(
        project="pablorocamora/tfm-tractography",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ODA0YzA2NS04MjczLTQyNzItOGE5Mi05ZmI5YjZkMmY3MDcifQ==",
    )

    
    

# Configuration class
class CFG:
    def __init__(self):
        self.seed = 42
        self.max_epochs = 4
        self.batch_size = 1024
        self.learning_rate = 1e-3
        self.max_batches_per_subject = 150# Un buen valor es 500
        self.optimizer = "AdamW"
        self.classification_weight = 1
        self.margin = 1.5
        self.encoder = "GCN"
        self.embedding_projection_dim = 512
        self.dataset = "HCP_105"#"Tractoinferno"

        dataset_paths = {
            "HCP_105": ("/app/dataset/HCP_105", 72),
            "HCP_105_without_CC": ("/app/dataset/HCP_105", 71),
            "Tractoinferno": ("/app/dataset/Tractoinferno/tractoinferno_preprocessed_mni", 32),
            "FiberCup": ("/app/dataset/Fibercup", 7)
        }
        self.ds_path, self.n_classes = dataset_paths.get(self.dataset, (None, None))

    
    
# Initialize configuration
cfg = CFG()
if log:
    writer = SummaryWriter(log_dir=f"runs/{cfg.dataset}_embedding_visualization_contrastive_pretraining", filename_suffix=f"{time.time()}")
    run["config"] = {
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "max_epochs": cfg.max_epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "max_batches_per_subject": cfg.max_batches_per_subject,
        "n_classes": cfg.n_classes,
        "embedding_projection_dim": cfg.embedding_projection_dim,
        "classification_weight": cfg.classification_weight,
        "margin": cfg.margin,
        "encoder": cfg.encoder,
        "optimizer": cfg.optimizer
    }

# Set the seed
seed_everything(cfg.seed)



# Cargar las rutas de los sujetos de entrenamiento, validación y test
if cfg.dataset == "HCP_105":
    handler = HCPHandler(path = cfg.ds_path, scope = "trainset")
    train_data = handler.get_data()

    handler = HCPHandler(path = cfg.ds_path, scope = "validset")
    valid_data = handler.get_data()

    handler = HCPHandler(path = cfg.ds_path, scope = "testset")
    test_data = handler.get_data()

elif cfg.dataset == "HCP_105_without_CC":
    handler = HCP_Without_CC_Handler(path = cfg.ds_path, scope = "trainset")
    train_data = handler.get_data()

    handler = HCP_Without_CC_Handler(path = cfg.ds_path, scope = "validset")
    valid_data = handler.get_data()

    handler = HCP_Without_CC_Handler(path = cfg.ds_path, scope = "testset")
    test_data = handler.get_data()

elif cfg.dataset == "Tractoinferno":
    handler = TractoinfernoHandler(path = cfg.ds_path, scope = "trainset")
    train_data = handler.get_data()
    train_data = fill_tracts_ds(train_data)# Hacer que todos los sujetos tengan el mismo número de tractos 

    handler = TractoinfernoHandler(path = cfg.ds_path, scope = "validset")
    valid_data = handler.get_data()

    handler = TractoinfernoHandler(path = cfg.ds_path, scope = "testset")
    test_data = handler.get_data()

elif cfg.dataset == "FiberCup":
    handler = FiberCupHandler(path = cfg.ds_path, scope = "trainset")
    train_data = handler.get_data()

    handler = FiberCupHandler(path = cfg.ds_path, scope = "validset")
    valid_data = handler.get_data()

    handler = FiberCupHandler(path = cfg.ds_path, scope = "testset")
    test_data = handler.get_data()





#=========================MODELO================================
# class GraphConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, dropout=0.0):
#         super(GraphConvBlock, self).__init__()
#         self.conv = GCNConv(in_channels, out_channels)
#         self.bn = BatchNorm(out_channels)
#         self.relu = nn.LeakyReLU()
#         self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

#     def forward(self, x, edge_index):
#         x = self.conv(x, edge_index)
#         x = self.bn(x)
#         x = self.relu(x)
#         if self.dropout:
#             x = self.dropout(x)
#         return x
    

# class GCNEncoder(nn.Module):
#     def __init__(self, in_channels, hidden_dim, out_channels, dropout, n_hidden_blocks):
#         super(GCNEncoder, self).__init__()
#         self.input_block = GraphConvBlock(in_channels, hidden_dim, dropout)
#         self.hidden_blocks = ModuleList([GraphConvBlock(hidden_dim, hidden_dim, dropout) for _ in range(n_hidden_blocks - 1)])
#         self.output_block = GraphConvBlock(hidden_dim, out_channels, dropout)
#         self.bn = BatchNorm(out_channels)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         x = self.input_block(x, edge_index)
#         for layer in self.hidden_blocks:
#             x = layer(x, edge_index)
#         x = self.output_block(x, edge_index)
#         x = self.bn(x)

#         return global_mean_pool(x, batch) # (batch_size, out_channels)


# class ProjectionHead(nn.Module):
#     """
#     Proyección de las embeddings de texto a un espacio de dimensión reducida.
#     """
#     def __init__(
#         self,
#         embedding_dim,# Salida del modelo de lenguaje (768)
#         projection_dim, # Dimensión de la proyección (256)
#         # dropout=0.1
#     ):
#         super(ProjectionHead, self).__init__()
#         self.projection = nn.Linear(embedding_dim, projection_dim)
#         self.gelu = nn.GELU()
#         self.fc = nn.Linear(projection_dim, projection_dim)
#         # self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(projection_dim)

#     def forward(self, x):
#         projected = self.projection(x)
#         x = self.gelu(projected)
#         x = self.fc(x)
#         # x = self.dropout(x)
#         x = x + projected
#         x = self.layer_norm(x)
#         return x
    

# class SiameseContrastiveGraphNetwork(nn.Module):
#     def __init__(self, encoder, projection_head):
#         super(SiameseContrastiveGraphNetwork, self).__init__()
#         self.encoder = encoder
#         self.projection_head = projection_head

#     def forward(self, graph):
#         x_1 = self.encoder(graph)
#         x_1 = self.projection_head(x_1)
#         x1_norm = F.normalize(x_1, p=2, dim=1)
#         return x1_norm

# # ========================================================================





# model = SiameseContrastiveGraphNetwork(
#     encoder = GCNEncoder(
#         in_channels = 3, 
#         hidden_dim = 128, 
#         out_channels = 512, 
#         dropout = 0.15, 
#         n_hidden_blocks = 2
#     ),

#     projection_head = ProjectionHead(
#         embedding_dim = 512, 
#         projection_dim = 128
#     )
# ).cuda()
# model = torch.compile(model, dynamic=True)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_mean_pool, TopKPooling

# class GraphEncoderWithPooling(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GraphEncoderWithPooling, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.pool1 = TopKPooling(hidden_channels, ratio=0.8)  # Manteniendo el 80% de los nodos
#         self.bn1 = BatchNorm(hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         self.pool2 = TopKPooling(hidden_channels, ratio=0.8)  # Manteniendo el 50% de los nodos
#         self.fc = nn.Linear(hidden_channels, out_channels)
        
        

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch

#         # Primera capa de convolución y pooling
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.bn1(x)
        
#         x, edge_index, edge_attr, batch, perm, score = self.pool1(x, edge_index, batch=batch)
        
#         # Segunda capa de convolución y pooling
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = self.bn1(x)
        
#         x, edge_index, edge_attr, batch, perm, score = self.pool2(x, edge_index, batch=batch)

#         # Pooling global (mean pooling)
#         x = global_mean_pool(x, batch)

#         # Capa lineal para obtener el embedding final
#         x = self.fc(x)
#         x = F.normalize(x, p=2, dim=1)

#         return x


# model = GraphEncoderWithPooling(
#     in_channels = 3,
#     hidden_channels = 512,
#     out_channels = 512
# ).cuda()
# model = torch.compile(model, dynamic=True)
import torch
from torch import Tensor
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.utils import add_self_loops, remove_self_loops, to_torch_csr_tensor

class GraphUNetEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, depth, n_classes,
                 pool_ratios = 0.5, act = 'relu'):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.depth = depth
        self.pool_ratios = [pool_ratios] * depth if isinstance(pool_ratios, float) else pool_ratios
        self.act = torch.nn.functional.relu if act == 'relu' else act

        channels = hidden_channels

        # Definimos las capas de downsampling (convolución y pooling)
        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(GCNConv(channels, channels, improved=True))
        
        self.classifier = torch.nn.Linear(channels, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()

    def forward(self, data) -> Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

        # global pooling
        embedding = global_mean_pool(x, batch)
        # normalización L2
        embedding = F.normalize(embedding, p=2, dim=-1)

        classifier_output = self.classifier(embedding)
        classifier_output = F.log_softmax(classifier_output, dim=1)
        
        return embedding, classifier_output  # Este tensor es el embedding de los nodos después del proceso de downsampling

    def augment_adj(self, edge_index: Tensor, edge_weight: Tensor, num_nodes: int):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=num_nodes)
        adj = to_torch_csr_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes))
        adj = (adj @ adj).to_sparse_coo()
        edge_index, edge_weight = adj.indices(), adj.values()
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, {self.hidden_channels}, '
                f'depth={self.depth}, pool_ratios={self.pool_ratios})')


model = GraphUNetEncoder(in_channels=3, hidden_channels=128, depth=3, n_classes=cfg.n_classes).cuda()


from loss_functions import MultiTaskTripletLoss
# Definir loss function, optimizador y scheduler

criterion = MultiTaskTripletLoss(
    classification_weight = cfg.classification_weight,
    margin = cfg.margin
).cuda()

# criterion = nn.TripletMarginLoss(
#     margin=cfg.margin, 
#     p=2.0
# ).cuda()

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr = cfg.learning_rate, # 1e-3
    weight_decay = 1e-4
)


# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
#                                            mode='min', 
#                                            factor=0.9, 
#                                            patience=5, 
#                                            verbose=True)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
#                                                        T_max=100, #50
#                                                        eta_min=0,
#                                                        verbose=True)



# Crear las métricas con torchmetrics
# subj_accuracy_train = MulticlassAccuracy(num_classes = cfg.n_classes, average='macro').cuda()
# subj_accuracy_val = MulticlassAccuracy(num_classes = cfg.n_classes, average='macro').cuda()

# subj_f1_train = MulticlassF1Score(num_classes = cfg.n_classes, average='macro').cuda()
# subj_f1_val = MulticlassF1Score(num_classes = cfg.n_classes, average='macro').cuda()

# subj_auroc_train = MulticlassAUROC(num_classes = cfg.n_classes).cuda()
# subj_auroc_val = MulticlassAUROC(num_classes = cfg.n_classes).cuda()

# subj_confusion_matrix = MulticlassConfusionMatrix(num_classes = cfg.n_classes).cuda()


# Definir loss function, optimizador y scheduler


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer = optimizer,
    mode = 'min', 
    factor = 0.7, 
    patience = 20, 
    verbose = True
)

from torch_geometric.transforms import Compose, ToUndirected, NormalizeFeatures, AddSelfLoops, GCNNorm
from torch_geometric.transforms import BaseTransform
from streamline_datasets import Cartesian2SphericalCoords



transforms = Compose([
    MaxMinNormalization(dataset = cfg.dataset),
    # AddSelfLoops(), 
    ToUndirected(),
    # Cartesian2SphericalCoords(), 
    # MaxMinNormalization(dataset = cfg.dataset),
    GCNNorm()
    
])


# Entrenar el modelo
for epoch in range(cfg.max_epochs):

    model.train()# Establecer el modelo en modo de entrenamiento
    
    # Iterar sobre los sujetos de entrenamiento
    for idx_suj, subject in enumerate(train_data):

        total_loss = 0

        # Crear el dataset y el dataloader
        train_ds = StreamlineTripletDataset(
            datadict = subject, 
            ds_handler = handler, 
            transform = transforms
        )
        
        train_dl = DataLoader(
            dataset = train_ds, 
            batch_size = cfg.batch_size, 
            shuffle = True, 
            num_workers = 4,
            collate_fn = collate_triplet_ds
        )
        
        # Bucle de entrenamiento del modelo
        prog_bar = tqdm(
            iterable = train_dl, 
            total = cfg.max_batches_per_subject
        )

        for i, (graph_anch, graph_pos, graph_neg) in enumerate(prog_bar):
            
            # Enviar a la gpu
            graph_anch = graph_anch.to('cuda')
            graph_pos = graph_pos.to('cuda')
            graph_neg = graph_neg.to('cuda')
   

            # Reiniciar los gradientes
            optimizer.zero_grad()

            # Forward pass
            embedding_a, pred_a = model(graph_anch)# Batch size necesario para calcular la media de los embeddings
            embedding_p, pred_p = model(graph_pos)
            embedding_n, pred_n = model(graph_neg)

            # Calcular la pérdida
            loss = criterion(embedding_a, 
                             embedding_p, 
                             embedding_n,
                             pred_a, 
                             pred_p, 
                             pred_n,
                             graph_anch.y, 
                             graph_pos.y, 
                             graph_neg.y)
                             

            total_loss += loss.item()
            
            # Backward pass
            loss.backward()

            # Actualizar los pesos
            optimizer.step()

            # Mostar métricas de entrenamiento cada 100 batches
            if i % 50 == 0:
                print(f"[TRAIN] Epoch {epoch+1}/{cfg.max_epochs} - Subj {idx_suj} - Batch {i} - Loss: {loss.item():.4f}")
                
                if log:# Loggear las métricas de batch
                    run["train/batch/loss"].log(loss.item())

            # Condicion de parada del bucle
            if i == cfg.max_batches_per_subject:
                break
        
        avg_loss = total_loss / cfg.max_batches_per_subject

        # Actualizar el learning rate
        scheduler.step(avg_loss)

        # Log del lr actual
        if log:
            run["train/lr"].log(optimizer.param_groups[0]['lr'])
            run["train/subject/loss"].log(avg_loss)

    # Save the model checkpoint
    checkpoint_name = f'checkpoint_{cfg.dataset}_v2_{epoch}_infonce_{avg_loss:.4f}.pth'
    save_checkpoint(epoch, model, optimizer, loss, filename=checkpoint_name)

# # Entrenar el modelo
# for epoch in range(cfg.max_epochs):

#     model.train()# Establecer el modelo en modo de entrenamiento
    
#     # Iterar sobre los sujetos de entrenamiento
#     for idx_suj, subject in enumerate(train_data):

#         # Crear el dataset y el dataloader
#         train_ds = StreamlineTripletDataset(
#             datadict = subject, 
#             ds_handler = handler, 
#             transform = MaxMinNormalization(dataset = cfg.dataset)
#         )
        
#         train_dl = DataLoader(
#             dataset = train_ds, 
#             batch_size = cfg.batch_size, 
#             shuffle = True, num_workers = 2,
#             collate_fn = collate_triplet_ds
#         )
        
#         # Bucle de entrenamiento del modelo
#         prog_bar = tqdm(
#             iterable = train_dl, 
#             total = cfg.max_batches_per_subject
#         )

#         for i, (graph_anch, graph_pos, graph_neg) in enumerate(prog_bar):

#             # Enviar a la gpu
#             graph_anch = graph_anch.to('cuda')
#             graph_pos = graph_pos.to('cuda')
#             graph_neg = graph_neg.to('cuda')
   

#             # Reiniciar los gradientes
#             optimizer.zero_grad()

#             # Forward pass
#             embedding_a = model(graph_anch)# Batch size necesario para calcular la media de los embeddings
#             embedding_p = model(graph_pos)
#             embedding_n = model(graph_neg)

#             # Calcular la pérdida
#             loss = criterion(embedding_a, embedding_p, embedding_n)
            
#             # Backward pass
#             loss.backward()

#             # Actualizar los pesos
#             optimizer.step()

#             # Mostar métricas de entrenamiento cada 100 batches
#             if i % 50 == 0:
#                 print(f"[TRAIN] Epoch {epoch+1}/{cfg.max_epochs} - Subj {idx_suj} - Batch {i} - Loss: {loss.item():.4f}")
                
#                 if log:# Loggear las métricas de batch
#                     run["train/batch/loss"].log(loss.item())

#             # Condicion de parada del bucle
#             if i == cfg.max_batches_per_subject:
#                 break
        
#         # Actualizar el learning rate
#         # scheduler.step()

#     # Save the model checkpoint
#     checkpoint_name = f'checkpoint_{cfg.dataset}_{cfg.encoder}_{cfg.embedding_projection_dim}_{epoch}.pth'
#     save_checkpoint(epoch, model, optimizer, loss, filename=checkpoint_name)















#     # Fase de validación del modelo
#     model.eval()
#     for idx_val, subject in enumerate(valid_data):
        
#         val_ds = StreamlineTestDataset(
#             subject, 
#             handler, 
#             transform = MaxMinNormalization(dataset = cfg.dataset)
#         )
        
#         val_dl = DataLoader(
#             val_ds, 
#             batch_size = cfg.batch_size, 
#             shuffle = False, 
#             num_workers = 1,
#             collate_fn = collate_test_ds
#         )
        
#         # Diccionario para guardar los embeddings de los grafos por clase cuyas claves son las etiquetas 0 - 71
#         embeddings_list_by_class = defaultdict(list)
#         max_embeddings_per_class = 100
        

#         with torch.no_grad():
#             for i, graph in enumerate(val_dl):
                
#                 # Enviar a la gpu
#                 graph = graph.to('cuda')
#                 target = graph.y

#                 # Forward pass
#                 embedding = model(graph)# Batch size necesario para calcular la media de los embeddings


#                 # Guardar los embeddings y etiquetas
#                 if idx_val == 0:

#                     embedding = embedding.cpu().numpy()
#                     target = graph.y.cpu().numpy()

#                     # Guardar embeddings y etiquetas en el diccionario por clase
#                     for emb, label in zip(embedding, target):
#                         if len(embeddings_list_by_class[label]) < max_embeddings_per_class:
#                             embeddings_list_by_class[label].append(emb)
                    
#                 # if i % 25 == 0:
#                 #     print(f"[VAL] Epoch {epoch+1}/{cfg.max_epochs} - Subj {idx_val} - Batch {i} - Acc.: {subj_accuracy_val.compute().item():.4f}, F1: {subj_f1_val.compute().item():.4f}, AUROC: {subj_auroc_val.compute().item():.4f}")

#             # Para el idx_val = 1, guardar los embeddings de los grafos con sus etiquetas para visualizarlos en TensorBoard Projector
#             if idx_val == 0:

#                 # Preparar los datos para TensorBoard
#                 all_embeddings = []
#                 all_labels = []

#                 for label, embeddings in embeddings_list_by_class.items():
#                     all_embeddings.extend(embeddings)
#                     # Obtener la etiqueta textual de la clase a través del handler
#                     label = handler.get_tract_from_label(label)
#                     print(label)
#                     all_labels.extend([label] * len(embeddings))

#                 # Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at /opt/conda/conda-bld/pytorch_1704987394225/work/torch/csrc/utils/tensor_new.cpp:275.)
#                 # Convertir listas a numpy.ndarray
#                 all_embeddings = np.array(all_embeddings)
#                 all_labels = np.array(all_labels)

#                 # Convertir a tensores
#                 all_embeddings = torch.tensor(all_embeddings)
#                 all_labels = all_labels.tolist()  # TensorBoard necesita etiquetas como lista de strings


#                 # Guardar los embeddings y etiquetas en TensorBoard
#                 writer.add_embedding(
#                     all_embeddings, 
#                     metadata = all_labels, 
#                     global_step = epoch,
#                     tag = f"{cfg.dataset}_{cfg.encoder}_{cfg.embedding_projection_dim}_{time.time()}"
#                 ) 

            


                

#         if idx_val == 1:
#             break

# # Cerrar el SummaryWriter
# writer.close()

                   
