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
from torch_geometric.transforms import Compose
from dataset_handlers import (FiberCupHandler, 
                              HCPHandler, 
                              TractoinfernoHandler, 
                              HCP_Without_CC_Handler)
from encoders import (
    ClassifierHead,
    GATEncoder,
    GCNEncoder,
    ProjectionHead,
    SiameseGraphNetwork
)
from graph_transformer import GraphTransformerEncoder
from loss_functions import MultiTaskTripletLoss
from streamline_datasets import (
    MaxMinNormalization,
    CartesianToPolar,
    StreamlineTestDataset,
    StreamlineTripletDataset_v2,
    StreamlineTripletDataset,
    collate_test_ds,
    collate_triplet_ds,
    fill_tracts_ds,
    save_checkpoint,
    seed_everything
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

# Comando para lanzar tensorboard en el navegador local a través del puerto 8888 reenviado por ssh:
# tensorboard --logdir=runs/embedding_visualization --host 0.0.0.0 --port 8888




#==================================CONFIG=======================================
class CFG:
    def __init__(self):
        self.seed = 42
        self.max_epochs = 10
        self.batch_size = 1024
        self.learning_rate = 1e-3
        self.max_batches_per_subject = 150# Un buen valor es 500
        self.optimizer = "AdamW"
        self.classification_weight = 0.04
        self.margin = 1.0
        self.encoder = "GCNEncoder_v2"
        self.embedding_projection_dim = 512
        self.dataset = "HCP_105"#"Tractoinferno"

        dataset_paths = {
            "HCP_105": ("/app/dataset/HCP_105", 72),
            "HCP_105_without_CC": ("/app/dataset/HCP_105", 71),
            "Tractoinferno": ("/app/dataset/Tractoinferno/tractoinferno_preprocessed_mni", 32),
            "FiberCup": ("/app/dataset/Fibercup", 7)
        }
        self.ds_path, self.n_classes = dataset_paths.get(self.dataset, (None, None))


       
       
torch.set_float32_matmul_precision('high')
cfg = CFG()
seed_everything(cfg.seed)    
log = True
#===============================================================================


#==================================LOGGING======================================
if log:
    run = neptune.init_run(
        project="pablorocamora/tfm-tractography",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ODA0YzA2NS04MjczLTQyNzItOGE5Mi05ZmI5YjZkMmY3MDcifQ==",
    )

    writer = SummaryWriter(
        log_dir=f"runs/{cfg.dataset}_embedding_visualization", 
        filename_suffix=f"{time.time()}"
    )

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
#===============================================================================


#==================================DATASET======================================
# Cargar las rutas de los sujetos de entrenamiento, validación y test
if cfg.dataset == "HCP_105":
    handler = HCPHandler(path = cfg.ds_path, scope = "trainset")
    train_data = handler.get_data()

    handler = HCPHandler(path = cfg.ds_path, scope = "validset")
    valid_data = handler.get_data()

    handler = HCPHandler(path = cfg.ds_path, scope = "testset")
    test_data = handler.get_data()

    class_weights =[0.00093538, 0.00133498, 0.00361582, 0.00346234, 0.06984161, 0.00137886,
                    0.00657724, 0.00141732, 0.008168  , 0.00233705, 0.00236711, 0.00263968,
                    0.00315052, 0.00140918, 0.00163606, 0.00973895, 0.01035299, 0.00589177,
                    0.00659963, 0.19821707, 0.26326884, 0.0130849 , 0.01460528, 0.00708661,
                    0.00518798, 0.01128727, 0.01429332, 0.00128066, 0.00131398, 0.00187007,
                    0.00824471, 0.01419559, 0.00564152, 0.00569818, 0.01403578, 0.0147423,
                    0.0093236 , 0.00590173, 0.00372287, 0.00249854, 0.00451997, 0.00396121,
                    0.03002945, 0.02163625, 0.01022905, 0.01267068, 0.00405242, 0.00442964,
                    0.00095304, 0.00106293, 0.00241588, 0.00288353, 0.00170544, 0.00187174,
                    0.00076665, 0.00088584, 0.01585927, 0.02146223, 0.00540888, 0.0092738,
                    0.00119916, 0.0015675 , 0.00383464, 0.00595095, 0.00237361, 0.00278225,
                    0.00110095, 0.00121093, 0.01575325, 0.01921575, 0.02086654, 0.00971133]


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
    # Hacer que todos los sujetos tengan el mismo número de tractos 
    train_data = fill_tracts_ds(train_data)

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
#===============================================================================















#==================================MODEL========================================

# # Crear el modelo, la función de pérdida y el optimizador
if cfg.encoder == "GAT":# Graph Attention Network
    encoder = GATEncoder(in_channels = 3, 
                         hidden_channels = 16, 
                         out_channels = 512)

elif cfg.encoder == "GCN":# Graph Convolutional Network
    encoder = GCNEncoder(in_channels = 3, 
                         hidden_dim = 128, 
                         out_channels = 128, 
                         dropout = 0.15, 
                         n_hidden_blocks = 2)
    

elif cfg.encoder == "GTE":# Graph Transformer Encoder
    encoder = GraphTransformerEncoder(in_channels = 3,
                                        hidden_channels = 128,
                                        out_channels = 512,
                                        num_heads = 8,
                                        dropout = 0.1)

elif cfg.encoder == "GCNEncoder_v2":
    model = SiameseGraphNetworkGCN_v2(n_classes = cfg.n_classes).cuda()



# model = SiameseGraphNetwork(
#     encoder = encoder,
#     projection_head = ProjectionHead(embedding_dim = 128, 
#                                      projection_dim = cfg.embedding_projection_dim),
#     classifier = ClassifierHead(projection_dim = cfg.embedding_projection_dim, 
#                                 n_classes = cfg.n_classes)
# ).cuda()
#============================================================================



# Nuevo modelo experimental
#============================================================================
# import torch.nn as nn
# from torch_geometric.nn import GCNConv, GATConv, BatchNorm, LayerNorm, global_mean_pool
# import torch
# import torch.nn as nn
# from torch_geometric.nn import GCNConv,GraphConv, global_mean_pool, BatchNorm
# from torch.nn import ModuleList
# import torch.nn.functional as F


# class GraphConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, dropout=0.0):
#         super(GraphConvBlock, self).__init__()
#         self.conv = GCNConv(in_channels, out_channels)
#         self.bn = BatchNorm(out_channels)
#         self.activation = nn.GELU()
#         self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

#     def forward(self, x, edge_index):
#         # Apply graph convolution
#         x = self.conv(x, edge_index)
#         # Apply batch normalization
#         x = self.bn(x)
#         # Apply activation function
#         x = self.activation(x)
#         # Apply dropout if defined
#         if self.dropout:
#             x = self.dropout(x)
#         return x

# class GCNEncoder_v2(nn.Module):
#     def __init__(self, in_channels, hidden_dim, out_channels, dropout=0.0, n_hidden_blocks=2):
#         super(GCNEncoder_v2, self).__init__()
#         self.input_block = GraphConvBlock(in_channels, hidden_dim, dropout)
#         self.hidden_blocks = self._make_hidden_layers(hidden_dim, dropout, n_hidden_blocks)
#         self.output_block = GraphConvBlock(hidden_dim, out_channels, dropout)
#         self.attention_block = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
#         self.layer_norm = LayerNorm(out_channels)

#     def _make_hidden_layers(self, hidden_dim, dropout, n_hidden_blocks):
#         layers = []
#         for _ in range(n_hidden_blocks - 1):
#             layers.append(GraphConvBlock(hidden_dim, hidden_dim, dropout))
#         return nn.ModuleList(layers)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         x = self.input_block(x, edge_index)
#         for layer in self.hidden_blocks:
#             x = layer(x, edge_index)
#         x = self.attention_block(x, edge_index)
#         x = self.output_block(x, edge_index)
#         x = self.layer_norm(x)
#         return global_mean_pool(x, batch)  # (batch_size, out_channels)

# class GraphConvBlock_v2(nn.Module):
#     def __init__(self, in_channels, out_channels, dropout=0.0):
#         super(GraphConvBlock_v2, self).__init__()
#         self.conv = GCNConv(in_channels, out_channels)
#         self.bn = BatchNorm(out_channels)
#         self.activation = nn.GELU()
#         self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
#         self.residual = (in_channels == out_channels)
        
#     def forward(self, x, edge_index):
#         identity = x
#         x = self.conv(x, edge_index)
#         x = self.bn(x)
#         x = self.activation(x)
#         if self.dropout:
#             x = self.dropout(x)
#         if self.residual:
#             x += identity
#         return x


# class ProjectionHead_v2(nn.Module):
#     def __init__(self, in_features, projection_dim):
#         super(ProjectionHead_v2, self).__init__()
#         self.projection = nn.Linear(in_features, projection_dim)
#         self.gelu = nn.GELU()
#         self.fc = nn.Linear(projection_dim, projection_dim)
#         self.layer_norm = nn.LayerNorm(projection_dim)

#     def forward(self, x):
#         x = self.projection(x)
#         x = self.gelu(x)
#         x = self.fc(x)
#         x = self.layer_norm(x)
#         return x
    
# class ClassifierHead_v2(nn.Module):
#     def __init__(self, in_features, num_classes):
#         super(ClassifierHead_v2, self).__init__()
#         self.fc = nn.Linear(in_features, num_classes)
#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, x):
#         x = self.fc(x)
#         x = self.softmax(x)
#         return x

# class SiameseGraphNetwork(nn.Module):
#     def __init__(self, encoder, projection_head, classifier, normalize=True):
#         super(SiameseGraphNetwork, self).__init__()
#         self.encoder = encoder
#         self.projection_head = projection_head
#         self.classifier = classifier
#         self.normalize = normalize

#     def forward(self, graph):
#         x_1 = self.encoder(graph)
#         x_1 = self.projection_head(x_1)
        
#         if self.normalize:
#             x1_norm = F.normalize(x_1, p=2, dim=1)

#         c1 = self.classifier(x1_norm)
#         return x1_norm, c1


# # Crear el modelo
# in_channels = 3  # Número de características de entrada
# hidden_dim = 128  # Dimensión de las capas ocultas
# out_channels = 64  # Dimensión de la salida del encoder
# projection_dim = 128  # Dimensión del embedding proyectado
# num_classes = 10  # Número de clases para la clasificación

# # Arquitectura
# n_hidden_blocks = 3  # Número de bloques ocultos

# # Dropout y Regularización
# dropout = 0.2  # Tasa de dropout


# # Configuración de la Pérdida
# # triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
# model = SiameseGraphNetwork(
#     encoder = GCNEncoder_v2(
#         in_channels = 3, 
#         hidden_dim = 128, 
#         out_channels = 256, # 64
#         dropout = 0.2, 
#         n_hidden_blocks = 3
#     ),

#     projection_head = ProjectionHead_v2(
#         in_features = 64, 
#         projection_dim = 128
#     ),

#     classifier = ClassifierHead_v2(
#         in_features = 128, 
#         num_classes = cfg.n_classes
#     )
# ).cuda()


#============================================================================

# Compile the model into an optimized version:
model = torch.compile(model, dynamic=True)





#=======================LOSS, OPTIMIZADOR y SCHEDULER===========================

# Loss function
criterion = MultiTaskTripletLoss(
    classification_weight = cfg.classification_weight,
    margin = cfg.margin,
    cross_entropy_weight_list=class_weights
).cuda()


# Optimizer
if cfg.optimizer == "Adam":
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr = cfg.learning_rate
    )

elif cfg.optimizer == "AdamW":
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr = cfg.learning_rate,
        weight_decay = 1e-4
    )

elif cfg.optimizer == "SGD":
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr = cfg.learning_rate,
        momentum = 0.9
    )


# Scheduler    
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer = optimizer,
    mode = 'min', 
    factor = 0.5, 
    patience = 15, 
    verbose = True
)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, 
#     T_max=50, 
#     eta_min=0
# )
#===============================================================================


#==================================METRICAS=====================================
subj_accuracy_train = MulticlassAccuracy(
    num_classes = cfg.n_classes, 
    average='macro'
).cuda()

subj_accuracy_val = MulticlassAccuracy(
    num_classes = cfg.n_classes, 
    average='macro'
).cuda()

subj_f1_train = MulticlassF1Score(
    num_classes = cfg.n_classes, 
    average='macro'
).cuda()

subj_f1_val = MulticlassF1Score(
    num_classes = cfg.n_classes, 
    average='macro'
).cuda()

subj_auroc_train = MulticlassAUROC(
    num_classes = cfg.n_classes
).cuda()

# subj_auroc_val = MulticlassAUROC(num_classes = cfg.n_classes).cuda()

subj_confusion_matrix = MulticlassConfusionMatrix(
    num_classes = cfg.n_classes
).cuda()
#===============================================================================



#==================================ENTRENAMIENTO================================
best_val_loss = float('inf')# Inicializar la mejor pérdida de validación

for epoch in range(cfg.max_epochs):

    for idx_suj, subject in enumerate(train_data):# Iterar sobre los sujetos de entrenamiento

        model.train()
        # incluir varios transforms 
        # transforms = [CartesianToPolar(), MaxMinNormalization(dataset=cfg.dataset)]
        # transforms 
        train_ds = StreamlineTripletDataset_v2(
            datadict = subject, 
            ds_handler = handler, 
            transform = Compose([
                MaxMinNormalization(dataset=cfg.dataset), 
                CartesianToPolar()
            ])#
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

        total_loss = 0
        for i, (graph_anch, graph_pos, graph_neg) in enumerate(prog_bar):

            # Enviar a la gpu
            graph_anch = graph_anch.to('cuda')
            graph_pos = graph_pos.to('cuda')
            graph_neg = graph_neg.to('cuda')
   

            # Reiniciar los gradientes
            optimizer.zero_grad()

            # Forward pass
            embedding_1, pred_1 = model(graph_anch)# Batch size necesario para calcular la media de los embeddings
            embedding_2, pred_2 = model(graph_pos)
            embedding_3, pred_3 = model(graph_neg)

            # Calcular la pérdida
            loss = criterion(
                x_anchor = embedding_1,
                x_positive = embedding_2,
                x_negative = embedding_3,
                class_anchor = pred_1,
                class_positive = pred_2,
                class_negative = pred_3,
                target_anchor = graph_anch.y,
                target_positive = graph_pos.y,
                target_negative = graph_neg.y
            )

            total_loss += loss.item()# Acumular la pérdida

            # Backward pass
            loss.backward()

            # Actualizar los pesos
            optimizer.step()

            # Concatenar las predicciones de los dos grafos
            preds = torch.cat(
                (
                    pred_1, 
                    pred_2, 
                    pred_3
                )
            )

            # Concatenar las etiquetas de los dos grafos
            targets = torch.cat(
                (
                    graph_anch.y, 
                    graph_pos.y, 
                    graph_neg.y
                )
            )


            # calcular métricas de entrenamiento
            subj_accuracy_train.update(preds, targets)
            subj_f1_train.update(preds, targets)
            subj_auroc_train.update(preds, targets)


            # Mostar métricas de entrenamiento cada 100 batches
            if i % 50 == 0:
                print(f"[TRAIN] Epoch {epoch+1}/{cfg.max_epochs} - Subj {idx_suj} - Batch {i} - Acc.: {subj_accuracy_train.compute().item():.4f}, F1: {subj_f1_train.compute().item():.4f}")
                
            if log:# Loggear las métricas de batch
                log_dict = criterion.log_dict
                run["train/batch/contrastive_loss"].log(log_dict["margin_loss"])
                run["train/batch/classification_loss"].log(log_dict["weighted_class_loss"])
                
                run["train/batch/loss"].log(loss.item())
                run["train/batch/acc"].log(subj_accuracy_train.compute().item())
                run["train/batch/f1"].log(subj_f1_train.compute().item())
              

            # Condicion de parada del bucle
            if i == cfg.max_batches_per_subject:
                avg_loss = total_loss / cfg.max_batches_per_subject

                if log:# Loggear las métricas de subject
                    run["train/subject/acc"].log(subj_accuracy_train.compute().item())
                    run["train/subject/f1"].log(subj_f1_train.compute().item())
                    run["train/subject/auroc"].log(subj_auroc_train.compute().item())
                    run["train/subject/loss"].log(avg_loss)
                    # Log lr
                    run["train/lr"].log(optimizer.param_groups[0]['lr'])
                
                # Reiniciar las métricas
                subj_accuracy_train.reset()
                subj_f1_train.reset()
                subj_auroc_train.reset()
                
                # Actualizar el learning rate
                scheduler.step(avg_loss)
                break


        # Validación del modelo cada 15 sujetos
        if idx_suj % 15 == 0 and idx_suj != 0:

            total_val_loss = 0 # Inicializar la pérdida total de validación
            model.eval()

            for idx_val, subject in enumerate(valid_data):
                subject_loss = 0
                
                val_ds = StreamlineTestDataset(
                    datadict = subject, 
                    ds_handler = handler, 
                    transform = Compose([
                        MaxMinNormalization(dataset=cfg.dataset), 
                        CartesianToPolar()
                    ])##MaxMinNormalization(dataset = cfg.dataset)
                )
                
                val_dl = DataLoader(
                    dataset = val_ds, 
                    batch_size = cfg.batch_size, 
                    shuffle = False, 
                    num_workers = 1,
                    collate_fn = collate_test_ds
                )

                for i, graph in enumerate(val_dl):
                        
                    # Enviar a la gpu
                    graph = graph.to('cuda')
                    target = graph.y

                    with torch.no_grad():
                        embedding, pred = model(graph)
                    
                    # calcular el loss de validación
                    loss = nn.functional.cross_entropy(pred, target)
                    subject_loss += loss.item()


                    # Calcular métricas de validación
                    subj_accuracy_val.update(pred, target)
                    subj_f1_val.update(pred, target)
                    subj_confusion_matrix.update(pred, target)
                    
                    if log:# Loggear las métricas de batch
                        run["val/batch/acc"].log(subj_accuracy_val.compute().item())
                        run["val/batch/f1"].log(subj_f1_val.compute().item())
                        run["val/batch/loss"].log(loss.item())

                    if i % 25 == 0:
                        print(f"[VAL] Epoch {epoch+1}/{cfg.max_epochs} - Subj {idx_val} - Batch {i} - Acc.: {subj_accuracy_val.compute().item():.4f}, F1: {subj_f1_val.compute().item():.4f}")

                avg_val_loss = subject_loss / len(val_dl)
                total_val_loss += avg_val_loss

                if log:# Loggear las métricas de subject
                    run["val/subject/acc"].log(subj_accuracy_val.compute().item())
                    run["val/subject/f1"].log(subj_f1_val.compute().item())
                    run["val/subject/loss"].log(avg_val_loss)
                   

                    cm = subj_confusion_matrix.compute()
                    # Convertir la matriz de confusión a numpy
                    cm = cm.cpu().numpy()

                    # Visualiza la matriz de confusión y guárdala como imagen
                    plt.figure(figsize=(35, 35))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

                    text_labels = [handler.get_tract_from_label(i) for i in range(cfg.n_classes)]
                    plt.xticks(ticks = range(cfg.n_classes), labels = text_labels, rotation = 90)
                    plt.yticks(ticks = range(cfg.n_classes), labels = text_labels, rotation = 0)
                    plt.xlabel('Predicted Labels')
                    plt.ylabel('True Labels')
                    plt.title(f'Confusion Matrix Subj {idx_val}')
                    plt.tight_layout()

                    # Guarda la imagen
                    img_path = f'/app/confusion_matrix_imgs/confusion_matrix_val_suj{idx_val}.png'
                    plt.savefig(img_path)
                    plt.close()

                    # Sube la imagen a Neptune
                    run["confusion_matrix_fig"].upload(img_path)
                
                subj_accuracy_val.reset()
                subj_f1_val.reset()
                subj_confusion_matrix.reset()

                # Si es el último sujeto de validación, comprobar si es el mejor modelo y guardarlo
                if idx_val == 4:

                    if total_val_loss < best_val_loss:
                        best_val_loss = total_val_loss
                        print(f"Saving best model with val loss: {best_val_loss}")
                        
                        # Save the model checkpoint
                        checkpoint_name = f'checkpoint_{cfg.dataset}_{cfg.encoder}_{cfg.embedding_projection_dim}_{epoch}.pth'
                        save_checkpoint(epoch, model, optimizer, loss, filename=checkpoint_name)
                    
                    break
                










# # Diccionario para guardar los embeddings de los grafos por clase cuyas claves son las etiquetas 0 - 71
                                    # # Diccionario para guardar los embeddings por clase
                                    # embeddings_list_by_class = defaultdict(list)
                                    # max_embeddings_per_class = 100
                    # # Guardar los embeddings y etiquetas
                    # if idx_val == 0:

                    #     embedding = embedding.cpu().numpy()
                    #     target = graph.y.cpu().numpy()

                    #     # Guardar embeddings y etiquetas en el diccionario por clase
                    #     for emb, label in zip(embedding, target):
                    #         if len(embeddings_list_by_class[label]) < max_embeddings_per_class:
                    #             embeddings_list_by_class[label].append(emb)
                # # Para el idx_val = 1, guardar los embeddings de los grafos con sus etiquetas para visualizarlos en TensorBoard Projector
                # if idx_val == 0:

                #     # Preparar los datos para TensorBoard
                #     all_embeddings = []
                #     all_labels = []

                #     for label, embeddings in embeddings_list_by_class.items():
                #         all_embeddings.extend(embeddings)
                #         # Obtener la etiqueta textual de la clase a través del handler
                #         label = handler.get_tract_from_label(label)
                #         print(label)
                #         all_labels.extend([label] * len(embeddings))

                #     # Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at /opt/conda/conda-bld/pytorch_1704987394225/work/torch/csrc/utils/tensor_new.cpp:275.)
                #     # Convertir listas a numpy.ndarray
                #     all_embeddings = np.array(all_embeddings)
                #     all_labels = np.array(all_labels)


                #     # Convertir a tensores
                #     # Convertir a tensores
                #     all_embeddings = torch.tensor(all_embeddings)
                #     all_labels = all_labels.tolist()  # TensorBoard necesita etiquetas como lista de strings


                #     # Guardar los embeddings y etiquetas en TensorBoard
                #     writer.add_embedding(
                #         all_embeddings, 
                #         metadata = all_labels, 
                #         global_step = epoch,
                #         tag = f"{cfg.dataset}_{cfg.encoder}_{cfg.embedding_projection_dim}_{time.time()}"
                #     ) # # Cerrar el SummaryWriter
# writer.close()