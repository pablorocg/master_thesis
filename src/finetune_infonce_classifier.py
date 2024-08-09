# Autor: Pablo Rocamora

import matplotlib.pyplot as plt
import neptune
import numpy as np

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
    StreamlineSingleDataset,
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoders import (
    ClassifierHead,
    GATEncoder,
    GCNEncoder,
    ProjectionHead,
    SiameseGraphNetwork
)

from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

from encoders import (
    ClassifierHead,
    GCNEncoder,
    ProjectionHead,
)


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
        if self.training:  # Aplicación de Dropout solo durante el entrenamiento
            x = self.dropout(x)
        
        return x
    

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dropout, n_hidden_blocks):
        super(GCNEncoder, self).__init__()
        self.input_block = GraphConvBlock(in_channels, hidden_dim, dropout)
        self.hidden_blocks = nn.ModuleList([GraphConvBlock(hidden_dim, hidden_dim, dropout) for _ in range(n_hidden_blocks - 1)])
        self.output_block = GraphConvBlock(hidden_dim, out_channels, dropout)
        # self.bn = BatchNorm(out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_block(x, edge_index)
        for layer in self.hidden_blocks:
            x = layer(x, edge_index)
        x = self.output_block(x, edge_index)
        # x = self.bn(x)

        return global_mean_pool(x, batch) # (batch_size, out_channels)


class ProjectionHead(nn.Module):
    """
    Proyección de las embeddings de texto a un espacio de dimensión reducida.
    """
    def __init__(
        self,
        embedding_dim,# Salida del modelo de lenguaje (768)
        projection_dim, # Dimensión de la proyección (256)
        # dropout=0.1
    ):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        # x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    

class SiameseContrastiveGraphNetwork(nn.Module):
    def __init__(self, encoder, projection_head):
        super(SiameseContrastiveGraphNetwork, self).__init__()
        self.encoder = encoder
        self.projection_head = projection_head

    def forward(self, graph):
        x_1 = self.encoder(graph)
        x_1 = self.projection_head(x_1)
        return x_1


model = SiameseContrastiveGraphNetwork(
    encoder = GCNEncoder(
        in_channels = 3, 
        hidden_dim = 64, 
        out_channels = 128, 
        dropout = 0.5, 
        n_hidden_blocks = 4
    ),

    projection_head = ProjectionHead(
        embedding_dim = 128, 
        projection_dim = 64
    )
)

model = torch.compile(model, dynamic=True)

# Cargar pesos preentrenados
checkpoint = torch.load('/app/trained_models/checkpoint_HCP_105_GCN_512_5_infonce_0.9312.pth')

model.load_state_dict(checkpoint['model_state_dict'])

# Eliminar capas Dropout y congelar los pesos
model.encoder.input_block.dropout.p = 0.0
for i in range(len(model.encoder.hidden_blocks)):
    model.encoder.hidden_blocks[i].dropout.p = 0.0
model.encoder.output_block.dropout.p = 0.0

# Eliminar la cabeza de proyección
model = model.encoder

# Congelar todos los parámetros del encoder
for param in model.parameters():
    param.requires_grad = False

# Definición del nuevo clasificador
class GraphClassifier(nn.Module):
    def __init__(self, encoder, n_classes):
        super(GraphClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = ClassifierHead(
            projection_dim=128, 
            n_classes=n_classes
        )   

    def forward(self, graph):
        x = self.encoder(graph)
        x = self.classifier(x)
        return x

# Instanciar el modelo de clasificación
model = GraphClassifier(encoder=model, n_classes=cfg.n_classes).cuda()
model = torch.compile(model, dynamic=True)


class_weights = [0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.018928406911583026, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.012988410533410502, 0.01292792165006919, 0.012958281051389589, 0.03744949301341532, 0.04831233844386465, 0.012962116949068162, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.0129685564603115, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.012961003067422231, 0.01292792165006919, 0.01292792165006919, 0.013229813391848906, 0.012962612069034894, 0.01313152278428651, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.014021172500152512, 0.013124793553369736, 0.012931246827824918, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.013054362475185993, 0.013347805473684228, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.013172810469463407, 0.013303589682533102, 0.013011816888483508, 0.01292792165006919]
# Loss function
criterion = nn.CrossEntropyLoss(
    # weight = torch.tensor(class_weights).cuda()
).cuda()
    
# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr = cfg.learning_rate,
    weight_decay = 1e-4
)

# Scheduler    
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer = optimizer,
    mode = 'min', 
    factor = 0.5, 
    patience = 15, 
    verbose = True
)

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

# subj_confusion_matrix = MulticlassConfusionMatrix(
#     num_classes = cfg.n_classes
# ).cuda()
from torch_geometric.transforms import Compose, ToUndirected, NormalizeFeatures, AddSelfLoops, GCNNorm

transforms = Compose([
    ToUndirected(), 
    GCNNorm() 
])



#==================================ENTRENAMIENTO================================
best_val_loss = float('inf')# Inicializar la mejor pérdida de validación

for epoch in range(cfg.max_epochs):

    for idx_suj, subject in enumerate(train_data):# Iterar sobre los sujetos de entrenamiento

        model.train()
        train_ds = StreamlineSingleDataset(
            datadict = subject, 
            ds_handler = handler, 
            transform = transforms#
        )
        
        train_dl = DataLoader(
            dataset = train_ds, 
            batch_size = cfg.batch_size,              
            shuffle = True, 
            num_workers = 4,
            collate_fn = collate_test_ds
        )
        
        # Bucle de entrenamiento del modelo
        prog_bar = tqdm(
            iterable = train_dl
        )

        train_loss = 0
        for i, graph_batch in enumerate(prog_bar):

            # Enviar a la gpu
            graph_batch = graph_batch.to('cuda')
            
            # Reiniciar los gradientes
            optimizer.zero_grad()

            # Forward pass
            prediction = model(graph_batch)
          
            # Calcular la pérdida
            loss = criterion(prediction, graph_batch.y)

            train_loss += loss.item()# Acumular la pérdida

            # Backward pass
            loss.backward()

            # Actualizar los pesos
            optimizer.step()

            # Calcular las métricas de entrenamiento
            subj_accuracy_train.update(prediction, graph_batch.y)
            subj_f1_train.update(prediction, graph_batch.y)
            # subj_confusion_matrix.update(prediction, graph_batch.y)


            # Mostar métricas de entrenamiento cada 20 batches
            if i % 20 == 0:
                print(f"[TRAIN] Epoch {epoch+1}/{cfg.max_epochs} - Subj {idx_suj} - Batch {i} - Acc.: {subj_accuracy_train.compute().item():.4f}, F1: {subj_f1_train.compute().item():.4f}")
                
            if log:# Loggear las métricas de batch
                run["train/batch/loss"].log(loss.item())
                run["train/batch/acc"].log(subj_accuracy_train.compute().item())
                run["train/batch/f1"].log(subj_f1_train.compute().item())
              

        avg_loss = train_loss / len(train_dl)

        if log:# Loggear las métricas de subject
            run["train/subject/acc"].log(subj_accuracy_train.compute().item())
            run["train/subject/f1"].log(subj_f1_train.compute().item())
            # run["train/subject/auroc"].log(subj_auroc_train.compute().item())
            run["train/subject/avg_loss"].log(avg_loss)
            # Log lr
            run["train/lr"].log(optimizer.param_groups[0]['lr'])
                
        # Reiniciar las métricas
        subj_accuracy_train.reset()
        subj_f1_train.reset()
        subj_auroc_train.reset()

        
        # Actualizar el learning rate
        scheduler.step(avg_loss)
                


        # Validación del modelo cada 15 sujetos
        if idx_suj % 25 == 0 and idx_suj != 0:
            
            val_loss = 0 # Inicializar la pérdida total de validación
            model.eval()

            for idx_val, subject in enumerate(valid_data):
                subj_val_loss = 0
                
                val_ds = StreamlineSingleDataset(
                    datadict = subject, 
                    ds_handler = handler, 
                    transform = transforms##MaxMinNormalization(dataset = cfg.dataset)
                )
                
                val_dl = DataLoader(
                    dataset = val_ds, 
                    batch_size = cfg.batch_size, 
                    shuffle = True, 
                    num_workers = 4,
                    collate_fn = collate_test_ds
                )

                for i, graph in enumerate(val_dl):
                        
                    # Enviar a la gpu
                    graph = graph.to('cuda')
                    target = graph.y

                    with torch.no_grad():
                        pred = model(graph)
                    
                    # calcular el loss de validación
                    loss = nn.functional.cross_entropy(pred, target)
                    subj_val_loss += loss.item()


                    # Calcular métricas de validación
                    subj_accuracy_val.update(pred, target)
                    subj_f1_val.update(pred, target)
                    
                    
                    if log:# Loggear las métricas de batch
                        run["val/batch/acc"].log(subj_accuracy_val.compute().item())
                        run["val/batch/f1"].log(subj_f1_val.compute().item())
                        run["val/batch/loss"].log(loss.item())

                    if i % 25 == 0:
                        print(f"[VAL] Epoch {epoch+1}/{cfg.max_epochs} - Subj {idx_val} - Batch {i} - Acc.: {subj_accuracy_val.compute().item():.4f}, F1: {subj_f1_val.compute().item():.4f}")

                subj_val_loss = subj_val_loss / len(val_dl)
                val_loss += subj_val_loss

                if log:# Loggear las métricas de subject
                    run["val/subject/acc"].log(subj_accuracy_val.compute().item())
                    run["val/subject/f1"].log(subj_f1_val.compute().item())
                    run["val/subject/loss"].log(subj_val_loss)
                
                subj_accuracy_val.reset()
                subj_f1_val.reset()
                
                # Si es el último sujeto de validación, comprobar si es el mejor modelo y guardarlo
                if idx_val == 10:
                    val_loss = val_loss / 10

                    if log:
                        run["val/avg_loss"].log(val_loss)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        print(f"Saving best model with val loss: {best_val_loss}")
                        
                        # Save the model checkpoint
                        checkpoint_name = f'checkpoint_{cfg.dataset}_finetuned_{cfg.encoder}_{epoch}_{val_loss:.2f}.pth'
                        save_checkpoint(epoch, model, optimizer, loss, filename=checkpoint_name)
                    
                    break
                
