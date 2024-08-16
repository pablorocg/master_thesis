from encoders import GraphFiberNet
from loss_functions import SupConLossWithCrossEntropy
from single_fiber_dataset import StreamlineSingleDataset, collate_single_ds
from custom_transforms import MaxMinNormalization
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch_geometric.transforms import Compose, ToUndirected, GCNNorm
from utils import save_checkpoint, get_dataset, seed_everything
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
import random
import string
import os
import neptune
# Ignorar advertencias
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="dipy.io.stateful_tractogram")



# Autor: Pablo Rocamora

import neptune
import numpy as np
import random
import time
import torch
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from dataset_handlers import (FiberCupHandler, 
                              HCPHandler, 
                              TractoinfernoHandler, 
                              HCP_Without_CC_Handler)

from sklearn.model_selection import StratifiedKFold

from custom_transforms import MaxMinNormalization
from utils import fill_tracts_ds

from loss_functions import SupConOutLoss, SupConLossWithCrossEntropy

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GCN, GAT, GraphSAGE, GIN
from torch_geometric.nn import global_add_pool
from tqdm import tqdm
from torch_geometric.transforms import Compose, ToUndirected, NormalizeFeatures, AddSelfLoops, GCNNorm
import os
import torch.nn.functional as F
from single_fiber_dataset import StreamlineSingleDataset, collate_single_ds
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from neptune.types import File
# Comando para lanzar tensorboard en el navegador local a través del puerto 8888 reenviado por ssh:
# tensorboard --logdir=/app/runs/HCP_105_without_CC_embedding_visualization_infonce_pretraining --host 0.0.0.0 --port 8888


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
        self.max_epochs = 15
        self.batch_size = 4096
        self.learning_rate = 3e-3
        # self.max_batches_per_subject = 200#350# Un buen valor es 500
        self.optimizer = "AdamW"
        self.encoder = "GCN"#"GCN"
        self.temperature = 0.07
        self.embedding_projection_dim = 256
        self.dataset = "HCP_105_without_CC"#"Tractoinferno"#"HCP_105"#

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
    run["config"] = {
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "max_epochs": cfg.max_epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        # "max_batches_per_subject": cfg.max_batches_per_subject,
        "embedding_projection_dim": cfg.embedding_projection_dim,
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

elif cfg.dataset == "HCP_105_without_CC":
    handler = HCP_Without_CC_Handler(path = cfg.ds_path, scope = "trainset")
    train_data = handler.get_data()

    handler = HCP_Without_CC_Handler(path = cfg.ds_path, scope = "validset")
    valid_data = handler.get_data()

elif cfg.dataset == "Tractoinferno":
    handler = TractoinfernoHandler(path = cfg.ds_path, scope = "trainset")
    train_data = handler.get_data()
    train_data = fill_tracts_ds(train_data)# Hacer que todos los sujetos tengan el mismo número de tractos 

    handler = TractoinfernoHandler(path = cfg.ds_path, scope = "validset")
    valid_data = handler.get_data()

elif cfg.dataset == "FiberCup":
    handler = FiberCupHandler(path = cfg.ds_path, scope = "trainset")
    train_data = handler.get_data()

    handler = FiberCupHandler(path = cfg.ds_path, scope = "validset")
    valid_data = handler.get_data()

transforms = Compose([
        MaxMinNormalization(dataset = cfg.dataset), 
        ToUndirected(), 
        GCNNorm()
    ])


if cfg.encoder == "GCN":
    model = GCN(
        in_channels = 3, 
        hidden_channels = cfg.embedding_projection_dim, 
        out_channels = cfg.embedding_projection_dim, 
        num_layers = 5
    ).cuda()
    # model = GraphFiberNet().cuda()

elif cfg.encoder == "GAT":
    model = GAT(
        in_channels = 3,
        hidden_channels = cfg.embedding_projection_dim,
        out_channels = cfg.embedding_projection_dim,
        num_layers = 5
    ).cuda()
    
elif cfg.encoder == "GraphSAGE":
    model = GraphSAGE(
        in_channels = 3,
        hidden_channels = cfg.embedding_projection_dim,
        out_channels = cfg.embedding_projection_dim,
        num_layers = 5
    ).cuda()

elif cfg.encoder == "GIN":
    model = GIN(
        in_channels = 3,
        hidden_channels = cfg.embedding_projection_dim,
        out_channels = cfg.embedding_projection_dim,
        num_layers = 5
    ).cuda()



# Definir loss function, optimizador y scheduler
criterion = SupConOutLoss(
    temperature = cfg.temperature
).cuda()

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr = cfg.learning_rate, # 1e-3
    weight_decay = 1e-2
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer = optimizer,
    mode = 'min', 
    factor = 0.7, 
    patience = 20, 
    verbose = True
)



best_f1 = 0
# Entrenar el modelo
for epoch in range(cfg.max_epochs):

    model.train()
    for idx_suj, subject in enumerate(train_data):

        total_loss = 0

        # Crear el dataset y el dataloader
        train_ds = StreamlineSingleDataset(
            datadict = subject, 
            ds_handler = handler, 
            transform = transforms,
            select_n_streamlines = 200
        )
        
        train_dl = DataLoader(
            dataset = train_ds, 
            batch_size = cfg.batch_size, 
            shuffle = True, 
            num_workers = 1,
            collate_fn = collate_single_ds
        )
        
        # Bucle de entrenamiento del modelo
        prog_bar = tqdm(
            iterable = train_dl
        )

        for i, graph_batch in enumerate(prog_bar):
            
            # Enviar a la gpu
            graph_batch = graph_batch.to('cuda')
            
            # Reiniciar los gradientes
            optimizer.zero_grad()

            # Forward pass
            node_embeddings = model(
                x = graph_batch.x,
                edge_index = graph_batch.edge_index,
                batch = graph_batch.batch,
                batch_size = cfg.batch_size
            )

            graph_embeddings = global_add_pool(node_embeddings, graph_batch.batch)
            
            
            # Calcular la pérdida
            loss = criterion(graph_embeddings, graph_batch.y)

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
                    

            
        
        avg_loss = total_loss / len(train_dl)

        # Actualizar el learning rate
        scheduler.step(avg_loss)

        # Log del lr actual
        if log:
            run["train/lr"].log(optimizer.param_groups[0]['lr'])
            run["train/subject/loss"].log(avg_loss)

        

        # Validación del modelo cada 15 sujetos
        if idx_suj % 25 == 0 and idx_suj > 0:
            print(f"[VALIDATION] Epoch {epoch+1}/{cfg.max_epochs}")
            # Tensor al que ir concatenando las embeddings de los sujetos
            embeddings = torch.tensor([]).to('cuda')
            # Tensor al que ir concatenando las etiquetas de los sujetos
            labels = torch.tensor([]).to('cuda')
            # Tensor al que ir concatenando los nombres de los sujetos
            subject_names = []

            for idx_suj, subject in enumerate(valid_data):# Iterar sobre los sujetos de entrenamiento

                valid_ds = StreamlineSingleDataset(
                    datadict = subject, 
                    ds_handler = handler, 
                    transform = transforms,
                    select_n_streamlines = 20
                )
                    
                valid_dl = DataLoader(
                    dataset = valid_ds, 
                    batch_size = cfg.batch_size,              
                    shuffle = False, 
                    num_workers = 1,
                    collate_fn = collate_single_ds
                )
                    
                # Bucle de entrenamiento del modelo
                prog_bar = tqdm(
                    iterable = valid_dl
                )

                for i, graph_batch in enumerate(prog_bar):
                    # Enviar a la gpu
                    graph_batch = graph_batch.to('cuda')
                    
                    
                    with torch.no_grad():
                        # Forward pass
                        node_embeddings = model(
                            x = graph_batch.x,
                            edge_index = graph_batch.edge_index,
                            batch = graph_batch.batch,
                            batch_size = cfg.batch_size
                        )

                    graph_embeddings = global_add_pool(node_embeddings, graph_batch.batch)
                    graph_embeddings = F.normalize(graph_embeddings, p=2, dim=-1)
                                           
                    # Calcular la pérdida

                        
                    # Concatenar las embeddings, etiquetas y nombres de sujetos
                    embeddings = torch.cat((embeddings, graph_embeddings), dim=0)
                    labels = torch.cat((labels, graph_batch.y), dim=0)
                    subject_names.extend([f'sujeto_{idx_suj}'] * graph_embeddings.size(0))


            # Convertir a CPU para t-SNE
            embeddings = embeddings.cpu().numpy()
            labels = labels.cpu().numpy()

            
            # Estandarizar las embeddings antes de aplicar t-SNE
            # scaler = StandardScaler()
            # embeddings_scaled = scaler.fit_transform(embeddings)

            # Aplicar t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_tsne = tsne.fit_transform(embeddings)

            # Crear DataFrame para visualización con Plotly
            
            df = pd.DataFrame({
                'TSNE Component 1': embeddings_tsne[:, 0],
                'TSNE Component 2': embeddings_tsne[:, 1],
                'Label': [handler.get_tract_from_label(int(label)) for label in labels],
                'Subject': subject_names
            })
            df['Label'] = df['Label'].astype(str) 
            # Visualizar los resultados con Plotly
            fig = px.scatter(
                df, x='TSNE Component 1', y='TSNE Component 2', color='Label',
                title='t-SNE Visualization of Embeddings',
                hover_data=['Subject'],
                labels={'Label': 'Class'},
                color_discrete_sequence=px.colors.qualitative.Set1,
                width=1000, height=1000
            )

            # Guardar la visualización como png
            
            fig.write_html(f"/app/pruebas/{cfg.dataset}_{cfg.encoder}_{epoch}_infonce_t-SNE_visualization.html")
            # Loggear la visualización en Neptune
            if log:
                run["html"].upload(File(f"/app/pruebas/{cfg.dataset}_{cfg.encoder}_{epoch}_infonce_t-SNE_visualization.html"))
                # run["val/t-SNE_visualization"].upload()
            
        
            # Clasificación con KNN
            accs, f1s, precisions, recalls = [], [], [], []
            folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            for train_index, test_index in folds.split(embeddings, labels):
                X_train, X_test = embeddings[train_index], embeddings[test_index]
                y_train, y_test = labels[train_index], labels[test_index]
                
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                
                accs.append(acc)
                f1s.append(f1)
                precisions.append(precision)
                recalls.append(recall)

            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            mean_f1 = np.mean(f1s)
            std_f1 = np.std(f1s)
            mean_precision = np.mean(precisions)
            std_precision = np.std(precisions)
            mean_recall = np.mean(recalls)
            std_recall = np.std(recalls)

            # Classification report con el ultimo fold
            cr = classification_report(
                y_test, 
                y_pred, 
                target_names=[handler.get_tract_from_label(int(label)) for label in np.unique(labels)]
            )

            print(cr)
            print(f"Accuracy: {mean_acc:.4} +/- {std_acc:.4}")
            print(f"F1: {mean_f1:.4} +/- {std_f1:.4}")
            print(f"Precision: {mean_precision:.4} +/- {std_precision:.4}")
            print(f"Recall: {mean_recall:.4} +/- {std_recall:.4}")
            

            if log:
                run["val/accuracy"].log(mean_acc)
                run["val/f1"].log(mean_f1)
                run["val/precision"].log(mean_precision)
                run["val/recall"].log(mean_recall)
                

            if best_f1 < mean_f1:
                best_f1 = mean_f1

                checkpoint_name = f'checkpoint_{cfg.dataset}_{cfg.embedding_projection_dim}_{cfg.encoder}_{epoch}_supconout_f1_{mean_f1:.4f}.pth'
                # guardar solo el mejor modelo por lo que hay que borrar el anterior mejor

                save_checkpoint(epoch, model, optimizer, loss, filename=checkpoint_name)





            
                
