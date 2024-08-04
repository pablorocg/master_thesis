# Modelo de clasificación de grafos de tractografías cerebrales sin contrastive loss

from dataset_handlers import Tractoinferno_handler, HCP_handler
import numpy as np
import random
from dipy.io.streamline import load_trk
import pathlib2 as pathlib
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import torch.nn as nn
# Importar categorical crossentropy loss
from torch.nn import CrossEntropyLoss

from torch_geometric.data import Data
from torch.nn import ModuleList
import torch.nn.functional as F
from torchmetrics.classification import (MulticlassAccuracy, 
                                         MulticlassF1Score,
                                         MulticlassAUROC)
import torch.optim as optim
from tqdm import tqdm

from encoders import SiameseGraphNetwork, GCNEncoder, ProjectionHead, ClassifierHead

from streamline_datasets import (MaxMinNormalization, 
                                 StreamlineTestDataset, collate_test_ds, 
                                 StreamlineTripletDataset, collate_triplet_ds,
                                 fill_tracts_ds)

from loss_functions import (MultiTaskTripletLoss, TripletLoss)
import neptune
from torch.utils.tensorboard import SummaryWriter
# Comando para lanzar tensorboard en el navegador local a traves del puerto 8888 reenviado por ssh:
# tensorboard --logdir=runs/embedding_visualization --host 0.0.0.0 --port 8888
from collections import defaultdict


# Habilitar TensorFloat32 para una mejor performance en operaciones de multiplicación de matrices
torch.set_float32_matmul_precision('high')



log = True

if log:
    run = neptune.init_run(
        project="pablorocamora/tfm-tractography",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ODA0YzA2NS04MjczLTQyNzItOGE5Mi05ZmI5YjZkMmY3MDcifQ==",
    )

    # Inicializar el SummaryWriter para TensorBoard
    writer = SummaryWriter(log_dir="runs/embedding_visualization")
    




class CFG:
    def __init__(self):
        self.seed = 42
        self.max_epochs = 1
        self.batch_size = 128
        self.learning_rate = 0.005
        self.max_batches_per_subject = 500
        self.classification_weight = 0.5
        self.margin = 1.25

        self.dataset = "Tractoinferno"#"HCP_105"

        if self.dataset == "HCP_105":
            self.ds_path = "/app/dataset/HCP_105"
            self.n_classes = 72

        elif self.dataset == "Tractoinferno":
            self.ds_path = "/app/dataset/Tractoinferno/tractoinferno_preprocessed_mni"
            self.n_classes = 32

        self.embedding_projection_dim = 128
        self.optimizer = "AdamW"
        
    
cfg = CFG()

if log:
    run["config/dataset"] = cfg.dataset
    run["config/seed"] = cfg.seed
    run["config/max_epochs"] = cfg.max_epochs
    run["config/batch_size"] = cfg.batch_size
    run["config/learning_rate"] = cfg.learning_rate
    run["config/max_batches_per_subject"] = cfg.max_batches_per_subject
    run["config/n_classes"] = cfg.n_classes
    run["config/embedding_projection_dim"] = cfg.embedding_projection_dim



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Establecer la semilla
seed_everything(cfg.seed)

# Cargar las rutas de los sujetos de entrenamiento, validación y test
if cfg.dataset == "HCP_105":
    handler = HCP_handler(path = cfg.ds_path, scope = "trainset")
    train_data = handler.get_data()

    handler = HCP_handler(path = cfg.ds_path, scope = "validset")
    valid_data = handler.get_data()

    handler = HCP_handler(path = cfg.ds_path, scope = "testset")
    test_data = handler.get_data()

elif cfg.dataset == "Tractoinferno":
    handler = Tractoinferno_handler(path = cfg.ds_path, scope = "trainset")
    train_data = handler.get_data()
    train_data = fill_tracts_ds(train_data)# Hacer que todos los sujetos tengan el mismo número de tractos 

    handler = Tractoinferno_handler(path = cfg.ds_path, scope = "validset")
    valid_data = handler.get_data()

    handler = Tractoinferno_handler(path = cfg.ds_path, scope = "testset")
    test_data = handler.get_data()


# Crear el modelo, la función de pérdida y el optimizador
model = SiameseGraphNetwork(
    encoder = GCNEncoder(in_channels = 3, 
                         hidden_dim = 512, 
                         out_channels = 512, 
                         dropout = 0.15, 
                         n_hidden_blocks = 4),
    projection_head = ProjectionHead(embedding_dim = 512, 
                                     projection_dim = cfg.embedding_projection_dim),
    classifier = ClassifierHead(projection_dim = cfg.embedding_projection_dim, 
                                n_classes = cfg.n_classes)
).cuda()

# Compile the model into an optimized version:
model = torch.compile(model, dynamic=True)

# Definir loss function, optimizador y scheduler
criterion = CrossEntropyLoss().cuda()



# Seleccionar el optimizador
if cfg.optimizer == "Adam":
    optimizer = torch.optim.Adam(model.parameters(),
                              lr = cfg.learning_rate)
elif cfg.optimizer == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(),
                                lr = cfg.learning_rate,
                                weight_decay = 0.01)
else:
    raise ValueError("Invalid optimizer")



scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                            step_size = 10, gamma = 0.1)


# Crear las métricas con torchmetrics
subj_accuracy_train = MulticlassAccuracy(num_classes = cfg.n_classes, average='macro').cuda()
subj_accuracy_val = MulticlassAccuracy(num_classes = cfg.n_classes, average='macro').cuda()

subj_f1_train = MulticlassF1Score(num_classes = cfg.n_classes, average='macro').cuda()
subj_f1_val = MulticlassF1Score(num_classes = cfg.n_classes, average='macro').cuda()

subj_auroc_train = MulticlassAUROC(num_classes = cfg.n_classes).cuda()
subj_auroc_val = MulticlassAUROC(num_classes = cfg.n_classes).cuda()




# Entrenar el modelo
for epoch in range(cfg.max_epochs):

    model.train()# Establecer el modelo en modo de entrenamiento

    for idx_suj, subject in enumerate(train_data):# Iterar sobre los sujetos de entrenamiento

        # Crear el dataset y el dataloader
        train_ds = StreamlineTestDataset(subject, handler, 
                                        transform = MaxMinNormalization())
        
        train_dl = DataLoader(train_ds, batch_size = cfg.batch_size * 4, 
                              shuffle = True, num_workers = 4,
                              collate_fn = collate_test_ds)
        
        
        # Bucle de entrenamiento del modelo
        prog_bar = tqdm(train_dl, total = cfg.max_batches_per_subject)

        for i, graph in enumerate(prog_bar):

            # Enviar a la gpu
            graph = graph.cuda()
            target = graph.y

            # Reiniciar los gradientes
            optimizer.zero_grad()

            # Forward pass
            embedding, pred = model(graph)

            # Calcular la pérdida
            loss = criterion(pred, target)
            
            # Backward pass
            loss.backward()

            # Actualizar los pesos
            optimizer.step()

            


            # calcular métricas de entrenamiento
            subj_accuracy_train.update(pred, target)
            subj_f1_train.update(pred, target)
            subj_auroc_train.update(pred, target)


            # Mostar métricas de entrenamiento cada 100 batches
            if i % 25 == 0:
                print(f"[TRAIN] Epoch {epoch+1}/{cfg.max_epochs} - Subj {idx_suj} - Batch {i} - Acc.: {subj_accuracy_train.compute().item():.4f}, F1: {subj_f1_train.compute().item():.4f}, AUROC: {subj_auroc_train.compute().item():.4f}")
                
                if log:# Loggear las métricas de batch
                    run["train/batch/loss"].log(loss.item())
                    run["train/batch/acc"].log(subj_accuracy_train.compute().item())
                    run["train/batch/f1"].log(subj_f1_train.compute().item())
                    run["train/batch/auroc"].log(subj_auroc_train.compute().item())

            # Condicion de parada del bucle
            if i == cfg.max_batches_per_subject:

                if log:# Loggear las métricas de subject
                    run["train/subject/acc"].log(subj_accuracy_train.compute().item())
                    run["train/subject/f1"].log(subj_f1_train.compute().item())
                    run["train/subject/auroc"].log(subj_auroc_train.compute().item())
                
                # Reiniciar las métricas
                subj_accuracy_train.reset()
                subj_f1_train.reset()
                subj_auroc_train.reset()

                break
        
        # Actualizar el learning rate
        scheduler.step()

    # Fase de validación del modelo

    model.eval()
    
    for idx_val, subject in enumerate(valid_data):
        val_ds = StreamlineTestDataset(subject, handler, 
                                        transform = MaxMinNormalization())
        
        val_dl = DataLoader(val_ds, batch_size = cfg.batch_size * 4, 
                              shuffle = False, num_workers = 4,
                              collate_fn = collate_test_ds)
        
        # Diccionario para guardar los embeddings de los grafos por clase cuyas claves son las etiquetas 0 - 71
        # Diccionario para guardar los embeddings por clase
        embeddings_list_by_class = defaultdict(list)
        max_embeddings_per_class = 100
        

        with torch.no_grad():
            for i, graph in enumerate(val_dl):
                
                # Enviar a la gpu
                graph = graph.to('cuda')
                target = graph.y

                # Forward pass
                embedding, pred = model(graph)

                # Calcular métricas de validación
                subj_accuracy_val.update(pred, target)
                subj_f1_val.update(pred, target)
                subj_auroc_val.update(pred, target)

                # Guardar los embeddings y etiquetas
                if idx_val == 0:

                    embedding = embedding.cpu().numpy()
                    target = graph.y.cpu().numpy()

                    # Guardar embeddings y etiquetas en el diccionario por clase
                    for emb, label in zip(embedding, target):
                        if len(embeddings_list_by_class[label]) < max_embeddings_per_class:
                            embeddings_list_by_class[label].append(emb)
                    

                if log:# Loggear las métricas de batch
                    run["val/batch/acc"].log(subj_accuracy_val.compute().item())
                    run["val/batch/f1"].log(subj_f1_val.compute().item())
                    run["val/batch/auroc"].log(subj_auroc_val.compute().item())

                if i % 25 == 0:
                    print(f"[VAL] Epoch {epoch+1}/{cfg.max_epochs} - Subj {idx_val} - Batch {i} - Acc.: {subj_accuracy_val.compute().item():.4f}, F1: {subj_f1_val.compute().item():.4f}, AUROC: {subj_auroc_val.compute().item():.4f}")

            # Para el idx_val = 1, guardar los embeddings de los grafos con sus etiquetas para visualizarlos en TensorBoard Projector
            if idx_val == 0:

                # Preparar los datos para TensorBoard
                all_embeddings = []
                all_labels = []

                for label, embeddings in embeddings_list_by_class.items():
                    all_embeddings.extend(embeddings)
                    all_labels.extend([label] * len(embeddings))

                # Convertir a tensores
                all_embeddings = torch.tensor(all_embeddings)
                all_labels = torch.tensor(all_labels)

                # Guardar los embeddings y etiquetas en TensorBoard
                writer.add_embedding(
                    all_embeddings, 
                    metadata = all_labels.tolist(), 
                    global_step = epoch,
                    tag = 'valid_embeddings_tractoinferno'
                ) 

            if log:# Loggear las métricas de subject
                run["val/subject/acc"].log(subj_accuracy_val.compute().item())
                run["val/subject/f1"].log(subj_f1_val.compute().item())
                run["val/subject/auroc"].log(subj_auroc_val.compute().item())

                subj_accuracy_val.reset()
                subj_f1_val.reset()
                subj_auroc_val.reset()

        if idx_val == 2:
            break

# Cerrar el SummaryWriter
writer.close()

               