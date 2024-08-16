# Autor: Pablo Rocamora

import neptune
import time
import torch
import torch.nn as nn
from torch_geometric.transforms import Compose
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAccuracy,
    MulticlassF1Score
)
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from encoders import GraphFiberNet
from torch_geometric.transforms import Compose, ToUndirected,GCNNorm
from torch_geometric.nn.models import GCN
from custom_transforms import MaxMinNormalization
from single_fiber_dataset import StreamlineSingleDataset, collate_single_ds
from utils import save_checkpoint, seed_everything, get_dataset


class CFG:
    def __init__(self):
        self.seed = 42
        self.max_epochs = 10
        self.batch_size = 1024
        self.learning_rate = 1e-3
        
        self.optimizer = "AdamW"
        
        self.encoder = "GCN"
        self.embedding_projection_dim = 256
        self.dataset = "HCP_105_without_CC"#"Tractoinferno"

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

if log:
    run = neptune.init_run(
        project="pablorocamora/tfm-tractography",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ODA0YzA2NS04MjczLTQyNzItOGE5Mi05ZmI5YjZkMmY3MDcifQ==",
    )



data = get_dataset(cfg.dataset, cfg.ds_path)
train_data = data["train"]
valid_data = data["valid"]
handler = data["handler"]

print(f"Train subjects: {len(train_data)}")
print(f"Valid subjects: {len(valid_data)}")




pretrained_encoder = GCN(
    in_channels = 3, 
    hidden_channels = 256, 
    out_channels = 256, 
    num_layers = 5
)

model = GraphFiberNet(
    encoder = pretrained_encoder,
    hidden_channels = 256,
    n_classes = cfg.n_classes,
    full_trainable = True
).cuda()


# Loss function
criterion = nn.CrossEntropyLoss().cuda()
    
# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr = cfg.learning_rate,
    weight_decay = 1e-4
)

# Scheduler    
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer = optimizer,
#     mode = 'min', 
#     factor = 0.5, 
#     patience = 15, 
#     verbose = True
# )
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_epochs * len(train_data), eta_min=1e-6)

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





transforms = Compose([
    MaxMinNormalization(dataset = cfg.dataset),
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
            transform = transforms,#
            select_n_streamlines=200
        )
        
        train_dl = DataLoader(
            dataset = train_ds, 
            batch_size = cfg.batch_size,              
            shuffle = True, 
            num_workers = 2,
            collate_fn = collate_single_ds
        )
        
        # Bucle de entrenamiento del modelo
        prog_bar = tqdm(iterable = train_dl)

        train_loss = 0
        for i, graph_batch in enumerate(prog_bar):
            graph_batch = graph_batch.to('cuda')
            optimizer.zero_grad()
            prediction = model(graph_batch)
            prediction = F.log_softmax(prediction, dim=1)
            loss = criterion(prediction, graph_batch.y)
            train_loss += loss.item()# Acumular la pérdida
            loss.backward()
            optimizer.step()
            subj_accuracy_train.update(prediction, graph_batch.y)
            subj_f1_train.update(prediction, graph_batch.y)

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
            run["train/subject/avg_loss"].log(avg_loss)
            run["train/lr"].log(optimizer.param_groups[0]['lr'])
                
        subj_accuracy_train.reset()
        subj_f1_train.reset()
        scheduler.step()
                

        if idx_suj % 25 == 0 and idx_suj != 0:
            val_loss = 0 
            model.eval()

            for idx_val, subject in enumerate(valid_data):
                subj_val_loss = 0
                
                val_ds = StreamlineSingleDataset(
                    datadict = subject, 
                    ds_handler = handler, 
                    transform = transforms,
                    select_n_streamlines=25
                )
                
                val_dl = DataLoader(
                    dataset = val_ds, 
                    batch_size = cfg.batch_size, 
                    shuffle = True, 
                    num_workers = 2,
                    collate_fn = collate_single_ds
                )

                for i, graph in enumerate(val_dl):
                        
                    
                    graph = graph.to('cuda')
                    target = graph.y

                    with torch.no_grad():
                        pred = model(graph)
                    
                    loss = nn.functional.cross_entropy(pred, target)
                    subj_val_loss += loss.item()
                    subj_accuracy_val.update(pred, target)
                    subj_f1_val.update(pred, target)
                    
                    
                    if log:
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
                
            val_loss = val_loss / len(valid_data)

            if log:
                run["val/avg_loss"].log(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Saving best model with val loss: {best_val_loss}")
                checkpoint_name = f'checkpoint_{cfg.dataset}_baseline_classifier.pth'
                print(f"Saving checkpoint: {checkpoint_name}")
                save_checkpoint(epoch, model, optimizer, loss, filename=checkpoint_name)
                    
                    
                
