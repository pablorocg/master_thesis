# Autor: Pablo Rocamora


import neptune


import time
import torch
import torch.nn as nn


from torch_geometric.transforms import Compose
from dataset_handlers import (FiberCupHandler, 
                              HCPHandler, 
                              TractoinfernoHandler, 
                              HCP_Without_CC_Handler)
# from encoders import (
#     ClassifierHead,
#     GCNEncoder,
#     ProjectionHead,
# )

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
# Comando para lanzar tensorboard en el navegador local a través del puerto 8888 reenviado por ssh:
# tensorboard --logdir=runs/embedding_visualization --host 0.0.0.0 --port 8888




#==================================CONFIG=======================================
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
       
        "n_classes": cfg.n_classes,
        "embedding_projection_dim": cfg.embedding_projection_dim,
      
        "encoder": cfg.encoder,
        "optimizer": cfg.optimizer
    }
#===============================================================================

data = get_dataset(cfg.dataset, cfg.ds_path)
train_data = data["train"]
valid_data = data["valid"]
print(f"Train subjects: {len(train_data)}")
print(f"Valid subjects: {len(valid_data)}")
test_data = data["test"]
handler = data["handler"]

pretrained_encoder = GCN(
    in_channels = 3, 
    hidden_channels = 256, 
    out_channels = 256, 
    num_layers = 5
)

pretrained_encoder.load_state_dict(torch.load('/app/trained_models/checkpoint_HCP_105_without_CC_256_GCN_6_supconout_f1_0.8648.pth')['model_state_dict'])

model = GraphFiberNet(
    encoder = pretrained_encoder,
    hidden_channels = 256,
    n_classes = cfg.n_classes
).cuda()


# model.encoder.load_state_dict(torch.load('/app/trained_models/checkpoint_Tractoinferno_256_GCN_3_supconout_f1_0.9154.pth')['model_state_dict'])


# class_weights = [0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.018928406911583026, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.012988410533410502, 0.01292792165006919, 0.012958281051389589, 0.03744949301341532, 0.04831233844386465, 0.012962116949068162, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.0129685564603115, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.012961003067422231, 0.01292792165006919, 0.01292792165006919, 0.013229813391848906, 0.012962612069034894, 0.01313152278428651, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.014021172500152512, 0.013124793553369736, 0.012931246827824918, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.013054362475185993, 0.013347805473684228, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.01292792165006919, 0.013172810469463407, 0.013303589682533102, 0.013011816888483508, 0.01292792165006919]
# Loss function
criterion = nn.CrossEntropyLoss(
    #weight = torch.tensor(class_weights).cuda()
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


transforms = Compose([
    MaxMinNormalization(dataset = cfg.dataset),
    # AddSelfLoops(), 
    ToUndirected(),
    # Cartesian2SphericalCoords(), 
    # MaxMinNormalization(dataset = cfg.dataset),
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
            num_workers = 4,
            collate_fn = collate_single_ds
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

            prediction = F.log_softmax(prediction, dim=1)
          
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
                    transform = transforms,
                    select_n_streamlines=25
                )
                
                val_dl = DataLoader(
                    dataset = val_ds, 
                    batch_size = cfg.batch_size, 
                    shuffle = True, 
                    num_workers = 4,
                    collate_fn = collate_single_ds
                )

                for i, graph in enumerate(val_dl):
                        
                    
                    graph = graph.to('cuda')
                    target = graph.y

                    with torch.no_grad():
                        pred = model(graph)
                    
                    
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
                
                
            val_loss = val_loss / len(valid_data)



            if log:
                run["val/avg_loss"].log(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Saving best model with val loss: {best_val_loss}")
                
                # Save the model checkpoint
                checkpoint_name = f'checkpoint_{cfg.dataset}_finetuned_{cfg.encoder}_{epoch}_{val_loss:.2f}.pth'
                print(f"Saving checkpoint: {checkpoint_name}")
                save_checkpoint(epoch, model, optimizer, loss, filename=checkpoint_name)
                    
                    
                
