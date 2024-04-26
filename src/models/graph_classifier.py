from encoders import GCN_Encoder, GAT_Encoder, CFG, Text_Encoder, ProjectionHead
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch_geometric
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


class GCN_Classifier(nn.Module):
    def __init__(self, 
                 graph_model_name = CFG.graph_model_name,
                 graph_embedding = CFG.graph_embedding,
                 graph_channels = CFG.graph_channels,
                 projection_dim = CFG.projection_dim,
                 device = CFG.device):
        
        super(GCN_Classifier, self).__init__()
        
        
        
        if graph_model_name == "GraphConvolutionalNetwork":
            self.graph_encoder = GCN_Encoder(graph_channels, graph_embedding)
        elif graph_model_name == "GraphAttentionNetwork":
            self.graph_encoder = GAT_Encoder(graph_channels, graph_embedding)
        else:
            raise ValueError("Invalid graph model name")
        
        
        self.graph_projection_head = ProjectionHead(graph_embedding, projection_dim)
        self.output_layer = nn.Linear(projection_dim, 32)
        
        
        self.device = device

        self.to(device)

       
        


    
    def forward(self, graph):
       
        graph_projections = self.graph_encoder(graph) # (batch_size, graph_embedding)
        graph_projections = self.graph_projection_head(graph_projections) # (batch_size, projection_dim)
        output = self.output_layer(graph_projections) # (batch_size, 32)
        
        return F.log_softmax(output, dim=1)
    
 

if __name__ == "__main__":
    
    import torchmetrics
    import torch
    import random
    import torch
    from torch_geometric.data import Data, Dataset
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence
    from torch_geometric.data import Batch as GeoBatch
    from torchvision.transforms import Compose
    import os
    from torch_geometric.transforms import BaseTransform
    from torch_geometric.data import Dataset
    from torch_geometric.data import Batch, Data
    

    

    class MaxMinNormalization(BaseTransform):
        def __init__(self, max_values=None, min_values=None):
            """
            Initialize the normalization transform with optional max and min values.
            If not provided, they should be computed from the dataset.
            """
            self.max_values = max_values if max_values is not None else torch.tensor([76.03170776367188, 77.9359130859375, 88.72427368164062], dtype=torch.float)
            self.min_values = min_values if min_values is not None else torch.tensor([-73.90082550048828, -112.23554992675781, -79.38320922851562], dtype=torch.float)

        def __call__(self, data: Data) -> Data:
            """
            Apply min-max normalization to the node features.
            """
            data.x = (data.x - self.min_values) / (self.max_values - self.min_values)
            return data
    
    class FiberGraphDataset(Dataset):
        def __init__(self, 
                     root, 
                     transform = Compose([MaxMinNormalization()]), 
                     pre_transform = None):
            super(FiberGraphDataset, self).__init__(root, transform, pre_transform)
        
        @property
        def processed_dir(self):
            return os.path.join(self.root)

        @property
        def processed_file_names(self):
            return os.listdir(self.root)
        
        def len(self):
            return len(self.processed_file_names)
        
        def get(self, idx):
            subject = self.processed_file_names[idx]# Seleccionar un sujeto
            graphs = torch.load(os.path.join(self.processed_dir, subject))
            
            if self.transform:
                graphs = self.transform(graphs)
            
            return graphs
        

    
        
    

    # importar logger para tensorboard
    import torch
    from torch.utils.tensorboard import SummaryWriter
    import neptune
    from config import CFG
    import time
    import matplotlib.pyplot as plt


    TRACT_LIST = {
            'AF_L': {'id': 0, 'tract': 'arcuate fasciculus', 'side' : 'left', 'type': 'association'},
            'AF_R': {'id': 1, 'tract': 'arcuate fasciculus','side' : 'right', 'type': 'association'},
            'CC_Fr_1': {'id': 2, 'tract': 'corpus callosum, frontal lobe', 'side' : 'most anterior part of the frontal lobe', 'type': 'commissural'},
            'CC_Fr_2': {'id': 3, 'tract': 'corpus callosum, frontal lobe', 'side' : 'most posterior part of the frontal lobe','type': 'commissural'},
            'CC_Oc': {'id': 4, 'tract': 'corpus callosum, occipital lobe', 'side' : 'central', 'type': 'commissural'},
            'CC_Pa': {'id': 5, 'tract': 'corpus callosum, parietal lobe', 'side' : 'central', 'type': 'commissural'},
            'CC_Pr_Po': {'id': 6, 'tract': 'corpus callosum, pre/post central gyri', 'side' : 'central', 'type': 'commissural'},
            'CG_L': {'id': 7, 'tract': 'cingulum', 'side' : 'left', 'type': 'association'},
            'CG_R': {'id': 8, 'tract': 'cingulum', 'side' : 'right', 'type': 'association'},
            'FAT_L': {'id': 9, 'tract': 'frontal aslant tract', 'side' : 'left', 'type': 'association'},
            'FAT_R': {'id': 10, 'tract': 'frontal aslant tract', 'side' : 'right', 'type': 'association'},
            'FPT_L': {'id': 11, 'tract': 'fronto-pontine tract', 'side' : 'left', 'type': 'association'},
            'FPT_R': {'id': 12, 'tract': 'fronto-pontine tract', 'side' : 'right', 'type': 'association'},
            'FX_L': {'id': 13, 'tract': 'fornix', 'side' : 'left', 'type': 'commissural'},
            'FX_R': {'id': 14, 'tract': 'fornix', 'side' : 'right', 'type': 'commissural'},
            'IFOF_L': {'id': 15, 'tract': 'inferior fronto-occipital fasciculus', 'side' : 'left', 'type': 'association'},
            'IFOF_R': {'id': 16, 'tract': 'inferior fronto-occipital fasciculus', 'side' : 'right', 'type': 'association'},
            'ILF_L': {'id': 17, 'tract': 'inferior longitudinal fasciculus', 'side' : 'left', 'type': 'association'},
            'ILF_R': {'id': 18, 'tract': 'inferior longitudinal fasciculus', 'side' : 'right', 'type': 'association'},
            'MCP': {'id': 19, 'tract': 'middle cerebellar peduncle', 'side' : 'central', 'type': 'commissural'},
            'MdLF_L': {'id': 20, 'tract': 'middle longitudinal fasciculus', 'side' : 'left', 'type': 'association'},
            'MdLF_R': {'id': 21, 'tract': 'middle longitudinal fasciculus', 'side' : 'right', 'type': 'association'},
            'OR_ML_L': {'id': 22, 'tract': 'optic radiation, Meyer loop', 'side' : 'left', 'type': 'projection'},
            'OR_ML_R': {'id': 23, 'tract': 'optic radiation, Meyer loop', 'side' : 'right', 'type': 'projection'},
            'POPT_L': {'id': 24, 'tract': 'pontine crossing tract', 'side' : 'left', 'type': 'commissural'},
            'POPT_R': {'id': 25, 'tract': 'pontine crossing tract', 'side' : 'right', 'type': 'commissural'},
            'PYT_L': {'id': 26, 'tract': 'pyramidal tract', 'side' : 'left', 'type': 'projection'},
            'PYT_R': {'id': 27, 'tract': 'pyramidal tract', 'side' : 'right', 'type': 'projection'},
            'SLF_L': {'id': 28, 'tract': 'superior longitudinal fasciculus', 'side' : 'left', 'type': 'association'},
            'SLF_R': {'id': 29, 'tract': 'superior longitudinal fasciculus', 'side' : 'right', 'type': 'association'},
            'UF_L': {'id': 30, 'tract': 'uncinate fasciculus', 'side' : 'left', 'type': 'association'},
            'UF_R': {'id': 31, 'tract': 'uncinate fasciculus', 'side' : 'right', 'type': 'association'}
        }
    LABELS = {value["id"]: key for key, value in TRACT_LIST.items()}# Diccionario id -> Etiqueta
    
        
    def generate_confusion_matrix(cm, class_names = range(32)):
        fig, ax = plt.subplots(figsize=(15, 15))
        im = ax.matshow(cm, cmap="Blues")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion matrix")
        fig.tight_layout()
        return fig
    

    log = True
    # Crear un writer para el logdir
    if log:
        name = f"model_{time.time()}"
        
        run = neptune.init_run(
            project="pablorocamora/multimodal-fiber-classification",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ODA0YzA2NS04MjczLTQyNzItOGE5Mi05ZmI5YjZkMmY3MDcifQ==",
            name=name
        )
        writer = SummaryWriter(f"runs/{name}")


    dataset = FiberGraphDataset(root='/app/dataset/tractoinferno_graphs/testset')
    

    params = {
    "lr": 1e-3,
    "batch_size": CFG.batch_size,
    "epochs": CFG.epochs,
    "temperature": CFG.temperature,
    "graph_model_name": CFG.graph_model_name,
    "text_encoder_model": CFG.text_encoder_model,
    "text_embedding": CFG.text_embedding,
    "graph_embedding": CFG.graph_embedding,
    "graph_channels": CFG.graph_channels,
    "projection_dim": CFG.projection_dim,
    "num_workers": CFG.num_workers
    
    
    }
    if log:
        run["parameters"] = params

    model = GCN_Classifier(graph_model_name = params["graph_model_name"],
                            graph_embedding = params["graph_embedding"],
                            graph_channels = params["graph_channels"],
                            projection_dim = params["projection_dim"],
                            device = CFG.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.95)
    scheduler = ReduceLROnPlateau(optimizer, patience=250, factor=0.95, verbose=True)
    
    accuracy_metric = torchmetrics.Accuracy(num_classes=32, task='multiclass').to(CFG.device)
    f1_metric = torchmetrics.F1Score(num_classes=32, average='weighted', task='multiclass').to(CFG.device)
    precision_metric = torchmetrics.Precision(num_classes=32, average='weighted', task='multiclass').to(CFG.device)
    recall_metric = torchmetrics.Recall(num_classes=32, average='weighted', task='multiclass').to(CFG.device)
    confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=32, task='multiclass').to(CFG.device)

    
  # Assuming accuracy, f1_score, precision, recall are imported and initialized here

for epoch in range(params["epochs"]):
    for subject in dataset:
        dataloader = DataLoader(subject, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'], collate_fn=GeoBatch.from_data_list)
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

        for i, graph_data in progress_bar:
            graph_data = graph_data.to(model.device)
            
            optimizer.zero_grad()
            output = model(graph_data)
            loss = F.nll_loss(output, graph_data.y)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)  # Make sure this matches your scheduler's expected usage
            
            with torch.no_grad():  # Metric calculations
                preds = output.argmax(dim=1)

                accuracy_metric.update(preds, graph_data.y)
                f1_metric.update(preds, graph_data.y)
                precision_metric.update(preds, graph_data.y)
                recall_metric.update(preds, graph_data.y)
                confusion_matrix.update(preds, graph_data.y)
                
                acc_score = accuracy_metric.compute()
                f1_score = f1_metric.compute()
                prec_score = precision_metric.compute()
                rec_score = recall_metric.compute()
                cm = confusion_matrix.compute()


                progress_bar.set_description(f'Epoch {epoch} Loss: {loss.item():.4f} Acc: {acc_score:.4f}')

                

            if log:
                writer.add_scalar('training loss', loss.item(), epoch * len(dataloader) + i)
                run['training loss'].log(loss.item())
                writer.add_scalar('accuracy', acc_score, epoch * len(dataloader) + i)
                run['accuracy'].log(acc_score)
                writer.add_scalar('f1', f1_score, epoch * len(dataloader) + i)
                run['f1'].log(f1_score)
                writer.add_scalar('precision', prec_score, epoch * len(dataloader) + i)
                run['precision'].log(prec_score)
                writer.add_scalar('recall', rec_score, epoch * len(dataloader) + i)
                run['recall'].log(rec_score)

                # writer.add_image('confusion matrix', cm, epoch * len(dataloader) + i)
                # Cada 100 batches subir la matriz de confusión
                if i % 100 == 0:
                    run['confusion matrix'].upload(generate_confusion_matrix(cm))

                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('learning rate', lr, epoch * len(dataloader) + i)
                run['learning rate'].log(lr)


                progress_bar.set_description(f'Epoch {epoch} - loss: {loss.item()} - accuracy: {acc_score} - f1: {f1_score} - precision: {prec_score} - recall: {rec_score} - lr: {lr}')
    run.stop()
    writer.close()

    
    
        
        

    
    
