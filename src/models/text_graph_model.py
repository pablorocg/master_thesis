from encoders import GCN_Encoder, GAT_Encoder, CFG, Text_Encoder, ProjectionHead
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch_geometric
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Multimodal_Text_Graph_Model(nn.Module):
    def __init__(self, 
                 text_encoder_model = CFG.text_encoder_model,
                 text_embedding = CFG.text_embedding,
                 graph_model_name = CFG.graph_model_name,
                 graph_embedding = CFG.graph_embedding,
                 graph_channels = CFG.graph_channels,
                 projection_dim = CFG.projection_dim,
                 temperature = CFG.temperature,
                 device = CFG.device):
        
        super(Multimodal_Text_Graph_Model, self).__init__()
        
        self.text_encoder = Text_Encoder(text_encoder_model)#, text_embedding)
        
        if graph_model_name == "GraphConvolutionalNetwork":
            self.graph_encoder = GCN_Encoder(graph_channels, graph_embedding)
        elif graph_model_name == "GraphAttentionNetwork":
            self.graph_encoder = GAT_Encoder(graph_channels, graph_embedding)
        else:
            raise ValueError("Invalid graph model name")
        
        # Proyectar representaciones a un espacio comun de menor dimension
        self.text_projection_head = ProjectionHead(text_embedding, projection_dim) 
        self.graph_projection_head = ProjectionHead(graph_embedding, projection_dim)
        self.temperature = temperature
        
        self.device = device

        self.to(device)

        # Configuracion de metricas para el modelo
        self.metrics = {
            'accuracy_tract': torchmetrics.Accuracy(num_classes=18, task='multiclass').to(device),
            'accuracy_side': torchmetrics.Accuracy(num_classes=5, task='multiclass').to(device),
            'accuracy_type': torchmetrics.Accuracy(num_classes=3, task='multiclass').to(device),

            'f1_tract': torchmetrics.F1Score(num_classes=18, average='weighted', task='multiclass').to(device),
            'f1_side': torchmetrics.F1Score(num_classes=5, average='weighted', task='multiclass').to(device),
            'f1_type': torchmetrics.F1Score(num_classes=3, average='weighted', task='multiclass').to(device),

            'precision_tract': torchmetrics.Precision(num_classes=18, average='weighted', task='multiclass').to(device),
            'precision_side': torchmetrics.Precision(num_classes=5, average='weighted', task='multiclass').to(device),
            'precision_type': torchmetrics.Precision(num_classes=3, average='weighted', task='multiclass').to(device),

            'recall_tract': torchmetrics.Recall(num_classes=18, average='weighted', task='multiclass').to(device),
            'recall_side': torchmetrics.Recall(num_classes=5, average='weighted', task='multiclass').to(device),
            'recall_type': torchmetrics.Recall(num_classes=3, average='weighted', task='multiclass').to(device)
        }
        


    
    def forward(self, graph, text):
        # Encode the text and project the representations
        text_projections = self.text_encoder(text) # (batch_size, text_embedding)
        graph_projections = self.graph_encoder(graph) # (batch_size, graph_embedding)
        # print(f'Text Shape: {text_projections.shape}')
        # print(f'Graph Shape: {graph_projections.shape}')
        
        text_projections = self.text_projection_head(text_projections) # (batch_size, projection_dim)
        graph_projections = self.graph_projection_head(graph_projections) # (batch_size, projection_dim)
        # print(f'Text Shape: {text_projections.shape}')
        # print(f'Graph Shape: {graph_projections.shape}')
        
        # Calculating the Loss
        logits = (text_projections @ graph_projections.T) / self.temperature # (batch_size, batch_size)
        graphs_similarity = graph_projections @ graph_projections.T # (batch_size, batch_size)
        texts_similarity = text_projections @ text_projections.T # (batch_size, batch_size)
        targets = F.softmax(
            (graphs_similarity + texts_similarity) / 2 * self.temperature, dim = -1
        )# (batch_size, batch_size)
        texts_loss = self.cross_entropy(logits, targets, reduction='none')# (batch_size)
        graphs_loss = self.cross_entropy(logits.T, targets.T, reduction='none')# (batch_size)
        loss =  (graphs_loss + texts_loss) / 2.0 # shape: (batch_size)
        
        return loss.mean()
    

    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
        
    def clasificar(self, graph_batch):
        """
        Clasifica los logits en una de las posibles etiquetas.
        """
        results = {}
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

        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        for category in ['tract', 'side', 'type']:
            
            posible_text_labels = list(set([value[category] for key, value in TRACT_LIST.items()]))
            
            tokenized_text_labels = tokenizer(posible_text_labels, padding=True, truncation=True, return_tensors="pt")
            tokenized_text_labels = {k: v.to(self.device) for k, v in tokenized_text_labels.items()}
            
            text_projections = self.text_encoder(tokenized_text_labels)
            text_projections = self.text_projection_head(text_projections)
            
            graph_projections = self.graph_encoder(graph_batch)
            graph_projections = self.graph_projection_head(graph_projections)
            
            graph_embeddings_norm = F.normalize(graph_projections, p=2, dim=-1)
            text_embeddings_norm = F.normalize(text_projections, p=2, dim=-1)
            
            dot_similarity = text_embeddings_norm @ graph_embeddings_norm.T
            max_similarities, max_indices = dot_similarity.max(dim=0)

            
            real_labels = graph_batch.y.tolist() #-> [0, 1, ...]
            real_labels = [LABELS[label] for label in real_labels] #-> ['AF_L', 'AF_R', ...]
            real_labels = [TRACT_LIST[label][category] for label in real_labels] #-> ['arcuate fasciculus', 'arcuate fasciculus', ...]
            real_labels = torch.tensor([posible_text_labels.index(label) for label in real_labels]).to(self.device) #-> [0, 1, ...]

            results[category] = {
                'pred_probs': max_similarities,
                'pred_labels': max_indices,
                'real_labels': real_labels,
                'accuracy': self.metrics[f'accuracy_{category}'](max_indices, real_labels),
                'f1': self.metrics[f'f1_{category}'](max_indices, real_labels),
                'precision': self.metrics[f'precision_{category}'](max_indices, real_labels),
                'recall': self.metrics[f'recall_{category}'](max_indices, real_labels)
            }
            
        
        return results

            


    

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
    # import os
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    

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
        

    def collate_function(batch):
        """Funcion para el DataLoader"""
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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
        caption_templates = [
                "A {type} fiber",
                "A {type} fiber on the {side} side",
                "{type} fiber on the {side} side",
                "A {type} fiber of the {tract}",
                "{type} fiber of the {tract}",
                "A {type} fiber of the {tract} on the {side} side",
                "{type} fiber of the {tract} on the {side} side",
                "{side} side",
                "{tract} tract",
                "{type} fiber",
                "The {type} fiber located in the {tract} tract",
                "This is a {type} fiber found on the {side} hemisphere",
                "Detailed view of a {type} fiber within the {tract}",
                "Observation of the {type} fiber, prominently on the {side} side",
                "The {tract} tract's remarkable {type} fiber",
                "Characteristics of a {type} fiber in the {tract} region",
                "Notable {type} fiber on the {side} hemisphere of the {tract}",
                "Insight into the {type} fiber's structure on the {side} side",
                "Exploring the complexity of the {type} fiber in the {tract}",
                "The anatomy of a {type} fiber on the {side} hemisphere",
                "The {tract} tract featuring a {type} fiber",
                "A comprehensive look at the {type} fiber, {side} orientation",
                "A closer look at the {type} fiber's path in the {tract}",
                "Unveiling the {type} fiber's role in the {tract} tract",
                "Decoding the structure of the {type} fiber on the {side}",
                "Highlighting the {type} fiber's significance in the {tract}",
                "The {type} fiber: A journey through the {tract} on the {side}",
                "A deep dive into the {type} fiber's dynamics in the {tract}",
                "The {type} fiber's contribution to {tract} tract functionality",
                "Mapping the {type} fiber's trajectory in the {tract} on the {side} side",
                "Navigating the intricate pathways of the {type} fiber within the {tract}",
                "The interplay of {type} fibers across the {side} hemisphere",
                "Traversing the {tract} with a {type} fiber",
                "The pivotal role of the {type} fiber in connecting the {tract}",
                "Showcasing the unique texture of {type} fibers in the {tract}",
                "Zooming in on the {type} fiber's impact on the {side} hemisphere",
                "The {type} fiber in the {tract}",
                "The {type} fiber as a conduit in the {tract} on the {side} side",
                "The {type} fiber's architectural marvel within the {tract}",
                "A journey alongside the {type} fiber through the {tract}",
                "The harmonious structure of the {type} fiber in the {tract}",
                "Unraveling the secrets of the {type} fiber in the {tract} tract",
                "The {type} fiber: A key player in {tract} dynamics",
                "Envisioning the {type} fiber's pathway in the {tract}",
                "The strategic placement of the {type} fiber in the {tract}",
                "Illuminating the {type} fiber's route through the {tract}",
                "The {type} fiber: An essential bridge within the {tract}",
                "Deciphering the network of {type} fibers in the {tract}",
                "Exploring the synergy between {type} fibers and the {tract}",
                "The {type} fiber's vital link in the neural network of the {tract}",
                "The {type} fiber's role in the {tract} on the {side} side",
                "The {type} fiber's intricate design within the {tract}",
                "The {type} fiber's impact on the {tract} on the {side} side",
                "The {type} fiber's influence on the {tract}",
                "The {type} fiber's significance in the {tract}",
                "The {type} fiber's contribution to the {tract}",
                "The {type} fiber's role in the {tract}",
                "The {type} fiber's function in the {tract}",
                "The {type} fiber's purpose in the {tract}",
                "The {type} fiber's importance in the {tract}",
                "A fiber of the {tract} tract located on the {side} side",
                "A {type} fiber located on the {side} side of the {tract} tract",
                "A {type} fiber located on the {side} side of the {tract}",
                "A {type} fiber located on the {side} side",
                "{type}",
                "{tract}",
                "{side} side",
                "{type}, {side}, {tract}",
                "{type}, {tract}",
                "{type}, {side}",
                "{tract}, {type}, {side}",

            ]
        
        
        # Extraer los labels de todos los grafos en el lote
        labels = [graph.y.item() for graph in batch]  # Asumiendo que `y` es el tensor de labels
        
        # Recuperar y tokenizar todos los captions necesarios en una sola llamada
        captions = [random.choice(caption_templates).format(**TRACT_LIST[LABELS[label]]) for label in labels]
        tokenized_texts_batch = tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
        

        return GeoBatch.from_data_list(batch), {'input_ids': tokenized_texts_batch['input_ids'], 'attention_mask': tokenized_texts_batch['attention_mask']}
        
    

    # importar logger para tensorboard
    import torch
    from torch.utils.tensorboard import SummaryWriter
    import neptune
    from config import CFG
    import time
     
        

    

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

    model = Multimodal_Text_Graph_Model()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.95)
    scheduler = ReduceLROnPlateau(optimizer, patience=100, factor=0.95, verbose=True)
    
    for epoch in range(params["epochs"]):
        for subject in dataset:
            dataloader = DataLoader(subject, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_function, num_workers=params['num_workers'])
            for i, (graph_data, text_data) in enumerate(dataloader):
                graph_data = graph_data.to(model.device)
                text_data = {k: v.to(model.device) for k, v in text_data.items()}
                optimizer.zero_grad()
                loss = model(graph_data, text_data)
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                with torch.no_grad():
                    results = model.clasificar(graph_data)
                if log:
                    writer.add_scalar('training loss', loss.item(), epoch * len(dataloader) + i)
                    run['training loss'].log(loss.item())
                    for category, metrics in results.items():
                        for metric_name, metric_value in metrics.items():
                            
                            if metric_name not in ['pred_probs', 'pred_labels', 'real_labels']:
                                writer.add_scalar(f'{metric_name}/{category}', metric_value.item(), epoch * len(dataloader) + i)
                                # Log the metrics to neptune
                                run[f'{metric_name}/{category}'].log(metric_value.item())
                    
                    # Obtener el valor del learning rate del optimizador para comprobar que se actualiza
                    lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning rate', lr, epoch * len(dataloader) + i)
                    run['learning rate'].log(lr)

                print(f"Epoch {epoch}, batch {i}, loss {loss.item()}")
    run.stop()
    writer.close()

    
    
        
        

    
    
