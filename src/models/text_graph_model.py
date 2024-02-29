from encoders import Text_Encoder, ProjectionHead
from torch_geometric.nn import global_mean_pool, GAE, VGAE
from autoencoders import GCN_Encoder
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch_geometric
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GAE, InnerProductDecoder, global_mean_pool
import torchmetrics
from config import CFG


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
        
        self.text_encoder = Text_Encoder(text_encoder_model) #(batch_size, text_embedding)
        self.graph_encoder = VGAE(GCN_Encoder(graph_channels, graph_embedding)) #(batch_size, text_embedding)
        
        # Proyectar representaciones a un espacio comun de menor dimension
        # self.text_projection_head = ProjectionHead(text_embedding, projection_dim) 
        # self.graph_projection_head = ProjectionHead(graph_embedding, projection_dim)
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
        


    
    def forward(self, graph_batch, text_batch):
        
        # Encode the text and project the representations
        text_projections = self.text_encoder(text_batch) # (batch_size, text_embedding)
        
        # Encode the graph and project the representations
        graph_encoded_projections = self.graph_encoder.encode(graph_batch.x, graph_batch.edge_index) # (num_nodes, graph_embedding)
        
        # Obtener los embeddings a nivel de grafo del batch
        graph_embeddings = global_mean_pool(graph_encoded_projections, graph_batch.batch)# (batch_size, graph_embedding)
        
        # Calculating the reconstruction loss
        loss = self.graph_encoder.recon_loss(graph_encoded_projections, graph_batch.edge_index)# (batch_size)
        
        reconstruction_loss = loss + (1 / graph_batch.num_graphs) * self.graph_encoder.kl_loss(graph_encoded_projections)
        
        
        
        # Calculating the contrastive Loss
        logits = (text_projections @ graph_embeddings.T) / self.temperature # (batch_size, batch_size)
        graphs_similarity = graph_embeddings @ graph_embeddings.T # (batch_size, batch_size)
        texts_similarity = text_projections @ text_projections.T # (batch_size, batch_size)
        targets = F.softmax(
            (graphs_similarity + texts_similarity) / 2 * self.temperature, dim = -1
        )# (batch_size, batch_size)
        texts_loss = self.cross_entropy(logits, targets, reduction='none')# (batch_size)
        graphs_loss = self.cross_entropy(logits.T, targets.T, reduction='none')# (batch_size)
        print(f"Graphs loss: {graphs_loss.mean().item()}, Texts loss: {texts_loss.mean().item()}, Reconstruction loss: {reconstruction_loss.item()}")
        contrastive_loss =  (graphs_loss + texts_loss) / 2.0 # shape: (batch_size)
        
        return contrastive_loss.mean() + reconstruction_loss # (1)
    

    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
        
    # def clasificar(self, graph_batch):
    #     """
    #     Clasifica los logits en una de las posibles etiquetas.
    #     """
    #     self.eval()
    #     results = {}
    #     TRACT_LIST = {
    #         'AF_L': {'id': 0, 'tract': 'arcuate fasciculus', 'side' : 'left', 'type': 'association'},
    #         'AF_R': {'id': 1, 'tract': 'arcuate fasciculus','side' : 'right', 'type': 'association'},
    #         'CC_Fr_1': {'id': 2, 'tract': 'corpus callosum, frontal lobe', 'side' : 'most anterior part of the frontal lobe', 'type': 'commissural'},
    #         'CC_Fr_2': {'id': 3, 'tract': 'corpus callosum, frontal lobe', 'side' : 'most posterior part of the frontal lobe','type': 'commissural'},
    #         'CC_Oc': {'id': 4, 'tract': 'corpus callosum, occipital lobe', 'side' : 'central', 'type': 'commissural'},
    #         'CC_Pa': {'id': 5, 'tract': 'corpus callosum, parietal lobe', 'side' : 'central', 'type': 'commissural'},
    #         'CC_Pr_Po': {'id': 6, 'tract': 'corpus callosum, pre/post central gyri', 'side' : 'central', 'type': 'commissural'},
    #         'CG_L': {'id': 7, 'tract': 'cingulum', 'side' : 'left', 'type': 'association'},
    #         'CG_R': {'id': 8, 'tract': 'cingulum', 'side' : 'right', 'type': 'association'},
    #         'FAT_L': {'id': 9, 'tract': 'frontal aslant tract', 'side' : 'left', 'type': 'association'},
    #         'FAT_R': {'id': 10, 'tract': 'frontal aslant tract', 'side' : 'right', 'type': 'association'},
    #         'FPT_L': {'id': 11, 'tract': 'fronto-pontine tract', 'side' : 'left', 'type': 'association'},
    #         'FPT_R': {'id': 12, 'tract': 'fronto-pontine tract', 'side' : 'right', 'type': 'association'},
    #         'FX_L': {'id': 13, 'tract': 'fornix', 'side' : 'left', 'type': 'commissural'},
    #         'FX_R': {'id': 14, 'tract': 'fornix', 'side' : 'right', 'type': 'commissural'},
    #         'IFOF_L': {'id': 15, 'tract': 'inferior fronto-occipital fasciculus', 'side' : 'left', 'type': 'association'},
    #         'IFOF_R': {'id': 16, 'tract': 'inferior fronto-occipital fasciculus', 'side' : 'right', 'type': 'association'},
    #         'ILF_L': {'id': 17, 'tract': 'inferior longitudinal fasciculus', 'side' : 'left', 'type': 'association'},
    #         'ILF_R': {'id': 18, 'tract': 'inferior longitudinal fasciculus', 'side' : 'right', 'type': 'association'},
    #         'MCP': {'id': 19, 'tract': 'middle cerebellar peduncle', 'side' : 'central', 'type': 'commissural'},
    #         'MdLF_L': {'id': 20, 'tract': 'middle longitudinal fasciculus', 'side' : 'left', 'type': 'association'},
    #         'MdLF_R': {'id': 21, 'tract': 'middle longitudinal fasciculus', 'side' : 'right', 'type': 'association'},
    #         'OR_ML_L': {'id': 22, 'tract': 'optic radiation, Meyer loop', 'side' : 'left', 'type': 'projection'},
    #         'OR_ML_R': {'id': 23, 'tract': 'optic radiation, Meyer loop', 'side' : 'right', 'type': 'projection'},
    #         'POPT_L': {'id': 24, 'tract': 'pontine crossing tract', 'side' : 'left', 'type': 'commissural'},
    #         'POPT_R': {'id': 25, 'tract': 'pontine crossing tract', 'side' : 'right', 'type': 'commissural'},
    #         'PYT_L': {'id': 26, 'tract': 'pyramidal tract', 'side' : 'left', 'type': 'projection'},
    #         'PYT_R': {'id': 27, 'tract': 'pyramidal tract', 'side' : 'right', 'type': 'projection'},
    #         'SLF_L': {'id': 28, 'tract': 'superior longitudinal fasciculus', 'side' : 'left', 'type': 'association'},
    #         'SLF_R': {'id': 29, 'tract': 'superior longitudinal fasciculus', 'side' : 'right', 'type': 'association'},
    #         'UF_L': {'id': 30, 'tract': 'uncinate fasciculus', 'side' : 'left', 'type': 'association'},
    #         'UF_R': {'id': 31, 'tract': 'uncinate fasciculus', 'side' : 'right', 'type': 'association'}
    #     }
    #     LABELS = {value["id"]: key for key, value in TRACT_LIST.items()}# Diccionario id -> Etiqueta

    #     tokenizer = AutoTokenizer.from_pretrained(CFG.text_encoder_model)#'distilbert-base-uncased'
        
    #     for category in ['tract', 'side', 'type']:
            
    #         posible_text_labels = list(set([value[category] for key, value in TRACT_LIST.items()]))
            
    #         tokenized_text_labels = tokenizer(posible_text_labels, padding=True, truncation=True, return_tensors="pt", max_length=512)
    #         tokenized_text_labels = {k: v.to(self.device) for k, v in tokenized_text_labels.items()}
            
    #         text_projections = self.text_encoder(tokenized_text_labels)
    #         # text_projections = self.text_projection_head(text_projections)
            
    #         graph_projections = self.graph_encoder.encode(graph_batch.x, graph_batch.edge_index)
    #         graph_projections = global_mean_pool(graph_projections, graph_batch.batch)
    #         # graph_projections = self.graph_projection_head(graph_projections)
            
    #         graph_embeddings_norm = F.normalize(graph_projections, p=2, dim=-1)
    #         text_embeddings_norm = F.normalize(text_projections, p=2, dim=-1)
            
    #         dot_similarity = text_embeddings_norm @ graph_embeddings_norm.T
    #         max_similarities, max_indices = dot_similarity.max(dim=0)

            
    #         real_labels = graph_batch.y.tolist() #-> [0, 1, ...]
    #         real_labels = [LABELS[label] for label in real_labels] #-> ['AF_L', 'AF_R', ...]
    #         real_labels = [TRACT_LIST[label][category] for label in real_labels] #-> ['arcuate fasciculus', 'arcuate fasciculus', ...]
    #         real_labels = torch.tensor([posible_text_labels.index(label) for label in real_labels]).to(self.device) #-> [0, 1, ...]

    #         results[category] = {
    #             'pred_probs': max_similarities,
    #             'pred_labels': max_indices,
    #             'real_labels': real_labels,
    #             'accuracy': self.metrics[f'accuracy_{category}'](max_indices, real_labels),
    #             'f1': self.metrics[f'f1_{category}'](max_indices, real_labels),
    #             'precision': self.metrics[f'precision_{category}'](max_indices, real_labels),
    #             'recall': self.metrics[f'recall_{category}'](max_indices, real_labels)
    #         }
            
        
    #     return results



            


    

if __name__ == "__main__":
    
    import torchmetrics
    import torch
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from config import CFG
    from torch.utils.tensorboard import SummaryWriter
    import neptune
    import time
    from torch_dataset import FiberGraphDataset, collate_function
    from tqdm import tqdm
    import numpy as np
    
    
    # 1. CONFIGURAR Todo PARA QUE SEA REPRODUCIBLE
    def set_seed(seed=42):
        # set all seeds for reproducibility
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.cuda.seed_all()
        

    set_seed()
    
    # Configuración de los parámetros
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
        "num_workers": CFG.num_workers,
        "log": True
    }

    # Crear un writer para el logdir
    if params["log"]:

        name = f"model_{time.time()}"
        
        run = neptune.init_run(
            project="pablorocamora/multimodal-fiber-classification",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ODA0YzA2NS04MjczLTQyNzItOGE5Mi05ZmI5YjZkMmY3MDcifQ==",
            name=name
        )
        tb_logger = SummaryWriter(f"runs/{name}")

        run["parameters"] = params# Log the parameters to Neptune
        run["config"] = CFG# Log the config to Neptune


    # Crear el dataset
    dataset = FiberGraphDataset(root='/app/dataset/tractoinferno_graphs/testset')
    
    # INSTANCIAR EL MODELO
    model = Multimodal_Text_Graph_Model()

    # INSTANCIAR EL OPTIMIZADOR Y EL SCHEDULER
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    # scheduler = ReduceLROnPlateau(optimizer, patience=1500, factor=0.95, verbose=True)


    # BUCLE DE ENTRENAMIENTO
    for epoch in range(params["epochs"]):
    
        for idx_suj, subject in enumerate(dataset):
            # if idx_suj == 0:
            #     break

            total_loss = 0

            dataloader = DataLoader(subject, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_function, num_workers=params['num_workers'])
            for i, (graph_data, text_data) in enumerate(dataloader):
                
                model.train()
                # send data to device
                graph_data = graph_data.to(model.device)
                text_data = {k: v.to(model.device) for k, v in text_data.items()}

                # reset gradients
                optimizer.zero_grad()

                # forward pass
                loss = model(graph_data, text_data)

                # sum the loss over the batch
                total_loss += loss.item()
                
                # Actualizar la barra de progreso con el batch loss
                print(f'\rBatch {i+1} - Loss: {loss.item()}')

                # backward pass
                loss.backward()
                
                # actualizar los parámetros
                optimizer.step()
                # scheduler.step(loss)

                if params["log"]:
                    run['train/batch/loss'].log(loss.item())# Log the loss to Neptune
            
            # Actualizar el loss del sujeto
            mean_loss = total_loss / len(dataloader)
            
            if params["log"]:
                run['train/subject/loss'].log(loss)

            
            # Actualizar la barra de progreso con el loss del sujeto
            print(f'\rSubject {idx_suj+1} - Loss: {mean_loss}')

        
        
        
        # 5. EVALUAR EL MODELO
        model.eval()
        # seleccionar un sujeto aleatorio para evaluar
        suj_eval = dataset[28]
        total_loss = 0
        dataloader = DataLoader(suj_eval, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_function, num_workers=params['num_workers'])
        progress_bar_validation = tqdm(dataloader, desc="Validation")
        name = f"model_{time.time()}"
        tb_logger = SummaryWriter(f"runs/{name}")
        all_embeddings, all_labels = [], []
        with torch.no_grad():
            for i, (graph_data, text_data) in enumerate(dataloader):
                
                graph_data = graph_data.to(model.device)
                text_data = {k: v.to(model.device) for k, v in text_data.items()}

                # save first 100 batches embeddings
                if i < 100:
                    graph_embeddings = model.graph_encoder.encode(graph_data.x, graph_data.edge_index)
                    graph_embeddings = global_mean_pool(graph_embeddings, graph_data.batch)
                    all_embeddings.append(graph_embeddings.detach().cpu().numpy())
                    all_labels.append(graph_data.y.detach().cpu().numpy())
                    if i == 99:
                        all_embeddings = np.concatenate(all_embeddings, axis=0)
                        all_labels = np.concatenate(all_labels, axis=0)
                        tb_logger.add_embedding(all_embeddings, metadata=all_labels, tag="embeddings")
                        tb_logger.close()
                    

                # forward pass
                loss = model(graph_data, text_data)
                total_loss += loss.item()

                if params["log"]:
                    run['eval/batch/loss'].log(loss.item())

            # Actualizar el loss del sujeto
            mean_loss = total_loss / len(dataloader)
            if params["log"]:
                run['eval/subject/loss'].log(loss)

                


        # 6. GUARDAR EL MODELO SI ES MEJOR QUE EL ANTERIOR
        
        model.train()
    # Cerrar el run de Neptune
    run.stop()
    
    

    
    
        
        

    
    
