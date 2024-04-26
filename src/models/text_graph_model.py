from encoders import TextEncoder, ProjectionHead, ClassifierHead, GCN_Encoder
# from torch_geometric.nn import global_mean_pool, GATv2Conv

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch_geometric
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tabulate import tabulate
import torchmetrics
from config import CFG


class Multimodal_Text_Graph_Model(nn.Module):
    def __init__(self, 
                 text_encoder_name = CFG.text_encoder_name,
                 text_embedding = CFG.text_encoder_embedding,
                 graph_model_name = CFG.graph_encoder_name,
                 graph_embedding = CFG.graph_encoder_graph_embedding,
                 graph_channels = CFG.graph_encoder_input_channels,
                 projection_dim = CFG.projection_head_output_dim,
                 n_classes = CFG.n_classes,
                 device = CFG.device):
        
        super(Multimodal_Text_Graph_Model, self).__init__()
        if graph_model_name == "GraphConvolutionalNetwork":
            self.graph_encoder = GCN_Encoder(graph_channels, graph_embedding)
        
        self.graph_projection_head = ProjectionHead(graph_embedding, projection_dim) #(batch_size, projection_dim)
        self.graph_embedding_classifier = ClassifierHead(projection_dim, n_classes)


        self.text_encoder = TextEncoder(text_encoder_name, trainable=False) #(batch_size, text_embedding)
        self.text_projection_head = ProjectionHead(text_embedding, projection_dim) #(batch_size, projection_dim)
        self.text_embedding_classifier = ClassifierHead(projection_dim, n_classes)
        
        self.device = device
        self.to(device)
    
    def forward(self, graph_batch, text_batch):
        
        # graph_batch = graph_batch.to(self.device)
        # text_batch = {k: v.to(self.device) for k, v in text_batch.items()}
        
        graph_projections = self.graph_encoder(graph_batch) # (batch_size, projection_dim)
        graph_projections = self.graph_projection_head(graph_projections) # (batch_size, projection_dim)

        graph_predicted_labels = self.graph_embedding_classifier(graph_projections) # (batch_size, n_classes)
        
        text_projections = self.text_encoder(text_batch) # (batch_size, text_embedding)
        text_projections = self.text_projection_head(text_projections) # (batch_size, projection_dim)

        text_predicted_labels = self.text_embedding_classifier(text_projections) # (batch_size, n_classes)

        return graph_projections, text_projections, graph_predicted_labels, text_predicted_labels
    

    
    
def get_streamline_weights():
    count_fibras = {0: 746528,
                    1: 329896,
                    2: 2793503,
                    3: 5128144,
                    4: 2059468,
                    5: 2765610,
                    6: 3367322,
                    7: 302274,
                    8: 236299,
                    9: 1263593,
                    10: 988040,
                    11: 659802,
                    12: 716805,
                    13: 569,
                    14: 254,
                    15: 696934,
                    16: 730783,
                    17: 1160519,
                    18: 979649,
                    19: 978293,
                    20: 371638,
                    21: 456242,
                    22: 480974,
                    23: 304006,
                    24: 540280,
                    25: 507532,
                    26: 1308236,
                    27: 1384632,
                    28: 230113,
                    29: 558167,
                    30: 165851,
                    31: 259382}
    weights = []
    for tipo, count in count_fibras.items():
        weights.append(1/count)
    weights = torch.tensor(weights)
    weights = weights/weights.sum()
    return weights

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.classification import multiclass_accuracy

class CategoricalContrastiveLoss(nn.Module):
    def __init__(self, theta=CFG.theta, margin = CFG.margin, dw = CFG.distance, weights = CFG.weighted_loss):
        super(CategoricalContrastiveLoss, self).__init__()
        self.theta = theta
        self.margin = margin  # margen
        self.classification_loss = nn.CrossEntropyLoss(weight=get_streamline_weights() if weights else None)
        self.dw = dw

    def forward(self, graph_emb, text_emb, graph_label, text_label, graph_pred_label, text_pred_label, y):
        # Calcula la pérdida de disimilitud como la distancia euclídea
        if self.dw == 'euclidean':
            graph_emb = F.normalize(graph_emb, p=2, dim=1)
            text_emb = F.normalize(text_emb, p=2, dim=1)
            dw = torch.norm(graph_emb - text_emb, p=2, dim=1)
        elif self.dw == 'cosine':
            graph_emb = F.normalize(graph_emb, p=2, dim=1)
            text_emb = F.normalize(text_emb, p=2, dim=1)
            dw = 1 - F.cosine_similarity(graph_emb, text_emb, dim=1)

        loss_similar = (1 - y) * torch.pow(dw, 2)# Pérdida para pares similares: Ls(Dw) = Dw^2
        loss_dissimilar = y * torch.pow(F.relu(self.margin - dw), 2)# Pérdida para pares disímiles: Ld(Dw) = max(0, m - Dw)^2
        contrastive_loss = loss_similar + loss_dissimilar
        
        graph_classification_loss = self.classification_loss(graph_pred_label, graph_label)
        text_classification_loss = self.classification_loss(text_pred_label, text_label)
        
        loss = contrastive_loss + self.theta * (graph_classification_loss + text_classification_loss)

        if True:#self.training
            run['train/contrastive_loss'].log(contrastive_loss.mean().item())
            run['train/graph_classification_loss'].log(graph_classification_loss.mean().item())
            run['train/text_classification_loss'].log(text_classification_loss.mean().item())
            run['train/loss'].log(loss.mean().item())

        return loss.mean()


def train(model, data_loader, optimizer, criterion, device="cuda"):
    model.train()  # Pone el modelo en modo entrenamiento
    loss_list, accuracy_g_list, accuracy_t_list = [], [], []

    for i, (graph_data, text_data, graph_label, text_label, type_of_pair) in enumerate(data_loader):
        
        # Enviar datos al dispositivo de una vez, incluyendo diccionarios de manera eficiente
        graph_data = graph_data.to(device)
        text_data = {k: v.to(device) for k, v in text_data.items()}
        graph_label = graph_label.to(device)
        text_label = text_label.to(device)
        type_of_pair = type_of_pair.to(device)

        optimizer.zero_grad()  # Reinicia los gradientes

        # Realiza el forward pass
        g_proj, t_proj, g_pred_lab, t_pred_lab = model(graph_data, text_data)
        loss = criterion(g_proj, t_proj, graph_label, text_label, g_pred_lab, t_pred_lab, type_of_pair)
        
        # Realiza el backward pass y actualiza los pesos
        loss.backward()
        optimizer.step()

        batch_graph_accuracy = multiclass_accuracy(g_pred_lab, graph_label, 32, average='micro')
        batch_text_accuracy = multiclass_accuracy(t_pred_lab, text_label, 32, average='micro')

        loss_list.append(loss.item())
        accuracy_g_list.append(batch_graph_accuracy.item())
        accuracy_t_list.append(batch_text_accuracy.item())


        if params["log"]:
            run['train/batch/graph_accuracy'].log(batch_graph_accuracy)
            run['train/batch/text_accuracy'].log(batch_text_accuracy)
            run['train/batch/loss'].log(loss)

    avg_loss = np.mean(loss_list)
    accuracy_g = np.mean(accuracy_g_list)
    accuracy_t = np.mean(accuracy_t_list)

    return avg_loss, accuracy_g, accuracy_t

def validate(model, data_loader, criterion, device):
    model.eval()  # Pone el modelo en modo evaluación
    loss_list, accuracy_g_list, accuracy_t_list = [], [], []

    with torch.no_grad():
        for i, (graph_data, text_data, graph_label, text_label, type_of_pair) in enumerate(data_loader):
            graph_data = graph_data.to(device)
            text_data = {k: v.to(device) for k, v in text_data.items}
            graph_label = graph_label.to(device)
            text_label = text_label.to(device)
            type_of_pair = type_of_pair.to(device)

            g_proj, t_proj, g_pred_lab, t_pred_lab = model(graph_data, text_data)
            loss = criterion(g_proj, t_proj, graph_label, text_label, g_pred_lab, t_pred_lab, type_of_pair)

            batch_graph_accuracy = multiclass_accuracy(g_pred_lab, graph_label, 32, average='micro')
            batch_text_accuracy = multiclass_accuracy(t_pred_lab, text_label, 32, average='micro')

            loss_list.append(loss.item())
            accuracy_g_list.append(batch_graph_accuracy.item())
            accuracy_t_list.append(batch_text_accuracy.item())

            if params["log"]:
                run['validation/batch/graph_accuracy'].log(batch_graph_accuracy)
                run['validation/batch/text_accuracy'].log(batch_text_accuracy)
                run['validation/batch/loss'].log(loss)

    avg_loss = np.mean(loss_list)
    accuracy_g = np.mean(accuracy_g_list)
    accuracy_t = np.mean(accuracy_t_list)

    return avg_loss, accuracy_g, accuracy_t

    

if __name__ == "__main__":
    
    import torchmetrics
    import torch
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from config import CFG
    # from torch.utils.tensorboard import SummaryWriter
    import neptune
    import time
    from torch_dataset import FiberGraphDataset, collate_function_v2
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
    config = CFG()
    
    # Convert the config to a dictionary
    params = {name: value for name, value in CFG.__dict__.items() if not name.startswith("__")}

    # Imprimir los parámetros con tabulate
    print(tabulate(params.items(), tablefmt="fancy_grid"))
    

    # Crear un writer para el logdir
    if params["log"]:

        name = f"model_{time.time()}"
        
        run = neptune.init_run(
            project="pablorocamora/multimodal-fiber-classification",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ODA0YzA2NS04MjczLTQyNzItOGE5Mi05ZmI5YjZkMmY3MDcifQ==",
            name=name
        )

        run["parameters"] = params# Log the parameters to Neptune
        run["config"] = CFG# Log the config to Neptune


    # Crear el dataset
    train_dataset = FiberGraphDataset(root='/app/dataset/Tractoinferno/tractoinferno_graphs/trainset')
    val_dataset = FiberGraphDataset(root='/app/dataset/Tractoinferno/tractoinferno_graphs/validset')

    model = Multimodal_Text_Graph_Model()
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=params['learning_rate'], 
                                  weight_decay=1e-3)
    criterion = CategoricalContrastiveLoss(theta=params['theta'], 
                                           margin=params['margin'], 
                                           dw=params['distance'], 
                                           weights=params['weighted_loss'])
    criterion.to(model.device)



    # BUCLE DE ENTRENAMIENTO
    for epoch in range(params['train_epochs']):
        loss, g_acc, t_acc = [], [], []
        # Entrenar el modelo
        for idx_suj, subject in enumerate(train_dataset):
            dataloader = DataLoader(subject, 
                                    batch_size=params['train_batch_size'], 
                                    shuffle=True, 
                                    collate_fn=collate_function_v2, 
                                    num_workers=params['train_num_workers'], 
                                    pin_memory=False, 
                                    drop_last=False)
            
            avg_subj_loss, accuracy_subj_g, accuracy_subj_t = train(model, 
                                                                    dataloader, 
                                                                    optimizer, 
                                                                    criterion, 
                                                                    device=model.device)
            

            print(f"\r [Train] Epoch {epoch} - Sujeto {idx_suj} - Loss: {avg_subj_loss} - Accuracy Graph: {accuracy_subj_g} - Accuracy Text: {accuracy_subj_t}")
            
            if params["log"]:
                run['train/subject/graph_accuracy'].log(accuracy_subj_g)
                run['train/subject/text_accuracy'].log(accuracy_subj_t)
                run['train/subject/loss'].log(avg_subj_loss)

            loss.append(avg_subj_loss)
            g_acc.append(accuracy_subj_g)
            t_acc.append(accuracy_subj_t)
        
        avg_loss = np.mean(loss)
        avg_accuracy_g = np.mean(g_acc)
        avg_accuracy_t = np.mean(t_acc)

        if params["log"]:
            run['train/epoch/graph_accuracy'].log(avg_accuracy_g)
            run['train/epoch/text_accuracy'].log(avg_accuracy_t)
            run['train/epoch/loss'].log(avg_loss)

            
        loss, g_acc, t_acc = [], [], []
        # Validar el modelo
        for idx_suj, subject in enumerate(val_dataset):
            dataloader = DataLoader(subject, 
                                    batch_size=params['train_batch_size'], 
                                    shuffle=True, 
                                    collate_fn=collate_function_v2, 
                                    num_workers=params['train_num_workers'], 
                                    pin_memory=False, 
                                    drop_last=False)
            
            avg_loss, accuracy_g, accuracy_t = validate(model, 
                                                        dataloader, 
                                                        criterion, 
                                                        device=model.device)
            
            print(f"\r[Validation] Epoch {epoch} - Sujeto {idx_suj} - Loss: {avg_loss} - Accuracy Graph: {accuracy_g} - Accuracy Text: {accuracy_t}")

            if params["log"]:
                run['validation/subject/graph_accuracy'].log(accuracy_g)
                run['validation/subject/text_accuracy'].log(accuracy_t)
                run['validation/subject/loss'].log(avg_loss)

            
            loss.append(avg_loss)
            g_acc.append(accuracy_g)
            t_acc.append(accuracy_t)
        
        avg_loss = np.mean(loss)
        avg_accuracy_g = np.mean(g_acc)
        avg_accuracy_t = np.mean(t_acc)

        if params["log"]:
            run['validation/epoch/graph_accuracy'].log(avg_accuracy_g)
            run['validation/epoch/text_accuracy'].log(avg_accuracy_t)
            run['validation/epoch/loss'].log(avg_loss)



        
    run.stop()
    
    

    
    
        
        

    
    
