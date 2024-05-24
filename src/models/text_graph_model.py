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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.classification import (multiclass_accuracy, 
                                                    multiclass_f1_score,
                                                    multiclass_auroc)


class Multimodal_Text_Graph_Model(nn.Module):
    def __init__(self, 
                 text_encoder_name = CFG.text_encoder_name,
                 text_embedding = CFG.text_encoder_embedding,
                 graph_embedding = CFG.graph_encoder_graph_embedding,
                 graph_channels = CFG.graph_encoder_input_channels,
                 projection_dim = CFG.projection_head_output_dim,
                 n_classes = CFG.n_classes):
        super(Multimodal_Text_Graph_Model, self).__init__()
        
        self.graph_encoder = GCN_Encoder(graph_channels, graph_embedding)
        self.graph_projection_head = ProjectionHead(graph_embedding, projection_dim) #(batch_size, projection_dim)
        self.graph_embedding_classifier = ClassifierHead(projection_dim, n_classes)


        self.text_encoder = TextEncoder(text_encoder_name, trainable=False) #(batch_size, text_embedding)
        self.text_projection_head = ProjectionHead(text_embedding, projection_dim) #(batch_size, projection_dim)
        self.text_embedding_classifier = ClassifierHead(projection_dim, n_classes)
    
    def forward(self, graph_batch, text_batch):
        graph_projections = self.graph_encoder(graph_batch) # (batch_size, projection_dim)
        graph_projections = self.graph_projection_head(graph_projections) # (batch_size, projection_dim)
        graph_predicted_labels = self.graph_embedding_classifier(graph_projections) # (batch_size, n_classes)
        
        text_projections = self.text_encoder(text_batch) # (batch_size, text_embedding)
        text_projections = self.text_projection_head(text_projections) # (batch_size, projection_dim)
        text_predicted_labels = self.text_embedding_classifier(text_projections) # (batch_size, n_classes)

        return graph_projections, text_projections, graph_predicted_labels, text_predicted_labels
    
    
class CategoricalContrastiveLoss(nn.Module):
    def __init__(self, theta=CFG.theta, margin = CFG.margin, dw = CFG.distance):
        super(CategoricalContrastiveLoss, self).__init__()
        self.theta = theta
        self.margin = margin
        self.classification_loss = nn.CrossEntropyLoss()#weight=self.get_streamline_weights()
        self.dw = dw

    def forward(self, graph_emb, text_emb, graph_label, text_label, graph_pred_label, text_pred_label, y):
        if self.dw == 'euclidean':
            graph_emb = F.normalize(graph_emb, p=2, dim=1)
            text_emb = F.normalize(text_emb, p=2, dim=1)
            dw = F.pairwise_distance(text_emb, graph_emb, p=2.0, eps=1e-6, keepdim=False)

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
        return loss.mean()
    
    def get_streamline_weights(self):
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
    

    
    





def train(model, data_loader, optimizer, criterion, device="cuda", log=True):
    
    # loss_list, accuracy_g_list, accuracy_t_list = [], [], []
    metrics = {
        "loss": [],
        "graph_accuracy": [],
        "text_accuracy": [],
        "graph_f1": [],
        "text_f1": [],
        "graph_auc": [],
        "text_auc": []
    }
    model.train()
    for graph_batch, text_batch, graph_label, text_label, type_of_pair in data_loader:


        graph_batch = graph_batch.to(device)
        text_batch = {k: v.to(device) for k, v in text_batch.items()}
        graph_label = graph_label.to(device)
        text_label = text_label.to(device)
        type_of_pair = type_of_pair.to(device)


        optimizer.zero_grad()  # Reinicia los gradientes

        # Realiza el forward pass
        g_proj, t_proj, g_pred_lab, t_pred_lab = model(graph_batch, text_batch)
        
        loss = criterion(g_proj, t_proj, graph_label, text_label, g_pred_lab, t_pred_lab, type_of_pair)
        
        batch_graph_accuracy = multiclass_accuracy(g_pred_lab, graph_label, 32, average='macro')
        batch_text_accuracy = multiclass_accuracy(t_pred_lab, text_label, 32, average='macro')
        batch_graph_f1 = multiclass_f1_score(g_pred_lab, graph_label, 32, average='macro')
        batch_text_f1 = multiclass_f1_score(t_pred_lab, text_label, 32, average='macro')
        batch_auc_graph = multiclass_auroc(g_pred_lab, graph_label, 32, average='macro')
        batch_auc_text = multiclass_auroc(t_pred_lab, text_label, 32, average='macro')


        loss.backward()
        optimizer.step()

        metrics["loss"].append(loss.item())
        metrics["graph_accuracy"].append(batch_graph_accuracy.item())
        metrics["text_accuracy"].append(batch_text_accuracy.item())
        metrics["graph_f1"].append(batch_graph_f1.item())
        metrics["text_f1"].append(batch_text_f1.item())
        metrics["graph_auc"].append(batch_auc_graph.item())
        metrics["text_auc"].append(batch_auc_text.item())

        if log:
            run['train/batch/graph_accuracy'].log(batch_graph_accuracy.item())
            run['train/batch/text_accuracy'].log(batch_text_accuracy.item())
            run['train/batch/graph_f1'].log(batch_graph_f1.item())
            run['train/batch/text_f1'].log(batch_text_f1.item())
            run['train/batch/graph_auc'].log(batch_auc_graph.item())
            run['train/batch/text_auc'].log(batch_auc_text.item())
            run['train/batch/loss'].log(loss.item())

    # Calcular la media de las métricas del diccionario
    avg_loss = np.mean(metrics["loss"])
    avg_graph_accuracy = np.mean(metrics["graph_accuracy"])
    avg_text_accuracy = np.mean(metrics["text_accuracy"])
    avg_graph_f1 = np.mean(metrics["graph_f1"])
    avg_text_f1 = np.mean(metrics["text_f1"])
    avg_graph_auc = np.mean(metrics["graph_auc"])
    avg_text_auc = np.mean(metrics["text_auc"])

    return {'loss': avg_loss,
            'graph_acc': avg_graph_accuracy,
            'text_acc': avg_text_accuracy,
            'graph_f1': avg_graph_f1,
            'text_f1': avg_text_f1,
            'graph_auc': avg_graph_auc,
            'text_auc': avg_text_auc}

    

def validate(model, data_loader, criterion, device="cuda", log=True):
    metrics = {
        "loss": [],
        "graph_accuracy": [],
        "text_accuracy": [],
        "graph_f1": [],
        "text_f1": [],
        "graph_auc": [],
        "text_auc": []
    }

    model.eval()
    with torch.no_grad():
        for graph_batch, text_batch, graph_label, text_label, type_of_pair in data_loader:
            
            graph_batch = graph_batch.to(device)
            text_batch = {k: v.to(device) for k, v in text_batch.items()}
            graph_label = graph_label.to(device)
            text_label = text_label.to(device)
            type_of_pair = type_of_pair.to(device)

            g_proj, t_proj, g_pred_lab, t_pred_lab = model(graph_batch, text_batch)
            loss = criterion(g_proj, t_proj, graph_label, text_label, g_pred_lab, t_pred_lab, type_of_pair)

            batch_graph_accuracy = multiclass_accuracy(g_pred_lab, graph_label, 32, average='macro')
            batch_text_accuracy = multiclass_accuracy(t_pred_lab, text_label, 32, average='macro')
            batch_graph_f1 = multiclass_f1_score(g_pred_lab, graph_label, 32, average='macro')
            batch_text_f1 = multiclass_f1_score(t_pred_lab, text_label, 32, average='macro')
            batch_auc_graph = multiclass_auroc(g_pred_lab, graph_label, 32, average='macro')
            batch_auc_text = multiclass_auroc(t_pred_lab, text_label, 32, average='macro')

            metrics["loss"].append(loss.item())
            metrics["graph_accuracy"].append(batch_graph_accuracy.item())
            metrics["text_accuracy"].append(batch_text_accuracy.item())
            metrics["graph_f1"].append(batch_graph_f1.item())
            metrics["text_f1"].append(batch_text_f1.item())
            metrics["graph_auc"].append(batch_auc_graph.item())
            metrics["text_auc"].append(batch_auc_text.item())

            if log:
                run['validation/batch/graph_accuracy'].log(batch_graph_accuracy.item())
                run['validation/batch/text_accuracy'].log(batch_text_accuracy.item())
                run['validation/batch/graph_f1'].log(batch_graph_f1.item())
                run['validation/batch/text_f1'].log(batch_text_f1.item())
                run['validation/batch/graph_auc'].log(batch_auc_graph.item())
                run['validation/batch/text_auc'].log(batch_auc_text.item())
                run['validation/batch/loss'].log(loss.item())

    avg_loss = np.mean(metrics["loss"])
    avg_graph_accuracy = np.mean(metrics["graph_accuracy"])
    avg_text_accuracy = np.mean(metrics["text_accuracy"])
    avg_graph_f1 = np.mean(metrics["graph_f1"])
    avg_text_f1 = np.mean(metrics["text_f1"])
    avg_graph_auc = np.mean(metrics["graph_auc"])
    avg_text_auc = np.mean(metrics["text_auc"])

    return {'loss': avg_loss, 
            'graph_acc': avg_graph_accuracy,
            'text_acc': avg_text_accuracy,
            'graph_f1': avg_graph_f1,
            'text_f1': avg_text_f1,
            'graph_auc': avg_graph_auc,
            'text_auc': avg_text_auc}




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
    test_dataset = FiberGraphDataset(root='/app/dataset/Tractoinferno/tractoinferno_graphs/testset')

    model = Multimodal_Text_Graph_Model()
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=params['learning_rate'], 
                                  weight_decay=1e-3)
    criterion = CategoricalContrastiveLoss(theta=params['theta'], 
                                           margin=params['margin'], 
                                           dw=params['distance'])
    model.to("cuda")
    criterion.to("cuda")



    # BUCLE DE ENTRENAMIENTO
    for epoch in range(params['train_epochs']):
        
        # Entrenar el modelo
        for idx_train_suj, train_subject in enumerate(train_dataset):
            
            train_dataloader = DataLoader(train_subject, 
                                    batch_size=params['train_batch_size'], 
                                    shuffle=True, 
                                    collate_fn=collate_function_v2, 
                                    num_workers=params['train_num_workers'], 
                                    pin_memory=False, 
                                    drop_last=False)
            
            metrics = train(model, train_dataloader, optimizer, criterion, device="cuda")
            

            print(f"""[Train] Epoch {epoch + 1} - Subj {idx_train_suj + 1} - Loss: {metrics['loss']} 
                            - Acc Graph: {metrics['graph_acc']} - Acc Text: {metrics['text_acc']} 
                            - F1 Graph: {metrics['graph_f1']} - F1 Text: {metrics['text_f1']} 
                            - AUC Graph: {metrics['graph_auc']} - AUC Text: {metrics['text_auc']}""")
            
            if params["log"]:
                run['train/subject/graph_accuracy'].log(metrics['graph_acc'])
                run['train/subject/text_accuracy'].log(metrics['text_acc'])
                run['train/subject/graph_f1'].log(metrics['graph_f1'])
                run['train/subject/text_f1'].log(metrics['text_f1'])
                run['train/subject/graph_auc'].log(metrics['graph_auc'])
                run['train/subject/text_auc'].log(metrics['text_auc'])
                run['train/subject/loss'].log(metrics['loss'])

            # Limpiar memoria de la GPU
            torch.cuda.empty_cache()
            del train_dataloader

            # Validar cada 5 sujetos
            if (idx_train_suj + 1) % 5 == 0:

                # Validar el modelo
                for idx_val_suj, validation_subject in enumerate(val_dataset):
                    if idx_val_suj == 3:
                        break
                    validation_dataloader = DataLoader(validation_subject, 
                                            batch_size=params['train_batch_size'], 
                                            shuffle=True, 
                                            collate_fn=collate_function_v2, 
                                            num_workers=params['train_num_workers'], 
                                            pin_memory=False, 
                                            drop_last=False)
                    
                    metrics = validate(model, validation_dataloader, criterion, device="cuda")
                    
                    print(f"""[Validation] Epoch {epoch + 1} - Subj {idx_val_suj + 1} - Loss: {metrics['loss']} 
                                    - Acc Graph: {metrics['graph_acc']} - Acc Text: {metrics['text_acc']} 
                                    - F1 Graph: {metrics['graph_f1']} - F1 Text: {metrics['text_f1']} 
                                    - AUC Graph: {metrics['graph_auc']} - AUC Text: {metrics['text_auc']}""")

                    if params["log"]:
                        run['validation/subject/graph_accuracy'].log(metrics['graph_acc'])
                        run['validation/subject/text_accuracy'].log(metrics['text_acc'])
                        run['validation/subject/graph_f1'].log(metrics['graph_f1'])
                        run['validation/subject/text_f1'].log(metrics['text_f1'])
                        run['validation/subject/graph_auc'].log(metrics['graph_auc'])
                        run['validation/subject/text_auc'].log(metrics['text_auc'])
                        run['validation/subject/loss'].log(metrics['loss'])

                    # Limpiar memoria de la GPU
                    torch.cuda.empty_cache()
                    del validation_dataloader

    # Evaluar el conjunto de test
    for idx_suj, subject in enumerate(test_dataset):
        dataloader = DataLoader(subject, 
                                batch_size=params['train_batch_size'], 
                                shuffle=True, 
                                collate_fn=collate_function_v2, 
                                num_workers=params['train_num_workers'], 
                                pin_memory=False, 
                                drop_last=False)

        metrics = validate(model, dataloader, criterion, device="cuda")

        print(f"""[Test] Epoch {epoch + 1} - Subj {idx_suj + 1} - Loss: {metrics['loss']} 
                        - Acc Graph: {metrics['graph_acc']} - Acc Text: {metrics['text_acc']} 
                        - F1 Graph: {metrics['graph_f1']} - F1 Text: {metrics['text_f1']} 
                        - AUC Graph: {metrics['graph_auc']} - AUC Text: {metrics['text_auc']}""")

        if params["log"]:
            run['test/subject/graph_accuracy'].log(metrics['graph_acc'])
            run['test/subject/text_accuracy'].log(metrics['text_acc'])
            run['test/subject/graph_f1'].log(metrics['graph_f1'])
            run['test/subject/text_f1'].log(metrics['text_f1'])
            run['test/subject/graph_auc'].log(metrics['graph_auc'])
            run['test/subject/text_auc'].log(metrics['text_auc'])
            run['test/subject/loss'].log(metrics['loss'])

        # Limpiar memoria de la GPU
        torch.cuda.empty_cache()
        del dataloader
    run.stop()
    
    

    
    
        
        

    
    
