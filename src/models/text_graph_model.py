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
    

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        return self
    
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
            # Si se utiliza la dist euclidea normalizar los embeddings
            graph_emb = F.normalize(graph_emb, p=2, dim=1)
            text_emb = F.normalize(text_emb, p=2, dim=1)
            dw = torch.norm(graph_emb - text_emb, p=2, dim=1)
        elif self.dw == 'cosine':
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
        # tb_logger = SummaryWriter(f"runs/{name}")

        run["parameters"] = params# Log the parameters to Neptune
        run["config"] = CFG# Log the config to Neptune


    # Crear el dataset
    dataset = FiberGraphDataset(root='/app/dataset/tractoinferno_graphs/testset')#r'C:\Users\pablo\GitHub\tfm_prg\tractoinferno_graphs\testset' 
    
    # INSTANCIAR EL MODELO
    model = Multimodal_Text_Graph_Model()

    # INSTANCIAR EL OPTIMIZADOR Y EL SCHEDULER
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=1e-3)
    # scheduler = ReduceLROnPlateau(optimizer, patience=1500, factor=0.95, verbose=True)

    # INSTANCIAR LA FUNCIÓN DE PÉRDIDA
    criterion = CategoricalContrastiveLoss(theta=params['theta'], margin=params['margin'], dw=params['distance'], weights=params['weighted_loss'])
    criterion.to(model.device)

    # DEFINIR LAS MÉTRICAS
    auroc_graphs = torchmetrics.classification.MulticlassAUROC(num_classes=CFG.n_classes, average="macro").to(model.device)
    auroc_texts = torchmetrics.classification.MulticlassAUROC(num_classes=CFG.n_classes, average="macro").to(model.device)

    accuracy_graphs = torchmetrics.classification.MulticlassAccuracy(num_classes=CFG.n_classes, average="macro").to(model.device)
    accuracy_texts = torchmetrics.classification.MulticlassAccuracy(num_classes=CFG.n_classes, average="macro").to(model.device)

    f1_graphs = torchmetrics.classification.MulticlassF1Score(num_classes=CFG.n_classes, average="macro").to(model.device)
    f1_texts = torchmetrics.classification.MulticlassF1Score(num_classes=CFG.n_classes, average="macro").to(model.device)


    # best loss 
    best_loss = np.inf


    # BUCLE DE ENTRENAMIENTO
    for epoch in range(params['train_epochs']):
        for idx_suj, subject in enumerate(dataset):
            # Average loss for the epoch
            avg_loss = 0

            dataloader = DataLoader(subject, batch_size=params['train_batch_size'], shuffle=True, collate_fn=collate_function_v2, num_workers=params['train_num_workers'], pin_memory=True, drop_last=True)
            for i, (graph_data, text_data, graph_label, text_label, type_of_pair) in enumerate(dataloader):
                
                model.train()
                
                # send data to device
                graph_data = graph_data.to(model.device)
                text_data = {k: v.to(model.device) for k, v in text_data.items()}
                graph_label = graph_label.to(model.device)
                text_label = text_label.to(model.device)
                type_of_pair = type_of_pair.to(model.device)


                # reset gradients
                optimizer.zero_grad()

                # forward pass
                g_proj, t_proj, g_pred_lab, t_pred_lab = model(graph_data, text_data)
                loss = criterion(g_proj, t_proj, graph_label, text_label, g_pred_lab, t_pred_lab, type_of_pair)

                # Calcular el accuracy y el f1
                g_pred = torch.argmax(g_pred_lab, dim=1)
                # Calculate metrics for graph data
                batch_graph_accuracy = accuracy_graphs(g_pred, graph_label)
                batch_graph_f1 = f1_graphs(g_pred, graph_label)
                batch_graph_aucroc = auroc_graphs(g_pred_lab, graph_label)
                

                t_pred = torch.argmax(t_pred_lab, dim=1)
                # Calculate metrics for text data
                batch_text_accuracy = accuracy_texts(t_pred, text_label)
                batch_text_f1 = f1_texts(t_pred, text_label)
                batch_text_aucroc = auroc_texts(t_pred_lab, text_label)

                if params["log"]:
                    run['train/batch/graph_accuracy'].log(batch_graph_accuracy.item())
                    run['train/batch/graph_f1'].log(batch_graph_f1.item())
                    run['train/batch/graph_aucroc'].log(batch_graph_aucroc.item())

                    run['train/batch/text_accuracy'].log(batch_text_accuracy.item())
                    run['train/batch/text_f1'].log(batch_text_f1.item())
                    run['train/batch/text_aucroc'].log(batch_text_aucroc.item())

                    run['train/batch/loss'].log(loss.item())
                
                # Actualizar la barra de progreso con el batch loss con 4 decimales
                print(f'\rBatch {i+1} - Loss: {loss.item():.4f} - Graph Accuracy: {batch_graph_accuracy.item():.4f} - Text Accuracy: {batch_text_accuracy.item():.4f}', end="")

                # backward pass
                loss.backward()
                
                # actualizar los parámetros
                optimizer.step()
                # scheduler.step(loss)

                # guardar el loss
                avg_loss += loss.item()

            # Actualizar el loss del sujeto
            avg_loss /= len(dataloader)
            if params['log']:
                run['train/subject/loss'].log(avg_loss)
            print(f"\nSubject {idx_suj} - Loss: {avg_loss}")

            # Computar las métricas para el sujeto
            graph_aucroc = auroc_graphs.compute()
            text_aucroc = auroc_texts.compute()

            graph_accuracy = accuracy_graphs.compute()
            text_accuracy = accuracy_texts.compute()

            graph_f1 = f1_graphs.compute()
            text_f1 = f1_texts.compute()

            if params["log"]:
                run['train/subject/graph_accuracy'].log(graph_accuracy.item())
                run['train/subject/graph_f1'].log(graph_f1.item())
                run['train/subject/graph_aucroc'].log(graph_aucroc.item())

                run['train/subject/text_accuracy'].log(text_accuracy.item())
                run['train/subject/text_f1'].log(text_f1.item())
                run['train/subject/text_aucroc'].log(text_aucroc.item())

            # resetear las métricas
            auroc_graphs.reset()
            auroc_texts.reset()
            accuracy_graphs.reset()
            accuracy_texts.reset()
            f1_graphs.reset()
            f1_texts.reset()



            if avg_loss < best_loss:
                best_loss = avg_loss
                # Guardar los pesos del modelo si el loss es mejor
                torch.save(model.state_dict(), f"/app/weights/model_{epoch}_suj_{idx_suj}_loss_{best_loss}.pt")



            # Cerrar el run de Neptune
            

            # Cuando termina el sujeto gurdar los pesos del modelo 
            # if loss.item() < best_loss:
            #     best_loss = loss.item()
            #     torch.save(model.state_dict(), f"model_{epoch}_suj{idx_suj}_loss_{best_loss}.pt")
                



                
            
            
            

        
        
        
        # # 5. EVALUAR EL MODELO
        # model.eval()
        # # seleccionar un sujeto aleatorio para evaluar
        # suj_eval = dataset[28]
        # total_loss = 0
        # dataloader = DataLoader(suj_eval, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_function, num_workers=params['num_workers'])
        # progress_bar_validation = tqdm(dataloader, desc="Validation")
        # name = f"model_{time.time()}"
        # tb_logger = SummaryWriter(f"runs/{name}")
        # all_embeddings, all_labels = [], []
        # with torch.no_grad():
        #     for i, (graph_data, text_data) in enumerate(dataloader):
                
        #         graph_data = graph_data.to(model.device)
        #         text_data = {k: v.to(model.device) for k, v in text_data.items()}

        #         # save first 100 batches embeddings
        #         if i < 100:
        #             graph_embeddings = model.graph_encoder.encode(graph_data.x, graph_data.edge_index)
        #             graph_embeddings = global_mean_pool(graph_embeddings, graph_data.batch)
        #             all_embeddings.append(graph_embeddings.detach().cpu().numpy())
        #             all_labels.append(graph_data.y.detach().cpu().numpy())
        #             if i == 99:
        #                 all_embeddings = np.concatenate(all_embeddings, axis=0)
        #                 all_labels = np.concatenate(all_labels, axis=0)
        #                 tb_logger.add_embedding(all_embeddings, metadata=all_labels, tag="embeddings")
        #                 tb_logger.close()
                    

        #         # forward pass
        #         loss = model(graph_data, text_data)
        #         total_loss += loss.item()

        #         if params["log"]:
        #             run['eval/batch/loss'].log(loss.item())

        #     # Actualizar el loss del sujeto
        #     mean_loss = total_loss / len(dataloader)
        #     if params["log"]:
        #         run['eval/subject/loss'].log(loss)

                


        # # 6. GUARDAR EL MODELO SI ES MEJOR QUE EL ANTERIOR
        
        # model.train()
    # Cerrar el run de Neptune
    run.stop()
    
    

    
    
        
        

    
    
