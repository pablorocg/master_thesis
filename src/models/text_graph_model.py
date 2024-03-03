from encoders import TextEncoder, ProjectionHead, ClassifierHead, GCN_Encoder
from torch_geometric.nn import global_mean_pool, GAE, VGAE

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch_geometric
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
                 
                 device = CFG.device):
        
        super(Multimodal_Text_Graph_Model, self).__init__()
        
        self.graph_encoder = GCN_Encoder(graph_channels, projection_dim) #(batch_size, text_embedding)
        self.graph_embedding_classifier = ClassifierHead(projection_dim, CFG.n_classes)


        self.text_encoder = TextEncoder(text_encoder_model) #(batch_size, text_embedding)
        self.text_projection_head = ProjectionHead(text_embedding, projection_dim, num_projection_layers=4) #(batch_size, projection_dim)
        self.text_embedding_classifier = ClassifierHead(projection_dim, CFG.n_classes)
        
        self.device = device
        self.to(device)
    
    def forward(self, graph_batch, text_batch):
        
        # graph_batch = graph_batch.to(self.device)
        # text_batch = {k: v.to(self.device) for k, v in text_batch.items()}
        
        graph_projections = self.graph_encoder(graph_batch) # (batch_size, projection_dim)
   
        graph_predicted_labels = self.graph_embedding_classifier(graph_projections) # (batch_size, n_classes)
        
        text_projections = self.text_encoder(text_batch) # (batch_size, text_embedding)
        text_projections = self.text_projection_head(text_projections) # (batch_size, projection_dim)
        text_predicted_labels = self.text_embedding_classifier(text_projections) # (batch_size, n_classes)

        return graph_projections, text_projections, graph_predicted_labels, text_predicted_labels
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class CategoricalContrastiveLoss(nn.Module):
    def __init__(self, theta):
        super(CategoricalContrastiveLoss, self).__init__()
        self.theta = theta
        self.margin = 1  # margen
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(self, graph_emb, text_emb, graph_label, text_label, graph_pred_label, text_pred_label, y):
        # Calcula la pérdida de disimilitud como la distancia euclídea
        dw = torch.norm(graph_emb - text_emb, p=2, dim=1)

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
    dataset = FiberGraphDataset(root=r'C:\Users\pablo\GitHub\tfm_prg\tractoinferno_graphs\testset')# '/app/dataset/tractoinferno_graphs/testset'
    
    # INSTANCIAR EL MODELO
    model = Multimodal_Text_Graph_Model()

    # INSTANCIAR EL OPTIMIZADOR Y EL SCHEDULER
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-3)
    # scheduler = ReduceLROnPlateau(optimizer, patience=1500, factor=0.95, verbose=True)

    # INSTANCIAR LA FUNCIÓN DE PÉRDIDA
    criterion = CategoricalContrastiveLoss(theta=0.35)
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
    for epoch in range(params["epochs"]):
        for idx_suj, subject in enumerate(dataset):
            dataloader = DataLoader(subject, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_function_v2, num_workers=params['num_workers'])
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
                
                # Actualizar la barra de progreso con el batch loss
                print(f'\rBatch {i+1} - Loss: {loss.item()} - G. Acc: {batch_graph_accuracy.item()} - T. Acc: {batch_text_accuracy.item()} - G F1: {batch_graph_f1.item()} - T F1: {batch_text_f1.item()} - G AUCROC: {batch_graph_aucroc.item()} - T AUCROC: {batch_text_aucroc.item()}', end="")

                # backward pass
                loss.backward()
                
                # actualizar los parámetros
                optimizer.step()
                # scheduler.step(loss)

                # guardar el loss 

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
    
    

    
    
        
        

    
    
