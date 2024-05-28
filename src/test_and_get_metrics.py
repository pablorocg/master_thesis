import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import (MulticlassAccuracy, 
                                         MulticlassF1Score,
                                         MulticlassAUROC,
                                         MulticlassConfusionMatrix)
import torch.optim as optim
from tqdm import tqdm

from dataset_handlers import (HCPHandler, 
                            TractoinfernoHandler,
                            FiberCupHandler)

from streamline_datasets import (MaxMinNormalization,
                                TestDataset, collate_test_ds,
                                StreamlineTripletDataset, collate_triplet_ds,
                                fill_tracts_ds)

from encoders import (SiameseGraphNetwork, GCNEncoder, GATEncoder,
                                ProjectionHead, ClassifierHead)
from hgp_sl_model import HGPSLEncoder

from loss_functions import (MultiTaskTripletLoss, TripletLoss)

import neptune
from torch.utils.tensorboard import SummaryWriter
# Comando para lanzar tensorboard en el navegador local a traves del puerto 8888 reenviado por ssh:
# tensorboard --logdir=runs/embedding_visualization --host 0.0.0.0 --port 8888
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import seaborn as sns


from dipy.tracking.utils import density_map
from dipy.io.streamline import load_trk


# Habilitar TensorFloat32 para una mejor performance en operaciones de multiplicación de matrices
torch.set_float32_matmul_precision('high')



log = False

if log:
    run = neptune.init_run(
        project="pablorocamora/tfm-tractography",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ODA0YzA2NS04MjczLTQyNzItOGE5Mi05ZmI5YjZkMmY3MDcifQ==",
    )

    
    




class CFG:
    def __init__(self):
        self.seed = 42
        self.max_epochs = 4
        self.batch_size = 4096
        self.learning_rate = 0.005
        self.max_batches_per_subject = 200
        self.classification_weight = 0.6
        self.margin = 1.25
        self.encoder = "GCN" # Las opciones son "GAT" o "GCN" o "HGPSL"
        self.optimizer = "AdamW"

        self.dataset = "HCP_105" # "Tractoinferno o "FiberCup" o "HCP_105"

        if self.dataset == "HCP_105":
            self.ds_path = "/app/dataset/HCP_105"
            self.pretrained_model_path = "/app/pretrained_models/encoder_HCP_105.pt"
            self.n_classes = 72 # 72 tractos o 71 tractos sin CC

        elif self.dataset == "Tractoinferno":
            self.ds_path = "/app/dataset/Tractoinferno/tractoinferno_preprocessed_mni"
            self.pretrained_model_path = "/app/pretrained_models/encoder_tractoinferno.pt"
            self.n_classes = 32
        
        elif self.dataset == "FiberCup":
            self.ds_path = "/app/dataset/Fibercup"
            self.pretrained_model_path = "/app/pretrained_models/encoder_fibercup.pt"
            self.n_classes = 7

        self.embedding_projection_dim = 128
        
    
cfg = CFG()

if log:
    # Inicializar el SummaryWriter para TensorBoard
    writer = SummaryWriter(log_dir=f"runs/{cfg.dataset}_embedding_visualization", filename_suffix=f"{time.time()}")
    
    run["config/dataset"] = cfg.dataset
    run["config/seed"] = cfg.seed
    run["config/max_epochs"] = cfg.max_epochs
    run["config/batch_size"] = cfg.batch_size
    run["config/learning_rate"] = cfg.learning_rate
    run["config/max_batches_per_subject"] = cfg.max_batches_per_subject
    run["config/n_classes"] = cfg.n_classes
    run["config/embedding_projection_dim"] = cfg.embedding_projection_dim
    run["config/classification_weight"] = cfg.classification_weight
    run["config/margin"] = cfg.margin
    run["config/encoder"] = cfg.encoder
    run["config/optimizer"] = cfg.optimizer


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

            
@torch.jit.script
def get_weighted_dice_coefficient(density_map_gt: torch.Tensor, 
                                  density_map_pred: torch.Tensor,
                                  gt_len:int, 
                                  pred_len:int) -> torch.Tensor:
    """
    Calcula el weighted dice coefficient entre dos mapas de densidad de fibras.
    """
    eps = 1e-6

    # Convertir a tensor plano con view(-1)
    density_map_gt = density_map_gt.view(-1).float()
    density_map_pred = density_map_pred.view(-1).float()

    # Dividir n de cada mapa de densidad por el numero de fibras en el tracto
    weighted_density_map_gt = density_map_gt / gt_len
    weighted_density_map_pred = density_map_pred / pred_len

    # Obtener la interseccion entre conjuntos
    intersection = 2 * torch.sum(torch.minimum(weighted_density_map_gt, weighted_density_map_pred))
    union = torch.sum(weighted_density_map_gt) + torch.sum(weighted_density_map_pred)

    wdice = intersection / union

    if wdice.isnan():
        wdice = 0.0

    return wdice


@torch.jit.script
def get_dice_coefficient(density_map_gt: torch.Tensor,
                            density_map_pred: torch.Tensor) -> torch.Tensor:
    """
    Calcula el dice coefficient entre dos mapas de densidad de fibras.
    """
    eps = 1e-6
    density_map_gt = density_map_gt > 0
    density_map_pred = density_map_pred > 0

    # Convertir a tensor plano con view(-1)
    density_map_gt = density_map_gt.view(-1).float()
    density_map_pred = density_map_pred.view(-1).float()

    # Obtener la interseccion entre conjuntos
    intersection = 2 * torch.sum(torch.minimum(density_map_gt, density_map_pred))
    union = torch.sum(density_map_gt) + torch.sum(density_map_pred)

    dice = (intersection) / union
    
    if dice.isnan():
        dice = 0.0

    return dice



# Establecer la semilla
seed_everything(cfg.seed)


# Cargar las rutas de los sujetos de entrenamiento, validación y test
if cfg.dataset == "HCP_105":
    handler = HCPHandler(path = cfg.ds_path, scope = "testset")
    test_data = handler.get_data()
    

elif cfg.dataset == "Tractoinferno":
    handler = TractoinfernoHandler(path = cfg.ds_path, scope = "testset")
    test_data = handler.get_data()

elif cfg.dataset == "FiberCup":
    handler = FiberCupHandler(path = cfg.ds_path, scope = "testset")
    test_data = handler.get_data()




# Crear el modelo, la función de pérdida y el optimizador
if cfg.encoder == "GAT":# Graph Attention Network
    encoder = GATEncoder(in_channels = 3, 
                         hidden_channels = 16, 
                         out_channels = 256)

elif cfg.encoder == "GCN":# Graph Convolutional Network
    encoder = GCNEncoder(in_channels = 3, 
                         hidden_dim = 128, 
                         out_channels = 128, 
                         dropout = 0.15, 
                         n_hidden_blocks = 2)
    
elif cfg.encoder == "HGPSL":# Hierarchical Graph Pooling with Structure Learning
    encoder = HGPSLEncoder(num_features = 3, 
                           nhid = 128, 
                           emb_dim = cfg.embedding_projection_dim, 
                           pooling_ratio = 0.5, 
                           dropout_ratio = 0.0, 
                           sample_neighbor = True, 
                           sparse_attention = True, 
                           structure_learning = True, 
                           lamb = 1.0)





model = SiameseGraphNetwork(
    encoder = encoder,
    projection_head = ProjectionHead(embedding_dim = 128, 
                                     projection_dim = cfg.embedding_projection_dim),
    classifier = ClassifierHead(projection_dim = cfg.embedding_projection_dim, 
                                n_classes = cfg.n_classes)
).cuda()

# Compile the model into an optimized version:
model = torch.compile(model, dynamic=True)



# Cargar el modelo preentrenado
# model.load_state_dict(torch.load(cfg.pretrained_model_path))

model.eval()

# Crear las métricas con torchmetrics
subj_accuracy = MulticlassAccuracy(num_classes = cfg.n_classes, average='macro').cuda()
ds_accuracy = MulticlassAccuracy(num_classes = cfg.n_classes, average='macro').cuda()

subj_f1 = MulticlassF1Score(num_classes = cfg.n_classes, average='macro').cuda()
ds_f1 = MulticlassF1Score(num_classes = cfg.n_classes, average='macro').cuda()

subj_auroc = MulticlassAUROC(num_classes = cfg.n_classes).cuda()
ds_auroc = MulticlassAUROC(num_classes = cfg.n_classes).cuda()

subj_confusion_matrix = MulticlassConfusionMatrix(num_classes = cfg.n_classes).cuda()
ds_confusion_matrix = MulticlassConfusionMatrix(num_classes = cfg.n_classes).cuda()


model.eval()

for idx_val, subject in enumerate(test_data):

    for file in subject['tracts']:
        print(f'Evaluando el archivo: {file}')
        correct_streamlines_idx = []

        val_ds = TestDataset(file, 
                            handler,
                            transform = MaxMinNormalization(dataset = cfg.dataset))

        val_dl = DataLoader(val_ds, batch_size = cfg.batch_size, 
                            shuffle = False, num_workers = 1,
                            collate_fn = collate_test_ds)

        classified_idxs = defaultdict(list)

        for i, graph in enumerate(val_dl):
            
            # Enviar a la gpu
            graph = graph.cuda()
            target = graph.y

            # Imprimir los valores unicos de las etiquetas
            # print(target.unique())
            
            with torch.no_grad():
                # Forward pass
                _, pred = model(graph)
                
            # Calcular métricas de test
            subj_accuracy.update(pred, target)
            subj_f1.update(pred, target)
            subj_auroc.update(pred, target)
            subj_confusion_matrix.update(pred, target)

            ds_accuracy.update(pred, target)
            ds_f1.update(pred, target)
            ds_auroc.update(pred, target)
            ds_confusion_matrix.update(pred, target)
            
            
            pred = pred.argmax(dim = -1)
            
            # Obtener los indices de posicion de los grafos clasificados correctamente
            # print(f'Predicciones: {pred.shape}')
            # print(f'Etiquetas: {target.shape}')
            
            correct_idxs = torch.where(pred == target)[0]
            correct_streamlines_idx.extend(correct_idxs.tolist())


        # Al completar el dataloader de un sujeto, guardar las métricas de subject
        # Al finalizar el dataloader de un sujeto tendremos los indices de los grafos clasificados correctamente en classified_idxs
        # A continuacion se cargara el fichero trk y se obtendra el tractograma, se seleccionaran las fibras clasificadas correctamente
        # y se guardaran en un nuevo tractograma que se guardara en un nuevo fichero trk
        print(f'Indices de los grafos clasificados correctamente: {correct_streamlines_idx}')
        
        # Cargar el archivo trk y obtener el tractograma
        tractogram = load_trk(str(file), reference='same', bbox_valid_check=False)
        # Calculo del weighted dice coefficient
        density_map_gt = density_map(streamlines = tractogram.streamlines, 
                                    affine = tractogram.affine, 
                                    vol_dims = tractogram.dimensions)
        

        # Calcular el mapa de densidad de las fibras clasificadas correctamente
        correct_streamlines = [tractogram.streamlines[i] for i in correct_streamlines_idx]

        density_map_pred = density_map(streamlines = correct_streamlines, 
                                    affine = tractogram.affine, 
                                    vol_dims = tractogram.dimensions)
        
        # Convertir a tensores
        density_map_gt = torch.tensor(density_map_gt)
        density_map_pred = torch.tensor(density_map_pred)
        
        # Calcular el weighted dice coefficient
        wdice = get_weighted_dice_coefficient(density_map_gt, density_map_pred,
                                            len(tractogram.streamlines), len(correct_streamlines))
        
        print(f'Weighted Dice Coefficient: {wdice}')
        # Calcular el dice coefficient
        dice = get_dice_coefficient(density_map_gt, density_map_pred)
        print(f'Dice Coefficient: {dice}')
        

                                        

        


    

    # Dividir n de cada mapa de densidad por el numero de fibras en el tracto
    # weighted_density_map_gt = density_map_gt / len(streamlines1.streamlines)
    # density_map_gt = density_map_gt.flatten().astype(float)
    # density_map_pred = density_map_pred.flatten().astype(float)

    # print(f'Max density map gt: {np.max(density_map_gt)}')
    # print(f'Max density map pred: {np.max(density_map_pred)}')

    # # Dividir n de cada mapa de densidad por el numero de fibras en el tracto
    # weighted_density_map_gt = density_map_gt / len(streamlines1.streamlines)
    # weighted_density_map_pred = density_map_pred / len(streamlines1.streamlines)


    # # Obtener la interseccion entre conjuntos 
    # intersection = 2 * np.sum(np.minimum(weighted_density_map_gt, weighted_density_map_pred))
    # union = np.sum(weighted_density_map_gt) + np.sum(weighted_density_map_pred)

    # dice = intersection / union

    # print(f'The weighted Dice coefficient is: {dice}')
           



        



        

        







        

    #         # Guardar los embeddings y etiquetas
    #         if idx_val == 0:

    #             embedding = embedding.cpu().numpy()
    #             target = graph.y.cpu().numpy()

    #             # Guardar embeddings y etiquetas en el diccionario por clase
    #             for emb, label in zip(embedding, target):
    #                 if len(embeddings_list_by_class[label]) < max_embeddings_per_class:
    #                     embeddings_list_by_class[label].append(emb)
                

    #         if log:# Loggear las métricas de batch
    #             run["val/batch/acc"].log(subj_accuracy_val.compute().item())
    #             run["val/batch/f1"].log(subj_f1_val.compute().item())
    #             run["val/batch/auroc"].log(subj_auroc_val.compute().item())

    #         if i % 25 == 0:
    #             print(f"[VAL] Epoch {epoch+1}/{cfg.max_epochs} - Subj {idx_val} - Batch {i} - Acc.: {subj_accuracy_val.compute().item():.4f}, F1: {subj_f1_val.compute().item():.4f}, AUROC: {subj_auroc_val.compute().item():.4f}")

    #     # Para el idx_val = 1, guardar los embeddings de los grafos con sus etiquetas para visualizarlos en TensorBoard Projector
    #     if idx_val == 0:

    #         # Preparar los datos para TensorBoard
    #         all_embeddings = []
    #         all_labels = []

    #         for label, embeddings in embeddings_list_by_class.items():
    #             all_embeddings.extend(embeddings)
    #             # Obtener la etiqueta textual de la clase a través del handler
    #             label = handler.get_tract_from_label(label)
    #             print(label)
    #             all_labels.extend([label] * len(embeddings))

    #         # Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at /opt/conda/conda-bld/pytorch_1704987394225/work/torch/csrc/utils/tensor_new.cpp:275.)
    #         # Convertir listas a numpy.ndarray
    #         all_embeddings = np.array(all_embeddings)
    #         all_labels = np.array(all_labels)


    #         # Convertir a tensores
    #         # Convertir a tensores
    #         all_embeddings = torch.tensor(all_embeddings)
    #         all_labels = all_labels.tolist()  # TensorBoard necesita etiquetas como lista de strings


    #         # Guardar los embeddings y etiquetas en TensorBoard
    #         writer.add_embedding(
    #             all_embeddings, 
    #             metadata = all_labels, 
    #             global_step = epoch,
    #             tag = f"{cfg.dataset}_{cfg.encoder}_{cfg.embedding_projection_dim}_{time.time()}"
    #         ) 

    #     if log:# Loggear las métricas de subject
    #         run["val/subject/acc"].log(subj_accuracy_val.compute().item())
    #         run["val/subject/f1"].log(subj_f1_val.compute().item())
    #         run["val/subject/auroc"].log(subj_auroc_val.compute().item())

    #         cm = subj_confusion_matrix.compute()
    #         # Convertir la matriz de confusión a numpy
    #         cm = cm.cpu().numpy()

    #         # Visualiza la matriz de confusión y guárdala como imagen
    #         plt.figure(figsize=(35, 35))
    #         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    #         text_labels = [handler.get_tract_from_label(i) for i in range(cfg.n_classes)]
    #         plt.xticks(ticks = range(cfg.n_classes), labels = text_labels, rotation = 90)
    #         plt.yticks(ticks = range(cfg.n_classes), labels = text_labels, rotation = 0)
    #         plt.xlabel('Predicted Labels')
    #         plt.ylabel('True Labels')
    #         plt.title(f'Confusion Matrix Subj {idx_val}')
    #         plt.tight_layout()

    #         # Guarda la imagen
    #         img_path = f'/app/confusion_matrix_imgs/confusion_matrix_val_suj{idx_val}.png'
    #         plt.savefig(img_path)
    #         plt.close()

    #         # Sube la imagen a Neptune
    #         run["confusion_matrix_fig"].upload(img_path)
        
    #     subj_accuracy_val.reset()
    #     subj_f1_val.reset()
    #     subj_auroc_val.reset()
    #     subj_confusion_matrix.reset()

    # if idx_val == 6:
    #     break

    # # Cerrar el SummaryWriter
    # writer.close()

                   
