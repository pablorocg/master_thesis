import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import (MulticlassAccuracy, 
                                         MulticlassF1Score,
                                         MulticlassAUROC,
                                         MulticlassPrecision,
                                         MulticlassRecall)
from tqdm import tqdm
from dataset_handlers import (HCPHandler,
                              HCP_Without_CC_Handler, 
                              TractoinfernoHandler,
                              FiberCupHandler)

from streamline_datasets import (MaxMinNormalization,
                                TestDataset, collate_test_ds)

from encoders import (SiameseGraphNetwork, GCNEncoder, 
                      GATEncoder, ProjectionHead, ClassifierHead)

from custom_metrics import get_dice_metrics

from torch.utils.tensorboard import SummaryWriter
# Comando para lanzar tensorboard en el navegador local a traves del puerto 8888 reenviado por ssh:
# tensorboard --logdir=runs/embedding_visualization --host 0.0.0.0 --port 8888


import pandas as pd
import warnings
warnings.filterwarnings('ignore')


from gcn_encoder_model_v2 import SiameseGraphNetworkGCN_v2

# Habilitar TensorFloat32 para una mejor performance en operaciones de multiplicación de matrices
torch.set_float32_matmul_precision('high')




class CFG:
    def __init__(self):
        self.seed = 42
      
        self.batch_size = 4096
        self.encoder = "GCNEncoder_v2" # Las opciones son "GAT" o "GCN" o "HGPSL"
        self.dataset = "HCP_105_without_CC"#"Tractoinferno" # "Tractoinferno o "FiberCup" o "HCP_105"

        if self.dataset == "HCP_105":
            self.ds_path = "/app/dataset/HCP_105"
            self.pretrained_model_path = "/app/pretrained_models/encoder_HCP_105.pt"
            self.n_classes = 72 # 72 tractos o 71 tractos sin CC

        elif self.dataset == "HCP_105_without_CC":
            self.ds_path = "/app/dataset/HCP_105"
            self.pretrained_model_path = "/app/trained_models/checkpoint_HCP_105_without_CC_GCNEncoder_v2_512_1.pth"
            self.n_classes = 71

        elif self.dataset == "Tractoinferno":
            self.ds_path = "/app/dataset/Tractoinferno/tractoinferno_preprocessed_mni"
            self.pretrained_model_path = "/app/trained_models/checkpoint_Tractoinferno_GCN_512_0.pth"
            self.n_classes = 32
        
        elif self.dataset == "FiberCup":
            self.ds_path = "/app/dataset/Fibercup"
            self.pretrained_model_path = "/app/pretrained_models/encoder_fibercup.pt"
            self.n_classes = 7

        self.embedding_projection_dim = 512
        
    
cfg = CFG()




def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Establecer la semilla
seed_everything(cfg.seed)


# Cargar las rutas de los sujetos de entrenamiento, validación y test
if cfg.dataset == "HCP_105":
    handler = HCPHandler(path = cfg.ds_path, scope = "testset")
    test_data = handler.get_data()

elif cfg.dataset == "HCP_105_without_CC":
    handler = HCP_Without_CC_Handler(path = cfg.ds_path, scope = "testset")
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
    

elif cfg.encoder == "GCNEncoder_v2":
    model = SiameseGraphNetworkGCN_v2(n_classes = cfg.n_classes).cuda()
    # Compile the model into an optimized version:
    model = torch.compile(model, dynamic=True)





# # Crear el modelo
# model = SiameseGraphNetwork(
#     encoder = encoder,
#     projection_head = ProjectionHead(embedding_dim = 128, 
#                                      projection_dim = cfg.embedding_projection_dim),
#     classifier = ClassifierHead(projection_dim = cfg.embedding_projection_dim, 
#                                 n_classes = cfg.n_classes)
# ).cuda()

# Cargar el modelo preentrenado
model.load_state_dict(torch.load(cfg.pretrained_model_path)["model_state_dict"])










# Crear un dataframe para almacenar las métricas de los sujetos
df_subjects = pd.DataFrame(columns = ['subject_id', 'tract', 'accuracy', 'precision', 'recall', 'F1', 'AUCROC', 'DICE', 'wDICE'])

model.eval()
# Iterar sobre los sujetos
for idx_val, subject in tqdm(enumerate(test_data), total = len(test_data), desc = "Subjects"):
    
    # Iterar sobre los tractos
    for file in tqdm(subject['tracts'], total = len(subject['tracts']), desc = "Tracts", leave = False):

        # Crear las métricas con torchmetrics para evaluar cada sujeto
        tract_accuracy = MulticlassAccuracy(num_classes = cfg.n_classes, average='macro').cuda()
        tract_f1 = MulticlassF1Score(num_classes = cfg.n_classes, average='macro').cuda()
        tract_precision = MulticlassPrecision(num_classes = cfg.n_classes, average='macro').cuda()
        tract_recall = MulticlassRecall(num_classes = cfg.n_classes, average='macro').cuda()
        tract_auroc = MulticlassAUROC(num_classes = cfg.n_classes).cuda()

        correct_streamlines_idx = []# Almacenar los indices de los grafos clasificados correctamente en el tracto actual
       

        print(f"Subject: {subject['subject']}, Tract: {file.stem}")
        ds = TestDataset(file, 
                            handler,
                            transform = MaxMinNormalization(dataset = cfg.dataset))

        dl = DataLoader(ds, batch_size = cfg.batch_size, 
                            shuffle = False, num_workers = 2,
                            collate_fn = collate_test_ds)
        if len(dl) != 0:
   
            for i, graph in enumerate(dl):

                graph = graph.cuda()
                target = graph.y
                
                with torch.no_grad():
                    _, pred = model(graph)# Forward pass
                    
                # Calcular métricas de test
                tract_accuracy.update(pred, target)
                tract_f1.update(pred, target)
                tract_auroc.update(pred, target)
                tract_precision.update(pred, target)
                tract_recall.update(pred, target)

            
                # Obtener los indices de los grafos clasificados correctamente
                pred = torch.argmax(pred, dim = -1)
                correct_idxs = torch.where(pred == target)[0]
                correct_streamlines_idx.extend(correct_idxs.tolist())

            

            # Calcular las métricas de DICE y wDICE
            wdice, dice = get_dice_metrics(file, correct_streamlines_idx)

            # Crear una fila con las métricas del sujeto actual
            # ['subject_id', 'tract', 'accuracy', 'precision', 'recall', 'F1', 'AUCROC', 'DICE', 'wDICE']
            row = [
                subject['subject'], # subject_id
                file.stem, # tract
                tract_accuracy.compute().item(), # accuracy
                tract_precision.compute().item(), # precision 
                tract_recall.compute().item(), # recall
                tract_f1.compute().item(), # F1
                tract_auroc.compute().item(), # AUCROC
                dice.item(), # DICE
                wdice.item() # wDICE
            ]

            print(row)

            # Agregar la fila al dataframe df.loc[len(df)] = row
            df_subjects.loc[len(df_subjects)] = row


            # Resetear las métricas de subject
            tract_accuracy.reset()
            tract_f1.reset()
            tract_auroc.reset()
            tract_precision.reset()
            tract_recall.reset()

# Guardar el dataframe en un archivo csv
df_subjects.to_csv(f"/app/resultados/results_{cfg.dataset}_{cfg.encoder}_v2.csv", index = False)


    
        

                                        

        


    

                   
