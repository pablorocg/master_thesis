# Load textgraph model with weights


import torch
import numpy as np
import pandas as pd
from torch_dataset import FiberGraphDataset, collate_function_v2
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from text_graph_model import Multimodal_Text_Graph_Model
from config import CFG
from tqdm import tqdm
# import tensorboard
from torch.utils.tensorboard import SummaryWriter

logger = SummaryWriter("/app/weights/tensorboard")

# Load model
model = model = Multimodal_Text_Graph_Model()

# Load textgraph model with weights
model.load_state_dict(torch.load("/app/weights/model_0_suj_20_loss_0.1034173529035748.pt", map_location=CFG.device))
model.to(CFG.device)
model.eval()
print(f'modelo cargado correctamente')


ds = FiberGraphDataset(root='/app/dataset/tractoinferno_graphs/testset')



print('Sujeto cargado correctamente ')
dl = DataLoader(ds[0], batch_size=1024, shuffle=False, collate_fn=collate_function_v2)

for i, (graph_data, text_data, graph_label, text_label, type_of_pair) in enumerate(dl):
                
    model.eval()
    
    # send data to device
    graph_data = graph_data.to(model.device)
    text_data = {k: v.to(model.device) for k, v in text_data.items()}
    graph_label = graph_label.to(model.device)
    text_label = text_label.to(model.device)
    type_of_pair = type_of_pair.to(model.device)

    # obtain the output
    with torch.no_grad():
        g_proj, t_proj, g_pred_lab, t_pred_lab = model(graph_data, text_data)

    # convert to numpy and save the projections to visualize them later using tensorboard
    g_proj = g_proj.detach().cpu().numpy()
    t_proj = t_proj.detach().cpu().numpy()

    

    


    # log the embeddings and labels to tensorboard
    logger.add_embedding(g_proj, metadata=graph_label, tag="graph_embeddings")
    logger.add_embedding(t_proj, metadata=text_label, tag="text_embeddings")
    
    # si la i vale 1 romper el bucle
    if i==1:
        print('Fin de la inferencia')
        break
    

    print(f'{i}/{len(dl)}')
    # # log the confusion matrix to tensorboard
    # logger.add_figure("Confusion Matrix", model.plot_confusion_matrix(graph_label, g_pred_lab), global_step=i)

# close the logger
logger.close()






