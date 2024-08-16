import numpy as np
import random
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from dataset_handlers import (HCPHandler,
                              HCP_Without_CC_Handler, 
                              TractoinfernoHandler,
                              FiberCupHandler)


from utils import seed_everything

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score)

import pathlib2 as pathlib
from torch_geometric.transforms import Compose, ToUndirected, GCNNorm
from torch_geometric.nn.models import GCN
from nibabel.streamlines import ArraySequence
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from custom_transforms import MaxMinNormalization

# Habilitar TensorFloat32 para una mejor performance en operaciones de multiplicación de matrices
torch.set_float32_matmul_precision('high')


from dipy.io.streamline import load_trk
from custom_metrics import get_weighted_dice_coefficient, get_dice_coefficient, density_map
from encoders import GraphFiberNet

from single_fiber_dataset import StreamlineSingleDataset, collate_single_ds
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

























def get_dice_metrics_v2(trk_files, predictions, handler) -> tuple[float, float]:
    streamlines_by_tract = {}
    all_streamlines = ArraySequence()
    all_labels = []
    affine = None
    dimensions = None

    for tract in trk_files:
        tract_label = handler.get_label_from_tract(tract.stem)# entero
        tractography_data = load_trk(str(tract), 'same', bbox_valid_check=False)
        tractography_data.remove_invalid_streamlines()
        streamlines = tractography_data.streamlines
        affine = tractography_data.affine
        dimensions = tractography_data.dimensions
        streamlines_by_tract[tract_label] = streamlines
        all_streamlines.extend(streamlines_by_tract[tract_label])
        all_labels.extend([tract_label] * len(streamlines_by_tract[tract_label]))
        del tractography_data

    wDICE, DICE = [], []
    for tract_label, streamlines in streamlines_by_tract.items():
        predicted_as_c = [i for i in range(len(predictions)) if predictions[i] == tract_label]
        predicted_streamlines = ArraySequence([all_streamlines[i] for i in predicted_as_c])
        density_map_gt = density_map(streamlines, affine, dimensions)
        density_map_pred = density_map(predicted_streamlines, affine, dimensions)
        density_map_gt = torch.tensor(density_map_gt)
        density_map_pred = torch.tensor(density_map_pred)
        wdice = get_weighted_dice_coefficient(density_map_gt, density_map_pred)
        dice = get_dice_coefficient(density_map_gt, density_map_pred)
        wDICE.append(wdice.item())
        DICE.append(dice.item())
    return wDICE, DICE




class CFG:
    def __init__(self):
        self.seed = 42
        self.batch_size = 2048
        self.encoder = "GCNEncoder_supcon_finetuned"
        self.dataset = "HCP_105_without_CC"
        self.ds_path = "/app/dataset/HCP_105"
        self.n_classes = 71
        
        
        
    
cfg = CFG()
seed_everything(cfg.seed)
handler = HCP_Without_CC_Handler(path = cfg.ds_path, scope = "testset")
test_data = handler.get_data()


transforms = Compose([
    MaxMinNormalization("Tractoinferno"),
    ToUndirected(), 
    GCNNorm() 
])





classifier_model = GraphFiberNet(
    encoder = GCN(in_channels = 3, hidden_channels = 256, out_channels = 256, num_layers = 5),
    hidden_channels = 256,
    n_classes = cfg.n_classes,
    full_trainable = True
).cuda()

classifier_model.load_state_dict(
    torch.load('/app/trained_models/checkpoint_HCP_105_without_CC_baseline_classifier.pth')['model_state_dict']
)





df = pd.DataFrame(columns = [
    'subject_id', 'tract', 'n_streamlines', 'accuracy', 
    'precision', 'recall', 'F1', 'AUCROC', 'DICE', 'wDICE'
])

classifier_model.eval()
for idx_val, subject in tqdm(enumerate(test_data), total=len(test_data), desc="Subjects"):
    
    
    test_ds = StreamlineSingleDataset(
        datadict=subject,
        ds_handler=handler,
        transform=transforms,
        select_n_streamlines=None#200#None
    )

    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_single_ds
    )

    print(subject)
    probs = torch.tensor([])  # Almacenar las probabilidades de los grafos en el tracto actual
    targets = torch.tensor([])  # Almacenar las etiquetas de los grafos en el tracto actual

    progbar = tqdm(enumerate(test_dl), total=len(test_dl), desc="Testing", leave=False)
    for i, graph in progbar:
        graph = graph.cuda()
        target = graph.y

        with torch.no_grad():
            pred = classifier_model(graph)
        
        pred = F.softmax(pred, dim=-1)
        
        pred = pred.cpu().detach()
        target = target.cpu().detach()

        probs = torch.cat((probs, pred), dim=0)
        targets = torch.cat((targets, target), dim=0)
    
    class_preds = torch.argmax(probs, dim=-1).tolist()
    targets_list = targets.tolist()

    # Recuento de clases en targets list
    fibers_per_tract = [(int(i), targets_list.count(i)) for i in set(targets_list)]

    # Valores unicos en targets list
    print([handler.get_tract_from_label(l) for l in set(targets_list)])
    

    accuracy = []
    precision = []
    recall = []
    f1 = []
    roc = []

    for c in set(targets_list):
        # c es la clase que se está evaluando en este momento
        
        # Obtener las predicciones y los targets para la clase c, para ello, binari
        class_preds_bin = [1 if p == c else 0 for p in class_preds]
        targets_bin = [1 if t == c else 0 for t in targets_list]

        # Calcular las métricas
        accuracy.append(accuracy_score(targets_bin, class_preds_bin))
        precision.append(precision_score(targets_bin, class_preds_bin, zero_division=0))
        recall.append(recall_score(targets_bin, class_preds_bin, zero_division=0))
        f1.append(f1_score(targets_bin, class_preds_bin, zero_division=0))
        
        # Calcular ROC AUC para la clase c usando one-vs-rest
        probs_bin = probs[:, int(c)].numpy()  # Probabilidad predicha para la clase c
        if len(np.unique(targets_bin)) > 1:  # Verifica que haya tanto positivos como negativos
            roc_auc = roc_auc_score(targets_bin, probs_bin)
        roc.append(roc_auc)
    
    
    # Ahora se deben calcular las métricas de DICE y wDICE para cada tracto del sujeto
    wdice, dice = get_dice_metrics_v2(subject['tracts'], class_preds, handler)

    print(len(accuracy), len(precision), len(recall), len(f1), len(roc), len(wdice), len(dice))


    # Procesar los resultados por clase
    for i, (a, p, r, f, roc, wd, d) in enumerate(zip(accuracy, precision, recall, f1, roc, wdice, dice)):
        row = [
            subject['subject'],  # subject_id
            handler.get_tract_from_label(fibers_per_tract[i][0]),  # tract
            fibers_per_tract[i][1],  # number of streamlines
            a,  # accuracy
            p,  # precision
            r,  # recall
            f,  # F1
            roc,  # AUCROC
            d,  # DICE
            wd  # wDICE
        ]

        print(row)

        # Agregar la fila al dataframe
        df.loc[len(df)] = row


df.to_csv(f"/app/resultados/results_hcp_105_without_cc_baseline_classifier.csv", index=False)