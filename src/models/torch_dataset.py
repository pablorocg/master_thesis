# Importar librerías
import os
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import BaseTransform, Compose
from config import CFG
from transformers import AutoTokenizer
import random
from torch_geometric.data import Batch as GeoBatch


#=============================== Transformaciones preprocesado =================================
class MaxMinNormalization(BaseTransform):
        def __init__(self, max_values=None, min_values=None):
            """
            Initialize the normalization transform with optional max and min values.
            If not provided, they should be computed from the dataset.
            """
            self.max_values = max_values if max_values is not None else torch.tensor([76.03170776367188, 
                                                                                      77.9359130859375, 
                                                                                      88.72427368164062], 
                                                                                      dtype=torch.float)
            self.min_values = min_values if min_values is not None else torch.tensor([-73.90082550048828, 
                                                                                      -112.23554992675781, 
                                                                                      -79.38320922851562], 
                                                                                      dtype=torch.float)

        def __call__(self, data: Data) -> Data:
            """
            Apply min-max normalization to the node features.
            """
            data.x = (data.x - self.min_values) / (self.max_values - self.min_values)
            return data


#=============================== Clase para el dataset =================================
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
    
# =============================== Clase para el batch de grafos ===============================
def collate_function(batch):
    """Funcion para el DataLoader"""
    tokenizer = AutoTokenizer.from_pretrained(CFG.text_encoder_model, model_max_length=128)#"bert-base-uncased"

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
            "In-depth analysis of the {type} fiber's architecture in the {tract}",
            "The {type} fiber's pivotal presence in the {tract} on the {side} side",
            "A {type} fiber's critical role in the {tract} tract's network",
            "Understanding the {type} fiber within the {tract}'s framework",
            "The {tract} tract's {type} fiber: A closer examination",
            "Insights into the {type} fiber's influence in the {tract}",
            "The {type} fiber's unique characteristics in the {tract}",
            "Exploring the {type} fiber's presence in the {tract} on the {side}",
            "The {type} fiber's essential functions within the {tract}",
            "Tracing the path of the {type} fiber through the {tract}",
            "The {type} fiber's strategic importance to the {tract}",
            "Investigating the {type} fiber's role in the {tract} tract's ecosystem",
            "The {type} fiber's contribution towards {tract} integrity",
            "Evaluating the {type} fiber's impact on the {tract} tract",
            "The {type} fiber's connectivity within the {tract}",
            "The {type} fiber and its significance to the {tract}'s structure",
            "A {type} fiber's journey across the {tract} on the {side} side",
            "The dynamic role of the {type} fiber in the {tract}",
            "Showcasing the {type} fiber within the {tract}'s complex network",
            "The {type} fiber's integration into the {tract} tract",
            "Dissecting the {type} fiber's function in the {tract}",
            "The {type} fiber's coordination with the {tract} tract",
            "A {type} fiber's influence on the {tract}'s functionality",
            "The {type} fiber's adaptation in the {tract} environment",
            "Charting the {type} fiber's relevance to the {tract}'s health",
            "The {type} fiber's orchestration within the {tract} tract on the {side}",
            "The {type} fiber's synergy with the {tract}'s architecture",
            "Unpacking the {type} fiber's functionality in the {tract}",
            "The {type} fiber within the {tract}: A study of precision",
            "The nuanced role of the {type} fiber in the {tract} tract",
            "A {type} fiber's impact on {tract} tract dynamics on the {side}",
            "The {type} fiber's contribution to the {tract}'s resilience",
            "A {type} fiber within the {tract}: Bridging critical gaps",
            "The {type} fiber's alignment with the {tract}'s objectives",
            "The {type} fiber's facilitation of communication in the {tract}",
            "Dissecting the {type} fiber's efficiency within the {tract}",
            "The {type} fiber's strategic placement in the {tract} for {side} hemisphere optimization",
            "Exploring the {type} fiber's adaptability in the {tract}",
            "The {type} fiber's role in enhancing the {tract}'s performance",
            "A {type} fiber's precision in navigating the {tract}",
            "The {type} fiber's contribution to the {tract}'s complexity",
            "A {type} fiber's resilience within the {tract} structure",
            "The {type} fiber: A cornerstone of the {tract}'s anatomy",
            "The {type} fiber's influence on {tract} tract health and functionality",
            "A {type} fiber's strategic role in the {tract}'s network on the {side}",
            "The {type} fiber's capacity for innovation in the {tract}",
            "Mapping the {type} fiber's influence across the {tract}",
            "The {type} fiber's essential service to the {tract}'s harmony",
            "The {type} fiber's role in the {tract}'s structural integrity",
            "A {type} fiber's signature within the {tract}: Unveiling its essence",
            "Profiling the {type} fiber: Key to the {tract}'s efficacy",
            "The {type} fiber as a linchpin in the {tract}'s neural circuitry",
            "Elevating {tract} tract insights: The {type} fiber's pivotal role",
            "A {type} fiber's blueprint within the {tract}'s neural landscape",
            "The {type} fiber: Sculpting the {tract}'s neuroanatomy",
            "Crafting connectivity: The {type} fiber's role in the {tract}",
            "The {type} fiber's imprint on the {tract}'s neural pathways",
            "A {type} fiber's strategic influence on {tract} tract coherence",
            "The {type} fiber's legacy within the {tract}'s neural networks",
            "A {type} fiber's mastery in shaping the {tract}",
            "Decoding the {type} fiber's essence in the {tract}'s framework",
            "The {type} fiber's orchestration of neural signals in the {tract}",
            "A {type} fiber's harmonious integration into the {tract}",
            "The {type} fiber as a navigator in the {tract}'s neural seas",
            "Crafting the {tract}'s narrative: The {type} fiber's role",
            "The {type} fiber's legacy in the {tract}'s functional architecture",
            "A {type} fiber's resonance within the {tract}'s neural harmony",
            "The {type} fiber: A beacon in the {tract}'s neural odyssey",
            "Navigating the {tract} with precision: The role of the {type} fiber",
            "The {type} fiber's symphony within the {tract}'s neural orchestra",
            "A {type} fiber's voyage through the {tract}'s neural corridors",
            "The {type} fiber: Crafting the backbone of the {tract}",
            "A {type} fiber's influence on the {tract}'s neural tapestry",
            "The {type} fiber as the {tract}'s architect: Designing neural pathways",
            "Unraveling the {type} fiber's narrative within the {tract}'s neural saga",
            "The {type} fiber's orchestral role in the neural symphony of the {tract}",
            "Navigating the nuances of the {tract} with the {type} fiber as our guide",
            "The {type} fiber: Bridging neural divides within the {tract}",
            "A {type} fiber's journey through the neural landscapes of the {tract}",
            "The {type} fiber: A neural harbinger in the {tract}'s evolution",
            "Architects of the brain: The {type} fiber's role in constructing the {tract}",
            "The {type} fiber's silent symphony within the {tract}'s neural network",
            "A {type} fiber's resilient passage through the {tract}'s neural terrain",
            "The {type} fiber: Shaping the {tract}'s neural destiny",
            "Unveiling the {type} fiber's neural tapestry in the {tract}",
            "The {type} fiber's dance across the neural stages of the {tract}",
            "A {type} fiber's imprint on the {tract}'s neural blueprint",
            "The {type} fiber: A conduit of connectivity in the {tract}",
            "Exploring the {tract}'s neural depth through the {type} fiber",
            "A {type} fiber's narrative within the neural anthology of the {tract}",
            "The {type} fiber's role in weaving the neural fabric of the {tract}",
            "A {type} fiber's echo in the neural chambers of the {tract}",
            "The {type} fiber as a neural pioneer in the {tract}'s frontier",
            "A {type} fiber's influence on the {tract}'s neural harmony",
            "The {type} fiber: A neural sentinel in the {tract}'s landscape",
            "Charting the {type} fiber's course in the {tract}'s neural voyage",
            "The {type} fiber's contribution to the {tract}'s neural mosaic",
            "A {type} fiber's legacy within the {tract}'s neural archives",
            "The {type} fiber: An architect of the {tract}'s neural pathways",
            "A {type} fiber's role in the {tract}'s neural symposium"
        ]
    
    
    # Extraer los labels de todos los grafos en el lote
    labels = [graph.y.item() for graph in batch]  # Asumiendo que `y` es el tensor de labels
    
    # Recuperar y tokenizar todos los captions necesarios en una sola llamada
    captions = [random.choice(caption_templates).format(**TRACT_LIST[LABELS[label]]) for label in labels]
    tokenized_texts_batch = tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
    

    return GeoBatch.from_data_list(batch), {'input_ids': tokenized_texts_batch['input_ids'], 'attention_mask': tokenized_texts_batch['attention_mask']}