from pytorch_lightning import LightningDataModule, Callback
import json
import os
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import BaseTransform, Compose
from transformers import AutoTokenizer
import random
from torch_geometric.data import Batch as GeoBatch
from torch.utils.data import DataLoader

class MaxMinNormalization(BaseTransform):
    def __init__(self, max_values=None, min_values=None):
        """
        Initialize the normalization transform with optional max and min values.
        If not provided, they should be computed from the dataset.
        """
        self.max_values = max_values if max_values is not None else \
            torch.tensor([76.03170776367188, 77.9359130859375, 88.72427368164062], 
                         dtype=torch.float)
        self.min_values = min_values if min_values is not None else \
            torch.tensor([-73.90082550048828, -112.23554992675781, -79.38320922851562], 
                         dtype=torch.float)
        
    def __call__(self, data: Data) -> Data:
        """
        Apply min-max normalization to the node features.
        """
        data.x = (data.x - self.min_values) / (self.max_values - self.min_values)
        return data



# class FiberGraphDataset(Dataset):
#     def __init__(self, root, transform=None, pre_transform=None):
#         super(FiberGraphDataset, self).__init__(root, transform, pre_transform)
#         
#     @property
#     def processed_dir(self):
#         return os.path.join(self.root)
# 
#     @property
#     def processed_file_names(self):
#         return os.listdir(self.root)
#     
#     def len(self):
#         return len(self.processed_file_names)
#     
#     def get(self, idx):
#         subject = self.processed_file_names[idx]# Seleccionar un sujeto
#         #subject = torch.load(os.path.join(self.processed_dir, subject))# Cargar los grafos del sujeto
#         graphs = torch.load(os.path.join(self.processed_dir, subject))# Cargar los grafos del sujeto
#         
#         if self.transform: # Aplicar transformaciones
#             graphs = self.transform(graphs)
#         
#         # Devolver los grafos de uno en uno
#         return graphs

# from torch.utils.data import IterableDataset as TorchDataset
from torch.utils.data import ChainDataset
# 
# class SubjectDS(TorchDataset):
#     # Dataset para un solo sujeto (un solo archivo)
#     def __init__(self, file, transform=None):
#         self.file = file
#         self.transform = transform
# 
#     def __len__(self):
#         return 1#torch.load(self.file).num_graphs
#     
#     def __getitem__(self, idx):
#         graphs = torch.load(self.file)
#         if self.transform:
#             graphs = self.transform(graphs)
# 
#         # Convert to list of Data objects
#         graphs = [Data(x=graph.x, edge_index=graph.edge_index, y=graph.y) for graph in graphs]
#         return graphs
import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Data

# class SubjectDS(IterableDataset):
#     def __init__(self, file, transform=None):
#         super().__init__()
#         self.file = file
#         self.transform = transform
# 
#     def __iter__(self):
#         # Load the file just once
#         data = torch.load(self.file)  # Assuming data is a list or iterable of Data objects
#         
#         if self.transform:
#             data = self.transform(data)
#         
#         # data es un DataBatch de Pytorch Geometric
#         # Devolver los grafos de uno en uno
#         graph_list = data.to_data_list()
#         # Mezclar los grafos aleatoriamente para evitar sesgos
#         random.shuffle(graph_list)
# 
#         # Samplear 1000 grafos de cada clase  o los que hayan
#         # TODO
#         
# 
# 
#         for graph in sampled_graph_list:
#             yield graph
        

import torch
import random
from collections import defaultdict
from torch.utils.data import IterableDataset
from torch_geometric.data import Data


class SubjectDS(IterableDataset):
    def __init__(self, file, transform=None, num_samples_per_class=2000):
        super().__init__()
        self.file = file
        self.transform = transform
        self.num_samples_per_class = num_samples_per_class

    def __iter__(self):
        # Load the file just once
        data_batch = torch.load(self.file)  # Load a DataBatch object

        if self.transform:
            data_batch = self.transform(data_batch)
        
        # Convert DataBatch to a list of Data objects
        graph_list = data_batch.to_data_list()

        # Mezclar los grafos aleatoriamente para evitar sesgos
        random.shuffle(graph_list)

        # Samplear grafos de cada clase
        class_samples = self.sample_graphs_by_class(graph_list, self.num_samples_per_class)
        random.shuffle(class_samples)
        for graph in class_samples:
            yield graph


    def sample_graphs_by_class(self, graphs, num_samples):
        # Organize graphs by class using a more streamlined approach
        class_dict = defaultdict(list)
        append_to_class = class_dict[None].append  # Handle graphs without labels more efficiently

        # Pre-process to reduce attribute access
        for graph in graphs:
            label = getattr(graph, 'y', None)
            if label is not None and label is not None:
                class_dict[label.item()].append(graph)
            else:
                append_to_class(graph)

        # Sample graphs
        sampled_graphs = []
        extend_sampled = sampled_graphs.extend
        for class_label, graph_list in class_dict.items():
            if len(graph_list) > num_samples:
                extend_sampled(random.sample(graph_list, num_samples))
            else:
                extend_sampled(graph_list)
        return sampled_graphs





#---------------------------------------

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os

class SubjectDataModule(pl.LightningDataModule):
    def __init__(self, 
                 train, 
                 val, 
                 test, 
                 fiber_captions_file, 
                 tokenizer, 
                 pos_neg_pair_ratio,
                 batch_size=32, 
                 num_workers=4):
        super().__init__()
        
        self.train = [os.path.join(train, file) for file in os.listdir(train)]# Rutas de los archivos de entrenamiento
        self.val = [os.path.join(val, file) for file in os.listdir(val)] # Rutas de los archivos de validación
        self.test = [os.path.join(test, file) for file in os.listdir(test)] # Rutas de los archivos de test

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.TRACT_LIST = self._load_json_file(fiber_captions_file) 

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.pos_neg_pair_ratio = pos_neg_pair_ratio

        self.batch_size = batch_size
        self.num_workers = num_workers
      
        self.transforms = Compose([
            MaxMinNormalization()
        ])

    def setup(self, stage=None):
        # Este método es utilizado para configurar datasets para cada etapa
        if stage == 'fit' or stage is None:
            # separar subject en ruta y nombre de archivo con pathlib
            datasets = [SubjectDS(file = subj, transform = self.transforms) for subj in self.train]
            self.train_ds = ChainDataset(datasets)

            # separar subject en ruta y nombre de archivo con pathlib
            datasets = [SubjectDS(file = subj, transform = self.transforms) for subj in self.val]
            self.val_ds = ChainDataset(datasets)
        
        if stage == 'test' or stage is None:
            # separar subject en ruta y nombre de archivo con pathlib
            
            datasets = [SubjectDS(file = subj, transform = self.transforms) for subj in self.test]
            
            self.test_ds = ChainDataset(datasets)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size = self.batch_size,  
                   collate_fn = self.collate_function_v2, num_workers = self.num_workers, 
                   pin_memory = False, drop_last = True)
        
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size = self.batch_size, 
                          shuffle = False, collate_fn = self.collate_function_v2, 
                          num_workers = self.num_workers, 
                          pin_memory = True, drop_last = True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size = self.batch_size,  
                          collate_fn = self.collate_function_v2, num_workers = self.num_workers, 
                          pin_memory = True, drop_last = True)
        

    def _load_json_file(self, file_path):
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
        
    def _save_json_file(data, filename):
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    def collate_function_v2(self, batch):
        """
        Funcion para el DataLoader
        """
        
        LABELS = {value["id"]: key for key, value in self.TRACT_LIST.items()}# Diccionario id -> Etiqueta
        caption_templates = [
                "Exploring the {type} tract of {tract} on the {side}, pivotal for {functional_significance}, characterized by {structural_characteristics}, with known {vulnerability_to_disease}.",
                "The {type} {tract} on the {side} side: A key player in {functional_significance}, structured with {structural_characteristics}, and susceptible to {vulnerability_to_disease}.",
                "Insights into the {tract}'s {type} fibers on the {side}: Essential for {functional_significance}, featuring {structural_characteristics}, and vulnerable to {vulnerability_to_disease}.",
                "{type} fibers of the {tract} on the {side} side underscore {functional_significance}, demonstrate {structural_characteristics}, and face threats from {vulnerability_to_disease}.",
                "Highlighting the {type} tract, {tract}, on the {side} side: Central to {functional_significance}, with {structural_characteristics}, and at risk of {vulnerability_to_disease}.",
                "The {tract}'s {type} fibers on the {side}: Integral for {functional_significance}, endowed with {structural_characteristics}, yet at risk due to {vulnerability_to_disease}.",
                "Delving into the {tract} on the {side} side: A {type} fiber crucial for {functional_significance}, defined by {structural_characteristics}, with a susceptibility to {vulnerability_to_disease}.",
                "A closer look at the {type} {tract} on the {side}, a cornerstone of {functional_significance}, built on {structural_characteristics}, with exposure to {vulnerability_to_disease}.",
                "Understanding the {type} {tract} on the {side}: Fundamental for {functional_significance}, based on {structural_characteristics}, with potential for {vulnerability_to_disease}.",
                "The {tract} on the {side}, a {type} tract: Vital for {functional_significance}, composed of {structural_characteristics}, and prone to {vulnerability_to_disease}.",
                "Navigating the {tract}'s {type} fibers on the {side}: Key to {functional_significance}, showcasing {structural_characteristics}, while facing {vulnerability_to_disease}.",
                "{type} fiber dynamics in the {tract} on the {side}: Driving {functional_significance}, supported by {structural_characteristics}, with concerns over {vulnerability_to_disease}.",
                "The role of the {type} {tract} on the {side} in {functional_significance}, its structure defined by {structural_characteristics}, and its challenges with {vulnerability_to_disease}.",
                "Spotlight on the {tract}'s {type} aspect on the {side}: A linchpin in {functional_significance}, with a basis in {structural_characteristics}, amid {vulnerability_to_disease}.",
                "Charting the course of the {type} {tract} on the {side}: Crucial for {functional_significance}, with a foundation of {structural_characteristics}, challenged by {vulnerability_to_disease}.",
                "Dissecting the {type} {tract} on the {side}: Central to {functional_significance}, with distinct {structural_characteristics}, and under threat from {vulnerability_to_disease}.",
                "The {type} fibers within the {tract} on the {side}: Pillars of {functional_significance}, with unique {structural_characteristics}, and susceptible to {vulnerability_to_disease}.",
                "Unveiling the {type} {tract} on the {side}: A conduit for {functional_significance}, designed with {structural_characteristics}, yet vulnerable to {vulnerability_to_disease}.",
                "The architecture of the {type} {tract} on the {side}: Scaffolding for {functional_significance}, erected on {structural_characteristics}, with a risk of {vulnerability_to_disease}.",
                "Deciphering the {type} {tract} on the {side}: Essential for {functional_significance}, with {structural_characteristics}, and facing risks of {vulnerability_to_disease}.",
                "The {tract}'s {type} path on the {side}: Spearheading {functional_significance}, built upon {structural_characteristics}, with vulnerabilities to {vulnerability_to_disease}.",
                "Insight into the {type} {tract} on the {side}: Spearheads {functional_significance}, relying on {structural_characteristics}, amidst threats from {vulnerability_to_disease}.",
                "Showcasing the {type} {tract} on the {side}: A crucible for {functional_significance}, framed by {structural_characteristics}, with exposure to {vulnerability_to_disease}.",
                "The essence of the {type} {tract} on the {side}: Harnessing {functional_significance}, through {structural_characteristics}, with a nod to {vulnerability_to_disease}.",
                "Evaluating the {type} {tract} on the {side}: A nexus for {functional_significance}, supported by {structural_characteristics}, with an eye on {vulnerability_to_disease}.",
                "Delineating the {side} {type} {tract}: A beacon for {functional_significance}, woven with {structural_characteristics}, under the shadow of {vulnerability_to_disease}.",
                "The {tract} on the {side}: A {type} network pivotal in {functional_significance}, structured via {structural_characteristics}, with an Achilles' heel of {vulnerability_to_disease}.",
                "A deep dive into the {type} {tract} on the {side}: Orchestrating {functional_significance}, through the lens of {structural_characteristics}, amidst battles with {vulnerability_to_disease}.",
                "Unraveling the {type} {tract} on the {side}: A cornerstone in {functional_significance}, pieced together by {structural_characteristics}, with a vulnerability to {vulnerability_to_disease}.",
                "Illuminating the {type} {tract} on the {side}: A catalyst for {functional_significance}, held together by {structural_characteristics}, yet dancing with {vulnerability_to_disease}.",
                "Navigating the nuances of the {type} {tract} on the {side}: Engineered for {functional_significance}, detailed by {structural_characteristics}, in the face of {vulnerability_to_disease}.",
                "The intricate web of the {type} {tract} on the {side}: A symphony of {functional_significance}, composed of {structural_characteristics}, with whispers of {vulnerability_to_disease}.",
                "The {tract} on the {side}: Where {type} fibers blend {functional_significance} with {structural_characteristics}, amidst the specter of {vulnerability_to_disease}.",
                "A glimpse into the {type} {tract} on the {side}: Harmonizing {functional_significance} with {structural_characteristics}, against the backdrop of {vulnerability_to_disease}.",
                "The {tract}'s {type} ensemble on the {side}: An odyssey of {functional_significance}, mapped through {structural_characteristics}, with a narrative of {vulnerability_to_disease}.",
                "The {type} {tract} on the {side}: A testament to {functional_significance}, sculpted by {structural_characteristics}, with tales of {vulnerability_to_disease}.",
                "Decoding the {type} {tract} on the {side}: A saga of {functional_significance}, etched with {structural_characteristics}, shadowed by {vulnerability_to_disease}.",
                "The {tract} on the {side}: A {type} odyssey defined by {functional_significance}, articulated through {structural_characteristics}, with challenges of {vulnerability_to_disease}.",
                "Embarking on the {type} {tract} journey on the {side}: Driven by {functional_significance}, carved out of {structural_characteristics}, with hurdles of {vulnerability_to_disease}.",
                "The {type} {tract} on the {side}: A crucible of {functional_significance}, molded by {structural_characteristics}, with the trials of {vulnerability_to_disease}.",
                "The {tract}'s {type} legacy on the {side}: Stitched with {functional_significance}, through the fabric of {structural_characteristics}, bracing against {vulnerability_to_disease}.",
                "Exploring the {type} {tract} on the {side}: A quilt of {functional_significance}, with patches of {structural_characteristics}, and threads of {vulnerability_to_disease}.",
                "The {tract} on the {side}: A {type} tapestry rich in {functional_significance}, woven with {structural_characteristics}, tinged with {vulnerability_to_disease}.",
                "The {type} {tract} on the {side}: A mosaic of {functional_significance}, pieced together by {structural_characteristics}, with cracks of {vulnerability_to_disease}.",
                "Navigating the {type} {tract} on the {side}: A journey through {functional_significance}, charted by {structural_characteristics}, navigating {vulnerability_to_disease}.",
                "The {tract} on the {side}: A {type} vista overlooking {functional_significance}, painted with {structural_characteristics}, under clouds of {vulnerability_to_disease}.",
                "Unlocking the {type} {tract} on the {side}: A vault of {functional_significance}, secured by {structural_characteristics}, under siege by {vulnerability_to_disease}.",
                "The {type} {tract} on the {side}: A beacon across {functional_significance}, shining through {structural_characteristics}, amid storms of {vulnerability_to_disease}.",
                "Journeying through the {type} {tract} on the {side}: A pathway lit by {functional_significance}, paved with {structural_characteristics}, weaving through {vulnerability_to_disease}.",
                "The {tract} on the {side}: A {type} symposium of {functional_significance}, curated with {structural_characteristics}, amidst the risks of {vulnerability_to_disease}.",
                "The {type} {tract} on the {side}: A dialogue between {functional_significance} and {structural_characteristics}, with an undercurrent of {vulnerability_to_disease}.",
                "Unraveling the {type} {tract} on the {side}: Threads of {functional_significance} intertwined with {structural_characteristics}, frayed by {vulnerability_to_disease}.",
                "The {type} {tract} on the {side}: A narrative woven from {functional_significance}, bound by {structural_characteristics}, and shadowed by {vulnerability_to_disease}.",
                "Charting the {type} {tract} on the {side}: A map of {functional_significance}, drawn with {structural_characteristics}, bordered by {vulnerability_to_disease}.",
                "The {tract} on the {side}: A {type} cornerstone anchoring {functional_significance}, chiseled from {structural_characteristics}, weathering {vulnerability_to_disease}.",
                "Amidst {vulnerability_to_disease}, the {type} {tract} on the {side} side stands out, marked by {structural_characteristics} and pivotal in {functional_significance}.",
                "Where {functional_significance} meets {structural_characteristics}, the {type} {tract} on the {side} navigates through {vulnerability_to_disease}.",
                "{vulnerability_to_disease} shadows the {type} {tract} on the {side}, a structure of {structural_characteristics} essential for {functional_significance}.",
                "In the realm of {functional_significance}, the {type} {tract} on the {side} emerges, built on {structural_characteristics}, yet tested by {vulnerability_to_disease}.",
                "Challenged by {vulnerability_to_disease}, the {type} {tract} on the {side} thrives, an epitome of {structural_characteristics} serving {functional_significance}.",
                "{structural_characteristics} define the {type} {tract} on the {side}, a beacon for {functional_significance} amidst {vulnerability_to_disease}.",
                "The {type} {tract} on the {side}, driven by {functional_significance}, showcases {structural_characteristics} while confronting {vulnerability_to_disease}.",
                "{functional_significance} is the hallmark of the {type} {tract} on the {side}, crafted from {structural_characteristics} and facing {vulnerability_to_disease}.",
                "Against the backdrop of {vulnerability_to_disease}, the {type} {tract} on the {side} excels in {functional_significance}, thanks to its {structural_characteristics}.",
                "The {type} {tract} on the {side}, a testament to {structural_characteristics}, champions {functional_significance} amid {vulnerability_to_disease}.",
                "Forged with {structural_characteristics}, the {type} {tract} on the {side} propels {functional_significance}, even as it grapples with {vulnerability_to_disease}.",
                "The {type} {tract} on the {side}, sculpted by {structural_characteristics}, enriches {functional_significance} against {vulnerability_to_disease}.",
                "Bearing {structural_characteristics}, the {type} {tract} on the {side} underpins {functional_significance}, resilient to {vulnerability_to_disease}.",
                "The {type} {tract} on the {side}, with its {structural_characteristics}, is a linchpin in {functional_significance}, enduring {vulnerability_to_disease}.",
                "Rooted in {structural_characteristics}, the {type} {tract} on the {side} elevates {functional_significance}, despite {vulnerability_to_disease}.",
                "Empowered by {structural_characteristics}, the {type} {tract} on the {side} fuels {functional_significance}, transcending {vulnerability_to_disease}.",
                "{vulnerability_to_disease} tests the {type} {tract} on the {side}, a construct of {structural_characteristics} and a vessel for {functional_significance}.",
                "The {type} {tract} on the {side}: a synthesis of {structural_characteristics}, serving {functional_significance}, amidst trials of {vulnerability_to_disease}.",
                "With {structural_characteristics} at its core, the {type} {tract} on the {side} epitomizes {functional_significance}, braving {vulnerability_to_disease}.",
                "A nexus of {structural_characteristics}, the {type} {tract} on the {side} radiates {functional_significance}, undeterred by {vulnerability_to_disease}.",
                "The {type} {tract} on the {side}, a paragon of {structural_characteristics}, emboldens {functional_significance}, despite {vulnerability_to_disease}.",
                "Anchored in {structural_characteristics}, the {type} {tract} on the {side} is a crucible for {functional_significance}, amidst {vulnerability_to_disease}.",
                "{structural_characteristics} are the foundation of the {type} {tract} on the {side}, powering {functional_significance}, amidst {vulnerability_to_disease}.",
                "A citadel of {structural_characteristics}, the {type} {tract} on the {side} is pivotal for {functional_significance}, despite {vulnerability_to_disease}.",
                "The {type} {tract} on the {side}, framed by {structural_characteristics}, is a bastion of {functional_significance}, facing {vulnerability_to_disease}.",
                "At the intersection of {structural_characteristics} and {functional_significance}, the {type} {tract} on the {side} confronts {vulnerability_to_disease}.",
                "The {type} {tract} on the {side}, a convergence of {structural_characteristics}, propels {functional_significance}, against the tide of {vulnerability_to_disease}.",
                "Imbued with {structural_characteristics}, the {type} {tract} on the {side} amplifies {functional_significance}, navigating {vulnerability_to_disease}.",
                "The {type} {tract} on the {side}, an embodiment of {structural_characteristics}, fortifies {functional_significance}, in defiance of {vulnerability_to_disease}.",
                "With {structural_characteristics} as its bedrock, the {type} {tract} on the {side} is instrumental in {functional_significance}, amidst the waves of {vulnerability_to_disease}."
            ]
        
        graph_labels = torch.tensor([graph.y.item() for graph in batch], dtype=torch.long)
        # # Obtener las etiquetas de los grafos
        # labels = [item.y for item in batch]
        # print(labels)
        # graph_labels = torch.stack(labels)
        # print(graph_labels)
        # # Convertir las etiquetas de los grafos a lista de enteros
        # graph_labels_2 = graph_labels.long()
        # # Convertir a list
        # graph_labels = graph_labels.tolist()
    
        
        # 1 positive pair -> text corresponds to graph
        # 0 negative pair -> text does not correspond to graph
        type_of_pair = (torch.rand(len(graph_labels)) < self.pos_neg_pair_ratio).long()
        
        text_labels = graph_labels.clone()
        for pos, value in enumerate(type_of_pair):
            if value == 0:
                text_labels[pos] = torch.randint(0, 32, (1,)).item()

        captions = []
        for label in text_labels:
            tract = self.TRACT_LIST[LABELS[label.item()]]
            input_data = {
                "type": tract["type"],
                "tract": tract["tract"],
                "side": tract["side"],
                "functional_significance": random.choice(tract["functional_significance"]),
                "structural_characteristics": random.choice(tract["structural_characteristics"]),
                "vulnerability_to_disease": random.choice(tract["vulnerability_to_disease"])
            }
            captions.append(random.choice(caption_templates).format(**input_data))
        
        tokenized_texts_batch = self.tokenizer(captions, 
                                               padding=True, 
                                               truncation=True, 
                                               return_tensors="pt")
        
        # Devolver el lote de grafos, los textos tokenizados, los graph_labels, los pos_neg_labels y los type_of_pair
        return GeoBatch.from_data_list(batch), \
            {'input_ids': tokenized_texts_batch['input_ids'], \
             'attention_mask': tokenized_texts_batch['attention_mask']}, \
              graph_labels, text_labels, type_of_pair # 1 positive pair -> text corresponds to graph, 0 negative pair -> text does not correspond to graph
            












#---------------------------------------




if __name__ == "__main__":

    from pytorch_lightning import Trainer
    from lightning import TextGraphModel

    dt_mod_args = {
        'train': '/app/dataset/Tractoinferno/tractoinferno_graphs/testset/raw',
        'val': '/app/dataset/Tractoinferno/tractoinferno_graphs/testset/raw',
        'test': '/app/dataset/Tractoinferno/tractoinferno_graphs/testset/raw',
        'fiber_captions_file': '/app/src/lightning_implementation/tractoinferno_text_data.json',
        'tokenizer': 'emilyalsentzer/Bio_ClinicalBERT',
        'pos_neg_pair_ratio': 0.5,
        'batch_size': 16,
        'num_workers': 4,
    }

    # 'val_files': 'dataset/Tractoinferno/tractoinferno_graphs/validset',
    # 'test_files': 'dataset/Tractoinferno/tractoinferno_graphs/testset',
    model_config = {
    "theta": 0.5,
    "margin": 0.5,
    "distance": 'euclidean',
    "weighted_loss": False,
    "n_classes": 32,
    "learning_rate": 1e-3,
    "batch_size": 256,
    "num_workers": 4,
    "text_encoder_name": "emilyalsentzer/Bio_ClinicalBERT",
    "text_embedding": 768,
    "graph_model_name": "GCN",
    "graph_embedding": 768,
    "graph_channels": 3,
    "projection_dim": 256,
    
    }
    
    
    from pytorch_lightning import Trainer
    import pathlib2 as pathlib
   
    
    
    # Initialize the model and the trainer
    datamodule = SubjectDataModule(**dt_mod_args)
    datamodule.setup(stage='fit')

    
    model = TextGraphModel(model_config)
    trainer = Trainer(devices=1, max_epochs=1, accelerator='gpu', limit_train_batches = 10000, limit_val_batches = 1000, limit_test_batches= 1000)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
      


 


    #datamod.setup(stage='test')
    #model = TextGraphModel(model_config)
    #trainer = Trainer(devices=1, max_epochs=1, accelerator='gpu')
    #trainer.fit(model, datamod)
    #trainer.test(model, datamod)
   

    
