# =================================================================================================
# MODEL
from encoders import CFG  # Importar la configuración de los modelos
from encoders import GCN_Encoder, GAT_Encoder, Text_Encoder, ProjectionHead  # Importar los modelos y capas necesarios
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer

class Multimodal_Text_Graph_Model(pl.LightningModule):
    """
    Modelo multimodal que combina la información de un grafo y un texto para calcular la similitud entre ambos.

    Args:
    - text_encoder_model: str. Nombre del modelo de lenguaje pre-entrenado a utilizar para codificar el texto.
    - text_embedding: int. Dimensión de la representación del texto.
    - graph_model_name: str. Nombre del modelo de grafos a utilizar.
    - graph_embedding: int. Dimensión de la representación del grafo.
    - graph_channels: int. Número de canales en la capa de convolución del modelo de grafos.
    - projection_dim: int. Dimensión de la representación proyectada.
    - temperature: float. Temperatura para suavizar la distribución de similitud entre los pares de ejemplos.

    Attributes:
    - text_encoder: Text_Encoder. Modelo de lenguaje pre-entrenado para codificar el texto.
    - graph_encoder: GCN_Encoder o GAT_Encoder. Modelo de grafos para codificar la información del grafo.
    - text_projection_head: ProjectionHead. Capa de proyección para la representación del texto.
    - graph_projection_head: ProjectionHead. Capa de proyección para la representación del grafo.
    - temperature: float. Temperatura para suavizar la distribución de similitud entre los pares de ejemplos.

    Methods:
    - forward(graph, text): Calcula la similitud entre el grafo y el texto.
    - training_step(batch, batch_idx): Realiza un paso de entrenamiento.
    - validation_step(batch, batch_idx): Realiza un paso de validación.
    - test_step(batch, batch_idx): Realiza un paso de test.
    - configure_optimizers(): Configura el optimizador y el scheduler.
    - predict_step(batch, batch_idx, dataloader_idx): Realiza un paso de predicción.
    - cross_entropy(preds, targets, reduction): Calcula la entropía cruzada entre las predicciones y los targets.
    """
    def __init__(self, 
                 text_encoder_model = CFG.text_encoder_model,
                 text_embedding = CFG.text_embedding,
                 graph_model_name = CFG.graph_model_name,
                 graph_embedding = CFG.graph_embedding,
                 graph_channels = CFG.graph_channels,
                 projection_dim = CFG.projection_dim,
                 temperature = CFG.temperature):
        super(Multimodal_Text_Graph_Model, self).__init__()
        
        self.text_encoder = Text_Encoder(text_encoder_model, trainable=False)
        if graph_model_name == "GraphConvolutionalNetwork":
            self.graph_encoder = GCN_Encoder(graph_channels, graph_embedding)
        elif graph_model_name == "GraphAttentionNetwork":
            self.graph_encoder = GAT_Encoder(graph_channels, graph_embedding)
        else:
            raise ValueError("Invalid graph model name")
        
        # Proyectar representaciones a un espacio comun de menor dimension
        self.text_projection_head = ProjectionHead(text_embedding, projection_dim) 
        self.graph_projection_head = ProjectionHead(graph_embedding, projection_dim)
        self.temperature = temperature
        
    def forward(self, graph, text):
        # Encode the text and project the representations
        text_projections = self.text_encoder(text) # (batch_size, text_embedding)
        graph_projections = self.graph_encoder(graph) # (batch_size, graph_embedding)
        
        text_projections = self.text_projection_head(text_projections) # (batch_size, projection_dim)
        graph_projections = self.graph_projection_head(graph_projections) # (batch_size, projection_dim)
        
        # Calculating the Loss
        # Temperaturas más bajas hacen la distribución más aguda (haciendo que las diferencias 
        # entre las mayores y menores similitudes sean más pronunciadas).
        logits = (text_projections @ graph_projections.T) / self.temperature 
        graphs_similarity = graph_projections @ graph_projections.T # (batch_size, batch_size)
        texts_similarity = text_projections @ text_projections.T
        targets = F.softmax(
            (graphs_similarity + texts_similarity) / 2 * self.temperature, dim = -1
        )
        texts_loss = self.cross_entropy(logits, targets, reduction='none')
        graphs_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (graphs_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

    def training_step(self, batch, batch_idx):
        graph, text = batch
        loss = self.forward(graph, text)
        self.log('train_loss', loss, batch_size=128)
        return loss
    
    def validation_step(self, batch, batch_idx):
        graph, text = batch
        loss = self.forward(graph, text)
        self.log('val_loss', loss, batch_size=128)
        return loss
    
    def test_step(self, batch, batch_idx):
        graph, text = batch
        loss = self.forward(graph, text)
        self.log('test_loss', loss, batch_size=128)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Falta implementar
        return self(batch)
        
    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()


# =================================================================================================
# DATASET 
import random
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch as GeoBatch

class RandomTextGraphDataset(Dataset):
    def __init__(self, root="", transform=None, pre_transform=None, num_samples=4000, num_max_nodes=30, num_node_features=4, return_tokenized_text=False):
        # Note: 'root' is required by PyG Dataset, but you might not need it for generating random data.
        super(RandomTextGraphDataset, self).__init__(root, transform, pre_transform)
        self.num_samples = num_samples # Numero de ejemplos en el dataset
        self.num_nodes = num_max_nodes # Numero maximo de nodos en cada grafo
        self.num_node_feat = num_node_features # Numero de caracteristicas de cada nodo
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.return_tokenized_text = return_tokenized_text
    
    def len(self):
        return self.num_samples

    def get(self, idx):
        graph_data = self.generate_random_graph_data()
        text = self.generate_random_text()
        if self.return_tokenized_text:
            return {'graph': graph_data, 'text': self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")}
        else:
            return {'graph': graph_data, 'text': text}

    def generate_random_text(self):
        words = ["left arcuate fasciculus", "right arcuate fasciculus", "left cingulum", "right fornix", "left fornix", "right cingulum", "left uncinate fasciculus", "right uncinate fasciculus"]
        return random.choice(words) # Seleccionar una palabra aleatoria

    def generate_random_graph_data(self):
        n = np.random.randint(20, self.num_nodes)
        x = torch.randn(n, self.num_node_feat)# (nun_nodes, num_node_features)
        edge_index = torch.tensor([[i, i+1] for i in range(n-1)] + [[i+1, i] for i in range(n-1)], dtype=torch.long).T # (2, num_edges)
        return Data(x=x, edge_index=edge_index) # Crear un objeto Data de PyG

# =================================================================================================
# DATA MODULE
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.data import Batch as GeoBatch
from torch.nn.utils.rnn import pad_sequence

class TextGraphDataModule(LightningDataModule):
    def __init__(self, dataset_class, batch_size=64, num_workers=4):
        super().__init__()
        self.dataset_class = dataset_class
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = self.dataset_class(num_samples=10000, return_tokenized_text=True)
        self.val_dataset = self.dataset_class(num_samples=1000, return_tokenized_text=True)
        self.test_dataset = self.dataset_class(num_samples=1000, return_tokenized_text=True)
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.custom_collate_fn, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.custom_collate_fn, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.custom_collate_fn, num_workers=self.num_workers, persistent_workers=True)
    

    @staticmethod
    def custom_collate_fn(batch):
        graphs = [item['graph'] for item in batch]
        input_ids = [item['text']['input_ids'].squeeze(0) for item in batch]
        attention_masks = [item['text']['attention_mask'].squeeze(0) for item in batch]

        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

        batched_graphs = GeoBatch.from_data_list(graphs)

        return batched_graphs, {'input_ids': padded_input_ids, 'attention_mask': padded_attention_masks}


if __name__ == "__main__":
    pass
    from pytorch_lightning import Trainer, LightningDataModule
    
    dataset = RandomTextGraphDataset(num_samples=4000, return_tokenized_text=True)
    datamodule = TextGraphDataModule(RandomTextGraphDataset, batch_size=128, num_workers=4)


    model = Multimodal_Text_Graph_Model()
    trainer = Trainer(max_epochs=10)  # `gpus=1` indica usar una GPU. Usa `gpus=-1` para usar todas las GPUs disponibles.
    trainer.fit(model, datamodule)

