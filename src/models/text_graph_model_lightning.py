from encoders import GCN_Encoder, GAT_Encoder, CFG, Text_Encoder, ProjectionHead
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch_geometric
from transformers import AutoTokenizer
import pytorch_lightning as pl

class Multimodal_Text_Graph_Model(pl.LightningModule):
    def __init__(self, 
                 text_encoder_model = CFG.text_encoder_model,
                 text_embedding = CFG.text_embedding,
                 graph_model_name = CFG.graph_model_name,
                 graph_embedding = CFG.graph_embedding,
                 graph_channels = CFG.graph_channels,
                 projection_dim = CFG.projection_dim,
                 temperature = CFG.temperature,
                 device = CFG.device):
        
        super(Multimodal_Text_Graph_Model, self).__init__()
        
        self.text_encoder = Text_Encoder(text_encoder_model)#, text_embedding)
        
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
        logits = (text_projections @ graph_projections.T) / self.temperature
        graphs_similarity = graph_projections @ graph_projections.T
        texts_similarity = text_projections @ text_projections.T
        targets = F.softmax(
            (graphs_similarity + texts_similarity) / 2 * self.temperature, dim = -1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        graphs_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (graphs_loss + texts_loss) / 2.0 # shape: (batch_size)
        
        return loss.mean()

    def training_step(self, batch, batch_idx):
        graph, text = batch
        loss = self.forward(graph, text)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
    

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


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
        self.train_dataset = self.dataset_class(split='train', return_tokenized_text=True)
        self.val_dataset = self.dataset_class(split='val', return_tokenized_text=True)
        self.test_dataset = self.dataset_class(split='test', return_tokenized_text=True)
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.custom_collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.custom_collate_fn, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.custom_collate_fn, num_workers=self.num_workers)
    

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
    
    from pytorch_lightning import Trainer, LightningDataModule
    from tools import 

    # Suponiendo que RandomTextGraphDataset está definido como antes
    dataset = RandomTextGraphDataset(num_samples=4000, return_tokenized_text=True)
    


    # Crear el DataLoader con la función de collate_fn personalizada
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, collate_fn=custom_collate_fn)


    model = MultimodalTextGraphModel()
    trainer = Trainer(max_epochs=100, gpus=1)  # `gpus=1` indica usar una GPU. Usa `gpus=-1` para usar todas las GPUs disponibles.
    trainer.fit(model, dataloader)

    # Para validación y prueba, puedes pasar dataloaders adicionales usando los argumentos `val_dataloaders` y `test_dataloaders` de Trainer.fit() y Trainer.test().
