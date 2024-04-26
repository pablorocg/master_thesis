import torch
from transformers import AutoModel
import torch.nn as nn
import torch.nn.functional as F
from config import CFG

class TextEncoder(nn.Module):
    """Text Encoder that wraps a pretrained transformer model for embedding extraction."""

    def __init__(self, 
                pretrained_model_name_or_path = CFG.text_encoder_name,
                trainable: bool = False) -> None:
        """
        Initializes the TextEncoder with a pretrained model.
        """
        super(TextEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        
        # Freeze the model's parameters if not trainable
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0
        

    def forward(self, tokenized_text: dict) -> torch.Tensor:
        """Performs a forward pass through the model to obtain embeddings."""

    
        input_ids, attention_mask = tokenized_text['input_ids'], tokenized_text['attention_mask']
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, self.target_token_idx, :]
        # model_output = self.model(**tokenized_text)
        # sentence_embeddings = self.meanpooling(model_output, tokenized_text['attention_mask'])
        # # Normalize embeddings
        # # sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        # return sentence_embeddings
        
    
    def meanpooling(self, output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Applies mean pooling on the model's output embeddings, taking the attention mask into account."""
        embeddings = output[0]  # The first element of model_output contains all token embeddings.
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

        

class ProjectionHead(nn.Module):
    """
    Proyección de las embeddings de texto a un espacio de dimensión reducida.
    """
    def __init__(
        self,
        embedding_dim,# Salida del modelo de lenguaje (768)
        projection_dim=CFG.projection_head_output_dim, # Dimensión de la proyección (256)
        dropout=CFG.projection_head_dropout
    ):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class ClassifierHead(nn.Module):
    """
    Capa FC con activación softmax para que clasifique la clase.
    """
    def __init__(
        self,
        projection_dim=CFG.projection_head_output_dim, # Dimensión de la proyección (512)
        n_classes=CFG.n_classes # Número de clases a clasificar (32)
    ):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Linear(projection_dim, n_classes)
        # self.softmax = nn.Softmax(dim=1)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.fc(x)
        return self.softmax(x)

# ======================================GRAPH ENCODER============================================================

import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from torch_geometric.data import Data
from torch.nn import ModuleList


class Graph_Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(Graph_Conv_Block, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.LeakyReLU()
        if dropout:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.relu(x)
        x = self.bn(x)
        if self.dropout:
            x = self.dropout(x)
        return x
    
class GCN_Encoder(nn.Module):
    def __init__(self, 
                 in_channels = CFG.graph_encoder_input_channels, 
                 hidden_dim = CFG.graph_encoder_hidden_channels, 
                 out_channels = CFG.graph_encoder_graph_embedding ,
                 dropout = CFG.graph_encoder_dropout,
                 n_hidden_blocks = CFG.graph_encoder_n_hidden_blocks
                 ):
        
        super(GCN_Encoder, self).__init__()
        self.input_block = Graph_Conv_Block(in_channels, hidden_dim, dropout)
        self.hidden_blocks = nn.ModuleList([Graph_Conv_Block(hidden_dim, hidden_dim, dropout) for _ in range(n_hidden_blocks - 1)])
        self.output_block = Graph_Conv_Block(hidden_dim, out_channels, dropout)
        

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_block(x, edge_index)
        for layer in self.hidden_blocks:
            x = layer(x, edge_index)
        x = self.output_block(x, edge_index)
        return global_mean_pool(x, batch) # (batch_size, out_channels)



# class GATv2_Block(nn.Module):
#     def __init__(self, in_channels, out_channels, heads, dropout=0.0):  # Establecer un valor predeterminado para dropout
#         super(GATv2_Block, self).__init__()
#         self.conv = GATv2Conv(in_channels, out_channels, heads=heads, concat=False, dropout=dropout)
#         self.relu = nn.GELU()
#         self.bn = BatchNorm(out_channels)
#         self.dropout = nn.Dropout(p=dropout)  # Siempre inicializar, pero el dropout será 0 si no se desea

#     def forward(self, x, edge_index):  # Cambiar la firma para aceptar componentes del grafo directamente
#         x = self.conv(x, edge_index)
#         x = self.relu(x)
#         x = self.bn(x)
#         x = self.dropout(x)
#         return x
    
# class GATv2_Encoder(nn.Module):
#     def __init__(self, 
#                  in_channels,  # Valores configurables
#                  hidden_channels, 
#                  out_channels, 
#                  heads, 
#                  dropout=0.1,
#                  n_hidden_blocks=1):
        
#         super(GATv2_Encoder, self).__init__()
#         # Asegúrate de que los canales de salida de la capa anterior coincidan con los canales de entrada de la siguiente
#         self.input_block = GATv2_Block(in_channels, hidden_channels, heads, dropout)
#         # No es necesario multiplicar por heads ya que concat=False en GATv2Conv reduce la dimensión
#         self.hidden_blocks = ModuleList([GATv2_Block(hidden_channels, hidden_channels, heads, dropout) for _ in range(n_hidden_blocks)])
#         self.output_block = GATv2_Block(hidden_channels, out_channels, 1, dropout)

#     def forward(self, graph:Data):
#         x, edge_index, batch = graph.x, graph.edge_index, graph.batch
#         x = self.input_block(x, edge_index)
#         for layer in self.hidden_blocks:
#             x = layer(x, edge_index)
#         x = self.output_block(x, edge_index)
#         return global_mean_pool(x, batch)

       