import torch
from transformers import AutoModel, AutoModelForMaskedLM
import torch.nn as nn
import torch.nn.functional as F
from config import CFG

class Text_Encoder(nn.Module):
    """Text Encoder that wraps a pretrained transformer model for embedding extraction."""

    def __init__(
                self, 
                pretrained_model_name_or_path: str = CFG.text_encoder_model,
                trainable: bool = False) -> None:
        """
        Initializes the TextEncoder with a pretrained model.
        """
        super(Text_Encoder, self).__init__()
        if pretrained_model_name_or_path == "distilbert-base-uncased":
            self.model = AutoModel.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float32)
        elif pretrained_model_name_or_path == "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext":
            self.model = AutoModelForMaskedLM.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", torch_dtype=torch.float32)

        self.trainable = trainable
        # Freeze the parameters of the pretrained model to prevent updates during training.
        if not self.trainable:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True

    def forward(self, tokenized_text: dict) -> torch.Tensor:
        """Performs a forward pass through the model to obtain embeddings."""

        model_output = self.model(**tokenized_text)
        # Perform pooling
        sentence_embeddings = self.meanpooling(model_output, tokenized_text['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        # Devolver el token [CLS]
        # return output.last_hidden_state[:, 0, :]
        return sentence_embeddings
    
    def meanpooling(self, output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Applies mean pooling on the model's output embeddings, taking the attention mask into account."""

        embeddings = output[0]  # The first element of model_output contains all token embeddings.
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)



class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
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
    

# ======================================GRAPH ENCODER============================================================

import torch.nn as nn
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool, BatchNorm
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.data import Data
from torch.nn import ModuleList

class GAT_Encoder(nn.Module):
    def __init__(self, 
                 in_channels = CFG.graph_channels, 
                 hidden_dim = CFG.graph_embedding, 
                 out_channels = CFG.graph_embedding,
                 dropout = CFG.dropout
                 ):
        
        super(GAT_Encoder, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_dim, heads=4, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout)
        self.conv3 = GATv2Conv(hidden_dim * 4, out_channels, heads=4, dropout=dropout)
        self.relu = nn.LeakyReLU()
        self.bn = BatchNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph:Data):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        out = global_mean_pool(x, batch)
        return out
    
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
        x = self.dropout(x)
        return x
    
class GCN_Encoder(nn.Module):
    def __init__(self, 
                 in_channels = CFG.graph_channels, 
                 hidden_dim = CFG.graph_embedding, 
                 out_channels = CFG.graph_embedding,
                 dropout = CFG.dropout
                 ):
        
        super(GCN_Encoder, self).__init__()
        self.first_block = Graph_Conv_Block(in_channels, hidden_dim, dropout)
        self.intermediate_layers = ModuleList([Graph_Conv_Block(hidden_dim, hidden_dim, dropout) for _ in range(5)])
        self.last_block = Graph_Conv_Block(hidden_dim, out_channels, dropout)
        

    def forward(self, graph:Data):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch

        x = self.first_block(x, edge_index)
        for layer in self.intermediate_layers:
            x = layer(x, edge_index)
        x = self.last_block(x, edge_index)

        return global_mean_pool(x, batch)



# ======================================GRAPH AUTOENCODER============================================================
import torch
from torch_geometric.nn import GAE, InnerProductDecoder


class CustomGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomGCNLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.relu = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_channels)
        

    def forward(self, graph:Data):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = self.conv(x, edge_index)
        x = self.relu(x)
        x = self.bn1(x)
        return x
    

    
    

    