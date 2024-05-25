import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from torch.nn import ModuleList
import torch.nn.functional as F



#================================================ENCODER=====================================================
class GraphConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(GraphConvBlock, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.relu(x)
        x = self.bn(x)
        if self.dropout:
            x = self.dropout(x)
        return x
    
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dropout, n_hidden_blocks):
        super(GCNEncoder, self).__init__()
        self.input_block = GraphConvBlock(in_channels, hidden_dim, dropout)
        self.hidden_blocks = nn.ModuleList([GraphConvBlock(hidden_dim, hidden_dim, dropout) for _ in range(n_hidden_blocks - 1)])
        self.output_block = GraphConvBlock(hidden_dim, out_channels, dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_block(x, edge_index)
        for layer in self.hidden_blocks:
            x = layer(x, edge_index)
        x = self.output_block(x, edge_index)
        return global_mean_pool(x, batch) # (batch_size, out_channels)




#================================================PROJECTION HEAD=====================================================
class ProjectionHead(nn.Module):
    """
    Proyección de las embeddings de texto a un espacio de dimensión reducida.
    """
    def __init__(
        self,
        embedding_dim,# Salida del modelo de lenguaje (768)
        projection_dim, # Dimensión de la proyección (256)
        dropout=0.1
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


#================================================CLASSIFIER HEADs=====================================================
class ClassifierHead(nn.Module):
    """
    Capa FC con activación softmax para que clasifique la clase.
    """
    def __init__(self, projection_dim, n_classes):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Linear(projection_dim, n_classes)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.fc(x)
        return self.softmax(x)


#================================================MODEL======================================================
class SiameseGraphNetwork(nn.Module):
    def __init__(self, encoder, projection_head, classifier):
        super(SiameseGraphNetwork, self).__init__()
        self.encoder = encoder
        self.projection_head = projection_head
        self.classifier = classifier

    def forward(self, graph):
        x_1 = self.encoder(graph)
        x_1 = self.projection_head(x_1)
        x1_norm = F.normalize(x_1, p=2, dim=1)

        c1 = self.classifier(x1_norm)
        return x1_norm, c1
    

#==================================================================================================================

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MLP, GINConv, global_add_pool

class GIN(torch.nn.Module):# GRaph Isomorphism Network
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)

    def forward(self, x, edge_index, batch, batch_size):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        # Pass the batch size to avoid CPU communication/graph breaks:
        x = global_add_pool(x, batch, size=batch_size)
        return self.mlp(x)
    

#     model = GIN(
#     in_channels=dataset.num_features,
#     hidden_channels=32,
#     out_channels=dataset.num_classes,
#     num_layers=5,
# ).to(device)

# # Compile the model into an optimized version:
# model = torch.compile(model, dynamic=True)