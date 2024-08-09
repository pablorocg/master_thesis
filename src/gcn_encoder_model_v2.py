import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, BatchNorm, LayerNorm, global_mean_pool
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv,GraphConv, global_mean_pool, BatchNorm
from torch.nn import ModuleList
import torch.nn.functional as F


# class GraphConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, dropout=0.0):
#         super(GraphConvBlock, self).__init__()
#         self.conv = GCNConv(in_channels, out_channels)
#         self.bn = BatchNorm(out_channels)
#         self.activation = nn.GELU()
#         self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

#     def forward(self, x, edge_index):
#         # Apply graph convolution
#         x = self.conv(x, edge_index)
#         # Apply batch normalization
#         x = self.bn(x)
#         # Apply activation function
#         x = self.activation(x)
#         # Apply dropout if defined
#         if self.dropout:
#             x = self.dropout(x)
#         return x

class GraphConvBlock_v2(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(GraphConvBlock_v2, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.bn = BatchNorm(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.residual = (in_channels == out_channels)
        
    def forward(self, x, edge_index):
        identity = x
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = self.activation(x)

        if self.dropout:
            x = self.dropout(x)

        if self.residual:
            x += identity

        return x
    

class GCNEncoder_v2(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dropout=0.0, n_hidden_blocks=2):
        super(GCNEncoder_v2, self).__init__()
        self.input_block = GraphConvBlock_v2(in_channels, hidden_dim, dropout)
        self.hidden_blocks = self._make_hidden_layers(hidden_dim, dropout, n_hidden_blocks)
        self.output_block = GraphConvBlock_v2(hidden_dim, out_channels, dropout)
        self.attention_block = GATConv(hidden_dim, hidden_dim, heads=5, concat=False)
        self.layer_norm = LayerNorm(out_channels)

    def _make_hidden_layers(self, hidden_dim, dropout, n_hidden_blocks):
        layers = []
        for _ in range(n_hidden_blocks - 1):
            layers.append(GraphConvBlock_v2(hidden_dim, hidden_dim, dropout))
        return nn.ModuleList(layers)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_block(x, edge_index)
        for layer in self.hidden_blocks:
            x = layer(x, edge_index)
        x = self.attention_block(x, edge_index)
        x = self.output_block(x, edge_index)
        x = self.layer_norm(x)
        return global_mean_pool(x, batch)  # (batch_size, out_channels)




class ProjectionHead_v2(nn.Module):
    def __init__(self, in_features, projection_dim):
        super(ProjectionHead_v2, self).__init__()
        self.projection = nn.Linear(in_features, projection_dim)
        self.bn = nn.BatchNorm1d(projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        
        x = self.projection(x)
        x = self.bn(x)
        x = self.gelu(x)
        
        identity = x
        
        x = self.fc(x)
        x = x + identity

        x = self.layer_norm(x)
        return x
    
class ClassifierHead_v2(nn.Module):
    def __init__(self, in_features, num_classes):
        super(ClassifierHead_v2, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x

class SiameseGraphNetworkGCN_v2(nn.Module):
    def __init__(self, n_classes, normalize=True):
        super(SiameseGraphNetworkGCN_v2, self).__init__()
        
        self.encoder = GCNEncoder_v2(
            in_channels = 3, 
            hidden_dim = 128, 
            out_channels = 512, # 64
            dropout = 0.5, #0.25
            n_hidden_blocks = 2
        )

        self.projection_head = ProjectionHead_v2(
            in_features = 512, 
            projection_dim = 128
        )

        self.classifier = ClassifierHead_v2(
            in_features = 128, 
            num_classes = n_classes
        )

        self.normalize = normalize

    def forward(self, graph):
        x_1 = self.encoder(graph) # (batch_size, out_channels)

        x_1 = self.projection_head(x_1) # (batch_size, projection_dim)
        
        if self.normalize:
            x1_norm = F.normalize(x_1, p=2, dim=1) # (batch_size, projection_dim)

        c1 = self.classifier(x1_norm) # (batch_size, num_classes)
        
        return x1_norm, c1
   