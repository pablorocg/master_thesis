import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GCN
from torch_geometric.nn import global_add_pool

class GraphFiberNet(nn.Module):
    def __init__(self, encoder, hidden_channels, n_classes, full_trainable=False):
        super(GraphFiberNet, self).__init__()

        self.encoder = encoder
        self.projection_head = nn.Linear(hidden_channels, n_classes)
        self.full_trainable = full_trainable

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if not self.full_trainable:
            self.encoder.eval()
            with torch.no_grad():
                x = self.encoder(x, edge_index)

        else:
            x = self.encoder(x, edge_index)

        x = global_add_pool(x, batch)
        x = F.normalize(x, p=2, dim=-1)

        x = self.projection_head(x)
        x = F.normalize(x, p=2, dim=-1)

        return x # Vector of size [batch_size, projection_dim]
    
# class GraphFiberNet(nn.Module):
#     def __init__(self, hidden_channels, n_classes):
#         super(GraphFiberNet, self).__init__()

#         self.encoder = GCN(
#             in_channels = 3, 
#             hidden_channels = hidden_channels, 
#             out_channels = hidden_channels
#         )
#         self.projection_head = nn.Linear(hidden_channels, n_classes)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
        
#         x = self.encoder(x, edge_index) # [num_nodes, hidden_channels]
#         x = global_add_pool(x, batch) # [batch_size, hidden_channels]
#         x = F.normalize(x, p=2, dim=-1) # [batch_size, hidden_channels]

#         x = self.projection_head(x)
#         x = F.normalize(x, p=2, dim=-1)
        
#         return x # Vector of size [batch_size, projection_dim]