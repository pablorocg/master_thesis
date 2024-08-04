import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from torch.nn import ModuleList
import torch.nn.functional as F



#================================================ENCODER=====================================================
# class GraphConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, dropout=0.0):
#         super(GraphConvBlock, self).__init__()
#         self.conv = GCNConv(in_channels, out_channels)
#         self.bn = BatchNorm(out_channels)
#         self.relu = nn.LeakyReLU()
#         self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

#     def forward(self, x, edge_index):
#         x = self.conv(x, edge_index)
#         x = self.relu(x)
#         x = self.bn(x)
#         if self.dropout:
#             x = self.dropout(x)
#         return x
    
# class GCNEncoder(nn.Module):
#     def __init__(self, in_channels, hidden_dim, out_channels, dropout, n_hidden_blocks):
#         super(GCNEncoder, self).__init__()
#         self.input_block = GraphConvBlock(in_channels, hidden_dim, dropout)
#         self.hidden_blocks = ModuleList([GraphConvBlock(hidden_dim, hidden_dim, dropout) for _ in range(n_hidden_blocks - 1)])
#         self.output_block = GraphConvBlock(hidden_dim, out_channels, dropout)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         x = self.input_block(x, edge_index)
#         for layer in self.hidden_blocks:
#             x = layer(x, edge_index)
#         x = self.output_block(x, edge_index)
#         return global_mean_pool(x, batch) # (batch_size, out_channels)
    

# Graph Transformer

# Architecture:
# - GCN layer
# - GCN layer
# - Transformer layer
# - Projection head


import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import BatchNorm
from torch_geometric.data import Data, Batch

class GraphTransformerEncoder(nn.Module):
    """
    Graph Transformer model for obtaining graph embeddings.
    This model processes graphs with a variable number of nodes but a fixed number of features per node.
    Nodes are connected by bidirectional edges in a sequential manner.
    Example of a graph: n1 <--> n2 <--> n3 <--> n4 <--> n5
    """
    def __init__(self, in_channels=3, hidden_channels=256, out_channels=512, 
                 num_heads=8, dropout=0.1):
        super(GraphTransformerEncoder, self).__init__()
        
        # Graph Convolutional layers for node embedding
        self.gcn_1 = GCNConv(in_channels, hidden_channels)
        self.gcn_2 = GCNConv(hidden_channels, hidden_channels)
        
        # Batch normalization layer
        self.bn = BatchNorm(hidden_channels)
        
        # Multi-head attention layer
        self.mha = nn.MultiheadAttention(embed_dim=hidden_channels, 
                                         num_heads=num_heads, 
                                         dropout=dropout,
                                         )
        
        # Final projection layer
        self.projection = nn.Linear(hidden_channels, out_channels)
        
        # Activation functions
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Node embedding using GCN layers
        x = self.gcn_1(x, edge_index) # [num_nodes, hidden_channels]
        x = self.gelu(x)
        x = self.bn(x)
        x = self.dropout(x)

        x = self.gcn_2(x, edge_index) # [num_nodes, hidden_channels]
        x = self.gelu(x)
        x = self.bn(x)
        x = self.dropout(x) 
        
        # Reshape for multi-head attention
        x = x.unsqueeze(1) # [num_nodes, 1, hidden_channels]

        # Apply multi-head attention
        x, nodes_weight = self.mha(x, x, x) # [num_nodes, 1, hidden_channels]
        x = x.squeeze(1)  # [num_nodes, hidden_channels]

        # Mostrar la suma de todos los elementos de los pesos de los nodos
        print("Suma de los pesos de los nodos: ", torch.sum(nodes_weight))
        # Calcular la media ponderada de los nodos con respecto a los pesos de los nodos obtenidos en la capa de atención
        x = torch.matmul(nodes_weight.squeeze(1).t(), x) # [1, hidden_channels]


        # Global pooling to get graph-level representation
        # x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # Project to output dimension
        x = self.projection(x)
        return x

if __name__ == "__main__":

    # Example usage
    model = GraphTransformerEncoder(in_channels=3, hidden_channels=256, out_channels=512, num_heads=8, dropout=0.1)

    nodes = torch.randn(100, 3)  # 10 nodes, 3 features
    edges = torch.tensor(
            [[i, i+1] for i in range(nodes.size(0)-1)] + [[i+1, i] for i in range(nodes.size(0)-1)], 
            dtype=torch.long
        ).t().contiguous()
    graph = Data(x=nodes, edge_index=edges)

    # Create batch tensor
    graph_batch = Batch.from_data_list([graph])

    # Forward pass
    output = model(graph_batch)
    print(output.size())  # torch.Size([1, 512])



    

        



