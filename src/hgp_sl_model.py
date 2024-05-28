import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv

from hgp_sl_layers import GCN, HGPSLPool


class HGPSLEncoder(torch.nn.Module):
    def __init__(self, 
                 num_features:int = 3, 
                 nhid:int = 128, 
                 emb_dim:int = 64, 
                 pooling_ratio:float = 0.5, 
                 dropout_ratio:float = 0.0, 
                 sample_neighbor:bool = True, 
                 sparse_attention:bool = True, 
                 structure_learning:bool = True, 
                 lamb:float = 1.0):
        super(HGPSLEncoder, self).__init__()
        self.num_features = num_features # number of features
        self.nhid = nhid # hidden dimension
        self.embedding_dim = emb_dim # embedding dimension
        self.pooling_ratio = pooling_ratio # pooling ratio
        self.dropout_ratio = dropout_ratio # dropout ratio
        self.sample = sample_neighbor # sample neighbor
        self.sparse = sparse_attention # sparse attention
        self.sl = structure_learning # structure learning
        self.lamb = lamb # lambda

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)

        self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin3 = torch.nn.Linear(self.nhid, self.embedding_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # x = F.log_softmax(self.lin3(x), dim=-1)
        x = self.lin3(x)
        # Normalize the output
        x = F.normalize(x, p=2, dim=-1)

        return x