import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import ChebConv, graclus, global_mean_pool, max_pool
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.pool import TopKPooling


# ========================================
# ChebNet
# pretrained gcn => used for perceptual loss
# ========================================
class ChebNet(nn.Module):
    def __init__(self, num_nodes, num_features, device, nf=16):
        super(ChebNet, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.nf = nf
        self.device = device

        # Encoder
        self.conv1 = ChebConv(num_features, nf, 2)
        self.norm1 = GraphNorm(nf)

        self.pool1 = TopKPooling(nf, 0.5)

        self.conv2 = ChebConv(nf, nf, 2)
        self.norm2 = GraphNorm(nf)

        self.pool2 = TopKPooling(nf, 0.5)

        self.conv3 = ChebConv(nf, nf, 2)
        self.norm3 = GraphNorm(nf)

        self.linear = nn.Linear(nf, num_nodes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, batch):
        outs = []

        # Conv 1
        x = F.relu(self.norm1(self.conv1(x, edge_index)))
        outs.append(x)

        # Pool 1
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index=edge_index, batch=batch)
        x = F.dropout(x, p=0.2, training=self.training)

        # Conv 2
        x = F.relu(self.norm2(self.conv2(x, edge_index, batch=batch)))
        outs.append(x)

        # Pool 2
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index=edge_index, batch=batch)
        x = F.dropout(x, p=0.2, training=self.training)

        # Conv 3
        x = F.relu(self.norm3(self.conv3(x, edge_index, batch=batch)))
        outs.append(x)

        return outs
