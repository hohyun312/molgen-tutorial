import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric import nn as gnn
import torch_geometric.transforms as T


class MLP(nn.Module):
    def __init__(self, dims, activation="leaky_relu"):
        assert len(dims) > 1
        super().__init__()
        self.dims = dims
        self.input_dim = dims[0]
        self.output_dim = dims[-1]

        self.layers = nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append(nn.Linear(dims[i - 1], dims[i]))

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

    def forward(self, input):
        output = input
        for layer in self.layers[:-1]:
            output = self.activation(layer(output))
        return self.layers[-1](output)
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        return self


class GPS(nn.Module):
    def __init__(
        self, env, emb_dim, pe_dim=8, heads=2, num_layers=3, activation="leaky_relu"
    ):
        super().__init__()
        self.env = env
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.transform = T.AddRandomWalkPE(walk_length=20, attr_name="pe")

        self.pe_lin = nn.Linear(20, pe_dim)
        self.pe_norm = nn.BatchNorm1d(20)

        self.node2emb = nn.Embedding(self.env.num_node_types, emb_dim - pe_dim)
        self.edge2emb = nn.Embedding(self.env.num_edge_types, emb_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            net = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim),
            )
            conv = gnn.GPSConv(emb_dim, gnn.GINEConv(net), heads=heads)
            self.layers.append(conv)

    def forward(self, g):
        self.transform(g)
        x_nt = self.node2emb(g.node_type)
        x_pe = self.pe_norm(g.pe)
        x = torch.cat([x_nt, self.pe_lin(x_pe)], 1)
        edge_attr = self.edge2emb(g.edge_type)

        for layer in self.layers:
            x = layer(x, g.edge_index, g.batch, edge_attr=edge_attr)

        return x


class RGCN(nn.Module):
    def __init__(self, env, emb_dim, num_layers=3, activation="leaky_relu"):
        super().__init__()
        self.env = env
        self.emb_dim = emb_dim
        self.num_layers = num_layers

        self.node2emb = nn.Embedding(self.env.num_node_types, emb_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = gnn.RGCNConv(emb_dim, emb_dim, self.env.num_edge_types, aggr="add")
            self.layers.append(conv)

        self.activation = getattr(F, activation)

    def forward(self, g):
        x = self.node2emb(g.node_type)
        for layer in self.layers:
            x = layer(x, g.edge_index, g.edge_type)
            x = self.activation(x)
        return x
    
    def reset_parameters(self):
        self.node2emb.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        return self
        
