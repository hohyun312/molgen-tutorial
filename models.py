from torch import nn
import torch.nn.functional as F

from torch_geometric import nn as gnn


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


class RGCN(nn.Module):
    def __init__(
        self,
        num_node_types,
        num_edge_types,
        emb_dim,
        num_layers=3,
        activation="leaky_relu",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.node2emb = nn.Embedding(num_node_types, emb_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = gnn.RGCNConv(emb_dim, emb_dim, num_edge_types, aggr="add")
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
