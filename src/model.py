import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv

class GraphTransformerAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.encoder = TransformerConv(in_channels, hidden_channels, heads=2)
        self.decoder = nn.Linear(hidden_channels*2, in_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if edge_index is None:
            # Use dummy edge_index for isolated nodes
            edge_index = torch.empty((2, 0), dtype=torch.long)
        z = self.encoder(x, edge_index)
        x_hat = self.decoder(z)
        return x_hat, z

    def reconstruction_error(self, x, x_hat):
        return ((x - x_hat)**2).mean(dim=1)
