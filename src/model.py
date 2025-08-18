import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GATConv
from torch_geometric.utils import to_dense_batch

class GraphTransformerEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads=4, dropout=0.1):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_channels * num_heads)
        self.conv2 = TransformerConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_channels * num_heads)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(self.norm1(x))
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        return x
        
class GraphTransformerDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_channels, out_channels, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_channels * 2)
        self.norm1 = nn.LayerNorm(hidden_channels * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, z):
        z = self.fc1(z)
        z = F.relu(self.norm1(z))
        z = self.dropout(z)
        z = self.fc2(z)
        z = F.relu(self.norm2(z))
        z = self.fc3(z)
        return z

class GraphTransformerAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads=4, latent_dim=None, dropout=0.1):
        super().__init__()
        if latent_dim is None:
            latent_dim = hidden_channels * num_heads
            
        self.encoder = GraphTransformerEncoder(in_channels, hidden_channels, num_heads, dropout)
        self.latent_proj = nn.Linear(hidden_channels * num_heads, latent_dim)
        self.decoder = GraphTransformerDecoder(latent_dim, hidden_channels, in_channels, dropout)
        self.latent_dim = latent_dim
        
    def encode(self, x, edge_index):
        h = self.encoder(x, edge_index)
        z = self.latent_proj(h)
        return z
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if isinstance(x, torch.Tensor) == False:
            x = torch.tensor(x, dtype=torch.float32)
        if edge_index is None:
            # Use dummy edge_index for isolated nodes
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
        
        # Encode
        z = self.encode(x, edge_index)
        
        # Decode
        x_hat = self.decode(z)
        
        return x_hat, z
    
    def reconstruction_error(self, x, x_hat):
        """Calculate MSE reconstruction error per node"""
        return ((x - x_hat)**2).mean(dim=1)
    
    def anomaly_score(self, data):
        """Calculate anomaly score for input data"""
        self.eval()
        with torch.no_grad():
            x_hat, _ = self.forward(data)
            scores = self.reconstruction_error(data.x, x_hat)
        return scores

class GraphTransformerVAE(nn.Module):
    """Variational Autoencoder version for more robust representation learning"""
    
    def __init__(self, in_channels, hidden_channels, num_heads=4, latent_dim=None, dropout=0.1):
        super().__init__()
        if latent_dim is None:
            latent_dim = hidden_channels * num_heads
            
        self.encoder = GraphTransformerEncoder(in_channels, hidden_channels, num_heads, dropout)
        
        # Mean and log variance projections
        self.mu_proj = nn.Linear(hidden_channels * num_heads, latent_dim)
        self.logvar_proj = nn.Linear(hidden_channels * num_heads, latent_dim)
        
        self.decoder = GraphTransformerDecoder(latent_dim, hidden_channels, in_channels, dropout)
        self.latent_dim = latent_dim
        
    def encode(self, x, edge_index):
        h = self.encoder(x, edge_index)
        mu = self.mu_proj(h)
        logvar = self.logvar_proj(h)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if isinstance(x, torch.Tensor) == False:
            x = torch.tensor(x, dtype=torch.float32)
        if edge_index is None:
            # Use dummy edge_index for isolated nodes
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
        
        # Encode
        mu, logvar = self.encode(x, edge_index)
        
        # Sample latent vector
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_hat = self.decode(z)
        
        return x_hat, mu, logvar
    
    def reconstruction_error(self, x, x_hat):
        """Calculate MSE reconstruction error per node"""
        return ((x - x_hat)**2).mean(dim=1)
    
    def kl_divergence(self, mu, logvar):
        """KL divergence between the learned distribution and standard normal"""
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kld
    
    def loss_function(self, x, x_hat, mu, logvar, kl_weight=0.1):
        """Total loss combining reconstruction error and KL divergence"""
        recon_loss = self.reconstruction_error(x, x_hat).mean()
        kl_loss = self.kl_divergence(mu, logvar).mean()
        return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss
    
    def anomaly_score(self, data):
        """Calculate anomaly score for input data"""
        self.eval()
        with torch.no_grad():
            x_hat, mu, logvar = self.forward(data)
            recon_error = self.reconstruction_error(data.x, x_hat)
            kl_div = self.kl_divergence(mu, logvar)
            # Combined score: reconstruction error + weighted KL divergence
            scores = recon_error + 0.1 * kl_div
        return scores
