from GinLayer import GIN
import torch.nn as nn
from dgl.nn.pytorch.glob import SumPooling

class GINModel(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_mlp_layers=2,
        num_gin_layers=2,
        dropout=0.2
    ):
        """
        Initialize the GINModel.
        
        Args:
            in_dim (int): Input feature dimension.
            hidden_dim (int): Hidden feature dimension.
            out_dim (int): Output feature dimension.
            num_mlp_layers (int): Number of MLP layers.
            num_gin_layers (int): Number of GIN layers.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.num_gin_layers = num_gin_layers
        self.gin = nn.ModuleList()
        for i in range(num_gin_layers):
            if i == 0:
                self.gin.append(GIN(in_dim, hidden_dim, num_mlps=num_mlp_layers))
            else:
                self.gin.append(GIN(hidden_dim, hidden_dim, num_mlps=num_mlp_layers))
        self.mlp_pool_layer = nn.ModuleList()
        for i in range(num_gin_layers + 1):
            if i == 0:
                self.mlp_pool_layer.append(nn.Linear(in_dim, out_dim))
            else:
                self.mlp_pool_layer.append(nn.Linear(hidden_dim, out_dim))
        self.pool = SumPooling()
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, feats):
        """
        Forward pass of the GINModel.
        
        Args:
            graph (DGLGraph): Input graph.
            feats (Tensor): Input features.
        
        Returns:
            Tensor: Output scores.
        """
        pool_h = [feats]
        for i in range(self.num_gin_layers):
            if i == 0:
                h = self.gin[i](graph, feats)
                h = self.dropout(h)
                pool_h.append(h)
                x = h
            elif i == 1:
                h = self.gin[i](graph, h)
                h = self.dropout(h)
                pool_h.append(h)
            else:
                x = x + h
                h = self.gin[i](graph, x)
                h = self.dropout(h)
                pool_h.append(h)
        pool_score = 0
        for i, h in enumerate(pool_h):
            pool_score += self.mlp_pool_layer[i](self.pool(graph, h))
        return pool_score
