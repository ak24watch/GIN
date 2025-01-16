import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class GIN(nn.Module):
    def __init__(self, in_dim, out_dim, init_eps=0, num_mlps=1):
        """
        Initialize the GIN layer.
        
        Args:
            in_dim (int): Input feature dimension.
            out_dim (int): Output feature dimension.
            init_eps (float): Initial epsilon value.
            num_mlps (int): Number of MLP layers.
        """
        super().__init__()
        self.mlp_layers = num_mlps
        self.mlp_f = nn.ModuleList()
        self.mlp_phy = nn.ModuleList()
        for i in range(num_mlps):
            if i == 0:
                self.mlp_f.append(nn.Linear(in_dim, out_dim))
            else:
                self.mlp_f.append(nn.Linear(out_dim, out_dim))
        for i in range(num_mlps):
            self.mlp_phy.append(nn.Linear(out_dim, out_dim))
        self.eps = nn.Parameter(torch.FloatTensor([init_eps]))

    def forward(self, graph, feats):
        """
        Forward pass of the GIN layer.
        
        Args:
            graph (DGLGraph): Input graph.
            feats (Tensor): Input features.
        
        Returns:
            Tensor: Output features.
        """
        with graph.local_scope():
            for i in range(self.mlp_layers):
                if i == 0:
                    h = F.relu(self.mlp_f[i](feats))
                else:
                    h = F.relu(self.mlp_f[i](h))
            graph.ndata["h"] = h
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "n"))
            h = 1 + self.eps * graph.ndata["h"] + graph.ndata["n"]
            for i in range(self.mlp_layers):
                h = F.relu(self.mlp_phy[i](h))
            return h
