import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class GATLayer(MessagePassing):
    """
    Single Graph Attention Layer with multi-head attention.
    """
    
    def __init__(self, in_channels, out_channels, num_heads=4, dropout=0.0, 
                 concat=True, edge_dim=None):
        super(GATLayer, self).__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat = concat
        self.edge_dim = edge_dim
        
        # Linear transformations for query, key, value
        self.lin_query = nn.Linear(in_channels, num_heads * out_channels)
        self.lin_key = nn.Linear(in_channels, num_heads * out_channels)
        self.lin_value = nn.Linear(in_channels, num_heads * out_channels)
        
        # Edge feature transformation (if edges have features)
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, num_heads * out_channels)
        else:
            self.lin_edge = None
        
        self.att = nn.Parameter(torch.Tensor(1, num_heads, 2 * out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_query.weight)
        nn.init.xavier_uniform_(self.lin_key.weight)
        nn.init.xavier_uniform_(self.lin_value.weight)
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att)
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass of GAT layer.
        
        """
        # Linear transformations
        query = self.lin_query(x).view(-1, self.num_heads, self.out_channels)
        key = self.lin_key(x).view(-1, self.num_heads, self.out_channels)
        value = self.lin_value(x).view(-1, self.num_heads, self.out_channels)
        
     
        out = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr)
        if self.concat:
            out = out.view(-1, self.num_heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        return out
    
    def message(self, query_i, key_j, value_j, edge_index_i, edge_attr, size_i):
        """
        Compute attention-weighted messages.
        """
        alpha = (torch.cat([query_i, key_j], dim=-1) * self.att).sum(dim=-1)
        if edge_attr is not None and self.lin_edge is not None:
            edge_feat = self.lin_edge(edge_attr).view(-1, self.num_heads, self.out_channels)
            alpha = alpha + (edge_feat * key_j).sum(dim=-1)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Apply attention to values
        return value_j * alpha.unsqueeze(-1)


class SharedGATEncoder(nn.Module):
    """
    Shared GAT Encoder for both DDQN and MASAC agents.
    """
    
    def __init__(self, node_dim, hidden_dim=128, output_dim=128, num_layers=2, 
                 num_heads=4, edge_dim=None, dropout=0.1):
        super(SharedGATEncoder, self).__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                gat = GATLayer(hidden_dim, hidden_dim // num_heads, num_heads,
                              dropout, concat=True, edge_dim=edge_dim)
            elif i == num_layers - 1:
                gat = GATLayer(hidden_dim, output_dim, num_heads,
                              dropout, concat=False, edge_dim=edge_dim)
            else:
                gat = GATLayer(hidden_dim, hidden_dim // num_heads, num_heads,
                              dropout, concat=True, edge_dim=edge_dim)

            self.gat_layers.append(gat)
            self.layer_norms.append(nn.LayerNorm(hidden_dim if i < num_layers - 1 else output_dim))

        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass of GAT encoder.
        """
        h = self.input_proj(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            h_new = gat(h, edge_index, edge_attr)

            # Residual connection (if dimensions match)
            if h.size(-1) == h_new.size(-1):
                h_new = h_new + h

            # Layer normalization
            h = norm(h_new)

            # Activation (except last layer)
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        # Output projection
        z = self.output_proj(h)

        return z

    def get_graph_embedding(self, z, batch=None):
        """
        Get graph-level embedding by pooling node embeddings.
        """
        if batch is None:
            return z.mean(dim=0, keepdim=True)
        else:
            from torch_geometric.nn import global_mean_pool
            return global_mean_pool(z, batch)

