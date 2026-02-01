"""
Graph Attention Network (GAT) Encoder

Implements multi-head graph attention for spatial object rearrangement.
Replaces RNN encoder with graph-based attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class GATLayer(MessagePassing):
    """
    Single Graph Attention Layer with multi-head attention.
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension per head
        num_heads: Number of attention heads
        dropout: Dropout rate
        concat: Whether to concatenate or average heads
        edge_dim: Edge feature dimension (optional)
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
        
        # Attention mechanism
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
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] (optional)
        
        Returns:
            out: Updated node features [num_nodes, num_heads * out_channels] or [num_nodes, out_channels]
        """
        # Linear transformations
        query = self.lin_query(x).view(-1, self.num_heads, self.out_channels)
        key = self.lin_key(x).view(-1, self.num_heads, self.out_channels)
        value = self.lin_value(x).view(-1, self.num_heads, self.out_channels)
        
        # Propagate messages
        out = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr)
        
        # Concatenate or average heads
        if self.concat:
            out = out.view(-1, self.num_heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        return out
    
    def message(self, query_i, key_j, value_j, edge_index_i, edge_attr, size_i):
        """
        Compute attention-weighted messages.
        
        Args:
            query_i: Query features of target nodes
            key_j: Key features of source nodes
            value_j: Value features of source nodes
            edge_index_i: Target node indices
            edge_attr: Edge features
            size_i: Number of target nodes
        
        Returns:
            Attention-weighted messages
        """
        # Compute attention scores
        alpha = (torch.cat([query_i, key_j], dim=-1) * self.att).sum(dim=-1)
        
        # Add edge features to attention (if available)
        if edge_attr is not None and self.lin_edge is not None:
            edge_feat = self.lin_edge(edge_attr).view(-1, self.num_heads, self.out_channels)
            alpha = alpha + (edge_feat * key_j).sum(dim=-1)
        
        # Softmax normalization
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Apply attention to values
        return value_j * alpha.unsqueeze(-1)


class SharedGATEncoder(nn.Module):
    """
    Shared GAT Encoder for both DDQN and MASAC agents.
    
    Multi-layer GAT with residual connections and layer normalization.
    
    Args:
        node_dim: Node feature dimension
        hidden_dim: Hidden dimension
        output_dim: Output embedding dimension
        num_layers: Number of GAT layers
        num_heads: Number of attention heads
        edge_dim: Edge feature dimension (optional)
        dropout: Dropout rate
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

        # Input projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                # First layer
                gat = GATLayer(hidden_dim, hidden_dim // num_heads, num_heads,
                              dropout, concat=True, edge_dim=edge_dim)
            elif i == num_layers - 1:
                # Last layer (average heads)
                gat = GATLayer(hidden_dim, output_dim, num_heads,
                              dropout, concat=False, edge_dim=edge_dim)
            else:
                # Middle layers
                gat = GATLayer(hidden_dim, hidden_dim // num_heads, num_heads,
                              dropout, concat=True, edge_dim=edge_dim)

            self.gat_layers.append(gat)
            self.layer_norms.append(nn.LayerNorm(hidden_dim if i < num_layers - 1 else output_dim))

        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass of GAT encoder.

        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] (optional)
            batch: Batch assignment for each node (optional)

        Returns:
            z: Node embeddings [num_nodes, output_dim]
        """
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # GAT layers with residual connections
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

        Args:
            z: Node embeddings [num_nodes, output_dim]
            batch: Batch assignment for each node (optional)

        Returns:
            graph_emb: Graph-level embedding [batch_size, output_dim] or [1, output_dim]
        """
        if batch is None:
            # Single graph: mean pooling
            return z.mean(dim=0, keepdim=True)
        else:
            # Multiple graphs: mean pooling per graph
            from torch_geometric.nn import global_mean_pool
            return global_mean_pool(z, batch)

