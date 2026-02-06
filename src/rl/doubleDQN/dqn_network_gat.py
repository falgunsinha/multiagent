import torch
import torch.nn as nn
import numpy as np
import sys
import os

marl_src_path = os.path.join(os.path.dirname(__file__), '../../../scripts/Reinforcement Learning/MARL/src')

if marl_src_path not in sys.path:
    sys.path.insert(0, marl_src_path)

from gat_cvd.gat_encoder import SharedGATEncoder


class DQNNetworkGAT(nn.Module):
    """Deep Q-Network with GAT encoder for object selection"""

    def __init__(self, node_dim=7, output_dim=10, hidden_dim=128, gat_output_dim=128,
                 num_layers=2, num_heads=4, edge_dim=3):
        super(DQNNetworkGAT, self).__init__()

        self.node_dim = node_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.gat_output_dim = gat_output_dim

        self.gat_encoder = SharedGATEncoder(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            output_dim=gat_output_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            edge_dim=edge_dim
        )

        self.q_head = nn.Sequential(
            nn.Linear(gat_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.object_projection = nn.Linear(gat_output_dim, output_dim)

    def forward(self, graph):
        """Forward pass: compute Q-values for each object"""
        z = self.gat_encoder(graph.x, graph.edge_index, graph.edge_attr)

        object_mask = graph.x[:, 6] >= 0
        z_objects = z[object_mask]

        if z_objects.size(0) > 0:
            q_values_objects = self.q_head(z_objects).squeeze(-1)
            if z_objects.size(0) < self.output_dim:
                padding = torch.full((self.output_dim - z_objects.size(0),),
                                    -float('inf'), device=z.device)
                q_values = torch.cat([q_values_objects, padding], dim=0)
            else:
                q_values = q_values_objects[:self.output_dim]
        else:
            q_values = torch.full((self.output_dim,), -float('inf'), device=z.device)

        return q_values.unsqueeze(0)

    def get_action(self, graph, epsilon=0.0, action_mask=None):
        """Select action using epsilon-greedy policy"""

        if np.random.random() < epsilon:
            if action_mask is not None:
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) > 0:
                    return np.random.choice(valid_actions)

            return np.random.randint(0, self.output_dim)
        else:

            with torch.no_grad():
                q_values = self.forward(graph).squeeze(0).cpu().numpy()

                if action_mask is not None:
                    q_values = np.where(action_mask, q_values, -np.inf)

                return np.argmax(q_values)

    def get_graph_embedding(self, graph):
        """Get graph-level embedding from GAT encoder"""
        z = self.gat_encoder(graph.x, graph.edge_index, graph.edge_attr)
        z_graph = z.mean(dim=0, keepdim=True)
        return z_graph


class DQNNetworkGATWrapper:
    """Wrapper to make DQNNetworkGAT compatible with existing DDQN agent"""

    def __init__(self, network, graph_builder):
        """Initialize wrapper"""
        self.network = network
        self.graph_builder = graph_builder

    def forward(self, state):
        """Convert state to graph and forward through network"""
        graph = self.graph_builder(state)
        return self.network.forward(graph)

    def get_action(self, state, epsilon=0.0, action_mask=None):
        """Convert state to graph and get action"""
        graph = self.graph_builder(state)
        return self.network.get_action(graph, epsilon, action_mask)

    def parameters(self):
        """Return network parameters"""
        return self.network.parameters()

    def state_dict(self):
        """Return network state dict"""
        return self.network.state_dict()

    def load_state_dict(self, state_dict):
        """Load network state dict"""
        return self.network.load_state_dict(state_dict)

