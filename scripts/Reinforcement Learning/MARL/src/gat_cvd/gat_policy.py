import torch
import torch.nn as nn
import torch.nn.functional as F
from .gat_encoder import SharedGATEncoder


class GATPolicy(nn.Module):
    """
    GAT-based policy network for MASAC Agent 2 (spatial manipulation)
    """
    
    def __init__(self, node_dim, n_actions, hidden_dim=128, output_dim=128, 
                 num_layers=2, num_heads=4, edge_dim=3, n_agents=2):
        super(GATPolicy, self).__init__()
        
        self.node_dim = node_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_agents = n_agents
        self.gat_encoder = SharedGATEncoder(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            edge_dim=edge_dim
        )
        
        self.policy_heads = nn.ModuleList()
        for i in range(n_agents):
            policy_head = nn.Sequential(
                nn.Linear(output_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions)
            )
            self.policy_heads.append(policy_head)
    
    def forward(self, graph, agent_id=None):
        """
        Forward pass: compute action probabilities.
        
        """
        z = self.gat_encoder(graph.x, graph.edge_index, graph.edge_attr)
        z_graph = z.mean(dim=0, keepdim=True)  # [1, output_dim]
        
        if agent_id is not None:
            # Single agent
            logits = self.policy_heads[agent_id](z_graph)  # [1, n_actions]
            action_probs = F.softmax(logits, dim=-1)
        else:
            # All agents
            action_probs_list = []
            for i in range(self.n_agents):
                logits = self.policy_heads[i](z_graph)  # [1, n_actions]
                probs = F.softmax(logits, dim=-1)
                action_probs_list.append(probs)
            action_probs = torch.cat(action_probs_list, dim=0)  # [n_agents, n_actions]
        
        return action_probs
    
    def get_action(self, graph, agent_id, epsilon=0.0):
        """
        Sample action from policy
        """
        if torch.rand(1).item() < epsilon:
            # Random action
            action = torch.randint(0, self.n_actions, (1,)).item()
        else:
            # Sample from policy
            with torch.no_grad():
                action_probs = self.forward(graph, agent_id)
                action = torch.multinomial(action_probs.squeeze(0), 1).item()
        
        return action
    
    def get_log_prob(self, graph, actions):
        """
        Compute log probabilities of actions
        """
        action_probs = self.forward(graph)  # [n_agents, n_actions]
        
        # Get log probs of taken actions
        if actions.dim() == 1:
            # Single timestep
            log_probs = torch.log(action_probs[range(self.n_agents), actions] + 1e-8)
        else:
        
            batch_size = actions.size(0)
            log_probs = []
            for b in range(batch_size):
                log_prob = torch.log(action_probs[range(self.n_agents), actions[b]] + 1e-8)
                log_probs.append(log_prob)
            log_probs = torch.stack(log_probs, dim=0)
        
        return log_probs
    
    def get_entropy(self, graph):
        """
        Compute policy entropy for exploration.
        """
        action_probs = self.forward(graph)  # [n_agents, n_actions]
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1)
        return entropy
    
    def get_graph_embedding(self, graph):
        """
        Get graph embedding from GAT encoder
        """
        z = self.gat_encoder(graph.x, graph.edge_index, graph.edge_attr)
        z_graph = z.mean(dim=0, keepdim=True)
        return z_graph

