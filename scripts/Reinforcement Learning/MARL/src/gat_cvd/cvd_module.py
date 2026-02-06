import torch
import torch.nn as nn
import torch.nn.functional as F


class CVDModule(nn.Module):
    """
    Counterfactual Value Decomposition Module
    """
    
    def __init__(self, input_dim, hidden_dim=64, n_agents=2):
        super(CVDModule, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents
        self.baseline_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.agent_value_nets = nn.ModuleList()
        for i in range(n_agents):
            agent_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.agent_value_nets.append(agent_net)
    
    def forward(self, z_graph, agent_actions=None):
        """
        Forward pass: compute baseline and individual values.
        """
        batch_size = z_graph.size(0)
        V_baseline = self.baseline_net(z_graph)  # [batch_size, 1]
        V_agents = []
        for i in range(self.n_agents):
            V_i = self.agent_value_nets[i](z_graph)  # [batch_size, 1]
            V_agents.append(V_i)
        
        V_agents = torch.cat(V_agents, dim=1)  # [batch_size, n_agents]
        
        # Total value: V_total = V_baseline + sum(V_i)
        V_total = V_baseline + V_agents.sum(dim=1, keepdim=True)
        
        return V_baseline, V_agents, V_total
    
    def compute_counterfactual_values(self, z_graph):
        """
        Compute counterfactual values for each agent.
        """
        V_baseline, V_agents, V_total = self.forward(z_graph)
        
        V_counterfactual = []
        
        for i in range(self.n_agents):
    
            V_without_i = V_baseline.clone()
            for j in range(self.n_agents):
                if j != i:
                    V_without_i = V_without_i + V_agents[:, j:j+1]
        
            V_i_cf = V_total - V_without_i
            V_counterfactual.append(V_i_cf)
        
        V_counterfactual = torch.cat(V_counterfactual, dim=1)  # [batch_size, n_agents]
        
        return V_counterfactual
    
    def compute_counterfactual_targets(self, rewards, next_z_graph, dones, gamma=0.99):
        """
        Compute counterfactual TD targets for training.
        """
        with torch.no_grad():
            V_next_cf = self.compute_counterfactual_values(next_z_graph)  # [batch_size, n_agents]
            if rewards.dim() == 1:
                rewards = rewards.unsqueeze(1)  # [batch_size, 1]
            if dones.dim() == 1:
                dones = dones.unsqueeze(1)  # [batch_size, 1]


            targets = rewards + gamma * V_next_cf * (1 - dones)

        return targets
    
    def compute_loss(self, z_graph, rewards, next_z_graph, dones, gamma=0.99):
        """
        Compute CVD loss for training.
        
        """
        V_cf = self.compute_counterfactual_values(z_graph)
        targets = self.compute_counterfactual_targets(rewards, next_z_graph, dones, gamma)
        loss = F.mse_loss(V_cf, targets)
        
        return loss

