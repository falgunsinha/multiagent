"""
Counterfactual Value Decomposition (CVD) Module

Implements counterfactual credit assignment for heterogeneous multi-agent RL.
Decomposes team value into individual agent contributions using counterfactual reasoning.

V_total = V_baseline + V_1 + V_2
where V_i = V_total - V_{-i} - V_baseline (counterfactual contribution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CVDModule(nn.Module):
    """
    Counterfactual Value Decomposition Module.
    
    Decomposes team value into:
    - V_baseline: Value when no agent acts
    - V_1: Agent 1's counterfactual contribution (DDQN)
    - V_2: Agent 2's counterfactual contribution (MASAC)
    
    Args:
        input_dim: Input dimension (graph embedding size)
        hidden_dim: Hidden layer dimension
        n_agents: Number of agents (default: 2)
    """
    
    def __init__(self, input_dim, hidden_dim=64, n_agents=2):
        super(CVDModule, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents
        
        # Baseline value network (no agent acts)
        self.baseline_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Individual value networks for each agent
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
        
        Args:
            z_graph: Graph embedding [batch_size, input_dim]
            agent_actions: Agent actions (optional, for action-conditioned values)
        
        Returns:
            V_baseline: Baseline value [batch_size, 1]
            V_agents: Individual agent values [batch_size, n_agents]
            V_total: Total team value [batch_size, 1]
        """
        batch_size = z_graph.size(0)
        
        # Compute baseline value
        V_baseline = self.baseline_net(z_graph)  # [batch_size, 1]
        
        # Compute individual agent values
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
        
        V_i^cf = V_total - V_{-i}
        where V_{-i} is the value when agent i doesn't act
        
        Args:
            z_graph: Graph embedding [batch_size, input_dim]
        
        Returns:
            V_counterfactual: Counterfactual values [batch_size, n_agents]
        """
        V_baseline, V_agents, V_total = self.forward(z_graph)
        
        # Counterfactual value for agent i = V_total - V_{-i}
        # V_{-i} = V_baseline + sum_{j != i} V_j
        V_counterfactual = []
        
        for i in range(self.n_agents):
            # Value without agent i
            V_without_i = V_baseline.clone()
            for j in range(self.n_agents):
                if j != i:
                    V_without_i = V_without_i + V_agents[:, j:j+1]
            
            # Counterfactual contribution of agent i
            V_i_cf = V_total - V_without_i
            V_counterfactual.append(V_i_cf)
        
        V_counterfactual = torch.cat(V_counterfactual, dim=1)  # [batch_size, n_agents]
        
        return V_counterfactual
    
    def compute_counterfactual_targets(self, rewards, next_z_graph, dones, gamma=0.99):
        """
        Compute counterfactual TD targets for training.
        
        Target_i = r + γ * V_i^cf(s')
        
        Args:
            rewards: Team rewards [batch_size, 1]
            next_z_graph: Next state graph embeddings [batch_size, input_dim]
            dones: Done flags [batch_size, 1]
            gamma: Discount factor
        
        Returns:
            targets: Counterfactual targets [batch_size, n_agents]
        """
        with torch.no_grad():
            # Compute next counterfactual values
            V_next_cf = self.compute_counterfactual_values(next_z_graph)  # [batch_size, n_agents]

            # Reshape rewards and dones to [batch_size, 1] for broadcasting to [batch_size, n_agents]
            if rewards.dim() == 1:
                rewards = rewards.unsqueeze(1)  # [batch_size, 1]
            if dones.dim() == 1:
                dones = dones.unsqueeze(1)  # [batch_size, 1]

            # TD targets: r + γ * V_next (if not done)
            # rewards [batch_size, 1] broadcasts to [batch_size, n_agents]
            targets = rewards + gamma * V_next_cf * (1 - dones)

        return targets
    
    def compute_loss(self, z_graph, rewards, next_z_graph, dones, gamma=0.99):
        """
        Compute CVD loss for training.
        
        Loss = MSE(V_i^cf, Target_i) for all agents
        
        Args:
            z_graph: Current state graph embeddings [batch_size, input_dim]
            rewards: Team rewards [batch_size, 1]
            next_z_graph: Next state graph embeddings [batch_size, input_dim]
            dones: Done flags [batch_size, 1]
            gamma: Discount factor
        
        Returns:
            loss: CVD loss (scalar)
        """
        # Compute current counterfactual values
        V_cf = self.compute_counterfactual_values(z_graph)
        
        # Compute targets
        targets = self.compute_counterfactual_targets(rewards, next_z_graph, dones, gamma)
        
        # MSE loss
        loss = F.mse_loss(V_cf, targets)
        
        return loss

