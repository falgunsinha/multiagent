import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from collections import deque
import random
from .gat_encoder import SharedGATEncoder
from .graph_utils import build_graph, compute_edge_features
from .cvd_module import CVDModule
from .gat_policy import GATPolicy
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.rl.doubleDQN.dqn_network_gat import DQNNetworkGAT


class GraphReplayBuffer:
    """
    Replay buffer for storing graph-based transitions.
    """

    def __init__(self, capacity=50000):
        """
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, graph, action_ddqn, action_masac, reward, next_graph, done,
             action_mask=None, next_action_mask=None):
        """
        Add transition to buffer
        """
        self.buffer.append({
            'graph': graph,
            'action_ddqn': action_ddqn,
            'action_masac': action_masac,
            'reward': reward,
            'next_graph': next_graph,
            'done': done,
            'action_mask': action_mask,
            'next_action_mask': next_action_mask
        })

    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        return {
            'graphs': [t['graph'] for t in batch],
            'actions_ddqn': torch.tensor([t['action_ddqn'] for t in batch], dtype=torch.long),
            'actions_masac': torch.tensor([t['action_masac'] for t in batch], dtype=torch.long),
            'rewards': torch.tensor([t['reward'] for t in batch], dtype=torch.float32),
            'next_graphs': [t['next_graph'] for t in batch],
            'dones': torch.tensor([t['done'] for t in batch], dtype=torch.float32),
            'action_masks': [t['action_mask'] for t in batch],
            'next_action_masks': [t['next_action_mask'] for t in batch]
        }

    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)


class GATCVDAgent:
    """
    Main GAT + CVD Agent for heterogeneous multi-agent RL.
    
    """
    
    def __init__(self, args, device='cpu'):
        self.args = args
        self.device = device
        self.n_agents = 2  # DDQN + MASAC

        # Hyperparameters
        self.node_dim = args.get('node_dim', 7)
        self.edge_dim = args.get('edge_dim', 3)
        self.hidden_dim = args.get('hidden_dim', 128)
        self.gat_output_dim = args.get('gat_output_dim', 128)
        self.num_layers = args.get('num_layers', 2)
        self.num_heads = args.get('num_heads', 4)
        self.n_actions_ddqn = args.get('n_actions_ddqn', 10)  # Max objects
        self.n_actions_masac = args.get('n_actions_masac', 5)  # Manipulation actions
        self.gamma = args.get('gamma', 0.99)
        self.tau = args.get('tau', 0.005)
        self.lr = args.get('lr', 3e-4)
        self.batch_size = args.get('batch_size', 32)
        self.buffer_size = args.get('replay_buffer_size', 50000)
        self._init_networks()
        self._init_optimizers()
        self.replay_buffer = GraphReplayBuffer(capacity=self.buffer_size)
        self.train_step = 0
        self.episodes = 0  # Track number of completed episodes (incremented at end of each episode)
        self.min_buffer_size = self.batch_size * 10  # Start training after 10 batches worth of data
    
    def _init_networks(self):
        """Initialize all networks."""
        
        self.shared_gat = SharedGATEncoder(
            node_dim=self.node_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.gat_output_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            edge_dim=self.edge_dim
        ).to(self.device)
        
        self.ddqn_policy = DQNNetworkGAT(
            node_dim=self.node_dim,
            output_dim=self.n_actions_ddqn,
            hidden_dim=self.hidden_dim,
            gat_output_dim=self.gat_output_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            edge_dim=self.edge_dim
        ).to(self.device)
        
        self.ddqn_target = DQNNetworkGAT(
            node_dim=self.node_dim,
            output_dim=self.n_actions_ddqn,
            hidden_dim=self.hidden_dim,
            gat_output_dim=self.gat_output_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            edge_dim=self.edge_dim
        ).to(self.device)
        
        self.ddqn_target.load_state_dict(self.ddqn_policy.state_dict())
        
        self.masac_policy = GATPolicy(
            node_dim=self.node_dim,
            n_actions=self.n_actions_masac,
            hidden_dim=self.hidden_dim,
            output_dim=self.gat_output_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            edge_dim=self.edge_dim,
            n_agents=1  # Single MASAC agent
        ).to(self.device)
        
        self.cvd_module = CVDModule(
            input_dim=self.gat_output_dim,
            hidden_dim=64,
            n_agents=self.n_agents
        ).to(self.device)
        
        self.critic_1 = nn.Sequential(
            nn.Linear(self.gat_output_dim + self.n_actions_masac, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        ).to(self.device)
        
        self.critic_2 = nn.Sequential(
            nn.Linear(self.gat_output_dim + self.n_actions_masac, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        ).to(self.device)
        
        self.critic_target_1 = nn.Sequential(
            nn.Linear(self.gat_output_dim + self.n_actions_masac, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        ).to(self.device)
        
        self.critic_target_2 = nn.Sequential(
            nn.Linear(self.gat_output_dim + self.n_actions_masac, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        ).to(self.device)
        
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        
        # 6. Entropy coefficient (auto-tuned)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
    
    def _init_optimizers(self):
        """Initialize optimizers."""
        self.ddqn_optimizer = torch.optim.Adam(self.ddqn_policy.parameters(), lr=self.lr)
        self.masac_optimizer = torch.optim.Adam(self.masac_policy.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), 
            lr=self.lr
        )
        self.cvd_optimizer = torch.optim.Adam(self.cvd_module.parameters(), lr=self.lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)

    def store_transition(self, graph, action_ddqn, action_masac, reward, next_graph, done,
                         action_mask=None, next_action_mask=None):
        """
        Store transition in replay buffer.
        """
        self.replay_buffer.push(
            graph, action_ddqn, action_masac, reward, next_graph, done,
            action_mask, next_action_mask
        )

    def can_train(self):
        """Check if buffer has enough samples to start training."""
        return len(self.replay_buffer) >= self.min_buffer_size

    def select_actions(self, graph, epsilon_ddqn=0.0, epsilon_masac=0.0, action_mask=None):
        """
        Select actions for both agents.
        """
        # DDQN action (Agent 1)
        action_ddqn = self.ddqn_policy.get_action(graph, epsilon_ddqn, action_mask)

        # MASAC action (Agent 2)
        action_masac = self.masac_policy.get_action(graph, agent_id=0, epsilon=epsilon_masac)

        return action_ddqn, action_masac

    def train_step_ddqn(self):
        """
        Train DDQN (Agent 1) with CVD.
        """
    
        if not self.can_train():
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        graphs = batch['graphs']
        actions = batch['actions_ddqn'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_graphs = batch['next_graphs']
        dones = batch['dones'].to(self.device)

        q_values = []
        for graph, action in zip(graphs, actions):
            q = self.ddqn_policy.forward(graph)
            q_values.append(q[0, action])
        q_values = torch.stack(q_values).unsqueeze(1)  # [batch_size, 1]

        # Compute target Q-values (Double DQN)
        with torch.no_grad():
            next_q_values = []
            for next_graph in next_graphs:
                # Select action with policy network
                q_policy = self.ddqn_policy.forward(next_graph)
                next_action = q_policy.argmax(dim=1).item()

                # Evaluate with target network
                q_target = self.ddqn_target.forward(next_graph)
                next_q_values.append(q_target[0, next_action])

            next_q_values = torch.stack(next_q_values).unsqueeze(1)  # [batch_size, 1]
            rewards_reshaped = rewards.unsqueeze(1)  # [batch_size, 1]
            dones_reshaped = dones.unsqueeze(1)  # [batch_size, 1]
            targets = rewards_reshaped + self.gamma * next_q_values * (1 - dones_reshaped)

        loss = F.mse_loss(q_values, targets)

        self.ddqn_optimizer.zero_grad()
        loss.backward()
        self.ddqn_optimizer.step()

        return loss.item()

    def train_step_masac(self):
        """
        Train MASAC (Agent 2) with CVD.
        """

        if not self.can_train():
            return None, None, None
        batch = self.replay_buffer.sample(self.batch_size)

        graphs = batch['graphs']
        actions = batch['actions_masac'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_graphs = batch['next_graphs']
        dones = batch['dones'].to(self.device)
        z_graphs = []
        for graph in graphs:
            z = self.masac_policy.get_graph_embedding(graph)
            z_graphs.append(z)
        z_graphs = torch.cat(z_graphs, dim=0)  # [batch_size, gat_output_dim]

        next_z_graphs = []
        for next_graph in next_graphs:
            z = self.masac_policy.get_graph_embedding(next_graph)
            next_z_graphs.append(z)
        next_z_graphs = torch.cat(next_z_graphs, dim=0)  # [batch_size, gat_output_dim]

        # Convert actions to one-hot
        actions_onehot = F.one_hot(actions.squeeze(), num_classes=self.n_actions_masac).float()

        with torch.no_grad():
            # Sample next actions from policy
            next_action_probs = []
            for next_graph in next_graphs:
                probs = self.masac_policy.forward(next_graph, agent_id=0)
                next_action_probs.append(probs)
            next_action_probs = torch.cat(next_action_probs, dim=0)  # [batch_size, n_actions]

            # Compute next Q-values (expectation over actions)
            next_q_1_all = []
            next_q_2_all = []
            for i, next_z in enumerate(next_z_graphs):
                q1_actions = []
                q2_actions = []
                for a in range(self.n_actions_masac):
                    action_vec = F.one_hot(torch.tensor([a]), num_classes=self.n_actions_masac).float().to(self.device)
                    q1 = self.critic_target_1(torch.cat([next_z.unsqueeze(0), action_vec], dim=1))
                    q2 = self.critic_target_2(torch.cat([next_z.unsqueeze(0), action_vec], dim=1))
                    q1_actions.append(q1)
                    q2_actions.append(q2)
                next_q_1_all.append(torch.cat(q1_actions, dim=1))
                next_q_2_all.append(torch.cat(q2_actions, dim=1))

            next_q_1_all = torch.cat(next_q_1_all, dim=0)  # [batch_size, n_actions]
            next_q_2_all = torch.cat(next_q_2_all, dim=0)  # [batch_size, n_actions]

            # Min of twin Q-values
            next_q = torch.min(next_q_1_all, next_q_2_all)

            # SAC target with entropy
            next_v = (next_action_probs * (next_q - self.alpha * torch.log(next_action_probs + 1e-8))).sum(dim=1, keepdim=True)
            # Reshape rewards and dones to [batch_size, 1] to match next_v
            rewards_reshaped = rewards.unsqueeze(1)  # [batch_size, 1]
            dones_reshaped = dones.unsqueeze(1)  # [batch_size, 1]
            targets = rewards_reshaped + self.gamma * next_v * (1 - dones_reshaped)

        q1 = self.critic_1(torch.cat([z_graphs, actions_onehot], dim=1))
        q2 = self.critic_2(torch.cat([z_graphs, actions_onehot], dim=1))

        # Critic loss
        critic_loss = F.mse_loss(q1, targets) + F.mse_loss(q2, targets)

        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update (policy gradient)
        z_graphs_actor = []
        for graph in graphs:
            z = self.masac_policy.get_graph_embedding(graph)
            z_graphs_actor.append(z)
        z_graphs_actor = torch.cat(z_graphs_actor, dim=0)  # [batch_size, gat_output_dim]

        action_probs = []
        for graph in graphs:
            probs = self.masac_policy.forward(graph, agent_id=0)
            action_probs.append(probs)
        action_probs = torch.cat(action_probs, dim=0)  # [batch_size, n_actions]

        q1_all = []
        for z in z_graphs_actor:
            q1_actions = []
            for a in range(self.n_actions_masac):
                action_vec = F.one_hot(torch.tensor([a]), num_classes=self.n_actions_masac).float().to(self.device)
                q1 = self.critic_1(torch.cat([z.unsqueeze(0), action_vec], dim=1))
                q1_actions.append(q1)
            q1_all.append(torch.cat(q1_actions, dim=1))
        q1_all = torch.cat(q1_all, dim=0)  # [batch_size, n_actions]


        actor_loss = (action_probs * (self.alpha * torch.log(action_probs + 1e-8) - q1_all)).sum(dim=1).mean()
        self.masac_optimizer.zero_grad()
        actor_loss.backward()
        self.masac_optimizer.step()
        alpha_loss = -(self.log_alpha * (torch.log(action_probs + 1e-8) + self.n_actions_masac).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

        return critic_loss.item(), actor_loss.item(), alpha_loss.item()

    def train_step_cvd(self):
        """
        Train CVD module for credit assignment
        """
        # Check if we have enough samples
        if not self.can_train():
            return None

        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)

        graphs = batch['graphs']
        rewards = batch['rewards'].to(self.device)
        next_graphs = batch['next_graphs']
        dones = batch['dones'].to(self.device)
        z_graphs = []
        for graph in graphs:
            z = self.shared_gat(graph.x, graph.edge_index, graph.edge_attr)
            z_graph = z.mean(dim=0, keepdim=True)
            z_graphs.append(z_graph)
        z_graphs = torch.cat(z_graphs, dim=0)  # [batch_size, gat_output_dim]

        next_z_graphs = []
        for next_graph in next_graphs:
            z = self.shared_gat(next_graph.x, next_graph.edge_index, next_graph.edge_attr)
            z_graph = z.mean(dim=0, keepdim=True)
            next_z_graphs.append(z_graph)
        next_z_graphs = torch.cat(next_z_graphs, dim=0)  # [batch_size, gat_output_dim]

        # Compute CVD loss
        loss = self.cvd_module.compute_loss(z_graphs, rewards, next_z_graphs, dones, self.gamma)

        # Update
        self.cvd_optimizer.zero_grad()
        loss.backward()
        self.cvd_optimizer.step()

        return loss.item()

    def soft_update_targets(self):
        """Soft update target networks."""
        for param, target_param in zip(self.ddqn_policy.parameters(), self.ddqn_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Critic targets
        for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path):
        """Save agent and replay buffer."""
        torch.save({
            'ddqn_policy': self.ddqn_policy.state_dict(),
            'masac_policy': self.masac_policy.state_dict(),
            'cvd_module': self.cvd_module.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'log_alpha': self.log_alpha,
            'replay_buffer': list(self.replay_buffer.buffer),  # Save replay buffer
            'train_step': self.train_step,
            'episodes': self.episodes,
        }, path)

    def load(self, path):
        """Load agent and replay buffer."""
        checkpoint = torch.load(path)
        self.ddqn_policy.load_state_dict(checkpoint['ddqn_policy'])
        self.masac_policy.load_state_dict(checkpoint['masac_policy'])
        self.cvd_module.load_state_dict(checkpoint['cvd_module'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp()
        if 'replay_buffer' in checkpoint:
            self.replay_buffer.buffer = deque(checkpoint['replay_buffer'], maxlen=self.buffer_size)
        if 'train_step' in checkpoint:
            self.train_step = checkpoint['train_step']
        if 'episodes' in checkpoint:
            self.episodes = checkpoint['episodes']

