"""
MAPPO Policy (Actor-Critic)

Implements actor and critic networks for MAPPO reshuffling agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Optional
import numpy as np
import sys
from pathlib import Path

# Add utils to path for observation normalizer
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from observation_normalizer import ObservationNormalizer


class Actor(nn.Module):
    """
    Actor network for MAPPO.

    Takes observation and outputs action probabilities.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize actor network.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
        """
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # Initialize weights with smaller values to prevent NaN
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights with very small gain to prevent NaN"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier uniform for more stable initialization
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Categorical:
        """
        Forward pass.
        
        Args:
            obs: Observation tensor (batch_size, obs_dim)
            action_mask: Action mask (batch_size, action_dim)
            
        Returns:
            Action distribution
        """
        # NEW: Check for NaN/Inf in input
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            print(f"[ERROR] NaN/Inf detected in observation!")
            print(f"  obs min: {obs.min().item()}, max: {obs.max().item()}")
            print(f"  NaN count: {torch.isnan(obs).sum().item()}")
            print(f"  Inf count: {torch.isinf(obs).sum().item()}")
            # Replace NaN/Inf with zeros
            obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)

        # NEW: Check for NaN/Inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"[ERROR] NaN/Inf detected in logits!")
            print(f"  logits min: {logits.min().item()}, max: {logits.max().item()}")
            print(f"  NaN count: {torch.isnan(logits).sum().item()}")
            print(f"  Inf count: {torch.isinf(logits).sum().item()}")
            # Replace NaN/Inf with large negative values
            logits = torch.nan_to_num(logits, nan=-1e8, posinf=1e8, neginf=-1e8)

        # Apply action mask if provided
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), float('-inf'))

        return Categorical(logits=logits)


class Critic(nn.Module):
    """
    Critic network for MAPPO.
    
    Takes observation and outputs value estimate.
    """
    
    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        """
        Initialize critic network.
        
        Args:
            obs_dim: Observation dimension
            hidden_dim: Hidden layer dimension
        """
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with very small gain to prevent NaN"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier uniform for more stable initialization
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: Observation tensor (batch_size, obs_dim)
            
        Returns:
            Value estimate (batch_size, 1)
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        
        return value


class MAPPOPolicy(nn.Module):
    """
    MAPPO Policy combining actor and critic.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        device: torch.device = torch.device("cpu"),
        use_obs_norm: bool = True
    ):
        """
        Initialize MAPPO policy.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            lr: Learning rate
            device: Device to run on
            use_obs_norm: Whether to use observation normalization
        """
        super(MAPPOPolicy, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.use_obs_norm = use_obs_norm

        # Create observation normalizer
        if use_obs_norm:
            self.obs_normalizer = ObservationNormalizer(obs_dim, clip_range=10.0)
        else:
            self.obs_normalizer = None

        # Create actor and critic
        self.actor = Actor(obs_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(obs_dim, hidden_dim).to(device)

        # Optimizer for both networks
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    
    def get_actions(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get actions from policy.

        Args:
            obs: Observation tensor
            action_mask: Action mask
            deterministic: Whether to use deterministic policy

        Returns:
            actions, action_log_probs, values
        """
        # Normalize observation if enabled
        if self.use_obs_norm and self.obs_normalizer is not None:
            obs = self.obs_normalizer.normalize_torch(obs, update_stats=False)

        # Get action distribution
        action_dist = self.actor(obs, action_mask)

        # Sample or take mode
        if deterministic:
            actions = action_dist.probs.argmax(dim=-1)
        else:
            actions = action_dist.sample()

        action_log_probs = action_dist.log_prob(actions)

        # Get value estimate
        values = self.critic(obs)

        return actions, action_log_probs, values
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions (for training).

        Args:
            obs: Observation tensor
            actions: Action tensor
            action_mask: Action mask

        Returns:
            values, action_log_probs, dist_entropy
        """
        # Normalize observation if enabled
        if self.use_obs_norm and self.obs_normalizer is not None:
            obs = self.obs_normalizer.normalize_torch(obs, update_stats=False)

        # Get action distribution
        action_dist = self.actor(obs, action_mask)

        # Get log probs and entropy
        action_log_probs = action_dist.log_prob(actions)
        dist_entropy = action_dist.entropy().mean()

        # Get value estimate
        values = self.critic(obs).squeeze(-1)

        return values, action_log_probs, dist_entropy

    def update_obs_normalizer(self, obs: np.ndarray):
        """Update observation normalizer statistics"""
        if self.use_obs_norm and self.obs_normalizer is not None:
            self.obs_normalizer.update(obs)

