"""
Replay Buffer for MAPPO

Stores trajectories and computes advantages using GAE (Generalized Advantage Estimation).
"""

import numpy as np
import torch
from typing import Dict, List, Tuple


class RolloutBuffer:
    """
    Rollout buffer for on-policy algorithms (PPO/MAPPO).
    
    Stores trajectories and computes advantages using GAE.
    """
    
    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cpu")
    ):
        """
        Initialize rollout buffer.
        
        Args:
            buffer_size: Maximum buffer size
            obs_dim: Observation dimension
            action_dim: Action dimension
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            device: Device to store tensors
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        
        # Storage
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.values = np.zeros((buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        self.action_masks = np.zeros((buffer_size, action_dim), dtype=np.float32)
        
        # Computed during finalize
        self.returns = np.zeros((buffer_size,), dtype=np.float32)
        self.advantages = np.zeros((buffer_size,), dtype=np.float32)
        
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = buffer_size
    
    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        action_mask: np.ndarray = None
    ):
        """
        Add transition to buffer.
        
        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            done: Whether episode terminated
            action_mask: Action mask
        """
        assert self.ptr < self.max_size, "Buffer overflow"
        
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        if action_mask is not None:
            self.action_masks[self.ptr] = action_mask
        
        self.ptr += 1
    
    def finish_path(self, last_value: float = 0.0):
        """
        Finish current trajectory and compute returns and advantages using GAE.
        
        Args:
            last_value: Value estimate for last state (bootstrap value)
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        
        # Compute GAE advantages
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        
        advantages = np.zeros_like(deltas)
        last_gae_lam = 0
        for t in reversed(range(len(deltas))):
            last_gae_lam = deltas[t] + self.gamma * self.gae_lambda * last_gae_lam
            advantages[t] = last_gae_lam
        
        # Compute returns
        returns = advantages + values[:-1]
        
        # Store
        self.advantages[path_slice] = advantages
        self.returns[path_slice] = returns
        
        self.path_start_idx = self.ptr
    
    def get(self) -> Dict[str, torch.Tensor]:
        """
        Get all data from buffer as PyTorch tensors.
        
        Returns:
            Dictionary of tensors
        """
        assert self.ptr == self.max_size, "Buffer not full"
        
        # Normalize advantages
        adv_mean = np.mean(self.advantages)
        adv_std = np.std(self.advantages)
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
        
        data = {
            'observations': torch.as_tensor(self.observations, dtype=torch.float32, device=self.device),
            'actions': torch.as_tensor(self.actions, dtype=torch.long, device=self.device),
            'values': torch.as_tensor(self.values, dtype=torch.float32, device=self.device),
            'log_probs': torch.as_tensor(self.log_probs, dtype=torch.float32, device=self.device),
            'advantages': torch.as_tensor(self.advantages, dtype=torch.float32, device=self.device),
            'returns': torch.as_tensor(self.returns, dtype=torch.float32, device=self.device),
            'action_masks': torch.as_tensor(self.action_masks, dtype=torch.float32, device=self.device),
        }
        
        return data
    
    def reset(self):
        """Reset buffer"""
        self.ptr = 0
        self.path_start_idx = 0

