"""
Observation Normalizer for MAPPO

Normalizes observations using running mean and standard deviation.
"""

import numpy as np
import torch


class ObservationNormalizer:
    """
    Normalizes observations using running mean and standard deviation.
    
    This helps prevent NaN values in the neural network by keeping
    observations in a reasonable range.
    """
    
    def __init__(self, obs_dim: int, clip_range: float = 10.0, epsilon: float = 1e-8, warmup_steps: int = 100):
        """
        Initialize observation normalizer.

        Args:
            obs_dim: Observation dimension
            clip_range: Clip normalized observations to [-clip_range, clip_range]
            epsilon: Small value to prevent division by zero
            warmup_steps: Number of observations to collect before normalizing
        """
        self.obs_dim = obs_dim
        self.clip_range = clip_range
        self.epsilon = epsilon
        self.warmup_steps = warmup_steps

        # Running statistics
        self.mean = np.zeros(obs_dim, dtype=np.float32)
        self.var = np.ones(obs_dim, dtype=np.float32)
        self.count = 0
    
    def update(self, obs: np.ndarray):
        """
        Update running statistics with new observation.
        
        Args:
            obs: Observation array (obs_dim,) or (batch_size, obs_dim)
        """
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        
        batch_mean = np.mean(obs, axis=0)
        batch_var = np.var(obs, axis=0)
        batch_count = obs.shape[0]
        
        # Update running statistics using Welford's online algorithm
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count
    
    def normalize(self, obs: np.ndarray, update_stats: bool = True) -> np.ndarray:
        """
        Normalize observation.

        Args:
            obs: Observation array (obs_dim,) or (batch_size, obs_dim)
            update_stats: Whether to update running statistics

        Returns:
            Normalized observation
        """
        if update_stats:
            self.update(obs)

        # During warmup, return observation as-is (no normalization)
        if self.count < self.warmup_steps:
            return obs.copy()

        # Normalize
        normalized = (obs - self.mean) / (np.sqrt(self.var) + self.epsilon)

        # Clip to prevent extreme values
        normalized = np.clip(normalized, -self.clip_range, self.clip_range)

        # Check for NaN/Inf and replace with zeros
        if np.isnan(normalized).any() or np.isinf(normalized).any():
            print(f"[WARNING] NaN/Inf detected in normalized observation! Replacing with zeros.")
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

        return normalized
    
    def normalize_torch(self, obs: torch.Tensor, update_stats: bool = False) -> torch.Tensor:
        """
        Normalize PyTorch tensor observation.

        Args:
            obs: Observation tensor (obs_dim,) or (batch_size, obs_dim)
            update_stats: Whether to update running statistics (not recommended for torch)

        Returns:
            Normalized observation tensor
        """
        # During warmup, return observation as-is (no normalization)
        if self.count < self.warmup_steps:
            return obs.clone()

        device = obs.device
        dtype = obs.dtype
        mean = torch.from_numpy(self.mean).to(device=device, dtype=dtype)
        std = torch.sqrt(torch.from_numpy(self.var).to(device=device, dtype=dtype)) + self.epsilon

        # Normalize
        normalized = (obs - mean) / std

        # Clip to prevent extreme values
        normalized = torch.clamp(normalized, -self.clip_range, self.clip_range)

        # Check for NaN/Inf and replace with zeros
        if torch.isnan(normalized).any() or torch.isinf(normalized).any():
            print(f"[WARNING] NaN/Inf detected in normalized torch observation! Replacing with zeros.")
            normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

        return normalized
    
    def save(self, path: str):
        """Save normalizer statistics"""
        np.savez(path, mean=self.mean, var=self.var, count=self.count)
    
    def load(self, path: str):
        """Load normalizer statistics"""
        data = np.load(path)
        self.mean = data['mean']
        self.var = data['var']
        self.count = data['count']

