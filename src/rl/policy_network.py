"""
Neural Network Policy for Object Selection
Custom policy network for Stable-Baselines3 PPO agent.

Architecture:
- Input: Object features (60 dims: 10 objects × 6 features)
- Feature extraction: MLP layers
- Attention mechanism: Learn to focus on relevant objects
- Output: Action probabilities (10 dims: which object to pick)

UPDATED: Changed from 10 features per object to 6 features per object
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np


class ObjectAttentionExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor with attention mechanism for object selection.
    
    Processes each object's features separately, then uses attention to
    aggregate information across all objects.
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Dynamically determine number of objects and features per object
        # Input shape: (batch, total_features)
        # Example: (batch, 60) for 10 objects × 6 features
        total_features = observation_space.shape[0]

        # Assume max_objects is determined by training_grid_size
        # For now, infer from observation space
        # Common configurations: 60 (10 objects × 6), 96 (16 objects × 6), 144 (24 objects × 6)
        self.features_per_object = 6  # UPDATED: Now 6 features per object
        self.num_objects = total_features // self.features_per_object

        print(f"[POLICY] Attention policy: {self.num_objects} objects × {self.features_per_object} features = {total_features} dims")
        
        # Per-object feature extraction
        self.object_encoder = nn.Sequential(
            nn.Linear(self.features_per_object, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Attention mechanism
        self.attention_query = nn.Linear(64, 32)
        self.attention_key = nn.Linear(64, 32)
        self.attention_value = nn.Linear(64, 32)
        
        # Final aggregation
        self.aggregator = nn.Sequential(
            nn.Linear(32 * self.num_objects, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention mechanism.

        Args:
            observations: (batch, total_features) tensor
                         Example: (batch, 60) for 10 objects × 6 features

        Returns:
            features: (batch, features_dim) tensor
        """
        batch_size = observations.shape[0]
        
        # Reshape to (batch, num_objects, features_per_object)
        obs_reshaped = observations.view(batch_size, self.num_objects, self.features_per_object)
        
        # Encode each object independently
        # (batch, num_objects, features_per_object) -> (batch, num_objects, 64)
        object_features = []
        for i in range(self.num_objects):
            obj_feat = self.object_encoder(obs_reshaped[:, i, :])
            object_features.append(obj_feat)
        
        object_features = torch.stack(object_features, dim=1)  # (batch, num_objects, 64)
        
        # Apply attention mechanism
        # Query, Key, Value: (batch, num_objects, 32)
        Q = self.attention_query(object_features)
        K = self.attention_key(object_features)
        V = self.attention_value(object_features)
        
        # Attention scores: (batch, num_objects, num_objects)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(32)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention: (batch, num_objects, 32)
        attended_features = torch.matmul(attention_weights, V)
        
        # Flatten and aggregate: (batch, num_objects * 32) -> (batch, features_dim)
        attended_flat = attended_features.view(batch_size, -1)
        features = self.aggregator(attended_flat)
        
        return features


class ObjectSelectionPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy for object selection.
    Uses attention-based feature extraction.
    """
    
    def __init__(self, *args, **kwargs):
        # Set custom feature extractor
        kwargs["features_extractor_class"] = ObjectAttentionExtractor
        kwargs["features_extractor_kwargs"] = {"features_dim": 256}
        
        # Network architecture for actor and critic
        kwargs["net_arch"] = [
            dict(pi=[256, 128], vf=[256, 128])
        ]
        
        super().__init__(*args, **kwargs)


class SimpleMLPExtractor(BaseFeaturesExtractor):
    """
    Simple MLP feature extractor (fallback if attention is too complex).
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(observations)


class SimpleMLPPolicy(ActorCriticPolicy):
    """
    Simple MLP policy with action masking support.
    Prevents selecting already-picked objects by masking invalid actions.
    """

    def __init__(self, *args, **kwargs):
        kwargs["features_extractor_class"] = SimpleMLPExtractor
        kwargs["features_extractor_kwargs"] = {"features_dim": 256}
        kwargs["net_arch"] = [dict(pi=[256, 128], vf=[256, 128])]

        super().__init__(*args, **kwargs)

    def forward(self, obs, deterministic=False, action_masks=None):
        """
        Forward pass with optional action masking.

        Args:
            obs: Observations
            deterministic: Whether to use deterministic actions
            action_masks: Boolean array where True = valid action, False = invalid

        Returns:
            actions, values, log_probs
        """
        # Get features from observation
        features = self.extract_features(obs)

        # Get latent representations
        latent_pi = self.mlp_extractor.forward_actor(features)
        latent_vf = self.mlp_extractor.forward_critic(features)

        # Get action distribution
        distribution = self._get_action_dist_from_latent(latent_pi)

        # Apply action masking if provided
        if action_masks is not None:
            # Convert to tensor if needed
            if isinstance(action_masks, np.ndarray):
                action_masks = torch.as_tensor(action_masks).to(obs.device)

            # Mask invalid actions by setting their logits to -inf
            # This ensures they have 0 probability
            if hasattr(distribution, 'logits'):
                # For Categorical distribution
                masked_logits = distribution.logits.clone()
                masked_logits[~action_masks] = -1e8  # Very large negative number
                distribution = Categorical(logits=masked_logits)

        # Sample actions
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        # Get values
        values = self.value_net(latent_vf)

        return actions, values, log_prob

