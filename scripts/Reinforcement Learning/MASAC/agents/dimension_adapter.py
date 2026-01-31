"""
Dimension Adapter for MASAC Models

Maps between different state/action dimensions using PCA and projection.
Allows using pretrained Tennis MASAC models (state_dim=24, action_dim=2)
for cube reshuffling tasks with different dimensions.
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Optional
import torch


class DimensionAdapter:
    """
    Adapts state and action dimensions between environments.
    
    Tennis MASAC: state_dim=24, action_dim=2
    Cube Reshuffling: state_dim varies (54, 96, etc.), action_dim=3
    """
    
    def __init__(
        self,
        source_state_dim: int,  # Cube environment state dimension
        target_state_dim: int = 24,  # Tennis model state dimension
        target_action_dim: int = 2,  # Tennis model action dimension
        required_action_dim: int = 3,  # Cube reshuffling action dimension
        use_pca: bool = True
    ):
        """
        Initialize dimension adapter.
        
        Args:
            source_state_dim: State dimension from cube environment
            target_state_dim: State dimension expected by Tennis model (24)
            target_action_dim: Action dimension from Tennis model (2)
            required_action_dim: Action dimension needed for reshuffling (3)
            use_pca: If True, use PCA for state reduction; else use simple projection
        """
        self.source_state_dim = source_state_dim
        self.target_state_dim = target_state_dim
        self.target_action_dim = target_action_dim
        self.required_action_dim = required_action_dim
        self.use_pca = use_pca
        
        # PCA for state dimension reduction
        self.pca: Optional[PCA] = None
        self.pca_fitted = False
        
        # Simple projection matrix (fallback if PCA not fitted)
        if source_state_dim > target_state_dim:
            # Reduce dimensions: use first target_state_dim features
            self.projection_matrix = np.eye(target_state_dim, source_state_dim)
        else:
            # Expand dimensions: pad with zeros
            self.projection_matrix = np.zeros((target_state_dim, source_state_dim))
            self.projection_matrix[:source_state_dim, :source_state_dim] = np.eye(source_state_dim)
    
    def fit_pca(self, states: np.ndarray):
        """
        Fit PCA on sample states from the environment.
        
        Args:
            states: Array of shape (n_samples, source_state_dim)
        """
        if not self.use_pca:
            return
        
        if self.source_state_dim <= self.target_state_dim:
            # No need for PCA if source is smaller
            return
        
        self.pca = PCA(n_components=self.target_state_dim)
        self.pca.fit(states)
        self.pca_fitted = True
    
    def adapt_state(self, state: np.ndarray) -> np.ndarray:
        """
        Adapt state from source dimension to target dimension.
        
        Args:
            state: State from cube environment (source_state_dim,)
        
        Returns:
            Adapted state for Tennis model (target_state_dim,)
        """
        if state.shape[-1] != self.source_state_dim:
            raise ValueError(f"Expected state dim {self.source_state_dim}, got {state.shape[-1]}")
        
        # Use PCA if fitted and enabled
        if self.use_pca and self.pca_fitted and self.source_state_dim > self.target_state_dim:
            adapted = self.pca.transform(state.reshape(1, -1)).flatten()
        else:
            # Use simple projection
            adapted = self.projection_matrix @ state
        
        return adapted
    
    def adapt_action(self, action: np.ndarray) -> np.ndarray:
        """
        Adapt action from Tennis model (2D) to reshuffling task (3D).
        
        Args:
            action: Action from Tennis model (target_action_dim,) in [-1, 1]
        
        Returns:
            Adapted action for reshuffling (required_action_dim,) in [-1, 1]
        """
        if action.shape[-1] != self.target_action_dim:
            raise ValueError(f"Expected action dim {self.target_action_dim}, got {action.shape[-1]}")
        
        # Expand action from 2D to 3D
        # action[0] -> cube selection
        # action[1] -> grid position (we'll split into x and y)
        adapted = np.zeros(self.required_action_dim)
        adapted[0] = action[0]  # Cube selection
        
        # Split second action into grid_x and grid_y
        # Use a simple mapping: action[1] -> both grid_x and grid_y
        # Alternative: use action[1] for x, and derive y from x
        adapted[1] = action[1]  # Grid X
        adapted[2] = -action[1]  # Grid Y (inverted for diversity)
        
        return adapted
    
    def collect_sample_states(self, env, n_samples: int = 1000) -> np.ndarray:
        """
        Collect sample states from environment for PCA fitting.

        Args:
            env: Environment to collect states from (TwoAgentEnv or base env)
            n_samples: Number of samples to collect

        Returns:
            Array of states (n_samples, source_state_dim)
        """
        states = []

        # Check if this is TwoAgentEnv (has agent2_action_dim) or base env (has action_space)
        is_two_agent_env = hasattr(env, 'agent2_action_dim')

        # Collect states until we have enough samples
        # This is more efficient than resetting n_samples times
        while len(states) < n_samples:
            if is_two_agent_env:
                obs, _ = env.reset()  # TwoAgentEnv returns (obs, info)
            else:
                obs = env.reset()

            states.append(obs)

            # Collect some random steps (up to 10 per episode)
            for _ in range(10):
                if len(states) >= n_samples:
                    break

                if is_two_agent_env:
                    # TwoAgentEnv: sample random action for agent 2
                    action = np.random.randint(0, env.agent2_action_dim)
                    obs, _, done, truncated, _ = env.step(action)
                    if done or truncated:
                        break
                else:
                    # Base env: use action_space.sample()
                    action = env.action_space.sample()
                    obs, _, done, _, _ = env.step(action)
                    if done:
                        break

                states.append(obs)

        states = np.array(states[:n_samples])
        return states

