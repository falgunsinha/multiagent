"""
MASAC Continuous Action Wrapper for Cube Reshuffling

Maps continuous MASAC actions [-1, 1] to discrete reshuffling actions:
- Continuous action[0]: Cube selection (mapped to cube_idx)
- Continuous action[1]: Grid X position (mapped to grid cell X)
- Continuous action[2]: Grid Y position (mapped to grid cell Y)
"""

import numpy as np
from typing import Tuple, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.masac_reshuffling_agent import MASACReshufflingAgent
from agents.dimension_adapter import DimensionAdapter


class MASACContinuousWrapper:
    """
    Wrapper that adapts MASAC (continuous actions) to reshuffling task (discrete grid).
    
    MASAC outputs continuous actions in [-1, 1]^3:
    - action[0]: Cube selection (continuous) -> mapped to cube_idx
    - action[1]: Grid X position (continuous) -> mapped to grid_x
    - action[2]: Grid Y position (continuous) -> mapped to grid_y
    """
    
    def __init__(
        self,
        state_dim: int,
        grid_size: int,
        num_cubes: int,
        grid_center: Tuple[float, float] = (0.45, -0.10),
        cube_spacing: float = 0.13,
        cube_height: float = 0.02575,
        pretrained_model_path: str = None,
        use_dimension_adapter: bool = True,
        **masac_kwargs
    ):
        """
        Initialize MASAC wrapper.

        Args:
            state_dim: Dimension of state observation from cube environment
            grid_size: Size of grid (e.g., 4 for 4x4)
            num_cubes: Number of cubes
            grid_center: Center of grid in world coordinates
            cube_spacing: Spacing between grid cells
            cube_height: Height to place cubes
            pretrained_model_path: Path to pretrained MASAC models (Tennis: 24 state, 2 action)
            use_dimension_adapter: If True, use dimension adapter for pretrained models
            **masac_kwargs: Additional arguments for MASAC agent
        """
        self.grid_size = grid_size
        self.num_cubes = num_cubes
        self.grid_center = np.array(grid_center)
        self.cube_spacing = cube_spacing
        self.cube_height = cube_height
        self.source_state_dim = state_dim

        # Calculate grid bounds
        grid_extent = (grid_size - 1) * cube_spacing
        self.grid_start_x = grid_center[0] - (grid_extent / 2.0)
        self.grid_start_y = grid_center[1] - (grid_extent / 2.0)

        # Dimension adapter for pretrained Tennis models
        self.dimension_adapter: Optional[DimensionAdapter] = None
        if use_dimension_adapter and pretrained_model_path is not None:
            self.dimension_adapter = DimensionAdapter(
                source_state_dim=state_dim,
                target_state_dim=24,  # Tennis model expects 24
                target_action_dim=2,  # Tennis model outputs 2
                required_action_dim=3,  # We need 3 for reshuffling
                use_pca=True
            )
            # Use Tennis model dimensions
            agent_state_dim = 24
            agent_action_dim = 2
        else:
            # Use actual dimensions
            agent_state_dim = state_dim
            agent_action_dim = 3

        # MASAC agent with continuous action space
        self.agent = MASACReshufflingAgent(
            state_dim=agent_state_dim,
            action_dim=agent_action_dim,
            pretrained_model_path=pretrained_model_path,
            **masac_kwargs
        )
    
    def select_action(self, state: np.ndarray, deterministic: bool = False, valid_cubes: list = None) -> dict:
        """
        Select reshuffling action with optional action masking.

        Args:
            state: Current state observation
            deterministic: If True, use deterministic policy
            valid_cubes: List of valid cube indices (for action masking). If None, all cubes are valid.

        Returns:
            Dictionary with:
                - cube_idx: Which cube to move
                - target_grid_x: Target grid cell X
                - target_grid_y: Target grid cell Y
                - target_world_pos: Target world position (3D)
                - continuous_action: Raw continuous action from MASAC
            Or None if no valid cubes available
        """
        # Check if there are any valid cubes
        if valid_cubes is not None and len(valid_cubes) == 0:
            return None  # No valid cubes to reshuffle

        # Adapt state if using dimension adapter
        if self.dimension_adapter is not None:
            adapted_state = self.dimension_adapter.adapt_state(state)
        else:
            adapted_state = state

        # Get continuous action from MASAC
        continuous_action = self.agent.select_action(adapted_state, deterministic)

        # Adapt action if using dimension adapter
        if self.dimension_adapter is not None:
            continuous_action = self.dimension_adapter.adapt_action(continuous_action)

        # Map continuous actions to discrete choices
        # action[0] in [-1, 1] -> cube_idx in [0, num_cubes-1]
        cube_idx = int((continuous_action[0] + 1) / 2 * self.num_cubes)
        cube_idx = np.clip(cube_idx, 0, self.num_cubes - 1)

        # ACTION MASKING: Enforce valid cubes
        if valid_cubes is not None:
            if cube_idx not in valid_cubes:
                # MASAC selected an invalid cube, replace with closest valid cube
                cube_idx = min(valid_cubes, key=lambda x: abs(x - cube_idx))

        # action[1] in [-1, 1] -> grid_x in [0, grid_size-1]
        grid_x = int((continuous_action[1] + 1) / 2 * self.grid_size)
        grid_x = np.clip(grid_x, 0, self.grid_size - 1)

        # action[2] in [-1, 1] -> grid_y in [0, grid_size-1]
        grid_y = int((continuous_action[2] + 1) / 2 * self.grid_size)
        grid_y = np.clip(grid_y, 0, self.grid_size - 1)

        # Convert grid coordinates to world position
        target_world_pos = self._grid_to_world(grid_x, grid_y)

        return {
            'cube_idx': cube_idx,
            'target_grid_x': grid_x,
            'target_grid_y': grid_y,
            'target_world_pos': target_world_pos,
            'continuous_action': continuous_action
        }
    
    def _grid_to_world(self, grid_x: int, grid_y: int) -> np.ndarray:
        """
        Convert grid coordinates to world position.
        
        Args:
            grid_x: Grid cell X coordinate [0, grid_size-1]
            grid_y: Grid cell Y coordinate [0, grid_size-1]
        
        Returns:
            World position (x, y, z)
        """
        world_x = self.grid_start_x + grid_x * self.cube_spacing
        world_y = self.grid_start_y + grid_y * self.cube_spacing
        world_z = self.cube_height
        
        return np.array([world_x, world_y, world_z], dtype=np.float32)
    
    def store_transition(self, state, action_dict, reward, next_state, done):
        """
        Store transition in replay buffer.
        
        Args:
            state: Current state
            action_dict: Action dictionary from select_action()
            reward: Reward received
            next_state: Next state
            done: Done flag
        """
        # Store the continuous action (not the discrete mapping)
        continuous_action = action_dict['continuous_action']
        self.agent.store_transition(state, continuous_action, reward, next_state, done)
    
    def update_model(self):
        """Update MASAC networks"""
        return self.agent.update_model()
    
    def set_test_mode(self, is_test: bool):
        """Set test mode"""
        self.agent.set_test_mode(is_test)

    def fit_dimension_adapter(self, env, n_samples: int = 1000):
        """
        Fit PCA dimension adapter on sample states from environment.

        Args:
            env: Environment to collect states from
            n_samples: Number of samples to collect
        """
        if self.dimension_adapter is None:
            return

        states = self.dimension_adapter.collect_sample_states(env, n_samples)
        self.dimension_adapter.fit_pca(states)
    
    def save(self, save_path: str):
        """Save model"""
        self.agent.save(save_path)

