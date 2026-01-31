"""
Heuristic Baseline Agents for Two-Agent System
Provides simple greedy/nearest-neighbor heuristics for both agents.
"""

import numpy as np
from typing import Tuple, List, Set


class HeuristicAgent1:
    """
    Heuristic Agent 1: Pick Sequence Selection
    Uses greedy nearest-neighbor strategy (picks closest unpicked cube)
    """
    
    def __init__(self, state_dim: int, action_dim: int, env=None):
        """
        Initialize heuristic agent for pick sequence

        Args:
            state_dim: State dimension (for compatibility)
            action_dim: Action dimension (number of cubes)
            env: Environment reference (for getting cube/robot positions)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = 0.0  # No exploration for heuristic
        self.env = env  # Store environment reference

    def select_action(self, state: np.ndarray, action_mask=None) -> int:
        """
        Select next cube to pick using greedy nearest-neighbor heuristic
        NOTE: Ignores action masking - pure heuristic baseline

        Args:
            state: Current state observation
            action_mask: Boolean mask for valid actions (ignored for pure heuristic)

        Returns:
            action: Index of cube to pick next
        """
        if self.env is None:
            # Fallback: random action if no environment provided
            if action_mask is not None:
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) > 0:
                    return np.random.choice(valid_actions)
            return np.random.randint(0, self.action_dim)

        # Get robot position (end effector)
        robot_pos = self.env.get_robot_position()

        # Get cube positions
        cube_positions = self.env.get_cube_positions()

        # Get already picked cubes
        picked_cubes = set(self.env.objects_picked) if hasattr(self.env, 'objects_picked') else set()

        # Find closest unpicked cube (IGNORES action masking - pure heuristic)
        min_distance = float('inf')
        best_action = 0

        for i in range(len(cube_positions)):
            if i in picked_cubes:
                continue  # Skip already picked cubes

            # Calculate Euclidean distance
            distance = np.linalg.norm(cube_positions[i][:2] - robot_pos[:2])

            if distance < min_distance:
                min_distance = distance
                best_action = i

        return best_action
    
    def load(self, path: str):
        """Dummy load method for compatibility"""
        pass


class HeuristicAgent2:
    """
    Heuristic Agent 2: Reshuffling
    Uses simple heuristic: always reshuffle to nearest empty grid position
    """
    
    def __init__(self, state_dim: int, action_dim: int = 3, grid_size: int = 4,
                 num_cubes: int = 9, cube_spacing: float = 0.13, env=None):
        """
        Initialize heuristic agent for reshuffling

        Args:
            state_dim: State dimension (for compatibility)
            action_dim: Action dimension (3 for continuous: cube_idx, grid_x, grid_y)
            grid_size: Grid size
            num_cubes: Number of cubes
            cube_spacing: Spacing between grid cells
            env: Environment reference (for getting cube/robot positions)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.grid_size = grid_size
        self.num_cubes = num_cubes
        self.cube_spacing = cube_spacing
        self.grid_center = np.array([0.45, -0.10])
        self.env = env  # Store environment reference

    def select_action(self, state: np.ndarray, deterministic: bool = False, valid_cubes=None) -> np.ndarray:
        """
        Select reshuffling action using greedy nearest-empty-spot heuristic
        NOTE: Ignores valid_cubes masking - pure heuristic baseline

        Args:
            state: Current state observation
            deterministic: For compatibility with MASAC (ignored for heuristic)
            valid_cubes: List of valid cube indices (ignored for pure heuristic)

        Returns:
            action: [cube_idx, grid_x, grid_y] in continuous space [-1, 1]
        """
        if self.env is None:
            # Fallback: random action
            return np.random.uniform(-1, 1, size=self.action_dim)

        # Get robot position
        robot_pos = self.env.get_robot_position()

        # Get cube positions
        cube_positions = self.env.get_cube_positions()

        # Get picked cubes
        picked_cubes = set(self.env.objects_picked) if hasattr(self.env, 'objects_picked') else set()

        # Find farthest unpicked cube (IGNORES valid_cubes masking - pure heuristic)
        max_distance = -1
        target_cube_idx = 0

        for i in range(len(cube_positions)):
            if i in picked_cubes:
                continue

            distance = np.linalg.norm(cube_positions[i][:2] - robot_pos[:2])
            if distance > max_distance:
                max_distance = distance
                target_cube_idx = i

        # Find nearest empty grid position to robot
        occupied_cells = self._get_occupied_cells(cube_positions)
        nearest_empty_cell = self._find_nearest_empty_cell(robot_pos, occupied_cells)

        # Convert to continuous action space [-1, 1]
        # Cube index: map [0, num_cubes-1] to [-1, 1]
        cube_action = (target_cube_idx / (self.num_cubes - 1)) * 2 - 1

        # Grid position: map [0, grid_size-1] to [-1, 1]
        grid_x_action = (nearest_empty_cell[0] / (self.grid_size - 1)) * 2 - 1
        grid_y_action = (nearest_empty_cell[1] / (self.grid_size - 1)) * 2 - 1

        return np.array([cube_action, grid_x_action, grid_y_action], dtype=np.float32)
    
    def _get_occupied_cells(self, cube_positions: List[np.ndarray]) -> Set[Tuple[int, int]]:
        """Get set of occupied grid cells"""
        occupied = set()
        for pos in cube_positions:
            grid_x, grid_y = self._world_to_grid(pos[:2])
            occupied.add((grid_x, grid_y))
        return occupied
    
    def _find_nearest_empty_cell(self, robot_pos: np.ndarray, occupied_cells: Set[Tuple[int, int]]) -> Tuple[int, int]:
        """Find nearest empty grid cell to robot position"""
        min_distance = float('inf')
        best_cell = (0, 0)
        
        for grid_x in range(self.grid_size):
            for grid_y in range(self.grid_size):
                if (grid_x, grid_y) in occupied_cells:
                    continue
                
                # Convert grid to world coordinates
                world_pos = self._grid_to_world(grid_x, grid_y)
                distance = np.linalg.norm(world_pos - robot_pos[:2])
                
                if distance < min_distance:
                    min_distance = distance
                    best_cell = (grid_x, grid_y)
        
        return best_cell
    
    def _world_to_grid(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        grid_extent = (self.grid_size - 1) * self.cube_spacing
        start_x = self.grid_center[0] - (grid_extent / 2.0)
        start_y = self.grid_center[1] - (grid_extent / 2.0)
        
        grid_x = int(round((world_pos[0] - start_x) / self.cube_spacing))
        grid_y = int(round((world_pos[1] - start_y) / self.cube_spacing))
        
        grid_x = max(0, min(self.grid_size - 1, grid_x))
        grid_y = max(0, min(self.grid_size - 1, grid_y))
        
        return grid_x, grid_y
    
    def _grid_to_world(self, grid_x: int, grid_y: int) -> np.ndarray:
        """Convert grid coordinates to world coordinates"""
        grid_extent = (self.grid_size - 1) * self.cube_spacing
        start_x = self.grid_center[0] - (grid_extent / 2.0)
        start_y = self.grid_center[1] - (grid_extent / 2.0)
        
        world_x = start_x + (grid_x * self.cube_spacing)
        world_y = start_y + (grid_y * self.cube_spacing)
        
        return np.array([world_x, world_y])
    
    def load(self, path: str):
        """Dummy load method for compatibility"""
        pass
    
    def set_test_mode(self, test_mode: bool):
        """Dummy method for compatibility with MASAC wrapper"""
        pass
    
    def fit_dimension_adapter(self, env, n_samples: int = 500):
        """Dummy method for compatibility with MASAC wrapper"""
        pass

