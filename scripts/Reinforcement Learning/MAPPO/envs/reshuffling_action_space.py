"""
Reshuffling Action Space

Defines the action space for Agent 2 (MAPPO Reshuffler):
- Which cube to move
- Where to move it (target grid cell)
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class ReshuffleAction:
    """Represents a reshuffling action"""
    cube_idx: int  # Which cube to move
    target_grid_x: int  # Target grid cell X
    target_grid_y: int  # Target grid cell Y
    target_world_pos: np.ndarray  # Target world position (3D)


class ReshufflingActionSpace:
    """
    Action space for reshuffling agent.
    
    Action encoding:
    - Discrete action space: [cube_idx * grid_size^2 + grid_y * grid_size + grid_x]
    - Total actions: num_cubes * grid_size * grid_size
    
    Example for 4x4 grid, 9 cubes:
    - Total actions: 9 * 4 * 4 = 144
    - Action 0: Move cube 0 to grid cell (0, 0)
    - Action 1: Move cube 0 to grid cell (0, 1)
    - ...
    - Action 15: Move cube 0 to grid cell (3, 3)
    - Action 16: Move cube 1 to grid cell (0, 0)
    - ...
    """
    
    def __init__(
        self,
        grid_size: int,
        num_cubes: int,
        grid_center: Tuple[float, float] = (0.45, -0.10),
        cube_spacing: float = 0.13,
        cube_height: float = 0.02575,  # Half of cube size (0.0515)
    ):
        """
        Initialize reshuffling action space.
        
        Args:
            grid_size: Size of grid (e.g., 4 for 4x4)
            num_cubes: Number of cubes
            grid_center: Center of grid in world coordinates (x, y)
            cube_spacing: Spacing between grid cells
            cube_height: Height to place cubes (z coordinate)
        """
        self.grid_size = grid_size
        self.num_cubes = num_cubes
        self.grid_center = np.array(grid_center)
        self.cube_spacing = cube_spacing
        self.cube_height = cube_height
        
        # Total action space size
        self.action_dim = num_cubes * grid_size * grid_size
        
        # Calculate grid bounds
        grid_extent = (grid_size - 1) * cube_spacing
        self.grid_start_x = grid_center[0] - (grid_extent / 2.0)
        self.grid_start_y = grid_center[1] - (grid_extent / 2.0)
    
    def decode_action(self, action: int) -> ReshuffleAction:
        """
        Decode discrete action into reshuffling action.
        
        Args:
            action: Discrete action index [0, action_dim)
            
        Returns:
            ReshuffleAction with cube index and target position
        """
        # Decode action
        cube_idx = action // (self.grid_size * self.grid_size)
        remaining = action % (self.grid_size * self.grid_size)
        grid_y = remaining // self.grid_size
        grid_x = remaining % self.grid_size
        
        # Convert grid coordinates to world coordinates
        world_x = self.grid_start_x + (grid_x * self.cube_spacing)
        world_y = self.grid_start_y + (grid_y * self.cube_spacing)
        world_z = self.cube_height
        
        target_world_pos = np.array([world_x, world_y, world_z])
        
        return ReshuffleAction(
            cube_idx=cube_idx,
            target_grid_x=grid_x,
            target_grid_y=grid_y,
            target_world_pos=target_world_pos
        )
    
    def encode_action(self, cube_idx: int, grid_x: int, grid_y: int) -> int:
        """
        Encode reshuffling action into discrete action.
        
        Args:
            cube_idx: Index of cube to move
            grid_x: Target grid cell X
            grid_y: Target grid cell Y
            
        Returns:
            Discrete action index
        """
        return cube_idx * (self.grid_size * self.grid_size) + grid_y * self.grid_size + grid_x
    
    def get_action_mask(
        self,
        cube_positions: np.ndarray,
        picked_cubes: List[int],
        occupied_cells: Optional[np.ndarray] = None,
        obstacle_positions: Optional[np.ndarray] = None,
        robot_position: Optional[np.ndarray] = None,
        base_env = None  # Base environment for reachability checking
    ) -> np.ndarray:
        """
        Get action mask for valid reshuffling actions.

        Args:
            cube_positions: Current cube positions (N, 3)
            picked_cubes: List of already picked cube indices
            occupied_cells: Boolean array of occupied grid cells (grid_size, grid_size)
            obstacle_positions: Array of obstacle positions (M, 3) - ADDED
            robot_position: Robot end-effector position (3,) - ADDED
            base_env: Base environment for reachability checking - ADDED

        Returns:
            Boolean mask of valid actions (action_dim,)
        """
        mask = np.zeros(self.action_dim, dtype=bool)

        # If no occupied cells provided, calculate from cube positions AND obstacles
        if occupied_cells is None:
            occupied_cells = self._get_occupied_cells(
                cube_positions,
                obstacle_positions=obstacle_positions
            )

        for action in range(self.action_dim):
            reshuffle_action = self.decode_action(action)

            # Check if cube is valid (not already picked)
            if reshuffle_action.cube_idx in picked_cubes:
                continue

            # Check if target cell is not occupied (by cubes OR obstacles)
            if occupied_cells[reshuffle_action.target_grid_y, reshuffle_action.target_grid_x]:
                continue

            # Check if not moving cube to its current position
            current_pos = cube_positions[reshuffle_action.cube_idx]
            current_grid_x, current_grid_y = self._world_to_grid(current_pos[:2])
            if current_grid_x == reshuffle_action.target_grid_x and current_grid_y == reshuffle_action.target_grid_y:
                continue

            # CRITICAL FIX: Check if target position is reachable by robot
            if robot_position is not None:
                target_pos = reshuffle_action.target_world_pos
                distance_to_robot = np.linalg.norm(target_pos - robot_position)

                # Check if within reachable distance range (0.3m to 0.9m)
                if distance_to_robot < 0.3 or distance_to_robot > 0.9:
                    continue  # Target position is unreachable

            # OPTIONAL: Advanced reachability check using A*/RRT (if base_env provided)
            # This ensures the target cell is actually reachable by path planning
            if base_env is not None and hasattr(base_env, 'astar_estimator'):
                # Use A* to check if robot can reach target position
                target_pos = reshuffle_action.target_world_pos
                is_reachable = base_env.astar_estimator.check_reachability(
                    robot_position,
                    target_pos
                )
                if not is_reachable:
                    continue  # Target position is blocked by obstacles
            elif base_env is not None and hasattr(base_env, 'rrt_estimator'):
                # Use RRT to check if robot can reach target position
                target_pos = reshuffle_action.target_world_pos
                try:
                    path_info = base_env.rrt_estimator.estimate_path_cost(
                        robot_position,
                        target_pos
                    )
                    if not path_info.get('success', False):
                        continue  # RRT failed - target is unreachable
                except:
                    pass  # If RRT fails, allow the action (conservative)

            # Valid action
            mask[action] = True

        return mask
    
    def _world_to_grid(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        grid_x = int(round((world_pos[0] - self.grid_start_x) / self.cube_spacing))
        grid_y = int(round((world_pos[1] - self.grid_start_y) / self.cube_spacing))

        # Clamp to grid bounds
        grid_x = max(0, min(self.grid_size - 1, grid_x))
        grid_y = max(0, min(self.grid_size - 1, grid_y))

        return grid_x, grid_y

    def _grid_to_world(self, grid_x: int, grid_y: int) -> np.ndarray:
        """
        Convert grid coordinates to world coordinates (cell center).

        Args:
            grid_x: Grid X coordinate
            grid_y: Grid Y coordinate

        Returns:
            World position (2,) [x, y] at cell center
        """
        world_x = self.grid_start_x + (grid_x * self.cube_spacing)
        world_y = self.grid_start_y + (grid_y * self.cube_spacing)

        return np.array([world_x, world_y], dtype=np.float32)
    
    def _get_occupied_cells(
        self,
        cube_positions: np.ndarray,
        obstacle_positions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get boolean array of occupied grid cells.

        Args:
            cube_positions: Current cube positions (N, 3)
            obstacle_positions: Obstacle positions (M, 3) - ADDED

        Returns:
            Boolean array of occupied cells (grid_size, grid_size)
        """
        occupied = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # Mark cells with cubes as occupied
        for pos in cube_positions:
            grid_x, grid_y = self._world_to_grid(pos[:2])
            occupied[grid_y, grid_x] = True

        # CRITICAL FIX: Mark cells with obstacles as occupied
        if obstacle_positions is not None and len(obstacle_positions) > 0:
            for obs_pos in obstacle_positions:
                grid_x, grid_y = self._world_to_grid(obs_pos[:2])
                occupied[grid_y, grid_x] = True

        return occupied

