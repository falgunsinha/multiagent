"""
RL Environment for Object Selection with RRT Path Planning - Visualization Version
Uses PythonRobotics RRT for fast visualization (not Isaac Sim native RRT).

This is separate from object_selection_env_rrt.py which uses Isaac Sim native RRT for training.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from .object_selection_env import ObjectSelectionEnv
from .path_estimators import RRTPathEstimator


class ObjectSelectionEnvRRTViz(ObjectSelectionEnv):
    """
    Environment variant that uses PythonRobotics RRT for visualization.
    
    This is for visualization only - training uses ObjectSelectionEnvRRT with Isaac Sim native RRT.
    """
    
    def __init__(
        self,
        franka_controller=None,
        max_objects: int = 10,
        max_steps: int = 50,
        num_cubes: int = 4,
        render_mode: Optional[str] = None,
        dynamic_obstacles: bool = True,
        training_grid_size: int = 6,
        execute_picks: bool = False
    ):
        # Use RRT Viz spacing (0.20/0.22)
        rrt_viz_spacing = 0.20 if training_grid_size > 3 else 0.22

        super().__init__(
            franka_controller=franka_controller,
            max_objects=max_objects,
            max_steps=max_steps,
            num_cubes=num_cubes,
            render_mode=render_mode,
            dynamic_obstacles=dynamic_obstacles,
            training_grid_size=training_grid_size,
            cube_spacing=rrt_viz_spacing  # Override with RRT Viz spacing
        )

        self.execute_picks = execute_picks

        # Initialize PythonRobotics RRT path estimator (for visualization)
        self.rrt_estimator = RRTPathEstimator(
            grid_size=training_grid_size,
            cell_size=rrt_viz_spacing,  # Match cube spacing
            franka_controller=franka_controller
        )

        # RRT performance tracking
        self.rrt_planning_times = []
        self.rrt_path_lengths = []
        self.rrt_success_count = 0
        self.rrt_failure_count = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment and update RRT occupancy grid"""
        obs, info = super().reset(seed=seed, options=options)

        # Update RRT occupancy grid with current object positions
        self._update_rrt_grid()

        return obs, info

    def _update_rrt_grid(self, target_cube_idx: Optional[int] = None):
        """
        Update RRT occupancy grid based on current object positions and obstacles.

        Args:
            target_cube_idx: Index of target cube to exclude from obstacles (for path planning to that cube)
                           If None, all cubes are treated as obstacles (for general grid update)
        """
        # Get obstacle positions from multiple sources
        obstacle_positions = []

        # 1. Lidar detected obstacles (if available)
        if self.franka_controller and hasattr(self.franka_controller, 'lidar_detected_obstacles'):
            for obs_name, obs_data in self.franka_controller.lidar_detected_obstacles.items():
                obs_pos = obs_data.get('position', None)
                if obs_pos is not None:
                    obstacle_positions.append(obs_pos)

        # 2. Random obstacles (for standalone training)
        if hasattr(self, 'random_obstacle_positions') and self.random_obstacle_positions:
            obstacle_positions.extend(self.random_obstacle_positions)

        # 3. Unpicked cubes as obstacles (EXCLUDING target cube and already picked cubes)
        unpicked_cube_positions = []
        for i in range(self.total_objects):
            if i not in self.objects_picked and i != target_cube_idx:
                unpicked_cube_positions.append(self.object_positions[i])

        # Update grid with unpicked cubes (excluding target) and static obstacles
        self.rrt_estimator.update_occupancy_grid(
            object_positions=unpicked_cube_positions,
            obstacle_positions=obstacle_positions
        )

    def _is_reachable(self, obj_idx: int) -> bool:
        """
        FULL RRT reachability check for action masking (overrides base class).

        Uses actual PythonRobotics RRT path planning to ensure agent only sees truly reachable cubes.
        This prevents the agent from learning to select blocked cubes.

        IMPORTANT: This is called during action masking for EVERY unpicked cube (~9 calls per step).
        Training will be slower but more accurate.

        NOTE: During testing with pretrained agents, use skip_reachability_check=True in action_masks()
        to avoid expensive RRT planning.

        Args:
            obj_idx: Index of object to check

        Returns:
            True if RRT can find a path to the object, False otherwise
        """
        if obj_idx in self.objects_picked:
            return False  # Already picked

        # Update RRT grid to exclude target cube from obstacles
        self._update_rrt_grid(target_cube_idx=obj_idx)

        # Use FULL PythonRobotics RRT path planning to check reachability
        return self.rrt_estimator.check_reachability(
            self.ee_position,
            self.object_positions[obj_idx]
        )

    def _calculate_reachability(self, obj_idx: int, dist_to_ee: float) -> float:
        """
        Calculate reachability for OBSERVATION (called for ALL objects).

        CRITICAL: This is called during _get_observation() for ALL objects (not just unpicked).
        Using full RRT here would mean 9 cubes × 10,000 iterations = 90,000 iterations per observation!

        Use FAST distance-based check instead. Full RRT is only used in reward calculation.

        Args:
            obj_idx: Index of object to check
            dist_to_ee: Euclidean distance from EE to object

        Returns:
            1.0 if within reachable distance, 0.0 if too far/close
        """
        # FAST: Distance-based check (no RRT planning)
        # This matches the fast check in _is_reachable()
        return 1.0 if (0.3 <= dist_to_ee <= 0.9) else 0.0

    def _calculate_reward(self, action: int) -> float:
        """
        Calculate reward using RRT path estimation.

        UPDATED: Now uses all 6 observation parameters for reward calculation:
        1. RRT path length (10 points max) - prioritizes nearest objects
        2. Distance to container (3 points max)
        3. Obstacle proximity (7 points max) - includes unpicked cubes
        4. Reachability flag (-10 penalty if unreachable)
        5. Path clearance (4 points max)
        6. Picked flag (handled via invalid action penalty)

        Additional penalties:
        - Path planning failure: -10 if RRT fails (returns 2.0×Euclidean)
        - Time penalty: -2 per step

        Bonuses:
        - First pick bonus: +5 if optimal first pick (shortest RRT path)
        - Completion bonus: +20 + time bonus (handled in step())
        """
        reward = 0.0
        obj_position = self.object_positions[action]

        # Base reward for successful pick
        reward += 10.0

        # IMPORTANT: Update RRT grid to exclude target cube from obstacles
        # This allows RRT to find a path TO the target cube
        self._update_rrt_grid(target_cube_idx=action)

        # 1. RRT path length reward (max 10 points) - INCREASED for nearest-first priority
        rrt_path_length = self.rrt_estimator.estimate_path_length(self.ee_position, obj_position)
        euclidean_distance = np.linalg.norm(obj_position[:2] - self.ee_position[:2])

        # Check if RRT planning failed (returns 2.0 × Euclidean as penalty)
        if rrt_path_length >= 2.0 * euclidean_distance:
            reward -= 10.0  # INCREASED path planning failure penalty

        # Normalize by typical path length (0.3m to 0.9m, same as A*)
        normalized_path_length = (rrt_path_length - 0.3) / 0.6
        normalized_path_length = np.clip(normalized_path_length, 0.0, 1.0)
        path_reward = 10.0 * (1.0 - normalized_path_length)  # INCREASED from 5.0
        reward += path_reward

        # 2. Distance to container reward (max 3 points)
        dist_to_container = np.linalg.norm(obj_position - self.container_position)
        container_reward = 3.0 * np.exp(-dist_to_container)
        reward += container_reward

        # 3. Obstacle proximity reward (max 7 points) - INCREASED
        # Now includes unpicked cubes as obstacles
        obstacle_score = self._calculate_obstacle_score_with_unpicked_cubes(obj_position, action)
        obstacle_reward = 7.0 * (1.0 - obstacle_score)  # INCREASED from 3.0
        reward += obstacle_reward

        # 4. Reachability penalty (-10 if unreachable) - INCREASED
        distance = np.linalg.norm(obj_position - self.ee_position)
        if not (0.3 <= distance <= 0.9):
            reward -= 10.0  # INCREASED from -5.0

        # 5. Path clearance reward (max 4 points) - INCREASED
        clearance_score = self._calculate_path_clearance(self.ee_position, obj_position)
        clearance_reward = 4.0 * clearance_score  # INCREASED from 2.0
        reward += clearance_reward

        # 6. Additional penalties for risky picks (close to masking thresholds)
        # These guide agent to prefer safer picks among valid actions
        if clearance_score < 0.30:  # Close to Layer 2 threshold (0.25)
            reward -= 5.0  # Risky pick - narrow path

        if obstacle_score > 0.60:  # Close to Layer 3 threshold (0.65)
            reward -= 5.0  # Risky pick - crowded area

        # Time penalty (encourage speed) - INCREASED
        reward -= 2.0  # INCREASED from -1.0

        # Sequential picking bonus: reward for picking closest object first
        # OPTIMIZED: Use Euclidean distance instead of RRT for first pick bonus
        # (calculating RRT for all 9 cubes would add 90,000 iterations to first step!)
        if len(self.objects_picked) == 0:
            # First pick: bonus if it's the closest object (by Euclidean distance)
            distances = [np.linalg.norm(pos[:2] - self.ee_position[:2]) for pos in self.object_positions]
            if action == np.argmin(distances):
                reward += 5.0

        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action and update RRT grid if dynamic obstacles enabled.
        """
        # Execute parent step
        obs, reward, terminated, truncated, info = super().step(action)

        # Update RRT grid if dynamic obstacles enabled
        if self.dynamic_obstacles and not terminated:
            self._update_rrt_grid()

        # Add RRT path length to info for logging
        # Note: Grid was already updated in _calculate_reward() with target excluded
        if action < self.total_objects and action not in self.objects_picked[:-1]:
            obj_position = self.object_positions[action]
            # Grid already has target excluded from _calculate_reward(), so this is correct
            rrt_length = self.rrt_estimator.estimate_path_length(self.ee_position, obj_position)
            info["rrt_path_length"] = rrt_length

        return obs, reward, terminated, truncated, info

