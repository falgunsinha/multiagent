import numpy as np
from typing import Dict, Optional, Tuple
from .object_selection_env import ObjectSelectionEnv
from .path_estimators import RRTPathEstimator


class ObjectSelectionEnvRRTViz(ObjectSelectionEnv):
    """Environment variant that uses PythonRobotics RRT for visualization"""

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
        rrt_viz_spacing = 0.20 if training_grid_size > 3 else 0.22

        super().__init__(
            franka_controller=franka_controller,
            max_objects=max_objects,
            max_steps=max_steps,
            num_cubes=num_cubes,
            render_mode=render_mode,
            dynamic_obstacles=dynamic_obstacles,
            training_grid_size=training_grid_size,
            cube_spacing=rrt_viz_spacing
        )

        self.execute_picks = execute_picks

        self.rrt_estimator = RRTPathEstimator(
            grid_size=training_grid_size,
            cell_size=rrt_viz_spacing,
            franka_controller=franka_controller
        )

        self.rrt_planning_times = []
        self.rrt_path_lengths = []
        self.rrt_success_count = 0
        self.rrt_failure_count = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment and update RRT occupancy grid"""
        obs, info = super().reset(seed=seed, options=options)

        self._update_rrt_grid()

        return obs, info

    def _update_rrt_grid(self, target_cube_idx: Optional[int] = None):
        """Update RRT occupancy grid based on current object positions and obstacles"""
        obstacle_positions = []

        if self.franka_controller and hasattr(self.franka_controller, 'lidar_detected_obstacles'):

            for _, obs_data in self.franka_controller.lidar_detected_obstacles.items():
                obs_pos = obs_data.get('position', None)

                if obs_pos is not None:
                    obstacle_positions.append(obs_pos)

        if hasattr(self, 'random_obstacle_positions') and self.random_obstacle_positions:
            obstacle_positions.extend(self.random_obstacle_positions)

        unpicked_cube_positions = []

        for i in range(self.total_objects):

            if i not in self.objects_picked and i != target_cube_idx:
                unpicked_cube_positions.append(self.object_positions[i])

        self.rrt_estimator.update_occupancy_grid(
            object_positions=unpicked_cube_positions,
            obstacle_positions=obstacle_positions
        )

    def _is_reachable(self, obj_idx: int) -> bool:
        """check for action masking (overrides base class)"""
        if obj_idx in self.objects_picked:
            return False

        self._update_rrt_grid(target_cube_idx=obj_idx)

        return self.rrt_estimator.check_reachability(
            self.ee_position,
            self.object_positions[obj_idx]
        )

    def _calculate_reachability(self, _: int, dist_to_ee: float) -> float:
        """Calculate reachability for observation using fast distance-based check"""
        return 1.0 if (0.3 <= dist_to_ee <= 0.9) else 0.0

    def _calculate_reward(self, action: int) -> float:
        """Calculate reward using RRT path estimation"""
        reward = 0.0
        obj_position = self.object_positions[action]

        reward += 10.0

        self._update_rrt_grid(target_cube_idx=action)

        rrt_path_length = self.rrt_estimator.estimate_path_length(self.ee_position, obj_position)
        euclidean_distance = np.linalg.norm(obj_position[:2] - self.ee_position[:2])

        if rrt_path_length >= 2.0 * euclidean_distance:
            reward -= 10.0

        normalized_path_length = (rrt_path_length - 0.3) / 0.6
        normalized_path_length = np.clip(normalized_path_length, 0.0, 1.0)
        path_reward = 10.0 * (1.0 - normalized_path_length)
        reward += path_reward

        dist_to_container = np.linalg.norm(obj_position - self.container_position)
        container_reward = 3.0 * np.exp(-dist_to_container)
        reward += container_reward

        obstacle_score = self._calculate_obstacle_score_with_unpicked_cubes(obj_position, action)
        obstacle_reward = 7.0 * (1.0 - obstacle_score)
        reward += obstacle_reward

        distance = np.linalg.norm(obj_position - self.ee_position)

        if not (0.3 <= distance <= 0.9):
            reward -= 10.0

        clearance_score = self._calculate_path_clearance(self.ee_position, obj_position)
        clearance_reward = 4.0 * clearance_score
        reward += clearance_reward

        if clearance_score < 0.30:
            reward -= 5.0

        if obstacle_score > 0.60:
            reward -= 5.0

        reward -= 2.0

        if len(self.objects_picked) == 0:
            distances = [np.linalg.norm(pos[:2] - self.ee_position[:2]) for pos in self.object_positions]

            if action == np.argmin(distances):
                reward += 5.0

        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and update RRT grid if dynamic obstacles enabled"""
        obs, reward, terminated, truncated, info = super().step(action)

        if self.dynamic_obstacles and not terminated:
            self._update_rrt_grid()

        if action < self.total_objects and action not in self.objects_picked[:-1]:
            obj_position = self.object_positions[action]
            rrt_length = self.rrt_estimator.estimate_path_length(self.ee_position, obj_position)
            info["rrt_path_length"] = rrt_length

        return obs, reward, terminated, truncated, info

