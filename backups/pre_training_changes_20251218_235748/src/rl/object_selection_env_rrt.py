"""
RL Environment for Object Selection with RRT Path Planning (Option 3)
Uses actual RRT path planning in Isaac Sim for training.
Requires Isaac Sim to be running during training.
"""

import numpy as np
import time
from typing import Dict, Optional, Tuple
from .object_selection_env import ObjectSelectionEnv
from .path_estimators_isaacsim import IsaacSimRRTPathEstimator


class ObjectSelectionEnvRRT(ObjectSelectionEnv):
    """
    Environment variant that uses actual RRT path planning for reward calculation.
    
    This requires Isaac Sim to be running and a Franka controller with RRT planner.
    Training will be slower but rewards will be based on actual RRT performance.
    """
    
    def __init__(
        self,
        franka_controller=None,  # Optional for visualization
        max_objects: int = 10,
        max_steps: int = 50,
        num_cubes: int = 4,
        render_mode: Optional[str] = None,
        dynamic_obstacles: bool = True,  # Usually True for RRT training
        training_grid_size: int = 6,
        execute_picks: bool = False,  # Whether to actually execute pick-and-place
        rrt_planner=None,  # Isaac Sim RRT planner instance
        kinematics_solver=None,  # Lula kinematics solver
        articulation_kinematics_solver=None,  # Articulation kinematics solver
        franka_articulation=None  # Franka robot articulation
    ):
        super().__init__(
            franka_controller=franka_controller,
            max_objects=max_objects,
            max_steps=max_steps,
            num_cubes=num_cubes,
            render_mode=render_mode,
            dynamic_obstacles=dynamic_obstacles,
            training_grid_size=training_grid_size
        )

        self.execute_picks = execute_picks

        # Initialize Isaac Sim RRT path estimator (for training)
        # UPDATED: Cell size accounts for gripper width (15.2cm) + safety margin
        self.rrt_estimator = IsaacSimRRTPathEstimator(
            grid_size=training_grid_size,
            cell_size=0.20 if training_grid_size > 3 else 0.22,  # 20cm for 4x4+, 22cm for 3x3
            rrt_planner=rrt_planner,
            kinematics_solver=kinematics_solver,
            articulation_kinematics_solver=articulation_kinematics_solver,
            franka_articulation=franka_articulation
        )

        # RRT performance tracking
        self.rrt_planning_times = []
        self.rrt_path_lengths = []
        self.rrt_success_count = 0
        self.rrt_failure_count = 0
        self.episode_count = 0
        self.stats_print_interval = 100  # Print stats every 100 episodes

    def set_rrt_components(self, rrt_planner, kinematics_solver,
                          articulation_kinematics_solver, franka_articulation):
        """Set all Isaac Sim RRT components"""
        self.rrt_estimator.set_rrt_planner(rrt_planner)
        self.rrt_estimator.set_kinematics_solver(kinematics_solver)
        self.rrt_estimator.set_articulation_kinematics_solver(articulation_kinematics_solver)
        self.rrt_estimator.set_franka_articulation(franka_articulation)

    def _update_rrt_grid(self, target_cube_idx: Optional[int] = None):
        """
        Update Isaac Sim RRT occupancy grid based on current object positions and obstacles.

        Args:
            target_cube_idx: Index of target cube to exclude from obstacles (for path planning to that cube)
                           If None, all cubes are treated as obstacles (for general grid update)
        """
        # Get obstacle positions from multiple sources
        obstacle_positions = []

        # 1. Lidar detected obstacles (if available - for RRT integration)
        if self.franka_controller and hasattr(self.franka_controller, 'lidar_detected_obstacles'):
            # Extract obstacle positions from Lidar detected obstacles
            for obs_name, obs_data in self.franka_controller.lidar_detected_obstacles.items():
                obs_pos = obs_data.get('position', None)
                if obs_pos is not None:
                    obstacle_positions.append(obs_pos)

        # 2. Random obstacles (for standalone training without Isaac Sim)
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
        FULL Isaac Sim RRT reachability check for action masking (overrides base class).

        Uses actual Isaac Sim RRT path planning to ensure agent only sees truly reachable cubes.
        This prevents the agent from learning to select blocked cubes.

        IMPORTANT: This is called during action masking for EVERY unpicked cube (~9 calls per step).
        Training will be slower but more accurate.

        Args:
            obj_idx: Index of object to check

        Returns:
            True if Isaac Sim RRT can find a path to the object, False otherwise
        """
        if obj_idx in self.objects_picked:
            return False  # Already picked

        # Update RRT grid to exclude target cube from obstacles
        self._update_rrt_grid(target_cube_idx=obj_idx)

        # Use FULL Isaac Sim RRT path planning to check reachability
        return self.rrt_estimator.check_reachability(
            self.ee_position,
            self.object_positions[obj_idx]
        )

    def _calculate_reachability(self, obj_idx: int, dist_to_ee: float) -> float:
        """
        Calculate reachability for OBSERVATION (called for ALL objects).

        CRITICAL: This is called during _get_observation() for ALL objects (not just unpicked).
        Using full Isaac Sim RRT here would mean ~9 cubes × RRT planning = significant slowdown.

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
        # (calculating RRT for all 9 cubes would add significant overhead to first step!)
        if len(self.objects_picked) == 0:
            # First pick: bonus if it's the closest object (by Euclidean distance)
            distances = [np.linalg.norm(pos[:2] - self.ee_position[:2]) for pos in self.object_positions]
            if action == np.argmin(distances):
                reward += 5.0

        return reward

    def _plan_rrt_path_to_object(self, obj_position: np.ndarray, obj_name: str) -> Dict:
        """
        Plan RRT path to object position.

        Args:
            obj_position: Target object position (x, y, z)
            obj_name: Name of the object

        Returns:
            Dictionary with:
                - success: bool
                - path_length: float (if success)
                - path: trajectory (if success)
                - reason: str (if failure)
        """
        if self.franka_controller is None:
            return {"success": False, "reason": "no_controller"}

        try:
            # Get current robot state
            current_joint_positions = self.franka_controller.franka.get_joint_positions()

            # Calculate target position (above object for picking)
            pick_height_offset = 0.15  # 15cm above object
            target_position = obj_position.copy()
            target_position[2] += pick_height_offset

            # Use Franka controller's RRT planner
            if hasattr(self.franka_controller, 'rrt') and self.franka_controller.rrt is not None:
                # Set target for RRT
                target_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Default orientation
                self.franka_controller.rrt.set_end_effector_target(target_position, target_orientation)
                self.franka_controller.rrt.update_world()

                # Get current joint positions
                if hasattr(self.franka_controller, 'path_planner_visualizer'):
                    active_joints = self.franka_controller.path_planner_visualizer.get_active_joints_subset()
                    start_pos = active_joints.get_joint_positions()
                else:
                    start_pos = current_joint_positions

                # Set max iterations
                max_iterations = 8000
                self.franka_controller.rrt.set_max_iterations(max_iterations)

                # Compute RRT path
                rrt_path = self.franka_controller.rrt.compute_path(start_pos, np.array([]))

                if rrt_path is not None and len(rrt_path) > 1:
                    # Calculate path length in joint space
                    path_length = 0.0
                    for i in range(len(rrt_path) - 1):
                        path_length += np.linalg.norm(rrt_path[i+1] - rrt_path[i])

                    return {
                        "success": True,
                        "path_length": path_length,
                        "path": rrt_path
                    }
                else:
                    return {"success": False, "reason": "no_path_found"}
            else:
                # Fallback: estimate path length using Euclidean distance
                ee_position = self.franka_controller.franka.end_effector.get_world_pose()[0]
                euclidean_distance = np.linalg.norm(target_position - ee_position)

                # Assume path is 1.5x Euclidean distance (rough estimate)
                estimated_path_length = euclidean_distance * 1.5

                return {
                    "success": True,
                    "path_length": estimated_path_length,
                    "path": None,
                    "estimated": True
                }

        except Exception as e:
            print(f"[ENV RRT] Error planning path: {e}")
            return {"success": False, "reason": "exception", "error": str(e)}

    def _execute_pick_place(self, action: int, rrt_path) -> bool:
        """
        Execute actual pick-and-place operation.

        Args:
            action: Object index to pick
            rrt_path: RRT path to follow

        Returns:
            True if successful, False otherwise
        """
        if not self.execute_picks or self.franka_controller is None:
            return True  # Assume success if not executing

        try:
            # This would call the actual pick-and-place execution
            # For now, return True (will be implemented when integrated with standalone script)
            return True

        except Exception as e:
            print(f"[ENV RRT] Error executing pick-place: {e}")
            return False

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment and clear RRT statistics"""
        obs, info = super().reset(seed=seed, options=options)

        # Add RRT statistics to info
        if len(self.rrt_planning_times) > 0:
            info["avg_rrt_planning_time"] = np.mean(self.rrt_planning_times)
            info["avg_rrt_path_length"] = np.mean(self.rrt_path_lengths)
            info["rrt_success_rate"] = self.rrt_success_count / (self.rrt_success_count + self.rrt_failure_count)

        # Reset statistics for new episode
        self.rrt_planning_times = []
        self.rrt_path_lengths = []
        self.rrt_success_count = 0
        self.rrt_failure_count = 0

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action with RRT planning"""
        obs, reward, terminated, truncated, info = super().step(action)

        # Add RRT statistics to info
        if len(self.rrt_planning_times) > 0:
            info["last_rrt_planning_time"] = self.rrt_planning_times[-1]
        if len(self.rrt_path_lengths) > 0:
            info["last_rrt_path_length"] = self.rrt_path_lengths[-1]

        return obs, reward, terminated, truncated, info

