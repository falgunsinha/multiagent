"""
RL Environment for Object Selection with RRT Path Planning (Option 3)
Uses actual RRT path planning in Isaac Sim for training.
Requires Isaac Sim to be running during training.
"""

import numpy as np
import time
from typing import Dict, Optional, Tuple
from .object_selection_env import ObjectSelectionEnv
from .path_estimators import RRTPathEstimator


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
        execute_picks: bool = False  # Whether to actually execute pick-and-place
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
        self.rrt_planner = None

        # Initialize RRT path estimator (for visualization and training)
        self.rrt_estimator = RRTPathEstimator(
            grid_size=training_grid_size,
            cell_size=0.13 if training_grid_size > 3 else 0.15,
            franka_controller=franka_controller
        )

        # RRT performance tracking
        self.rrt_planning_times = []
        self.rrt_path_lengths = []
        self.rrt_success_count = 0
        self.rrt_failure_count = 0
    
    def set_rrt_planner(self, rrt_planner):
        """Set the RRT planner instance"""
        self.rrt_planner = rrt_planner
    
    def _calculate_reward(self, action: int) -> float:
        """
        Calculate reward using RRT path estimation.

        MUST match A* reward formula exactly for fair comparison.
        """
        reward = 0.0

        # Base reward for successful pick (same as A*)
        reward += 10.0

        # Use stored ee_position (updated in _get_observation from parent class)
        obj_position = self.object_positions[action]

        # RRT path length reward (instead of Euclidean distance)
        rrt_path_length = self.rrt_estimator.estimate_path_length(self.ee_position, obj_position)

        # Reward inversely proportional to RRT path length (max 5 points, same as A*)
        # Normalize by typical path length (0.3m to 0.9m, same as A*)
        normalized_path_length = (rrt_path_length - 0.3) / 0.6
        normalized_path_length = np.clip(normalized_path_length, 0.0, 1.0)
        path_reward = 5.0 * (1.0 - normalized_path_length)
        reward += path_reward

        # Obstacle avoidance reward (max 3 points, same as A*)
        obstacle_score = self.obstacle_scores[action]
        obstacle_reward = 3.0 * (1.0 - obstacle_score)
        reward += obstacle_reward

        # Time penalty (encourage speed, same as A*)
        reward -= 1.0

        # Sequential picking bonus: reward for picking object with shortest RRT path first (same as A*)
        if len(self.objects_picked) == 0:
            # First pick: bonus if it's the object with shortest RRT path
            rrt_lengths = [
                self.rrt_estimator.estimate_path_length(self.ee_position, pos)
                for pos in self.object_positions[:self.total_objects]
            ]
            if action == np.argmin(rrt_lengths):
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

