import numpy as np
import time
import random
from typing import Dict, Optional, Tuple
from .object_selection_env import ObjectSelectionEnv
from .path_estimators_isaacsim import IsaacSimRRTPathEstimator
from .distance_utils import calculate_placement_position


class ObjectSelectionEnvRRT(ObjectSelectionEnv):
    """Environment variant that uses actual RRT path planning for reward calculation"""

    def __init__(
        self,
        franka_controller=None,
        max_objects: int = 10,
        max_steps: int = 50,
        num_cubes: int = 4,
        render_mode: Optional[str] = None,
        dynamic_obstacles: bool = True,
        training_grid_size: int = 6,
        execute_picks: bool = False,
        rrt_planner=None,
        kinematics_solver=None,
        articulation_kinematics_solver=None,
        franka_articulation=None
    ):
        ddqn_spacing = 0.13 if training_grid_size > 3 else 0.15

        super().__init__(
            franka_controller=franka_controller,
            max_objects=max_objects,
            max_steps=max_steps,
            num_cubes=num_cubes,
            render_mode=render_mode,
            dynamic_obstacles=dynamic_obstacles,
            training_grid_size=training_grid_size,
            cube_spacing=ddqn_spacing
        )

        self.execute_picks = execute_picks

        self.rrt_estimator = IsaacSimRRTPathEstimator(
            grid_size=training_grid_size,
            cell_size=0.13 if training_grid_size > 3 else 0.15,
            rrt_planner=rrt_planner,
            kinematics_solver=kinematics_solver,
            articulation_kinematics_solver=articulation_kinematics_solver,
            franka_articulation=franka_articulation
        )

        self.rrt_planning_times = []
        self.rrt_path_lengths = []
        self.rrt_optimal_path_lengths = []
        self.rrt_success_count = 0
        self.rrt_failure_count = 0
        self.episode_count = 0
        self.stats_print_interval = 100

        self.episode_rrt_failures = 0
        self.episode_pick_failures = 0
        self.episode_collisions = 0
        self.episode_successful_picks = 0
        self.episode_unreachable_cubes = 0

        self.cubes_skipped = []

    def set_rrt_components(self, rrt_planner, kinematics_solver,
                          articulation_kinematics_solver, franka_articulation):
        """Set all RRT components"""
        self.rrt_estimator.set_rrt_planner(rrt_planner)
        self.rrt_estimator.set_kinematics_solver(kinematics_solver)
        self.rrt_estimator.set_articulation_kinematics_solver(articulation_kinematics_solver)
        self.rrt_estimator.set_franka_articulation(franka_articulation)

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

        if self.franka_controller is not None:
            obj_position = self.object_positions[obj_idx]
            rrt_result = self._plan_rrt_path_to_object(obj_position, self.object_names[obj_idx], max_retries=1)
            return rrt_result["success"]

        elif hasattr(self, 'rrt_estimator') and self.rrt_estimator is not None:
            obj_position = self.object_positions[obj_idx]

            self._update_rrt_grid(target_cube_idx=obj_idx)

            return self.rrt_estimator.check_reachability(
                self.ee_position,
                obj_position
            )

        else:
            dist = np.linalg.norm(self.object_positions[obj_idx] - self.ee_position)
            return 0.3 <= dist <= 0.9

    def _calculate_reachability(self, _: int, dist_to_ee: float) -> float:
        """Calculate reachability for OBSERVATION (called for ALL objects)"""
        return 1.0 if (0.3 <= dist_to_ee <= 0.9) else 0.0

    def _calculate_reward(self, action: int) -> float:
        """Calculate reward using RRT path estimation"""
        reward = 0.0
        obj_position = self.object_positions[action]
        reward += 10.0

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

    def _plan_rrt_path_to_object(self, obj_position: np.ndarray, obj_name: str, max_retries: int = 3) -> Dict:
        """Plan RRT path to object position with retry """
        if self.franka_controller is None:
            return {"success": False, "reason": "no_controller", "attempts": 0}

        try:
            current_joint_positions = self.franka_controller.franka.get_joint_positions()

            pick_height_offset = 0.15
            target_position = obj_position.copy()
            target_position[2] += pick_height_offset

            if hasattr(self.franka_controller, 'rrt') and self.franka_controller.rrt is not None:
                target_orientation = np.array([1.0, 0.0, 0.0, 0.0])
                self.franka_controller.rrt.set_end_effector_target(target_position, target_orientation)
                self.franka_controller.rrt.update_world()

                if hasattr(self.franka_controller, 'path_planner_visualizer'):
                    active_joints = self.franka_controller.path_planner_visualizer.get_active_joints_subset()
                    start_pos = active_joints.get_joint_positions()
                else:
                    start_pos = current_joint_positions

                max_iterations = 8000
                self.franka_controller.rrt.set_max_iterations(max_iterations)

                for attempt in range(1, max_retries + 1):
                    start_time = time.time()
                    rrt_path = self.franka_controller.rrt.compute_path(start_pos, np.array([]))
                    planning_time = time.time() - start_time

                    if rrt_path is not None and len(rrt_path) > 1:
                        path_length = 0.0

                        for i in range(len(rrt_path) - 1):
                            path_length += np.linalg.norm(rrt_path[i+1] - rrt_path[i])

                        optimal_path_length = np.linalg.norm(rrt_path[-1] - rrt_path[0])

                        if attempt > 1:
                            print(f"[ENV RRT] Path found for {obj_name} on attempt {attempt}/{max_retries}")

                        return {
                            "success": True,
                            "path_length": path_length,
                            "optimal_path_length": optimal_path_length,
                            "path": rrt_path,
                            "attempts": attempt,
                            "planning_time": planning_time
                        }

                    else:

                        if attempt < max_retries:
                            print(f"[ENV RRT] Attempt {attempt}/{max_retries} failed for {obj_name}, retrying...")
                        else:
                            print(f"[ENV RRT] All {max_retries} attempts failed for {obj_name}")

                return {"success": False, "reason": "no_path_found_after_retries", "attempts": max_retries}

            else:
                ee_position = self.franka_controller.franka.end_effector.get_world_pose()[0]
                euclidean_distance = np.linalg.norm(target_position - ee_position)

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

    def _execute_pick_place(self, action: int, _) -> Tuple[bool, str]:
        """Execute actual pick-and-place operation"""
        if not self.execute_picks or self.franka_controller is None:
            return True, ""

        try:

            if hasattr(self.franka_controller, 'pick_and_place_cube'):
                result = self.franka_controller.pick_and_place_cube(action)

                if isinstance(result, tuple) and len(result) >= 2:
                    success, message = result[0], result[1]

                    if not success:

                        if "collision" in message.lower() or "obstacle" in message.lower():
                            return False, "collision"
                        else:
                            return False, "execution_error"

                    return True, ""

                else:
                    return True, ""

            else:
                return True, ""

        except Exception as e:
            print(f"[ENV RRT] Error executing pick-place: {e}")
            return False, "execution_error"

    def _get_cube_spacing_for_deployment(self) -> float:
        """Get cube spacing that matches deployment script"""
        if self.num_cubes <= 4:
            return 0.14

        elif self.num_cubes <= 9:
            return 0.12

        else:
            return 0.10

    def _generate_random_objects(self, use_pygame_style=True):
        """Override to use deployment spacing instead of training spacing"""
        grid_size = self.training_grid_size

        self.object_positions = []
        self.object_types = []
        self.object_names = []
        self.obstacle_scores = []

        cube_spacing = self._get_cube_spacing_for_deployment()
        grid_center_x = 0.45
        grid_center_y = -0.10
        grid_extent_x = (grid_size - 1) * cube_spacing
        grid_extent_y = (grid_size - 1) * cube_spacing
        start_x = grid_center_x - (grid_extent_x / 2.0)
        start_y = grid_center_y - (grid_extent_y / 2.0)
        random_offset_range = 0.03

        if use_pygame_style:
            total_cells = grid_size * grid_size
            n_objects = min(self.num_cubes, total_cells - 1)

            ee_home_row = grid_size - 1
            ee_home_col = grid_size // 2
            ee_home_cell = ee_home_row * grid_size + ee_home_col

            available_cells = [i for i in range(total_cells) if i != ee_home_cell]
            occupied_cells = random.sample(available_cells, n_objects)

            occupied_cells_set = set()

            for cell_idx in occupied_cells + [ee_home_cell]:
                grid_x = cell_idx % grid_size
                grid_y = cell_idx // grid_size
                occupied_cells_set.add((grid_x, grid_y))

            self.random_obstacle_positions = self._generate_random_obstacles(
                grid_size=grid_size,
                occupied_cells=occupied_cells_set
            )

            for cell_idx in occupied_cells:
                row = cell_idx // grid_size
                col = cell_idx % grid_size

                x = start_x + col * cube_spacing
                y = start_y + row * cube_spacing

                x += np.random.uniform(-random_offset_range, random_offset_range)
                y += np.random.uniform(-random_offset_range, random_offset_range)
                z = 0.5

                self.object_positions.append(np.array([x, y, z]))
                self.object_types.append(0)
                self.object_names.append(f"cube_{len(self.object_positions)}")

            self.total_objects = len(self.object_positions)

            for i in range(self.total_objects):
                score = self._calculate_obstacle_score_with_unpicked_cubes(
                    self.object_positions[i], i
                )
                self.obstacle_scores.append(score)

    def _get_observation(self, recalculate_obstacles: bool = False) -> np.ndarray:
        """Override to calculate distance to actual placement position instead of container center"""
        obs = np.zeros((self.max_objects, 6), dtype=np.float32)

        if self.franka_controller:
            self.ee_position = self.franka_controller.franka.end_effector.get_world_pose()[0]
            container_position = self.franka_controller.container.get_world_pose()[0]
            container_dimensions = getattr(self.franka_controller, 'container_dimensions', np.array([0.48, 0.36, 0.128]))
        else:
            container_position = self.container_position
            container_dimensions = np.array([0.48, 0.36, 0.128])

        ee_position = self.ee_position

        picked_count = len(self.objects_picked)

        for i in range(self.total_objects):
            obj_pos = self.object_positions[i]

            dist_to_ee = np.linalg.norm(obj_pos - ee_position)
            obs[i, 0] = dist_to_ee

            placement_position = calculate_placement_position(
                cube_index=picked_count,
                total_cubes=self.num_cubes,
                container_center=container_position,
                container_dimensions=container_dimensions
            )
            dist_to_container = np.linalg.norm(obj_pos - placement_position)
            obs[i, 1] = dist_to_container

            if recalculate_obstacles:
                obstacle_score = self._calculate_obstacle_score_with_unpicked_cubes(obj_pos, i)
                obs[i, 2] = obstacle_score
            else:
                obs[i, 2] = self.obstacle_scores[i]

            reachability = self._calculate_reachability(i, dist_to_ee)
            obs[i, 3] = reachability

            path_clearance = self._calculate_path_clearance(ee_position, obj_pos)
            obs[i, 4] = path_clearance

            obs[i, 5] = 1.0 if i in self.objects_picked else 0.0

        return obs.flatten()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment and clear RRT statistics"""
        obs, info = super().reset(seed=seed, options=options)

        if len(self.rrt_planning_times) > 0:
            info["avg_rrt_planning_time"] = np.mean(self.rrt_planning_times)
            info["avg_rrt_path_length"] = np.mean(self.rrt_path_lengths)
            info["total_rrt_path_length"] = np.sum(self.rrt_path_lengths)
            info["avg_rrt_optimal_path_length"] = np.mean(self.rrt_optimal_path_lengths) if len(self.rrt_optimal_path_lengths) > 0 else 0.0
            info["total_rrt_optimal_path_length"] = np.sum(self.rrt_optimal_path_lengths) if len(self.rrt_optimal_path_lengths) > 0 else 0.0
            info["rrt_success_rate"] = self.rrt_success_count / (self.rrt_success_count + self.rrt_failure_count) if (self.rrt_success_count + self.rrt_failure_count) > 0 else 0.0
        else:
            info["avg_rrt_planning_time"] = 0.0
            info["avg_rrt_path_length"] = 0.0
            info["total_rrt_path_length"] = 0.0
            info["avg_rrt_optimal_path_length"] = 0.0
            info["total_rrt_optimal_path_length"] = 0.0
            info["rrt_success_rate"] = 0.0

        info["episode_rrt_failures"] = self.episode_rrt_failures
        info["episode_pick_failures"] = self.episode_pick_failures
        info["episode_collisions"] = self.episode_collisions
        info["episode_successful_picks"] = self.episode_successful_picks
        info["episode_unreachable_cubes"] = self.episode_unreachable_cubes

        self.rrt_planning_times = []
        self.rrt_path_lengths = []
        self.rrt_optimal_path_lengths = []
        self.rrt_success_count = 0
        self.rrt_failure_count = 0

        self.episode_rrt_failures = 0
        self.episode_pick_failures = 0
        self.episode_collisions = 0
        self.episode_successful_picks = 0
        self.episode_unreachable_cubes = 0

        self.cubes_skipped = []

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action with RRT planning and proper success tracking"""
        self.current_step += 1

        if action >= self.total_objects or action in self.objects_picked:
            print(f"[WARNING] Invalid action {action} selected (already picked or out of range)")
            reward = -10.0
            terminated = False
            truncated = self.current_step >= self.max_steps
            obs = self._get_observation()
            info = self._get_info()
            info["invalid_action"] = True
            info["action_mask"] = self.action_masks()
            return obs, reward, terminated, truncated, info

        info = {}

        obj_position = self.object_positions[action]
        obj_name = self.object_names[action]
        rrt_result = self._plan_rrt_path_to_object(obj_position, obj_name, max_retries=1)

        if rrt_result["success"]:
            self.rrt_success_count += 1

            if "path_length" in rrt_result:
                self.rrt_path_lengths.append(rrt_result["path_length"])

            if "optimal_path_length" in rrt_result:
                self.rrt_optimal_path_lengths.append(rrt_result["optimal_path_length"])

            if "planning_time" in rrt_result:
                self.rrt_planning_times.append(rrt_result["planning_time"])

            reward = self._calculate_reward(action)

            pick_success, failure_reason = self._execute_pick_place(action, rrt_result.get("path"))

            if pick_success:
                self.objects_picked.append(action)
                self.episode_successful_picks += 1

                info["pick_success"] = True
                info["rrt_success"] = True
                info["rrt_attempts"] = rrt_result.get("attempts", 1)

            else:
                self.episode_pick_failures += 1

                if failure_reason == "collision":
                    self.episode_collisions += 1
                    reward = -20.0
                    info["collision"] = True
                    info["pick_failed_reason"] = "collision"
                else:
                    reward = -15.0
                    info["pick_failed_reason"] = failure_reason

                info["pick_success"] = False
                info["rrt_success"] = True

        else:
            self.rrt_failure_count += 1
            self.episode_rrt_failures += 1

            attempts = rrt_result.get("attempts", 3)
            info["cube_skipped"] = False
            info["skip_reason"] = f"rrt_unreachable_after_{attempts}_attempts"
            print(f"[ENV RRT] Cube {action} unreachable after {attempts} RRT attempts.")

            reward = -25.0

            info["rrt_success"] = False
            info["rrt_failed_reason"] = rrt_result.get("reason", "unknown")
            info["rrt_attempts"] = attempts
            info["pick_success"] = False

        successfully_picked = len(self.objects_picked) - len(self.cubes_skipped)
        terminated = successfully_picked >= self.num_cubes
        truncated = self.current_step >= self.max_steps

        action_mask = self.action_masks()

        if not terminated and not truncated:

            if not action_mask.any():
                unpicked_count = self.num_cubes - len(self.objects_picked)
                self.episode_unreachable_cubes = unpicked_count

                print(f"[ENV RRT] No valid actions remaining (all remaining cubes unreachable)")
                print(f"[ENV RRT] Successfully picked: {successfully_picked}/{self.num_cubes}, Unreachable: {unpicked_count}")
                terminated = True
                info["termination_reason"] = "no_valid_actions"

        obs = self._get_observation(recalculate_obstacles=self.dynamic_obstacles)
        info["action_mask"] = action_mask

        if terminated:
            time_bonus = max(0, (self.max_steps - self.current_step) * 0.5)
            reward += 20.0 + time_bonus
            info["success"] = True

        if len(self.rrt_planning_times) > 0:
            info["last_rrt_planning_time"] = self.rrt_planning_times[-1]

        if len(self.rrt_path_lengths) > 0:
            info["last_rrt_path_length"] = self.rrt_path_lengths[-1]

        info["episode_rrt_failures"] = self.episode_rrt_failures
        info["episode_pick_failures"] = self.episode_pick_failures
        info["episode_collisions"] = self.episode_collisions
        info["episode_successful_picks"] = self.episode_successful_picks
        info["episode_unreachable_cubes"] = self.episode_unreachable_cubes
        info["episode_cubes_skipped"] = len(self.cubes_skipped)
        info["successfully_picked_count"] = successfully_picked

        if terminated or truncated:

            if len(self.rrt_planning_times) > 0:
                info["avg_rrt_planning_time"] = np.mean(self.rrt_planning_times)
                info["avg_rrt_path_length"] = np.mean(self.rrt_path_lengths)
                info["total_rrt_path_length"] = np.sum(self.rrt_path_lengths)
                info["avg_rrt_optimal_path_length"] = np.mean(self.rrt_optimal_path_lengths) if len(self.rrt_optimal_path_lengths) > 0 else 0.0
                info["total_rrt_optimal_path_length"] = np.sum(self.rrt_optimal_path_lengths) if len(self.rrt_optimal_path_lengths) > 0 else 0.0
                info["rrt_success_rate"] = self.rrt_success_count / (self.rrt_success_count + self.rrt_failure_count) if (self.rrt_success_count + self.rrt_failure_count) > 0 else 0.0
            else:
                info["avg_rrt_planning_time"] = 0.0
                info["avg_rrt_path_length"] = 0.0
                info["total_rrt_path_length"] = 0.0
                info["avg_rrt_optimal_path_length"] = 0.0
                info["total_rrt_optimal_path_length"] = 0.0
                info["rrt_success_rate"] = 0.0

        return obs, reward, terminated, truncated, info

