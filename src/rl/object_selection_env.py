import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import time
from .distance_utils import (
    calculate_cube_to_cube_edge_distance,
    calculate_object_to_obstacle_edge_distance_conservative
)


class ObjectSelectionEnv(gym.Env):
    """Gym environment for training RL agent to select optimal pick order"""

    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        franka_controller=None,
        max_objects: int = 10,
        max_steps: int = 50,
        num_cubes: int = 4,
        render_mode: Optional[str] = None,
        dynamic_obstacles: bool = False,
        training_grid_size: int = 6,
        cube_spacing: Optional[float] = None
    ):
        super().__init__()
        self.franka_controller = franka_controller
        self.max_objects = max_objects
        self.max_steps = max_steps
        self.num_cubes = num_cubes
        self.render_mode = render_mode
        self.dynamic_obstacles = dynamic_obstacles
        self.training_grid_size = training_grid_size
        self.custom_cube_spacing = cube_spacing
        self.test_mode = False
        self.action_space = spaces.Discrete(max_objects)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(max_objects * 6,),
            dtype=np.float32
        )
        self.current_step = 0
        self.objects_picked = []
        self.total_objects = 0
        self.episode_start_time = 0
        self.object_positions = []
        self.object_types = []
        self.object_names = []
        self.obstacle_scores = []
        self.random_obstacle_positions = []
        self.container_position = np.array([0.6, 0.0, 0.0])
        self.ee_position = np.array([0.45, 0.40, 0.5])
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        self.current_step = 0
        self.objects_picked = []
        self.episode_start_time = time.time()
        if self.franka_controller:
            self._update_object_data()
        else:
            self._generate_random_objects(use_pygame_style=True)
        obs = self._get_observation(recalculate_obstacles=self.dynamic_obstacles)
        info = self._get_info()
        info["action_mask"] = self.action_masks()
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action (select object to pick)"""
        self.current_step += 1
        if action >= self.total_objects or action in self.objects_picked:
            reward = -10.0
            terminated = False
            truncated = self.current_step >= self.max_steps
            obs = self._get_observation()
            info = self._get_info()
            info["invalid_action"] = True
            info["action_mask"] = self.action_masks()
            return obs, reward, terminated, truncated, info
        reward = self._calculate_reward(action)
        self.objects_picked.append(action)
        terminated = len(self.objects_picked) >= self.total_objects
        truncated = self.current_step >= self.max_steps
        obs = self._get_observation(recalculate_obstacles=self.dynamic_obstacles)
        info = self._get_info()
        info["action_mask"] = self.action_masks()
        if terminated:
            time_bonus = max(0, (self.max_steps - self.current_step) * 0.5)
            reward += 20.0 + time_bonus
            info["success"] = True
        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, action: int) -> float:
        """Calculate reward for picking a specific object"""
        reward = 0.0
        obj_position = self.object_positions[action]
        reward += 10.0
        distance = np.linalg.norm(obj_position - self.ee_position)
        if self._check_straight_line_collision(self.ee_position, obj_position, action):
            reward -= 10.0
            effective_distance = 2.0 * distance
        else:
            effective_distance = distance
        distance_reward = 10.0 * np.exp(-effective_distance)
        reward += distance_reward
        dist_to_container = np.linalg.norm(obj_position - self.container_position)
        container_reward = 3.0 * np.exp(-dist_to_container)
        reward += container_reward
        obstacle_score = self._calculate_obstacle_score_with_unpicked_cubes(obj_position, action)
        obstacle_reward = 7.0 * (1.0 - obstacle_score)
        reward += obstacle_reward
        if not (0.3 <= distance <= 0.9):
            reward -= 50.0
        clearance_score = self._calculate_path_clearance(self.ee_position, obj_position)
        clearance_reward = 4.0 * clearance_score
        reward += clearance_reward
        if clearance_score < 0.40:
            reward -= 5.0
        if obstacle_score > 0.45:
            reward -= 20.0
        reward -= 2.0
        if len(self.objects_picked) == 0:
            distances = [np.linalg.norm(pos - self.ee_position) for pos in self.object_positions]
            if action == np.argmin(distances):
                reward += 5.0
        return reward

    def _get_observation(self, recalculate_obstacles=True) -> np.ndarray:
        """Get current observation"""
        obs = np.zeros((self.max_objects, 6), dtype=np.float32)
        if self.franka_controller:
            self.ee_position = self.franka_controller.franka.end_effector.get_world_pose()[0]
            self.container_position = self.franka_controller.container.get_world_pose()[0]
        ee_position = self.ee_position
        container_position = self.container_position
        for i in range(self.total_objects):
            obj_pos = self.object_positions[i]
            dist_to_ee = np.linalg.norm(obj_pos - ee_position)
            obs[i, 0] = dist_to_ee
            dist_to_container = np.linalg.norm(obj_pos - container_position)
            obs[i, 1] = dist_to_container
            if recalculate_obstacles:
                obstacle_score = self._calculate_obstacle_score_with_unpicked_cubes(obj_pos, i)
                obs[i, 2] = obstacle_score
            else:
                obs[i, 2] = self.obstacle_scores[i]
            reachable = self._calculate_reachability(i, dist_to_ee)
            obs[i, 3] = reachable
            path_clearance = self._calculate_path_clearance(ee_position, obj_pos)
            obs[i, 4] = path_clearance
            picked_flag = 1.0 if i in self.objects_picked else 0.0
            obs[i, 5] = picked_flag
        return obs.flatten()

    def action_masks(self, skip_reachability_check: Optional[bool] = None) -> np.ndarray:
        """Return action mask for valid actions"""
        if skip_reachability_check is None:
            skip_reachability_check = self.test_mode
        mask = np.zeros(self.max_objects, dtype=bool)
        if skip_reachability_check:
            for i in range(self.total_objects):
                if i not in self.objects_picked:
                    mask[i] = True
        else:
            if hasattr(self, '_update_astar_grid'):
                self._update_astar_grid()
            elif hasattr(self, '_update_rrt_grid'):
                self._update_rrt_grid()
            for i in range(self.total_objects):
                if i not in self.objects_picked:
                    mask[i] = self._is_reachable_robust(i)
            if not mask.any():
                for i in range(self.total_objects):
                    if i not in self.objects_picked:
                        mask[i] = self._is_reachable(i)
        return mask

    def _is_reachable(self, obj_idx: int) -> bool:
        """Basic reachability check (base implementation)"""
        distance = np.linalg.norm(self.object_positions[obj_idx] - self.ee_position)
        return 0.3 <= distance <= 0.9

    def _is_reachable_robust(self, obj_idx: int) -> bool:
        """Robust 3-layer reachability check"""
        if not self._is_reachable(obj_idx):
            return False
        path_clearance = self._calculate_path_clearance(
            self.ee_position, self.object_positions[obj_idx]
        )
        if path_clearance < 0.20:
            return False
        obstacle_score = self._calculate_obstacle_score_with_unpicked_cubes(
            self.object_positions[obj_idx], obj_idx
        )
        if obstacle_score > 0.45:
            return False
        return True

    def _get_info(self) -> Dict:
        """Get additional info"""
        return {
            "objects_picked": len(self.objects_picked),
            "total_objects": self.total_objects,
            "current_step": self.current_step,
            "episode_time": time.time() - self.episode_start_time,
            "invalid_action": False,
            "success": False,
            "action_mask": self.action_masks()
        }

    def _update_object_data(self):
        """Update object data from Franka controller"""
        if not self.franka_controller:
            return
        if hasattr(self.franka_controller, 'randomize_cube_positions'):
            self.franka_controller.randomize_cube_positions()
        if hasattr(self.franka_controller, 'randomize_obstacle_positions'):
            self.franka_controller.randomize_obstacle_positions()
        if hasattr(self.franka_controller, 'world') and self.franka_controller.world is not None:
            for _ in range(5):
                self.franka_controller.world.step(render=False)
        self.object_positions = []
        self.object_types = []
        self.object_names = []
        self.obstacle_scores = []
        for cube, cube_name in self.franka_controller.cubes:
            pos, _ = cube.get_world_pose()
            self.object_positions.append(pos)
            self.object_types.append("cube")
            self.object_names.append(cube_name)
            obstacle_score = self._calculate_obstacle_score(pos)
            self.obstacle_scores.append(obstacle_score)
        self.total_objects = len(self.object_positions)

    def _generate_random_obstacles(self, grid_size: int, occupied_cells: set) -> list:
        """Generate random obstacle positions in empty grid cells"""
        total_cells = grid_size * grid_size
        num_cubes = len(occupied_cells)
        available_cells = total_cells - num_cubes - 1
        max_obstacles = max(0, min(1, available_cells))
        num_obstacles = 1 if max_obstacles > 0 else 0
        empty_cells = []
        for grid_x in range(grid_size):
            for grid_y in range(grid_size):
                if (grid_x, grid_y) not in occupied_cells:
                    empty_cells.append((grid_x, grid_y))
        obstacle_positions = []
        if len(empty_cells) >= num_obstacles:
            np.random.shuffle(empty_cells)
            selected_cells = empty_cells[:num_obstacles]
            cell_size = 0.20 if grid_size > 3 else 0.22
            grid_center = np.array([0.45, -0.10])
            grid_extent = (grid_size - 1) * cell_size
            start_x = grid_center[0] - (grid_extent / 2.0)
            start_y = grid_center[1] - (grid_extent / 2.0)
            for grid_x, grid_y in selected_cells:
                world_x = start_x + (grid_x * cell_size)
                world_y = start_y + (grid_y * cell_size)
                world_z = 0.055
                obstacle_positions.append(np.array([world_x, world_y, world_z]))
        return obstacle_positions

    def _calculate_random_obstacle_score(self, position: np.ndarray, obstacle_positions: list) -> float:
        """Calculate obstacle score based on random obstacle positions"""
        if not obstacle_positions:
            return 0.0
        min_distance = float('inf')
        for obs_pos in obstacle_positions:
            distance = np.linalg.norm(position[:2] - obs_pos[:2])
            min_distance = min(min_distance, distance)
        if min_distance < 0.10:
            return 1.0
        elif min_distance > 0.30:
            return 0.0
        else:
            return 1.0 - (min_distance - 0.10) / 0.20

    def _calculate_obstacle_score(self, position: np.ndarray) -> float:
        """Calculate obstacle proximity score for a position"""
        if not self.franka_controller:
            return 0.0
        if hasattr(self.franka_controller, 'lidar_detected_obstacles'):
            obstacle_count = 0
            min_distance = float('inf')
            for _, obs_data in self.franka_controller.lidar_detected_obstacles.items():
                obs_pos = obs_data.get('position', None)
                if obs_pos is not None:
                    distance = np.linalg.norm(position - obs_pos)
                    if distance < 0.3:
                        obstacle_count += 1
                        min_distance = min(min_distance, distance)
            if obstacle_count == 0:
                return 0.0
            else:
                count_score = min(obstacle_count / 3.0, 1.0)
                distance_score = 1.0 - min(min_distance / 0.3, 1.0)
                return (count_score + distance_score) / 2.0
        return 0.0

    def _calculate_path_clearance(self, start_pos: np.ndarray, end_pos: np.ndarray) -> float:
        """Calculate path clearance score for straight-line path from start to end"""
        if not self.franka_controller:
            return 1.0
        num_samples = 5
        min_clearance = float('inf')
        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0.5
            sample_pos = start_pos + t * (end_pos - start_pos)
            if hasattr(self.franka_controller, 'lidar_detected_obstacles'):
                for _, obs_data in self.franka_controller.lidar_detected_obstacles.items():
                    obs_pos = obs_data.get('position', None)
                    if obs_pos is not None:
                        edge_distance = calculate_object_to_obstacle_edge_distance_conservative(
                            obj_pos=sample_pos,
                            obs_pos=obs_pos,
                            obj_radius=0.076,
                            obs_radius=0.05
                        )
                        min_clearance = min(min_clearance, edge_distance)
        if min_clearance == float('inf'):
            return 1.0
        else:
            clearance_score = min(min_clearance / 0.3, 1.0)
            return clearance_score

    def _check_straight_line_collision(self, start_pos: np.ndarray, end_pos: np.ndarray, target_action: int) -> bool:
        """Check if straight-line path from start to end collides with obstacles or unpicked cubes"""
        num_samples = 10
        collision_threshold = 0.10
        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0.5
            sample_pos = start_pos + t * (end_pos - start_pos)
            for obs_pos in self.random_obstacle_positions:
                distance = np.linalg.norm(sample_pos[:2] - obs_pos[:2])
                if distance < collision_threshold:
                    return True
            for obj_idx in range(self.total_objects):
                if obj_idx == target_action or obj_idx in self.objects_picked:
                    continue
                cube_pos = self.object_positions[obj_idx]
                distance = np.linalg.norm(sample_pos[:2] - cube_pos[:2])
                if distance < collision_threshold:
                    return True
        return False

    def _calculate_obstacle_score_with_unpicked_cubes(self, position: np.ndarray, target_action: int) -> float:
        """Calculate obstacle proximity score including unpicked cubes as obstacles"""
        min_distance = float('inf')
        for obs_pos in self.random_obstacle_positions:
            edge_distance = calculate_object_to_obstacle_edge_distance_conservative(
                obj_pos=position,
                obs_pos=obs_pos,
                obj_radius=0.0354,
                obs_radius=0.05
            )
            min_distance = min(min_distance, edge_distance)
        for i in range(self.total_objects):
            if i == target_action or i in self.objects_picked:
                continue
            cube_pos = self.object_positions[i]
            edge_distance = calculate_cube_to_cube_edge_distance(
                pos1=position,
                pos2=cube_pos,
                half_edge=0.025
            )
            min_distance = min(min_distance, edge_distance)
        if min_distance < 0.10:
            return 1.0
        elif min_distance > 0.30:
            return 0.0
        else:
            return 1.0 - (min_distance - 0.10) / 0.20

    def _calculate_reachability(self, obj_idx: int, dist_to_ee: float) -> float:
        """Calculate reachability flag for an object"""
        if obj_idx in self.objects_picked:
            return 0.0
        return 1.0 if (0.3 <= dist_to_ee <= 0.9) else 0.0

    def _generate_random_objects(self, use_pygame_style=True):
        """Generate random objects for testing without Franka controller"""
        grid_size = self.training_grid_size
        self.object_positions = []
        self.object_types = []
        self.object_names = []
        self.obstacle_scores = []
        if self.custom_cube_spacing is not None:
            cube_spacing = self.custom_cube_spacing
        else:
            cube_spacing = 0.26 if grid_size > 3 else 0.28
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
            ee_home_row = grid_size // 2
            ee_home_col = grid_size - 1
            ee_home_idx = ee_home_row * grid_size + ee_home_col
            available_cells = [i for i in range(total_cells) if i != ee_home_idx]
            selected_indices = np.random.choice(available_cells, size=n_objects, replace=False)
            self.total_objects = n_objects
            occupied_cells = set()
            occupied_cells.add((ee_home_row, ee_home_col))
            for idx in selected_indices:
                row = idx // grid_size
                col = idx % grid_size
                occupied_cells.add((row, col))
                base_x = start_x + (row * cube_spacing)
                base_y = start_y + (col * cube_spacing)
                random_offset_x = np.random.uniform(-random_offset_range, random_offset_range)
                random_offset_y = np.random.uniform(-random_offset_range, random_offset_range)
                x = base_x + random_offset_x
                y = base_y + random_offset_y
                z = 0.05
                self.object_positions.append(np.array([x, y, z]))
                obj_type = np.random.choice(["cube", "cylinder", "sphere"])
                self.object_types.append(obj_type)
                self.object_names.append(f"object_{row}_{col}")
            self.random_obstacle_positions = self._generate_random_obstacles(grid_size, occupied_cells)
            for pos in self.object_positions:
                obstacle_score = self._calculate_random_obstacle_score(pos, self.random_obstacle_positions)
                self.obstacle_scores.append(obstacle_score)
        else:
            total_cells = grid_size * grid_size
            self.total_objects = min(self.max_objects, total_cells)
            for row in range(grid_size):
                for col in range(grid_size):
                    base_x = start_x + (row * cube_spacing)
                    base_y = start_y + (col * cube_spacing)
                    random_offset_x = np.random.uniform(-random_offset_range, random_offset_range)
                    random_offset_y = np.random.uniform(-random_offset_range, random_offset_range)
                    x = base_x + random_offset_x
                    y = base_y + random_offset_y
                    z = 0.05
                    self.object_positions.append(np.array([x, y, z]))
                    obj_type = np.random.choice(["cube", "cylinder", "sphere"])
                    self.object_types.append(obj_type)
                    self.object_names.append(f"object_{row}_{col}")
                    self.obstacle_scores.append(np.random.uniform(0.0, 0.5))

    def render(self):
        """Render environment (handled by Isaac Sim)"""
        if self.render_mode == "human":
            pass

    def close(self):
        """Clean up resources"""
        pass

    def get_cube_positions(self) -> np.ndarray:
        """Get positions of all cubes (including picked ones)"""
        cube_positions = []
        for i in range(min(self.total_objects, self.num_cubes)):
            cube_positions.append(self.object_positions[i])
        return np.array(cube_positions, dtype=np.float32)

    def get_robot_position(self) -> np.ndarray:
        """Get robot BASE position (NOT end-effector)"""
        if self.franka_controller and hasattr(self.franka_controller, 'franka'):
            base_pos, _ = self.franka_controller.franka.get_world_pose()
            return base_pos.copy()
        return np.array([-0.1, 0.0, 0.6], dtype=np.float32)

    def get_obstacle_positions(self) -> np.ndarray:
        """Get obstacle positions (random obstacles + unpicked cubes)"""
        obstacle_positions = []
        for obs_pos in self.random_obstacle_positions:
            obstacle_positions.append(obs_pos)
        for i in range(self.total_objects):
            if i not in self.objects_picked:
                obstacle_positions.append(self.object_positions[i])
        return np.array(obstacle_positions, dtype=np.float32) if obstacle_positions else np.zeros((0, 3), dtype=np.float32)

    def move_cube(self, cube_idx: int, target_pos: np.ndarray):
        """Move a cube to a target position (for reshuffling)"""
        if cube_idx < 0 or cube_idx >= self.total_objects:
            return
        if cube_idx in self.objects_picked:
            return
        self.object_positions[cube_idx] = target_pos.copy()
        if self.franka_controller and hasattr(self.franka_controller, 'cubes'):
            if cube_idx < len(self.franka_controller.cubes):
                cube, cube_name = self.franka_controller.cubes[cube_idx]
                _, current_orientation = cube.get_world_pose()
                cube.set_world_pose(position=target_pos, orientation=current_orientation)

    def remove_picked_cube(self, cube_idx: int):
        """Remove a picked cube from the environment (for testing performance optimization)"""
        if cube_idx < 0 or cube_idx >= self.total_objects:
            return
        if cube_idx not in self.objects_picked:
            return
        far_away_pos = np.array([10.0, 10.0, 10.0])
        self.object_positions[cube_idx] = far_away_pos.copy()
        if self.franka_controller and hasattr(self.franka_controller, 'cubes'):
            if cube_idx < len(self.franka_controller.cubes):
                cube, cube_name = self.franka_controller.cubes[cube_idx]
                _, current_orientation = cube.get_world_pose()
                cube.set_world_pose(position=far_away_pos, orientation=current_orientation)


