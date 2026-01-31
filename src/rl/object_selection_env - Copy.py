"""
RL Environment for Intelligent Object Selection in Franka Pick-and-Place
Wraps the existing Franka RRT v1.9 code with Gym interface for RL training.

The agent learns to select which object to pick first based on:
- Distance to robot
- Obstacle proximity
- Object type (cube, cylinder, etc.)
- Time efficiency
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import time


class ObjectSelectionEnv(gym.Env):
    """
    Gym environment for training RL agent to select optimal pick order.
    
    Observation Space:
        For each object (max 10 objects):
        - Position (x, y, z): 3 values
        - Distance to robot EE: 1 value
        - Distance to container: 1 value
        - Object type (one-hot): 3 values (cube, cylinder, sphere)
        - Obstacle proximity score: 1 value
        - Already picked flag: 1 value
        Total per object: 10 values
        Total observation: 10 objects × 10 = 100 values
        
    Action Space:
        Discrete(10): Select which object index to pick next (0-9)
    
    Reward:
        - +10 for successful pick
        - +5 for picking closest object
        - +3 for picking object with fewer obstacles
        - -1 for each timestep (encourage speed)
        - +20 bonus for completing all objects
        - -10 penalty for failed pick
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        franka_controller=None,
        max_objects: int = 10,
        max_steps: int = 50,
        num_cubes: int = 4,
        render_mode: Optional[str] = None,
        dynamic_obstacles: bool = False,
        training_grid_size: int = 6
    ):
        super().__init__()

        self.franka_controller = franka_controller
        self.max_objects = max_objects
        self.max_steps = max_steps
        self.num_cubes = num_cubes  # Number of cubes per episode
        self.render_mode = render_mode
        self.dynamic_obstacles = dynamic_obstacles  # Enable real-time obstacle recalculation
        self.training_grid_size = training_grid_size  # Fixed grid size for training (e.g., 6x6)

        # Action space: select which object to pick (0 to max_objects-1)
        self.action_space = spaces.Discrete(max_objects)

        # Observation space: features for each object
        # UPDATED: 6 values per object (added picked flag)
        # 1. Distance to robot EE
        # 2. Distance to container
        # 3. Obstacle proximity score
        # 4. Reachability flag
        # 5. Path clearance score
        # 6. Picked flag (0.0 = available, 1.0 = already picked)
        # Shape: (max_objects, 6) flattened to (60,) for max_objects=10
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(max_objects * 6,),
            dtype=np.float32
        )

        # Episode state
        self.current_step = 0
        self.objects_picked = []  # List of picked object indices
        self.total_objects = 0
        self.episode_start_time = 0

        # Object data (will be populated by Franka controller)
        self.object_positions = []
        self.object_types = []
        self.object_names = []
        self.obstacle_scores = []
        self.random_obstacle_positions = []  # Random obstacles in empty cells

        # Container position (for visualization and distance calculations)
        self.container_position = np.array([0.6, 0.0, 0.0])  # Default position

        # EE position: For A* path planning, EE must be at a valid grid cell CENTER
        # Grid center: [0.45, -0.10], cell_size: 0.13, grid_size: 6
        # Grid range: X: 0.125 to 0.775, Y: -0.425 to 0.225
        # For bottom-middle in SCREEN coordinates (where Y increases downward):
        # - Grid cell (2, 5) center: (0.385, 0.225)
        # Cell (2, 5) boundaries: X [0.32, 0.45], Y [0.16, 0.29]
        # IMPORTANT: EE must be at cell CENTER to avoid rounding issues with cubes in same cell
        self.ee_position = np.array([0.385, 0.225, 0.5])  # At cell (2, 5) CENTER
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.objects_picked = []
        self.episode_start_time = time.time()
        
        # Get object data from Franka controller
        if self.franka_controller:
            self._update_object_data()
        else:
            # For testing without controller, generate random objects
            # Always use PyGame-style: random placement in fixed training grid
            self._generate_random_objects(use_pygame_style=True)

        obs = self._get_observation(recalculate_obstacles=self.dynamic_obstacles)
        info = self._get_info()

        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action (select object to pick)
        
        Args:
            action: Index of object to pick (0 to max_objects-1)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        
        # Validate action
        if action >= self.total_objects or action in self.objects_picked:
            # Invalid action: already picked or out of range
            reward = -10.0
            terminated = False
            truncated = self.current_step >= self.max_steps
            obs = self._get_observation()
            info = self._get_info()
            info["invalid_action"] = True
            return obs, reward, terminated, truncated, info
        
        # Calculate reward for this pick
        reward = self._calculate_reward(action)

        # Mark object as picked
        self.objects_picked.append(action)

        # Check if episode is done
        terminated = len(self.objects_picked) >= self.total_objects
        truncated = self.current_step >= self.max_steps

        # Get new observation (with dynamic obstacle recalculation if enabled)
        obs = self._get_observation(recalculate_obstacles=self.dynamic_obstacles)
        info = self._get_info()
        
        # Bonus for completing all objects
        if terminated:
            time_bonus = max(0, (self.max_steps - self.current_step) * 0.5)
            reward += 20.0 + time_bonus
            info["success"] = True
        
        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, action: int) -> float:
        """
        Calculate reward for picking a specific object.

        Reward components:
        1. Distance reward: Closer objects get higher reward
        2. Obstacle avoidance reward: Objects with fewer obstacles get higher reward
        3. Time penalty: Encourage faster completion
        4. Sequential picking bonus: Reward for picking in optimal order
        """
        reward = 0.0

        # Base reward for successful pick
        reward += 10.0

        # Distance reward: Normalize distance and invert (closer = better)
        # Use stored ee_position (updated in _get_observation)
        obj_position = self.object_positions[action]
        distance = np.linalg.norm(obj_position - self.ee_position)

        # Reward inversely proportional to distance (max 5 points)
        distance_reward = 5.0 * np.exp(-distance)
        reward += distance_reward

        # Obstacle avoidance reward (max 3 points)
        obstacle_score = self.obstacle_scores[action]
        obstacle_reward = 3.0 * (1.0 - obstacle_score)  # Lower obstacle score = higher reward
        reward += obstacle_reward

        # Time penalty (encourage speed)
        reward -= 1.0

        # Sequential picking bonus: reward for picking closest object first
        if len(self.objects_picked) == 0:
            # First pick: bonus if it's the closest object
            distances = [np.linalg.norm(pos - self.ee_position) for pos in self.object_positions]
            if action == np.argmin(distances):
                reward += 5.0

        return reward

    def _get_observation(self, recalculate_obstacles=True) -> np.ndarray:
        """
        Get current observation.

        UPDATED: 6 values per object (added picked flag)
        1. Distance to robot EE: 1 value
        2. Distance to container: 1 value
        3. Obstacle proximity score: 1 value
        4. Reachability flag: 1 value
        5. Path clearance score: 1 value
        6. Picked flag: 1 value (0.0 = available, 1.0 = already picked)

        Args:
            recalculate_obstacles: If True, recalculates obstacle scores in real-time
                                  (enables dynamic obstacle handling)
        """
        obs = np.zeros((self.max_objects, 6), dtype=np.float32)

        # Get robot EE position and container position
        if self.franka_controller:
            self.ee_position = self.franka_controller.franka.end_effector.get_world_pose()[0]
            self.container_position = self.franka_controller.container.get_world_pose()[0]
        # else: use default values set in __init__

        ee_position = self.ee_position
        container_position = self.container_position

        for i in range(self.total_objects):
            obj_pos = self.object_positions[i]

            # 1. Distance to EE (1 value)
            dist_to_ee = np.linalg.norm(obj_pos - ee_position)
            obs[i, 0] = dist_to_ee

            # 2. Distance to container (1 value)
            dist_to_container = np.linalg.norm(obj_pos - container_position)
            obs[i, 1] = dist_to_container

            # 3. Obstacle proximity score (1 value)
            if recalculate_obstacles:
                # DYNAMIC: Recalculate obstacle score in real-time
                obstacle_score = self._calculate_obstacle_score(obj_pos)
                obs[i, 2] = obstacle_score
            else:
                # STATIC: Use pre-calculated score from reset()
                obs[i, 2] = self.obstacle_scores[i]

            # 4. Reachability flag (1 value) - NEW
            # Simple heuristic: reachable if within workspace limits
            # Franka workspace: ~30cm to 90cm from base
            reachable = 1.0 if (0.3 <= dist_to_ee <= 0.9) else 0.0
            obs[i, 3] = reachable

            # 5. Path clearance score (1 value)
            # Measure free space around straight-line path from EE to object
            path_clearance = self._calculate_path_clearance(ee_position, obj_pos)
            obs[i, 4] = path_clearance

            # 6. Picked flag (1 value) - NEW
            # 0.0 = available to pick, 1.0 = already picked
            picked_flag = 1.0 if i in self.objects_picked else 0.0
            obs[i, 5] = picked_flag

        # Flatten to 1D array
        return obs.flatten()

    def _get_info(self) -> Dict:
        """Get additional info"""
        return {
            "objects_picked": len(self.objects_picked),
            "total_objects": self.total_objects,
            "current_step": self.current_step,
            "episode_time": time.time() - self.episode_start_time,
            "invalid_action": False,
            "success": False
        }

    def _update_object_data(self):
        """Update object data from Franka controller"""
        if not self.franka_controller:
            return

        self.object_positions = []
        self.object_types = []
        self.object_names = []
        self.obstacle_scores = []

        # Get cubes from Franka controller
        for cube, cube_name in self.franka_controller.cubes:
            pos, _ = cube.get_world_pose()
            self.object_positions.append(pos)
            self.object_types.append("cube")
            self.object_names.append(cube_name)

            # Calculate obstacle proximity score (0 = no obstacles, 1 = many obstacles)
            obstacle_score = self._calculate_obstacle_score(pos)
            self.obstacle_scores.append(obstacle_score)

        self.total_objects = len(self.object_positions)

    def _generate_random_obstacles(self, grid_size: int, occupied_cells: set) -> list:
        """
        Generate random obstacle positions in empty grid cells.
        Smart logic: fewer obstacles when grid is small or has many cubes.

        Args:
            grid_size: Grid size (e.g., 3 for 3x3, 4 for 4x4, 6 for 6x6)
            occupied_cells: Set of (grid_x, grid_y) tuples for cells with cubes

        Returns:
            List of obstacle positions (x, y, z) in world coordinates
        """
        # Calculate available empty cells
        total_cells = grid_size * grid_size
        num_cubes = len(occupied_cells)
        empty_cells_count = total_cells - num_cubes

        # Smart obstacle count: 0-3 obstacles based on grid size and cube count
        # If many cubes (>50% of grid), use fewer obstacles
        cube_density = num_cubes / total_cells

        if cube_density > 0.5:
            # High cube density: 0-1 obstacles
            num_obstacles = np.random.randint(0, 2)
        elif cube_density > 0.3:
            # Medium cube density: 0-2 obstacles
            num_obstacles = np.random.randint(0, 3)
        else:
            # Low cube density: 0-3 obstacles
            num_obstacles = np.random.randint(0, 4)

        # Ensure we don't exceed available empty cells
        num_obstacles = min(num_obstacles, empty_cells_count)

        # Get all empty cells (cells without cubes)
        empty_cells = []
        for grid_x in range(grid_size):
            for grid_y in range(grid_size):
                if (grid_x, grid_y) not in occupied_cells:
                    empty_cells.append((grid_x, grid_y))

        # Randomly select obstacle cells
        obstacle_positions = []
        if len(empty_cells) >= num_obstacles:
            # Randomly shuffle and pick first num_obstacles cells
            np.random.shuffle(empty_cells)
            selected_cells = empty_cells[:num_obstacles]

            # Convert grid coordinates to world coordinates
            cell_size = 0.13 if grid_size > 3 else 0.15
            grid_center = np.array([0.45, -0.10])
            grid_extent = (grid_size - 1) * cell_size
            start_x = grid_center[0] - (grid_extent / 2.0)
            start_y = grid_center[1] - (grid_extent / 2.0)

            for grid_x, grid_y in selected_cells:
                world_x = start_x + (grid_x * cell_size)
                world_y = start_y + (grid_y * cell_size)
                world_z = 0.055  # Standard obstacle height
                obstacle_positions.append(np.array([world_x, world_y, world_z]))

        return obstacle_positions

    def _calculate_random_obstacle_score(self, position: np.ndarray, obstacle_positions: list) -> float:
        """
        Calculate obstacle score based on random obstacle positions.

        Args:
            position: Object position (x, y, z)
            obstacle_positions: List of obstacle positions

        Returns:
            Score from 0.0 (no obstacles) to 1.0 (very close to obstacle)
        """
        if not obstacle_positions:
            return 0.0

        # Calculate minimum distance to any obstacle
        min_distance = float('inf')
        for obs_pos in obstacle_positions:
            distance = np.linalg.norm(position[:2] - obs_pos[:2])  # 2D distance
            min_distance = min(min_distance, distance)

        # Convert distance to score (0.0 = far, 1.0 = very close)
        # Within 10cm = 1.0, beyond 30cm = 0.0
        if min_distance < 0.10:
            return 1.0
        elif min_distance > 0.30:
            return 0.0
        else:
            # Linear interpolation between 0.10 and 0.30
            return 1.0 - (min_distance - 0.10) / 0.20

    def _calculate_obstacle_score(self, position: np.ndarray) -> float:
        """
        Calculate obstacle proximity score for a position.
        Uses Lidar data if available, otherwise uses distance to known obstacles.

        Returns:
            Score from 0.0 (no obstacles) to 1.0 (many obstacles nearby)
        """
        if not self.franka_controller:
            return 0.0

        # Use Lidar detected obstacles if available
        if hasattr(self.franka_controller, 'lidar_detected_obstacles'):
            obstacle_count = 0
            min_distance = float('inf')

            for obs_name, obs_data in self.franka_controller.lidar_detected_obstacles.items():
                obs_pos = obs_data.get('position', None)
                if obs_pos is not None:
                    distance = np.linalg.norm(position - obs_pos)
                    if distance < 0.3:  # Within 30cm
                        obstacle_count += 1
                        min_distance = min(min_distance, distance)

            # Score based on obstacle count and proximity
            if obstacle_count == 0:
                return 0.0
            else:
                # More obstacles and closer = higher score
                count_score = min(obstacle_count / 3.0, 1.0)  # Normalize by 3 obstacles
                distance_score = 1.0 - min(min_distance / 0.3, 1.0)  # Closer = higher
                return (count_score + distance_score) / 2.0

        return 0.0

    def _calculate_path_clearance(self, start_pos: np.ndarray, end_pos: np.ndarray) -> float:
        """
        Calculate path clearance score for straight-line path from start to end.
        Measures free space around the path.

        Args:
            start_pos: Start position (e.g., robot EE)
            end_pos: End position (e.g., object)

        Returns:
            Score from 0.0 (blocked path) to 1.0 (clear path)
        """
        if not self.franka_controller:
            return 1.0  # Assume clear if no controller

        # Sample points along the path
        num_samples = 5
        min_clearance = float('inf')

        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0.5
            sample_pos = start_pos + t * (end_pos - start_pos)

            # Check distance to obstacles at this sample point
            if hasattr(self.franka_controller, 'lidar_detected_obstacles'):
                for obs_name, obs_data in self.franka_controller.lidar_detected_obstacles.items():
                    obs_pos = obs_data.get('position', None)
                    if obs_pos is not None:
                        distance = np.linalg.norm(sample_pos - obs_pos)
                        min_clearance = min(min_clearance, distance)

        # Convert distance to score (0.0 = blocked, 1.0 = clear)
        if min_clearance == float('inf'):
            return 1.0  # No obstacles detected
        else:
            # Normalize: 0cm = 0.0, 30cm+ = 1.0
            clearance_score = min(min_clearance / 0.3, 1.0)
            return clearance_score

    def _generate_random_objects(self, use_pygame_style=True):
        """
        Generate random objects for testing without Franka controller.
        Uses PyGame-style placement: fixed training grid (e.g., 6x6),
        but only spawn a subset of objects in random cells.

        Args:
            use_pygame_style: If True, uses PyGame-style random placement
                            (variable count, random cells in FIXED training grid)
        """
        # Use FIXED training grid size (e.g., 6x6 = 36 cells)
        # This is independent of actual number of cubes spawned
        grid_size = self.training_grid_size

        self.object_positions = []
        self.object_types = []
        self.object_names = []
        self.obstacle_scores = []

        # Grid configuration (matching Franka workspace)
        cube_spacing = 0.15
        grid_center_x = 0.45
        grid_center_y = -0.10
        grid_extent_x = (grid_size - 1) * cube_spacing
        grid_extent_y = (grid_size - 1) * cube_spacing
        start_x = grid_center_x - (grid_extent_x / 2.0)
        start_y = grid_center_y - (grid_extent_y / 2.0)
        random_offset_range = 0.03  # ±3cm random offset

        if use_pygame_style:
            # PyGame-style: Fixed number of objects in random cells of FIXED grid
            # Example: 4 cubes randomly placed in 3x3 grid (9 cells)
            total_cells = grid_size * grid_size
            n_objects = min(self.num_cubes, total_cells - 1)  # Reserve 1 cell for EE home position

            # EE home position: bottom-middle cell for any grid size
            # Grid indexing: idx = row * grid_size + col
            # row affects X (horizontal), col affects Y (vertical)
            # EE is at grid position (grid_x=grid_size//2, grid_y=grid_size-1)
            # In cube placement: row=grid_x, col=grid_y
            ee_home_row = grid_size // 2  # Middle row (grid_x = 2 for 6x6)
            ee_home_col = grid_size - 1  # Last col (grid_y = 5 for 6x6, bottom in screen)
            ee_home_idx = ee_home_row * grid_size + ee_home_col  # idx = 2*6+5 = 17 for 6x6

            # Create list of available cells (excluding EE home position)
            available_cells = [i for i in range(total_cells) if i != ee_home_idx]

            # Randomly select which grid cells to fill (no duplicates)
            selected_indices = np.random.choice(available_cells, size=n_objects, replace=False)

            self.total_objects = n_objects

            # Track occupied cells for obstacle generation (including EE home)
            occupied_cells = set()
            occupied_cells.add((ee_home_row, ee_home_col))  # Reserve EE home cell

            for idx in selected_indices:
                row = idx // grid_size
                col = idx % grid_size

                # Track occupied cell
                occupied_cells.add((row, col))

                # Base grid position (using FIXED training grid size)
                base_x = start_x + (row * cube_spacing)
                base_y = start_y + (col * cube_spacing)

                # Add random offset within cell
                random_offset_x = np.random.uniform(-random_offset_range, random_offset_range)
                random_offset_y = np.random.uniform(-random_offset_range, random_offset_range)

                x = base_x + random_offset_x
                y = base_y + random_offset_y
                z = 0.05

                self.object_positions.append(np.array([x, y, z]))

                # Random type
                obj_type = np.random.choice(["cube", "cylinder", "sphere"])
                self.object_types.append(obj_type)

                self.object_names.append(f"object_{row}_{col}")

            # Generate random obstacles in empty cells
            self.random_obstacle_positions = self._generate_random_obstacles(grid_size, occupied_cells)

            # Calculate obstacle scores based on random obstacles
            for pos in self.object_positions:
                obstacle_score = self._calculate_random_obstacle_score(pos, self.random_obstacle_positions)
                self.obstacle_scores.append(obstacle_score)
        else:
            # Fallback: Fill all grid cells (not used anymore)
            total_cells = grid_size * grid_size
            self.total_objects = min(self.max_objects, total_cells)

            for row in range(grid_size):
                for col in range(grid_size):
                    # Base grid position
                    base_x = start_x + (row * cube_spacing)
                    base_y = start_y + (col * cube_spacing)

                    # Add random offset within cell
                    random_offset_x = np.random.uniform(-random_offset_range, random_offset_range)
                    random_offset_y = np.random.uniform(-random_offset_range, random_offset_range)

                    x = base_x + random_offset_x
                    y = base_y + random_offset_y
                    z = 0.05

                    self.object_positions.append(np.array([x, y, z]))

                    # Random type
                    obj_type = np.random.choice(["cube", "cylinder", "sphere"])
                    self.object_types.append(obj_type)

                    self.object_names.append(f"object_{row}_{col}")

                    # Random obstacle score
                    self.obstacle_scores.append(np.random.uniform(0.0, 0.5))

    def render(self):
        """Render environment (handled by Isaac Sim)"""
        if self.render_mode == "human":
            pass  # Isaac Sim handles rendering

    def close(self):
        """Clean up resources"""
        pass


