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
from .distance_utils import (
    calculate_cube_to_cube_edge_distance,
    calculate_object_to_obstacle_edge_distance_conservative
)


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
        training_grid_size: int = 6,
        cube_spacing: Optional[float] = None  # Override cube spacing (None = use default)
    ):
        super().__init__()

        self.franka_controller = franka_controller
        self.max_objects = max_objects
        self.max_steps = max_steps
        self.num_cubes = num_cubes  # Number of cubes per episode
        self.render_mode = render_mode
        self.dynamic_obstacles = dynamic_obstacles  # Enable real-time obstacle recalculation
        self.training_grid_size = training_grid_size  # Fixed grid size for training (e.g., 6x6)
        self.custom_cube_spacing = cube_spacing  # Custom cube spacing override (None = use default)

        # Test mode flag: When True, skip expensive reachability checks in action_masks()
        # This speeds up testing and PCA fitting by 10-100x for RRT-based environments
        self.test_mode = False

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
        # Grid center: [0.45, -0.10], cell_size: 0.20 (UPDATED for gripper width), grid_size: 6
        # UPDATED: Cell size increased from 0.13m to 0.20m to fit gripper palm (15.2cm wide)
        # Grid range: X: -0.05 to 0.95, Y: -0.60 to 0.40
        # For bottom-middle in SCREEN coordinates (where Y increases downward):
        # - Grid cell (2, 5) center: (0.45, 0.40)
        # Cell (2, 5) boundaries: X [0.35, 0.55], Y [0.30, 0.50]
        # IMPORTANT: EE must be at cell CENTER to avoid rounding issues with cubes in same cell
        self.ee_position = np.array([0.45, 0.40, 0.5])  # Bottom-middle cell center (UPDATED for 0.20m cells)
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        # DEBUG: Print when reset is called
        import traceback
        print(f"\n{'='*80}")
        print(f"[DEBUG] ObjectSelectionEnv.reset() called!")
        print(f"[DEBUG] Cubes picked before reset: {len(self.objects_picked)}/{getattr(self, 'total_objects', 0)}")
        print(f"[DEBUG] Call stack:")
        for line in traceback.format_stack()[-6:-1]:
            print(f"  {line.strip()}")
        print(f"{'='*80}\n")

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
        info["action_mask"] = self.action_masks()  # Add action mask to initial info

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

        # Validate action (should never happen with action masking, but keep as safety check)
        if action >= self.total_objects or action in self.objects_picked:
            # Invalid action: already picked or out of range
            # This should be prevented by action masking, but handle gracefully
            print(f"[WARNING] Invalid action {action} selected (already picked or out of range)")
            reward = -10.0
            terminated = False
            truncated = self.current_step >= self.max_steps
            obs = self._get_observation()
            info = self._get_info()
            info["invalid_action"] = True
            info["action_mask"] = self.action_masks()
            return obs, reward, terminated, truncated, info

        # Calculate reward for this pick
        reward = self._calculate_reward(action)

        # Mark object as picked
        self.objects_picked.append(action)

        # Check if episode is done
        terminated = len(self.objects_picked) >= self.total_objects
        truncated = self.current_step >= self.max_steps

        # DEBUG: Print when all cubes are picked
        if terminated:
            print(f"\n{'='*80}")
            print(f"[DEBUG] ALL CUBES PICKED! Setting terminated=True")
            print(f"[DEBUG] Cubes picked: {len(self.objects_picked)}/{self.total_objects}")
            print(f"[DEBUG] objects_picked = {self.objects_picked}")
            print(f"{'='*80}\n")

        # Get new observation (with dynamic obstacle recalculation if enabled)
        obs = self._get_observation(recalculate_obstacles=self.dynamic_obstacles)
        info = self._get_info()
        info["action_mask"] = self.action_masks()  # Add action mask to info

        # Bonus for completing all objects
        if terminated:
            time_bonus = max(0, (self.max_steps - self.current_step) * 0.5)
            reward += 20.0 + time_bonus
            info["success"] = True

        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, action: int) -> float:
        """
        Calculate reward for picking a specific object.

        UPDATED: Now uses all 6 observation parameters for reward calculation:
        1. Distance to EE / Path length (5 points max)
        2. Distance to container (3 points max) - NEW!
        3. Obstacle proximity (3 points max)
        4. Reachability flag (-5 penalty if unreachable) - NEW!
        5. Path clearance (2 points max) - NEW!
        6. Picked flag (handled via invalid action penalty)

        Additional penalties:
        - Collision penalty: -5 if straight-line path blocked (Heuristic only)
        - Time penalty: -1 per step

        Bonuses:
        - First pick bonus: +5 if optimal first pick
        - Completion bonus: +20 + time bonus (handled in step())
        """
        reward = 0.0
        obj_position = self.object_positions[action]

        # Base reward for successful pick
        reward += 10.0

        # 1. Distance to EE reward (max 10 points) - INCREASED for nearest-first priority
        # Check for straight-line collision (Heuristic method)
        distance = np.linalg.norm(obj_position - self.ee_position)

        if self._check_straight_line_collision(self.ee_position, obj_position, action):
            # Collision detected - apply penalty
            reward -= 10.0  # INCREASED path planning failure penalty
            # Use 2.0 × Euclidean for distance calculation (same as A*/RRT failure)
            effective_distance = 2.0 * distance
        else:
            effective_distance = distance

        # Distance reward inversely proportional to distance (max 10 points) - INCREASED
        distance_reward = 10.0 * np.exp(-effective_distance)  # INCREASED from 5.0
        reward += distance_reward

        # 2. Distance to container reward (max 3 points)
        dist_to_container = np.linalg.norm(obj_position - self.container_position)
        container_reward = 3.0 * np.exp(-dist_to_container)
        reward += container_reward

        # 3. Obstacle proximity reward (max 7 points) - INCREASED
        # Now includes unpicked cubes as obstacles
        obstacle_score = self._calculate_obstacle_score_with_unpicked_cubes(obj_position, action)
        obstacle_reward = 7.0 * (1.0 - obstacle_score)  # INCREASED from 3.0
        reward += obstacle_reward

        # 4. Reachability penalty (INCREASED: -50 if unreachable, was -10)
        if not (0.3 <= distance <= 0.9):
            reward -= 50.0  # HARSH penalty for unreachable cubes

        # 5. Path clearance reward (max 4 points) - INCREASED
        clearance_score = self._calculate_path_clearance(self.ee_position, obj_position)
        clearance_reward = 4.0 * clearance_score  # INCREASED from 2.0
        reward += clearance_reward

        # 6. Additional penalties for risky picks (close to masking thresholds)
        # UPDATED thresholds to match new Layer 2 (0.35) and Layer 3 (0.50) values
        if clearance_score < 0.40:  # Close to Layer 2 threshold (0.35)
            reward -= 5.0  # Risky pick - narrow path

        if obstacle_score > 0.45:  # Close to Layer 3 threshold (0.50)
            reward -= 20.0  # HARSH penalty for crowded area (was -5.0)

        # Time penalty (encourage speed) - INCREASED
        reward -= 2.0  # INCREASED from -1.0

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
            # FIX 2: Use consistent obstacle score calculation (includes unpicked cubes)
            if recalculate_obstacles:
                # DYNAMIC: Recalculate obstacle score in real-time
                obstacle_score = self._calculate_obstacle_score_with_unpicked_cubes(obj_pos, i)
                obs[i, 2] = obstacle_score
            else:
                # STATIC: Use pre-calculated score from reset()
                obs[i, 2] = self.obstacle_scores[i]

            # 4. Reachability flag (1 value)
            # FIX 2: Use actual path planning instead of distance heuristic
            # Subclasses override _calculate_reachability() with A*/RRT checks
            reachable = self._calculate_reachability(i, dist_to_ee)
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

    def action_masks(self, skip_reachability_check: Optional[bool] = None) -> np.ndarray:
        """
        Return action mask for valid actions.

        Args:
            skip_reachability_check: If True, only mask already picked cubes (fast, for testing).
                                    If False, use full 3-layer reachability check (for training).
                                    If None, use self.test_mode flag (default behavior).

        Returns:
            Boolean array where True = valid action, False = invalid action
            Shape: (max_objects,)
        """
        # Use test_mode flag if skip_reachability_check not explicitly provided
        if skip_reachability_check is None:
            skip_reachability_check = self.test_mode

        mask = np.zeros(self.max_objects, dtype=bool)

        if skip_reachability_check:
            # FAST: Only mask already picked cubes (for testing with pretrained agents)
            # Pretrained agents already learned to avoid unreachable/unsafe cubes during training
            for i in range(self.total_objects):
                if i not in self.objects_picked:
                    mask[i] = True
        else:
            # FULL: 3-layer reachability check (for training)
            # Force obstacle grid update for dynamic obstacles (external obstacles, Lidar-detected)
            # This ensures action masks always use latest obstacle positions
            if hasattr(self, '_update_astar_grid'):
                self._update_astar_grid()  # Update A* grid with latest obstacles
            elif hasattr(self, '_update_rrt_grid'):
                self._update_rrt_grid()  # Update RRT grid with latest obstacles

            # Try robust reachability check first (3-layer validation)
            for i in range(self.total_objects):
                if i not in self.objects_picked:
                    mask[i] = self._is_reachable_robust(i)

            # Fallback: If no valid actions, use basic reachability check
            if not mask.any():
                for i in range(self.total_objects):
                    if i not in self.objects_picked:
                        mask[i] = self._is_reachable(i)

            # REMOVED: Last resort fallback that forced blocked cubes to be valid
            # If no valid actions remain, episode will terminate naturally
            # This prevents infinite loops trying to pick unreachable cubes

        return mask

    def _is_reachable(self, obj_idx: int) -> bool:
        """
        Basic reachability check (base implementation).
        Subclasses override this with path planner-specific checks (A*/RRT).

        Args:
            obj_idx: Index of object to check

        Returns:
            True if object is reachable, False otherwise
        """
        # Base implementation: Simple distance heuristic
        distance = np.linalg.norm(self.object_positions[obj_idx] - self.ee_position)
        return 0.3 <= distance <= 0.9

    def _is_reachable_robust(self, obj_idx: int) -> bool:
        """
        Robust 3-layer reachability check.
        Combines path existence, path quality, and safety margin.

        Layer 1: Path existence (can path planner find a path?)
        Layer 2: Path clearance (is path wide enough/safe?)
        Layer 3: Obstacle proximity (is target too close to obstacles?)

        Subclasses can override for path planner-specific implementations.

        Args:
            obj_idx: Index of object to check

        Returns:
            True if object passes all 3 safety checks, False otherwise
        """
        # Layer 1: Basic reachability (path exists)
        if not self._is_reachable(obj_idx):
            return False

        # Layer 2: Path clearance check
        # UPDATED: 20cm clearance (was 35cm) - adjusted for new 26cm cell spacing
        # With 26cm spacing, minimum gap between objects is ~18cm (cube-obstacle)
        # Gripper width = 15cm, so 20cm clearance ensures safe passage
        path_clearance = self._calculate_path_clearance(
            self.ee_position, self.object_positions[obj_idx]
        )
        if path_clearance < 0.20:  # Minimum 20cm clearance for safe navigation
            return False

        # Layer 3: Obstacle proximity check
        # UPDATED: 45% max obstacle proximity (was 50%) - slightly tighter for larger cell spacing
        obstacle_score = self._calculate_obstacle_score_with_unpicked_cubes(
            self.object_positions[obj_idx], obj_idx
        )
        if obstacle_score > 0.45:  # Max 45% obstacle proximity (too crowded)
            return False

        return True  # All 3 layers passed!

    def _get_info(self) -> Dict:
        """Get additional info"""
        return {
            "objects_picked": len(self.objects_picked),
            "total_objects": self.total_objects,
            "current_step": self.current_step,
            "episode_time": time.time() - self.episode_start_time,
            "invalid_action": False,
            "success": False,
            "action_mask": self.action_masks()  # Include action mask in info
        }

    def _update_object_data(self):
        """Update object data from Franka controller"""
        if not self.franka_controller:
            return

        # CRITICAL FIX: Randomize cube and obstacle positions each episode
        # This provides diverse training scenarios instead of fixed positions
        if hasattr(self.franka_controller, 'randomize_cube_positions'):
            self.franka_controller.randomize_cube_positions()

        if hasattr(self.franka_controller, 'randomize_obstacle_positions'):
            self.franka_controller.randomize_obstacle_positions()

        # Step the world a few times to let physics settle after randomization
        if hasattr(self.franka_controller, 'world') and self.franka_controller.world is not None:
            for _ in range(5):
                self.franka_controller.world.step(render=False)

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
        # Match visualizer logic: available_cells = grid_capacity - num_cubes - 1 (EE home cell)
        total_cells = grid_size * grid_size
        num_cubes = len(occupied_cells)
        available_cells = total_cells - num_cubes - 1  # -1 for EE home cell

        # Generate exactly 1 random obstacle (REDUCED from 1-3 to reduce RRT failures)
        max_obstacles = max(0, min(1, available_cells))  # Cap at 1, or 0 if no room
        min_obstacles = 1 if max_obstacles > 0 else 0    # Exactly 1 if room exists
        num_obstacles = 1 if max_obstacles > 0 else 0

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
            # UPDATED: Cell size accounts for gripper width (15.2cm) + safety margin
            # Gripper palm width: 15.2cm, so cells must be >= 20cm to fit gripper comfortably
            cell_size = 0.20 if grid_size > 3 else 0.22  # 20cm for 4x4+, 22cm for 3x3
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

        UPDATED: Uses edge-to-edge distance instead of center-to-center for accurate clearance.

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

            # Check edge-to-edge distance to obstacles at this sample point
            if hasattr(self.franka_controller, 'lidar_detected_obstacles'):
                for obs_name, obs_data in self.franka_controller.lidar_detected_obstacles.items():
                    obs_pos = obs_data.get('position', None)
                    if obs_pos is not None:
                        # Use conservative edge-to-edge distance
                        # Gripper radius ~0.076m (15.2cm width / 2), obstacle radius ~0.05m
                        edge_distance = calculate_object_to_obstacle_edge_distance_conservative(
                            obj_pos=sample_pos,
                            obs_pos=obs_pos,
                            obj_radius=0.076,  # Gripper half-width
                            obs_radius=0.05    # Conservative obstacle radius
                        )
                        min_clearance = min(min_clearance, edge_distance)

        # Convert distance to score (0.0 = blocked, 1.0 = clear)
        if min_clearance == float('inf'):
            return 1.0  # No obstacles detected
        else:
            # Normalize: 0cm = 0.0, 30cm+ = 1.0
            clearance_score = min(min_clearance / 0.3, 1.0)
            return clearance_score

    def _check_straight_line_collision(self, start_pos: np.ndarray, end_pos: np.ndarray, target_action: int) -> bool:
        """
        Check if straight-line path from start to end collides with obstacles or unpicked cubes.

        Args:
            start_pos: Start position (robot EE)
            end_pos: End position (target object)
            target_action: Index of target object (excluded from collision check)

        Returns:
            True if collision detected, False if path is clear
        """
        # Sample points along the straight-line path
        num_samples = 10
        # UPDATED: Collision threshold accounts for gripper half-width (7.6cm) + sphere radius (3cm)
        collision_threshold = 0.10  # 10cm - gripper palm half-width + safety margin

        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0.5
            sample_pos = start_pos + t * (end_pos - start_pos)

            # Check collision with random obstacles
            for obs_pos in self.random_obstacle_positions:
                distance = np.linalg.norm(sample_pos[:2] - obs_pos[:2])
                if distance < collision_threshold:
                    return True  # Collision detected

            # Check collision with unpicked cubes (except target)
            for i in range(self.total_objects):
                if i == target_action or i in self.objects_picked:
                    continue  # Skip target and already picked cubes

                cube_pos = self.object_positions[i]
                distance = np.linalg.norm(sample_pos[:2] - cube_pos[:2])
                if distance < collision_threshold:
                    return True  # Collision detected

        return False  # No collision

    def _calculate_obstacle_score_with_unpicked_cubes(self, position: np.ndarray, target_action: int) -> float:
        """
        Calculate obstacle proximity score including unpicked cubes as obstacles.

        UPDATED: Uses edge-to-edge distance instead of center-to-center for accurate collision detection.

        Args:
            position: Object position to check
            target_action: Index of target object (excluded from obstacle check)

        Returns:
            Score from 0.0 (no obstacles) to 1.0 (very close to obstacle)
        """
        min_distance = float('inf')

        # Check distance to random obstacles (external obstacles)
        # Use conservative edge-to-edge distance
        for obs_pos in self.random_obstacle_positions:
            edge_distance = calculate_object_to_obstacle_edge_distance_conservative(
                obj_pos=position,
                obs_pos=obs_pos,
                obj_radius=0.0354,  # Cube bounding radius (0.025 * sqrt(2))
                obs_radius=0.05     # Conservative 5cm obstacle radius
            )
            min_distance = min(min_distance, edge_distance)

        # Check distance to unpicked cubes (except target)
        # Use edge-to-edge distance for cube-to-cube
        for i in range(self.total_objects):
            if i == target_action or i in self.objects_picked:
                continue  # Skip target and already picked cubes

            cube_pos = self.object_positions[i]
            edge_distance = calculate_cube_to_cube_edge_distance(
                pos1=position,
                pos2=cube_pos,
                half_edge=0.025  # 5cm cube half-edge
            )
            min_distance = min(min_distance, edge_distance)

        # Convert distance to score (0.0 = far, 1.0 = very close)
        # Within 10cm = 1.0, beyond 30cm = 0.0
        if min_distance < 0.10:
            return 1.0
        elif min_distance > 0.30:
            return 0.0
        else:
            # Linear interpolation between 0.10 and 0.30
            return 1.0 - (min_distance - 0.10) / 0.20

    def _calculate_reachability(self, obj_idx: int, dist_to_ee: float) -> float:
        """
        Calculate reachability flag for an object.
        Base implementation uses distance heuristic.
        Subclasses override with A*/RRT checks.

        Args:
            obj_idx: Index of object to check
            dist_to_ee: Euclidean distance from EE to object

        Returns:
            1.0 if reachable, 0.0 if unreachable
        """
        # Base implementation: Simple distance heuristic
        # Franka workspace: ~30cm to 90cm from base
        if obj_idx in self.objects_picked:
            return 0.0  # Already picked

        return 1.0 if (0.3 <= dist_to_ee <= 0.9) else 0.0

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
        # CRITICAL: cube_spacing MUST match cell_size used by path estimators!
        # Use custom spacing if provided, otherwise use default
        if self.custom_cube_spacing is not None:
            cube_spacing = self.custom_cube_spacing
        else:
            # Default: Updated from 0.20/0.22 to 0.26/0.28 to ensure gripper (15cm) can fit between objects
            # With 26cm spacing: cube-to-cube gap = 26cm - 5.15cm = 20.85cm (gripper 15cm fits with 5.85cm clearance)
            cube_spacing = 0.26 if grid_size > 3 else 0.28  # Match path estimator cell_size
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

    # ========================================================================
    # Methods for Two-Agent Environment (MAPPO Reshuffling)
    # ========================================================================

    def get_cube_positions(self) -> np.ndarray:
        """
        Get positions of all cubes (including picked ones).

        Returns:
            Array of cube positions (num_cubes, 3)
        """
        # Return only the actual cubes (not padding)
        cube_positions = []
        for i in range(min(self.total_objects, self.num_cubes)):
            cube_positions.append(self.object_positions[i])

        return np.array(cube_positions, dtype=np.float32)

    def get_robot_position(self) -> np.ndarray:
        """
        Get robot BASE position (NOT end-effector).

        IMPORTANT: Changed from EE to base for consistency with A*/RRT visualizers.
        The base is fixed (like the circle in A*/RRT viz), while EE moves during pick/place.
        This gives more stable distance measurements for reshuffling decisions.

        Returns:
            Robot base position (3,) - typically [-0.1, 0.0, 0.6] for Franka
        """
        # If using Franka controller, get actual base position
        if self.franka_controller and hasattr(self.franka_controller, 'franka'):
            base_pos, _ = self.franka_controller.franka.get_world_pose()
            return base_pos.copy()

        # Default base position for standalone training (matches Isaac Sim Franka base)
        # Franka base is at [-0.1, 0.0, 0.6] in Isaac Sim
        return np.array([-0.1, 0.0, 0.6], dtype=np.float32)

    def get_obstacle_positions(self) -> np.ndarray:
        """
        Get obstacle positions (random obstacles + unpicked cubes).

        Returns:
            Array of obstacle positions (M, 3)
        """
        obstacle_positions = []

        # Add random obstacles
        for obs_pos in self.random_obstacle_positions:
            obstacle_positions.append(obs_pos)

        # Add unpicked cubes as obstacles (for path planning)
        for i in range(self.total_objects):
            if i not in self.objects_picked:
                obstacle_positions.append(self.object_positions[i])

        return np.array(obstacle_positions, dtype=np.float32) if obstacle_positions else np.zeros((0, 3), dtype=np.float32)

    def move_cube(self, cube_idx: int, target_pos: np.ndarray):
        """
        Move a cube to a target position (for reshuffling).

        Args:
            cube_idx: Index of cube to move
            target_pos: Target position (3,) [x, y, z]
        """
        if cube_idx < 0 or cube_idx >= self.total_objects:
            print(f"[WARNING] Invalid cube index: {cube_idx}")
            return

        if cube_idx in self.objects_picked:
            print(f"[WARNING] Cannot move already picked cube: {cube_idx}")
            return

        # Update cube position in environment state
        self.object_positions[cube_idx] = target_pos.copy()

        # If using Franka controller (Isaac Sim), move the actual cube
        if self.franka_controller and hasattr(self.franka_controller, 'cubes'):
            if cube_idx < len(self.franka_controller.cubes):
                cube, cube_name = self.franka_controller.cubes[cube_idx]
                # Set new position (keep current orientation)
                _, current_orientation = cube.get_world_pose()
                cube.set_world_pose(position=target_pos, orientation=current_orientation)
                print(f"[RESHUFFLE] Moved cube {cube_idx} ({cube_name}) to {target_pos}")
        else:
            print(f"[RESHUFFLE] Updated cube {cube_idx} position to {target_pos} (simulation only)")

    def remove_picked_cube(self, cube_idx: int):
        """
        Remove a picked cube from the environment (for testing performance optimization).

        This moves the cube far away from the workspace to prevent it from being
        considered as an obstacle during path planning.

        IMPORTANT: Only use during testing! Training needs picked cubes to remain
        as obstacles to learn proper collision avoidance.

        Args:
            cube_idx: Index of cube to remove
        """
        if cube_idx < 0 or cube_idx >= self.total_objects:
            print(f"[WARNING] Invalid cube index: {cube_idx}")
            return

        if cube_idx not in self.objects_picked:
            print(f"[WARNING] Cannot remove unpicked cube: {cube_idx}")
            return

        # Move cube far away from workspace (effectively removing it from obstacle checks)
        # Use a position far outside the workspace (10 meters away)
        far_away_pos = np.array([10.0, 10.0, 10.0])

        # Update cube position in environment state
        self.object_positions[cube_idx] = far_away_pos.copy()

        # If using Franka controller (Isaac Sim), move the actual cube far away
        if self.franka_controller and hasattr(self.franka_controller, 'cubes'):
            if cube_idx < len(self.franka_controller.cubes):
                cube, cube_name = self.franka_controller.cubes[cube_idx]
                # Move cube far away (keep current orientation)
                _, current_orientation = cube.get_world_pose()
                cube.set_world_pose(position=far_away_pos, orientation=current_orientation)
                print(f"[REMOVE] Moved picked cube {cube_idx} ({cube_name}) far away (testing optimization)")
        else:
            print(f"[REMOVE] Moved picked cube {cube_idx} far away (testing optimization)")


