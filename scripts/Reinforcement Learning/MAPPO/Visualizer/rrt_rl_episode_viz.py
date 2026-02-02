"""
RL + RRT Episode Visualizer

Visualizes RL agent selecting cubes and RRT planning obstacle-free paths.
Shows one complete episode with tree exploration and interactive charts.
"""

import os
import sys
import argparse
import warnings
import time
from typing import Optional, List, Tuple
from collections import deque

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

import numpy as np
import pygame
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import FancyArrowPatch
import seaborn as sns

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Catmull-Rom spline from local utils
from src.utils.catmull_rom_spline_path import catmull_rom_spline

from src.rl.object_selection_env_rrt_viz import ObjectSelectionEnvRRTViz
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from src.rl.doubleDQN import DoubleDQNAgent

# Colors
BG_COLOR = (20, 20, 30)
GRID_COLOR = (60, 60, 80)
CUBE_COLOR = (100, 200, 100)
OBSTACLE_COLOR = (200, 50, 50)
AGENT_COLOR = (100, 150, 255)
PATH_COLOR = (255, 200, 0)
SELECTED_BORDER_COLOR = (0, 150, 255)
TEXT_COLOR = (220, 220, 220)
HEADER_COLOR = (100, 200, 255)  # Bright blue for headers
LABEL_COLOR = (180, 180, 200)   # Light gray for labels
VALUE_COLOR = (255, 255, 255)   # White for values
BOX_COLOR = (40, 40, 50)        # Dark box background
BOX_BORDER_COLOR = (80, 80, 100)  # Box border


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='RL + RRT Episode Visualizer')
    parser.add_argument('--grid_size', type=int, default=6, help='Grid size (default: 6)')
    parser.add_argument('--num_cubes', type=int, default=25, help='Number of cubes (default: 25)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model (optional)')
    parser.add_argument('--window_width', type=int, default=1600, help='Window width (default: 1600)')
    parser.add_argument('--window_height', type=int, default=900, help='Window height (default: 900)')
    parser.add_argument('--fps', type=int, default=30, help='FPS (default: 30)')
    return parser.parse_args()


def cubic_spline(points: List[Tuple[float, float]], num_points: int = 100) -> List[Tuple[float, float]]:
    """
    Generate smooth curve through points using Cubic Spline interpolation.

    Parameters:
        points: List of (x, y) waypoints from RRT path
        num_points: Number of interpolated points for smooth curve

    Returns:
        List of interpolated (x, y) points forming a smooth curve
    """
    if len(points) < 2:
        return points

    if len(points) == 2:
        # For 2 points, create linear interpolation (straight line with multiple points)
        x1, y1 = points[0]
        x2, y2 = points[1]
        t_vals = np.linspace(0, 1, num_points)
        interpolated = [(int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))) for t in t_vals]
        return interpolated

    import math
    import bisect

    # Extract x and y coordinates
    x_list = [p[0] for p in points]
    y_list = [p[1] for p in points]

    # Calculate cumulative distance along path (arc-length parameterization)
    dx = np.diff(x_list)
    dy = np.diff(y_list)
    ds = [math.hypot(idx, idy) for idx, idy in zip(dx, dy)]
    s = [0.0]
    s.extend(np.cumsum(ds).tolist())

    # Generate evenly spaced parameter values for smooth interpolation
    total_length = s[-1]
    if total_length == 0:
        return points

    t = np.linspace(0, total_length, num_points)

    # Cubic spline interpolation for x and y separately
    def spline_1d(s_list, val_list, t_vals):
        """
        1D cubic spline interpolation.
        Solves for cubic polynomial coefficients: f(s) = a + b*ds + c*ds^2 + d*ds^3
        """
        num = len(s_list)
        a = val_list.copy()
        b, c, d = [], [], []
        h = np.diff(s_list)

        # Build tridiagonal matrix for natural cubic spline
        A = np.zeros((num, num))
        B = np.zeros(num)

        # Natural boundary conditions: second derivative = 0 at endpoints
        A[0, 0] = 1.0
        A[num - 1, num - 1] = 1.0

        # Interior points: continuity of second derivative
        for i in range(1, num - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2.0 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            B[i] = 3.0 * ((a[i + 1] - a[i]) / h[i] - (a[i] - a[i - 1]) / h[i - 1])

        # Solve for c coefficients
        c = np.linalg.solve(A, B).tolist()

        # Calculate b and d coefficients
        for i in range(num - 1):
            d.append((c[i + 1] - c[i]) / (3.0 * h[i]))
            b.append((a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0)

        # Evaluate spline at requested parameter values
        result = []
        for it in t_vals:
            if it <= s_list[0]:
                result.append(val_list[0])
            elif it >= s_list[-1]:
                result.append(val_list[-1])
            else:
                # Find the segment containing this parameter value
                i = bisect.bisect(s_list, it) - 1
                i = min(i, num - 2)  # Ensure we don't go out of bounds
                ds = it - s_list[i]
                # Evaluate cubic polynomial
                value = a[i] + b[i] * ds + c[i] * ds**2 + d[i] * ds**3
                result.append(value)

        return result

    # Interpolate x and y coordinates separately
    path_x = spline_1d(s, x_list, t)
    path_y = spline_1d(s, y_list, t)

    # Combine into list of (x, y) points
    curve_points = [(int(x), int(y)) for x, y in zip(path_x, path_y)]

    return curve_points


class EpisodeData:
    """Stores data for one complete episode"""
    def __init__(self):
        self.actions = []
        self.paths = []
        self.tree_edges = []  # RRT tree edges for visualization
        self.rewards = []
        self.cube_positions = []
        self.obstacle_positions = []
        self.ee_start_position = None
        self.picked_cubes = []
        self.metrics = {
            'path_lengths': [],
            'obstacle_proximities': [],
            'reachability_flags': [],
            'path_clearances': [],
            'distances_to_ee': [],
            'distances_to_container': [],
            'total_reward': 0.0,
            'episode_length': 0
        }

    def add_step(self, action, path, reward, path_length, obstacle_proximity,
                 reachability, path_clearance, dist_to_ee, dist_to_container, tree_edges=None):
        """Add a step to the episode"""
        self.actions.append(action)
        self.paths.append(path)
        self.tree_edges.append(tree_edges if tree_edges else [])
        self.rewards.append(reward)
        self.metrics['path_lengths'].append(path_length)
        self.metrics['obstacle_proximities'].append(obstacle_proximity)
        self.metrics['reachability_flags'].append(1.0 if reachability else 0.0)
        self.metrics['path_clearances'].append(path_clearance)
        self.metrics['distances_to_ee'].append(dist_to_ee)
        self.metrics['distances_to_container'].append(dist_to_container)
        self.metrics['total_reward'] += reward
        self.metrics['episode_length'] += 1


class RRTRLEpisodeVisualizer:
    """Episode visualizer for RL + RRT pick-and-place"""

    def __init__(self, grid_size: int, num_cubes: int, model_path: Optional[str],
                 window_width: int, window_height: int, initial_fps: int):
        """Initialize visualizer"""
        # Configuration
        self.grid_size = grid_size
        self.num_cubes = num_cubes
        self.window_width = window_width
        self.window_height = window_height
        self.fps = initial_fps

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
        pygame.display.set_caption("Decision making + RRT Motion Planning")
        self.clock = pygame.time.Clock()

        # Use Trebuchet MS font
        try:
            self.font_header = pygame.font.SysFont("trebuchetms", 20, bold=True)
            self.font_large = pygame.font.SysFont("trebuchetms", 16, bold=True)
            self.font_medium = pygame.font.SysFont("trebuchetms", 13)
            self.font_small = pygame.font.SysFont("trebuchetms", 12)
        except:
            # Fallback to default font
            self.font_header = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 20)
            self.font_medium = pygame.font.Font(None, 16)
            self.font_small = pygame.font.Font(None, 14)

        # Load model metadata to get max_objects if model is provided
        self.model = None
        self.model_type = None  # 'ppo' or 'ddqn'
        model_max_objects = None
        model_grid_size = grid_size

        if model_path and os.path.exists(model_path):
            print(f"[MODEL] Loading model from: {model_path}")

            # Detect model type
            if model_path.endswith('.pt'):
                self.model_type = 'ddqn'
                print(f"[MODEL] Detected DDQN model (.pt)")
            else:
                self.model_type = 'ppo'
                print(f"[MODEL] Detected PPO model (.zip)")

            # Try to load metadata JSON file (simpler and more reliable)
            # Handle both _final and _step_XXXXX checkpoints
            if "_step_" in model_path:
                # Extract base name before _step_XXXXX
                base_name = model_path.rsplit("_step_", 1)[0]
                metadata_path = base_name + "_metadata.json"
            else:
                metadata_path = model_path.replace("_final.zip", "_metadata.json").replace("_final.pt", "_metadata.json")

            if os.path.exists(metadata_path):
                import json
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        model_max_objects = metadata.get("max_objects", None)
                        model_grid_size = metadata.get("training_grid_size", grid_size)
                        print(f"[MODEL] Loaded metadata: grid={model_grid_size}x{model_grid_size}, max_objects={model_max_objects}, num_cubes={metadata.get('num_cubes', num_cubes)}")

                        # Warn if mismatch
                        if model_grid_size != grid_size:
                            print(f"[WARNING] Model trained on {model_grid_size}x{model_grid_size} grid, but running with {grid_size}x{grid_size}")
                            print(f"[INFO] Using model's grid size: {model_grid_size}x{model_grid_size}")
                            grid_size = model_grid_size
                            self.grid_size = grid_size
                except Exception as e:
                    print(f"[WARNING] Could not read metadata file: {e}")
            else:
                print(f"[WARNING] Metadata file not found: {metadata_path}")

            # If metadata didn't provide max_objects, extract from model zip file
            if model_max_objects is None:
                print(f"[INFO] Extracting max_objects from model zip file...")
                import zipfile
                import json
                try:
                    with zipfile.ZipFile(model_path, 'r') as archive:
                        data_json = archive.read('data').decode('utf-8')
                        data = json.loads(data_json)

                        # Try different ways to extract observation space shape
                        if 'observation_space' in data:
                            obs_space = data['observation_space']
                            if isinstance(obs_space, dict):
                                if 'shape' in obs_space:
                                    obs_space_shape = obs_space['shape'][0]
                                elif '_shape' in obs_space:
                                    obs_space_shape = obs_space['_shape'][0]
                                else:
                                    print(f"[WARNING] observation_space keys: {obs_space.keys()}")
                                    raise KeyError("Cannot find shape in observation_space")
                            else:
                                obs_space_shape = obs_space[0]
                        else:
                            raise KeyError("observation_space not found in model data")

                        model_max_objects = obs_space_shape // 6  # 6 features per object
                        print(f"[MODEL] Extracted from zip: {obs_space_shape} dims = {model_max_objects} objects × 6 features")
                except Exception as e:
                    print(f"[ERROR] Could not extract max_objects from model zip: {e}")
                    print(f"[ERROR] Cannot load model without knowing max_objects!")
                    print(f"[INFO] Running in greedy baseline mode (no model)")
                    model_path = None  # Disable model loading

        # Create environment with correct max_objects (using PythonRobotics RRT for visualization)
        max_objects = model_max_objects if model_max_objects else num_cubes
        print(f"[ENV] Creating environment with max_objects={max_objects}, num_cubes={num_cubes}, grid={grid_size}x{grid_size}")

        base_env = ObjectSelectionEnvRRTViz(
            franka_controller=None,
            max_objects=max_objects,  # Use model's max_objects
            max_steps=num_cubes * 2,
            num_cubes=num_cubes,
            render_mode=None,
            dynamic_obstacles=False,
            training_grid_size=grid_size,
            execute_picks=False
        )
        # Keep reference to unwrapped environment for direct access
        self.env = base_env

        # Wrap with ActionMasker for action masking support
        def mask_fn(env):
            return env.action_masks()
        self.wrapped_env = ActionMasker(base_env, mask_fn)
        vec_env = DummyVecEnv([lambda: self.wrapped_env])
        self.vec_env = vec_env

        # Load model with matching environment
        if model_path and os.path.exists(model_path):
            try:
                if self.model_type == 'ddqn':
                    # Load DDQN model
                    checkpoint = torch.load(model_path, map_location='cpu')
                    state_dim = checkpoint['state_dim']
                    action_dim = checkpoint['action_dim']

                    self.model = DoubleDQNAgent(
                        state_dim=state_dim,
                        action_dim=action_dim,
                        gamma=checkpoint['gamma'],
                        epsilon_start=0.0,  # No exploration during visualization
                        epsilon_end=0.0,
                        epsilon_decay=1.0,
                        batch_size=checkpoint['batch_size'],
                        target_update_freq=checkpoint['target_update_freq']
                    )
                    self.model.load(model_path)
                    self.model.epsilon = 0.0  # Ensure greedy action selection
                    print(f"[MODEL] DDQN model loaded successfully!")
                else:
                    # Load PPO model
                    self.model = MaskablePPO.load(model_path, env=vec_env)
                    print(f"[MODEL] PPO model loaded successfully!")
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
                print(f"[INFO] Running in greedy baseline mode instead")
                self.model = None
                self.model_type = None
        else:
            if model_path:
                print(f"[WARNING] Model file not found: {model_path}")
            print(f"[INFO] Running in greedy baseline mode (no model)")

        # Episode state
        self.current_episode = None
        self.episode_step = 0
        self.episode_count = 0
        self.episode_history = deque(maxlen=50)
        self.chart_fig = None

        # Visualization state
        self.selected_cube_idx = None
        self.current_path = None
        self.rrt_tree = []  # Track RRT tree nodes for visualization
        self.static_obstacles = []
        self.running = True
        self.paused = True
        self.playback_speed = 1.0
        self.auto_advance = True

        # Animation state with phases
        # Phase 1: Show selected cube (blue border)
        # Phase 2: Draw path animation (path grows towards target)
        # Phase 3: Cube disappears, wait before next step
        self.animation_phase = 0  # 0=idle, 1=selection, 2=path_animation, 3=post_animation_delay
        self.phase_timer = 0
        self.selection_delay = 30  # frames to show selected cube before path
        self.path_progress = 0.0  # 0.0 to 1.0 for animated path drawing
        self.path_animation_duration = 90  # frames for path drawing animation
        self.post_animation_delay = 30  # frames to wait after path before cube disappears

        # Reward components for current step (calculated from observation values)
        self.current_reward_components = {
            'r_total': 0.0,
            'r_path': 0.0,           # Path/distance reward (max 5.0)
            'r_container': 0.0,      # Container distance reward (max 3.0)
            'r_obstacle': 0.0,       # Obstacle proximity reward (max 3.0)
            'r_reachability': 0.0,   # Reachability penalty (0.0 or -5.0)
            'r_clearance': 0.0       # Path clearance reward (max 2.0)
        }

        # Current observation components
        self.current_obs_components = {
            'dist_to_ee': 0.0,
            'dist_to_container': 0.0,
            'obstacle_proximity': 0.0,
            'reachability': 0.0,
            'path_clearance': 0.0,
            'items_left': 0,
            'dist_to_origin': 0.0
        }

        # Reward history for spike graph
        self.reward_history = []

        # Episode timer
        self.episode_start_time = 0
        self.episode_elapsed_time = 0  # in seconds
        self.timer_started = False  # Flag to track if timer has started

        # Graph caching for performance
        self.cached_reward_graph = None
        self.cached_reward_history_len = 0
        self.cached_rrt_graph = None
        self.cached_rrt_progress = -1.0
        self.cached_rrt_phase = -1
        self.cached_rrt_paths_len = 0
        self.last_graph_step = -1

        # Cache for arrow calculations to avoid recalculating every frame
        self.cached_arrow_data = None
        self.cached_path_progress = -1.0

        # Accumulated paths for RRT graph (all paths from all steps in current episode)
        self.accumulated_paths = []  # List of (path, tree_edges, target_pos) tuples

        # Episode generation state
        self.generating_episode = False
        self.episode_ready = False

        # Don't generate episode in __init__ - do it on first frame
        # This allows window to show immediately
        print("\nInitializing visualizer...")
        print("Visualizer ready!")

    def generate_new_episode(self):
        """Generate a new episode by running the RL agent"""
        self.generating_episode = True
        self.episode_ready = False
        print(f"\nGenerating episode {self.episode_count + 1}...")

        # Reset environment (use wrapped env for reset/step)
        obs, info = self.wrapped_env.reset()

        # Reset episode timer (will start when Step 1 begins)
        self.episode_start_time = 0
        self.episode_elapsed_time = 0
        self.timer_started = False

        # Generate random obstacles (1-3, but only if there's room)
        grid_capacity = self.grid_size * self.grid_size
        available_cells = grid_capacity - self.num_cubes - 1  # -1 for EE home cell
        max_obstacles = max(0, min(3, available_cells))
        min_obstacles = 1 if max_obstacles > 0 else 0
        num_obstacles = np.random.randint(min_obstacles, max_obstacles + 1) if max_obstacles > 0 else 0
        self._add_random_obstacles(num_obstacles)

        print(f"  {self.num_cubes} cubes, {len(self.static_obstacles)} obstacles")

        # Create episode data
        episode = EpisodeData()
        episode.cube_positions = self.env.object_positions[:self.env.total_objects].copy()
        episode.obstacle_positions = [self.env.rrt_estimator._grid_to_world(ox, oy)
                                     for ox, oy in self.static_obstacles]
        episode.ee_start_position = self.env.ee_position.copy()

        # Run episode
        done = False
        step = 0
        successful_picks = 0
        failed_picks = 0

        while not done and step < self.env.max_steps:
            # Get action from model or greedy baseline
            if self.model is not None:
                # Get action mask (prevent selecting already-picked objects)
                action_mask = self.wrapped_env.action_masks()

                if self.model_type == 'ddqn':
                    # DDQN: flatten observation and use policy network
                    obs_flat = obs.flatten()
                    obs_tensor = torch.FloatTensor(obs_flat).to(self.model.device)
                    action = self.model.policy_net.get_action(obs_tensor, epsilon=0.0, action_mask=action_mask)
                else:
                    # PPO: use predict method
                    action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
                    action = int(action)
            else:
                action = self._greedy_action()

            # Update RRT grid with current unpicked cubes as obstacles
            # (exclude the target cube we're planning to pick)
            self._update_rrt_grid_for_planning(action)

            # Get RRT path before taking action (with tree edges for visualization)
            path, tree_edges = self._get_rrt_path(action, return_tree=True)

            # Track success/failure based on RRT result
            path_points = len(path) if path else 0
            tree_size = len(tree_edges) if tree_edges else 0

            if path and len(path) > 0:
                successful_picks += 1
                print(f"  Step {step+1}: Cube {action}, Path points: {path_points}, Tree edges: {tree_size} ✓")
            else:
                failed_picks += 1
                print(f"  Step {step+1}: Cube {action}, Path points: {path_points}, Tree edges: {tree_size} ✗ RRT FAILED")

            # Calculate metrics
            path_length = self._calculate_path_length(path) if path else 999.0
            obstacle_proximity = self._calculate_obstacle_proximity(action)
            dist_to_ee = np.linalg.norm(self.env.object_positions[action][:2] - self.env.ee_position[:2])
            # Reachability based on distance (0.3m to 0.9m range, matches environment)
            reachability = 1.0 if (0.3 <= dist_to_ee <= 0.9) else 0.0
            path_clearance = self._calculate_path_clearance(path) if path else 0.0
            dist_to_container = np.linalg.norm(self.env.object_positions[action][:2] - self.env.container_position[:2])

            # Take action (use wrapped env for step)
            obs, reward, terminated, truncated, info = self.wrapped_env.step(action)
            done = terminated or truncated

            # NOTE: EE position stays at home position (not updated)
            # Path planning always starts from same EE position for all picks

            # Record step (including tree edges for visualization)
            episode.add_step(action, path, reward, path_length, obstacle_proximity,
                           reachability, path_clearance, dist_to_ee, dist_to_container, tree_edges)
            episode.picked_cubes.append(action)

            step += 1

        self.current_episode = episode
        self.episode_step = 0
        self.episode_count += 1
        self.episode_history.append(episode)

        # Clear graph cache for new episode
        self.cached_reward_graph = None
        self.cached_reward_history_len = 0
        self.cached_rrt_graph = None
        self.cached_rrt_progress = -1.0
        self.cached_rrt_phase = -1
        self.cached_rrt_paths_len = 0
        self.last_graph_step = -1

        # Clear accumulated paths for new episode
        self.accumulated_paths = []

        # Mark episode as ready
        self.generating_episode = False
        self.episode_ready = True

        # Print accurate summary (without reward)
        print(f"Done! ({successful_picks} successful, {failed_picks} failed)")

    def _add_random_obstacles(self, num_obstacles: int):
        """Add random obstacles to empty grid cells"""
        self.static_obstacles = []

        if num_obstacles == 0:
            return

        # Get EE home position to avoid
        ee_grid_x, ee_grid_y = self.env.rrt_estimator._world_to_grid(self.env.ee_position[:2])

        # Get cube positions to avoid
        cube_cells = set()
        cube_cells.add((ee_grid_x, ee_grid_y))  # Exclude EE home cell

        for i in range(self.env.total_objects):
            pos = self.env.object_positions[i]
            grid_col, grid_row = self.env.rrt_estimator._world_to_grid(pos[:2])
            cube_cells.add((grid_col, grid_row))

        # Get all empty cells
        empty_cells = []
        for grid_x in range(self.grid_size):
            for grid_y in range(self.grid_size):
                if (grid_x, grid_y) not in cube_cells:
                    empty_cells.append((grid_x, grid_y))

        # Randomly select obstacle positions
        if len(empty_cells) >= num_obstacles:
            np.random.shuffle(empty_cells)
            selected_cells = empty_cells[:num_obstacles]

            for grid_x, grid_y in selected_cells:
                self.static_obstacles.append((grid_x, grid_y))

    def _update_rrt_grid_for_planning(self, target_cube_idx: int):
        """
        Update RRT occupancy grid with current obstacles.

        Obstacles include:
        1. Static obstacles (randomly placed)
        2. Unpicked cubes (EXCEPT the target cube we're planning to pick)

        Args:
            target_cube_idx: Index of the cube we're planning to pick (not treated as obstacle)
        """
        # Get static obstacle positions
        obstacle_positions = []
        for grid_x, grid_y in self.static_obstacles:
            world_pos = self.env.rrt_estimator._grid_to_world(grid_x, grid_y)
            obstacle_positions.append(np.array([world_pos[0], world_pos[1], 0.0]))

        # Get unpicked cube positions (excluding target cube)
        unpicked_cube_positions = []
        for i in range(self.env.total_objects):
            if i not in self.env.objects_picked and i != target_cube_idx:
                unpicked_cube_positions.append(self.env.object_positions[i])

        # Update RRT grid
        self.env.rrt_estimator.update_occupancy_grid(
            object_positions=unpicked_cube_positions,
            obstacle_positions=obstacle_positions
        )

    def _greedy_action(self) -> int:
        """
        Greedy baseline: pick closest unpicked cube using RRT path length.

        This ensures we pick cubes that are actually reachable, not just physically close.
        """
        min_path_length = float('inf')
        best_action = 0

        for i in range(self.env.total_objects):
            if i not in self.env.objects_picked:
                # Update grid for this candidate (exclude this cube from obstacles)
                self._update_rrt_grid_for_planning(i)

                # Use RRT path length instead of Euclidean distance
                # This accounts for obstacles and ensures we pick reachable cubes
                path_length = self.env.rrt_estimator.estimate_path_length(
                    self.env.ee_position[:2],
                    self.env.object_positions[i][:2]
                )

                if path_length < min_path_length:
                    min_path_length = path_length
                    best_action = i

        return best_action

    def _get_rrt_path(self, cube_idx: int, skip_grid_update: bool = False, return_tree: bool = False) -> tuple:
        """
        Get RRT path from end-effector to cube.

        Args:
            cube_idx: Index of target cube
            skip_grid_update: If True, skip grid update (caller already updated grid)
            return_tree: If True, also return tree edges for visualization

        Returns:
            If return_tree=False: path (list of tuples)
            If return_tree=True: (path, tree_edges)
        """
        if cube_idx >= self.env.total_objects:
            return (None, None) if return_tree else None

        ee_grid = self.env.rrt_estimator._world_to_grid(self.env.ee_position[:2])
        cube_pos = self.env.object_positions[cube_idx]
        goal_grid = self.env.rrt_estimator._world_to_grid(cube_pos[:2])

        # Check if start == goal
        if ee_grid == goal_grid:
            return ([], []) if return_tree else []

        # Update RRT grid for this specific target (unless caller already did it)
        if not skip_grid_update:
            self._update_rrt_grid_for_planning(cube_idx)

        # Run RRT search (using PythonRobotics planning method)
        if return_tree:
            rx, ry, tree_edges = self.env.rrt_estimator.planning(ee_grid[0], ee_grid[1], goal_grid[0], goal_grid[1], return_tree=True)
        else:
            rx, ry = self.env.rrt_estimator.planning(ee_grid[0], ee_grid[1], goal_grid[0], goal_grid[1], return_tree=False)
            tree_edges = None

        # Convert to list of tuples
        # NOTE: RRT returns path from START to GOAL (already in correct order)
        if rx is None or ry is None:
            return ([], []) if return_tree else []
        else:
            path = [(rx[i], ry[i]) for i in range(len(rx))]
            return (path, tree_edges) if return_tree else path

    def _get_astar_fallback_path(self, cube_idx: int, ee_grid: Tuple[int, int], goal_grid: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Fallback to A* when RRT fails"""
        from src.rl.path_estimators import AStarPathEstimator

        # Create A* estimator with same grid parameters
        astar = AStarPathEstimator(
            grid_size=self.grid_size,
            cell_size=0.26 if self.grid_size > 3 else 0.28
        )

        # Update A* grid with same obstacles as RRT
        obstacle_positions = []
        object_positions = []

        for i in range(self.env.total_objects):
            if i != cube_idx and i not in self.env.objects_picked:
                pos = self.env.object_positions[i]
                object_positions.append(pos)

        # Add static obstacles
        for obs_grid_x, obs_grid_y in self.static_obstacles:
            # Convert grid to world coordinates
            cell_size = 0.26 if self.grid_size > 3 else 0.28
            grid_center = np.array([0.45, -0.10])
            grid_extent = (self.grid_size - 1) * cell_size
            start_x = grid_center[0] - (grid_extent / 2.0)
            start_y = grid_center[1] - (grid_extent / 2.0)

            world_x = start_x + (obs_grid_x * cell_size)
            world_y = start_y + (obs_grid_y * cell_size)
            world_z = 0.055
            obstacle_positions.append(np.array([world_x, world_y, world_z]))

        # Update A* grid
        astar.update_grid(object_positions, obstacle_positions)

        # Run A* search
        path = astar.find_path(ee_grid[0], ee_grid[1], goal_grid[0], goal_grid[1])

        return path

    def _calculate_path_length(self, path: List[Tuple[int, int]]) -> float:
        """Calculate path length in meters"""
        if not path or len(path) < 2:
            return 0.0

        length = 0.0
        for i in range(len(path) - 1):
            pos1 = self.env.rrt_estimator._grid_to_world(path[i][0], path[i][1])
            pos2 = self.env.rrt_estimator._grid_to_world(path[i+1][0], path[i+1][1])
            length += np.linalg.norm(pos2 - pos1)

        return length

    def _calculate_obstacle_proximity(self, cube_idx: int) -> float:
        """Calculate minimum distance to obstacles"""
        if not self.static_obstacles:
            return 999.0

        cube_pos = self.env.object_positions[cube_idx][:2]
        min_dist = float('inf')

        for ox, oy in self.static_obstacles:
            obs_pos = self.env.rrt_estimator._grid_to_world(ox, oy)
            dist = np.linalg.norm(cube_pos - obs_pos)
            min_dist = min(min_dist, dist)

        return min_dist

    def _calculate_path_clearance(self, path: List[Tuple[int, int]]) -> float:
        """Calculate minimum clearance from path to obstacles"""
        if not path or not self.static_obstacles:
            return 999.0

        min_clearance = float('inf')

        for px, py in path:
            for ox, oy in self.static_obstacles:
                dist = np.sqrt((px - ox)**2 + (py - oy)**2)
                min_clearance = min(min_clearance, dist)

        return min_clearance

    def calculate_reward_components(self, path_length, dist_to_container, obstacle_proximity,
                                    reachability, path_clearance, dist_to_ee):
        """
        Calculate individual reward components from observation values.
        Matches the reward calculation in ObjectSelectionEnvRRT._calculate_reward()

        Args:
            path_length: RRT path length in meters
            dist_to_container: Distance to container in meters
            obstacle_proximity: Obstacle score (0.0 = far, 1.0 = very close)
            reachability: Reachability flag (1.0 = reachable, 0.0 = unreachable)
            path_clearance: Path clearance score (0.0 = blocked, 1.0 = clear)
            dist_to_ee: Euclidean distance to EE in meters

        Returns:
            Dictionary with individual reward components
        """
        components = {}

        # 1. Path length reward (max 5.0)
        # Check if RRT planning failed (path_length >= 2.0 × Euclidean)
        euclidean_distance = dist_to_ee
        planning_failed = (path_length >= 2.0 * euclidean_distance)

        # Normalize by typical path length (0.3m to 0.9m)
        normalized_path_length = (path_length - 0.3) / 0.6
        normalized_path_length = np.clip(normalized_path_length, 0.0, 1.0)
        path_reward = 5.0 * (1.0 - normalized_path_length)

        # Subtract failure penalty if planning failed
        if planning_failed:
            path_reward -= 5.0

        components['r_path'] = path_reward

        # 2. Container distance reward (max 3.0)
        container_reward = 3.0 * np.exp(-dist_to_container)
        components['r_container'] = container_reward

        # 3. Obstacle proximity reward (max 3.0)
        obstacle_reward = 3.0 * (1.0 - obstacle_proximity)
        components['r_obstacle'] = obstacle_reward

        # 4. Reachability penalty (0.0 or -10.0)
        # reachability flag: 1.0 = reachable (0.3m to 0.9m), 0.0 = unreachable
        reachability_penalty = 0.0 if reachability >= 0.5 else -10.0
        components['r_reachability'] = reachability_penalty

        # 5. Path clearance reward (max 2.0)
        clearance_reward = 2.0 * path_clearance
        components['r_clearance'] = clearance_reward

        return components

    def create_reward_graph_surface(self, width, height):
        """Create reward gradient area graph using matplotlib (optimized with caching)"""
        if len(self.reward_history) < 2:
            # Return empty surface
            surf = pygame.Surface((width, height))
            surf.fill(BOX_COLOR)
            return surf

        # Check if we can use cached graph (only regenerate when reward history changes)
        if (self.cached_reward_graph is not None and
            self.cached_reward_history_len == len(self.reward_history)):
            return self.cached_reward_graph

        # Create matplotlib figure with adjusted layout to prevent label cutoff
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100, facecolor='#282828')
        ax.set_facecolor('#282828')

        # Create gradient area chart
        steps = np.array(range(1, len(self.reward_history) + 1))
        rewards = np.array(self.reward_history)

        # Normalize rewards to 0-1 range for colormap
        min_reward = min(rewards)

        # Shift rewards to be positive for gradient effect
        rewards_shifted = rewards - min_reward + 0.1  # Add small offset to avoid zero

        # Choose colormap - Blues for a nice gradient effect
        cmap = plt.get_cmap('Blues')
        n_levels = 60  # Reduced from 80 for better performance

        # Create layered gradient fill
        for i in range(n_levels):
            alpha_level = (i + 1) / n_levels
            y_level = rewards_shifted * alpha_level
            ax.fill_between(steps, 0, y_level, color=cmap(alpha_level), alpha=0.05)

        # Styling - NO GRID
        ax.set_xlabel('Step', color='white', fontsize=7)
        ax.set_ylabel('Reward', color='white', fontsize=7)
        ax.set_title('Reward Vs Step', color='#64C8FF', fontsize=9, fontweight='bold')
        ax.tick_params(colors='white', labelsize=6)
        ax.grid(False)  # No grid
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Format tick labels as integers
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Adjust y-axis to show actual reward values (not shifted) as integers
        y_ticks = ax.get_yticks()
        ax.set_yticklabels([f'{int(y + min_reward - 0.1)}' for y in y_ticks])

        # Set x-axis to start from 1 (steps start from 1, but xlim starts from 0.5 to fill from y-axis)
        ax.set_xlim(0.5, len(self.reward_history) + 0.5)

        # Adjust layout to prevent label cutoff - more space for title and bottom label
        plt.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.18)

        # Convert to pygame surface (fix for newer matplotlib)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        # Get the RGBA buffer and convert to RGB
        buf = canvas.buffer_rgba()
        size = canvas.get_width_height()

        # Create surface from RGBA buffer
        surf = pygame.image.frombuffer(buf, size, "RGBA")
        plt.close(fig)

        # Cache the surface
        self.cached_reward_graph = surf
        self.cached_reward_history_len = len(self.reward_history)

        return surf

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self.running = False

                elif event.key == pygame.K_SPACE:
                    was_paused = self.paused
                    self.paused = not self.paused
                    # Reset timer when resuming to prevent skipping animations
                    if was_paused and not self.paused:
                        self.phase_timer = 0

                elif event.key == pygame.K_n:
                    # Next step
                    if self.episode_step < len(self.current_episode.actions):
                        self.episode_step += 1
                        self.animating = True
                        self.animation_progress = 0.0

                elif event.key == pygame.K_r:
                    # Reset episode
                    self.episode_step = 0
                    self.animating = False

                elif event.key == pygame.K_g:
                    # Generate new episode
                    self.generate_new_episode()

                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.playback_speed = min(5.0, self.playback_speed + 0.5)

                elif event.key == pygame.K_MINUS:
                    self.playback_speed = max(0.5, self.playback_speed - 0.5)

            elif event.type == pygame.VIDEORESIZE:
                self.window_width = event.w
                self.window_height = event.h
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

    def draw_grid(self, cell_size, grid_x, grid_y):
        """Draw grid"""
        for row in range(self.grid_size + 1):
            y = grid_y + row * cell_size
            pygame.draw.line(self.screen, GRID_COLOR,
                           (grid_x, y), (grid_x + self.grid_size * cell_size, y), 2)

        for col in range(self.grid_size + 1):
            x = grid_x + col * cell_size
            pygame.draw.line(self.screen, GRID_COLOR,
                           (x, grid_y), (x, grid_y + self.grid_size * cell_size), 2)

    def draw_obstacles(self, cell_size, grid_x, grid_y):
        """Draw obstacles as rectangles (half width, slightly reduced height)"""
        for ox, oy in self.static_obstacles:
            # Center position in cell
            center_x = grid_x + ox * cell_size + cell_size // 2
            center_y = grid_y + oy * cell_size + cell_size // 2

            # Half width, slightly reduced height
            width = cell_size // 2
            height = int(cell_size * 0.85)

            # Draw centered rectangle
            pygame.draw.rect(self.screen, OBSTACLE_COLOR,
                           (center_x - width // 2, center_y - height // 2, width, height))

    def draw_cubes(self, cell_size, grid_x, grid_y):
        """Draw cubes - only unpicked cubes, selected cube has blue border"""
        if not self.current_episode:
            return

        # Determine which cubes to hide based on animation phase and step
        # episode_step tracks how many cubes have been COMPLETED (picked and removed)
        #
        # Phase 0 (idle): All completed cubes are hidden
        # Phase 1 (selection): Show next cube to pick with blue border
        # Phase 2 (path animation): Show cube with blue border + path
        # Phase 3 (disappearing): Show cube + path, then both disappear together

        # Hide all cubes that have been completed (picked and removed)
        picked_cubes = set(self.current_episode.picked_cubes[:self.episode_step])

        # Get EE grid position to avoid drawing cubes at same location
        ee_col, ee_row = self.env.rrt_estimator._world_to_grid(self.env.ee_position[:2])

        for i in range(len(self.current_episode.cube_positions)):
            # Skip picked cubes
            if i in picked_cubes:
                continue

            pos = self.current_episode.cube_positions[i]
            grid_col, grid_row = self.env.rrt_estimator._world_to_grid(pos[:2])

            # Skip cubes at EE position (they're hidden under the EE)
            if grid_col == ee_col and grid_row == ee_row:
                continue

            if 0 <= grid_col < self.grid_size and 0 <= grid_row < self.grid_size:
                x = grid_x + grid_col * cell_size + cell_size // 2
                y = grid_y + grid_row * cell_size + cell_size // 2

                # Draw cube
                size = cell_size // 2

                # Hide cube during phase 3 if it's the one being picked (disappearing)
                if i == self.selected_cube_idx and self.animation_phase == 3:
                    continue  # Skip drawing - cube disappears in phase 3

                pygame.draw.rect(self.screen, CUBE_COLOR,
                               (x - size // 2, y - size // 2, size, size))

                # Highlight selected cube with BLUE border during phases 1 and 2
                if i == self.selected_cube_idx and self.animation_phase in [1, 2]:
                    border_width = 4
                    pygame.draw.rect(self.screen, SELECTED_BORDER_COLOR,
                                   (x - size // 2 - border_width, y - size // 2 - border_width,
                                    size + border_width * 2, size + border_width * 2), border_width)
                    # Draw blue circle at center of target cube
                    pygame.draw.circle(self.screen, SELECTED_BORDER_COLOR, (x, y), 5)

    def draw_agent(self, cell_size, grid_x, grid_y, grid_h):
        """Draw agent (end-effector) at its actual grid cell position"""
        # Get EE grid position from world coordinates (same as used for obstacle exclusion)
        ee_col, ee_row = self.env.rrt_estimator._world_to_grid(self.env.ee_position[:2])

        # Position EE at the CENTER of its grid cell
        agent_x = grid_x + ee_col * cell_size + cell_size // 2
        agent_y = grid_y + ee_row * cell_size + cell_size // 2

        # Draw blue circle for EE
        radius = int(cell_size * 0.3)  # Larger icon to be visible in cell
        pygame.draw.circle(self.screen, AGENT_COLOR, (agent_x, agent_y), radius)
        pygame.draw.circle(self.screen, AGENT_COLOR, (agent_x, agent_y), radius, 2)

        # Draw label
        label = self.font_small.render("EE", True, TEXT_COLOR)
        label_rect = label.get_rect(center=(agent_x, agent_y))
        self.screen.blit(label, label_rect)

        return agent_x, agent_y, radius

    def draw_path(self, cell_size, grid_x, grid_y, grid_h, agent_pos):
        """Draw RRT path using Catmull-Rom Spline interpolation"""
        # Draw path ONLY during phase 2 (path disappears in phase 3 along with cube)
        if self.animation_phase != 2:
            return

        # Don't draw if path is empty or invalid
        if self.current_path is None or len(self.current_path) == 0:
            return

        # Don't draw if path has only 1 waypoint (start == goal)
        if len(self.current_path) == 1:
            return

        agent_x, agent_y, agent_radius = agent_pos

        # Build path points from RRT grid coordinates
        points = []
        points.append((agent_x, agent_y))  # Start from agent position

        # Add all RRT waypoints EXCEPT the first one (EE position)
        for i, (grid_col, grid_row) in enumerate(self.current_path):
            if i == 0:
                continue  # Skip first waypoint (EE position)
            x = grid_x + grid_col * cell_size + cell_size // 2
            y = grid_y + grid_row * cell_size + cell_size // 2
            points.append((x, y))

        # If we only have 1 point (agent position), don't draw
        if len(points) < 2:
            return

        # Generate smooth curved path using Catmull-Rom Spline interpolation
        if len(points) >= 2:
            # Calculate adaptive number of interpolation points based on path length
            total_dist = sum(np.sqrt((points[i+1][0] - points[i][0])**2 +
                                    (points[i+1][1] - points[i][1])**2)
                           for i in range(len(points)-1))
            num_interp_points = max(50, int(total_dist / 2))

            # Catmull-Rom spline requires at least 3 points
            if len(points) == 2:
                # Linear interpolation for straight line
                x1, y1 = points[0]
                x2, y2 = points[1]
                t_vals = np.linspace(0, 1, num_interp_points)
                curved_points = [(int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))) for t in t_vals]
            else:
                # Use Catmull-Rom spline from PythonRobotics (requires 3+ points)
                spline_x, spline_y = catmull_rom_spline(points, num_interp_points)
                curved_points = [(int(x), int(y)) for x, y in zip(spline_x, spline_y)]

            # Draw curved path with animation (path grows towards target)
            if len(curved_points) >= 2:
                # Calculate how many points to draw based on animation progress
                num_points_to_draw = max(2, int(len(curved_points) * self.path_progress))
                animated_points = curved_points[:num_points_to_draw]

                if len(animated_points) >= 2:
                    # Draw red filled circle at EE (start of path)
                    start_x, start_y = animated_points[0]
                    pygame.draw.circle(self.screen, (255, 0, 0), (int(start_x), int(start_y)), 5)

                    # Draw the main path curve
                    pygame.draw.lines(self.screen, PATH_COLOR, False, animated_points, 3)

                    # Calculate and draw arrow head (optimized with caching)
                    # Only recalculate if path progress changed significantly (> 1%)
                    if (self.cached_arrow_data is None or
                        abs(self.path_progress - self.cached_path_progress) > 0.01):

                        arrow_tip = None
                        arrow_left = None
                        arrow_right = None

                        if len(animated_points) >= 2:
                            # Get the last point
                            end_x, end_y = animated_points[-1]

                            # Get previous point for direction
                            # Look back further for more stable direction, but ensure we don't go out of bounds
                            lookback = min(10, len(animated_points) - 1)
                            prev_x, prev_y = animated_points[-1 - lookback]

                            # Calculate direction vector
                            dx = end_x - prev_x
                            dy = end_y - prev_y
                            length = np.sqrt(dx**2 + dy**2)

                            if length > 1.0:  # Reduced threshold from 2.0 to 1.0 to show arrow more often
                                # Normalize direction
                                dx /= length
                                dy /= length

                                # Arrow head parameters (smaller)
                                arrow_length = 6
                                arrow_width = 4

                                # Calculate arrow head points
                                arrow_tip = (int(end_x), int(end_y))
                                base_x = end_x - dx * arrow_length
                                base_y = end_y - dy * arrow_length
                                perp_x = -dy
                                perp_y = dx
                                arrow_left = (int(base_x + perp_x * arrow_width), int(base_y + perp_y * arrow_width))
                                arrow_right = (int(base_x - perp_x * arrow_width), int(base_y - perp_y * arrow_width))

                        # Cache the arrow data
                        self.cached_arrow_data = (arrow_tip, arrow_left, arrow_right)
                        self.cached_path_progress = self.path_progress

                    # Draw arrow head using cached data
                    if self.cached_arrow_data:
                        arrow_tip, arrow_left, arrow_right = self.cached_arrow_data
                        if arrow_tip and arrow_left and arrow_right:
                            pygame.draw.line(self.screen, PATH_COLOR, arrow_tip, arrow_left, 2)
                            pygame.draw.line(self.screen, PATH_COLOR, arrow_tip, arrow_right, 2)

    def draw_box(self, x, y, width, height, title=None):
        """Draw a styled box with optional title"""
        # Draw box background
        pygame.draw.rect(self.screen, BOX_COLOR, (x, y, width, height))
        # Draw box border
        pygame.draw.rect(self.screen, BOX_BORDER_COLOR, (x, y, width, height), 2)

        # Draw title if provided
        if title:
            title_surf = self.font_large.render(title, True, HEADER_COLOR)
            self.screen.blit(title_surf, (x + 10, y + 8))
            return y + 35  # Return y position after title
        return y + 10

    def draw_info_panel(self, x, y, width, height):
        """Draw comprehensive info panel on right side"""
        if not self.current_episode:
            return

        current_y = y

        # Header
        header_surf = self.font_header.render("Decision making + RRT motion planning", True, HEADER_COLOR)
        self.screen.blit(header_surf, (x, current_y))
        current_y += 40

        # Episode Info Box
        box_y = self.draw_box(x, current_y, width, 195, "Episode Info")
        current_reward = sum(self.current_episode.rewards[:self.episode_step]) if self.episode_step > 0 else 0.0

        # Get distance to target cube (Dist->Target)
        dist_to_target = 0.0
        if self.selected_cube_idx is not None and self.selected_cube_idx < len(self.env.object_positions):
            target_pos = self.env.object_positions[self.selected_cube_idx][:2]
            dist_to_target = np.linalg.norm(target_pos - self.env.ee_position[:2])

        # Get playback status
        status_text = "Paused" if self.paused else "Playing"

        # Format episode timer as HH:MM:SS
        hours = int(self.episode_elapsed_time // 3600)
        minutes = int((self.episode_elapsed_time % 3600) // 60)
        seconds = int(self.episode_elapsed_time % 60)
        timer_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        info_items = [
            ("Episode:", f"{self.episode_count}"),
            ("Step:", f"{self.episode_step}/{len(self.current_episode.actions)}"),
            ("FPS:", f"{int(self.clock.get_fps())}"),
            ("Items left:", f"{len(self.current_episode.actions) - self.episode_step}"),
            ("Dist->Target:", f"{dist_to_target:.3f}"),
            ("Total Time:", timer_text),
            ("Status:", status_text),
        ]

        for label, value in info_items:
            label_surf = self.font_small.render(label, True, LABEL_COLOR)
            value_surf = self.font_small.render(value, True, VALUE_COLOR)
            self.screen.blit(label_surf, (x + 15, box_y))
            self.screen.blit(value_surf, (x + 200, box_y))
            box_y += 20

        current_y += 205

        # Cumulative Rewards Box
        box_y = self.draw_box(x, current_y, width, 70, "Cumulative Rewards")
        total_label = self.font_large.render("Total:", True, LABEL_COLOR)
        total_value = self.font_large.render(f"{current_reward:.2f}", True, HEADER_COLOR)
        self.screen.blit(total_label, (x + 15, box_y))
        self.screen.blit(total_value, (x + 200, box_y))

        current_y += 80

        # Performance Metrics Box
        box_y = self.draw_box(x, current_y, width, 160, "Performance Metrics")

        reward_items = [
            ("Distance to EE reward:", f"{self.current_reward_components['r_path']:.3f}"),
            ("Distance to container reward:", f"{self.current_reward_components['r_container']:.3f}"),
            ("Obstacle proximity reward:", f"{self.current_reward_components['r_obstacle']:.3f}"),
            ("Reachability penalty:", f"{self.current_reward_components['r_reachability']:.3f}"),
            ("Path clearance reward:", f"{self.current_reward_components['r_clearance']:.3f}"),
        ]

        for label, value in reward_items:
            label_surf = self.font_small.render(label, True, LABEL_COLOR)
            value_surf = self.font_small.render(value, True, VALUE_COLOR)
            self.screen.blit(label_surf, (x + 15, box_y))
            self.screen.blit(value_surf, (x + 250, box_y))
            box_y += 20

    def draw_reward_graph(self, x, y, width, height):
        """Draw reward spike graph at bottom of right panel with progress bar"""
        # Draw progress bar above graph (moved up more)
        progress_bar_height = 28
        progress_bar_y = y - progress_bar_height - 30

        # Progress bar width should match Observation Metrics box width (420)
        progress_bar_width = 420

        if self.current_episode:
            total_steps = len(self.current_episode.actions)
            current_step = self.episode_step
            progress = current_step / total_steps if total_steps > 0 else 0.0

            # Draw progress bar background (white)
            pygame.draw.rect(self.screen, (255, 255, 255), (x, progress_bar_y, progress_bar_width, progress_bar_height))
            pygame.draw.rect(self.screen, BOX_BORDER_COLOR, (x, progress_bar_y, progress_bar_width, progress_bar_height), 2)

            # Draw progress fill (green)
            fill_width = int((progress_bar_width - 4) * progress)
            if fill_width > 0:
                pygame.draw.rect(self.screen, (0, 200, 0), (x + 2, progress_bar_y + 2, fill_width, progress_bar_height - 4))

            # Draw progress text (dark black, in middle of current progress)
            progress_text = f"{progress*100:.1f}%"
            text_surf = self.font_small.render(progress_text, True, (0, 0, 0))
            # Position text in middle of filled area (or at start if progress is very low)
            text_x = x + max(fill_width // 2, 30)
            text_rect = text_surf.get_rect(center=(text_x, progress_bar_y + progress_bar_height//2))
            self.screen.blit(text_surf, text_rect)

        # Draw graph - only show when playing and have data
        if len(self.reward_history) < 2:
            # Draw empty box (no text)
            pygame.draw.rect(self.screen, BOX_COLOR, (x, y, width, height))
            pygame.draw.rect(self.screen, BOX_BORDER_COLOR, (x, y, width, height), 2)
            return

        # Use cached graph if step hasn't changed
        if self.cached_reward_graph is None or self.last_graph_step != self.episode_step:
            self.cached_reward_graph = self.create_reward_graph_surface(width, height)
            self.last_graph_step = self.episode_step

        self.screen.blit(self.cached_reward_graph, (x, y))

    def create_rrt_graph_surface(self, width, height):
        """Create RRT path planning visualization showing only current target cube"""
        if len(self.accumulated_paths) == 0:
            # Return empty surface
            surf = pygame.Surface((width, height))
            surf.fill(BOX_COLOR)
            return surf

        # Check if we can use cached graph (only regenerate when progress changes significantly)
        # Regenerate every 2% progress change for smooth animation
        # Also regenerate when step changes (to show new target)
        current_step_idx = len(self.accumulated_paths) - 1
        if (self.cached_rrt_graph is not None and
            abs(self.path_progress - self.cached_rrt_progress) < 0.02 and
            self.animation_phase == self.cached_rrt_phase and
            current_step_idx == self.cached_rrt_paths_len):
            return self.cached_rrt_graph

        # Create matplotlib figure with white background
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100, facecolor='white')
        ax.set_facecolor('white')

        # Set fixed axis limits for all steps (inverted Y-axis so EE is at bottom)
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(self.grid_size - 0.5, -0.5)  # Inverted Y-axis
        ax.set_aspect('equal')

        # Draw EE position (blue circle at bottom - same color as grid)
        ee_grid = self.env.rrt_estimator._world_to_grid(self.env.ee_position[:2])
        # Convert AGENT_COLOR (100, 150, 255) to matplotlib format (0-1 range)
        ee_color = (100/255, 150/255, 255/255)
        ax.plot(ee_grid[0], ee_grid[1], "o", color=ee_color, markersize=6)

        # Draw static obstacles as very thin red rectangles
        for ox, oy in self.static_obstacles:
            # Create very thin rectangle centered at (ox, oy)
            rect_width = 0.08  # Very thin width
            rect_height = 0.85  # Slightly reduced height
            rect = plt.Rectangle((ox - rect_width/2, oy - rect_height/2),
                                rect_width, rect_height,
                                color='red', linewidth=0)
            ax.add_patch(rect)

        # Draw ONLY the current target cube (not cumulative)
        current_idx = len(self.accumulated_paths) - 1
        _, _, target_grid = self.accumulated_paths[current_idx]
        square_size = 0.15
        square = plt.Rectangle((target_grid[0] - square_size/2, target_grid[1] - square_size/2),
                              square_size, square_size,
                              color='green', linewidth=0)
        ax.add_patch(square)

        # Draw ONLY the current path (not all accumulated paths)
        path, tree_edges, target_grid = self.accumulated_paths[current_idx]

        # Animate based on phase and progress
        if self.animation_phase == 2:
            # During animation - draw progressively
            if self.path_progress < 0.5:
                # First half: Draw tree edges progressively (0.0 to 0.5 -> 0% to 100% of edges)
                if tree_edges:
                    progress_in_first_half = self.path_progress / 0.5  # Map 0.0-0.5 to 0.0-1.0
                    num_edges = max(1, int(len(tree_edges) * progress_in_first_half))
                    for j in range(num_edges):
                        (px, py), (cx, cy) = tree_edges[j]
                        ax.plot([px, cx], [py, cy], "-", color='blue', alpha=0.3, linewidth=1)
            else:
                # Second half: Draw all tree edges + growing red line (0.5 to 1.0 -> 0% to 100% of path)
                # Draw all tree edges
                if tree_edges:
                    for (px, py), (cx, cy) in tree_edges:
                        ax.plot([px, cx], [py, cy], "-", color='blue', alpha=0.3, linewidth=1)

                # Draw growing red line (reduced thickness to half: 1.0 -> 0.5)
                if path and len(path) > 1:
                    progress_in_second_half = (self.path_progress - 0.5) / 0.5  # Map 0.5-1.0 to 0.0-1.0
                    num_points = max(2, int(len(path) * progress_in_second_half))
                    if num_points >= 2:
                        path_x = [p[0] for p in path[:num_points]]
                        path_y = [p[1] for p in path[:num_points]]
                        ax.plot(path_x, path_y, "-r", linewidth=0.5)
        else:
            # After animation completes - show full tree and path
            # Draw all tree edges (blue lines)
            if tree_edges:
                for (px, py), (cx, cy) in tree_edges:
                    ax.plot([px, cx], [py, cy], "-", color='blue', alpha=0.3, linewidth=1)

            # Draw full path (thin red line - reduced thickness to half: 1.0 -> 0.5)
            if path and len(path) > 1:
                path_x = [p[0] for p in path]
                path_y = [p[1] for p in path]
                ax.plot(path_x, path_y, "-r", linewidth=0.5)

        # Styling - no axis labels, no ticks, black border
        ax.set_title('RRT Path Planning', color='#64C8FF', fontsize=9, fontweight='bold', pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        # Set black border
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

        # Adjust layout to maximize graph area (reduce top/bottom white margins)
        plt.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.12)

        # Convert to pygame surface
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        # Get the RGBA buffer
        buf = canvas.buffer_rgba()
        size = canvas.get_width_height()

        # Create surface from RGBA buffer
        surf = pygame.image.frombuffer(buf, size, "RGBA")
        plt.close(fig)

        # Cache the surface
        self.cached_rrt_graph = surf
        self.cached_rrt_progress = self.path_progress
        self.cached_rrt_phase = self.animation_phase
        self.cached_rrt_paths_len = current_step_idx  # Cache current step index

        return surf

    def draw_rrt_graph(self, x, y, width, height):
        """Draw RRT path planning graph"""
        # Draw graph - show accumulated paths
        if len(self.accumulated_paths) == 0:
            # Draw empty box (no text)
            pygame.draw.rect(self.screen, BOX_COLOR, (x, y, width, height))
            pygame.draw.rect(self.screen, BOX_BORDER_COLOR, (x, y, width, height), 2)
            return

        # Generate and display graph surface
        graph_surf = self.create_rrt_graph_surface(width, height)
        self.screen.blit(graph_surf, (x, y))

    def render(self):
        """Render the visualization"""
        # Clear screen
        self.screen.fill(BG_COLOR)

        # Show loading screen if generating episode
        if self.generating_episode:
            win_w, win_h = self.screen.get_size()
            loading_text = f"Generating episode {self.episode_count + 1}..."
            loading_surf = self.font_large.render(loading_text, True, HEADER_COLOR)
            loading_rect = loading_surf.get_rect(center=(win_w // 2, win_h // 2 - 20))
            self.screen.blit(loading_surf, loading_rect)

            info_text = f"Planning {self.num_cubes} RRT paths..."
            info_surf = self.font_medium.render(info_text, True, TEXT_COLOR)
            info_rect = info_surf.get_rect(center=(win_w // 2, win_h // 2 + 20))
            self.screen.blit(info_surf, info_rect)

            pygame.display.flip()
            return

        # Calculate layout - grid on left, info panel on right
        win_w, win_h = self.screen.get_size()
        info_width = 420  # Info panel width
        graph_width = 350  # Reduced graph width
        graph_height = 220  # Height for bottom graph
        gap = 80  # Increased gap between grid and info panel

        grid_area_w = win_w - info_width - gap
        grid_area_h = win_h - 60

        # Calculate grid size (HALF the previous size)
        cell_size = min(grid_area_w // self.grid_size, grid_area_h // self.grid_size) // 2
        grid_w = self.grid_size * cell_size
        grid_h = self.grid_size * cell_size

        # Center grid in left area
        grid_x = (grid_area_w - grid_w) // 2
        grid_y = (win_h - grid_h) // 2

        # Draw components
        self.draw_grid(cell_size, grid_x, grid_y)
        self.draw_obstacles(cell_size, grid_x, grid_y)
        self.draw_cubes(cell_size, grid_x, grid_y)
        agent_pos = self.draw_agent(cell_size, grid_x, grid_y, grid_h)
        self.draw_path(cell_size, grid_x, grid_y, grid_h, agent_pos)

        # Draw info panel on right side
        info_x = grid_x + grid_w + gap
        info_y = 20
        info_panel_height = win_h - graph_height - 60
        self.draw_info_panel(info_x, info_y, info_width, info_panel_height)

        # Draw graphs at bottom of right panel (side by side)
        graph_y = info_y + info_panel_height + 20
        graph_gap = 20

        # Reward graph on left
        self.draw_reward_graph(info_x, graph_y, graph_width, graph_height)

        # RRT graph on right
        rrt_graph_x = info_x + graph_width + graph_gap
        self.draw_rrt_graph(rrt_graph_x, graph_y, graph_width, graph_height)

        # Update display
        pygame.display.flip()

    def update_playback(self):
        """Update episode playback with 3-phase animation

        Phase flow:
        - Phase 0: Idle - brief pause before starting next pick
        - Phase 1: Select cube (show blue border) - 30 frames
        - Phase 2: Show path (blue border + path) - 90 frames
        - Phase 3: Disappear (cube + path fade out together) - 30 frames
        - After Phase 3: Increment episode_step, return to Phase 0
        """
        if not self.current_episode or self.paused:
            return

        if not self.auto_advance:
            return

        # Start timer when Step 1 begins (episode_step == 0 and animation starts)
        if not self.timer_started and self.episode_step == 0 and self.animation_phase == 0:
            self.episode_start_time = time.time()
            self.timer_started = True

        # Update episode timer (only when not paused and timer has started)
        if not self.paused and self.timer_started and self.episode_step < len(self.current_episode.actions):
            self.episode_elapsed_time = time.time() - self.episode_start_time

        # Phase 0: Idle - brief pause before starting next pick
        if self.animation_phase == 0:
            self.phase_timer += self.playback_speed
            if self.phase_timer >= self.playback_speed:  # Wait just 1 frame
                # Check if there are more cubes to pick
                if self.episode_step < len(self.current_episode.actions):
                    # Select next cube to pick
                    action_idx = self.episode_step
                    self.selected_cube_idx = self.current_episode.actions[action_idx]
                    self.current_path = self.current_episode.paths[action_idx]

                    # Update observation components (use correct metric keys)
                    self.current_obs_components['dist_to_ee'] = self.current_episode.metrics['distances_to_ee'][action_idx]
                    self.current_obs_components['dist_to_container'] = self.current_episode.metrics['distances_to_container'][action_idx]
                    self.current_obs_components['obstacle_proximity'] = self.current_episode.metrics['obstacle_proximities'][action_idx]
                    self.current_obs_components['reachability'] = self.current_episode.metrics['reachability_flags'][action_idx]
                    self.current_obs_components['path_clearance'] = self.current_episode.metrics['path_clearances'][action_idx]
                    self.current_obs_components['items_left'] = len(self.current_episode.actions) - action_idx
                    self.current_obs_components['dist_to_origin'] = np.linalg.norm(self.env.object_positions[self.selected_cube_idx][:2])

                    # Calculate reward components from observation values
                    path_length = self.current_episode.metrics['path_lengths'][action_idx]
                    reward_components = self.calculate_reward_components(
                        path_length=path_length,
                        dist_to_container=self.current_obs_components['dist_to_container'],
                        obstacle_proximity=self.current_obs_components['obstacle_proximity'],
                        reachability=self.current_obs_components['reachability'],
                        path_clearance=self.current_obs_components['path_clearance'],
                        dist_to_ee=self.current_obs_components['dist_to_ee']
                    )

                    # Update reward components
                    self.current_reward_components['r_total'] = self.current_episode.rewards[action_idx]
                    self.current_reward_components['r_path'] = reward_components['r_path']
                    self.current_reward_components['r_container'] = reward_components['r_container']
                    self.current_reward_components['r_obstacle'] = reward_components['r_obstacle']
                    self.current_reward_components['r_reachability'] = reward_components['r_reachability']
                    self.current_reward_components['r_clearance'] = reward_components['r_clearance']

                    # Add to reward history
                    self.reward_history.append(self.current_episode.rewards[action_idx])

                    # Start phase 1: Show selection
                    self.animation_phase = 1
                    self.phase_timer = 0
                else:
                    # Episode complete
                    self.generate_new_episode()

        # Phase 1: Show selected cube (blue border only)
        elif self.animation_phase == 1:
            self.phase_timer += self.playback_speed
            if self.phase_timer >= self.selection_delay:
                # Check if we have a valid path to animate
                # Path is valid if it exists and is a list (even if empty or has 1 point)
                # We want to show the RRT graph even for short/empty paths
                has_valid_path = (self.current_path is not None and
                                 isinstance(self.current_path, list))

                if has_valid_path:
                    # Add current path to accumulated paths for RRT graph
                    action_idx = self.episode_step
                    if self.current_episode and action_idx < len(self.current_episode.tree_edges):
                        tree_edges = self.current_episode.tree_edges[action_idx]
                        target_pos = self.current_episode.cube_positions[self.selected_cube_idx]
                        target_grid = self.env.rrt_estimator._world_to_grid(target_pos[:2])
                        self.accumulated_paths.append((self.current_path, tree_edges, target_grid))

                    # Move to phase 2: Path animation (even if path is short)
                    self.animation_phase = 2
                    self.phase_timer = 0
                    self.path_progress = 0.0  # Reset path animation
                else:
                    # Skip phase 2 only if path is None (should not happen if RRT succeeded)
                    self.animation_phase = 3
                    self.phase_timer = 0

        # Phase 2: Path animation (path grows towards target)
        elif self.animation_phase == 2:
            self.phase_timer += self.playback_speed
            # Update path progress (0.0 to 1.0)
            self.path_progress = min(1.0, self.phase_timer / self.path_animation_duration)

            if self.phase_timer >= self.path_animation_duration:
                # Move to phase 3: Cube + path disappear together
                self.animation_phase = 3
                self.phase_timer = 0
                self.path_progress = 1.0  # Ensure full path is shown

        # Phase 3: Cube + path disappear together
        elif self.animation_phase == 3:
            self.phase_timer += self.playback_speed
            if self.phase_timer >= self.post_animation_delay:
                # Cube is now completed - increment step counter
                self.episode_step += 1

                # Clear selection to prevent re-rendering
                self.selected_cube_idx = None
                self.current_path = None

                # Return to phase 0: Brief idle before next cube
                self.animation_phase = 0
                self.phase_timer = 0

    def run(self):
        """Main loop"""
        print(f"\nRRT Episode Visualizer - {self.grid_size}x{self.grid_size} grid, {self.num_cubes} cubes")
        print("Controls: SPACE=Play/Pause, N=Next, R=Reset, G=New Episode, +/-=Speed, ESC=Quit\n")

        # Generate first episode on first frame (non-blocking)
        first_frame = True

        while self.running:
            # Generate episode on first frame
            if first_frame:
                first_frame = False
                self.generate_new_episode()

            # Handle events
            self.handle_events()

            # Update playback
            self.update_playback()

            # Render
            self.render()

            # Control frame rate
            self.clock.tick(self.fps)

        # Cleanup
        pygame.quit()
        print(f"\nVisualizer closed. Total episodes: {self.episode_count}")


def main():
    """Main function"""
    args = parse_args()

    # Validate inputs
    if args.num_cubes > args.grid_size * args.grid_size:
        print(f"ERROR: Number of cubes ({args.num_cubes}) exceeds grid capacity")
        sys.exit(1)

    # Create and run visualizer
    visualizer = RRTRLEpisodeVisualizer(
        grid_size=args.grid_size,
        num_cubes=args.num_cubes,
        model_path=args.model_path,
        window_width=args.window_width,
        window_height=args.window_height,
        initial_fps=args.fps
    )

    visualizer.run()


if __name__ == "__main__":
    main()
