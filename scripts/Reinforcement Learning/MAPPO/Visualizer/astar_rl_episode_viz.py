"""
MAPPO Two-Agent + A* Episode Visualizer

Visualizes MAPPO two-agent system:
- Agent 1 (DDQN): Selects cubes to pick
- Agent 2 (MAPPO): Decides when/how to reshuffle cubes
Shows one complete episode with curved paths, reshuffling events, and interactive charts.
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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add MAPPO module to path
mappo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if mappo_root not in sys.path:
    sys.path.insert(0, mappo_root)

# Import Catmull-Rom spline from local utils
from src.utils.catmull_rom_spline_path import catmull_rom_spline

from src.rl.object_selection_env_astar import ObjectSelectionEnvAStar
from src.rl.path_estimators import Node
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from src.rl.doubleDQN import DoubleDQNAgent

# Import MAPPO modules
from algorithms.mappo_policy import MAPPOPolicy
from envs.two_agent_env import TwoAgentEnv
from envs.reshuffling_decision import ReshufflingReason

# Colors
BG_COLOR = (20, 20, 30)
GRID_COLOR = (60, 60, 80)
CUBE_COLOR = (100, 200, 100)
OBSTACLE_COLOR = (200, 50, 50)
AGENT_COLOR = (100, 150, 255)
PATH_COLOR = (255, 200, 0)
SELECTED_BORDER_COLOR = (0, 150, 255)  # Blue border for picked cubes
RESHUFFLE_BORDER_COLOR = (255, 255, 0)  # Yellow border for reshuffled cubes
TEXT_COLOR = (220, 220, 220)
HEADER_COLOR = (100, 200, 255)  # Bright blue for headers
LABEL_COLOR = (180, 180, 200)   # Light gray for labels
VALUE_COLOR = (255, 255, 255)   # White for values
BOX_COLOR = (40, 40, 50)        # Dark box background
BOX_BORDER_COLOR = (80, 80, 100)  # Box border


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='MAPPO Two-Agent + A* Episode Visualizer')
    parser.add_argument('--grid_size', type=int, default=4, help='Grid size (default: 4)')
    parser.add_argument('--num_cubes', type=int, default=9, help='Number of cubes (default: 9)')
    parser.add_argument('--ddqn_model_path', type=str, default=None, help='Path to trained DDQN model (Agent 1)')
    parser.add_argument('--mappo_model_path', type=str, default=None, help='Path to trained MAPPO model (Agent 2)')
    parser.add_argument('--window_width', type=int, default=1600, help='Window width (default: 1600)')
    parser.add_argument('--window_height', type=int, default=900, help='Window height (default: 900)')
    parser.add_argument('--fps', type=int, default=30, help='FPS (default: 30)')
    return parser.parse_args()


def cubic_spline(points: List[Tuple[float, float]], num_points: int = 100) -> List[Tuple[float, float]]:
    """
    Generate smooth curve through points using Cubic Spline interpolation.

    Parameters:
        points: List of (x, y) waypoints from A* path
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
        self.rewards = []
        self.cube_positions = []
        self.obstacle_positions = []
        self.ee_start_position = None
        self.picked_cubes = []

        # Reshuffling tracking
        self.reshuffles = []  # List of reshuffle events
        self.reshuffle_rewards = []  # Reshuffle rewards per step
        self.pick_rewards = []  # Pick rewards per step

        self.metrics = {
            'path_lengths': [],
            'obstacle_proximities': [],
            'reachability_flags': [],
            'path_clearances': [],
            'distances_to_ee': [],
            'distances_to_container': [],
            'total_reward': 0.0,
            'episode_length': 0,
            'total_reshuffles': 0
        }

    def add_step(self, action, path, reward, path_length, obstacle_proximity,
                 reachability, path_clearance, dist_to_ee, dist_to_container,
                 reshuffle_info=None, reshuffle_reward=0.0, pick_reward=0.0):
        """Add a step to the episode"""
        self.actions.append(action)
        self.paths.append(path)
        self.rewards.append(reward)
        self.reshuffle_rewards.append(reshuffle_reward)
        self.pick_rewards.append(pick_reward)

        if reshuffle_info:
            self.reshuffles.append(reshuffle_info)
            self.metrics['total_reshuffles'] += 1

        self.metrics['path_lengths'].append(path_length)
        self.metrics['obstacle_proximities'].append(obstacle_proximity)
        self.metrics['reachability_flags'].append(1.0 if reachability else 0.0)
        self.metrics['path_clearances'].append(path_clearance)
        self.metrics['distances_to_ee'].append(dist_to_ee)
        self.metrics['distances_to_container'].append(dist_to_container)
        self.metrics['total_reward'] += reward
        self.metrics['episode_length'] += 1


class AStarRLEpisodeVisualizer:
    """Episode visualizer for MAPPO Two-Agent + A* pick-and-place"""

    def __init__(self, grid_size: int, num_cubes: int,
                 ddqn_model_path: Optional[str] = None,
                 mappo_model_path: Optional[str] = None,
                 window_width: int = 1600, window_height: int = 900, initial_fps: int = 30):
        """Initialize visualizer"""
        # Configuration
        self.grid_size = grid_size
        self.num_cubes = num_cubes
        self.window_width = window_width
        self.window_height = window_height
        self.fps = initial_fps
        self.ddqn_model_path = ddqn_model_path
        self.mappo_model_path = mappo_model_path

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
        pygame.display.set_caption("MAPPO Two-Agent + A* Motion Planning")
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

        # Initialize models
        self.ddqn_agent = None
        self.mappo_agent = None
        model_max_objects = None
        model_grid_size = grid_size

        # Load DDQN model metadata if provided
        if ddqn_model_path and os.path.exists(ddqn_model_path):
            print(f"[DDQN] Loading DDQN model from: {ddqn_model_path}")

            # Try to load metadata JSON file
            if "_step_" in ddqn_model_path:
                base_name = ddqn_model_path.rsplit("_step_", 1)[0]
                metadata_path = base_name + "_metadata.json"
            else:
                metadata_path = ddqn_model_path.replace("_final.pt", "_metadata.json").replace(".pt", "_metadata.json")

            if os.path.exists(metadata_path):
                import json
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        model_max_objects = metadata.get("max_objects", None)
                        model_grid_size = metadata.get("training_grid_size", grid_size)
                        print(f"[DDQN] Loaded metadata: grid={model_grid_size}x{model_grid_size}, max_objects={model_max_objects}, num_cubes={metadata.get('num_cubes', num_cubes)}")

                        # Warn if mismatch
                        if model_grid_size != grid_size:
                            print(f"[WARNING] DDQN trained on {model_grid_size}x{model_grid_size} grid, but running with {grid_size}x{grid_size}")
                            print(f"[INFO] Using DDQN's grid size: {model_grid_size}x{model_grid_size}")
                            grid_size = model_grid_size
                            self.grid_size = grid_size
                except Exception as e:
                    print(f"[WARNING] Could not read DDQN metadata file: {e}")
            else:
                print(f"[WARNING] DDQN metadata file not found: {metadata_path}")



        # Create base environment (A* for Agent 1)
        max_objects = model_max_objects if model_max_objects else num_cubes
        print(f"[ENV] Creating base A* environment with max_objects={max_objects}, num_cubes={num_cubes}, grid={grid_size}x{grid_size}")

        base_env = ObjectSelectionEnvAStar(
            franka_controller=None,
            max_objects=max_objects,
            max_steps=num_cubes * 2,
            num_cubes=num_cubes,
            render_mode=None,
            dynamic_obstacles=False,
            training_grid_size=grid_size
        )

        # Wrap in TwoAgentEnv for MAPPO
        print(f"[ENV] Wrapping in TwoAgentEnv for MAPPO two-agent system")
        self.env = TwoAgentEnv(
            base_env=base_env,
            grid_size=grid_size,
            num_cubes=num_cubes,
            max_reshuffles_per_episode=5,
            reshuffle_reward_scale=1.0
        )

        # Keep reference to base environment for visualization
        self.base_env = base_env
        self.wrapped_env = self.env  # For compatibility

        # Load Agent 1 (DDQN) model
        if ddqn_model_path and os.path.exists(ddqn_model_path):
            try:
                print(f"[DDQN] Loading DDQN model...")
                checkpoint = torch.load(ddqn_model_path, map_location='cpu')
                state_dim = checkpoint['state_dim']
                action_dim = checkpoint['action_dim']

                self.ddqn_agent = DoubleDQNAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    gamma=checkpoint['gamma'],
                    epsilon_start=0.0,
                    epsilon_end=0.0,
                    epsilon_decay=1.0,
                    batch_size=checkpoint['batch_size'],
                    target_update_freq=checkpoint['target_update_freq']
                )
                self.ddqn_agent.load(ddqn_model_path)
                self.ddqn_agent.epsilon = 0.0  # Greedy action selection
                print(f"[DDQN] Model loaded successfully!")
            except Exception as e:
                print(f"[ERROR] Failed to load DDQN model: {e}")
                print(f"[INFO] Agent 1 will use random actions")
                self.ddqn_agent = None

        # Load Agent 2 (MAPPO) model
        if mappo_model_path and os.path.exists(mappo_model_path):
            try:
                print(f"[MAPPO] Loading MAPPO model from: {mappo_model_path}")

                # Get observation and action dimensions from environment
                obs_dim = self.env.observation_space.shape[0]
                action_dim = self.env.action_space.n

                self.mappo_agent = MAPPOPolicy(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_dim=256
                )

                checkpoint = torch.load(mappo_model_path, map_location='cpu')
                self.mappo_agent.load_state_dict(checkpoint)
                self.mappo_agent.eval()
                print(f"[MAPPO] Model loaded successfully!")
            except Exception as e:
                print(f"[ERROR] Failed to load MAPPO model: {e}")
                print(f"[INFO] Agent 2 will use random actions")
                self.mappo_agent = None

        # Print model status
        if not self.ddqn_agent and not self.mappo_agent:
            print(f"[INFO] Running with random actions for both agents")

        # Episode state
        self.current_episode = None
        self.episode_step = 0
        self.episode_count = 0
        self.episode_history = deque(maxlen=50)
        self.chart_fig = None

        # Cache for graph surfaces to avoid regenerating every frame
        self.cached_reward_graph = None
        self.cached_reward_history_len = 0

        # Cache for arrow calculations to avoid recalculating every frame
        self.cached_arrow_data = None
        self.cached_path_progress = -1.0

        # Accumulated paths for A* graph (all paths from all steps in current episode)
        self.accumulated_paths = []  # List of (path, explored_nodes, target_pos) tuples

        # Cache for A* graph to avoid regenerating every frame
        self.cached_astar_graph = None
        self.cached_astar_progress = -1.0
        self.cached_astar_phase = -1
        self.cached_astar_paths_len = 0

        # Visualization state
        self.selected_cube_idx = None
        self.current_path = None
        self.explored_nodes = []  # Track A* explored nodes for visualization
        self.static_obstacles = []
        self.reshuffled_cube_idx = None  # Track cube being reshuffled (for yellow border)
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
        self.path_animation_duration = 60  # frames for path drawing animation (reduced from 90 for faster movement)
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
            'invalid_pick': 0.0,
            'items_left': 0,
            'dist_to_origin': 0.0
        }

        # Reward history for spike graph
        self.reward_history = []

        # Episode timer
        self.episode_start_time = 0
        self.episode_elapsed_time = 0  # in seconds
        self.timer_started = False  # Flag to track if timer has started

        # Generate first episode
        self.generate_new_episode()

    def generate_new_episode(self):
        """Generate a new episode by running the two-agent system"""
        # Reset environment
        agent2_obs = self.env.reset()

        # Reset episode timer (will start when Step 1 begins)
        self.episode_start_time = 0
        self.episode_elapsed_time = 0
        self.timer_started = False

        # Clear accumulated paths for new episode
        self.accumulated_paths = []

        # Generate random obstacles (1-3, but only if there's room)
        grid_capacity = self.grid_size * self.grid_size
        available_cells = grid_capacity - self.num_cubes - 1  # -1 for EE home cell
        max_obstacles = max(0, min(3, available_cells))
        min_obstacles = 1 if max_obstacles > 0 else 0
        num_obstacles = np.random.randint(min_obstacles, max_obstacles + 1) if max_obstacles > 0 else 0
        self._add_random_obstacles(num_obstacles)

        # Create episode data
        episode = EpisodeData()
        episode.cube_positions = self.base_env.object_positions[:self.base_env.total_objects].copy()
        episode.obstacle_positions = [self.base_env.astar_estimator._grid_to_world(ox, oy)
                                     for ox, oy in self.static_obstacles]
        episode.ee_start_position = self.base_env.ee_position.copy()

        # Run episode
        done = False
        step = 0
        successful_picks = 0
        failed_picks = 0

        print(f"\nGenerating episode {self.episode_count + 1}...")
        print(f"  {self.num_cubes} cubes, {num_obstacles} obstacles")

        while not done and step < self.base_env.max_steps:
            # Agent 1 (DDQN) selects cube to pick
            agent1_obs = self.base_env._get_observation()

            if self.ddqn_agent is not None:
                # Get action mask from base environment
                action_mask = self.base_env.action_masks()
                obs_flat = agent1_obs.flatten()
                obs_tensor = torch.FloatTensor(obs_flat).to(self.ddqn_agent.device)
                agent1_action = self.ddqn_agent.policy_net.get_action(obs_tensor, epsilon=0.0, action_mask=action_mask)
            else:
                # Random action (greedy baseline)
                agent1_action = self._greedy_action()

            # Agent 2 (MAPPO) decides reshuffling
            if self.mappo_agent is not None:
                agent2_obs_tensor = torch.FloatTensor(agent2_obs).unsqueeze(0)
                with torch.no_grad():
                    action_logits, _ = self.mappo_agent(agent2_obs_tensor)
                    agent2_action = torch.argmax(action_logits, dim=-1).item()
            else:
                # Random action
                agent2_action = np.random.randint(0, self.env.action_space.n)

            # Update A* grid with current unpicked cubes as obstacles
            # (exclude the target cube we're planning to pick)
            self._update_astar_grid_for_planning(agent1_action)

            # Get A* path before taking action
            path = self._get_astar_path(agent1_action)

            # Track success/failure based on A* result
            if path and len(path) > 0:
                successful_picks += 1
                print(f"  Step {step+1}: Cube {agent1_action}, Path points: {len(path)} ✓")
            else:
                failed_picks += 1
                print(f"  Step {step+1}: Cube {agent1_action}, Path points: 0 ✗ A* FAILED")

            # Calculate metrics
            path_length = self._calculate_path_length(path) if path else 999.0
            obstacle_proximity = self._calculate_obstacle_proximity(agent1_action)
            dist_to_ee = np.linalg.norm(self.base_env.object_positions[agent1_action][:2] - self.base_env.ee_position[:2])
            # Reachability based on distance (0.3m to 0.9m range, matches environment)
            reachability = 1.0 if (0.3 <= dist_to_ee <= 0.9) else 0.0
            path_clearance = self._calculate_path_clearance(path) if path else 0.0
            dist_to_container = 0.5

            # Take action (both agents)
            next_agent2_obs, reward, terminated, truncated, info = self.env.step(agent1_action, agent2_action)
            done = terminated or truncated
            agent2_obs = next_agent2_obs

            # Extract reshuffling info
            reshuffle_info = None
            reshuffle_reward = 0.0
            pick_reward = 0.0

            if info.get('reshuffled_this_step', False):
                reshuffle_history = info.get('reshuffle_history', [])
                if reshuffle_history:
                    last_reshuffle = reshuffle_history[-1]
                    reshuffle_info = {
                        'step': step,
                        'cube_idx': last_reshuffle['cube_idx'],
                        'target_pos': last_reshuffle['target_pos'],
                        'reason': last_reshuffle['reason'],
                        'priority': last_reshuffle['priority']
                    }
                    print(f"    → Reshuffle: Cube {last_reshuffle['cube_idx']} (Priority {last_reshuffle['priority']}, {last_reshuffle['reason']})")

            # NOTE: EE position stays at home position (not updated)
            # Path planning always starts from same EE position for all picks

            # Record step
            episode.add_step(agent1_action, path, reward, path_length, obstacle_proximity,
                           reachability, path_clearance, dist_to_ee, dist_to_container,
                           reshuffle_info, reshuffle_reward, pick_reward)
            episode.picked_cubes.append(agent1_action)

            step += 1

        self.current_episode = episode
        self.episode_step = 0
        self.episode_count += 1
        self.episode_history.append(episode)

        # Print accurate summary (without reward)
        print(f"Done! ({successful_picks} successful, {failed_picks} failed)\n")

    def _add_random_obstacles(self, num_obstacles: int):
        """Add random obstacles to empty grid cells"""
        self.static_obstacles = []

        if num_obstacles == 0:
            return

        # Get EE home position to avoid
        ee_grid_x, ee_grid_y = self.base_env.astar_estimator._world_to_grid(self.base_env.ee_position[:2])

        # Get cube positions to avoid
        cube_cells = set()
        cube_cells.add((ee_grid_x, ee_grid_y))  # Exclude EE home cell

        for i in range(self.base_env.total_objects):
            pos = self.base_env.object_positions[i]
            grid_col, grid_row = self.base_env.astar_estimator._world_to_grid(pos[:2])
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

    def _update_astar_grid_for_planning(self, target_cube_idx: int):
        """
        Update A* occupancy grid with current obstacles.

        Obstacles include:
        1. Static obstacles (randomly placed)
        2. Unpicked cubes (EXCEPT the target cube we're planning to pick)

        Args:
            target_cube_idx: Index of the cube we're planning to pick (not treated as obstacle)
        """
        # Get static obstacle positions
        obstacle_positions = []
        for grid_x, grid_y in self.static_obstacles:
            world_pos = self.base_env.astar_estimator._grid_to_world(grid_x, grid_y)
            obstacle_positions.append(np.array([world_pos[0], world_pos[1], 0.0]))

        # Get unpicked cube positions (excluding target cube)
        unpicked_cube_positions = []
        for i in range(self.base_env.total_objects):
            if i not in self.base_env.objects_picked and i != target_cube_idx:
                unpicked_cube_positions.append(self.base_env.object_positions[i])

        # Update A* grid
        self.base_env.astar_estimator.update_occupancy_grid(
            object_positions=unpicked_cube_positions,
            obstacle_positions=obstacle_positions
        )

    def _greedy_action(self) -> int:
        """
        Greedy baseline: pick closest unpicked cube using A* path length.

        This ensures we pick cubes that are actually reachable, not just physically close.
        """
        min_path_length = float('inf')
        best_action = 0

        for i in range(self.base_env.total_objects):
            if i not in self.base_env.objects_picked:
                # Use A* path length instead of Euclidean distance
                # This accounts for obstacles and ensures we pick reachable cubes
                path_length = self.base_env.astar_estimator.estimate_path_length(
                    self.base_env.ee_position[:2],
                    self.base_env.object_positions[i][:2]
                )

                if path_length < min_path_length:
                    min_path_length = path_length
                    best_action = i

        return best_action

    def _get_astar_path(self, cube_idx: int) -> Optional[List[Tuple[int, int]]]:
        """Get A* path from end-effector to cube"""
        if cube_idx >= self.base_env.total_objects:
            return None

        ee_grid = self.base_env.astar_estimator._world_to_grid(self.base_env.ee_position[:2])
        cube_pos = self.base_env.object_positions[cube_idx]
        goal_grid = self.base_env.astar_estimator._world_to_grid(cube_pos[:2])

        # Check if start == goal
        if ee_grid == goal_grid:
            return []  # Return empty list

        # Run A* search (using PythonRobotics planning method)
        rx, ry = self.base_env.astar_estimator.planning(ee_grid[0], ee_grid[1], goal_grid[0], goal_grid[1])

        # Convert to list of tuples
        # NOTE: PythonRobotics A* returns path from GOAL to START, so we need to REVERSE it
        if rx is None or ry is None:
            return []  # No path found
        else:
            path = [(rx[i], ry[i]) for i in range(len(rx))]
            path.reverse()  # Reverse to get path from START to GOAL
            return path

    def _get_astar_path_with_explored(self, cube_idx: int) -> Tuple[Optional[List[Tuple[int, int]]], List[Tuple[int, int]]]:
        """Get A* path and explored nodes from end-effector to cube"""
        if cube_idx >= self.base_env.total_objects:
            return None, []

        ee_grid = self.base_env.astar_estimator._world_to_grid(self.base_env.ee_position[:2])
        cube_pos = self.base_env.object_positions[cube_idx]
        goal_grid = self.base_env.astar_estimator._world_to_grid(cube_pos[:2])

        # Check if start == goal
        if ee_grid == goal_grid:
            return [], []  # Return empty lists

        # Run A* search and get closed_set (explored nodes)
        start_node = Node(ee_grid[0], ee_grid[1], 0.0, -1)
        goal_node = Node(goal_grid[0], goal_grid[1], 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.base_env.astar_estimator.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                return [], []  # No path found

            c_id = min(open_set, key=lambda o: open_set[o].cost + self.base_env.astar_estimator.calc_heuristic(goal_node, open_set[o]))
            current = open_set[c_id]

            if current.x == goal_node.x and current.y == goal_node.y:
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            del open_set[c_id]
            closed_set[c_id] = current

            for i, _ in enumerate(self.base_env.astar_estimator.motion):
                node = Node(
                    current.x + self.base_env.astar_estimator.motion[i][0],
                    current.y + self.base_env.astar_estimator.motion[i][1],
                    current.cost + self.base_env.astar_estimator.motion[i][2],
                    c_id
                )
                n_id = self.base_env.astar_estimator.calc_grid_index(node)

                if not self.base_env.astar_estimator.verify_node(node, goal_node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node
                else:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node

        # Extract path
        rx, ry = self.base_env.astar_estimator.calc_final_path(goal_node, closed_set)

        # Extract explored nodes
        explored = [(closed_set[key].x, closed_set[key].y) for key in closed_set]

        if rx is None or ry is None:
            return [], explored
        else:
            path = [(rx[i], ry[i]) for i in range(len(rx))]
            path.reverse()  # Reverse to get path from START to GOAL
            return path, explored

    def _calculate_path_length(self, path: List[Tuple[int, int]]) -> float:
        """Calculate path length in meters"""
        if not path or len(path) < 2:
            return 0.0

        length = 0.0
        for i in range(len(path) - 1):
            pos1 = self.base_env.astar_estimator._grid_to_world(path[i][0], path[i][1])
            pos2 = self.base_env.astar_estimator._grid_to_world(path[i+1][0], path[i+1][1])
            length += np.linalg.norm(pos2 - pos1)

        return length

    def _calculate_obstacle_proximity(self, cube_idx: int) -> float:
        """Calculate minimum distance to obstacles"""
        if not self.static_obstacles:
            return 999.0

        cube_pos = self.base_env.object_positions[cube_idx][:2]
        min_dist = float('inf')

        for ox, oy in self.static_obstacles:
            obs_pos = self.base_env.astar_estimator._grid_to_world(ox, oy)
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
        Matches the reward calculation in ObjectSelectionEnvAStar._calculate_reward()

        Args:
            path_length: A* path length in meters
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
        # Check if A* planning failed (path_length >= 2.0 × Euclidean)
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
                    self.paused = not self.paused

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
        """Draw cubes - only unpicked cubes, selected cube has blue border, reshuffled cube has yellow border"""
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
        ee_col, ee_row = self.base_env.astar_estimator._world_to_grid(self.base_env.ee_position[:2])

        for i in range(len(self.current_episode.cube_positions)):
            # Skip picked cubes
            if i in picked_cubes:
                continue

            pos = self.current_episode.cube_positions[i]
            grid_col, grid_row = self.base_env.astar_estimator._world_to_grid(pos[:2])

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

                # Highlight selected cube with BLUE border during phases 1 and 2 (being picked)
                if i == self.selected_cube_idx and self.animation_phase in [1, 2]:
                    border_width = 4
                    pygame.draw.rect(self.screen, SELECTED_BORDER_COLOR,
                                   (x - size // 2 - border_width, y - size // 2 - border_width,
                                    size + border_width * 2, size + border_width * 2), border_width)
                    # Draw blue circle at center of target cube
                    pygame.draw.circle(self.screen, SELECTED_BORDER_COLOR, (x, y), 5)

                # Highlight reshuffled cube with YELLOW border (being reshuffled)
                elif i == self.reshuffled_cube_idx:
                    border_width = 4
                    pygame.draw.rect(self.screen, RESHUFFLE_BORDER_COLOR,
                                   (x - size // 2 - border_width, y - size // 2 - border_width,
                                    size + border_width * 2, size + border_width * 2), border_width)
                    # Draw yellow circle at center of reshuffled cube
                    pygame.draw.circle(self.screen, RESHUFFLE_BORDER_COLOR, (x, y), 5)

    def draw_agent(self, cell_size, grid_x, grid_y, grid_h):
        """Draw agent (end-effector) at its actual grid cell position"""
        # Get EE grid position from world coordinates (same as used for obstacle exclusion)
        ee_col, ee_row = self.base_env.astar_estimator._world_to_grid(self.base_env.ee_position[:2])

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
        """Draw CURVED A* path using Catmull-Rom Spline interpolation"""
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

        # Build path points from A* grid coordinates
        points = []
        points.append((agent_x, agent_y))  # Start from agent position

        # Add all A* waypoints EXCEPT the first one (EE position)
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

                            # Find a previous point that's far enough away to get stable direction
                            # Look back through points to find one at least 10 pixels away for more stability
                            prev_x, prev_y = animated_points[-2]
                            for i in range(len(animated_points) - 2, -1, -1):
                                px, py = animated_points[i]
                                dist = np.sqrt((end_x - px)**2 + (end_y - py)**2)
                                if dist >= 10:
                                    prev_x, prev_y = px, py
                                    break

                            # Calculate direction vector
                            dx = end_x - prev_x
                            dy = end_y - prev_y
                            length = np.sqrt(dx**2 + dy**2)

                            if length > 2.0:  # Only draw arrow if direction is stable
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
        header_surf = self.font_header.render("Decision making + A* motion planning", True, HEADER_COLOR)
        self.screen.blit(header_surf, (x, current_y))
        current_y += 40

        # Episode Info Box
        box_y = self.draw_box(x, current_y, width, 195, "Episode Info")
        current_reward = sum(self.current_episode.rewards[:self.episode_step]) if self.episode_step > 0 else 0.0

        # Get distance to target cube (Dist->Target)
        dist_to_target = 0.0
        if self.selected_cube_idx is not None and self.selected_cube_idx < len(self.base_env.object_positions):
            target_pos = self.base_env.object_positions[self.selected_cube_idx][:2]
            dist_to_target = np.linalg.norm(target_pos - self.base_env.ee_position[:2])

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

        # Generate and display graph surface
        graph_surf = self.create_reward_graph_surface(width, height)
        self.screen.blit(graph_surf, (x, y))

    def create_astar_graph_surface(self, width, height):
        """Create A* path planning visualization showing all accumulated paths step-by-step"""
        if len(self.accumulated_paths) == 0:
            # Return empty surface
            surf = pygame.Surface((width, height))
            surf.fill(BOX_COLOR)
            return surf

        # Check if we can use cached graph (only regenerate when progress changes significantly)
        # Regenerate every 2% progress change for smooth animation
        if (self.cached_astar_graph is not None and
            abs(self.path_progress - self.cached_astar_progress) < 0.02 and
            self.animation_phase == self.cached_astar_phase and
            len(self.accumulated_paths) == self.cached_astar_paths_len):
            return self.cached_astar_graph

        # Create matplotlib figure with white background
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100, facecolor='white')
        ax.set_facecolor('white')

        # Set fixed axis limits for all steps (inverted Y-axis so EE is at bottom)
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(self.grid_size - 0.5, -0.5)  # Inverted Y-axis
        ax.set_aspect('equal')

        # Draw EE position (blue circle at bottom - same color as grid)
        ee_grid = self.base_env.astar_estimator._world_to_grid(self.base_env.ee_position[:2])
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

        # Draw all target squares up to current step (cumulative)
        num_targets_to_show = len(self.accumulated_paths)
        if self.animation_phase == 2:
            # During animation, show all targets including current one
            num_targets_to_show = len(self.accumulated_paths)
        else:
            # After animation completes, show all targets
            num_targets_to_show = len(self.accumulated_paths)

        for i in range(num_targets_to_show):
            _, _, target_grid = self.accumulated_paths[i]
            square_size = 0.15
            square = plt.Rectangle((target_grid[0] - square_size/2, target_grid[1] - square_size/2),
                                  square_size, square_size,
                                  color='green', linewidth=0)
            ax.add_patch(square)

        # Draw all accumulated paths
        for i, (path, explored_nodes, target_grid) in enumerate(self.accumulated_paths):
            is_current_step = (i == len(self.accumulated_paths) - 1) and (self.animation_phase == 2)

            if is_current_step:
                # Current step being animated - draw progressively
                # First half of animation: show explored nodes appearing one by one
                # Second half of animation: show yellow line growing

                if self.path_progress < 0.5:
                    # First half: Draw explored nodes progressively (0.0 to 0.5 -> 0% to 100% of nodes)
                    if explored_nodes:
                        progress_in_first_half = self.path_progress / 0.5  # Map 0.0-0.5 to 0.0-1.0
                        num_explored = max(1, int(len(explored_nodes) * progress_in_first_half))
                        # Debug output
                        if i == len(self.accumulated_paths) - 1 and self.phase_timer % 30 == 0:
                            print(f"  Drawing {num_explored}/{len(explored_nodes)} explored nodes, progress={self.path_progress:.2f}")
                        for j in range(num_explored):
                            node = explored_nodes[j]
                            # Draw hollow blue circle
                            ax.plot(node[0], node[1], "o", color='blue', markersize=3,
                                   markerfacecolor='none', markeredgewidth=0.5)
                else:
                    # Second half: Draw all explored nodes + growing dark blue line (0.5 to 1.0 -> 0% to 100% of path)
                    # Draw all explored nodes
                    if explored_nodes:
                        for node in explored_nodes:
                            # Draw hollow blue circle
                            ax.plot(node[0], node[1], "o", color='blue', markersize=3,
                                   markerfacecolor='none', markeredgewidth=0.5)

                    # Draw growing dark blue line
                    if path and len(path) > 1:
                        progress_in_second_half = (self.path_progress - 0.5) / 0.5  # Map 0.5-1.0 to 0.0-1.0
                        num_points = max(2, int(len(path) * progress_in_second_half))
                        # Debug output
                        if i == len(self.accumulated_paths) - 1 and self.phase_timer % 30 == 0:
                            print(f"  Drawing {num_points}/{len(path)} path points, progress={self.path_progress:.2f}")
                        if num_points >= 2:
                            path_x = [p[0] for p in path[:num_points]]
                            path_y = [p[1] for p in path[:num_points]]
                            ax.plot(path_x, path_y, "-", color='darkblue', linewidth=0.4)
            else:
                # Previous steps - draw fully
                # Draw all explored nodes (hollow blue circles)
                if explored_nodes:
                    for node in explored_nodes:
                        # Draw hollow blue circle
                        ax.plot(node[0], node[1], "o", color='blue', markersize=3,
                               markerfacecolor='none', markeredgewidth=0.5)

                # Draw full path (thin dark blue line connecting waypoints)
                if path and len(path) > 1:
                    path_x = [p[0] for p in path]
                    path_y = [p[1] for p in path]
                    ax.plot(path_x, path_y, "-", color='darkblue', linewidth=0.4)

        # Styling - no axis labels, no ticks, black border
        ax.set_title('A* Path Planning', color='#64C8FF', fontsize=9, fontweight='bold', pad=8)
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
        self.cached_astar_graph = surf
        self.cached_astar_progress = self.path_progress
        self.cached_astar_phase = self.animation_phase
        self.cached_astar_paths_len = len(self.accumulated_paths)

        return surf

    def draw_astar_graph(self, x, y, width, height):
        """Draw A* path planning graph showing all accumulated paths"""
        # Draw graph - show all accumulated paths (don't clear between steps)
        if len(self.accumulated_paths) == 0:
            # Draw empty box (no text)
            pygame.draw.rect(self.screen, BOX_COLOR, (x, y, width, height))
            pygame.draw.rect(self.screen, BOX_BORDER_COLOR, (x, y, width, height), 2)
            return

        # Generate and display graph surface
        graph_surf = self.create_astar_graph_surface(width, height)
        self.screen.blit(graph_surf, (x, y))

    def render(self):
        """Render the visualization"""
        # Clear screen
        self.screen.fill(BG_COLOR)

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

        # A* path planning graph on right
        astar_graph_x = info_x + graph_width + graph_gap
        self.draw_astar_graph(astar_graph_x, graph_y, graph_width, graph_height)

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

                    # Check if there's a reshuffle for this step
                    self.reshuffled_cube_idx = None
                    for reshuffle in self.current_episode.reshuffles:
                        if reshuffle['step'] == action_idx:
                            self.reshuffled_cube_idx = reshuffle['cube_idx']
                            break

                    # Update observation components (use correct metric keys)
                    self.current_obs_components['dist_to_ee'] = self.current_episode.metrics['distances_to_ee'][action_idx]
                    self.current_obs_components['dist_to_container'] = self.current_episode.metrics['distances_to_container'][action_idx]
                    self.current_obs_components['obstacle_proximity'] = self.current_episode.metrics['obstacle_proximities'][action_idx]
                    self.current_obs_components['reachability'] = self.current_episode.metrics['reachability_flags'][action_idx]
                    self.current_obs_components['path_clearance'] = self.current_episode.metrics['path_clearances'][action_idx]
                    self.current_obs_components['invalid_pick'] = 0.0
                    self.current_obs_components['items_left'] = len(self.current_episode.actions) - action_idx

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
                    self.current_obs_components['dist_to_origin'] = np.linalg.norm(self.base_env.object_positions[self.selected_cube_idx][:2])

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
                has_valid_path = (self.current_path is not None and
                                 len(self.current_path) > 1)

                if has_valid_path:
                    # Move to phase 2: Path animation
                    self.animation_phase = 2
                    self.phase_timer = 0
                    self.path_progress = 0.0  # Reset path animation

                    # Get path with explored nodes for visualization
                    _, explored_nodes = self._get_astar_path_with_explored(self.selected_cube_idx)

                    # Get target position
                    target_pos = self.current_episode.cube_positions[self.selected_cube_idx][:2]
                    target_grid = self.base_env.astar_estimator._world_to_grid(target_pos)

                    # Debug: Print number of explored nodes
                    print(f"Step {self.episode_step}: Explored {len(explored_nodes)} nodes, Path length: {len(self.current_path)}")

                    # Add current path to accumulated paths for A* graph
                    self.accumulated_paths.append((self.current_path.copy(), explored_nodes, target_grid))
                else:
                    # Skip phase 2 if no valid path (cube is at EE position)
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
                self.reshuffled_cube_idx = None  # Clear reshuffled cube

                # Return to phase 0: Brief idle before next cube
                self.animation_phase = 0
                self.phase_timer = 0

    def run(self):
        """Main loop"""
        print("\n" + "=" * 70)
        print("RL + A* EPISODE VISUALIZER")
        print("=" * 70)
        print(f"Grid Size: {self.grid_size}x{self.grid_size}")
        print(f"Number of Cubes: {self.num_cubes}")
        print(f"Model: {'Trained RL Agent' if self.model else 'Greedy Baseline'}")
        print("\nVisualization shows:")
        print("  - RL agent selecting best cube to pick (blue border)")
        print("  - A* planning curved obstacle-free path (yellow)")
        print("  - Cubes removed one by one after path execution")
        print("  - Complete episode without grid refresh")
        print("\nControls:")
        print("  SPACE: Play/Pause")
        print("  N: Next Step")
        print("  R: Reset Episode")
        print("  G: Generate New Episode")
        print("  +/-: Speed Control")
        print("  ESC/Q: Quit")
        print("=" * 70)
        print("\nPress SPACE to start!\n")

        while self.running:
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
        print("\n" + "=" * 70)
        print("VISUALIZER CLOSED")
        print("=" * 70)
        print(f"Total Episodes: {self.episode_count}")
        print("=" * 70)


def main():
    """Main function"""
    args = parse_args()

    # Validate inputs
    if args.num_cubes > args.grid_size * args.grid_size:
        print(f"ERROR: Number of cubes ({args.num_cubes}) exceeds grid capacity")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("INITIALIZING MAPPO TWO-AGENT + A* EPISODE VISUALIZER")
    print("=" * 70)
    print(f"Grid Size: {args.grid_size}x{args.grid_size}")
    print(f"Number of Cubes: {args.num_cubes}")
    print(f"DDQN Model (Agent 1): {args.ddqn_model_path if args.ddqn_model_path else 'None (random)'}")
    print(f"MAPPO Model (Agent 2): {args.mappo_model_path if args.mappo_model_path else 'None (random)'}")
    print("=" * 70)

    # Create and run visualizer
    visualizer = AStarRLEpisodeVisualizer(
        grid_size=args.grid_size,
        num_cubes=args.num_cubes,
        ddqn_model_path=args.ddqn_model_path,
        mappo_model_path=args.mappo_model_path,
        window_width=args.window_width,
        window_height=args.window_height,
        initial_fps=args.fps
    )

    visualizer.run()


if __name__ == "__main__":
    main()
