"""
RL + RRT Episode Visualizer

Visualizes RL agent selecting cubes and RRT planning obstacle-free paths.
Shows one complete episode with tree exploration and interactive charts.
"""

import os
import sys
import argparse
import warnings
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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

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
                 reachability, path_clearance, dist_to_ee, dist_to_container):
        """Add a step to the episode"""
        self.actions.append(action)
        self.paths.append(path)
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

        # Create environment (using PythonRobotics RRT for visualization)
        self.env = ObjectSelectionEnvRRTViz(
            franka_controller=None,
            max_objects=num_cubes,
            max_steps=num_cubes * 2,
            num_cubes=num_cubes,
            render_mode=None,
            dynamic_obstacles=False,
            training_grid_size=grid_size,
            execute_picks=False
        )
        vec_env = DummyVecEnv([lambda: self.env])
        self.vec_env = vec_env

        # Load model if provided
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = PPO.load(model_path, env=vec_env)
        else:
            pass  # Using greedy baseline

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

        # Reward components for current step
        self.current_reward_components = {
            'r_total': 0.0,
            'r_dist_ee': 0.0,
            'r_dist_c': 0.0,
            'r_reach': 0.0,
            'r_clear': 0.0,
            'r_invalid': 0.0
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

        # Generate first episode
        self.generate_new_episode()

    def generate_new_episode(self):
        """Generate a new episode by running the RL agent"""
        print(f"\nGenerating episode {self.episode_count + 1}...")

        # Reset environment
        obs, info = self.env.reset()

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

        while not done and step < self.env.max_steps:
            # Get action from model or greedy baseline
            if self.model is not None:
                norm_obs = self.vec_env.normalize_obs(obs)
                action, _ = self.model.predict(norm_obs, deterministic=True)
                action = int(action)
            else:
                action = self._greedy_action()

            # Update RRT grid with current unpicked cubes as obstacles
            # (exclude the target cube we're planning to pick)
            self._update_rrt_grid_for_planning(action)

            # Get RRT path before taking action
            path = self._get_rrt_path(action)

            # Calculate metrics
            path_length = self._calculate_path_length(path) if path else 999.0
            obstacle_proximity = self._calculate_obstacle_proximity(action)
            reachability = path is not None
            path_clearance = self._calculate_path_clearance(path) if path else 0.0
            dist_to_ee = np.linalg.norm(self.env.object_positions[action][:2] - self.env.ee_position[:2])
            dist_to_container = np.linalg.norm(self.env.object_positions[action][:2] - self.env.container_position[:2])

            # Take action
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # NOTE: EE position stays at home position (not updated)
            # Path planning always starts from same EE position for all picks

            # Record step
            episode.add_step(action, path, reward, path_length, obstacle_proximity,
                           reachability, path_clearance, dist_to_ee, dist_to_container)
            episode.picked_cubes.append(action)

            step += 1

        self.current_episode = episode
        self.episode_step = 0
        self.episode_count += 1
        self.episode_history.append(episode)

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

    def _get_rrt_path(self, cube_idx: int) -> Optional[List[Tuple[int, int]]]:
        """Get RRT path from end-effector to cube"""
        if cube_idx >= self.env.total_objects:
            return None

        ee_grid = self.env.rrt_estimator._world_to_grid(self.env.ee_position[:2])
        cube_pos = self.env.object_positions[cube_idx]
        goal_grid = self.env.rrt_estimator._world_to_grid(cube_pos[:2])

        # Check if start == goal
        if ee_grid == goal_grid:
            return []  # Return empty list

        # Update RRT grid for this specific target
        self._update_rrt_grid_for_planning(cube_idx)

        # Run RRT search (using PythonRobotics planning method)
        rx, ry = self.env.rrt_estimator.planning(ee_grid[0], ee_grid[1], goal_grid[0], goal_grid[1])

        # Convert to list of tuples
        # NOTE: RRT returns path from START to GOAL (already in correct order)
        if rx is None or ry is None:
            print(f"RRT failed for cube {cube_idx}")
            return []  # No path found
        else:
            path = [(rx[i], ry[i]) for i in range(len(rx))]
            return path

    def _get_astar_fallback_path(self, cube_idx: int, ee_grid: Tuple[int, int], goal_grid: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Fallback to A* when RRT fails"""
        from src.rl.path_estimators import AStarPathEstimator

        # Create A* estimator with same grid parameters
        astar = AStarPathEstimator(
            grid_size=self.grid_size,
            cell_size=0.13 if self.grid_size > 3 else 0.15
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
            cell_size = 0.13 if self.grid_size > 3 else 0.15
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

    def create_reward_graph_surface(self, width, height):
        """Create reward spike graph using seaborn/matplotlib"""
        if len(self.reward_history) < 2:
            # Return empty surface
            surf = pygame.Surface((width, height))
            surf.fill(BOX_COLOR)
            return surf

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100, facecolor='#282828')
        ax.set_facecolor('#282828')

        # Create gradient area chart
        steps = np.array(range(1, len(self.reward_history) + 1))
        rewards = np.array(self.reward_history)

        # Calculate reward thresholds for color grading
        min_reward = min(rewards)
        max_reward = max(rewards)
        reward_range = max_reward - min_reward if max_reward != min_reward else 1

        # Create smooth gradient colors for each point
        colors = []
        for r in rewards:
            # Normalize reward to 0-1 range
            normalized = (r - min_reward) / reward_range if reward_range > 0 else 0.5

            if normalized < 0.33:  # Low rewards - red gradient
                intensity = normalized / 0.33
                colors.append((0.5 + 0.5*intensity, 0, 0))
            elif normalized < 0.67:  # Medium rewards - orange gradient
                intensity = (normalized - 0.33) / 0.34
                colors.append((1.0, 0.5*intensity, 0))
            else:  # High rewards - green gradient
                intensity = (normalized - 0.67) / 0.33
                colors.append((1.0 - intensity, 0.5 + 0.5*intensity, 0))

        # Plot gradient area chart (fill between with varying colors)
        for i in range(len(steps) - 1):
            # Create gradient fill for this segment
            ax.fill_between(steps[i:i+2], rewards[i:i+2],
                          color=colors[i], alpha=0.7, linewidth=0)

        # Draw smooth line on top
        ax.plot(steps, rewards, color='white', linewidth=1.5, alpha=0.8)

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

        # Convert to pygame surface (fix for newer matplotlib)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        # Get the RGBA buffer and convert to RGB
        buf = canvas.buffer_rgba()
        size = canvas.get_width_height()

        # Create surface from RGBA buffer
        surf = pygame.image.frombuffer(buf, size, "RGBA")
        plt.close(fig)

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
                    pygame.draw.lines(self.screen, PATH_COLOR, False, animated_points, 3)

                    # Draw red filled circle at EE (start of path)
                    start_x, start_y = animated_points[0]
                    pygame.draw.circle(self.screen, (255, 0, 0), (int(start_x), int(start_y)), 5)

                    # Draw small yellow circle at current end of path
                    end_x, end_y = animated_points[-1]
                    pygame.draw.circle(self.screen, (255, 200, 0), (int(end_x), int(end_y)), 4)

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
        box_y = self.draw_box(x, current_y, width, 175, "Episode Info")
        current_reward = sum(self.current_episode.rewards[:self.episode_step]) if self.episode_step > 0 else 0.0

        # Get distance to target cube (Dist->Target)
        dist_to_target = 0.0
        if self.selected_cube_idx is not None and self.selected_cube_idx < len(self.env.object_positions):
            target_pos = self.env.object_positions[self.selected_cube_idx][:2]
            dist_to_target = np.linalg.norm(target_pos - self.env.ee_position[:2])

        # Get playback status
        status_text = "Paused" if self.paused else "Playing"

        info_items = [
            ("Episode:", f"{self.episode_count}"),
            ("Step:", f"{self.episode_step}/{len(self.current_episode.actions)}"),
            ("FPS:", f"{int(self.clock.get_fps())}"),
            ("Items left:", f"{len(self.current_episode.actions) - self.episode_step}"),
            ("Dist->Target:", f"{dist_to_target:.3f}"),
            ("Status:", status_text),
        ]

        for label, value in info_items:
            label_surf = self.font_small.render(label, True, LABEL_COLOR)
            value_surf = self.font_small.render(value, True, VALUE_COLOR)
            self.screen.blit(label_surf, (x + 15, box_y))
            self.screen.blit(value_surf, (x + 200, box_y))
            box_y += 20

        current_y += 185

        # Cumulative Rewards Box
        box_y = self.draw_box(x, current_y, width, 70, "Cumulative Rewards")
        total_label = self.font_large.render("Total:", True, LABEL_COLOR)
        total_value = self.font_large.render(f"{current_reward:.2f}", True, HEADER_COLOR)
        self.screen.blit(total_label, (x + 15, box_y))
        self.screen.blit(total_value, (x + 200, box_y))

        current_y += 80

        # Observation Metrics Box - Show REWARD COMPONENTS calculated from metrics
        box_y = self.draw_box(x, current_y, width, 160, "Observation Metrics")

        # Only show rewards if episode has started (episode_step > 0)
        if self.episode_step > 0 and self.current_episode:
            # Calculate reward components from observation metrics (matching A* formula exactly)
            # 1. Path length reward (max 5.0) - using distance to EE as proxy for path length
            dist_to_ee = self.current_obs_components['dist_to_ee']
            normalized_path = (dist_to_ee - 0.3) / 0.6
            normalized_path = np.clip(normalized_path, 0.0, 1.0)
            r_path = 5.0 * (1.0 - normalized_path)

            # 2. Obstacle proximity reward (max 3.0)
            obstacle_score = self.current_obs_components['obstacle_proximity']
            r_obstacle = 3.0 * (1.0 - obstacle_score)

            # 3. Reachability (1.0 if reachable, 0.0 if not)
            r_reachability = 1.0 if self.current_obs_components['reachability'] > 0.5 else 0.0

            # 4. Path clearance (already a score 0-1)
            r_clearance = self.current_obs_components['path_clearance']

            # 5. Distance to container reward (not used in actual reward formula, but shown for consistency)
            # Normalize similar to path length: shorter distance = higher reward
            dist_container = self.current_obs_components['dist_to_container']
            normalized_container = (dist_container - 0.3) / 0.6
            normalized_container = np.clip(normalized_container, 0.0, 1.0)
            r_container = 5.0 * (1.0 - normalized_container)
        else:
            # Before episode starts, all values are 0
            r_path = 0.0
            r_obstacle = 0.0
            r_reachability = 0.0
            r_clearance = 0.0
            r_container = 0.0

        obs_items = [
            ("Path length reward:", f"{r_path:.3f}"),
            ("Obstacle avoid reward:", f"{r_obstacle:.3f}"),
            ("Reachability reward:", f"{r_reachability:.3f}"),
            ("Path clearance reward:", f"{r_clearance:.3f}"),
            ("Dist to container reward:", f"{r_container:.3f}"),
        ]

        for label, value in obs_items:
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

    def create_rrt_graph_surface(self, width, height):
        """Create RRT path planning visualization with tree exploration (PythonRobotics style)"""
        if not self.current_path or len(self.current_path) < 2:
            # Return empty surface
            surf = pygame.Surface((width, height))
            surf.fill(BOX_COLOR)
            return surf

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100, facecolor='#282828')
        ax.set_facecolor('#282828')

        # Collect obstacle points (PythonRobotics style: plot as black dots)
        ox_list = []
        oy_list = []

        # Add static obstacles
        for ox, oy in self.static_obstacles:
            ox_list.append(ox)
            oy_list.append(oy)

        # Add unpicked cubes as obstacles
        if self.current_episode:
            picked_cubes = set(self.current_episode.picked_cubes[:self.episode_step])
            for i in range(len(self.current_episode.cube_positions)):
                if i not in picked_cubes and i != self.selected_cube_idx:
                    pos = self.current_episode.cube_positions[i]
                    grid_col, grid_row = self.env.rrt_estimator._world_to_grid(pos[:2])
                    ox_list.append(grid_col)
                    oy_list.append(grid_row)

        # Simulate RRT tree visualization (green branches)
        # RRT builds a tree by randomly sampling and connecting nodes
        if self.animation_phase == 2 and self.path_progress > 0:
            path_x = [p[0] for p in self.current_path]
            path_y = [p[1] for p in self.current_path]

            # Calculate how many path points to show based on animation progress
            num_points = max(2, int(len(self.current_path) * self.path_progress))

            # Simulate RRT tree branches (green lines showing exploration)
            # Draw simplified tree branches along the path
            for i in range(1, num_points):
                # Draw main branch (parent to child connection)
                ax.plot([path_x[i-1], path_x[i]], [path_y[i-1], path_y[i]],
                       "-g", linewidth=1, alpha=0.3)

                # Add some random exploration branches (to simulate RRT's random sampling)
                if i % 2 == 0 and i < num_points - 1:
                    # Random branch offset
                    offset_x = np.random.uniform(-0.5, 0.5)
                    offset_y = np.random.uniform(-0.5, 0.5)
                    branch_x = path_x[i] + offset_x
                    branch_y = path_y[i] + offset_y

                    # Check if branch is within bounds
                    if 0 <= branch_x < self.grid_size and 0 <= branch_y < self.grid_size:
                        ax.plot([path_x[i], branch_x], [path_y[i], branch_y],
                               "-g", linewidth=0.5, alpha=0.2)

        # Plot obstacles (black dots - PythonRobotics style)
        if ox_list:
            ax.plot(ox_list, oy_list, ".k", markersize=10)

        # Get start and goal from path
        path_x = [p[0] for p in self.current_path]
        path_y = [p[1] for p in self.current_path]

        # Plot start (green circle - PythonRobotics style)
        ax.plot(path_x[0], path_y[0], "og", markersize=10)

        # Plot goal (blue X - PythonRobotics style)
        ax.plot(path_x[-1], path_y[-1], "xb", markersize=12)

        # Draw animated portion of final path (red line - PythonRobotics style)
        num_points = max(2, int(len(self.current_path) * self.path_progress))
        animated_x = path_x[:num_points]
        animated_y = path_y[:num_points]
        ax.plot(animated_x, animated_y, "-r", linewidth=2)

        # Styling (PythonRobotics style)
        ax.set_xlabel('X', color='white', fontsize=7)
        ax.set_ylabel('Y', color='white', fontsize=7)
        ax.set_title('RRT Path Planning', color='#64C8FF', fontsize=9, fontweight='bold')
        ax.tick_params(colors='white', labelsize=6)
        ax.grid(True, alpha=0.3, color='white')
        ax.set_aspect('equal')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')

        # Convert to pygame surface
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        # Get the RGBA buffer
        buf = canvas.buffer_rgba()
        size = canvas.get_width_height()

        # Create surface from RGBA buffer
        surf = pygame.image.frombuffer(buf, size, "RGBA")
        plt.close(fig)

        return surf

    def draw_rrt_graph(self, x, y, width, height):
        """Draw RRT path planning graph"""
        # Draw graph - only show when path is being animated
        if self.animation_phase != 2 or not self.current_path or len(self.current_path) < 2:
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

                    # Update reward components and observation metrics
                    self.current_reward_components['r_total'] = self.current_episode.rewards[action_idx]
                    # For now, set individual components to 0 (would need to extract from env)
                    self.current_reward_components['r_dist_ee'] = 0.0
                    self.current_reward_components['r_dist_c'] = 0.0
                    self.current_reward_components['r_reach'] = 0.0
                    self.current_reward_components['r_clear'] = 0.0
                    self.current_reward_components['r_invalid'] = 0.0

                    # Update observation components (use correct metric keys)
                    self.current_obs_components['dist_to_ee'] = self.current_episode.metrics['distances_to_ee'][action_idx]
                    self.current_obs_components['dist_to_container'] = self.current_episode.metrics['distances_to_container'][action_idx]
                    self.current_obs_components['obstacle_proximity'] = self.current_episode.metrics['obstacle_proximities'][action_idx]
                    self.current_obs_components['reachability'] = self.current_episode.metrics['reachability_flags'][action_idx]
                    self.current_obs_components['path_clearance'] = self.current_episode.metrics['path_clearances'][action_idx]
                    self.current_obs_components['items_left'] = len(self.current_episode.actions) - action_idx
                    self.current_obs_components['dist_to_origin'] = np.linalg.norm(self.env.object_positions[self.selected_cube_idx][:2])

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
                # Path is valid if it exists, is not empty, and has more than 1 waypoint
                has_valid_path = (self.current_path is not None and
                                 isinstance(self.current_path, list) and
                                 len(self.current_path) > 1)

                if has_valid_path:
                    # Move to phase 2: Path animation
                    self.animation_phase = 2
                    self.phase_timer = 0
                    self.path_progress = 0.0  # Reset path animation
                else:
                    # Skip phase 2 if no valid path (cube is at EE position or RRT failed)
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
