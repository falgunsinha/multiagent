"""
RL + A* Episode Visualizer

Visualizes RL agent selecting cubes and A* planning obstacle-free paths.
Shows one complete episode with curved paths and interactive charts.
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Catmull-Rom spline from local utils
from src.utils.catmull_rom_spline_path import catmull_rom_spline

from src.rl.object_selection_env_astar import ObjectSelectionEnvAStar
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


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='RL + A* Episode Visualizer')
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


class AStarRLEpisodeVisualizer:
    """Episode visualizer for RL + A* pick-and-place"""

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
        pygame.display.set_caption("RL + A* Episode Visualizer")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 22)

        # Create environment
        self.env = ObjectSelectionEnvAStar(
            franka_controller=None,
            max_objects=num_cubes,
            max_steps=num_cubes * 2,
            num_cubes=num_cubes,
            render_mode=None,
            dynamic_obstacles=False,
            training_grid_size=grid_size
        )
        vec_env = DummyVecEnv([lambda: self.env])
        self.vec_env = vec_env

        # Load model if provided
        self.model = None
        if model_path and os.path.exists(model_path):
            print(f"[MODEL] Loading trained model from {model_path}")
            self.model = PPO.load(model_path, env=vec_env)
            print("[MODEL] Model loaded successfully")
        else:
            print("[MODEL] No model provided - using greedy baseline")

        # Episode state
        self.current_episode = None
        self.episode_step = 0
        self.episode_count = 0
        self.episode_history = deque(maxlen=50)
        self.chart_fig = None

        # Visualization state
        self.selected_cube_idx = None
        self.current_path = None
        self.static_obstacles = []
        self.running = True
        self.paused = True
        self.playback_speed = 1.0
        self.auto_advance = True

        # Animation state
        self.animation_progress = 0.0
        self.animation_duration = 90  # frames (increased for better visibility)
        self.animating = False

        # Generate first episode
        self.generate_new_episode()

    def generate_new_episode(self):
        """Generate a new episode by running the RL agent"""
        print(f"\n[EPISODE] Generating episode #{self.episode_count + 1}...")

        # Reset environment
        obs, info = self.env.reset()

        # Generate random obstacles (0-3, but only if there's room)
        # With many cubes, there may not be room for obstacles
        grid_capacity = self.grid_size * self.grid_size
        available_cells = grid_capacity - self.num_cubes
        max_obstacles = max(0, min(3, available_cells))
        num_obstacles = np.random.randint(0, max_obstacles + 1) if max_obstacles > 0 else 0
        self._add_random_obstacles(num_obstacles)

        # Create episode data
        episode = EpisodeData()
        episode.cube_positions = self.env.object_positions[:self.env.total_objects].copy()
        episode.obstacle_positions = [self.env.astar_estimator._grid_to_world(ox, oy)
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

            # Update A* grid with current unpicked cubes as obstacles
            # (exclude the target cube we're planning to pick)
            self._update_astar_grid_for_planning(action)

            # Get A* path before taking action
            path = self._get_astar_path(action)

            # Calculate metrics
            path_length = self._calculate_path_length(path) if path else 999.0
            obstacle_proximity = self._calculate_obstacle_proximity(action)
            reachability = path is not None
            path_clearance = self._calculate_path_clearance(path) if path else 0.0
            dist_to_ee = np.linalg.norm(self.env.object_positions[action][:2] - self.env.ee_position[:2])
            dist_to_container = 0.5

            # Take action
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # NOTE: EE position stays at home position (not updated)
            # Path planning always starts from same EE position for all picks

            # Record step
            episode.add_step(action, path, reward, path_length, obstacle_proximity,
                           reachability, path_clearance, dist_to_ee, dist_to_container)
            episode.picked_cubes.append(action)

            # LOG: Verify path is stored correctly
            print(f"  → Stored path for cube {action}: {len(path) if path else 0} waypoints")

            step += 1

        self.current_episode = episode
        self.episode_step = 0
        self.episode_count += 1
        self.episode_history.append(episode)

        print(f"[EPISODE] Generated episode with {len(episode.actions)} picks, total reward: {episode.metrics['total_reward']:.2f}")

        # Update charts
        if len(self.episode_history) >= 2:
            self.update_charts()

    def _add_random_obstacles(self, num_obstacles: int):
        """Add random obstacles to empty grid cells"""
        self.static_obstacles = []

        if num_obstacles == 0:
            return

        # Get cube positions to avoid
        cube_cells = set()
        for i in range(self.env.total_objects):
            pos = self.env.object_positions[i]
            grid_col, grid_row = self.env.astar_estimator._world_to_grid(pos[:2])
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

        print(f"[OBSTACLES] Added {len(self.static_obstacles)} random obstacles")
        # Note: A* grid will be updated before each path planning with current unpicked cubes

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
            world_pos = self.env.astar_estimator._grid_to_world(grid_x, grid_y)
            obstacle_positions.append(np.array([world_pos[0], world_pos[1], 0.0]))

        # Get unpicked cube positions (excluding target cube)
        unpicked_cube_positions = []
        for i in range(self.env.total_objects):
            if i not in self.env.objects_picked and i != target_cube_idx:
                unpicked_cube_positions.append(self.env.object_positions[i])

        # Update A* grid
        self.env.astar_estimator.update_occupancy_grid(
            object_positions=unpicked_cube_positions,
            obstacle_positions=obstacle_positions
        )

        print(f"[A* GRID] Updated for cube {target_cube_idx}: {len(unpicked_cube_positions)} unpicked cubes as obstacles, {len(obstacle_positions)} static obstacles")

        # DEBUG: Print obstacle map to see obstacles
        if target_cube_idx in [13, 18, 14]:  # First few cubes that fail
            print(f"[A* GRID] Obstacle map for cube {target_cube_idx}:")
            for row in range(self.grid_size):
                row_str = f"  Row {row}: "
                for col in range(self.grid_size):
                    val = self.env.astar_estimator.obstacle_map[col][row]
                    row_str += f"{int(val)} "
                print(row_str)

    def _greedy_action(self) -> int:
        """
        Greedy baseline: pick closest unpicked cube using A* path length.

        This ensures we pick cubes that are actually reachable, not just physically close.
        """
        min_path_length = float('inf')
        best_action = 0

        for i in range(self.env.total_objects):
            if i not in self.env.objects_picked:
                # Use A* path length instead of Euclidean distance
                # This accounts for obstacles and ensures we pick reachable cubes
                path_length = self.env.astar_estimator.estimate_path_length(
                    self.env.ee_position[:2],
                    self.env.object_positions[i][:2]
                )

                if path_length < min_path_length:
                    min_path_length = path_length
                    best_action = i

        return best_action

    def _get_astar_path(self, cube_idx: int) -> Optional[List[Tuple[int, int]]]:
        """Get A* path from end-effector to cube"""
        if cube_idx >= self.env.total_objects:
            return None

        ee_grid = self.env.astar_estimator._world_to_grid(self.env.ee_position[:2])
        cube_pos = self.env.object_positions[cube_idx]
        goal_grid = self.env.astar_estimator._world_to_grid(cube_pos[:2])

        # LOG: Print detailed path information
        print(f"\n[A* PATH] Cube {cube_idx}:")
        print(f"  EE Position (world): {self.env.ee_position[:2]}")
        print(f"  EE Position (grid): {ee_grid}")
        print(f"  Cube Position (world): {cube_pos[:2]}")
        print(f"  Cube Position (grid):  {goal_grid}")

        # Check if start == goal (should never happen now - EE home cell is reserved)
        if ee_grid == goal_grid:
            print(f"  ⚠ Start and goal are the same! Returning empty path.")
            return []  # Return empty list instead of None

        # Run A* search (using PythonRobotics planning method)
        rx, ry = self.env.astar_estimator.planning(ee_grid[0], ee_grid[1], goal_grid[0], goal_grid[1])

        # Convert to list of tuples
        # NOTE: PythonRobotics A* returns path from GOAL to START, so we need to REVERSE it
        if rx is None or ry is None:
            path = None
        else:
            path = [(rx[i], ry[i]) for i in range(len(rx))]
            path.reverse()  # Reverse to get path from START to GOAL

        print(f"  A* Path (start->goal): {path}")
        print(f"  Path Length: {len(path) if path else 0}")
        if path:
            print(f"  Path waypoints: {' -> '.join([f'({p[0]},{p[1]})' for p in path])}")
        elif path is None:
            print(f"  ⚠ A* returned None - no path found! Returning empty list.")
            return []  # Return empty list instead of None

        return path

    def _calculate_path_length(self, path: List[Tuple[int, int]]) -> float:
        """Calculate path length in meters"""
        if not path or len(path) < 2:
            return 0.0

        length = 0.0
        for i in range(len(path) - 1):
            pos1 = self.env.astar_estimator._grid_to_world(path[i][0], path[i][1])
            pos2 = self.env.astar_estimator._grid_to_world(path[i+1][0], path[i+1][1])
            length += np.linalg.norm(pos2 - pos1)

        return length

    def _calculate_obstacle_proximity(self, cube_idx: int) -> float:
        """Calculate minimum distance to obstacles"""
        if not self.static_obstacles:
            return 999.0

        cube_pos = self.env.object_positions[cube_idx][:2]
        min_dist = float('inf')

        for ox, oy in self.static_obstacles:
            obs_pos = self.env.astar_estimator._grid_to_world(ox, oy)
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

    def update_charts(self):
        """Update Plotly interactive charts"""
        if len(self.episode_history) < 2:
            return

        # Prepare data
        episodes = list(range(1, len(self.episode_history) + 1))
        rewards = [ep.metrics['total_reward'] for ep in self.episode_history]
        path_lengths = [np.mean(ep.metrics['path_lengths']) if ep.metrics['path_lengths'] else 0
                       for ep in self.episode_history]
        obstacle_proximities = [np.mean(ep.metrics['obstacle_proximities']) if ep.metrics['obstacle_proximities'] else 0
                               for ep in self.episode_history]

        # Create subplots
        self.chart_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Episode Rewards', 'Average Path Length',
                          'Obstacle Proximity', 'Path Clearance Distribution'),
            vertical_spacing=0.12,
            horizontal_spacing=0.10
        )

        # Reward curve
        self.chart_fig.add_trace(
            go.Scatter(x=episodes, y=rewards, mode='lines+markers', name='Reward',
                      line=dict(color='rgb(100, 200, 100)', width=2)),
            row=1, col=1
        )

        # Path length curve
        self.chart_fig.add_trace(
            go.Scatter(x=episodes, y=path_lengths, mode='lines+markers', name='Path Length',
                      line=dict(color='rgb(255, 200, 0)', width=2)),
            row=1, col=2
        )

        # Obstacle proximity
        self.chart_fig.add_trace(
            go.Scatter(x=episodes, y=obstacle_proximities, mode='lines+markers', name='Obstacle Proximity',
                      line=dict(color='rgb(255, 100, 100)', width=2)),
            row=2, col=1
        )

        # Path clearance distribution
        all_clearances = []
        for ep in self.episode_history:
            all_clearances.extend(ep.metrics['path_clearances'])

        if all_clearances:
            self.chart_fig.add_trace(
                go.Histogram(x=all_clearances, nbinsx=20, name='Path Clearance',
                           marker=dict(color='rgb(100, 150, 255)')),
                row=2, col=2
            )

        # Update layout
        self.chart_fig.update_layout(
            height=600,
            showlegend=False,
            template='plotly_dark',
            title_text=f"RL + A* Performance Metrics (Last {len(self.episode_history)} Episodes)"
        )

        # Show chart in browser (silent)
        self.chart_fig.show()

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
                    print(f"[PLAYBACK] {'Paused' if self.paused else 'Playing'}")

                elif event.key == pygame.K_n:
                    # Next step
                    if self.episode_step < len(self.current_episode.actions):
                        self.episode_step += 1
                        self.animating = True
                        self.animation_progress = 0.0
                        print(f"[PLAYBACK] Step {self.episode_step}/{len(self.current_episode.actions)}")

                elif event.key == pygame.K_r:
                    # Reset episode
                    self.episode_step = 0
                    self.animating = False
                    print("[PLAYBACK] Reset to start")

                elif event.key == pygame.K_g:
                    # Generate new episode
                    self.generate_new_episode()

                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.playback_speed = min(5.0, self.playback_speed + 0.5)
                    print(f"[PLAYBACK] Speed: {self.playback_speed}x")

                elif event.key == pygame.K_MINUS:
                    self.playback_speed = max(0.5, self.playback_speed - 0.5)
                    print(f"[PLAYBACK] Speed: {self.playback_speed}x")

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
        """Draw obstacles"""
        for ox, oy in self.static_obstacles:
            x_center = grid_x + ox * cell_size + cell_size // 2
            y_center = grid_y + oy * cell_size + cell_size // 2

            size = cell_size // 2
            pygame.draw.rect(self.screen, OBSTACLE_COLOR,
                           (x_center - size // 2, y_center - size // 2, size, size))

    def draw_cubes(self, cell_size, grid_x, grid_y):
        """Draw cubes - only unpicked cubes, selected cube has blue border"""
        if not self.current_episode:
            return

        # Get list of picked cubes from PREVIOUS steps (not including current step)
        # When episode_step = 1, we're showing step 1 (action index 0), so hide nothing
        # When episode_step = 2, we're showing step 2 (action index 1), so hide cubes from step 1
        # After animation completes, the cube disappears for the next step
        if self.animating:
            # During animation: show the cube we're picking (don't hide it yet)
            picked_cubes = set(self.current_episode.picked_cubes[:self.episode_step - 1])
        else:
            # After animation: hide the cube we just picked
            picked_cubes = set(self.current_episode.picked_cubes[:self.episode_step])

        for i in range(len(self.current_episode.cube_positions)):
            # Skip picked cubes
            if i in picked_cubes:
                continue

            pos = self.current_episode.cube_positions[i]
            grid_col, grid_row = self.env.astar_estimator._world_to_grid(pos[:2])

            if 0 <= grid_col < self.grid_size and 0 <= grid_row < self.grid_size:
                x = grid_x + grid_col * cell_size + cell_size // 2
                y = grid_y + grid_row * cell_size + cell_size // 2

                # Draw cube
                size = cell_size // 2
                pygame.draw.rect(self.screen, CUBE_COLOR,
                               (x - size // 2, y - size // 2, size, size))

                # Highlight selected cube with BLUE border ONLY
                if i == self.selected_cube_idx:
                    border_width = 4
                    pygame.draw.rect(self.screen, SELECTED_BORDER_COLOR,
                                   (x - size // 2 - border_width, y - size // 2 - border_width,
                                    size + border_width * 2, size + border_width * 2), border_width)

    def draw_agent(self, cell_size, grid_x, grid_y, grid_h):
        """Draw agent (end-effector) inside the bottom-middle grid cell"""
        # EE is at grid cell (grid_size//2, grid_size-1) - bottom-middle cell
        # In screen coordinates: row 0 is TOP, row (grid_size-1) is BOTTOM
        ee_col = self.grid_size // 2  # Middle column
        ee_row = self.grid_size - 1  # Bottom row (last row in screen coordinates)

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
        # DEBUG: Log why we might skip drawing
        if not self.animating:
            # Don't log every frame, only when state changes
            return

        # current_path should never be None now (always empty list [] if no path)
        # But keep this check for safety
        if self.current_path is None:
            if self.animation_progress < 2:
                print(f"[DRAW PATH] Skipping - current_path is None (shouldn't happen!)")
            return

        # Skip drawing if path is empty (cube is at same position as EE)
        if len(self.current_path) == 0:
            if self.animation_progress < 2:
                print(f"[DRAW PATH] Skipping - path is empty (cube at same position as EE)")
            return

        agent_x, agent_y, agent_radius = agent_pos

        # Use last waypoint in A* path as target
        target_grid_col, target_grid_row = self.current_path[-1]
        if self.animation_progress < 2:
            print(f"[DRAW PATH] Using A* path target: ({target_grid_col}, {target_grid_row})")

        target_x = grid_x + target_grid_col * cell_size + cell_size // 2
        target_y = grid_y + target_grid_row * cell_size + cell_size // 2

        # Build path points from A* grid coordinates
        # A* path includes start position (EE grid cell), but we want to start from agent position (outside grid)
        # So we skip the first waypoint (EE position) and start from agent position outside grid
        points = []

        # Add agent position as start (outside grid at bottom)
        points.append((agent_x, agent_y))

        # Add all A* waypoints EXCEPT the first one (which is the EE position)
        # The first waypoint is the EE grid cell, which we're already representing with agent position
        for i, (grid_col, grid_row) in enumerate(self.current_path):
            if i == 0:
                continue  # Skip first waypoint (EE position)
            x = grid_x + grid_col * cell_size + cell_size // 2
            y = grid_y + grid_row * cell_size + cell_size // 2
            points.append((x, y))

        # LOG: Print path drawing information (only once per animation start)
        if self.animation_progress < 2:  # Only log at start of animation
            print(f"\n[DRAW PATH] Cube {self.selected_cube_idx}:")
            print(f"  Agent pos: ({agent_x}, {agent_y}), radius: {agent_radius}")
            print(f"  Target grid: ({target_grid_col}, {target_grid_row})")
            print(f"  Target screen: ({target_x}, {target_y})")
            print(f"  Points for spline ({len(points)}): {points[:5]}{'...' if len(points) > 5 else ''}")

        # Generate smooth curved path using Catmull-Rom Spline interpolation
        if len(points) >= 2:
            # Calculate adaptive number of interpolation points based on path length
            total_dist = sum(np.sqrt((points[i+1][0] - points[i][0])**2 +
                                    (points[i+1][1] - points[i][1])**2)
                           for i in range(len(points)-1))
            # More points for longer paths, ensuring smooth visualization
            num_interp_points = max(50, int(total_dist / 2))

            # Catmull-Rom spline requires at least 3 points
            # For 2 points, use linear interpolation
            if len(points) == 2:
                # Linear interpolation for straight line
                x1, y1 = points[0]
                x2, y2 = points[1]
                t_vals = np.linspace(0, 1, num_interp_points)
                curved_points = [(int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))) for t in t_vals]
            else:
                # Use Catmull-Rom spline from PythonRobotics (requires 3+ points)
                # Returns (x_coords, y_coords) as numpy arrays
                spline_x, spline_y = catmull_rom_spline(points, num_interp_points)
                curved_points = [(int(x), int(y)) for x, y in zip(spline_x, spline_y)]

            # LOG: Print spline result
            if self.animation_progress < 2:
                print(f"  Total distance: {total_dist:.1f}px")
                print(f"  Interpolation points: {num_interp_points}")
                print(f"  Curved points generated: {len(curved_points)}")

            # Draw curved path
            if len(curved_points) >= 2:
                # DEBUG: Draw test line to verify pygame is working
                if self.animation_progress < 2:
                    print(f"  Drawing path with color {PATH_COLOR}, width 3")
                    print(f"  First point: {curved_points[0]}, Last point: {curved_points[-1]}")
                    print(f"  Screen size: {self.screen.get_size()}")

                pygame.draw.lines(self.screen, PATH_COLOR, False, curved_points, 3)

                # DEBUG: Draw a test circle at start and end to verify rendering
                pygame.draw.circle(self.screen, (0, 255, 0), curved_points[0], 10)  # Green circle at start
                pygame.draw.circle(self.screen, (255, 0, 0), curved_points[-1], 10)  # Red circle at end

                if self.animation_progress < 2:
                    print(f"  ✓ Path drawn successfully!")
            else:
                if self.animation_progress < 2:
                    print(f"  ✗ Not enough curved points to draw!")
        else:
            if self.animation_progress < 2:
                print(f"  ✗ Not enough points for spline ({len(points)} < 2)!")

    def draw_info(self):
        """Draw info panel"""
        if not self.current_episode:
            return

        y_offset = 20
        line_height = 30

        # Title
        title = self.font_large.render("RL + A* Episode Visualizer", True, TEXT_COLOR)
        self.screen.blit(title, (20, y_offset))
        y_offset += line_height + 10

        # Calculate reward up to current step
        current_reward = 0.0
        if self.episode_step > 0:
            current_reward = sum(self.current_episode.rewards[:self.episode_step])

        # Episode info
        info_lines = [
            f"Episode: {self.episode_count}",
            f"Step: {self.episode_step}/{len(self.current_episode.actions)}",
            f"Total Reward: {current_reward:.2f}",
            f"Playback Speed: {self.playback_speed}x",
            f"Status: {'Paused' if self.paused else 'Playing'}",
        ]

        for line in info_lines:
            text = self.font_medium.render(line, True, TEXT_COLOR)
            self.screen.blit(text, (20, y_offset))
            y_offset += line_height

        # Controls
        y_offset += 20
        controls_title = self.font_medium.render("Controls:", True, TEXT_COLOR)
        self.screen.blit(controls_title, (20, y_offset))
        y_offset += line_height

        controls = [
            "SPACE: Play/Pause",
            "N: Next Step",
            "R: Reset Episode",
            "G: Generate New Episode",
            "+/-: Speed Control",
            "ESC/Q: Quit"
        ]

        for control in controls:
            text = self.font_small.render(control, True, TEXT_COLOR)
            self.screen.blit(text, (20, y_offset))
            y_offset += 25

    def render(self):
        """Render the visualization"""
        # Clear screen
        self.screen.fill(BG_COLOR)

        # Calculate layout
        win_w, win_h = self.screen.get_size()
        info_width = 350
        grid_area_w = win_w - info_width - 60
        grid_area_h = win_h - 100

        # Calculate grid size
        cell_size = min(grid_area_w // self.grid_size, grid_area_h // self.grid_size)
        grid_w = self.grid_size * cell_size
        grid_h = self.grid_size * cell_size

        # Center grid
        grid_x = info_width + (win_w - info_width - grid_w) // 2
        grid_y = (win_h - grid_h) // 2 - 30

        # Draw components
        self.draw_grid(cell_size, grid_x, grid_y)
        self.draw_obstacles(cell_size, grid_x, grid_y)
        self.draw_cubes(cell_size, grid_x, grid_y)
        agent_pos = self.draw_agent(cell_size, grid_x, grid_y, grid_h)
        self.draw_path(cell_size, grid_x, grid_y, grid_h, agent_pos)
        self.draw_info()

        # Update display
        pygame.display.flip()

    def update_playback(self):
        """Update episode playback"""
        if not self.current_episode or self.paused:
            return

        # Auto-advance if enabled
        if self.auto_advance and not self.animating:
            if self.episode_step < len(self.current_episode.actions):
                self.episode_step += 1
                self.animating = True
                self.animation_progress = 0.0

                # Update selected cube and path
                if self.episode_step > 0:
                    action_idx = self.episode_step - 1
                    self.selected_cube_idx = self.current_episode.actions[action_idx]
                    self.current_path = self.current_episode.paths[action_idx]
                    print(f"[ANIMATION] Step {self.episode_step}: Selected cube {self.selected_cube_idx}, Path length: {len(self.current_path) if self.current_path else 0}")
                    if self.current_path:
                        print(f"  Path waypoints: {self.current_path}")
            else:
                # Episode complete - generate new one
                print("[EPISODE] Complete! Generating new episode...")
                self.generate_new_episode()

        # Update animation
        if self.animating:
            self.animation_progress += self.playback_speed

            if self.animation_progress >= self.animation_duration:
                self.animating = False
                self.animation_progress = 0.0

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
    print("INITIALIZING RL + A* EPISODE VISUALIZER")
    print("=" * 70)
    print(f"Grid Size: {args.grid_size}x{args.grid_size}")
    print(f"Number of Cubes: {args.num_cubes}")
    print(f"Model Path: {args.model_path if args.model_path else 'None (greedy baseline)'}")
    print("=" * 70)

    # Create and run visualizer
    visualizer = AStarRLEpisodeVisualizer(
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
