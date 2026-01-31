"""
RL + A* Episode Visualizer
Shows complete pick-and-place episodes with RL decision-making and A* path planning.

Visualization:
- RL agent selects best cube to pick (based on 6 parameters)
- A* plans obstacle-free path from end-effector to selected cube
- Shows complete episode without grid refresh
- Curved path visualization using Catmull-Rom splines
- Interactive charts for rewards, path planning, and obstacle avoidance KPIs

Controls:
- SPACE: Start/Pause episode
- N: New episode
- R: Replay current episode
- +/-: Speed up/slow down
- ESC/Q: Quit

Usage:
    py -3.11 astar_rl_training_viz.py --grid_size 6 --num_cubes 25
    py -3.11 astar_rl_training_viz.py --grid_size 6 --num_cubes 25 --model_path models/ppo_final.zip
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pygame
import time
from datetime import datetime
import os
from typing import List, Tuple, Optional
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Suppress seaborn/matplotlib messages
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('seaborn').setLevel(logging.WARNING)

# Add project root to path
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Stable-Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Custom imports
from src.rl.object_selection_env_astar import ObjectSelectionEnvAStar
from src.rl.path_estimators import AStarPathEstimator

# Plotly for interactive charts (embedded in pygame window)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="RL + A* Episode Visualizer")
    parser.add_argument("--grid_size", type=int, default=6, help="Grid size (e.g., 6 for 6x6)")
    parser.add_argument("--num_cubes", type=int, default=25, help="Number of cubes to place")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained RL model (optional)")
    parser.add_argument("--window_width", type=int, default=1600, help="Window width")
    parser.add_argument("--window_height", type=int, default=900, help="Window height")
    parser.add_argument("--fps", type=int, default=5, help="Initial FPS (actions per second)")
    return parser.parse_args()


# Colors
BG_COLOR = (24, 24, 24)
GRID_COLOR = (60, 60, 60)
TEXT_COLOR = (220, 220, 220)
ACCENT_COLOR = (160, 200, 255)

# Object colors
CUBE_COLOR = (100, 200, 100)  # Green cubes
CUBE_PICKED_COLOR = (50, 100, 50)  # Dark green (picked)
OBSTACLE_COLOR = (255, 50, 50)  # Red obstacles
CONTAINER_COLOR = (100, 100, 200)  # Blue container

# Agent colors
AGENT_COLOR = (63, 127, 255)  # Blue agent (end-effector)
SELECTED_BORDER_COLOR = (63, 127, 255)  # Blue border for selected cube

# Path colors
PATH_COLOR = (255, 200, 0)  # Yellow curved path

# Grid cell colors
CELL_BG = (35, 35, 40)  # Dark background


def catmull_rom_spline(points: List[Tuple[float, float]], num_segments: int = 20) -> List[Tuple[float, float]]:
    """
    Generate smooth curve through points using Catmull-Rom spline.

    Args:
        points: List of (x, y) control points
        num_segments: Number of segments between each pair of points

    Returns:
        List of (x, y) points forming smooth curve
    """
    if len(points) < 2:
        return points

    if len(points) == 2:
        # Linear interpolation for 2 points
        return points

    # Add duplicate endpoints for Catmull-Rom
    extended_points = [points[0]] + points + [points[-1]]

    curve_points = []

    for i in range(len(points) - 1):
        p0 = extended_points[i]
        p1 = extended_points[i + 1]
        p2 = extended_points[i + 2]
        p3 = extended_points[i + 3] if i + 3 < len(extended_points) else extended_points[i + 2]

        for t in range(num_segments):
            t_norm = t / num_segments
            t2 = t_norm * t_norm
            t3 = t2 * t_norm

            # Catmull-Rom formula
            x = 0.5 * (
                (2 * p1[0]) +
                (-p0[0] + p2[0]) * t_norm +
                (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )

            y = 0.5 * (
                (2 * p1[1]) +
                (-p0[1] + p2[1]) * t_norm +
                (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )

            curve_points.append((x, y))

    # Add final point
    curve_points.append(points[-1])

    return curve_points


class EpisodeData:
    """Stores data for one complete episode"""

    def __init__(self):
        self.actions = []  # List of cube indices selected by RL
        self.paths = []  # List of A* paths (grid coordinates)
        self.rewards = []  # List of rewards for each action
        self.cube_positions = []  # Initial cube positions
        self.obstacle_positions = []  # Obstacle positions
        self.ee_start_position = None  # End-effector start position
        self.picked_cubes = []  # Track which cubes have been picked
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

    def add_step(self, action: int, path: Optional[List[Tuple[int, int]]], reward: float,
                 path_length: float, obstacle_proximity: float, reachability: bool,
                 path_clearance: float, dist_to_ee: float, dist_to_container: float):
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


class AStarRLVisualizer:
    """Episode visualizer for RL + A* pick-and-place"""

    def __init__(self, grid_size: int, num_cubes: int, model_path: Optional[str],
                 window_width: int, window_height: int, initial_fps: int):
        self.grid_size = grid_size
        self.num_cubes = num_cubes
        self.model_path = model_path
        self.window_width = window_width
        self.window_height = window_height
        self.fps = initial_fps

        # Validate inputs
        if num_cubes > grid_size * grid_size:
            raise ValueError(f"Number of cubes ({num_cubes}) exceeds grid capacity ({grid_size}x{grid_size}={grid_size*grid_size})")

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
        pygame.display.set_caption(f"RL + A* Episode Visualizer - {grid_size}x{grid_size} Grid, {num_cubes} Cubes")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_large = pygame.font.SysFont("segoeui", 26, bold=True)
        self.font_medium = pygame.font.SysFont("segoeui", 20)
        self.font_small = pygame.font.SysFont("segoeui", 18)

        # Create RL environment
        max_objects = grid_size * grid_size
        max_steps = max(50, min(200, num_cubes * 3))

        self.env = ObjectSelectionEnvAStar(
            franka_controller=None,
            max_objects=max_objects,
            max_steps=max_steps,
            num_cubes=num_cubes,
            training_grid_size=grid_size,
            render_mode=None
        )

        # Load RL model if provided
        self.model = None
        if model_path and os.path.exists(model_path):
            print(f"[VISUALIZER] Loading RL model from: {model_path}")
            # Create vectorized environment for model
            def make_env():
                env = ObjectSelectionEnvAStar(
                    franka_controller=None,
                    max_objects=max_objects,
                    max_steps=max_steps,
                    num_cubes=num_cubes,
                    training_grid_size=grid_size,
                    render_mode=None
                )
                return env

            vec_env = DummyVecEnv([make_env])
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

            self.model = PPO.load(model_path, env=vec_env)
            self.vec_env = vec_env
            print("[VISUALIZER] RL model loaded successfully")
        else:
            print("[VISUALIZER] No model provided - using greedy baseline")

        # Episode state
        self.running = True
        self.paused = True  # Start paused
        self.current_episode = None  # EpisodeData instance
        self.episode_step = 0  # Current step in episode playback
        self.episode_count = 0

        # Visualization state
        self.current_path = None
        self.selected_cube_idx = None
        self.ee_position = None  # Current end-effector position

        # Animation state
        self.action_delay = 1000 // initial_fps  # Milliseconds between actions
        self.last_action_time = 0

        # Store obstacle positions
        self.static_obstacles = []

        # Episode history for charts
        self.episode_history = deque(maxlen=50)  # Keep last 50 episodes

        # Plotly chart figure
        self.chart_fig = None

        # Reset environment and generate first episode
        self.generate_new_episode()

    def generate_new_episode(self):
        """Generate a new episode by running the RL agent"""
        print(f"\n[EPISODE] Generating new episode #{self.episode_count + 1}...")

        # Reset environment
        obs, info = self.env.reset()

        # Generate random obstacles (0-3, inversely proportional to cube count)
        max_obstacles = max(0, min(3, int(3 * (1 - self.num_cubes / (self.grid_size * self.grid_size)))))
        num_obstacles = np.random.randint(0, max_obstacles + 1)
        self._add_random_obstacles(num_obstacles)

        # Create episode data
        episode = EpisodeData()
        episode.cube_positions = self.env.object_positions[:self.env.total_objects].copy()
        episode.obstacle_positions = [self.env.astar_estimator._grid_to_world(ox, oy) for ox, oy in self.static_obstacles]
        episode.ee_start_position = self.env.ee_position.copy()

        # Run episode
        done = False
        step = 0

        while not done and step < self.env.max_steps:
            # Get action from model or greedy baseline
            if self.model is not None:
                # Normalize observation
                norm_obs = self.vec_env.normalize_obs(obs)
                action, _ = self.model.predict(norm_obs, deterministic=True)
                action = int(action)
            else:
                # Greedy baseline: pick closest unpicked cube
                action = self._greedy_action()

            # Get A* path before taking action
            path = self._get_astar_path(action)

            # Calculate metrics
            path_length = self._calculate_path_length(path) if path else 999.0
            obstacle_proximity = self._calculate_obstacle_proximity(action)
            reachability = path is not None
            path_clearance = self._calculate_path_clearance(path) if path else 0.0
            dist_to_ee = np.linalg.norm(self.env.object_positions[action][:2] - self.env.ee_position[:2])
            dist_to_container = 0.5  # Fixed container distance

            # Take action
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Record step
            episode.add_step(action, path, reward, path_length, obstacle_proximity,
                           reachability, path_clearance, dist_to_ee, dist_to_container)
            episode.picked_cubes.append(action)

            step += 1

        self.current_episode = episode
        self.episode_step = 0
        self.episode_count += 1
        self.episode_history.append(episode)

        print(f"[EPISODE] Generated episode with {len(episode.actions)} actions, total reward: {episode.metrics['total_reward']:.2f}")

        # Update charts
        self.update_charts()

    def _add_random_obstacles(self, num_obstacles: int):
        """Add random obstacles to empty grid cells"""
        self.static_obstacles = []

        if num_obstacles == 0:
            return

        # Get cube positions to avoid placing obstacles on cubes
        cube_cells = set()
        for i in range(self.env.total_objects):
            pos = self.env.object_positions[i]
            grid_col, grid_row = self.env.astar_estimator._world_to_grid(pos[:2])
            cube_cells.add((grid_col, grid_row))

        # Get ALL empty cells (cells without cubes)
        empty_cells = []
        for grid_x in range(self.grid_size):
            for grid_y in range(self.grid_size):
                if (grid_x, grid_y) not in cube_cells:
                    empty_cells.append((grid_x, grid_y))

        # Randomly select obstacle positions from empty cells
        obstacle_positions = []
        if len(empty_cells) >= num_obstacles:
            np.random.shuffle(empty_cells)
            selected_cells = empty_cells[:num_obstacles]

            for grid_x, grid_y in selected_cells:
                self.static_obstacles.append((grid_x, grid_y))
                world_pos = self.env.astar_estimator._grid_to_world(grid_x, grid_y)
                obstacle_positions.append(np.array([world_pos[0], world_pos[1], 0.0]))

        # Update A* grid with obstacles
        self.env.astar_estimator.update_occupancy_grid(
            object_positions=self.env.object_positions[:self.env.total_objects],
            obstacle_positions=obstacle_positions
        )

        print(f"[VISUALIZER] Added {len(self.static_obstacles)} random obstacles")

    def _greedy_action(self) -> int:
        """Greedy baseline: pick closest unpicked cube"""
        min_dist = float('inf')
        best_action = 0

        for i in range(self.env.total_objects):
            if i not in self.env.objects_picked:
                dist = np.linalg.norm(self.env.object_positions[i][:2] - self.env.ee_position[:2])
                if dist < min_dist:
                    min_dist = dist
                    best_action = i

        return best_action

    def _get_astar_path(self, cube_idx: int) -> Optional[List[Tuple[int, int]]]:
        """Get A* path from end-effector to cube"""
        if cube_idx >= self.env.total_objects:
            return None

        # Get current EE position and goal position
        ee_grid = self.env.astar_estimator._world_to_grid(self.env.ee_position[:2])
        cube_pos = self.env.object_positions[cube_idx]
        goal_grid = self.env.astar_estimator._world_to_grid(cube_pos[:2])

        # Run A* search (using PythonRobotics planning method)
        rx, ry = self.env.astar_estimator.planning(ee_grid[0], ee_grid[1], goal_grid[0], goal_grid[1])

        # Convert to list of tuples
        if rx is None or ry is None:
            return None

        path = [(rx[i], ry[i]) for i in range(len(rx))]
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
                          'Obstacle Proximity', 'Path Clearance'),
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

        # Show chart in browser
        self.chart_fig.show()

    def compute_layout(self):
        """Compute layout dimensions based on window size"""
        win_w, win_h = self.screen.get_size()

        # Sidebar width (right side) - larger for graphs
        sidebar_w = max(450, int(win_w * 0.4))

        # Available space for grid (left half)
        available_w = win_w - sidebar_w - 40  # 40px padding
        available_h = win_h - 40  # 40px padding

        # Cell size (DOUBLED - larger cells for better visibility)
        cell_size = min(available_w // self.grid_size, available_h // self.grid_size)
        cell_size = max(60, min(cell_size, 100))  # Between 60-100px (doubled from 30-50)

        # Grid dimensions
        grid_w = self.grid_size * cell_size
        grid_h = self.grid_size * cell_size

        # Grid offset (centered in left half)
        grid_x = (available_w - grid_w) // 2 + 20
        grid_y = (available_h - grid_h) // 2 + 20

        return cell_size, grid_x, grid_y, grid_w, grid_h, sidebar_w

    def draw_grid(self, cell_size, grid_x, grid_y):
        """Draw the grid with simple dark background (no maroon colors)"""
        # Draw grid cells with uniform dark background
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x = grid_x + col * cell_size
                y = grid_y + row * cell_size

                # All cells have same dark background
                pygame.draw.rect(self.screen, CELL_BG, (x, y, cell_size, cell_size))
                pygame.draw.rect(self.screen, GRID_COLOR, (x, y, cell_size, cell_size), 2)

    def draw_obstacles(self, cell_size, grid_x, grid_y):
        """Draw static obstacles as solid red squares (same style as cubes)"""
        for (obs_col, obs_row) in self.static_obstacles:
            x_center = grid_x + obs_col * cell_size + cell_size // 2
            y_center = grid_y + obs_row * cell_size + cell_size // 2

            # Draw obstacle as solid red square (same size as cubes)
            size = cell_size // 2
            pygame.draw.rect(self.screen, OBSTACLE_COLOR,
                           (x_center - size // 2, y_center - size // 2, size, size))

    def draw_cubes(self, cell_size, grid_x, grid_y):
        """Draw cubes - only unpicked cubes are shown, selected cube has blue border"""
        if not self.current_episode:
            return

        # Get list of picked cubes up to current step
        picked_cubes = set(self.current_episode.picked_cubes[:self.episode_step])

        for i in range(len(self.current_episode.cube_positions)):
            # Skip picked cubes - they vanish from grid
            if i in picked_cubes:
                continue

            pos = self.current_episode.cube_positions[i]

            # Convert world to grid coordinates
            grid_col, grid_row = self.env.astar_estimator._world_to_grid(pos[:2])

            if 0 <= grid_col < self.grid_size and 0 <= grid_row < self.grid_size:
                x = grid_x + grid_col * cell_size + cell_size // 2
                y = grid_y + grid_row * cell_size + cell_size // 2

                # Draw cube as square
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
        """Draw agent (end-effector position) OUTSIDE grid at bottom middle - SMALLER"""
        # Agent position: exactly at center of bottom grid line
        grid_center_x = grid_x + (self.grid_size * cell_size) // 2
        agent_y = grid_y + grid_h + int(cell_size * 0.8)

        # Draw agent as SMALLER blue circle
        radius = int(cell_size * 0.25)  # Smaller circle (was 0.5)
        pygame.draw.circle(self.screen, AGENT_COLOR, (grid_center_x, agent_y), radius)
        pygame.draw.circle(self.screen, AGENT_COLOR, (grid_center_x, agent_y), radius, 2)

        # Draw label
        label = self.font_small.render("EE", True, TEXT_COLOR)
        label_rect = label.get_rect(center=(grid_center_x, agent_y + radius + 15))
        self.screen.blit(label, label_rect)

        # Return agent position for path drawing
        return grid_center_x, agent_y, radius

    def draw_path(self, cell_size, grid_x, grid_y, grid_h, agent_pos):
        """Draw CURVED A* path from agent to selected cube using Catmull-Rom spline"""
        if self.current_path is None or len(self.current_path) < 1:
            return

        # Agent position (passed from draw_agent)
        agent_x, agent_y, agent_radius = agent_pos

        # Build path points starting from agent position
        points = [(agent_x, agent_y)]

        # Add grid cell centers for the path
        for grid_col, grid_row in self.current_path:
            x = grid_x + grid_col * cell_size + cell_size // 2
            y = grid_y + grid_row * cell_size + cell_size // 2
            points.append((x, y))

        # Adjust first point to be on agent circle boundary
        if len(points) >= 2:
            dx = points[1][0] - agent_x
            dy = points[1][1] - agent_y
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                points[0] = (
                    int(agent_x + (dx / length) * agent_radius),
                    int(agent_y + (dy / length) * agent_radius)
                )

        # Generate smooth curved path using Catmull-Rom spline
        if len(points) >= 2:
            curved_points = catmull_rom_spline(points, num_segments=15)

            # Draw curved path
            if len(curved_points) >= 2:
                pygame.draw.lines(self.screen, PATH_COLOR, False, curved_points, 3)

    def draw_mini_graph(self, x, y, width, height, data, title, color, max_val=None, show_grid=True):
        """Draw an enhanced mini line graph with grid and better styling"""
        if len(data) < 2:
            return y + height + 10

        # Draw background
        pygame.draw.rect(self.screen, (25, 25, 30), (x, y, width, height))
        pygame.draw.rect(self.screen, (80, 80, 90), (x, y, width, height), 2)

        # Draw title
        title_surf = self.font_medium.render(title, True, ACCENT_COLOR)
        self.screen.blit(title_surf, (x + 8, y + 6))

        # Get data values
        values = [d[1] for d in data]
        if max_val is None:
            max_val = max(values) if max(values) > 0 else 1.0
        min_val = min(values) if min(values) < 0 else 0.0
        val_range = max_val - min_val if max_val != min_val else 1.0

        # Graph area
        graph_x = x + 45
        graph_y = y + 35
        graph_w = width - 55
        graph_h = height - 45

        # Draw grid lines
        if show_grid:
            for i in range(5):
                grid_y_pos = graph_y + (i * graph_h // 4)
                pygame.draw.line(self.screen, (50, 50, 55),
                               (graph_x, grid_y_pos),
                               (graph_x + graph_w, grid_y_pos), 1)

        # Draw zero line if data crosses zero
        if min_val < 0 < max_val:
            zero_y = graph_y + graph_h - ((-min_val) / val_range) * graph_h
            pygame.draw.line(self.screen, (100, 100, 100),
                           (graph_x, int(zero_y)),
                           (graph_x + graph_w, int(zero_y)), 2)

        # Draw graph line
        points = []
        for i, (timestep, value) in enumerate(data):
            px = graph_x + (i / max(len(data) - 1, 1)) * graph_w
            py = graph_y + graph_h - ((value - min_val) / val_range) * graph_h
            points.append((int(px), int(py)))

        if len(points) >= 2:
            # Draw filled area under curve
            if min_val >= 0:
                filled_points = points + [(graph_x + graph_w, graph_y + graph_h), (graph_x, graph_y + graph_h)]
                pygame.draw.polygon(self.screen, (*color, 50), filled_points)

            # Draw line
            pygame.draw.lines(self.screen, color, False, points, 3)

            # Draw points
            for px, py in points[-10:]:  # Last 10 points
                pygame.draw.circle(self.screen, color, (px, py), 3)

        # Draw Y-axis labels
        for i in range(5):
            val = max_val - (i * val_range / 4)
            label_y = graph_y + (i * graph_h // 4)
            label = self.font_small.render(f"{val:.1f}", True, (180, 180, 180))
            self.screen.blit(label, (x + 5, label_y - 8))

        # Draw current value (REMOVED to avoid overlap - value visible in graph)
        # if len(data) > 0:
        #     current_val = data[-1][1]
        #     current_label = self.font_small.render(f"Current: {current_val:.2f}", True, color)
        #     self.screen.blit(current_label, (x + 8, y + height - 18))

        return y + height + 10  # Reduced spacing since no "Current" label

    def draw_sidebar(self, sidebar_w):
        """Draw sidebar with training info (top half) and live graphs (bottom half)"""
        win_w, win_h = self.screen.get_size()
        x0 = win_w - sidebar_w + 20
        y = 20

        # Calculate half-height for layout
        half_height = win_h // 2

        def draw_text(text, color=TEXT_COLOR, font=None, pad=5):
            nonlocal y
            if font is None:
                font = self.font_medium
            surf = font.render(text, True, color)
            self.screen.blit(surf, (x0, y))
            y += surf.get_height() + pad

        def draw_small(text, color=TEXT_COLOR, pad=3):
            draw_text(text, color, self.font_small, pad)

        # ===== TOP HALF: LABELS AND INFO =====
        # Header
        draw_text("A* RL Training", ACCENT_COLOR, self.font_large)
        draw_small(f"Grid: {self.grid_size}x{self.grid_size}, Cubes: {self.num_cubes}")
        y += 8

        # Episode completion status
        cubes_picked = len(self.env.objects_picked)
        cubes_total = self.env.total_objects
        episode_complete = cubes_picked == cubes_total

        if episode_complete:
            draw_text("EPISODE COMPLETE!", (100, 255, 100), self.font_medium)
            draw_small("All cubes picked! Resetting...")
            y += 5

        # Training state
        status = "TRAINING" if not self.paused else "PAUSED"
        status_color = (100, 255, 100) if not self.paused else (255, 200, 100)
        draw_text(f"Status: {status}", status_color)

        # Training progress
        progress = (self.timesteps_done / self.total_timesteps) * 100
        draw_small(f"Progress: {progress:.1f}%")
        draw_small(f"Training Steps: {self.timesteps_done}/{self.total_timesteps}")

        # Use total_episodes from callback (more accurate)
        episode_count = max(self.episode, self.total_episodes)
        draw_small(f"Episodes: {episode_count}")
        draw_small(f"Speed: {1000/self.frame_delay:.1f} steps/sec")
        y += 8

        # Episode info
        draw_text("Current Episode", ACCENT_COLOR)
        if hasattr(self.env, 'current_step'):
            # Episode steps (picks within this episode)
            draw_small(f"Episode Steps: {self.env.current_step}/{self.env.max_steps}")
            draw_small(f"  (1 step = 1 pick decision)", (150, 150, 150))

            # Warning if max_steps is too low
            if self.env.max_steps < self.env.total_objects * 2:
                draw_small(f"WARNING: max_steps too low!", (255, 100, 100))
                draw_small(f"Need ~{self.env.total_objects * 2} for {self.env.total_objects} cubes", (255, 150, 100))

            picked_color = (100, 255, 100) if cubes_picked == cubes_total else TEXT_COLOR
            draw_small(f"Cubes Picked: {cubes_picked}/{cubes_total}", picked_color)
        draw_small(f"Episode Reward: {self.episode_reward:.2f}")
        y += 8

        # Last action info
        if self.last_action is not None:
            draw_text("Last Action", ACCENT_COLOR)
            draw_small(f"Cube: {self.last_action}")
            draw_small(f"Reward: {self.last_reward:.2f}")
            if self.current_path:
                draw_small(f"Path: {len(self.current_path)} cells")
        y += 8

        # Reward Parameters (Live)
        draw_text("Reward Parameters", ACCENT_COLOR)
        draw_small(f"Pick: +{self.reward_params['base_pick']:.0f}", (100, 255, 100))
        draw_small(f"Distance: +{self.reward_params['distance_max']:.0f}", (100, 255, 100))
        draw_small(f"Time: {self.reward_params['time_penalty']:.0f}", (255, 150, 100))
        draw_small(f"Sequential: +{self.reward_params['sequential_bonus']:.0f}", (100, 255, 100))
        draw_small(f"Invalid: {self.reward_params['invalid_penalty']:.0f}", (255, 100, 100))
        y += 8

        # Observation features (for selected cube)
        if self.selected_cube_idx is not None and self.selected_cube_idx < self.env.total_objects:
            if self.selected_cube_idx not in self.env.objects_picked:  # Only show if not picked
                draw_text("Selected Cube", ACCENT_COLOR)
                obs = self.env._get_observation()
                start_idx = self.selected_cube_idx * 6

                draw_small(f"Dist EE: {obs[start_idx]:.2f}")
                draw_small(f"Dist Container: {obs[start_idx + 1]:.2f}")
                draw_small(f"Obstacle: {obs[start_idx + 2]:.2f}")

        # ===== BOTTOM HALF: INTERACTIVE GRAPHS =====
        # Start graphs at half-height with some padding
        y = max(y + 20, half_height + 20)  # More padding to avoid overlap

        # Success metrics
        if self.total_episodes > 0:
            success_rate = (self.success_count / self.total_episodes) * 100
            draw_text(f"Success: {success_rate:.1f}% ({self.success_count}/{self.total_episodes})",
                     (100, 255, 100) if success_rate > 50 else (255, 150, 100))
            y += 10  # Add spacing after success metrics

        # Live Reward Graph (ONLY graph in sidebar)
        if len(self.reward_history) > 0:
            graph_w = sidebar_w - 50
            graph_h = 150  # Larger since it's the only graph
            y = self.draw_mini_graph(x0, y, graph_w, graph_h,
                                     self.reward_history, "Episode Rewards",
                                     (100, 220, 100))
            y += 10

        # Interactive charts hint
        hint_text = "Seaborn charts auto-opened at start"
        hint_surf = self.font_small.render(hint_text, True, (150, 200, 255))
        self.screen.blit(hint_surf, (x0, y))
        y += hint_surf.get_height() + 10

        # Episode progress bar (at bottom)
        if hasattr(self.env, 'current_step') and self.env.max_steps > 0:
            bar_w = sidebar_w - 50
            bar_h = 25
            bar_x = x0
            bar_y = y

            # Background
            pygame.draw.rect(self.screen, (40, 40, 45), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, (80, 80, 90), (bar_x, bar_y, bar_w, bar_h), 2)

            # Progress fill
            progress_pct = self.env.current_step / self.env.max_steps
            fill_w = int(bar_w * progress_pct)
            pygame.draw.rect(self.screen, (100, 200, 100), (bar_x, bar_y, fill_w, bar_h))

            # Label
            label = self.font_small.render(f"Episode Progress: {int(progress_pct * 100)}%", True, TEXT_COLOR)
            label_rect = label.get_rect(center=(bar_x + bar_w // 2, bar_y + bar_h // 2))
            self.screen.blit(label, label_rect)

            y += bar_h + 10

    def step_training(self):
        """Execute one PPO training step"""
        if self.timesteps_done >= self.total_timesteps:
            print(f"\n[VISUALIZER] Training complete! {self.total_timesteps} timesteps done.")
            self.paused = True
            return

        # Learn for ONLY 1 step at a time for visualization (slow, watchable)
        # This allows you to see each cube being picked one by one
        steps_per_update = 1  # Process 1 step at a time for visibility

        # Train model
        self.model.learn(
            total_timesteps=steps_per_update,
            reset_num_timesteps=False,
            callback=VisualizationCallback(self)
        )

        self.timesteps_done += steps_per_update
        self.total_steps += steps_per_update

        # Update episode reward
        if self.last_reward != 0.0:
            self.episode_reward += self.last_reward
            self.total_reward += self.last_reward

            # Add to reward history for graph
            self.reward_history.append((self.timesteps_done, self.last_reward))
            if len(self.reward_history) > self.max_graph_points:
                self.reward_history.pop(0)

        # Update episode stats
        prev_episode = self.episode
        self.update_episode_stats()

        # If episode changed, add episode reward to graph and update charts
        if self.episode > prev_episode and len(self.episode_rewards) > 0:
            episode_total = self.episode_rewards[-1]
            self.reward_history.append((self.timesteps_done, episode_total))
            if len(self.reward_history) > self.max_graph_points:
                self.reward_history.pop(0)

            # Episode completed - trigger pause
            self.episode_complete_time = pygame.time.get_ticks()

            # Update interactive charts every 10 episodes
            if self.episode % 10 == 0:
                self.update_interactive_charts()

        # Update visualization state
        if self.last_action is not None and self.last_action < self.env.total_objects:
            # Check if this cube was just picked
            if self.last_action in self.env.objects_picked and self.just_picked_cube != self.last_action:
                self.just_picked_cube = self.last_action
                self.pick_animation_frames = self.max_animation_frames
                self.last_pick_time = pygame.time.get_ticks()  # Record pick time for pause

            self.selected_cube_idx = self.last_action
            self.current_path = self.get_astar_path(self.last_action)

        # Decrement animation counter
        if self.pick_animation_frames > 0:
            self.pick_animation_frames -= 1
            if self.pick_animation_frames == 0:
                self.just_picked_cube = None

    def get_astar_path(self, cube_idx):
        """Get A* path from entry point to cube"""
        if cube_idx is None or cube_idx >= self.env.total_objects:
            return None

        # Entry point: bottom middle of grid (where agent is positioned)
        entry_col = self.grid_size // 2
        entry_row = self.grid_size - 1  # Bottom row

        cube_pos = self.env.object_positions[cube_idx]
        goal_col, goal_row = self.env.astar_estimator._world_to_grid(cube_pos[:2])

        # Get A* path from entry point to cube (using PythonRobotics planning method)
        rx, ry = self.env.astar_estimator.planning(entry_col, entry_row, goal_col, goal_row)

        # Convert to list of tuples
        if rx is None or ry is None:
            return None

        path = [(rx[i], ry[i]) for i in range(len(rx))]
        return path

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False

                elif event.key == pygame.K_SPACE:
                    if self.timesteps_done >= self.total_timesteps:
                        print("[VISUALIZER] Training already complete!")
                    else:
                        self.paused = not self.paused
                        print(f"Training {'PAUSED' if self.paused else 'RESUMED'}")

                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    # Decrease delay = faster
                    self.frame_delay = max(10, self.frame_delay - 50)
                    print(f"Speed increased: {1000/self.frame_delay:.1f} steps/sec (delay: {self.frame_delay}ms)")

                elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                    # Increase delay = slower
                    self.frame_delay = min(5000, self.frame_delay + 200)
                    print(f"Speed decreased: {1000/self.frame_delay:.1f} steps/sec (delay: {self.frame_delay}ms)")

            elif event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

    def show_interactive_charts(self):
        """Show interactive Seaborn charts in separate window (auto-updating)"""
        try:
            # Set Seaborn style to match reference images
            sns.set_theme(style="whitegrid")
            plt.style.use('seaborn-v0_8-whitegrid')

            # Create figure with 2 subplots (matching reference layout)
            self.fig, self.axes = plt.subplots(2, 1, figsize=(12, 8))
            self.fig.suptitle('A* RL Training Progress', fontsize=14, fontweight='bold')

            # Enable interactive mode
            plt.ion()
            plt.show(block=False)

            print("[VISUALIZER] Interactive Seaborn charts opened! They will update automatically during training.")

        except Exception as e:
            print(f"[ERROR] Failed to show interactive charts: {e}")
            import traceback
            traceback.print_exc()

    def update_interactive_charts(self):
        """Update the interactive charts with latest data"""
        if not hasattr(self, 'fig') or not hasattr(self, 'axes'):
            return

        try:
            # Clear previous plots
            for ax in self.axes:
                ax.clear()

            # Chart 1: Episode Rewards Over Time (matching reference style)
            if len(self.episode_rewards) > 10:
                # Plot multiple lines with different styles (like reference)
                timesteps = list(range(len(self.episode_rewards)))
                rewards = self.episode_rewards

                # Main line (solid)
                self.axes[0].plot(timesteps, rewards, color='#E74C3C', linestyle='-', linewidth=1.5, alpha=0.8, label='Episode Reward')

                # Moving average (dotted)
                if len(rewards) > 20:
                    window = 20
                    moving_avg = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
                    self.axes[0].plot(timesteps, moving_avg, color='#F39C12', linestyle=':', linewidth=2, alpha=0.9, label=f'{window}-Episode MA')

                # Smoothed line (dashed)
                if len(rewards) > 50:
                    window = 50
                    smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
                    self.axes[0].plot(timesteps, smoothed, color='#27AE60', linestyle='--', linewidth=2, alpha=0.9, label=f'{window}-Episode MA')

                self.axes[0].set_title('Training Progress: Episode Rewards Over Time', fontsize=11, fontweight='bold')
                self.axes[0].set_xlabel('Episode', fontsize=10)
                self.axes[0].set_ylabel('Reward Mean', fontsize=10)
                self.axes[0].legend(loc='upper left', fontsize=9)
                self.axes[0].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

            # Chart 2: Value Loss (matching reference style)
            if len(self.reward_history) > 10:
                # Extract timesteps and rewards
                timesteps = [d[0] for d in self.reward_history]
                rewards = [d[1] for d in self.reward_history]

                # Plot with dotted/dashed lines
                self.axes[1].plot(timesteps, rewards, color='#3498DB', linestyle=':', linewidth=1.5, alpha=0.7, label='Step Reward')

                # Moving average
                if len(rewards) > 100:
                    df = pd.DataFrame({'timestep': timesteps, 'reward': rewards})
                    df['ma'] = df['reward'].rolling(window=100, min_periods=1).mean()
                    self.axes[1].plot(df['timestep'], df['ma'], color='#E67E22', linestyle='--', linewidth=2, alpha=0.9, label='100-Step MA')

                self.axes[1].set_title('Value Loss', fontsize=11, fontweight='bold')
                self.axes[1].set_xlabel('Timesteps', fontsize=10)
                self.axes[1].set_ylabel('Value Loss', fontsize=10)
                self.axes[1].legend(loc='upper right', fontsize=9)
                self.axes[1].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

            # Refresh canvas
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        except Exception as e:
            # Silently ignore chart update errors (window might be closed)
            pass

    def render(self):
        """Render the visualization"""
        # Clear screen
        self.screen.fill(BG_COLOR)

        # Compute layout
        cell_size, grid_x, grid_y, grid_w, grid_h, sidebar_w = self.compute_layout()

        # Draw components in correct order
        self.draw_grid(cell_size, grid_x, grid_y)
        self.draw_obstacles(cell_size, grid_x, grid_y)
        self.draw_cubes(cell_size, grid_x, grid_y)

        # Draw agent and get its position for path drawing
        agent_pos = self.draw_agent(cell_size, grid_x, grid_y, grid_h)

        # Draw path from agent to target
        self.draw_path(cell_size, grid_x, grid_y, grid_h, agent_pos)

        # Draw sidebar
        self.draw_sidebar(sidebar_w)

        # Update display
        pygame.display.flip()

    def run(self):
        """Main loop"""
        print("\n" + "=" * 70)
        print("A* PPO TRAINING VISUALIZER (LIVE)")
        print("=" * 70)
        print(f"Grid Size: {self.grid_size}x{self.grid_size}")
        print(f"Number of Cubes: {self.num_cubes}")
        print(f"Max Steps per Episode: {self.env.max_steps}")
        print(f"Total Timesteps: {self.total_timesteps}")
        print(f"Initial Speed: {1000/self.frame_delay:.1f} steps/sec (SLOW for watching)")
        print("\nThis visualizes ACTUAL PPO training steps!")
        print("The agent learns using PPO algorithm in real-time.")
        print("\nYou will see:")
        print("  - Agent picks cubes ONE BY ONE (slow, watchable)")
        print("  - Yellow path with blue dots shows route to each cube")
        print("  - Picked cubes flash white and vanish")
        print("  - Episode completes when all cubes are picked")
        print("  - Grid resets for next episode with new cube positions")
        print("\nSpeed Control:")
        print("  - Visualization is SLOWED DOWN (2 picks/sec) so you can watch")
        print("  - Use +/- keys to speed up or slow down further")
        print("  - Training still happens at full speed internally")
        print("\nControls:")
        print("  SPACE: Start/Pause training")
        print("  +: Speed up (faster picks)")
        print("  -: Slow down (slower picks)")
        print("  ESC/Q: Quit")
        print("=" * 70)
        print("\nStarting visualization (PAUSED)...")
        print("Opening interactive Seaborn charts...")

        # Auto-open Seaborn charts at start
        Thread(target=self.show_interactive_charts, daemon=True).start()

        print("Press SPACE to start PPO training!\n")

        last_step_time = pygame.time.get_ticks()

        while self.running:
            # Handle events
            self.handle_events()

            # Step training if not paused (with frame delay for visibility)
            if not self.paused:
                current_time = pygame.time.get_ticks()

                # Check if we're in pick pause (extra delay after picking a cube)
                time_since_pick = current_time - self.last_pick_time
                in_pick_pause = (self.last_pick_time > 0 and time_since_pick < self.pick_pause_delay)

                # Check if we're in episode complete pause
                time_since_episode_complete = current_time - self.episode_complete_time
                in_episode_pause = (self.episode_complete_time > 0 and time_since_episode_complete < self.episode_complete_pause)

                # Only step if enough time has passed AND not in any pause
                if not in_pick_pause and not in_episode_pause and current_time - last_step_time >= self.frame_delay:
                    self.step_training()
                    last_step_time = current_time

            # Render
            self.render()

            # Control frame rate (60 FPS for smooth animation)
            self.clock.tick(60)

        # Cleanup
        pygame.quit()
        print("\n" + "=" * 70)
        print("TRAINING VISUALIZATION CLOSED")
        print("=" * 70)
        print(f"Total Episodes: {self.episode}")
        print(f"Timesteps Completed: {self.timesteps_done}/{self.total_timesteps}")
        print(f"Total Reward: {self.total_reward:.2f}")
        print(f"Average Reward (last 10): {self.avg_reward:.2f}")
        print("=" * 70)


def main():
    """Main function"""
    args = parse_args()

    # Validate inputs
    if args.num_cubes > args.grid_size * args.grid_size:
        print(f"ERROR: Number of cubes ({args.num_cubes}) exceeds grid capacity ({args.grid_size}x{args.grid_size}={args.grid_size * args.grid_size})")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("RL + A* EPISODE VISUALIZER")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Grid Size: {args.grid_size}x{args.grid_size}")
    print(f"  Number of Cubes: {args.num_cubes}")
    print(f"  Model Path: {args.model_path if args.model_path else 'None (greedy baseline)'}")
    print("=" * 70)
    print("\nInitializing visualizer...")

    # Create and run visualizer
    visualizer = AStarRLVisualizer(
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

