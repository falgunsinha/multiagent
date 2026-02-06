import os
import sys
import argparse
import warnings
import time
from typing import Optional, List, Tuple
from collections import deque

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

import numpy as np
import pygame
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.catmull_rom_spline_path import catmull_rom_spline
from src.rl.object_selection_env_astar import ObjectSelectionEnvAStar
from src.rl.path_estimators import Node
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from src.rl.doubleDQN import DoubleDQNAgent

BG_COLOR = (20, 20, 30)
GRID_COLOR = (60, 60, 80)
CUBE_COLOR = (100, 200, 100)
OBSTACLE_COLOR = (200, 50, 50)
AGENT_COLOR = (100, 150, 255)
PATH_COLOR = (255, 200, 0)
SELECTED_BORDER_COLOR = (0, 150, 255)
TEXT_COLOR = (220, 220, 220)
HEADER_COLOR = (100, 200, 255)
LABEL_COLOR = (180, 180, 200)
VALUE_COLOR = (255, 255, 255)
BOX_COLOR = (40, 40, 50)
BOX_BORDER_COLOR = (80, 80, 100)


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
    """Generate smooth curve through points using Cubic Spline interpolation"""

    if len(points) < 2:
        return points

    if len(points) == 2:
        x1, y1 = points[0]
        x2, y2 = points[1]
        t_vals = np.linspace(0, 1, num_points)
        interpolated = [(int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))) for t in t_vals]
        return interpolated

    import math
    import bisect

    x_list = [p[0] for p in points]
    y_list = [p[1] for p in points]

    dx = np.diff(x_list)
    dy = np.diff(y_list)
    ds = [math.hypot(idx, idy) for idx, idy in zip(dx, dy)]
    s = [0.0]
    s.extend(np.cumsum(ds).tolist())

    total_length = s[-1]

    if total_length == 0:
        return points

    t = np.linspace(0, total_length, num_points)

    def spline_1d(s_list, val_list, t_vals):
        """1D cubic spline interpolation"""
        num = len(s_list)
        a = val_list.copy()
        b, c, d = [], [], []
        h = np.diff(s_list)

        A = np.zeros((num, num))
        B = np.zeros(num)

        A[0, 0] = 1.0
        A[num - 1, num - 1] = 1.0

        for i in range(1, num - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2.0 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            B[i] = 3.0 * ((a[i + 1] - a[i]) / h[i] - (a[i] - a[i - 1]) / h[i - 1])

        c = np.linalg.solve(A, B).tolist()

        for i in range(num - 1):
            d.append((c[i + 1] - c[i]) / (3.0 * h[i]))
            b.append((a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0)

        result = []

        for it in t_vals:
            if it <= s_list[0]:
                result.append(val_list[0])
            elif it >= s_list[-1]:
                result.append(val_list[-1])
            else:
                i = bisect.bisect(s_list, it) - 1
                i = min(i, num - 2)
                ds = it - s_list[i]
                value = a[i] + b[i] * ds + c[i] * ds**2 + d[i] * ds**3
                result.append(value)

        return result

    path_x = spline_1d(s, x_list, t)
    path_y = spline_1d(s, y_list, t)

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
        self.grid_size = grid_size
        self.num_cubes = num_cubes
        self.window_width = window_width
        self.window_height = window_height
        self.fps = initial_fps

        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
        pygame.display.set_caption("Decision making + A* Motion Planning")
        self.clock = pygame.time.Clock()

        try:
            self.font_header = pygame.font.SysFont("trebuchetms", 20, bold=True)
            self.font_large = pygame.font.SysFont("trebuchetms", 16, bold=True)
            self.font_medium = pygame.font.SysFont("trebuchetms", 13)
            self.font_small = pygame.font.SysFont("trebuchetms", 12)
        except:
            self.font_header = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 20)
            self.font_medium = pygame.font.Font(None, 16)
            self.font_small = pygame.font.Font(None, 14)

        self.model = None
        self.model_type = None
        model_max_objects = None
        model_grid_size = grid_size

        if model_path and os.path.exists(model_path):
            print(f"[MODEL] Loading model from: {model_path}")
            if model_path.endswith('.pt'):
                self.model_type = 'ddqn'
                print(f"[MODEL] Detected DDQN model (.pt)")
            else:
                self.model_type = 'ppo'
                print(f"[MODEL] Detected PPO model (.zip)")
            if "_step_" in model_path:
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

                        if model_grid_size != grid_size:
                            print(f"[WARNING] Model trained on {model_grid_size}x{model_grid_size} grid, but running with {grid_size}x{grid_size}")
                            print(f"[INFO] Using model's grid size: {model_grid_size}x{model_grid_size}")
                            grid_size = model_grid_size
                            self.grid_size = grid_size
                except Exception as e:
                    print(f"[WARNING] Could not read metadata file: {e}")
            else:
                print(f"[WARNING] Metadata file not found: {metadata_path}")

            if model_max_objects is None:
                print(f"[INFO] Extracting max_objects from model zip file...")
                import zipfile
                import json
                try:
                    with zipfile.ZipFile(model_path, 'r') as archive:
                        data_json = archive.read('data').decode('utf-8')
                        data = json.loads(data_json)

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

                        model_max_objects = obs_space_shape // 6
                        print(f"[MODEL] Extracted from zip: {obs_space_shape} dims = {model_max_objects} objects × 6 features")
                except Exception as e:
                    print(f"[ERROR] Could not extract max_objects from model zip: {e}")
                    print(f"[ERROR] Cannot load model without knowing max_objects!")
                    print(f"[INFO] Running in greedy baseline mode (no model)")
                    model_path = None

        max_objects = model_max_objects if model_max_objects else num_cubes
        print(f"[ENV] Creating environment with max_objects={max_objects}, num_cubes={num_cubes}, grid={grid_size}x{grid_size}")

        self.env = ObjectSelectionEnvAStar(
            franka_controller=None,
            max_objects=max_objects,
            max_steps=num_cubes * 2,
            num_cubes=num_cubes,
            render_mode=None,
            dynamic_obstacles=False,
            training_grid_size=grid_size
        )

        def mask_fn(env):
            return env.action_masks()

        wrapped_env = ActionMasker(self.env, mask_fn)
        vec_env = DummyVecEnv([lambda: wrapped_env])
        self.vec_env = vec_env
        self.wrapped_env = wrapped_env

        if model_path and os.path.exists(model_path):
            try:

                if self.model_type == 'ddqn':
                    checkpoint = torch.load(model_path, map_location='cpu')
                    state_dim = checkpoint['state_dim']
                    action_dim = checkpoint['action_dim']

                    self.model = DoubleDQNAgent(
                        state_dim=state_dim,
                        action_dim=action_dim,
                        gamma=checkpoint['gamma'],
                        epsilon_start=0.0,
                        epsilon_end=0.0,
                        epsilon_decay=1.0,
                        batch_size=checkpoint['batch_size'],
                        target_update_freq=checkpoint['target_update_freq']
                    )
                    self.model.load(model_path)
                    self.model.epsilon = 0.0
                    print(f"[MODEL] DDQN model loaded successfully!")
                else:
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

        self.cached_reward_graph = None
        self.cached_reward_history_len = 0

        self.cached_arrow_data = None
        self.cached_path_progress = -1.0

        self.accumulated_paths = []

        self.cached_astar_graph = None
        self.cached_astar_progress = -1.0
        self.cached_astar_phase = -1
        self.cached_astar_paths_len = 0

        # Visualization state
        self.selected_cube_idx = None
        self.current_path = None
        self.explored_nodes = []
        self.static_obstacles = []
        self.running = True
        self.paused = True
        self.playback_speed = 1.0
        self.auto_advance = True

        self.animation_phase = 0
        self.phase_timer = 0
        self.selection_delay = 30
        self.path_progress = 0.0
        self.path_animation_duration = 60
        self.post_animation_delay = 30

        self.current_reward_components = {
            'r_total': 0.0,
            'r_path': 0.0,
            'r_container': 0.0,
            'r_obstacle': 0.0,
            'r_reachability': 0.0,
            'r_clearance': 0.0
        }

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

        self.reward_history = []

        self.episode_start_time = 0
        self.episode_elapsed_time = 0
        self.timer_started = False

        self.generate_new_episode()

    def generate_new_episode(self):
        """Generate a new episode by running the RL agent"""
        obs, _ = self.wrapped_env.reset()

        self.episode_start_time = 0
        self.episode_elapsed_time = 0
        self.timer_started = False

        self.accumulated_paths = []

        grid_capacity = self.grid_size * self.grid_size
        available_cells = grid_capacity - self.num_cubes - 1
        max_obstacles = max(0, min(3, available_cells))
        min_obstacles = 1 if max_obstacles > 0 else 0
        num_obstacles = np.random.randint(min_obstacles, max_obstacles + 1) if max_obstacles > 0 else 0
        self._add_random_obstacles(num_obstacles)

        episode = EpisodeData()
        episode.cube_positions = self.env.object_positions[:self.env.total_objects].copy()
        episode.obstacle_positions = [self.env.astar_estimator._grid_to_world(ox, oy)
                                     for ox, oy in self.static_obstacles]
        episode.ee_start_position = self.env.ee_position.copy()

        done = False
        step = 0
        successful_picks = 0
        failed_picks = 0

        print(f"\nGenerating episode {self.episode_count + 1}...")
        print(f"  {self.num_cubes} cubes, {num_obstacles} obstacles")

        while not done and step < self.env.max_steps:

            if self.model is not None:
                action_mask = self.wrapped_env.action_masks()

                if self.model_type == 'ddqn':
                    obs_flat = obs.flatten()
                    obs_tensor = torch.FloatTensor(obs_flat).to(self.model.device)
                    action = self.model.policy_net.get_action(obs_tensor, epsilon=0.0, action_mask=action_mask)
                else:
                    action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
                    action = int(action)
            else:
                action = self._greedy_action()

            self._update_astar_grid_for_planning(action)

            path = self._get_astar_path(action)

            if path and len(path) > 0:
                successful_picks += 1
                print(f"  Step {step+1}: Cube {action}, Path points: {len(path)} ✓")
            else:
                failed_picks += 1
                print(f"  Step {step+1}: Cube {action}, Path points: 0 ✗ A* FAILED")

            path_length = self._calculate_path_length(path) if path else 999.0
            obstacle_proximity = self._calculate_obstacle_proximity(action)
            dist_to_ee = np.linalg.norm(self.env.object_positions[action][:2] - self.env.ee_position[:2])
            reachability = 1.0 if (0.3 <= dist_to_ee <= 0.9) else 0.0
            path_clearance = self._calculate_path_clearance(path) if path else 0.0
            dist_to_container = 0.5

            obs, reward, terminated, truncated, _ = self.wrapped_env.step(action)
            done = terminated or truncated

            episode.add_step(action, path, reward, path_length, obstacle_proximity,
                           reachability, path_clearance, dist_to_ee, dist_to_container)
            episode.picked_cubes.append(action)

            step += 1

        self.current_episode = episode
        self.episode_step = 0
        self.episode_count += 1
        self.episode_history.append(episode)

        print(f"Done! ({successful_picks} successful, {failed_picks} failed)\n")

    def _add_random_obstacles(self, num_obstacles: int):
        """Add random obstacles to empty grid cells"""
        self.static_obstacles = []

        if num_obstacles == 0:
            return

        ee_grid_x, ee_grid_y = self.env.astar_estimator._world_to_grid(self.env.ee_position[:2])

        cube_cells = set()
        cube_cells.add((ee_grid_x, ee_grid_y))

        for i in range(self.env.total_objects):
            pos = self.env.object_positions[i]
            grid_col, grid_row = self.env.astar_estimator._world_to_grid(pos[:2])
            cube_cells.add((grid_col, grid_row))

        empty_cells = []

        for grid_x in range(self.grid_size):

            for grid_y in range(self.grid_size):

                if (grid_x, grid_y) not in cube_cells:
                    empty_cells.append((grid_x, grid_y))

        if len(empty_cells) >= num_obstacles:
            np.random.shuffle(empty_cells)
            selected_cells = empty_cells[:num_obstacles]

            for grid_x, grid_y in selected_cells:
                self.static_obstacles.append((grid_x, grid_y))

    def _update_astar_grid_for_planning(self, target_cube_idx: int):
        """Update A* occupancy grid with current obstacles"""
        obstacle_positions = []

        for grid_x, grid_y in self.static_obstacles:
            world_pos = self.env.astar_estimator._grid_to_world(grid_x, grid_y)
            obstacle_positions.append(np.array([world_pos[0], world_pos[1], 0.0]))

        unpicked_cube_positions = []

        for i in range(self.env.total_objects):

            if i not in self.env.objects_picked and i != target_cube_idx:
                unpicked_cube_positions.append(self.env.object_positions[i])

        self.env.astar_estimator.update_occupancy_grid(
            object_positions=unpicked_cube_positions,
            obstacle_positions=obstacle_positions
        )

    def _greedy_action(self) -> int:
        """Greedy baseline: pick closest unpicked cube using A* path length"""
        min_path_length = float('inf')
        best_action = 0

        for i in range(self.env.total_objects):

            if i not in self.env.objects_picked:
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

        if ee_grid == goal_grid:
            return []

        rx, ry = self.env.astar_estimator.planning(ee_grid[0], ee_grid[1], goal_grid[0], goal_grid[1])

        if rx is None or ry is None:
            return []
        else:
            path = [(rx[i], ry[i]) for i in range(len(rx))]
            path.reverse()
            return path

    def _get_astar_path_with_explored(self, cube_idx: int) -> Tuple[Optional[List[Tuple[int, int]]], List[Tuple[int, int]]]:
        """Get A* path and explored nodes from end-effector to cube"""

        if cube_idx >= self.env.total_objects:
            return None, []

        ee_grid = self.env.astar_estimator._world_to_grid(self.env.ee_position[:2])
        cube_pos = self.env.object_positions[cube_idx]
        goal_grid = self.env.astar_estimator._world_to_grid(cube_pos[:2])

        if ee_grid == goal_grid:
            return [], []

        start_node = Node(ee_grid[0], ee_grid[1], 0.0, -1)
        goal_node = Node(goal_grid[0], goal_grid[1], 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.env.astar_estimator.calc_grid_index(start_node)] = start_node

        while True:

            if len(open_set) == 0:
                return [], []

            c_id = min(open_set, key=lambda o: open_set[o].cost + self.env.astar_estimator.calc_heuristic(goal_node, open_set[o]))
            current = open_set[c_id]

            if current.x == goal_node.x and current.y == goal_node.y:
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            del open_set[c_id]
            closed_set[c_id] = current

            for i, _ in enumerate(self.env.astar_estimator.motion):
                node = Node(
                    current.x + self.env.astar_estimator.motion[i][0],
                    current.y + self.env.astar_estimator.motion[i][1],
                    current.cost + self.env.astar_estimator.motion[i][2],
                    c_id
                )
                n_id = self.env.astar_estimator.calc_grid_index(node)

                if not self.env.astar_estimator.verify_node(node, goal_node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node
                else:

                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node

        rx, ry = self.env.astar_estimator.calc_final_path(goal_node, closed_set)

        explored = [(closed_set[key].x, closed_set[key].y) for key in closed_set]

        if rx is None or ry is None:
            return [], explored
        else:
            path = [(rx[i], ry[i]) for i in range(len(rx))]
            path.reverse()
            return path, explored

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

    def calculate_reward_components(self, path_length, dist_to_container, obstacle_proximity,
                                    reachability, path_clearance, dist_to_ee):
        """Calculate individual reward components from observation values"""
        components = {}

        euclidean_distance = dist_to_ee
        planning_failed = (path_length >= 2.0 * euclidean_distance)

        normalized_path_length = (path_length - 0.3) / 0.6
        normalized_path_length = np.clip(normalized_path_length, 0.0, 1.0)
        path_reward = 5.0 * (1.0 - normalized_path_length)

        if planning_failed:
            path_reward -= 5.0

        components['r_path'] = path_reward

        container_reward = 3.0 * np.exp(-dist_to_container)
        components['r_container'] = container_reward

        obstacle_reward = 3.0 * (1.0 - obstacle_proximity)
        components['r_obstacle'] = obstacle_reward

        reachability_penalty = 0.0 if reachability >= 0.5 else -10.0
        components['r_reachability'] = reachability_penalty

        clearance_reward = 2.0 * path_clearance
        components['r_clearance'] = clearance_reward

        return components

    def create_reward_graph_surface(self, width, height):
        """Create reward gradient area graph using matplotlib"""

        if len(self.reward_history) < 2:
            surf = pygame.Surface((width, height))
            surf.fill(BOX_COLOR)
            return surf

        if (self.cached_reward_graph is not None and
            self.cached_reward_history_len == len(self.reward_history)):
            return self.cached_reward_graph

        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100, facecolor='#282828')
        ax.set_facecolor('#282828')

        steps = np.array(range(1, len(self.reward_history) + 1))
        rewards = np.array(self.reward_history)

        min_reward = min(rewards)

        rewards_shifted = rewards - min_reward + 0.1

        cmap = plt.get_cmap('Blues')
        n_levels = 60

        for i in range(n_levels):
            alpha_level = (i + 1) / n_levels
            y_level = rewards_shifted * alpha_level
            ax.fill_between(steps, 0, y_level, color=cmap(alpha_level), alpha=0.05)

        ax.set_xlabel('Step', color='white', fontsize=7)
        ax.set_ylabel('Reward', color='white', fontsize=7)
        ax.set_title('Reward Vs Step', color='#64C8FF', fontsize=9, fontweight='bold')
        ax.tick_params(colors='white', labelsize=6)
        ax.grid(False)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        y_ticks = ax.get_yticks()
        ax.set_yticklabels([f'{int(y + min_reward - 0.1)}' for y in y_ticks])

        ax.set_xlim(0.5, len(self.reward_history) + 0.5)

        plt.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.18)

        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        buf = canvas.buffer_rgba()
        size = canvas.get_width_height()

        surf = pygame.image.frombuffer(buf, size, "RGBA")
        plt.close(fig)

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

                    if self.episode_step < len(self.current_episode.actions):
                        self.episode_step += 1
                        self.animating = True
                        self.animation_progress = 0.0

                elif event.key == pygame.K_r:
                    self.episode_step = 0
                    self.animating = False

                elif event.key == pygame.K_g:
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

        picked_cubes = set(self.current_episode.picked_cubes[:self.episode_step])

        ee_col, ee_row = self.env.astar_estimator._world_to_grid(self.env.ee_position[:2])

        for i in range(len(self.current_episode.cube_positions)):

            if i in picked_cubes:
                continue

            pos = self.current_episode.cube_positions[i]
            grid_col, grid_row = self.env.astar_estimator._world_to_grid(pos[:2])

            if grid_col == ee_col and grid_row == ee_row:
                continue

            if 0 <= grid_col < self.grid_size and 0 <= grid_row < self.grid_size:
                x = grid_x + grid_col * cell_size + cell_size // 2
                y = grid_y + grid_row * cell_size + cell_size // 2

                size = cell_size // 2

                if i == self.selected_cube_idx and self.animation_phase == 3:
                    continue

                pygame.draw.rect(self.screen, CUBE_COLOR,
                               (x - size // 2, y - size // 2, size, size))

                if i == self.selected_cube_idx and self.animation_phase in [1, 2]:
                    border_width = 4
                    pygame.draw.rect(self.screen, SELECTED_BORDER_COLOR,
                                   (x - size // 2 - border_width, y - size // 2 - border_width,
                                    size + border_width * 2, size + border_width * 2), border_width)
                    pygame.draw.circle(self.screen, SELECTED_BORDER_COLOR, (x, y), 5)

    def draw_agent(self, cell_size, grid_x, grid_y, _):
        """Draw agent (end-effector) at its actual grid cell position"""
        ee_col, ee_row = self.env.astar_estimator._world_to_grid(self.env.ee_position[:2])

        agent_x = grid_x + ee_col * cell_size + cell_size // 2
        agent_y = grid_y + ee_row * cell_size + cell_size // 2

        radius = int(cell_size * 0.3)
        pygame.draw.circle(self.screen, AGENT_COLOR, (agent_x, agent_y), radius)
        pygame.draw.circle(self.screen, AGENT_COLOR, (agent_x, agent_y), radius, 2)

        label = self.font_small.render("EE", True, TEXT_COLOR)
        label_rect = label.get_rect(center=(agent_x, agent_y))
        self.screen.blit(label, label_rect)

        return agent_x, agent_y, radius

    def draw_path(self, cell_size, grid_x, grid_y, _, agent_pos):
        """Draw CURVED A* path using Catmull-Rom Spline interpolation"""

        if self.animation_phase != 2:
            return

        if self.current_path is None or len(self.current_path) == 0:
            return

        if len(self.current_path) == 1:
            return

        agent_x, agent_y, _ = agent_pos

        points = []
        points.append((agent_x, agent_y))

        for i, (grid_col, grid_row) in enumerate(self.current_path):

            if i == 0:
                continue

            x = grid_x + grid_col * cell_size + cell_size // 2
            y = grid_y + grid_row * cell_size + cell_size // 2
            points.append((x, y))

        if len(points) < 2:
            return

        if len(points) >= 2:
            total_dist = sum(np.sqrt((points[i+1][0] - points[i][0])**2 +
                                    (points[i+1][1] - points[i][1])**2)
                           for i in range(len(points)-1))
            num_interp_points = max(50, int(total_dist / 2))

            if len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                t_vals = np.linspace(0, 1, num_interp_points)
                curved_points = [(int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))) for t in t_vals]
            else:
                spline_x, spline_y = catmull_rom_spline(points, num_interp_points)
                curved_points = [(int(x), int(y)) for x, y in zip(spline_x, spline_y)]

            if len(curved_points) >= 2:
                num_points_to_draw = max(2, int(len(curved_points) * self.path_progress))
                animated_points = curved_points[:num_points_to_draw]

                if len(animated_points) >= 2:
                    start_x, start_y = animated_points[0]
                    pygame.draw.circle(self.screen, (255, 0, 0), (int(start_x), int(start_y)), 5)

                    pygame.draw.lines(self.screen, PATH_COLOR, False, animated_points, 3)

                    if (self.cached_arrow_data is None or
                        abs(self.path_progress - self.cached_path_progress) > 0.01):

                        arrow_tip = None
                        arrow_left = None
                        arrow_right = None

                        if len(animated_points) >= 2:
                            end_x, end_y = animated_points[-1]

                            prev_x, prev_y = animated_points[-2]

                            for i in range(len(animated_points) - 2, -1, -1):
                                px, py = animated_points[i]
                                dist = np.sqrt((end_x - px)**2 + (end_y - py)**2)

                                if dist >= 10:
                                    prev_x, prev_y = px, py
                                    break

                            dx = end_x - prev_x
                            dy = end_y - prev_y
                            length = np.sqrt(dx**2 + dy**2)

                            if length > 2.0:
                                dx /= length
                                dy /= length

                                arrow_length = 6
                                arrow_width = 4

                                arrow_tip = (int(end_x), int(end_y))
                                base_x = end_x - dx * arrow_length
                                base_y = end_y - dy * arrow_length
                                perp_x = -dy
                                perp_y = dx
                                arrow_left = (int(base_x + perp_x * arrow_width), int(base_y + perp_y * arrow_width))
                                arrow_right = (int(base_x - perp_x * arrow_width), int(base_y - perp_y * arrow_width))

                        self.cached_arrow_data = (arrow_tip, arrow_left, arrow_right)
                        self.cached_path_progress = self.path_progress

                    if self.cached_arrow_data:
                        arrow_tip, arrow_left, arrow_right = self.cached_arrow_data

                        if arrow_tip and arrow_left and arrow_right:
                            pygame.draw.line(self.screen, PATH_COLOR, arrow_tip, arrow_left, 2)
                            pygame.draw.line(self.screen, PATH_COLOR, arrow_tip, arrow_right, 2)

    def draw_box(self, x, y, width, height, title=None):
        """Draw a styled box with optional title"""
        pygame.draw.rect(self.screen, BOX_COLOR, (x, y, width, height))
        pygame.draw.rect(self.screen, BOX_BORDER_COLOR, (x, y, width, height), 2)

        if title:
            title_surf = self.font_large.render(title, True, HEADER_COLOR)
            self.screen.blit(title_surf, (x + 10, y + 8))
            return y + 35

        return y + 10

    def draw_info_panel(self, x, y, width, _):
        """Draw comprehensive info panel on right side"""

        if not self.current_episode:
            return

        current_y = y

        header_surf = self.font_header.render("Decision making + A* motion planning", True, HEADER_COLOR)
        self.screen.blit(header_surf, (x, current_y))
        current_y += 40

        box_y = self.draw_box(x, current_y, width, 195, "Episode Info")
        current_reward = sum(self.current_episode.rewards[:self.episode_step]) if self.episode_step > 0 else 0.0

        dist_to_target = 0.0

        if self.selected_cube_idx is not None and self.selected_cube_idx < len(self.env.object_positions):
            target_pos = self.env.object_positions[self.selected_cube_idx][:2]
            dist_to_target = np.linalg.norm(target_pos - self.env.ee_position[:2])

        status_text = "Paused" if self.paused else "Playing"

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

        box_y = self.draw_box(x, current_y, width, 70, "Cumulative Rewards")
        total_label = self.font_large.render("Total:", True, LABEL_COLOR)
        total_value = self.font_large.render(f"{current_reward:.2f}", True, HEADER_COLOR)
        self.screen.blit(total_label, (x + 15, box_y))
        self.screen.blit(total_value, (x + 200, box_y))

        current_y += 80

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
        progress_bar_height = 28
        progress_bar_y = y - progress_bar_height - 30

        progress_bar_width = 420

        if self.current_episode:
            total_steps = len(self.current_episode.actions)
            current_step = self.episode_step
            progress = current_step / total_steps if total_steps > 0 else 0.0

            pygame.draw.rect(self.screen, (255, 255, 255), (x, progress_bar_y, progress_bar_width, progress_bar_height))
            pygame.draw.rect(self.screen, BOX_BORDER_COLOR, (x, progress_bar_y, progress_bar_width, progress_bar_height), 2)

            fill_width = int((progress_bar_width - 4) * progress)

            if fill_width > 0:
                pygame.draw.rect(self.screen, (0, 200, 0), (x + 2, progress_bar_y + 2, fill_width, progress_bar_height - 4))

            progress_text = f"{progress*100:.1f}%"
            text_surf = self.font_small.render(progress_text, True, (0, 0, 0))
            text_x = x + max(fill_width // 2, 30)
            text_rect = text_surf.get_rect(center=(text_x, progress_bar_y + progress_bar_height//2))
            self.screen.blit(text_surf, text_rect)

        if len(self.reward_history) < 2:
            pygame.draw.rect(self.screen, BOX_COLOR, (x, y, width, height))
            pygame.draw.rect(self.screen, BOX_BORDER_COLOR, (x, y, width, height), 2)
            return

        graph_surf = self.create_reward_graph_surface(width, height)
        self.screen.blit(graph_surf, (x, y))

    def create_astar_graph_surface(self, width, height):
        """Create A* path planning visualization showing all accumulated paths step-by-step"""

        if len(self.accumulated_paths) == 0:
            surf = pygame.Surface((width, height))
            surf.fill(BOX_COLOR)
            return surf

        if (self.cached_astar_graph is not None and
            abs(self.path_progress - self.cached_astar_progress) < 0.02 and
            self.animation_phase == self.cached_astar_phase and
            len(self.accumulated_paths) == self.cached_astar_paths_len):
            return self.cached_astar_graph

        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100, facecolor='white')
        ax.set_facecolor('white')

        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(self.grid_size - 0.5, -0.5)
        ax.set_aspect('equal')

        ee_grid = self.env.astar_estimator._world_to_grid(self.env.ee_position[:2])
        ee_color = (100/255, 150/255, 255/255)
        ax.plot(ee_grid[0], ee_grid[1], "o", color=ee_color, markersize=6)

        for ox, oy in self.static_obstacles:
            rect_width = 0.08
            rect_height = 0.85
            rect = plt.Rectangle((ox - rect_width/2, oy - rect_height/2),
                                rect_width, rect_height,
                                color='red', linewidth=0)
            ax.add_patch(rect)

        num_targets_to_show = len(self.accumulated_paths)

        if self.animation_phase == 2:
            num_targets_to_show = len(self.accumulated_paths)
        else:
            num_targets_to_show = len(self.accumulated_paths)

        for i in range(num_targets_to_show):
            _, _, target_grid = self.accumulated_paths[i]
            square_size = 0.15
            square = plt.Rectangle((target_grid[0] - square_size/2, target_grid[1] - square_size/2),
                                  square_size, square_size,
                                  color='green', linewidth=0)
            ax.add_patch(square)

        for i, (path, explored_nodes, target_grid) in enumerate(self.accumulated_paths):
            is_current_step = (i == len(self.accumulated_paths) - 1) and (self.animation_phase == 2)

            if is_current_step:

                if self.path_progress < 0.5:

                    if explored_nodes:
                        progress_in_first_half = self.path_progress / 0.5
                        num_explored = max(1, int(len(explored_nodes) * progress_in_first_half))

                        for j in range(num_explored):
                            node = explored_nodes[j]
                            ax.plot(node[0], node[1], "o", color='blue', markersize=3,
                                   markerfacecolor='none', markeredgewidth=0.5)
                else:

                    if explored_nodes:

                        for node in explored_nodes:
                            ax.plot(node[0], node[1], "o", color='blue', markersize=3,
                                   markerfacecolor='none', markeredgewidth=0.5)

                    if path and len(path) > 1:
                        progress_in_second_half = (self.path_progress - 0.5) / 0.5
                        num_points = max(2, int(len(path) * progress_in_second_half))

                        if num_points >= 2:
                            path_x = [p[0] for p in path[:num_points]]
                            path_y = [p[1] for p in path[:num_points]]
                            ax.plot(path_x, path_y, "-", color='darkblue', linewidth=0.4)
            else:

                if explored_nodes:

                    for node in explored_nodes:
                        ax.plot(node[0], node[1], "o", color='blue', markersize=3,
                               markerfacecolor='none', markeredgewidth=0.5)

                if path and len(path) > 1:
                    path_x = [p[0] for p in path]
                    path_y = [p[1] for p in path]
                    ax.plot(path_x, path_y, "-", color='darkblue', linewidth=0.4)

        ax.set_title('A* Path Planning', color='#64C8FF', fontsize=9, fontweight='bold', pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

        plt.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.12)

        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        buf = canvas.buffer_rgba()
        size = canvas.get_width_height()

        surf = pygame.image.frombuffer(buf, size, "RGBA")
        plt.close(fig)

        self.cached_astar_graph = surf
        self.cached_astar_progress = self.path_progress
        self.cached_astar_phase = self.animation_phase
        self.cached_astar_paths_len = len(self.accumulated_paths)

        return surf

    def draw_astar_graph(self, x, y, width, height):
        """Draw A* path planning graph showing all accumulated paths"""

        if len(self.accumulated_paths) == 0:
            pygame.draw.rect(self.screen, BOX_COLOR, (x, y, width, height))
            pygame.draw.rect(self.screen, BOX_BORDER_COLOR, (x, y, width, height), 2)
            return

        graph_surf = self.create_astar_graph_surface(width, height)
        self.screen.blit(graph_surf, (x, y))

    def render(self):
        """Render the visualization"""
        self.screen.fill(BG_COLOR)

        win_w, win_h = self.screen.get_size()
        info_width = 420
        graph_width = 350
        graph_height = 220
        gap = 80

        grid_area_w = win_w - info_width - gap
        grid_area_h = win_h - 60

        cell_size = min(grid_area_w // self.grid_size, grid_area_h // self.grid_size) // 2
        grid_w = self.grid_size * cell_size
        grid_h = self.grid_size * cell_size

        grid_x = (grid_area_w - grid_w) // 2
        grid_y = (win_h - grid_h) // 2

        self.draw_grid(cell_size, grid_x, grid_y)
        self.draw_obstacles(cell_size, grid_x, grid_y)
        self.draw_cubes(cell_size, grid_x, grid_y)
        agent_pos = self.draw_agent(cell_size, grid_x, grid_y, grid_h)
        self.draw_path(cell_size, grid_x, grid_y, grid_h, agent_pos)

        info_x = grid_x + grid_w + gap
        info_y = 20
        info_panel_height = win_h - graph_height - 60
        self.draw_info_panel(info_x, info_y, info_width, info_panel_height)

        graph_y = info_y + info_panel_height + 20
        graph_gap = 20

        self.draw_reward_graph(info_x, graph_y, graph_width, graph_height)

        astar_graph_x = info_x + graph_width + graph_gap
        self.draw_astar_graph(astar_graph_x, graph_y, graph_width, graph_height)

        pygame.display.flip()

    def update_playback(self):
        """Update episode playback with 3-phase animation"""

        if not self.current_episode or self.paused:
            return

        if not self.auto_advance:
            return

        if not self.timer_started and self.episode_step == 0 and self.animation_phase == 0:
            self.episode_start_time = time.time()
            self.timer_started = True

        if not self.paused and self.timer_started and self.episode_step < len(self.current_episode.actions):
            self.episode_elapsed_time = time.time() - self.episode_start_time

        if self.animation_phase == 0:
            self.phase_timer += self.playback_speed

            if self.phase_timer >= self.playback_speed:

                if self.episode_step < len(self.current_episode.actions):
                    action_idx = self.episode_step
                    self.selected_cube_idx = self.current_episode.actions[action_idx]
                    self.current_path = self.current_episode.paths[action_idx]

                    self.current_obs_components['dist_to_ee'] = self.current_episode.metrics['distances_to_ee'][action_idx]
                    self.current_obs_components['dist_to_container'] = self.current_episode.metrics['distances_to_container'][action_idx]
                    self.current_obs_components['obstacle_proximity'] = self.current_episode.metrics['obstacle_proximities'][action_idx]
                    self.current_obs_components['reachability'] = self.current_episode.metrics['reachability_flags'][action_idx]
                    self.current_obs_components['path_clearance'] = self.current_episode.metrics['path_clearances'][action_idx]
                    self.current_obs_components['invalid_pick'] = 0.0
                    self.current_obs_components['items_left'] = len(self.current_episode.actions) - action_idx

                    path_length = self.current_episode.metrics['path_lengths'][action_idx]
                    reward_components = self.calculate_reward_components(
                        path_length=path_length,
                        dist_to_container=self.current_obs_components['dist_to_container'],
                        obstacle_proximity=self.current_obs_components['obstacle_proximity'],
                        reachability=self.current_obs_components['reachability'],
                        path_clearance=self.current_obs_components['path_clearance'],
                        dist_to_ee=self.current_obs_components['dist_to_ee']
                    )

                    self.current_reward_components['r_total'] = self.current_episode.rewards[action_idx]
                    self.current_reward_components['r_path'] = reward_components['r_path']
                    self.current_reward_components['r_container'] = reward_components['r_container']
                    self.current_reward_components['r_obstacle'] = reward_components['r_obstacle']
                    self.current_reward_components['r_reachability'] = reward_components['r_reachability']
                    self.current_reward_components['r_clearance'] = reward_components['r_clearance']
                    self.current_obs_components['dist_to_origin'] = np.linalg.norm(self.env.object_positions[self.selected_cube_idx][:2])

                    self.reward_history.append(self.current_episode.rewards[action_idx])

                    self.animation_phase = 1
                    self.phase_timer = 0
                else:
                    self.generate_new_episode()

        elif self.animation_phase == 1:
            self.phase_timer += self.playback_speed

            if self.phase_timer >= self.selection_delay:
                has_valid_path = (self.current_path is not None and
                                 len(self.current_path) > 1)

                if has_valid_path:
                    self.animation_phase = 2
                    self.phase_timer = 0
                    self.path_progress = 0.0

                    _, explored_nodes = self._get_astar_path_with_explored(self.selected_cube_idx)

                    target_pos = self.current_episode.cube_positions[self.selected_cube_idx][:2]
                    target_grid = self.env.astar_estimator._world_to_grid(target_pos)

                    self.accumulated_paths.append((self.current_path.copy(), explored_nodes, target_grid))
                else:
                    self.animation_phase = 3
                    self.phase_timer = 0

        elif self.animation_phase == 2:
            self.phase_timer += self.playback_speed
            self.path_progress = min(1.0, self.phase_timer / self.path_animation_duration)

            if self.phase_timer >= self.path_animation_duration:
                self.animation_phase = 3
                self.phase_timer = 0
                self.path_progress = 1.0

        elif self.animation_phase == 3:
            self.phase_timer += self.playback_speed

            if self.phase_timer >= self.post_animation_delay:
                self.episode_step += 1

                self.selected_cube_idx = None
                self.current_path = None

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
            self.handle_events()

            self.update_playback()

            self.render()

            self.clock.tick(self.fps)

        pygame.quit()
        print("\n" + "=" * 70)
        print("VISUALIZER CLOSED")
        print("=" * 70)
        print(f"Total Episodes: {self.episode_count}")
        print("=" * 70)


def main():
    """Main function"""
    args = parse_args()

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
