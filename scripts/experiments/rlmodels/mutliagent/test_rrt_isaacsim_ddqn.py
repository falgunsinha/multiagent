"""
Test Trained RL Models with RRT Path Planning in Isaac Sim
Tests multiple trained models (DDQN, PPO, SAC, etc.) in Isaac Sim environment.

Usage:
    C:\isaacsim\python.bat test_rrt_isaacsim_ddqn.py --experiment exp1
"""

import argparse
import sys
from pathlib import Path

# Parse command-line arguments BEFORE importing Isaac Sim
parser = argparse.ArgumentParser(description="Test trained RL models with Isaac Sim RRT")
parser.add_argument("--experiment", type=str, required=True,
                   help="Experiment config name (exp1, exp2, etc.)")
parser.add_argument("--grid_size", type=int, default=4,
                   help="Grid size (default: 4)")
parser.add_argument("--num_cubes", type=int, default=9,
                   help="Number of cubes (default: 9)")
parser.add_argument("--episodes", type=int, default=99,
                   help="Number of test episodes per model (default: 99)")
parser.add_argument("--skip_viz", action="store_true",
                   help="Skip visualization generation for faster testing")
parser.add_argument("--headless", action="store_true",
                   help="Run in headless mode")
parser.add_argument("--use_wandb", action="store_true",
                   help="Use Weights & Biases for logging")
args = parser.parse_args()

# Create SimulationApp BEFORE importing any Isaac Sim modules
try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import os
import time
import numpy as np
from datetime import datetime
import json
import omni.timeline
import omni.usd

# Isaac Sim imports
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.robot_motion.motion_generation.lula import RRT
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver, PathPlannerVisualizer
from pxr import UsdGeom, UsdPhysics

# Add project root to path
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.manipulators import SingleManipulator
from src.grippers import ParallelGripper

import torch
from src.rl.object_selection_env_rrt import ObjectSelectionEnvRRT
from src.rl.doubleDQN import DoubleDQNAgent

# Import adapters
try:
    from adapters import (
        FeatureAggregationAdapter,
        PCAStateAdapter,
        RandomProjectionAdapter,
        DiscreteActionMapper,
        ContinuousToDiscreteAdapter,
        WeightedActionAdapter,
        ProbabilisticActionAdapter
    )
    ADAPTERS_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Adapters not available: {e}")
    ADAPTERS_AVAILABLE = False


class MetricsCollector:
    """Collect and aggregate metrics during testing"""

    def __init__(self):
        self.episode_data = []
        self.current_episode = {}

    def start_episode(self, episode_num, model_name, seed=None):
        """Start tracking a new episode"""
        self.current_episode = {
            'episode': episode_num,
            'model': model_name,
            'reward': 0,
            'length': 0,
            'success': 0,
            'picks': 0,
            'start_time': time.time(),
            'seed': seed,
            'collisions': 0,
            'rrt_failures': 0,  # NEW: Track RRT planning failures
            'pick_failures': 0,  # NEW: Track pick execution failures
            'successful_picks': 0,  # NEW: Track successful picks
            'unreachable_cubes': 0,  # NEW: Track cubes that were never reachable
            'actions': [],
            'optimal_steps': None  # Will be set when we know the goal
        }

    def update_step(self, reward, action=None, collision=False, info=None):
        """Update metrics for current step"""
        self.current_episode['reward'] += reward
        self.current_episode['length'] += 1
        if action is not None:
            self.current_episode['actions'].append(action)
        if collision:
            self.current_episode['collisions'] += 1

        # NEW: Track RRT and pick failures from info dict
        if info is not None:
            if info.get('rrt_success') == False:
                self.current_episode['rrt_failures'] += 1
            if info.get('pick_success') == False and info.get('rrt_success') == True:
                self.current_episode['pick_failures'] += 1
            if info.get('pick_success') == True:
                self.current_episode['successful_picks'] += 1

    def set_optimal_steps(self, optimal_steps):
        """Set the optimal number of steps for path efficiency calculation"""
        self.current_episode['optimal_steps'] = optimal_steps

    def end_episode(self, success, picks, total_cubes, info=None):
        """Finish current episode"""
        self.current_episode['success'] = 1 if success else 0
        self.current_episode['picks'] = picks
        self.current_episode['total_cubes'] = total_cubes
        self.current_episode['duration'] = time.time() - self.current_episode['start_time']

        # NEW: Track unreachable cubes and RRT path lengths from info dict
        if info is not None:
            self.current_episode['unreachable_cubes'] = info.get('episode_unreachable_cubes', 0)
            # Track RRT path lengths (joint space distance)
            self.current_episode['total_rrt_path_length'] = info.get('total_rrt_path_length', 0.0)
            self.current_episode['avg_rrt_path_length'] = info.get('avg_rrt_path_length', 0.0)
            self.current_episode['total_rrt_optimal_path_length'] = info.get('total_rrt_optimal_path_length', 0.0)
            self.current_episode['avg_rrt_optimal_path_length'] = info.get('avg_rrt_optimal_path_length', 0.0)
            self.current_episode['avg_rrt_planning_time'] = info.get('avg_rrt_planning_time', 0.0)
            self.current_episode['rrt_success_rate'] = info.get('rrt_success_rate', 0.0)
        else:
            self.current_episode['total_rrt_path_length'] = 0.0
            self.current_episode['avg_rrt_path_length'] = 0.0
            self.current_episode['total_rrt_optimal_path_length'] = 0.0
            self.current_episode['avg_rrt_optimal_path_length'] = 0.0
            self.current_episode['avg_rrt_planning_time'] = 0.0
            self.current_episode['rrt_success_rate'] = 0.0

        # UPDATED: Calculate unattempted picks and add to pick_failures
        # Unattempted picks = cubes that were never attempted (due to RRT failures or truncation)
        successful_picks = self.current_episode['successful_picks']
        execution_failures = self.current_episode['pick_failures']  # Only execution failures so far
        unattempted_picks = total_cubes - successful_picks - execution_failures

        # Total pick failures = execution failures + unattempted picks
        self.current_episode['pick_failures'] = execution_failures + unattempted_picks
        self.current_episode['unattempted_picks'] = unattempted_picks  # Track separately for analysis

        # Calculate additional metrics
        actions = self.current_episode['actions']

        # RRT Path efficiency: optimal_path_length / actual_path_length
        # (1.0 = perfect straight line, <1.0 = took longer path, >1.0 = impossible/error)
        # This measures how close the RRT path is to the optimal straight-line path in joint space
        total_actual = self.current_episode['total_rrt_path_length']
        total_optimal = self.current_episode['total_rrt_optimal_path_length']

        if total_actual > 0 and total_optimal > 0:
            self.current_episode['path_efficiency'] = total_optimal / total_actual
        else:
            # No RRT paths were executed (all failures or no picks attempted)
            self.current_episode['path_efficiency'] = 0.0

        # Action diversity metrics
        if len(actions) > 0:
            unique_actions = len(set(actions))
            self.current_episode['unique_actions'] = unique_actions
            self.current_episode['action_diversity_ratio'] = unique_actions / len(actions)

            # Action entropy (Shannon entropy)
            action_counts = np.bincount(actions)
            action_probs = action_counts[action_counts > 0] / len(actions)
            self.current_episode['action_entropy'] = -np.sum(action_probs * np.log2(action_probs + 1e-10))
        else:
            self.current_episode['unique_actions'] = 0
            self.current_episode['action_diversity_ratio'] = 0.0
            self.current_episode['action_entropy'] = 0.0

        # Remove actions list to save memory (we only need the metrics)
        del self.current_episode['actions']

        self.episode_data.append(self.current_episode.copy())

    def get_model_stats(self, model_name):
        """Get statistics for a specific model"""
        model_episodes = [ep for ep in self.episode_data if ep['model'] == model_name]
        if not model_episodes:
            return {}

        return {
            'avg_reward': np.mean([ep['reward'] for ep in model_episodes]),
            'std_reward': np.std([ep['reward'] for ep in model_episodes]),
            'success_rate': np.mean([ep['success'] for ep in model_episodes]),
            'avg_length': np.mean([ep['length'] for ep in model_episodes]),
            'avg_picks': np.mean([ep['picks'] for ep in model_episodes]),
            'avg_duration': np.mean([ep['duration'] for ep in model_episodes]),
            'avg_collisions': np.mean([ep['collisions'] for ep in model_episodes]),
            'avg_rrt_failures': np.mean([ep['rrt_failures'] for ep in model_episodes]),  # NEW
            'avg_pick_failures': np.mean([ep['pick_failures'] for ep in model_episodes]),  # NEW
            'avg_successful_picks': np.mean([ep['successful_picks'] for ep in model_episodes]),  # NEW
            'avg_unreachable_cubes': np.mean([ep.get('unreachable_cubes', 0) for ep in model_episodes]),  # NEW
            'avg_path_efficiency': np.mean([ep['path_efficiency'] for ep in model_episodes if ep['path_efficiency'] != float('inf')]),
            'avg_unique_actions': np.mean([ep['unique_actions'] for ep in model_episodes]),
            'avg_action_diversity': np.mean([ep['action_diversity_ratio'] for ep in model_episodes]),
            'avg_action_entropy': np.mean([ep['action_entropy'] for ep in model_episodes]),
            'episodes': len(model_episodes)
        }

    def get_all_data(self):
        """Get all episode data"""
        return self.episode_data


class FrankaRRTTrainer:
    """
    Franka controller for RRT-based Double DQN training.
    Simplified version focused on training.
    """

    def __init__(self, num_cubes=9, training_grid_size=4):
        self.num_cubes = num_cubes
        self.training_grid_size = training_grid_size
        self.world = None
        self.franka = None
        self.gripper = None
        self.rrt = None
        self.path_planner_visualizer = None
        self.container = None
        self.container_dimensions = None
        self.cubes = []
        self.cube_positions = []
        self.obstacle_prims = []  # Track obstacle prims for regeneration

        print(f"[TRAINER] Initializing Franka RRT Trainer for Double DQN")
        print(f"[TRAINER] Grid: {training_grid_size}x{training_grid_size}, Cubes: {num_cubes}")

    def setup_scene(self):
        """Setup Isaac Sim scene with Franka and cubes"""
        print("[TRAINER] Setting up scene...")

        # Create world
        self.world = World(stage_units_in_meters=1.0)

        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Add Franka robot USD to stage
        assets_root_path = get_assets_root_path()
        franka_prim_path = "/World/Franka"

        franka_usd_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        robot_prim = add_reference_to_stage(usd_path=franka_usd_path, prim_path=franka_prim_path)
        robot_prim.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
        robot_prim.GetVariantSet("Mesh").SetVariantSelection("Quality")

        # Create gripper
        self.gripper = ParallelGripper(
            end_effector_prim_path=f"{franka_prim_path}/panda_rightfinger",
            joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
            joint_opened_positions=np.array([0.04, 0.04]),
            joint_closed_positions=np.array([0.0, 0.0]),
            action_deltas=np.array([0.01, 0.01])
        )

        # Create Franka manipulator
        self.franka = self.world.scene.add(
            SingleManipulator(
                prim_path=franka_prim_path,
                name="franka",
                end_effector_prim_path=f"{franka_prim_path}/panda_rightfinger",
                gripper=self.gripper,
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0])
            )
        )

        # Initialize RRT planner
        print("[TRAINER] Initializing RRT planner...")
        self._setup_rrt_planner()

        # Add container
        print("[TRAINER] Adding container...")
        self._setup_container()

        # Spawn cubes
        self._spawn_cubes()

        # Create random obstacles
        print("[TRAINER] Creating random obstacles in empty cells...")
        self._create_random_obstacles()

        # Reset world
        self.world.reset()

        print("[TRAINER] Scene setup complete")

    def _setup_rrt_planner(self):
        """Setup RRT path planner"""
        try:
            mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")

            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(script_dir, "..", "..", "..")
            robot_description_file = os.path.join(project_root, "assets", "franka_conservative_spheres_robot_description.yaml")
            robot_description_file = os.path.normpath(robot_description_file)

            urdf_path = os.path.join(mg_extension_path, "motion_policy_configs", "franka", "lula_franka_gen.urdf")
            rrt_config_file = os.path.join(mg_extension_path, "path_planner_configs", "franka", "rrt", "franka_planner_config.yaml")

            if not os.path.exists(robot_description_file):
                print(f"[TRAINER WARNING] Robot description not found: {robot_description_file}")
                robot_description_file = os.path.join(mg_extension_path, "motion_policy_configs", "franka", "rmpflow", "robot_descriptor.yaml")

            self.rrt = RRT(
                robot_description_path=robot_description_file,
                urdf_path=urdf_path,
                rrt_config_path=rrt_config_file,
                end_effector_frame_name="right_gripper"
            )
            self.rrt.set_max_iterations(10000)

            self.path_planner_visualizer = PathPlannerVisualizer(
                robot_articulation=self.franka,
                path_planner=self.rrt
            )

            self.kinematics_solver = LulaKinematicsSolver(
                robot_description_path=robot_description_file,
                urdf_path=urdf_path
            )
            self.articulation_kinematics_solver = ArticulationKinematicsSolver(
                self.franka,
                self.kinematics_solver,
                "right_gripper"
            )

            print("[TRAINER] RRT planner and kinematics solvers initialized")
        except Exception as e:
            print(f"[TRAINER ERROR] Failed to initialize RRT: {e}")
            import traceback
            traceback.print_exc()
            self.rrt = None
            self.kinematics_solver = None
            self.articulation_kinematics_solver = None

    def _setup_container(self):
        """Setup container for placing cubes"""
        from isaacsim.core.utils.stage import get_current_stage

        container_prim_path = "/World/Container"
        container_usd_path = f"{get_assets_root_path()}/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Storage/Containers/Container_I/Container_I04_160x120x64cm_PR_V_NVD_01.usd"
        add_reference_to_stage(usd_path=container_usd_path, prim_path=container_prim_path)

        container_position = np.array([0.30, 0.50, 0.0])
        scale = np.array([0.3, 0.3, 0.2])
        original_size = np.array([1.60, 1.20, 0.64])
        self.container_dimensions = original_size * scale

        self.container = self.world.scene.add(
            SingleXFormPrim(
                prim_path=container_prim_path,
                name="container",
                translation=container_position,
                scale=scale
            )
        )

        stage = get_current_stage()
        container_prim = stage.GetPrimAtPath(container_prim_path)
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(container_prim)
        rigid_body_api.CreateKinematicEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(container_prim)

    def _world_to_grid(self, world_pos):
        """Convert world coordinates to grid coordinates"""
        cube_spacing = 0.13 if self.training_grid_size > 3 else 0.15
        grid_center = np.array([0.45, -0.10])
        grid_extent = (self.training_grid_size - 1) * cube_spacing
        start_x = grid_center[0] - (grid_extent / 2.0)
        start_y = grid_center[1] - (grid_extent / 2.0)

        grid_x = int(round((world_pos[0] - start_x) / cube_spacing))
        grid_y = int(round((world_pos[1] - start_y) / cube_spacing))

        grid_x = max(0, min(self.training_grid_size - 1, grid_x))
        grid_y = max(0, min(self.training_grid_size - 1, grid_y))

        return grid_x, grid_y

    def _grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        cube_spacing = 0.13 if self.training_grid_size > 3 else 0.15
        grid_center = np.array([0.45, -0.10])
        grid_extent = (self.training_grid_size - 1) * cube_spacing
        start_x = grid_center[0] - (grid_extent / 2.0)
        start_y = grid_center[1] - (grid_extent / 2.0)

        world_x = start_x + (grid_x * cube_spacing)
        world_y = start_y + (grid_y * cube_spacing)

        return world_x, world_y

    def _remove_obstacles(self):
        """Remove existing obstacles from the scene AND from RRT planner"""
        if not hasattr(self, 'obstacle_prims') or not self.obstacle_prims:
            return

        for obs_prim in self.obstacle_prims:
            try:
                # CRITICAL FIX: Remove from RRT planner first
                if self.rrt is not None:
                    self.rrt.remove_obstacle(obs_prim)

                # Then remove from scene
                if obs_prim.is_valid():
                    self.world.scene.remove_object(obs_prim.name)
            except Exception as e:
                # Silently continue if removal fails
                pass

        self.obstacle_prims = []

        # CRITICAL FIX: Update RRT world after removing obstacles
        if self.rrt is not None:
            self.rrt.update_world()

    def _create_random_obstacles(self):
        """Create random obstacles in empty grid cells"""
        # Remove existing obstacles first
        self._remove_obstacles()

        # UPDATED: Exactly 1 obstacle for all grid sizes (for consistent testing)
        num_obstacles_map = {
            3: 1,  # 3x3 grid: 1 obstacle (fixed)
            4: 1,  # 4x4 grid: 1 obstacle (fixed) - REDUCED for testing
            6: 1,  # 6x6 grid: 1 obstacle (fixed) - REDUCED for testing
        }
        num_obstacles = num_obstacles_map.get(self.training_grid_size, 1)

        cube_cells = set()
        for i in range(self.num_cubes):
            cube_pos = self.cube_positions[i]
            grid_col, grid_row = self._world_to_grid(cube_pos[:2])
            cube_cells.add((grid_col, grid_row))

        empty_cells = []
        for grid_x in range(self.training_grid_size):
            for grid_y in range(self.training_grid_size):
                if (grid_x, grid_y) not in cube_cells:
                    empty_cells.append((grid_x, grid_y))

        if len(empty_cells) < num_obstacles:
            num_obstacles = len(empty_cells)

        if num_obstacles == 0:
            return

        np.random.shuffle(empty_cells)
        selected_cells = empty_cells[:num_obstacles]

        for idx, (grid_x, grid_y) in enumerate(selected_cells):
            world_pos = self._grid_to_world(grid_x, grid_y)
            obs_position = np.array([world_pos[0], world_pos[1], 0.055])
            obs_name = f"Obstacle_{idx}"

            obstacle = self.world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/{obs_name}",
                    name=obs_name,
                    position=obs_position,
                    scale=np.array([0.11, 0.11, 0.11]),
                    color=np.array([1.0, 0.0, 0.0])
                )
            )

            # CRITICAL FIX: Add obstacle to RRT planner for collision avoidance!
            # Without this, RRT doesn't know about the obstacles and plans through them
            if self.rrt is not None:
                self.rrt.add_obstacle(obstacle, static=False)

            # Store obstacle prim for later removal/regeneration
            self.obstacle_prims.append(obstacle)

        # CRITICAL FIX: Update RRT world after adding all obstacles
        if self.rrt is not None and num_obstacles > 0:
            self.rrt.update_world()
            print(f"[TRAINER] Added {num_obstacles} obstacles to RRT collision checker")

    def _spawn_cubes(self):
        """Spawn cubes in grid pattern"""
        cube_size = 0.0515
        cube_spacing = 0.13 if self.training_grid_size > 3 else 0.15
        grid_center_x = 0.45
        grid_center_y = -0.10

        grid_extent_x = (self.training_grid_size - 1) * cube_spacing
        grid_extent_y = (self.training_grid_size - 1) * cube_spacing
        start_x = grid_center_x - (grid_extent_x / 2.0)
        start_y = grid_center_y - (grid_extent_y / 2.0)

        total_cells = self.training_grid_size * self.training_grid_size
        selected_indices = np.random.choice(total_cells, size=self.num_cubes, replace=False)
        selected_cells = set(selected_indices)

        cube_count = 0
        for row in range(self.training_grid_size):
            for col in range(self.training_grid_size):
                cell_index = row * self.training_grid_size + col
                if cell_index not in selected_cells:
                    continue

                base_x = start_x + (row * cube_spacing)
                base_y = start_y + (col * cube_spacing)

                offset_x = np.random.uniform(-0.02, 0.02)
                offset_y = np.random.uniform(-0.02, 0.02)

                position = np.array([base_x + offset_x, base_y + offset_y, cube_size / 2.0])

                cube_number = cube_count + 1
                cube_name = f"Cube_{cube_number}"
                cube = self.world.scene.add(
                    DynamicCuboid(
                        prim_path=f"/World/{cube_name}",
                        name=cube_name,
                        position=position,
                        scale=np.array([cube_size, cube_size, cube_size]),
                        color=np.array([0.0, 0.5, 1.0])
                    )
                )
                self.cubes.append((cube, cube_name))
                self.cube_positions.append(position)
                cube_count += 1

    def get_cube_positions(self):
        """Get current cube positions"""
        positions = []
        for cube, cube_name in self.cubes:
            pos, _ = cube.get_world_pose()
            positions.append(pos)
        return positions

    def randomize_cube_positions(self):
        """Randomize cube positions for each episode (adds stochasticity)"""
        if not self.cubes:
            return

        # Get grid parameters
        cube_spacing = 0.15
        grid_center_x = 0.3
        grid_center_y = 0.0
        cube_size = 0.05

        grid_extent_x = (self.training_grid_size - 1) * cube_spacing
        grid_extent_y = (self.training_grid_size - 1) * cube_spacing
        start_x = grid_center_x - (grid_extent_x / 2.0)
        start_y = grid_center_y - (grid_extent_y / 2.0)

        # Select random cells for cubes
        total_cells = self.training_grid_size * self.training_grid_size
        selected_indices = np.random.choice(total_cells, size=self.num_cubes, replace=False)
        selected_cells = set(selected_indices)

        # Update cube positions
        cube_idx = 0
        self.cube_positions = []
        for row in range(self.training_grid_size):
            for col in range(self.training_grid_size):
                cell_index = row * self.training_grid_size + col
                if cell_index not in selected_cells:
                    continue

                if cube_idx >= len(self.cubes):
                    break

                # Calculate new position with random offset
                base_x = start_x + (row * cube_spacing)
                base_y = start_y + (col * cube_spacing)
                offset_x = np.random.uniform(-0.02, 0.02)
                offset_y = np.random.uniform(-0.02, 0.02)
                position = np.array([base_x + offset_x, base_y + offset_y, cube_size / 2.0])

                # Update cube position
                cube, cube_name = self.cubes[cube_idx]
                cube.set_world_pose(position=position)
                self.cube_positions.append(position)
                cube_idx += 1


def main():
    """Main testing loop"""
    # Base directory for models (relative to this script)
    base_dir = Path(__file__).parent

    # Experiment configurations (matching run_isaac_sim_headless_exp.py)
    EXPERIMENTS = {
        "exp1": {
            "name": "Discrete Model Comparison",
            "models": [
                # Custom-DDQN: Trained on 4x4 grid with 9 cubes - TEST FIRST!
                # Grid: 4x4 = 16 positions, State: 16 objects × 6 features = 96D, Actions: 16
                {"name": "Custom-DDQN", "path": str(base_dir / "models/pretrained/custom_ddqn_grid4_cubes9.pt"), "type": "ddqn", "state_dim": 96, "action_dim": 16, "custom": True},
                {"name": "Duel-DDQN", "path": str(base_dir / "models/pretrained/duel_ddqn_lunarlander.pth"), "type": "ddqn", "state_dim": 8, "action_dim": 4},
                {"name": "PER-DDQN-Light", "path": str(base_dir / "models/pretrained/per_ddqn_light_lunarlander.pth"), "type": "ddqn", "state_dim": 8, "action_dim": 4},
                {"name": "PER-DDQN-Full", "path": str(base_dir / "models/pretrained/per_ddqn_full_lunarlander.pth"), "type": "ddqn", "state_dim": 8, "action_dim": 4},
                {"name": "C51-DDQN", "path": str(base_dir / "models/pretrained/c51_ddqn_lunarlander.pth"), "type": "ddqn", "state_dim": 8, "action_dim": 4},
                {"name": "SAC-Discrete", "path": str(base_dir / "models/pretrained/sac_discrete_lunarlander_actor.pth"), "type": "pytorch", "state_dim": 8, "action_dim": 4},
                {"name": "PPO-Discrete", "path": str(base_dir / "models/pretrained/ppo_discrete_lunarlander_actor.pth"), "type": "pytorch", "state_dim": 4, "action_dim": 2},
            ]
        },
        "exp2": {
            "name": "Continuous Model Comparison",
            "models": [
                # Custom-DDQN: Trained on 4x4 grid with 9 cubes - TEST FIRST!
                {"name": "Custom-DDQN", "path": str(base_dir / "models/pretrained/custom_ddqn_grid4_cubes9.pt"), "type": "ddqn", "state_dim": 96, "action_dim": 16, "custom": True},
                {"name": "DDPG", "path": str(base_dir / "models/pretrained/ddpg_pendulum_actor.pth"), "type": "continuous", "state_dim": 3, "action_dim": 1},
                {"name": "TD3", "path": str(base_dir / "models/pretrained/td3_bipedalwalker_actor.pth"), "type": "continuous", "state_dim": 24, "action_dim": 4},
                {"name": "PPO-Continuous", "path": str(base_dir / "models/pretrained/ppo_continuous_pendulum_actor.pth"), "type": "continuous", "state_dim": 3, "action_dim": 1},
                {"name": "SAC-Continuous", "path": str(base_dir / "models/pretrained/sac_continuous_bipedalwalker_actor.pth"), "type": "continuous", "state_dim": 24, "action_dim": 4},
            ]
        }
    }

    # Get experiment config
    if args.experiment not in EXPERIMENTS:
        print(f"ERROR: Unknown experiment '{args.experiment}'")
        print(f"Available experiments: {list(EXPERIMENTS.keys())}")
        sys.exit(1)

    exp_config = EXPERIMENTS[args.experiment]
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_config['name']}")
    print(f"{'='*60}")
    print(f"Grid: {args.grid_size}x{args.grid_size}")
    print(f"Cubes: {args.num_cubes}")
    print(f"Episodes per model: {args.episodes}")
    print(f"Total models: {len(exp_config['models'])}")
    print(f"{'='*60}\n")

    # Initialize metrics collector
    metrics_collector = MetricsCollector()

    # Initialize W&B if requested
    if args.use_wandb:
        try:
            import wandb
            import shutil

            # Delete old wandb folder to start fresh
            wandb_dir = Path(__file__).parent / "wandb"
            if wandb_dir.exists():
                print(f"🗑️  Deleting old wandb folder: {wandb_dir}")
                try:
                    shutil.rmtree(wandb_dir)
                    print("   ✓ Old wandb folder deleted")
                except Exception as e:
                    print(f"   ⚠️  Could not delete wandb folder: {e}")
                    print("   Continuing anyway...")

            # Set environment variables to handle SSL and mode
            os.environ['WANDB_MODE'] = 'offline'  # Use offline mode to avoid SSL/network issues

            # Disable SSL warnings
            try:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except:
                pass

            # Initialize in offline mode
            wandb.init(
                project="exp-rl-models",
                entity="falgunsinha",
                name=f"{args.experiment}_{exp_config['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "experiment": args.experiment,
                    "experiment_name": exp_config['name'],
                    "grid_size": args.grid_size,
                    "num_cubes": args.num_cubes,
                    "episodes_per_model": args.episodes,
                    "total_models": len(exp_config['models']),
                    "total_episodes": len(exp_config['models']) * args.episodes,
                },
                tags=[args.experiment, "testing", "isaac_sim"],
                mode="offline",  # Offline mode - sync later with 'wandb sync'
                settings=wandb.Settings(
                    start_method="thread",
                    _disable_stats=True,
                    _disable_meta=True
                )
            )
            print("✅ W&B logging enabled (offline mode)")
            print(f"   Project: exp-rl-models")
            print(f"   Run: {wandb.run.name}")
            print(f"   Data saved locally - sync later with: wandb sync wandb/run-*")
        except Exception as e:
            print(f"⚠️  WARNING: W&B initialization failed: {e}")
            print(f"   Continuing without W&B logging...")
            print(f"   Results will still be saved to CSV files")
            args.use_wandb = False

    # Create trainer
    trainer = FrankaRRTTrainer(
        num_cubes=args.num_cubes,
        training_grid_size=args.grid_size
    )

    # Setup scene
    trainer.setup_scene()

    # Create RL environment
    print("\n[TEST] Creating RRT-based RL environment...")
    max_objects = args.grid_size * args.grid_size

    env = ObjectSelectionEnvRRT(
        franka_controller=trainer,
        max_objects=max_objects,
        max_steps=50,
        num_cubes=args.num_cubes,
        training_grid_size=args.grid_size,
        execute_picks=True,  # Always execute picks during testing
        rrt_planner=trainer.rrt,
        kinematics_solver=trainer.kinematics_solver,
        articulation_kinematics_solver=trainer.articulation_kinematics_solver,
        franka_articulation=trainer.franka
    )

    # Results storage
    import csv
    results_dir = Path(r"C:\isaacsim\cobotproject\scripts\experiments\rlmodels\results") / args.experiment
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "comparison_results.csv"

    all_results = []
    all_episode_data = {}  # Store episode-level data for advanced visualizations

    # Test each model
    for model_idx, model_config in enumerate(exp_config['models'], 1):
        model_name = model_config['name']
        model_path = model_config['path']
        model_type = model_config['type']

        print(f"\n{'='*60}")
        print(f"[{model_idx}/{len(exp_config['models'])}] Testing: {model_name}")
        print(f"{'='*60}")
        print(f"Model path: {model_path}")
        print(f"Model type: {model_type}")

        # Check if model exists
        if not os.path.exists(model_path):
            print(f"WARNING: Model not found: {model_path}")
            print("Skipping this model...")
            continue

        # Load model based on type
        import torch
        import torch.nn as nn

        # Isaac Sim environment dimensions
        # State: max_objects × 6 features (distance_ee, distance_container, obstacle_score, reachability, path_clearance, picked)
        # Actions: grid_size × grid_size positions
        # NOTE: max_objects = grid_size × grid_size (e.g., 9 for 3x3 grid)
        #       but only num_cubes are actually used (e.g., 4 cubes)
        isaac_state_dim = max_objects * 6  # 54D for 3x3 grid (9 objects × 6 features)
        isaac_action_dim = max_objects  # 9 actions for 3x3 grid

        print(f"\nIsaac Sim Environment:")
        print(f"  Grid: {args.grid_size}x{args.grid_size}, Cubes: {args.num_cubes}")
        print(f"  State dimension: {isaac_state_dim}D ({max_objects} objects × 6 features)")
        print(f"  Action dimension: {isaac_action_dim} actions")

        # Initialize adapters
        state_adapter = None
        action_adapter = None
        continuous_adapter = None

        # Determine if this is a custom model or pretrained
        is_custom = model_config.get('custom', False)

        # Get model's expected dimensions
        model_state_dim = model_config.get('state_dim', isaac_state_dim)
        model_action_dim = model_config.get('action_dim', isaac_action_dim)

        # Setup adapters for pretrained models
        if not is_custom and ADAPTERS_AVAILABLE:
            print(f"\nSetting up adapters for pretrained model...")
            print(f"  Isaac Sim: {isaac_state_dim}D state → {isaac_action_dim} actions")
            print(f"  Model expects: {model_state_dim}D state → {model_action_dim} actions")

            # State adapter: isaac_state_dim → 8D, 4D, or 3D
            # (e.g., 54D for 3x3 grid → 8D for LunarLander)
            if model_state_dim == 8:
                state_adapter = PCAStateAdapter(input_dim=isaac_state_dim, output_dim=8)
                print(f"  ✓ State adapter: PCA ({isaac_state_dim}D → 8D)")
            elif model_state_dim == 4:
                state_adapter = PCAStateAdapter(input_dim=isaac_state_dim, output_dim=4)
                print(f"  ✓ State adapter: PCA ({isaac_state_dim}D → 4D)")
            elif model_state_dim == 3:
                state_adapter = PCAStateAdapter(input_dim=isaac_state_dim, output_dim=3)
                print(f"  ✓ State adapter: PCA ({isaac_state_dim}D → 3D)")

            # Action adapter for discrete models
            if model_action_dim == 4 and model_type != "continuous":
                action_adapter = DiscreteActionMapper(source_actions=4, target_actions=isaac_action_dim)
                print(f"  ✓ Action adapter: Discrete Mapper (4 → {isaac_action_dim})")
            elif model_action_dim == 2 and model_type != "continuous":
                action_adapter = DiscreteActionMapper(source_actions=2, target_actions=isaac_action_dim)
                print(f"  ✓ Action adapter: Discrete Mapper (2 → {isaac_action_dim})")

            # Continuous-to-discrete adapter for continuous models
            if model_type == "continuous":
                continuous_adapter = ContinuousToDiscreteAdapter(
                    continuous_dim=model_action_dim,
                    num_cubes=isaac_action_dim
                )
                print(f"  ✓ Continuous adapter: Continuous→Discrete ({model_action_dim}D → {isaac_action_dim})")

        if model_type == "ddqn":
            if is_custom:
                # Custom DDQN: Load directly with Isaac Sim dimensions (no adaptation)
                # For grid4_cubes9: 4x4 grid = 16 positions, state_dim=96 (16×6), action_dim=16
                print(f"Loading custom DDQN model...")
                print(f"  Model trained for: {args.grid_size}x{args.grid_size} grid, {args.num_cubes} cubes")
                print(f"  Dimensions: {isaac_state_dim}D state, {isaac_action_dim} actions")

                agent = DoubleDQNAgent(
                    state_dim=isaac_state_dim,
                    action_dim=isaac_action_dim,
                    learning_rate=1e-3,
                    gamma=0.99,
                    epsilon_start=0.0,
                    epsilon_end=0.0,
                    epsilon_decay=1.0,
                    batch_size=64,
                    buffer_capacity=10000,
                    target_update_freq=1000
                )
                try:
                    agent.load(model_path, weights_only=False)
                    print(f"✓ Loaded custom DDQN model from {model_path}")
                except Exception as e:
                    print(f"ERROR loading custom DDQN model: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            else:
                # Pretrained DDQN: Need to create network with correct architecture
                print(f"Loading pretrained DDQN model...")
                try:
                    # Load the state dict
                    state_dict = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)

                    # Create wrapper agent
                    class PretrainedDQNWrapper:
                        def __init__(self, state_dict, state_dim, action_dim, model_name):
                            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            self.model_name = model_name

                            # Determine architecture from state_dict keys
                            if 'hidden.0.weight' in state_dict:
                                # Dueling DQN architecture
                                hidden_size = state_dict['hidden.0.weight'].shape[0]
                                self.network = nn.Sequential(
                                    nn.Linear(state_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU()
                                ).to(self.device)
                                self.value_stream = nn.Linear(hidden_size, 1).to(self.device)
                                self.advantage_stream = nn.Linear(hidden_size, action_dim).to(self.device)

                                # Load weights
                                self.network[0].load_state_dict({'weight': state_dict['hidden.0.weight'], 'bias': state_dict['hidden.0.bias']})
                                self.network[2].load_state_dict({'weight': state_dict['hidden.2.weight'], 'bias': state_dict['hidden.2.bias']})
                                self.value_stream.load_state_dict({'weight': state_dict['V.weight'], 'bias': state_dict['V.bias']})
                                self.advantage_stream.load_state_dict({'weight': state_dict['A.weight'], 'bias': state_dict['A.bias']})
                                self.is_dueling = True
                            elif 'Q.0.weight' in state_dict:
                                # Standard DQN architecture
                                hidden_size = state_dict['Q.0.weight'].shape[0]
                                self.network = nn.Sequential(
                                    nn.Linear(state_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, action_dim)
                                ).to(self.device)
                                self.network[0].load_state_dict({'weight': state_dict['Q.0.weight'], 'bias': state_dict['Q.0.bias']})
                                self.network[2].load_state_dict({'weight': state_dict['Q.2.weight'], 'bias': state_dict['Q.2.bias']})
                                self.network[4].load_state_dict({'weight': state_dict['Q.4.weight'], 'bias': state_dict['Q.4.bias']})
                                self.is_dueling = False
                            elif 'net.0.weight' in state_dict:
                                # C51 DQN architecture
                                hidden_size = state_dict['net.0.weight'].shape[0]
                                output_size = state_dict['net.4.weight'].shape[0]
                                self.network = nn.Sequential(
                                    nn.Linear(state_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, output_size)
                                ).to(self.device)
                                self.network[0].load_state_dict({'weight': state_dict['net.0.weight'], 'bias': state_dict['net.0.bias']})
                                self.network[2].load_state_dict({'weight': state_dict['net.2.weight'], 'bias': state_dict['net.2.bias']})
                                self.network[4].load_state_dict({'weight': state_dict['net.4.weight'], 'bias': state_dict['net.4.bias']})
                                self.is_dueling = False
                                self.is_c51 = True
                                self.num_atoms = output_size // action_dim
                            else:
                                raise ValueError(f"Unknown architecture for {model_name}")

                            self.network.eval()
                            if hasattr(self, 'value_stream'):
                                self.value_stream.eval()
                                self.advantage_stream.eval()

                        def select_action(self, state, action_mask=None):
                            with torch.no_grad():
                                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                                if self.is_dueling:
                                    features = self.network(state_tensor)
                                    value = self.value_stream(features)
                                    advantage = self.advantage_stream(features)
                                    q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
                                elif hasattr(self, 'is_c51') and self.is_c51:
                                    # C51: Get distribution and compute expected Q-values
                                    logits = self.network(state_tensor)
                                    logits = logits.view(-1, self.num_atoms)
                                    probs = torch.softmax(logits, dim=1)
                                    # Simplified: just use argmax of mean
                                    q_values = probs.mean(dim=1).unsqueeze(0)
                                else:
                                    q_values = self.network(state_tensor)

                                if action_mask is not None:
                                    q_values = q_values.masked_fill(~torch.BoolTensor(action_mask).to(self.device), float('-inf'))

                                action = torch.argmax(q_values, dim=1).item()
                            return action

                        def predict(self, state, deterministic=True):
                            action = self.select_action(state)
                            return action, None

                    agent = PretrainedDQNWrapper(state_dict, model_state_dim, model_action_dim, model_name)
                    print(f"✓ Loaded pretrained DDQN model from {model_path}")
                except Exception as e:
                    print(f"ERROR loading pretrained DDQN model: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        elif model_type == "pytorch":
            # Load raw PyTorch models (SAC-Discrete, PPO-Discrete actor networks)
            print(f"Loading pretrained PyTorch actor model...")
            try:
                # Load the state dict
                state_dict = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)

                # Create wrapper for PyTorch actor networks
                class PyTorchActorWrapper:
                    def __init__(self, state_dict, state_dim, action_dim, model_name, device='cuda'):
                        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
                        self.model_name = model_name

                        # Determine architecture from state_dict keys
                        if 'P.0.weight' in state_dict:
                            # SAC-Discrete architecture
                            hidden_size = state_dict['P.0.weight'].shape[0]
                            self.actor = nn.Sequential(
                                nn.Linear(state_dim, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, action_dim),
                                nn.Softmax(dim=-1)
                            ).to(self.device)
                            self.actor[0].load_state_dict({'weight': state_dict['P.0.weight'], 'bias': state_dict['P.0.bias']})
                            self.actor[2].load_state_dict({'weight': state_dict['P.2.weight'], 'bias': state_dict['P.2.bias']})
                            self.actor[4].load_state_dict({'weight': state_dict['P.4.weight'], 'bias': state_dict['P.4.bias']})
                        elif 'l1.weight' in state_dict:
                            # PPO-Discrete architecture
                            hidden_size = state_dict['l1.weight'].shape[0]
                            self.actor = nn.Sequential(
                                nn.Linear(state_dim, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, action_dim),
                                nn.Softmax(dim=-1)
                            ).to(self.device)
                            self.actor[0].load_state_dict({'weight': state_dict['l1.weight'], 'bias': state_dict['l1.bias']})
                            self.actor[2].load_state_dict({'weight': state_dict['l2.weight'], 'bias': state_dict['l2.bias']})
                            self.actor[4].load_state_dict({'weight': state_dict['l3.weight'], 'bias': state_dict['l3.bias']})
                        else:
                            raise ValueError(f"Unknown architecture for {model_name}")

                        self.actor.eval()

                    def select_action(self, state, action_mask=None):
                        with torch.no_grad():
                            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                            action_probs = self.actor(state_tensor)
                            if action_mask is not None:
                                # Apply action mask
                                action_probs = action_probs * torch.FloatTensor(action_mask).to(self.device)
                                action_probs = action_probs / action_probs.sum()
                            action = torch.argmax(action_probs, dim=-1).item()
                        return action

                    def predict(self, state, deterministic=True):
                        action = self.select_action(state)
                        return action, None

                agent = PyTorchActorWrapper(state_dict, model_state_dim, model_action_dim, model_name)
                print(f"✓ Loaded pretrained PyTorch actor model from {model_path}")
            except Exception as e:
                print(f"ERROR loading PyTorch model: {e}")
                import traceback
                traceback.print_exc()
                continue

        elif model_type == "continuous":
            # Load continuous action models (DDPG, TD3, SAC, PPO)
            print(f"Loading pretrained continuous actor model...")

            if not ADAPTERS_AVAILABLE:
                print(f"⚠️  Skipping continuous model {model_name} - adapters not available")
                continue

            try:
                # Load the state dict
                state_dict = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)

                # Create wrapper for continuous actor networks
                class ContinuousActorWrapper:
                    def __init__(self, state_dict, state_dim, action_dim, model_name, device='cuda'):
                        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
                        self.model_name = model_name
                        self.action_dim = action_dim
                        self.actual_state_dim = None  # Will be set after detecting architecture
                        self.actual_action_dim = None  # Will be set after detecting architecture

                        # Debug: Print available keys
                        print(f"  [DEBUG] State dict keys: {list(state_dict.keys())}")

                        # Determine architecture from state_dict keys
                        if 'l1.weight' in state_dict and 'l3.weight' in state_dict:
                            # DDPG/TD3/PPO architecture (l1, l2, l3 or l1, l2, mu_head)
                            # Detect layer sizes from state dict (use ORIGINAL dimensions from checkpoint)
                            original_state_dim = state_dict['l1.weight'].shape[1]  # Input dimension from checkpoint
                            l1_out = state_dict['l1.weight'].shape[0]
                            l2_out = state_dict['l2.weight'].shape[0]
                            original_action_dim = state_dict.get('l3.weight', state_dict.get('mu_head.weight')).shape[0]

                            # Store actual dimensions
                            self.actual_state_dim = original_state_dim
                            self.actual_action_dim = original_action_dim

                            # Check if it's PPO (has mu_head) or DDPG/TD3 (has l3)
                            if 'mu_head.weight' in state_dict:
                                # PPO-Continuous architecture
                                print(f"  Detected PPO-Continuous architecture")
                                print(f"  Original model: {original_state_dim}D state → {original_action_dim}D action")
                                self.actor = nn.Sequential(
                                    nn.Linear(original_state_dim, l1_out),
                                    nn.ReLU(),
                                    nn.Linear(l1_out, l2_out),
                                    nn.ReLU(),
                                    nn.Linear(l2_out, original_action_dim),
                                    nn.Sigmoid()  # PPO uses sigmoid
                                ).to(self.device)
                                self.actor[0].load_state_dict({'weight': state_dict['l1.weight'], 'bias': state_dict['l1.bias']})
                                self.actor[2].load_state_dict({'weight': state_dict['l2.weight'], 'bias': state_dict['l2.bias']})
                                self.actor[4].load_state_dict({'weight': state_dict['mu_head.weight'], 'bias': state_dict['mu_head.bias']})
                            else:
                                # DDPG/TD3 architecture
                                print(f"  Detected DDPG/TD3 architecture")
                                print(f"  Original model: {original_state_dim}D state → {original_action_dim}D action")
                                self.actor = nn.Sequential(
                                    nn.Linear(original_state_dim, l1_out),
                                    nn.ReLU(),
                                    nn.Linear(l1_out, l2_out),
                                    nn.ReLU(),
                                    nn.Linear(l2_out, original_action_dim),
                                    nn.Tanh()  # DDPG/TD3 use tanh
                                ).to(self.device)
                                self.actor[0].load_state_dict({'weight': state_dict['l1.weight'], 'bias': state_dict['l1.bias']})
                                self.actor[2].load_state_dict({'weight': state_dict['l2.weight'], 'bias': state_dict['l2.bias']})
                                self.actor[4].load_state_dict({'weight': state_dict['l3.weight'], 'bias': state_dict['l3.bias']})

                        elif 'a_net.0.weight' in state_dict and 'mu_layer.weight' in state_dict:
                            # SAC-Continuous architecture (a_net + mu_layer + log_std_layer)
                            print(f"  Detected SAC-Continuous architecture")
                            # Get dimensions from checkpoint
                            original_state_dim = state_dict['a_net.0.weight'].shape[1]
                            h1 = state_dict['a_net.0.weight'].shape[0]
                            h2 = state_dict['a_net.2.weight'].shape[0] if 'a_net.2.weight' in state_dict else h1
                            original_action_dim = state_dict['mu_layer.weight'].shape[0]

                            # Store actual dimensions
                            self.actual_state_dim = original_state_dim
                            self.actual_action_dim = original_action_dim

                            print(f"  Original model: {original_state_dim}D state → {original_action_dim}D action")

                            self.actor = nn.Sequential(
                                nn.Linear(original_state_dim, h1),
                                nn.ReLU(),
                                nn.Linear(h1, h2),
                                nn.ReLU(),
                                nn.Linear(h2, original_action_dim),
                                nn.Tanh()
                            ).to(self.device)
                            self.actor[0].load_state_dict({'weight': state_dict['a_net.0.weight'], 'bias': state_dict['a_net.0.bias']})
                            if 'a_net.2.weight' in state_dict:
                                self.actor[2].load_state_dict({'weight': state_dict['a_net.2.weight'], 'bias': state_dict['a_net.2.bias']})
                            self.actor[4].load_state_dict({'weight': state_dict['mu_layer.weight'], 'bias': state_dict['mu_layer.bias']})

                        elif 'alpha_head.weight' in state_dict and 'beta_head.weight' in state_dict:
                            # PPO-Continuous architecture (dual-head with alpha/beta for Beta distribution)
                            print(f"  Detected PPO-Continuous architecture")
                            # Get dimensions from checkpoint
                            original_state_dim = state_dict['l1.weight'].shape[1]
                            h1 = state_dict['l1.weight'].shape[0]
                            h2 = state_dict['l2.weight'].shape[0]
                            original_action_dim = state_dict['alpha_head.weight'].shape[0]

                            # Store actual dimensions
                            self.actual_state_dim = original_state_dim
                            self.actual_action_dim = original_action_dim

                            print(f"  Original model: {original_state_dim}D state → {original_action_dim}D action")

                            # Use alpha head for mean action (ignore beta for simplicity)
                            self.actor = nn.Sequential(
                                nn.Linear(original_state_dim, h1),
                                nn.ReLU(),
                                nn.Linear(h1, h2),
                                nn.ReLU(),
                                nn.Linear(h2, original_action_dim),
                                nn.Tanh()
                            ).to(self.device)
                            self.actor[0].load_state_dict({'weight': state_dict['l1.weight'], 'bias': state_dict['l1.bias']})
                            self.actor[2].load_state_dict({'weight': state_dict['l2.weight'], 'bias': state_dict['l2.bias']})
                            self.actor[4].load_state_dict({'weight': state_dict['alpha_head.weight'], 'bias': state_dict['alpha_head.bias']})

                        else:
                            raise ValueError(f"Unknown continuous architecture for {model_name}. Keys: {list(state_dict.keys())}")

                        self.actor.eval()

                    def select_action(self, state):
                        """Select continuous action"""
                        with torch.no_grad():
                            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                            action = self.actor(state_tensor).cpu().numpy()[0]
                        return action

                    def predict(self, state, deterministic=True):
                        action = self.select_action(state)
                        return action, None

                agent = ContinuousActorWrapper(state_dict, model_state_dim, model_action_dim, model_name)
                print(f"✓ Loaded pretrained continuous actor model from {model_path}")

                # Recreate adapters with ACTUAL dimensions from the loaded model
                if hasattr(agent, 'actual_state_dim') and agent.actual_state_dim is not None:
                    actual_state_dim = agent.actual_state_dim
                    actual_action_dim = agent.actual_action_dim

                    print(f"\n⚠️  Recreating adapters with actual model dimensions:")
                    print(f"  Isaac Sim: {isaac_state_dim}D state → {isaac_action_dim} actions")
                    print(f"  Model ACTUALLY expects: {actual_state_dim}D state → {actual_action_dim}D action")

                    # Recreate state adapter with actual dimensions
                    if actual_state_dim != isaac_state_dim:
                        state_adapter = PCAStateAdapter(input_dim=isaac_state_dim, output_dim=actual_state_dim)
                        print(f"  ✓ State adapter: PCA ({isaac_state_dim}D → {actual_state_dim}D)")
                    else:
                        state_adapter = None
                        print(f"  ✓ No state adapter needed (dimensions match)")

                    # Recreate continuous adapter with actual dimensions
                    continuous_adapter = ContinuousToDiscreteAdapter(
                        continuous_dim=actual_action_dim,
                        num_cubes=isaac_action_dim
                    )
                    print(f"  ✓ Continuous adapter: Continuous→Discrete ({actual_action_dim}D → {isaac_action_dim})")

                    # Update model_action_dim for later use
                    model_action_dim = actual_action_dim

                print(f"  Model outputs {model_action_dim}D continuous action")
            except Exception as e:
                print(f"ERROR loading continuous model: {e}")
                import traceback
                traceback.print_exc()
                continue

        else:
            # Load Stable-Baselines3 models (fallback)
            try:
                from stable_baselines3 import PPO, SAC, DDPG, TD3
                if "ppo" in model_name.lower():
                    agent = PPO.load(model_path)
                elif "sac" in model_name.lower():
                    agent = SAC.load(model_path)
                elif "ddpg" in model_name.lower():
                    agent = DDPG.load(model_path)
                elif "td3" in model_name.lower():
                    agent = TD3.load(model_path)
                else:
                    print(f"ERROR: Unknown model type for {model_name}")
                    continue
                print(f"Loaded {model_name} model from {model_path}")
            except Exception as e:
                print(f"ERROR loading Stable-Baselines3 model: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Test the model for N episodes
        model_rewards = []
        model_successes = []
        model_lengths = []
        model_picks = []

        # Reset state adapter if present
        if state_adapter is not None:
            state_adapter.reset()

        for ep in range(args.episodes):
            print(f"\n  Episode {ep+1}/{args.episodes}...")

            # Set random seed for this episode (for reproducibility and diversity)
            episode_seed = 42 + model_idx * 1000 + ep
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)

            # Start metrics tracking
            metrics_collector.start_episode(ep + 1, model_name, seed=episode_seed)

            # ENHANCED STOCHASTICITY: Randomize cube positions AND obstacles for each episode
            if hasattr(env, 'franka_controller') and env.franka_controller is not None:
                # Randomize cube positions (adds spatial variation)
                env.franka_controller.randomize_cube_positions()
                # Regenerate obstacles (1 obstacle for 4x4 grid, random position)
                env.franka_controller._create_random_obstacles()

            state, info = env.reset()

            # Add observation noise for robustness testing (10% Gaussian noise - INCREASED)
            noise_std = 0.10  # Increased from 0.05 to 0.10 for more challenge
            state = state + np.random.normal(0, noise_std, state.shape)

            # Calculate optimal steps (minimum number of cubes to pick)
            optimal_steps = env.num_cubes
            metrics_collector.set_optimal_steps(optimal_steps)

            episode_reward = 0
            episode_length = 0
            done = False
            episode_actions = []

            # Reset state adapter for new episode
            if state_adapter is not None:
                state_adapter.reset()

            while not done:
                # Get action mask (Isaac Sim: 9 actions)
                isaac_action_mask = info.get('action_mask', env.action_masks())

                # Adapt state if needed (24D → 8D, 4D, or 3D)
                if state_adapter is not None:
                    adapted_state = state_adapter.transform(state)
                else:
                    adapted_state = state

                # Add observation noise to adapted state as well
                adapted_state = adapted_state + np.random.normal(0, noise_std, adapted_state.shape)

                # Select action based on model type
                if model_type == "continuous":
                    # Continuous model: outputs continuous action
                    continuous_action = agent.select_action(adapted_state)

                    # Convert continuous to discrete using adapter
                    if continuous_adapter is not None:
                        isaac_action = continuous_adapter.map_action(continuous_action)
                    else:
                        # Fallback: use first dimension and discretize
                        isaac_action = int(np.clip(continuous_action[0], 0, isaac_action_dim - 1))

                    # Ensure action is valid
                    if not isaac_action_mask[isaac_action]:
                        valid_actions = np.where(isaac_action_mask)[0]
                        if len(valid_actions) > 0:
                            isaac_action = valid_actions[0]
                        else:
                            isaac_action = 0

                elif model_type == "ddqn" or model_type == "pytorch":
                    # Discrete model with action mask
                    if action_adapter is not None:
                        # Use all actions in model space
                        model_action_mask = np.ones(model_action_dim, dtype=bool)
                    else:
                        model_action_mask = isaac_action_mask

                    model_action = agent.select_action(adapted_state, model_action_mask)

                    # Adapt action back to Isaac Sim space
                    if action_adapter is not None:
                        # CRITICAL FIX: Pass action mask to adapter to filter invalid actions
                        isaac_action = action_adapter.map_action(model_action, state, action_mask=isaac_action_mask)
                        # Double-check action is valid (safety check)
                        if not isaac_action_mask[isaac_action]:
                            valid_actions = np.where(isaac_action_mask)[0]
                            if len(valid_actions) > 0:
                                isaac_action = valid_actions[0]
                            else:
                                isaac_action = 0
                    else:
                        isaac_action = model_action

                else:
                    # Stable-Baselines3 models
                    model_action, _ = agent.predict(adapted_state, deterministic=True)
                    isaac_action = int(model_action)

                episode_actions.append(int(isaac_action))

                # Take step in Isaac Sim environment
                next_state, reward, terminated, truncated, info = env.step(isaac_action)
                done = terminated or truncated

                # Detect collision (from info dict - now properly tracked by environment)
                collision = info.get('collision', False)

                state = next_state
                # Add observation noise
                state = state + np.random.normal(0, noise_std, state.shape)

                episode_reward += reward
                episode_length += 1

                # Update metrics with action, collision, and info tracking
                # NEW: Pass info dict to track RRT/pick failures
                metrics_collector.update_step(reward, action=isaac_action, collision=collision, info=info)

            # Episode finished
            picks = len(env.objects_picked)
            # UPDATED: Success as percentage (0.0-1.0) based on cubes picked
            success = picks / env.num_cubes  # 8/9 = 0.889 (88.9%)

            model_rewards.append(episode_reward)
            model_successes.append(success)
            model_lengths.append(episode_length)
            model_picks.append(picks)

            # End metrics tracking (binary success for completion tracking)
            # Pass info dict to collect unreachable cubes
            metrics_collector.end_episode(picks == env.num_cubes, picks, env.num_cubes, info=info)

            # Get episode stats for enhanced logging
            episode_stats = metrics_collector.episode_data[-1]
            print(f"    Reward: {episode_reward:.2f}, Success: {success:.1%}, Length: {episode_length}, Picks: {picks}/{env.num_cubes}")
            print(f"    Collisions: {episode_stats['collisions']}, RRT Fails: {episode_stats['rrt_failures']}, "
                  f"Pick Fails: {episode_stats['pick_failures']} (Unattempted: {episode_stats['unattempted_picks']}), Successful: {episode_stats['successful_picks']}")
            print(f"    Unreachable: {episode_stats.get('unreachable_cubes', 0)}")
            print(f"    RRT Path Eff: {episode_stats['path_efficiency']:.3f} (1.0=optimal, <1.0=longer path), "
                  f"Action Entropy: {episode_stats['action_entropy']:.2f}, Unique Actions: {episode_stats['unique_actions']}")

            # Log to W&B (per episode)
            if args.use_wandb:
                import wandb
                wandb.log({
                    f"{model_name}/episode_reward": episode_reward,
                    f"{model_name}/episode_success": success,
                    f"{model_name}/episode_length": episode_length,
                    f"{model_name}/episode_picks": picks,
                    f"{model_name}/episode_collisions": episode_stats['collisions'],
                    f"{model_name}/episode_path_efficiency": episode_stats['path_efficiency'],
                    f"{model_name}/episode_action_entropy": episode_stats['action_entropy'],
                    f"{model_name}/episode_unique_actions": episode_stats['unique_actions'],
                    f"{model_name}/episode_duration": episode_stats['duration'],
                    f"{model_name}/episode": ep + 1,
                    "global_episode": model_idx * args.episodes + ep + 1
                })

        # Calculate statistics
        avg_reward = np.mean(model_rewards)
        std_reward = np.std(model_rewards)
        success_rate = np.mean(model_successes)
        avg_length = np.mean(model_lengths)
        avg_picks = np.mean(model_picks)

        # Get detailed stats from metrics collector
        model_stats = metrics_collector.get_model_stats(model_name)

        print(f"\n  Results for {model_name}:")
        print(f"    Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"    Success Rate: {success_rate:.2%}")
        print(f"    Avg Length: {int(avg_length + 0.5)} steps")  # Round step count (0.5 rounds up)
        print(f"    Avg Picks: {int(avg_picks + 0.5)}/{env.num_cubes}")
        print(f"    Avg Duration: {model_stats.get('avg_duration', 0):.2f}s per episode")
        print(f"    Avg Collisions: {int(model_stats.get('avg_collisions', 0) + 0.5)}")
        print(f"    Avg RRT Failures: {int(model_stats.get('avg_rrt_failures', 0) + 0.5)}")
        print(f"    Avg Pick Failures: {int(model_stats.get('avg_pick_failures', 0) + 0.5)}")
        print(f"    Avg Successful Picks: {int(model_stats.get('avg_successful_picks', 0) + 0.5)}")
        print(f"    Avg Unreachable Cubes: {int(model_stats.get('avg_unreachable_cubes', 0) + 0.5)}")
        print(f"    Avg RRT Path Efficiency: {model_stats.get('avg_path_efficiency', 0):.3f} (1.0=optimal, <1.0=longer path)")
        print(f"    Avg Action Entropy: {model_stats.get('avg_action_entropy', 0):.2f}")
        print(f"    Avg Unique Actions: {int(model_stats.get('avg_unique_actions', 0) + 0.5)}")

        # Store results
        # Apply proper rounding (0.5 rounds up) to count-based metrics for CSV
        all_results.append({
            "model": model_name,
            "model_type": model_type,
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "success_rate": success_rate,
            "avg_length": int(avg_length + 0.5),  # Round count metrics
            "avg_picks": int(avg_picks + 0.5),  # Round count metrics
            "avg_duration": model_stats.get('avg_duration', 0),
            "avg_collisions": int(model_stats.get('avg_collisions', 0) + 0.5),  # Round count metrics
            "avg_rrt_failures": int(model_stats.get('avg_rrt_failures', 0) + 0.5),  # Round count metrics
            "avg_pick_failures": int(model_stats.get('avg_pick_failures', 0) + 0.5),  # Round count metrics
            "avg_successful_picks": int(model_stats.get('avg_successful_picks', 0) + 0.5),  # Round count metrics
            "avg_unreachable_cubes": int(model_stats.get('avg_unreachable_cubes', 0) + 0.5),  # Round count metrics
            "avg_path_efficiency": model_stats.get('avg_path_efficiency', 0),
            "avg_action_entropy": model_stats.get('avg_action_entropy', 0),
            "avg_unique_actions": int(model_stats.get('avg_unique_actions', 0) + 0.5),  # Round count metrics
            "avg_action_diversity": model_stats.get('avg_action_diversity', 0),
            "episodes": args.episodes
        })

        # Store episode-level data for advanced visualizations
        all_episode_data[model_name] = {
            'rewards': model_rewards,
            'successes': model_successes,
            'lengths': model_lengths
        }

        # Enhanced W&B logging with visualizations
        if args.skip_viz:
            print(f"   ⏭️  Skipping visualizations (--skip_viz flag enabled)")

        if args.use_wandb and not args.skip_viz:
            import wandb
            import pandas as pd
            import sys
            import warnings

            # Suppress font and deprecation warnings
            warnings.filterwarnings('ignore', message='.*Linux Libertine.*')
            warnings.filterwarnings('ignore', category=DeprecationWarning)

            # Add visualization module to path
            viz_path = Path(__file__).parent / "visualization"
            if str(viz_path) not in sys.path:
                sys.path.insert(0, str(viz_path))

            # Summary metrics
            wandb.log({
                f"{model_name}/summary/avg_reward": avg_reward,
                f"{model_name}/summary/std_reward": std_reward,
                f"{model_name}/summary/success_rate": success_rate,
                f"{model_name}/summary/avg_length": avg_length,
                f"{model_name}/summary/avg_picks": avg_picks,
                f"{model_name}/summary/avg_duration": model_stats.get('avg_duration', 0),
                f"{model_name}/summary/avg_collisions": model_stats.get('avg_collisions', 0),
                f"{model_name}/summary/avg_path_efficiency": model_stats.get('avg_path_efficiency', 0),
                f"{model_name}/summary/avg_action_entropy": model_stats.get('avg_action_entropy', 0),
                f"{model_name}/summary/avg_unique_actions": model_stats.get('avg_unique_actions', 0),
                f"{model_name}/summary/avg_action_diversity": model_stats.get('avg_action_diversity', 0),
                f"{model_name}/summary/num_successes": int(success_rate * args.episodes),
                f"{model_name}/summary/num_failures": int((1 - success_rate) * args.episodes),
            })

            # Log reward distribution histogram (works in offline mode)
            wandb.log({
                f"{model_name}/reward_distribution": wandb.Histogram(model_rewards)
            })

            # Try to generate visualizations
            try:
                import pandas as pd
                import matplotlib.pyplot as plt
                import warnings

                # Pandas compatibility patch for older Plotly versions
                if not hasattr(pd.DataFrame, 'iteritems'):
                    pd.DataFrame.iteritems = pd.DataFrame.items

                from individual_models.seaborn.success_rate import plot_success_rate_seaborn
                from individual_models.seaborn.reward_distribution import plot_reward_distribution_seaborn
                from individual_models.seaborn.steps_distribution import plot_steps_distribution_seaborn
                from individual_models.seaborn.multi_metric_dashboard import plot_multi_metric_dashboard_seaborn
                from individual_models.seaborn.performance_radar import plot_performance_radar_seaborn

                from individual_models.plotly.success_rate import plot_success_rate_plotly
                from individual_models.plotly.reward_distribution import plot_reward_distribution_plotly
                from individual_models.plotly.steps_distribution import plot_steps_distribution_plotly
                from individual_models.plotly.multi_metric_dashboard import plot_multi_metric_dashboard_plotly
                from individual_models.plotly.performance_radar import plot_performance_radar_plotly

                # Prepare data for visualization functions
                viz_df = pd.DataFrame({
                    'episode': list(range(1, len(model_rewards) + 1)),
                    'reward': model_rewards,
                    'success': model_successes,
                    'success_rate': [s * 100 for s in model_successes],
                    'steps': model_lengths,
                    'model_name': [model_name] * len(model_rewards),
                    'model': [model_name] * len(model_rewards)
                })

                import matplotlib.pyplot as plt

                # Generate and log visualizations
                # 1. Success Rate (Seaborn)
                fig = plot_success_rate_seaborn(viz_df, model_name)
                wandb.log({f"{model_name}/seaborn/success_rate": wandb.Image(fig)})
                plt.close(fig)

                # 2. Success Rate (Plotly)
                fig = plot_success_rate_plotly(viz_df, model_name)
                wandb.log({f"{model_name}/plotly/success_rate": wandb.Plotly(fig)})

                # 3. Reward Distribution (Seaborn)
                fig = plot_reward_distribution_seaborn(viz_df, model_name)
                wandb.log({f"{model_name}/seaborn/reward_distribution": wandb.Image(fig)})
                plt.close(fig)

                # 4. Reward Distribution (Plotly)
                fig = plot_reward_distribution_plotly(viz_df, model_name)
                wandb.log({f"{model_name}/plotly/reward_distribution": wandb.Plotly(fig)})

                # 5. Steps Distribution (Seaborn)
                fig = plot_steps_distribution_seaborn(viz_df, model_name)
                wandb.log({f"{model_name}/seaborn/steps_distribution": wandb.Image(fig)})
                plt.close(fig)

                # 6. Steps Distribution (Plotly)
                fig = plot_steps_distribution_plotly(viz_df, model_name)
                wandb.log({f"{model_name}/plotly/steps_distribution": wandb.Plotly(fig)})

                # 7. Performance Radar (Seaborn)
                fig = plot_performance_radar_seaborn(viz_df, model_name)
                wandb.log({f"{model_name}/seaborn/performance_radar": wandb.Image(fig)})
                plt.close(fig)

                # 8. Performance Radar (Plotly)
                fig = plot_performance_radar_plotly(viz_df, model_name)
                wandb.log({f"{model_name}/plotly/performance_radar": wandb.Plotly(fig)})

                # 9. Multi-Metric Dashboard (Seaborn)
                fig = plot_multi_metric_dashboard_seaborn(viz_df, model_name)
                wandb.log({f"{model_name}/seaborn/multi_metric_dashboard": wandb.Image(fig)})
                plt.close(fig)

                # 10. Multi-Metric Dashboard (Plotly)
                fig = plot_multi_metric_dashboard_plotly(viz_df, model_name)
                wandb.log({f"{model_name}/plotly/multi_metric_dashboard": wandb.Plotly(fig)})

                print(f"   ✅ Logged 10 visualizations (5 Seaborn + 5 Plotly) for {model_name}")

            except Exception as e:
                print(f"   ⚠️  Visualization generation failed: {e}")
                print(f"   Continuing with basic metrics only...")

    # Save summary results to CSV
    with open(results_file, 'w', newline='') as f:
        import csv
        fieldnames = [
            "model", "model_type", "avg_reward", "std_reward", "success_rate",
            "avg_length", "avg_picks", "avg_duration", "avg_collisions",
            "avg_rrt_failures", "avg_pick_failures", "avg_successful_picks",
            "avg_unreachable_cubes",  # NEW
            "avg_path_efficiency", "avg_action_entropy", "avg_unique_actions",
            "avg_action_diversity", "episodes"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Save detailed episode-level data to separate CSV
    episode_results_file = results_dir / "episode_results.csv"
    all_episode_data_flat = metrics_collector.get_all_data()
    if all_episode_data_flat:
        with open(episode_results_file, 'w', newline='') as f:
            import csv
            # Get all keys from first episode
            fieldnames = list(all_episode_data_flat[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_episode_data_flat)
        print(f"Episode-level results saved to: {episode_results_file}")

    print(f"\n{'='*60}")
    print("TESTING COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {results_file}")

    # Helper function to print summary table
    def print_summary_table(results, title):
        """Print a formatted summary table for a set of results"""
        print(f"\n{'='*160}")
        print(f"{title}")
        print(f"{'='*160}")
        print(f"{'Model':<20} {'Success':<9} {'Reward':<9} {'Picks':<7} {'Collis':<8} {'RRTFail':<9} {'PickFail':<9} {'SuccPick':<9} {'Unreach':<8} {'RRTPathEff':<11} {'Entropy':<9}")
        print(f"{'-'*160}")
        for result in results:
            # Count metrics are already integers (rounded when stored in CSV)
            # If reading from old CSV files, they might be floats, so handle both cases
            # Keep decimals for continuous metrics (Success %, Reward, PathEff ratio, Entropy)
            def safe_int(val):
                """Convert to int, handling both int and float inputs"""
                return int(val) if isinstance(val, int) else int(val + 0.5)

            print(f"{result['model']:<20} {result['success_rate']:>7.1%} {result['avg_reward']:>8.2f} "
                  f"{safe_int(result['avg_picks']):>5d} {safe_int(result['avg_collisions']):>7d} "
                  f"{safe_int(result['avg_rrt_failures']):>8d} {safe_int(result['avg_pick_failures']):>8d} "
                  f"{safe_int(result['avg_successful_picks']):>8d} {safe_int(result.get('avg_unreachable_cubes', 0)):>7d} "
                  f"{result['avg_path_efficiency']:>10.3f} {result['avg_action_entropy']:>8.2f}")
        print(f"{'='*160}")

    # Print current experiment summary
    print_summary_table(all_results, f"COMPREHENSIVE METRICS SUMMARY - {exp_config['name']}")

    # If this is exp2 (continuous), also load and print exp1 (discrete) summary
    if args.experiment == "exp2":
        exp1_results_file = Path(r"C:\isaacsim\cobotproject\scripts\experiments\rlmodels\results") / "exp1" / "comparison_results.csv"
        if exp1_results_file.exists():
            try:
                import csv
                with open(exp1_results_file, 'r') as f:
                    reader = csv.DictReader(f)
                    exp1_results = list(reader)
                    # Convert string values to appropriate types
                    # Count metrics are stored as integers, continuous metrics as floats
                    for r in exp1_results:
                        r['success_rate'] = float(r['success_rate'])
                        r['avg_reward'] = float(r['avg_reward'])
                        r['avg_picks'] = int(r['avg_picks'])  # Count metric (integer)
                        r['avg_collisions'] = int(r['avg_collisions'])  # Count metric (integer)
                        r['avg_rrt_failures'] = int(r['avg_rrt_failures'])  # Count metric (integer)
                        r['avg_pick_failures'] = int(r['avg_pick_failures'])  # Count metric (integer)
                        r['avg_successful_picks'] = int(r['avg_successful_picks'])  # Count metric (integer)
                        r['avg_unreachable_cubes'] = int(r.get('avg_unreachable_cubes', 0))  # Count metric (integer)
                        r['avg_path_efficiency'] = float(r['avg_path_efficiency'])
                        r['avg_action_entropy'] = float(r['avg_action_entropy'])

                # Print exp1 summary
                print_summary_table(exp1_results, "COMPREHENSIVE METRICS SUMMARY - Discrete Action Models (EXP1)")

                print(f"\n{'='*160}")
                print("NOTE: Both experiment summaries are shown above for easy comparison")
                print(f"{'='*160}")
            except Exception as e:
                print(f"\n⚠️  Could not load exp1 results for comparison: {e}")

    # Print detailed metrics explanation
    print(f"\nMETRICS LEGEND:")
    print(f"  Success   : % of episodes where all cubes were picked")
    print(f"  Reward    : Average total reward per episode")
    print(f"  Picks     : Average number of cubes picked per episode")
    print(f"  Collis    : Average collision count per episode")
    print(f"  RRTFail   : Average RRT failures (agent selected unreachable cube)")
    print(f"  PickFail  : Average pick failures (RRT succeeded but pick failed)")
    print(f"  SuccPick  : Average successful picks per episode")
    print(f"  Unreach   : Average unreachable cubes (never reachable, episode terminated)")
    print(f"  PathEff   : Path efficiency (actual steps / optimal steps)")
    print(f"  Entropy   : Action entropy (diversity of actions)")
    print(f"{'='*160}\n")

    # Enhanced W&B logging with cross-model comparisons
    if args.use_wandb:
        import wandb

        try:
            # Log summary metrics for each model
            for r in all_results:
                model_prefix = r['model'].replace(' ', '_').replace('-', '_')
                wandb.log({
                    f"final/{model_prefix}/avg_reward": r['avg_reward'],
                    f"final/{model_prefix}/std_reward": r['std_reward'],
                    f"final/{model_prefix}/success_rate": r['success_rate'],
                    f"final/{model_prefix}/avg_length": r['avg_length'],
                    f"final/{model_prefix}/avg_picks": r['avg_picks'],
                    f"final/{model_prefix}/avg_duration": r['avg_duration'],
                    f"final/{model_prefix}/avg_collisions": r['avg_collisions'],
                    f"final/{model_prefix}/avg_path_efficiency": r['avg_path_efficiency'],
                    f"final/{model_prefix}/avg_action_entropy": r['avg_action_entropy'],
                    f"final/{model_prefix}/avg_unique_actions": r['avg_unique_actions'],
                    f"final/{model_prefix}/avg_action_diversity": r['avg_action_diversity'],
                })

            # Generate cross-model comparison visualizations
            print(f"\n{'='*60}")
            print("GENERATING CROSS-MODEL VISUALIZATIONS")
            print("12 Seaborn + 11 Plotly + 4 Advanced = 27 Total Charts")
            print(f"{'='*60}")

            try:
                # Import cross-model visualization functions
                from cross_model.seaborn.grouped_bar_with_errors import (
                    plot_grouped_bars_with_ci_seaborn,
                    plot_multi_metric_grouped_bars
                )
                from cross_model.seaborn.pairplot_matrix import (
                    plot_cross_model_pairplot_seaborn,
                    plot_cross_model_corner_pairplot_seaborn
                )
                from cross_model.seaborn.model_comparison import plot_parallel_coordinates_seaborn
                from cross_model.seaborn.reward_comparison import plot_reward_box_seaborn
                from cross_model.seaborn.ranking_table import (
                    plot_distribution_histograms_seaborn,
                    plot_distribution_kde_seaborn
                )
                from cross_model.seaborn.performance_line_chart import plot_performance_line_seaborn
                from cross_model.seaborn.success_rate_comparison import plot_success_rate_line_seaborn
                from cross_model.seaborn.success_vs_inference_scatter import plot_success_vs_inference_seaborn

                from cross_model.plotly.grouped_bar_with_errors import (
                    plot_grouped_bars_with_ci_plotly,
                    plot_multi_metric_grouped_bars_plotly
                )
                from cross_model.plotly.pairplot_matrix import (
                    plot_cross_model_pairplot_plotly,
                    plot_cross_model_3d_scatter_plotly
                )
                from cross_model.plotly.interactive_comparison import plot_parallel_coordinates_plotly
                from cross_model.plotly.ranking_table import plot_distribution_distplot_plotly
                from cross_model.plotly.performance_line_chart import plot_performance_line_plotly
                from cross_model.plotly.reward_comparison import plot_reward_box_plotly
                from cross_model.plotly.success_rate_comparison import plot_success_rate_line_plotly
                from cross_model.plotly.success_vs_inference_scatter import plot_success_vs_inference_plotly

                # Advanced cross-model visualizations
                from cross_model.plotly.box_plot_with_points import plot_box_with_points_plotly
                from cross_model.plotly.histogram_with_kde import plot_histogram_with_kde_plotly
                from cross_model.plotly.timeseries_with_ci import plot_timeseries_with_ci_plotly
                from cross_model.seaborn.timeseries_with_ci import plot_timeseries_with_ci_seaborn

                # Prepare cross-model comparison data
                cross_df = pd.DataFrame(all_results)

                # SEABORN VISUALIZATIONS
                print("\n   Seaborn Visualizations:")

                # 1. grouped_bar_with_errors.py - Success Rate
                try:
                    fig = plot_grouped_bars_with_ci_seaborn(cross_df, x='model', y='success_rate')
                    wandb.log({"cross_model/seaborn/grouped_bar_success_rate": wandb.Image(fig)})
                    plt.close(fig)
                    print("   ✓ Success rate (bar chart with CI)")
                except Exception as e:
                    print(f"   ⚠️  Success rate bar chart failed: {str(e)[:80]}")

                # 2. grouped_bar_with_errors.py - Reward
                try:
                    fig = plot_grouped_bars_with_ci_seaborn(cross_df, x='model', y='avg_reward')
                    wandb.log({"cross_model/seaborn/grouped_bar_reward": wandb.Image(fig)})
                    plt.close(fig)
                    print("   ✓ Reward (bar chart with CI)")
                except Exception as e:
                    print(f"   ⚠️  Reward bar chart failed: {str(e)[:80]}")

                # 3. grouped_bar_with_errors.py - Multi-Metric
                try:
                    fig = plot_multi_metric_grouped_bars(cross_df, metrics=['success_rate', 'avg_reward', 'avg_length'])
                    wandb.log({"cross_model/seaborn/multi_metric_bars": wandb.Image(fig)})
                    plt.close(fig)
                    print("   ✓ Multi-metric (grouped bar charts)")
                except Exception as e:
                    print(f"   ⚠️  Multi-metric bar charts failed: {str(e)[:80]}")



                # 4. pairplot_matrix.py - Full Scatter Matrix
                try:
                    if len(cross_df) >= 3 and cross_df['avg_reward'].std() > 1e-6:
                        fig = plot_cross_model_pairplot_seaborn(cross_df, metrics=['success_rate', 'avg_reward', 'avg_length'], hue='model')
                        wandb.log({"cross_model/seaborn/pairplot_full": wandb.Image(fig)})
                        plt.close(fig)
                        print("   ✓ Pairplot - full scatter matrix")
                    else:
                        print("   ⊘ Pairplot - full scatter matrix (insufficient variance)")
                except Exception as e:
                    print(f"   ⚠️  Pairplot full failed: {str(e)[:80]}")

                # 5. pairplot_matrix.py - Corner (Lower Triangle)
                try:
                    if len(cross_df) >= 3 and cross_df['avg_reward'].std() > 1e-6:
                        fig = plot_cross_model_corner_pairplot_seaborn(cross_df, metrics=['success_rate', 'avg_reward', 'avg_length'], hue='model')
                        wandb.log({"cross_model/seaborn/pairplot_corner": wandb.Image(fig)})
                        plt.close(fig)
                        print("   ✓ Pairplot - corner (lower triangle)")
                    else:
                        print("   ⊘ Pairplot - corner (insufficient variance)")
                except Exception as e:
                    print(f"   ⚠️  Pairplot corner failed: {str(e)[:80]}")

                # 6. reward_comparison.py - Box Plot
                try:
                    fig = plot_reward_box_seaborn(cross_df)
                    wandb.log({"cross_model/seaborn/reward_box": wandb.Image(fig)})
                    plt.close(fig)
                    print("   ✓ Reward distribution (box plot)")
                except Exception as e:
                    print(f"   ⚠️  Reward box plot failed: {str(e)[:80]}")

                # 7. model_comparison.py - Parallel Coordinates
                try:
                    fig = plot_parallel_coordinates_seaborn(cross_df)
                    wandb.log({"cross_model/seaborn/parallel_coordinates": wandb.Image(fig)})
                    plt.close(fig)
                    print("   ✓ Parallel coordinates")
                except Exception as e:
                    print(f"   ⚠️  Parallel coordinates failed: {str(e)[:80]}")

                # 8. ranking_table.py - Distribution Histograms
                try:
                    fig = plot_distribution_histograms_seaborn(cross_df)
                    wandb.log({"cross_model/seaborn/distribution_histograms": wandb.Image(fig)})
                    plt.close(fig)
                    print("   ✓ Distribution histograms (stacked)")
                except Exception as e:
                    print(f"   ⚠️  Distribution histograms failed: {str(e)[:80]}")

                # 9. ranking_table.py - Distribution KDE
                try:
                    fig = plot_distribution_kde_seaborn(cross_df)
                    wandb.log({"cross_model/seaborn/distribution_kde": wandb.Image(fig)})
                    plt.close(fig)
                    print("   ✓ Distribution KDE plots")
                except Exception as e:
                    print(f"   ⚠️  Distribution KDE failed: {str(e)[:80]}")

                # 10. performance_line_chart.py - Faceted Relplot
                try:
                    fig = plot_performance_line_seaborn(cross_df)
                    wandb.log({"cross_model/seaborn/performance_line": wandb.Image(fig)})
                    plt.close(fig)
                    print("   ✓ Performance line chart (faceted relplot)")
                except Exception as e:
                    print(f"   ⚠️  Performance line chart failed: {str(e)[:80]}")

                # 11. success_rate_comparison.py - Line Chart with CI
                try:
                    fig = plot_success_rate_line_seaborn(cross_df)
                    wandb.log({"cross_model/seaborn/success_rate_line": wandb.Image(fig)})
                    plt.close(fig)
                    print("   ✓ Success rate over time (line chart with CI)")
                except Exception as e:
                    print(f"   ⚠️  Success rate line chart failed: {str(e)[:80]}")

                # 12. success_vs_inference_scatter.py - Scatter Plot
                try:
                    fig = plot_success_vs_inference_seaborn(cross_df)
                    wandb.log({"cross_model/seaborn/success_vs_inference": wandb.Image(fig)})
                    plt.close(fig)
                    print("   ✓ Success vs inference (scatter)")
                except Exception as e:
                    print(f"   ⚠️  Success vs inference scatter failed: {str(e)[:80]}")

                # PLOTLY VISUALIZATIONS
                print("\n   Plotly Visualizations:")

                # 1. grouped_bar_with_errors.py - Success Rate
                try:
                    fig = plot_grouped_bars_with_ci_plotly(cross_df, x='model', y='success_rate')
                    wandb.log({"cross_model/plotly/grouped_bar_success_rate": wandb.Plotly(fig)})
                    print("   ✓ Success rate (interactive bar chart with CI)")
                except Exception as e:
                    print(f"   ⚠️  Success rate bar chart failed: {str(e)[:80]}")

                # 2. grouped_bar_with_errors.py - Reward
                try:
                    fig = plot_grouped_bars_with_ci_plotly(cross_df, x='model', y='avg_reward')
                    wandb.log({"cross_model/plotly/grouped_bar_reward": wandb.Plotly(fig)})
                    print("   ✓ Reward (interactive bar chart with CI)")
                except Exception as e:
                    print(f"   ⚠️  Reward bar chart failed: {str(e)[:80]}")

                # 3. grouped_bar_with_errors.py - Multi-Metric
                try:
                    fig = plot_multi_metric_grouped_bars_plotly(cross_df, metrics=['success_rate', 'avg_reward', 'avg_length'])
                    wandb.log({"cross_model/plotly/multi_metric_bars": wandb.Plotly(fig)})
                    print("   ✓ Multi-metric (interactive grouped bar charts)")
                except Exception as e:
                    print(f"   ⚠️  Multi-metric bar charts failed: {str(e)[:80]}")

                # 4. pairplot_matrix.py - Interactive Scatter Matrix
                try:
                    if len(cross_df) >= 3 and cross_df['avg_reward'].std() > 1e-6:
                        fig = plot_cross_model_pairplot_plotly(cross_df, metrics=['success_rate', 'avg_reward', 'avg_length'], color='model')
                        wandb.log({"cross_model/plotly/pairplot_interactive": wandb.Plotly(fig)})
                        print("   ✓ Pairplot - interactive scatter matrix")
                    else:
                        print("   ⊘ Pairplot - interactive scatter matrix (insufficient variance)")
                except Exception as e:
                    print(f"   ⚠️  Pairplot interactive failed: {str(e)[:80]}")

                # 5. pairplot_matrix.py - 3D Scatter
                try:
                    fig = plot_cross_model_3d_scatter_plotly(cross_df, metrics=['success_rate', 'avg_reward', 'avg_length'], color='model')
                    wandb.log({"cross_model/plotly/3d_scatter": wandb.Plotly(fig)})
                    print("   ✓ 3D scatter plot")
                except Exception as e:
                    print(f"   ⚠️  3D scatter plot failed: {str(e)[:80]}")

                # 6. interactive_comparison.py - Parallel Coordinates
                try:
                    fig = plot_parallel_coordinates_plotly(cross_df)
                    wandb.log({"cross_model/plotly/parallel_coordinates": wandb.Plotly(fig)})
                    print("   ✓ Parallel coordinates (interactive)")
                except Exception as e:
                    print(f"   ⚠️  Parallel coordinates failed: {str(e)[:80]}")

                # 7. ranking_table.py - Distribution Distplot
                try:
                    fig = plot_distribution_distplot_plotly(cross_df)
                    wandb.log({"cross_model/plotly/distribution_distplot": wandb.Plotly(fig)})
                    print("   ✓ Distribution distplot (histogram + KDE)")
                except Exception as e:
                    print(f"   ⚠️  Distribution distplot failed: {str(e)[:80]}")

                # 8. performance_line_chart.py - Interactive Faceted
                try:
                    fig = plot_performance_line_plotly(cross_df)
                    wandb.log({"cross_model/plotly/performance_line": wandb.Plotly(fig)})
                    print("   ✓ Performance line chart (interactive faceted)")
                except Exception as e:
                    print(f"   ⚠️  Performance line chart failed: {str(e)[:80]}")

                # 9. reward_comparison.py - Interactive Box Plot
                try:
                    fig = plot_reward_box_plotly(cross_df)
                    wandb.log({"cross_model/plotly/reward_box": wandb.Plotly(fig)})
                    print("   ✓ Reward distribution (interactive box plot)")
                except Exception as e:
                    print(f"   ⚠️  Reward box plot failed: {str(e)[:80]}")

                # 10. success_rate_comparison.py - Interactive Line Chart with CI
                try:
                    fig = plot_success_rate_line_plotly(cross_df)
                    wandb.log({"cross_model/plotly/success_rate_line": wandb.Plotly(fig)})
                    print("   ✓ Success rate over time (interactive line chart with CI)")
                except Exception as e:
                    print(f"   ⚠️  Success rate line chart failed: {str(e)[:80]}")

                # 11. success_vs_inference_scatter.py - Interactive Scatter
                try:
                    fig = plot_success_vs_inference_plotly(cross_df)
                    wandb.log({"cross_model/plotly/success_vs_inference": wandb.Plotly(fig)})
                    print("   ✓ Success vs inference (interactive scatter)")
                except Exception as e:
                    print(f"   ⚠️  Success vs inference scatter failed: {str(e)[:80]}")

                # ADVANCED VISUALIZATIONS
                print("\n   Advanced Visualizations:")

                # 1. timeseries_with_ci.py (Seaborn) - Episode-Level Time Series
                try:
                    # Prepare episode-level data for time series
                    episode_data = []
                    for model_name, results in all_episode_data.items():
                        for ep_idx, (reward, success, length) in enumerate(zip(
                            results['rewards'], results['successes'], results['lengths']
                        )):
                            episode_data.append({
                                'model': model_name,
                                'episode': ep_idx + 1,
                                'reward': reward,
                                'success': success,
                                'steps': length
                            })
                    episode_df = pd.DataFrame(episode_data)

                    fig = plot_timeseries_with_ci_seaborn(episode_df, x='episode', y='reward', hue='model', ci=95)
                    wandb.log({"cross_model/advanced/timeseries_ci_seaborn": wandb.Image(fig)})
                    plt.close(fig)
                    print("   ✓ Time series with CI (Seaborn)")
                except Exception as e:
                    print(f"   ⚠️  Time series CI (Seaborn) failed: {str(e)[:80]}")

                # 2. timeseries_with_ci.py (Plotly) - Interactive Episode-Level Time Series
                try:
                    fig = plot_timeseries_with_ci_plotly(episode_df, x='episode', y='reward', hue='model')
                    wandb.log({"cross_model/advanced/timeseries_ci_plotly": wandb.Plotly(fig)})
                    print("   ✓ Time series with CI (Plotly)")
                except Exception as e:
                    print(f"   ⚠️  Time series CI (Plotly) failed: {str(e)[:80]}")

                # 3. box_plot_with_points.py - Box Plot with All Data Points
                try:
                    fig = plot_box_with_points_plotly(episode_df, metric='reward', group_by='model')
                    wandb.log({"cross_model/advanced/box_with_points": wandb.Plotly(fig)})
                    print("   ✓ Box plot with all data points (Plotly)")
                except Exception as e:
                    print(f"   ⚠️  Box plot with points failed: {str(e)[:80]}")

                # 4. histogram_with_kde.py - Histogram with KDE Overlay
                try:
                    fig = plot_histogram_with_kde_plotly(episode_df, metric='reward', group_by='model')
                    wandb.log({"cross_model/advanced/histogram_kde": wandb.Plotly(fig)})
                    print("   ✓ Histogram with KDE (Plotly)")
                except Exception as e:
                    print(f"   ⚠️  Histogram with KDE failed: {str(e)[:80]}")

                print(f"\n{'='*60}\n")

            except Exception as e:
                print(f"   ⚠️  Cross-model visualization generation failed: {str(e)[:100]}")

            print(f"\n✅ W&B data saved locally with all visualizations")
            print(f"   Run: {wandb.run.name}")
            print(f"   Sync to dashboard with: wandb sync {wandb.run.dir}")

        except Exception as e:
            print(f"⚠️  W&B final logging failed (continuing anyway): {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                wandb.finish()
            except Exception:
                pass

    simulation_app.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
