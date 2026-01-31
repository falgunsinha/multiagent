"""
Test Two-Agent System (Agent 1: Pick Sequence + Agent 2: Reshuffling)
Tests multiple model combinations with comprehensive logging.

Usage:
    # Discrete models with 2 seeds
    C:\isaacsim\python.bat test_two_agent_system.py --action_space discrete --seeds 42 123 --episodes 5
    
    # Continuous models with 2 seeds
    C:\isaacsim\python.bat test_two_agent_system.py --action_space continuous --seeds 42 123 --episodes 5
"""

import sys
from pathlib import Path

# Add project root to path (absolute path for reliability)
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add MASAC and MAPPO to path (for agents/ and envs/ imports)
masac_path = Path(r"C:\isaacsim\cobotproject\scripts\Reinforcement Learning\MASAC")
if str(masac_path) not in sys.path:
    sys.path.insert(0, str(masac_path))
mappo_path = Path(r"C:\isaacsim\cobotproject\scripts\Reinforcement Learning\MAPPO")
if str(mappo_path) not in sys.path:
    sys.path.insert(0, str(mappo_path))

# Add adapters path (for pretrained model adapters)
adapters_path = Path(r"C:\isaacsim\cobotproject\scripts\experiments\rlmodels")
if str(adapters_path) not in sys.path:
    sys.path.insert(0, str(adapters_path))

# Import Isaac Sim components (must be before other imports)
from isaacsim import SimulationApp
import argparse

# Parse arguments BEFORE creating SimulationApp
parser = argparse.ArgumentParser(description="Test Two-Agent System")
parser.add_argument("--action_space", type=str, default="both", choices=["discrete", "continuous", "both"],
                   help="Action space type: discrete, continuous, or both (default: both)")
parser.add_argument("--seeds", type=int, nargs='+', default=[42, 123],
                   help="Random seeds for testing (default: 42 123)")
parser.add_argument("--episodes", type=int, default=5,
                   help="Number of episodes per model (default: 5)")
parser.add_argument("--grid_size", type=int, default=4,
                   help="Grid size (default: 4)")
parser.add_argument("--num_cubes", type=int, default=9,
                   help="Number of cubes (default: 9)")
parser.add_argument("--headless", action="store_true",
                   help="Run in headless mode")
args = parser.parse_args()

# Create SimulationApp
simulation_app = SimulationApp({"headless": True})  # Always headless for batch testing

# Now import everything else
import numpy as np
import json
import csv
import time
from datetime import datetime
import os
import yaml

# Import Isaac Sim modules
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot_motion.motion_generation import PathPlannerVisualizer, ArticulationKinematicsSolver, LulaKinematicsSolver
from isaacsim.robot_motion.motion_generation.lula import RRT
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper
from pxr import UsdGeom, UsdPhysics

# Import RL components
from src.rl.doubleDQN import DoubleDQNAgent
from src.rl.object_selection_env_rrt import ObjectSelectionEnvRRT
from agents.masac_continuous_wrapper import MASACContinuousWrapper
from envs.two_agent_env import TwoAgentEnv

# Import our new components
from two_agent_logger import TwoAgentLogger
from heuristic_agents import HeuristicAgent1, HeuristicAgent2

# Import adapters for non-custom models
try:
    from adapters import (
        FeatureAggregationAdapter,
        PCAStateAdapter,
        DiscreteActionMapper,
        ContinuousToDiscreteAdapter
    )
    ADAPTERS_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Adapters not available: {e}")
    ADAPTERS_AVAILABLE = False


class FrankaRRTTrainer:
    """
    Franka controller for RRT-based testing (reachability checks only, no execution).
    Adapted from MASAC test_masac_grid4_cubes9_isaacsim.py - EXACT COPY for consistency.
    """

    def __init__(self, num_cubes=9, training_grid_size=4):
        self.num_cubes = num_cubes
        self.training_grid_size = training_grid_size
        self.world = None
        self.franka = None
        self.gripper = None
        self.rrt = None
        self.path_planner_visualizer = None
        self.kinematics_solver = None
        self.articulation_kinematics_solver = None
        self.container = None
        self.cubes = []
        self.cube_positions = []
        self.obstacle_prims = []

        print(f"[TRAINER] Initializing Franka RRT Trainer for MASAC Testing")
        print(f"[TRAINER] Grid: {training_grid_size}x{training_grid_size}, Cubes: {num_cubes}")

    def setup_scene(self):
        """Setup Isaac Sim scene with Franka and cubes"""
        print("[TRAINER] Setting up scene...")

        # Create world
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # Add Franka robot
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
            # Go up 4 levels: mutliagent -> rlmodels -> experiments -> scripts -> cobotproject
            project_root = os.path.join(script_dir, "..", "..", "..", "..")
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
        """Setup container for cubes"""
        container_prim_path = "/World/Container"
        container_usd_path = f"{get_assets_root_path()}/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Storage/Containers/Container_I/Container_I04_160x120x64cm_PR_V_NVD_01.usd"
        add_reference_to_stage(usd_path=container_usd_path, prim_path=container_prim_path)

        container_position = np.array([0.45, -0.10, 0.0])  # MATCH MASAC
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

        print(f"[TRAINER] Container added at {container_position}")

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
                # Remove from RRT planner first
                if self.rrt is not None:
                    self.rrt.remove_obstacle(obs_prim)

                # Then remove from scene
                if obs_prim.is_valid():
                    self.world.scene.remove_object(obs_prim.name)
            except Exception as e:
                pass

        self.obstacle_prims = []

        # Update RRT world after removing obstacles
        if self.rrt is not None:
            self.rrt.update_world()

    def _create_random_obstacles(self):
        """Create random obstacles in empty cells (MATCHES test_masac_grid4_cubes9_isaacsim.py)"""
        # For 4x4 grid, create 1 obstacle
        num_obstacles = 1 if self.training_grid_size == 4 else 2

        cube_spacing = 0.13 if self.training_grid_size > 3 else 0.15
        grid_center_x = 0.45
        grid_center_y = -0.10

        grid_extent_x = (self.training_grid_size - 1) * cube_spacing
        grid_extent_y = (self.training_grid_size - 1) * cube_spacing
        start_x = grid_center_x - (grid_extent_x / 2.0)
        start_y = grid_center_y - (grid_extent_y / 2.0)

        # Find occupied cells
        occupied_cells = set()
        for row in range(self.training_grid_size):
            for col in range(self.training_grid_size):
                cell_x = start_x + (row * cube_spacing)
                cell_y = start_y + (col * cube_spacing)
                for cube_pos in self.cube_positions:
                    if np.linalg.norm(cube_pos[:2] - np.array([cell_x, cell_y])) < 0.05:
                        occupied_cells.add((row, col))
                        break

        # Find empty cells
        empty_cells = []
        for row in range(self.training_grid_size):
            for col in range(self.training_grid_size):
                if (row, col) not in occupied_cells:
                    empty_cells.append((row, col))

        # Select random empty cells for obstacles
        if len(empty_cells) < num_obstacles:
            num_obstacles = len(empty_cells)

        selected_cells = np.random.choice(len(empty_cells), size=num_obstacles, replace=False)

        # Create obstacles
        for idx in selected_cells:
            row, col = empty_cells[idx]
            obs_x = start_x + (row * cube_spacing)
            obs_y = start_y + (col * cube_spacing)
            obs_position = np.array([obs_x, obs_y, 0.05])  # Match MASAC height

            obs_name = f"Obstacle_{len(self.obstacle_prims) + 1}"
            obstacle = self.world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/{obs_name}",
                    name=obs_name,
                    position=obs_position,
                    size=0.08,
                    color=np.array([0.8, 0.2, 0.2])
                )
            )
            self.obstacle_prims.append(obstacle)

        # Update RRT world to include obstacles (MATCHES MASAC - no add_obstacle calls)
        if self.rrt:
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
        cube_spacing = 0.13 if self.training_grid_size > 3 else 0.15
        grid_center_x = 0.45
        grid_center_y = -0.10
        cube_size = 0.0515

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


# Load model configurations from YAML files
def load_model_configs_from_yaml():
    """Load discrete and continuous model configs from YAML files"""
    configs_dir = project_root / "scripts/experiments/rlmodels/configs"

    # Load discrete models config
    discrete_config_path = configs_dir / "exp1_discrete_comparison.yaml"
    continuous_config_path = configs_dir / "exp2_continuous_comparison.yaml"

    discrete_models = []
    continuous_models = []

    # Load discrete models
    if discrete_config_path.exists():
        with open(discrete_config_path, 'r') as f:
            discrete_config = yaml.safe_load(f)
            for model_cfg in discrete_config.get('models', []):
                # Skip custom DDQN (we add it manually later)
                if model_cfg.get('custom', False):
                    continue

                # Only add models with state_dim and action_dim
                if 'state_dim' not in model_cfg or 'action_dim' not in model_cfg:
                    print(f"⚠️  Skipping {model_cfg['name']} - missing state_dim or action_dim")
                    continue

                discrete_models.append({
                    "name": f"{model_cfg['name']}+MASAC",
                    "agent1_type": model_cfg['type'],
                    "agent1_path": model_cfg['path'],
                    "agent1_state_dim": model_cfg['state_dim'],
                    "agent1_action_dim": model_cfg['action_dim'],
                    "agent2_type": "masac",
                    "agent2_path": None
                })
    else:
        print(f"⚠️  Discrete config not found: {discrete_config_path}")

    # Load continuous models
    if continuous_config_path.exists():
        with open(continuous_config_path, 'r') as f:
            continuous_config = yaml.safe_load(f)
            for model_cfg in continuous_config.get('models', []):
                # Skip custom models (we add DDQN manually in discrete)
                if model_cfg.get('custom', False):
                    continue

                # Skip discrete models in continuous config
                if model_cfg.get('type') in ['ddqn'] or 'discrete' in model_cfg.get('name', '').lower():
                    continue

                # Only add models with state_dim and action_dim
                if 'state_dim' not in model_cfg or 'action_dim' not in model_cfg:
                    print(f"⚠️  Skipping {model_cfg['name']} - missing state_dim or action_dim")
                    continue

                continuous_models.append({
                    "name": f"{model_cfg['name']}+MASAC",
                    "agent1_type": model_cfg['type'],
                    "agent1_path": model_cfg['path'],
                    "agent1_state_dim": model_cfg['state_dim'],
                    "agent1_action_dim": model_cfg['action_dim'],
                    "agent2_type": "masac",
                    "agent2_path": None
                })
    else:
        print(f"⚠️  Continuous config not found: {continuous_config_path}")

    print(f"\n✅ Loaded {len(discrete_models)} discrete models from YAML")
    print(f"✅ Loaded {len(continuous_models)} continuous models from YAML")

    return discrete_models, continuous_models


# Load pretrained models from YAML configs
_discrete_from_yaml, _continuous_from_yaml = load_model_configs_from_yaml()

# Model configurations
# NOTE: Heuristic and Custom DDQN are added manually, rest loaded from YAML
DISCRETE_MODELS = [
    # Heuristic + Heuristic (Test first - simplest baseline)
    {
        "name": "Heuristic",
        "agent1_type": "heuristic",
        "agent1_path": None,
        "agent2_type": "heuristic",
        "agent2_path": None
    },
] + _discrete_from_yaml + [
    # Custom-DDQN + MASAC (Test last - most important, our trained model)
    # MATCHES test_masac_grid4_cubes9_isaacsim.py
    {
        "name": "DDQN+MASAC",
        "agent1_type": "custom_ddqn",
        "agent1_path": str(project_root / "scripts/Reinforcement Learning/doubleDQN_script/models/ddqn_rrt_isaacsim_grid4_cubes9_20251224_185752_final.pt"),
        "agent2_type": "masac",
        "agent2_path": None  # Uses pretrained MASAC
    }
]

# Continuous models loaded from YAML config
CONTINUOUS_MODELS = _continuous_from_yaml


def create_isaacsim_environment(grid_size: int, num_cubes: int, max_steps: int = 50):
    """
    Create Isaac Sim environment with Franka robot and RRT planner.
    Uses FrankaRRTTrainer for comprehensive scene setup with cubes, obstacles, and RRT.
    """
    print("[ENV] Creating Isaac Sim environment...")
    print(f"[ENV] Grid: {grid_size}x{grid_size}, Cubes: {num_cubes}")

    # Create trainer (handles all scene setup)
    trainer = FrankaRRTTrainer(
        num_cubes=num_cubes,
        training_grid_size=grid_size
    )

    # Setup scene
    trainer.setup_scene()

    # Create RL environment
    print("[ENV] Creating RRT-based RL environment...")
    max_objects = grid_size * grid_size

    env = ObjectSelectionEnvRRT(
        franka_controller=trainer,
        max_objects=max_objects,
        max_steps=max_steps,
        num_cubes=num_cubes,
        training_grid_size=grid_size,
        execute_picks=False,  # Planning only (no execution) - faster and more reliable
        rrt_planner=trainer.rrt,
        kinematics_solver=trainer.kinematics_solver,
        articulation_kinematics_solver=trainer.articulation_kinematics_solver,
        franka_articulation=trainer.franka
    )

    print("[ENV] Isaac Sim environment created successfully")
    return env, trainer.world


def load_agent1(model_config: dict, base_env, grid_size: int, num_cubes: int):
    """
    Load Agent 1 (Pick Sequence) based on model configuration

    Returns:
        agent: Loaded agent
    """
    agent_type = model_config["agent1_type"]
    agent_path = model_config.get("agent1_path")

    if agent_type == "custom_ddqn":
        # Load custom DDQN agent
        from src.rl.doubleDQN import DoubleDQNAgent

        state_dim = base_env.observation_space.shape[0]
        action_dim = base_env.action_space.n

        agent = DoubleDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=0.0,  # No exploration during testing
            epsilon_end=0.0,
            epsilon_decay=1.0,
            buffer_capacity=10000,
            batch_size=64
        )

        # Load weights
        agent.load(agent_path)
        print(f"✅ Loaded Custom DDQN from: {agent_path}")
        return agent

    elif agent_type == "heuristic":
        # Create heuristic agent
        state_dim = base_env.observation_space.shape[0]
        action_dim = base_env.action_space.n

        agent = HeuristicAgent1(state_dim=state_dim, action_dim=action_dim, env=base_env)
        print(f"✅ Created Heuristic Agent 1")
        return agent

    elif agent_type in ["ddqn", "pytorch"]:
        # Load pretrained model with adapters
        if not ADAPTERS_AVAILABLE:
            raise RuntimeError("Adapters not available for pretrained models")

        import torch
        import torch.nn as nn

        # Get model configuration
        model_path = model_config["agent1_path"]
        model_state_dim = model_config["agent1_state_dim"]
        model_action_dim = model_config["agent1_action_dim"]

        # Get Isaac Sim environment dimensions
        isaac_state_dim = base_env.observation_space.shape[0]
        isaac_action_dim = base_env.action_space.n

        print(f"  Isaac Sim: {isaac_state_dim}D state → {isaac_action_dim} actions")
        print(f"  Model expects: {model_state_dim}D state → {model_action_dim} actions")

        # Create state adapter (PCA to reduce dimensions)
        state_adapter = PCAStateAdapter(input_dim=isaac_state_dim, output_dim=model_state_dim)
        print(f"  ✓ State adapter: PCA ({isaac_state_dim}D → {model_state_dim}D)")

        # Create action adapter
        # Check if model is discrete or continuous based on path
        model_name = model_config.get("name", "")
        is_discrete_model = "Discrete" in model_name or agent_type == "ddqn"

        if is_discrete_model:
            # Discrete models (DDQN, SAC-Discrete, PPO-Discrete): discrete action space
            action_adapter = DiscreteActionMapper(source_actions=model_action_dim, target_actions=isaac_action_dim)
            print(f"  ✓ Action adapter: Discrete Mapper ({model_action_dim} → {isaac_action_dim})")
        else:
            # Continuous models (DDPG, TD3, SAC-Continuous, PPO-Continuous): continuous action space
            # NOTE: We use default grid for continuous adapter because cube positions
            # are not initialized until environment reset, which happens later.
            # The adapter will map continuous actions to the nearest cube in a normalized grid.
            # IMPORTANT: Use num_cubes (9), NOT isaac_action_dim (16)!
            action_adapter = ContinuousToDiscreteAdapter(
                continuous_dim=model_action_dim,
                num_cubes=num_cubes,  # Use actual number of cubes (9), not grid size (16)
                cube_positions=None  # Use default normalized grid
            )
            print(f"  ✓ Action adapter: Continuous→Discrete ({model_action_dim}D → {num_cubes} cubes) using normalized grid")

        # Load pretrained model
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

        # Create model architecture based on type
        if agent_type == "ddqn":
            # DDQN variants: Dynamically detect architecture from state_dict
            print(f"  Detecting DDQN architecture from state_dict...")
            print(f"  State dict keys: {list(state_dict.keys())[:10]}")  # Show first 10 keys

            # Create wrapper agent with dynamic architecture detection
            class DDQNAdapterAgent:
                def __init__(self, state_dict, state_dim, action_dim, state_adapter, action_adapter):
                    self.device = torch.device('cpu')
                    self.state_adapter = state_adapter
                    self.action_adapter = action_adapter

                    # Determine architecture from state_dict keys
                    if 'hidden.0.weight' in state_dict:
                        # Dueling DQN architecture
                        print(f"  Detected Dueling DQN architecture")
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
                        print(f"  Detected Standard DQN architecture")
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
                        print(f"  Detected C51 DQN architecture")
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
                        raise ValueError(f"Unknown DDQN architecture. Keys: {list(state_dict.keys())}")

                    self.network.eval()
                    if hasattr(self, 'value_stream'):
                        self.value_stream.eval()
                        self.advantage_stream.eval()

                def select_action(self, state, action_mask=None):
                    # Adapt state (flatten first, then transform returns 1D)
                    adapted_state = self.state_adapter.transform(state.flatten())

                    # Get Q-values
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(adapted_state).unsqueeze(0).to(self.device)

                        if self.is_dueling:
                            features = self.network(state_tensor)
                            value = self.value_stream(features)
                            advantage = self.advantage_stream(features)
                            # Handle both 1D and 2D tensors - ensure batch dimension
                            if advantage.dim() == 1:
                                advantage = advantage.unsqueeze(0)
                                value = value.unsqueeze(0)
                            # Now compute Q-values with proper dimensions
                            q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
                        elif hasattr(self, 'is_c51') and self.is_c51:
                            logits = self.network(state_tensor)
                            logits = logits.view(-1, self.num_atoms)
                            probs = torch.softmax(logits, dim=1)
                            q_values = probs.mean(dim=1).unsqueeze(0)
                        else:
                            q_values = self.network(state_tensor)

                    # Select best action
                    model_action = q_values.argmax().item()

                    # Adapt action (pass action_mask to prevent out-of-bounds)
                    isaac_action = self.action_adapter.map_action(model_action, action_mask=action_mask)
                    return isaac_action

            agent = DDQNAdapterAgent(state_dict, model_state_dim, model_action_dim, state_adapter, action_adapter)
            print(f"✅ Loaded DDQN variant with adapters")
            return agent

        else:  # pytorch (SAC, PPO, DDPG, TD3)
            # PyTorch actor models: Dynamically detect architecture from state_dict
            print(f"  Detecting PyTorch actor architecture from state_dict...")
            print(f"  State dict keys: {list(state_dict.keys())[:10]}")  # Show first 10 keys

            # Create wrapper agent with dynamic architecture detection
            class ContinuousAdapterAgent:
                def __init__(self, state_dict, state_dim, action_dim, state_adapter, action_adapter, is_discrete_model=False):
                    self.device = torch.device('cpu')
                    self.state_adapter = state_adapter
                    self.action_adapter = action_adapter
                    self.is_discrete = False  # Default to continuous
                    self.is_discrete_model = is_discrete_model  # Store for adapter recreation

                    # Determine architecture from state_dict keys
                    if 'P.0.weight' in state_dict:
                        # SAC-Discrete architecture
                        print(f"  Detected SAC-Discrete architecture")
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
                        self.is_discrete = True

                    elif 'l1.weight' in state_dict and 'l3.weight' in state_dict:
                        # Could be DDPG/TD3 (continuous) or PPO-Discrete (discrete)
                        # Check output dimension to distinguish
                        original_state_dim = state_dict['l1.weight'].shape[1]
                        l1_out = state_dict['l1.weight'].shape[0]
                        l2_out = state_dict['l2.weight'].shape[0]
                        original_action_dim = state_dict['l3.weight'].shape[0]

                        print(f"  Detected model with l1/l2/l3 layers:")
                        print(f"    Actual dimensions: {original_state_dim}D state → {original_action_dim}D action")
                        print(f"    YAML dimensions: {state_dim}D state → {action_dim}D action")

                        # Check for dimension mismatch
                        if original_state_dim != state_dim or original_action_dim != action_dim:
                            print(f"  ⚠️  WARNING: YAML dimensions don't match actual model!")
                            print(f"  ⚠️  Using actual model dimensions: {original_state_dim}D → {original_action_dim}D")
                            # Update dimensions to match actual model
                            state_dim = original_state_dim
                            action_dim = original_action_dim
                            # Recreate state adapter with correct dimensions
                            self.state_adapter = PCAStateAdapter(input_dim=isaac_state_dim, output_dim=state_dim)
                            print(f"  ✓ Recreated state adapter: PCA ({isaac_state_dim}D → {state_dim}D)")
                            # Recreate action adapter with correct dimensions
                            if self.is_discrete_model:
                                self.action_adapter = DiscreteActionMapper(source_actions=action_dim, target_actions=isaac_action_dim)
                                print(f"  ✓ Recreated action adapter: Discrete Mapper ({action_dim} → {isaac_action_dim})")
                            else:
                                # Continuous models: use normalized grid adapter
                                # IMPORTANT: Use num_cubes from environment, not isaac_action_dim!
                                # We need to get num_cubes from the outer scope
                                # For now, infer from isaac_action_dim (16 actions = 4x4 grid, but only 9 cubes)
                                actual_num_cubes = 9  # Hardcoded for now - should be passed as parameter
                                self.action_adapter = ContinuousToDiscreteAdapter(
                                    continuous_dim=action_dim,
                                    num_cubes=actual_num_cubes,
                                    cube_positions=None
                                )
                                print(f"  ✓ Recreated action adapter: Continuous→Discrete ({action_dim}D → {actual_num_cubes} cubes)")

                        # Determine if this is PPO-Discrete or DDPG/TD3 based on is_discrete_model flag
                        # PPO-Discrete has discrete action space, DDPG/TD3 have continuous action space
                        if self.is_discrete_model:
                            # PPO-Discrete: discrete action probabilities
                            print(f"  Detected PPO-Discrete architecture")
                            print(f"  Model: {original_state_dim}D state → {original_action_dim}D action (discrete)")

                            self.actor = nn.Sequential(
                                nn.Linear(original_state_dim, l1_out),
                                nn.ReLU(),
                                nn.Linear(l1_out, l2_out),
                                nn.ReLU(),
                                nn.Linear(l2_out, original_action_dim),
                                nn.Softmax(dim=-1)  # PPO-Discrete uses softmax for action probabilities
                            ).to(self.device)
                            self.actor[0].load_state_dict({'weight': state_dict['l1.weight'], 'bias': state_dict['l1.bias']})
                            self.actor[2].load_state_dict({'weight': state_dict['l2.weight'], 'bias': state_dict['l2.bias']})
                            self.actor[4].load_state_dict({'weight': state_dict['l3.weight'], 'bias': state_dict['l3.bias']})
                            self.is_discrete = True
                        else:
                            # DDPG/TD3 architecture (l1, l2, l3) - continuous
                            print(f"  Detected DDPG/TD3 architecture (continuous)")
                            print(f"  Model: {original_state_dim}D state → {original_action_dim}D action (continuous)")

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
                            self.is_discrete = False

                    elif 'l1.weight' in state_dict and 'mu_head.weight' in state_dict:
                        # PPO-Continuous architecture (l1, l2, mu_head)
                        print(f"  Detected PPO-Continuous architecture")
                        original_state_dim = state_dict['l1.weight'].shape[1]
                        l1_out = state_dict['l1.weight'].shape[0]
                        l2_out = state_dict['l2.weight'].shape[0]
                        original_action_dim = state_dict['mu_head.weight'].shape[0]
                        print(f"  Original model: {original_state_dim}D state → {original_action_dim}D action")

                        self.actor = nn.Sequential(
                            nn.Linear(original_state_dim, l1_out),
                            nn.ReLU(),
                            nn.Linear(l1_out, l2_out),
                            nn.ReLU(),
                            nn.Linear(l2_out, original_action_dim),
                            nn.Tanh()  # PPO-Continuous uses tanh
                        ).to(self.device)
                        self.actor[0].load_state_dict({'weight': state_dict['l1.weight'], 'bias': state_dict['l1.bias']})
                        self.actor[2].load_state_dict({'weight': state_dict['l2.weight'], 'bias': state_dict['l2.bias']})
                        self.actor[4].load_state_dict({'weight': state_dict['mu_head.weight'], 'bias': state_dict['mu_head.bias']})
                        self.is_discrete = False

                    elif 'alpha_head.weight' in state_dict and 'beta_head.weight' in state_dict:
                        # PPO-Continuous architecture (dual-head with alpha/beta for Beta distribution)
                        print(f"  Detected PPO-Continuous (Beta) architecture")
                        original_state_dim = state_dict['l1.weight'].shape[1]
                        l1_out = state_dict['l1.weight'].shape[0]
                        l2_out = state_dict['l2.weight'].shape[0]
                        original_action_dim = state_dict['alpha_head.weight'].shape[0]
                        print(f"  Original model: {original_state_dim}D state → {original_action_dim}D action")

                        # Use alpha_head as the mean action (ignore beta for simplicity)
                        self.actor = nn.Sequential(
                            nn.Linear(original_state_dim, l1_out),
                            nn.ReLU(),
                            nn.Linear(l1_out, l2_out),
                            nn.ReLU(),
                            nn.Linear(l2_out, original_action_dim),
                            nn.Sigmoid()  # Beta distribution uses sigmoid
                        ).to(self.device)
                        self.actor[0].load_state_dict({'weight': state_dict['l1.weight'], 'bias': state_dict['l1.bias']})
                        self.actor[2].load_state_dict({'weight': state_dict['l2.weight'], 'bias': state_dict['l2.bias']})
                        self.actor[4].load_state_dict({'weight': state_dict['alpha_head.weight'], 'bias': state_dict['alpha_head.bias']})
                        self.is_discrete = False

                    elif 'a_net.0.weight' in state_dict and 'mu_layer.weight' in state_dict:
                        # SAC-Continuous architecture (a_net + mu_layer)
                        print(f"  Detected SAC-Continuous architecture")
                        original_state_dim = state_dict['a_net.0.weight'].shape[1]
                        h1 = state_dict['a_net.0.weight'].shape[0]
                        h2 = state_dict['a_net.2.weight'].shape[0] if 'a_net.2.weight' in state_dict else h1
                        original_action_dim = state_dict['mu_layer.weight'].shape[0]
                        print(f"  Original model: {original_state_dim}D state → {original_action_dim}D action")

                        self.actor = nn.Sequential(
                            nn.Linear(original_state_dim, h1),
                            nn.ReLU(),
                            nn.Linear(h1, h2),
                            nn.ReLU(),
                            nn.Linear(h2, original_action_dim),
                            nn.Tanh()  # SAC uses tanh
                        ).to(self.device)
                        self.actor[0].load_state_dict({'weight': state_dict['a_net.0.weight'], 'bias': state_dict['a_net.0.bias']})
                        if 'a_net.2.weight' in state_dict:
                            self.actor[2].load_state_dict({'weight': state_dict['a_net.2.weight'], 'bias': state_dict['a_net.2.bias']})
                        self.actor[4].load_state_dict({'weight': state_dict['mu_layer.weight'], 'bias': state_dict['mu_layer.bias']})
                        self.is_discrete = False

                    else:
                        raise ValueError(f"Unknown PyTorch architecture. Keys: {list(state_dict.keys())}")

                    self.actor.eval()

                def select_action(self, state, action_mask=None):
                    # Adapt state (flatten first, then transform returns 1D)
                    adapted_state = self.state_adapter.transform(state.flatten())

                    # Get action
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(adapted_state).unsqueeze(0).to(self.device)
                        if self.is_discrete:
                            action_probs = self.actor(state_tensor)
                            model_action = torch.argmax(action_probs, dim=-1).item()
                            # Adapt action (pass action_mask to prevent out-of-bounds)
                            isaac_action = self.action_adapter.map_action(model_action, action_mask=action_mask)
                        else:
                            continuous_action = self.actor(state_tensor).cpu().numpy().flatten()
                            # Adapt continuous action to discrete cube index
                            isaac_action = self.action_adapter.map_action(continuous_action)

                    return isaac_action

            agent = ContinuousAdapterAgent(state_dict, model_state_dim, model_action_dim, state_adapter, action_adapter, is_discrete_model)
            print(f"✅ Loaded PyTorch actor model with adapters")
            return agent

    else:
        raise ValueError(f"Unknown agent1_type: {agent_type}")


def load_agent2(model_config: dict, grid_size: int, num_cubes: int, cube_spacing: float = 0.13):
    """
    Load Agent 2 (Reshuffling) based on model configuration

    Returns:
        agent: Loaded agent
    """
    agent_type = model_config["agent2_type"]

    # Calculate Agent 2 state dimension
    agent2_state_dim = (num_cubes * 3) + 3 + num_cubes + (grid_size * grid_size) + 10

    if agent_type == "masac":
        # Load MASAC agent
        pretrained_path = project_root / "scripts" / "Reinforcement Learning" / "MASAC" / "pretrained_models"

        agent = MASACContinuousWrapper(
            state_dim=agent2_state_dim,
            grid_size=grid_size,
            num_cubes=num_cubes,
            cube_spacing=cube_spacing,
            pretrained_model_path=str(pretrained_path),
            use_dimension_adapter=True,
            memory_size=10000,
            batch_size=64
        )
        print(f"✅ Created MASAC Agent 2")
        return agent

    elif agent_type == "heuristic":
        # Create heuristic agent
        # Note: env will be set later when TwoAgentEnv is created
        agent = HeuristicAgent2(
            state_dim=agent2_state_dim,
            action_dim=3,
            grid_size=grid_size,
            num_cubes=num_cubes,
            cube_spacing=cube_spacing,
            env=None  # Will be set to TwoAgentEnv later
        )
        print(f"✅ Created Heuristic Agent 2")
        return agent

    else:
        raise ValueError(f"Unknown agent2_type: {agent_type}")


def test_model_with_seed(model_config: dict, seed: int, num_episodes: int,
                         grid_size: int, num_cubes: int, logger: TwoAgentLogger,
                         base_env, world):
    """
    Test a single model with a specific seed

    Args:
        model_config: Model configuration dictionary
        seed: Random seed
        num_episodes: Number of episodes to test
        grid_size: Grid size
        num_cubes: Number of cubes
        logger: TwoAgentLogger instance
        base_env: Pre-created base environment (reused across models)
        world: Pre-created world (reused across models)

    Returns:
        episode_results: List of episode results
    """
    model_name = model_config["name"]
    print(f"\n{'='*80}")
    print(f"Testing Model: {model_name} | Seed: {seed}")
    print(f"{'='*80}")

    # Load Agent 1
    print("✓ Loading Agent 1...")
    agent1 = load_agent1(model_config, base_env, grid_size, num_cubes)

    # Load Agent 2
    print("✓ Loading Agent 2...")
    agent2 = load_agent2(model_config, grid_size, num_cubes, cube_spacing=0.13)

    # Create two-agent environment
    two_agent_env = TwoAgentEnv(
        base_env=base_env,
        ddqn_agent=agent1,  # Agent 1 (pick sequence)
        grid_size=grid_size,
        num_cubes=num_cubes,
        max_reshuffles_per_episode=5,
        reshuffle_reward_scale=1.0,
        max_episode_steps=50,
        verbose=False
    )

    # Set environment reference for heuristic agents
    if isinstance(agent1, HeuristicAgent1):
        agent1.env = base_env
    if isinstance(agent2, HeuristicAgent2):
        agent2.env = two_agent_env

    # Relax reshuffling thresholds for testing
    print("[TEST] Relaxing reshuffling thresholds...")
    two_agent_env.reshuffle_decision.min_reachable_distance = 0.30
    two_agent_env.reshuffle_decision.max_reachable_distance = 0.90
    two_agent_env.reshuffle_decision.path_length_ratio_threshold = 1.5
    two_agent_env.reshuffle_decision.crowded_threshold = 2
    two_agent_env.reshuffle_decision.rrt_failure_window = 2
    two_agent_env.reshuffle_decision.min_clearance = 0.35
    two_agent_env.reshuffle_decision.far_cube_ratio = 1.1
    two_agent_env.reshuffle_decision.batch_reshuffle_count = 2

    # Enable test mode for fast PCA fitting
    print("[TEST] Enabling test mode for fast PCA fitting...")
    base_env.test_mode = True

    # Fit PCA dimension adapter (if needed)
    if hasattr(agent2, 'fit_dimension_adapter'):
        agent2.fit_dimension_adapter(two_agent_env, n_samples=100)  # Reduced from 500 for faster setup

    # Set test mode
    if hasattr(agent2, 'set_test_mode'):
        agent2.set_test_mode(True)

    # Test episodes
    episode_results = []
    start_time = datetime.now().isoformat()

    for episode in range(num_episodes):
        # DO NOT randomize cube positions - use fixed positions for consistent testing
        # This ensures all models are tested on the same cube configurations
        # if hasattr(base_env, 'franka_controller') and base_env.franka_controller is not None:
        #     base_env.franka_controller.randomize_cube_positions()

        obs, reset_info = two_agent_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        agent1_reward = 0
        agent2_reward = 0
        episode_length = 0
        reshuffles_performed = 0
        episode_start_time = time.time()

        print(f"\n[Episode {episode+1}/{num_episodes}] Starting...")

        # Check if this is a heuristic model (skip reward calculation)
        is_heuristic = (model_name == "Heuristic")

        while not (done or truncated) and episode_length < 50:
            # NOTE: Episode length limit of 50 steps
            # - Most episodes timeout at 50 because agents fail to pick all 9 cubes
            # - Only successful episodes (all cubes picked) have episode_length < 50
            # - Episode length = number of timesteps taken, NOT number of cubes picked
            # Calculate valid cubes for action masking
            valid_cubes = [
                i for i in range(num_cubes)
                if i not in two_agent_env.base_env.objects_picked
                and two_agent_env.reshuffle_count_per_cube.get(i, 0) < 2
            ]

            # Agent 2 selects reshuffling action
            if hasattr(agent2, 'select_action'):
                if model_config["agent2_type"] == "masac":
                    action_dict = agent2.select_action(obs, deterministic=True, valid_cubes=valid_cubes)
                elif model_config["agent2_type"] == "heuristic":
                    # Pass valid_cubes to heuristic agent (now supports action masking)
                    action_continuous = agent2.select_action(obs, deterministic=True, valid_cubes=valid_cubes)
                    # Convert continuous to dictionary
                    cube_idx = int((action_continuous[0] + 1) / 2 * (num_cubes - 1))
                    grid_x = int((action_continuous[1] + 1) / 2 * (grid_size - 1))
                    grid_y = int((action_continuous[2] + 1) / 2 * (grid_size - 1))
                    action_dict = {'cube_idx': cube_idx, 'target_grid_x': grid_x, 'target_grid_y': grid_y}
            else:
                action_dict = None

            # Skip if no valid action
            if action_dict is None:
                print(f"  [WARNING] No valid cubes to reshuffle, skipping step")
                break

            # Convert dictionary action to integer action
            action_int = two_agent_env.reshuffle_action_space.encode_action(
                cube_idx=action_dict['cube_idx'],
                grid_x=action_dict['target_grid_x'],
                grid_y=action_dict['target_grid_y']
            )

            # Execute action
            next_obs, reward, done, truncated, info = two_agent_env.step(action_int)

            episode_length += 1

            # Track rewards (skip for heuristic - it's not an RL agent)
            if not is_heuristic:
                episode_reward += reward

                # Track Agent 1 and Agent 2 rewards separately using info dict
                # Agent 1 reward: picking reward (from base environment)
                # Agent 2 reward: reshuffling reward (from reshuffle actions)
                pick_reward = info.get('pick_reward', 0.0)
                reshuffle_reward = info.get('reshuffle_reward', 0.0)

                agent1_reward += pick_reward
                agent2_reward += reshuffle_reward

            if info.get('reshuffled_this_step', False):
                reshuffles_performed += 1

            # Log timestep (all models including heuristic)
            timestep_data = {
                'episode': episode + 1,
                'step_in_episode': episode_length,
                'model': model_name,
                'reward': float(reward) if not is_heuristic else '',  # Blank for heuristic
                'cumulative_reward': float(episode_reward) if not is_heuristic else '',  # Blank for heuristic
                'reshuffled': info.get('reshuffled_this_step', False),
                'cubes_picked_so_far': len(two_agent_env.base_env.objects_picked),
                'done': done,
                'truncated': truncated,
                'timestamp': datetime.now().isoformat(),
                'planner': 'Isaac Sim RRT'
            }
            logger.log_timestep(timestep_data)

            obs = next_obs

        # Episode complete
        episode_duration = time.time() - episode_start_time
        cubes_picked = len(two_agent_env.base_env.objects_picked)
        success = (cubes_picked == num_cubes)

        # Get additional metrics from environment
        pick_failures = getattr(base_env, 'episode_pick_failures', 0)
        successful_picks = cubes_picked
        unreachable_cubes = num_cubes - cubes_picked

        # Calculate path efficiency (optimal_path_length / actual_path_length)
        # This measures how close RRT paths are to optimal straight-line paths
        path_efficiency = 0.0
        if hasattr(base_env, 'rrt_path_lengths') and hasattr(base_env, 'rrt_optimal_path_lengths'):
            if len(base_env.rrt_path_lengths) > 0 and len(base_env.rrt_optimal_path_lengths) > 0:
                total_actual = sum(base_env.rrt_path_lengths)
                total_optimal = sum(base_env.rrt_optimal_path_lengths)
                if total_actual > 0:
                    path_efficiency = total_optimal / total_actual

        # Calculate action entropy (not applicable for deterministic testing with epsilon=0)
        # For DDQN agents in testing mode, entropy is 0 (greedy policy)
        action_entropy = 0.0

        # Log episode (all models including heuristic, but set rewards to blank for heuristic)
        episode_data = {
            'episode': episode + 1,
            'model': model_name,
            'agent1_reward': float(agent1_reward) if not is_heuristic else '',  # Blank for heuristic
            'success': success,
            'cubes_picked': cubes_picked,
            'pick_failures': pick_failures,
            'successful_picks': successful_picks,
            'unreachable_cubes': unreachable_cubes,
            'path_efficiency': float(path_efficiency),
            'action_entropy': float(action_entropy),
            'agent2_reward': float(agent2_reward) if not is_heuristic else '',  # Blank for heuristic
            'reshuffles_performed': reshuffles_performed,
            'total_distance_reduced': float(two_agent_env.total_distance_reduced),
            'total_time_saved': float(two_agent_env.total_time_saved),
            'total_reward': float(episode_reward) if not is_heuristic else '',  # Blank for heuristic
            'episode_length': episode_length,
            'duration': episode_duration,
            'timestamp': datetime.now().isoformat(),
            'planner': 'Isaac Sim RRT',
            'grid_size': grid_size,
            'num_cubes': num_cubes
        }
        logger.log_episode(episode_data)
        episode_results.append(episode_data)

        # Print episode summary
        if not is_heuristic:
            print(f"Episode {episode+1}/{num_episodes}: "
                  f"Reward={episode_reward:.2f} (A1={agent1_reward:.2f}, A2={agent2_reward:.2f}), "
                  f"Reshuffles={reshuffles_performed}, "
                  f"Cubes={cubes_picked}/{num_cubes}")
        else:
            # For heuristic, don't show rewards (they're 0 anyway)
            print(f"Episode {episode+1}/{num_episodes}: "
                  f"Reshuffles={reshuffles_performed}, "
                  f"Cubes={cubes_picked}/{num_cubes}, "
                  f"Duration={episode_duration:.1f}s")

    # Write summary for this model (DISABLED - JSON not needed, use CSV analysis instead)
    # logger.write_summary_for_model(model_name, start_time)

    return episode_results


def main():
    """Main testing function"""
    print(f"\n{'='*80}")
    print(f"TWO-AGENT SYSTEM TESTING")
    print(f"{'='*80}")
    print(f"Action Space: {args.action_space}")
    print(f"Seeds: {args.seeds}")
    print(f"Episodes per model: {args.episodes}")
    print(f"Grid Size: {args.grid_size}x{args.grid_size}")
    print(f"Number of Cubes: {args.num_cubes}")
    print(f"{'='*80}\n")

    # Select models based on action space
    # NOTE: Continuous models tested FIRST to catch errors early
    if args.action_space == "discrete":
        models_to_test = [("discrete", DISCRETE_MODELS)]
    elif args.action_space == "continuous":
        models_to_test = [("continuous", CONTINUOUS_MODELS)]
    else:  # both
        models_to_test = [("continuous", CONTINUOUS_MODELS), ("discrete", DISCRETE_MODELS)]

    # Filter models if MODEL_FILTER environment variable is set
    import os
    model_filter = os.environ.get("MODEL_FILTER", None)
    if model_filter:
        print(f"🔍 Filtering models: Only testing '{model_filter}'")
        filtered_models = []
        for action_space, models in models_to_test:
            filtered = [m for m in models if m.get("name") == model_filter]
            if filtered:
                filtered_models.append((action_space, filtered))
        models_to_test = filtered_models
        if not models_to_test:
            print(f"❌ No models found matching filter '{model_filter}'")
            return

    total_models = sum(len(models) for _, models in models_to_test)
    print(f"Testing {total_models} models ({args.action_space}) with {len(args.seeds)} seeds each")
    print(f"Total test runs: {total_models * len(args.seeds)}")
    print(f"Total episodes: {total_models * len(args.seeds) * args.episodes}\n")

    # Test each seed
    for seed in args.seeds:
        print(f"\n{'#'*80}")
        print(f"# SEED: {seed}")
        print(f"{'#'*80}\n")

        # Set random seed
        np.random.seed(seed)

        # Create Isaac Sim environment ONCE per seed (reuse for all models)
        print("✓ Creating Isaac Sim environment...")
        base_env, world = create_isaacsim_environment(
            grid_size=args.grid_size,
            num_cubes=args.num_cubes,
            max_steps=50
        )

        # Test each action space and its models
        for action_space, models in models_to_test:
            print(f"\n{'='*80}")
            print(f"Testing {action_space.upper()} models (Seed: {seed})")
            print(f"{'='*80}\n")

            # Create logger for this action space and seed
            script_dir = Path(__file__).parent
            results_base_dir = script_dir / "two_agent_results"

            print(f"\n📊 Creating logger for action_space='{action_space}', seed={seed}")
            logger = TwoAgentLogger(
                base_dir=str(results_base_dir),
                action_space=action_space,
                seed=seed
            )

            # Test each model (reusing the same environment)
            for i, model_config in enumerate(models, 1):
                print(f"\n[{i}/{len(models)}] Testing model: {model_config['name']}")
                print(f"  Action space: {action_space}")
                print(f"  Model type: {model_config.get('agent1_type', 'unknown')}")
                print(f"  State dim: {model_config.get('agent1_state_dim', 'unknown')} → Action dim: {model_config.get('agent1_action_dim', 'unknown')}")

                try:
                    test_model_with_seed(
                        model_config=model_config,
                        seed=seed,
                        num_episodes=args.episodes,
                        grid_size=args.grid_size,
                        num_cubes=args.num_cubes,
                        logger=logger,
                        base_env=base_env,
                        world=world
                    )
                    print(f"✅ Completed: {model_config['name']}")
                except Exception as e:
                    print(f"❌ Error testing model {model_config['name']}: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"⚠️  Skipping {model_config['name']} and continuing with next model...")
                    continue

        # Clean up after all models for this seed
        print("\n[CLEANUP] Cleaning up Isaac Sim environment...")
        try:
            if world is not None:
                print("[CLEANUP] Stopping world...")
                world.stop()
                print("[CLEANUP] Clearing scene...")
                world.scene.clear()
                print("[CLEANUP] Resetting world...")
                world.clear_instance()
            from isaacsim.core.utils.stage import clear_stage
            print("[CLEANUP] Clearing stage...")
            clear_stage()
            print("[CLEANUP] Stage cleared successfully")
        except Exception as e:
            print(f"[CLEANUP] Warning during cleanup: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"TWO-AGENT SYSTEM TESTING COMPLETE")
    print(f"{'='*80}")
    script_dir = Path(__file__).parent
    results_base_path = script_dir / "two_agent_results"
    if args.action_space == "both":
        print(f"Results saved to:")
        print(f"  - Discrete: {results_base_path / 'discrete'}")
        print(f"  - Continuous: {results_base_path / 'continuous'}")
    else:
        print(f"Results saved to: {results_base_path / args.action_space}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always clean up Isaac Sim
        print("\n[CLEANUP] Closing Isaac Sim...")
        try:
            from omni.isaac.core.utils.stage import clear_stage
            clear_stage()
            print("[CLEANUP] Stage cleared")
        except Exception as cleanup_error:
            print(f"[CLEANUP] Warning: Could not clear stage: {cleanup_error}")

        try:
            simulation_app.close()
            print("[CLEANUP] Isaac Sim closed successfully")
        except Exception as close_error:
            print(f"[CLEANUP] Warning: Error closing Isaac Sim: {close_error}")

