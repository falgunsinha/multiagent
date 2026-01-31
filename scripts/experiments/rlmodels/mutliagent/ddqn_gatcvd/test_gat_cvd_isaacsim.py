"""
Test GAT+CVD Multi-Agent System (DDQN+GAT + MASAC+GAT + CVD)
Tests GAT+CVD model against discrete baseline models with comprehensive logging.

Usage:
    # Test with default settings (50 episodes, 2 seeds)
    C:\isaacsim\python.bat test_gat_cvd_isaacsim.py
    
    # Test with custom settings
    C:\isaacsim\python.bat test_gat_cvd_isaacsim.py --episodes 100 --seeds 42 123 456
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

# Add MARL source to path (for GAT+CVD)
marl_src = Path(r"C:\isaacsim\cobotproject\scripts\Reinforcement Learning\MARL\src")
if str(marl_src) not in sys.path:
    sys.path.insert(0, str(marl_src))

# Import Isaac Sim components (must be before other imports)
from isaacsim import SimulationApp
import argparse

# Parse arguments BEFORE creating SimulationApp
parser = argparse.ArgumentParser(description="Test GAT+CVD Multi-Agent System")
parser.add_argument("--episodes", type=int, default=50,
                   help="Number of episodes per model (default: 50)")
parser.add_argument("--seeds", type=int, nargs='+', default=[42, 123],
                   help="Random seeds for testing (default: 42 123)")
parser.add_argument("--grid_size", type=int, default=4,
                   help="Grid size (default: 4)")
parser.add_argument("--num_cubes", type=int, default=9,
                   help="Number of cubes (default: 9)")
parser.add_argument("--checkpoint", type=str, 
                   default=r"C:\isaacsim\cobotproject\scripts\Reinforcement Learning\MARL\src\gat_cvd\models\gat_cvd_isaacsim_grid4_cubes9_20260123_132522_step_20000.pt",
                   help="Path to GAT+CVD checkpoint to test")
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
import torch

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

# Import GAT+CVD components
from gat_cvd.gat_cvd_agent import GATCVDAgent
from gat_cvd.graph_utils import build_graph, compute_edge_features

# Import our components from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
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


print("=" * 80)
print("GAT+CVD MULTI-AGENT TESTING - ISAAC SIM")
print("=" * 80)
print(f"Checkpoint: {args.checkpoint}")
print(f"Episodes per model: {args.episodes}")
print(f"Seeds: {args.seeds}")
print(f"Grid: {args.grid_size}x{args.grid_size}")
print(f"Cubes: {args.num_cubes}")
print("=" * 80)


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

        print(f"[TRAINER] Initializing Franka RRT Trainer for GAT+CVD Testing")
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
            # Go up 5 levels: ddqn_gatcvd -> mutliagent -> rlmodels -> experiments -> scripts -> cobotproject
            project_root = os.path.join(script_dir, "..", "..", "..", "..", "..")
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

    def get_obstacle_positions(self):
        """Get current obstacle positions"""
        positions = []
        for obstacle in self.obstacle_prims:
            pos, _ = obstacle.get_world_pose()
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


# Model configurations for GAT+CVD testing (ALL DISCRETE MODELS)
# NOTE: GAT+CVD was trained as a single integrated system (DDQN+GAT+MASAC+GAT+CVD)
# We CANNOT swap pretrained DDQN variants with GAT+CVD's MASAC - they were never trained together
# So baseline models use pretrained MASAC instead (renamed as SAC for simplicity)
DISCRETE_MODELS = [
    # 1. DDQN+GAT (Full integrated system - your trained model) - TEST FIRST
    {
        "name": "DDQN+GAT",
        "agent1_type": "gat_cvd",
        "agent1_path": args.checkpoint,
        "agent2_type": "gat_cvd",
        "agent2_path": args.checkpoint
    },

    # 2. Heuristic (Baseline - simplest)
    {
        "name": "Heuristic",
        "agent1_type": "heuristic",
        "agent1_path": None,
        "agent2_type": "heuristic",
        "agent2_path": None
    },

    # 3. Duel-DDQN + SAC (Pretrained on LunarLander + Pretrained MASAC)
    {
        "name": "Duel-DDQN+SAC",
        "agent1_type": "ddqn",
        "agent1_path": str(project_root / "scripts/experiments/rlmodels/models/pretrained/duel_ddqn_lunarlander.pth"),
        "agent1_state_dim": 8,
        "agent1_action_dim": 4,
        "agent2_type": "masac",
        "agent2_path": None  # MASAC loads from pretrained_models directory
    },

    # 4. PER-DDQN-Full + SAC (Pretrained on LunarLander + Pretrained MASAC)
    {
        "name": "PER-DDQN-Full+SAC",
        "agent1_type": "ddqn",
        "agent1_path": str(project_root / "scripts/experiments/rlmodels/models/pretrained/per_ddqn_full_lunarlander.pth"),
        "agent1_state_dim": 8,
        "agent1_action_dim": 4,
        "agent2_type": "masac",
        "agent2_path": None
    },

    # 5. PER-DDQN-Light + SAC (Pretrained on LunarLander + Pretrained MASAC)
    {
        "name": "PER-DDQN-Light+SAC",
        "agent1_type": "ddqn",
        "agent1_path": str(project_root / "scripts/experiments/rlmodels/models/pretrained/per_ddqn_light_lunarlander.pth"),
        "agent1_state_dim": 8,
        "agent1_action_dim": 4,
        "agent2_type": "masac",
        "agent2_path": None
    },

    # 6. C51-DDQN + SAC (Pretrained on LunarLander + Pretrained MASAC)
    {
        "name": "C51-DDQN+SAC",
        "agent1_type": "ddqn",
        "agent1_path": str(project_root / "scripts/experiments/rlmodels/models/pretrained/c51_ddqn_lunarlander.pth"),
        "agent1_state_dim": 8,
        "agent1_action_dim": 4,
        "agent2_type": "masac",
        "agent2_path": None
    },

    # 7. PPO-Discrete + SAC (Pretrained on CartPole + Pretrained MASAC)
    {
        "name": "PPO-Discrete+SAC",
        "agent1_type": "pytorch",
        "agent1_path": str(project_root / "scripts/experiments/rlmodels/models/pretrained/ppo_discrete_lunarlander_actor.pth"),
        "agent1_state_dim": 4,
        "agent1_action_dim": 2,
        "agent2_type": "masac",
        "agent2_path": None
    },

    # 8. SAC-Discrete + SAC (Pretrained on LunarLander + Pretrained MASAC)
    {
        "name": "SAC-Discrete+SAC",
        "agent1_type": "pytorch",
        "agent1_path": str(project_root / "scripts/experiments/rlmodels/models/pretrained/sac_discrete_lunarlander_actor.pth"),
        "agent1_state_dim": 8,
        "agent1_action_dim": 4,
        "agent2_type": "masac",
        "agent2_path": None
    }
]


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
    return env, trainer.world, trainer


def load_agent1(model_config: dict, base_env, grid_size: int, num_cubes: int, trainer, device):
    """
    Load Agent 1 (Pick Sequence) based on model configuration

    Returns:
        agent: Loaded agent
    """
    agent_type = model_config["agent1_type"]
    agent_path = model_config.get("agent1_path")

    if agent_type == "gat_cvd":
        # Load GAT+CVD agent
        from gat_cvd_test_utils import load_gat_cvd_checkpoint, create_gat_cvd_wrapper

        # Load GAT+CVD config
        config_path = Path(r"C:\isaacsim\cobotproject\scripts\Reinforcement Learning\MARL\src\gat_cvd\config_gat_cvd.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Update config with environment-specific parameters
        config['node_dim'] = 7
        config['edge_dim'] = 3
        config['n_actions_ddqn'] = grid_size * grid_size
        config['n_actions_masac'] = num_cubes * grid_size * grid_size

        # Load checkpoint
        agent = load_gat_cvd_checkpoint(agent_path, config, device)

        # Create wrapper for TwoAgentEnv compatibility
        wrapper = create_gat_cvd_wrapper(agent, trainer, config, device)

        print(f"✅ Loaded GAT+CVD Agent 1 (DDQN+GAT)")
        return wrapper

    elif agent_type == "custom_ddqn":
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
        # Load pretrained model with adapters (same as test_two_agent_system.py)
        if not ADAPTERS_AVAILABLE:
            raise RuntimeError("Adapters not available for pretrained models")

        import torch as torch_lib
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

        # Create action adapter (discrete only)
        action_adapter = DiscreteActionMapper(source_actions=model_action_dim, target_actions=isaac_action_dim)
        print(f"  ✓ Action adapter: Discrete Mapper ({model_action_dim} → {isaac_action_dim})")

        # Load pretrained model
        state_dict = torch_lib.load(model_path, map_location='cpu', weights_only=False)

        # Create wrapper agent (simplified version - only DDQN variants for discrete models)
        class DDQNAdapterAgent:
            def __init__(self, state_dict, state_dim, action_dim, state_adapter, action_adapter):
                self.device = torch_lib.device('cpu')
                self.state_adapter = state_adapter
                self.action_adapter = action_adapter
                self.is_c51 = False  # Default: not C51 distributional RL

                # Determine architecture from state_dict keys
                keys = list(state_dict.keys())
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
                elif any(k.startswith('Q.') for k in keys):
                    # PER-DDQN (Q.* naming) - map to fc* naming
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
                elif any(k.startswith('net.') for k in keys):
                    # C51-DDQN (net.* naming) - Distributional RL with atoms
                    # Output shape: [num_atoms * action_dim] (e.g., 51 atoms × 4 actions = 204)
                    print(f"  Detected C51-DDQN (Distributional RL) architecture")
                    hidden_size = state_dict['net.0.weight'].shape[0]
                    output_size = state_dict['net.4.weight'].shape[0]  # e.g., 204
                    num_atoms = output_size // action_dim  # e.g., 204 / 4 = 51
                    print(f"  C51 params: {num_atoms} atoms, {action_dim} actions, output_size={output_size}")

                    self.network = nn.Sequential(
                        nn.Linear(state_dim, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, output_size)  # Full distributional output
                    ).to(self.device)
                    self.network[0].load_state_dict({'weight': state_dict['net.0.weight'], 'bias': state_dict['net.0.bias']})
                    self.network[2].load_state_dict({'weight': state_dict['net.2.weight'], 'bias': state_dict['net.2.bias']})
                    self.network[4].load_state_dict({'weight': state_dict['net.4.weight'], 'bias': state_dict['net.4.bias']})
                    self.is_dueling = False
                    self.is_c51 = True
                    self.num_atoms = num_atoms
                    self.action_dim = action_dim
                elif any(k.startswith('l1') for k in keys):
                    # PPO-Discrete (l1/l2/l3 naming) - map to fc* naming
                    hidden_size = state_dict['l1.weight'].shape[0]
                    self.network = nn.Sequential(
                        nn.Linear(state_dim, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, action_dim)
                    ).to(self.device)
                    self.network[0].load_state_dict({'weight': state_dict['l1.weight'], 'bias': state_dict['l1.bias']})
                    self.network[2].load_state_dict({'weight': state_dict['l2.weight'], 'bias': state_dict['l2.bias']})
                    self.network[4].load_state_dict({'weight': state_dict['l3.weight'], 'bias': state_dict['l3.bias']})
                    self.is_dueling = False
                elif any(k.startswith('P.') for k in keys):
                    # SAC-Discrete (P.* naming) - map to fc* naming
                    hidden_size = state_dict['P.0.weight'].shape[0]
                    self.network = nn.Sequential(
                        nn.Linear(state_dim, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, action_dim)
                    ).to(self.device)
                    self.network[0].load_state_dict({'weight': state_dict['P.0.weight'], 'bias': state_dict['P.0.bias']})
                    self.network[2].load_state_dict({'weight': state_dict['P.2.weight'], 'bias': state_dict['P.2.bias']})
                    self.network[4].load_state_dict({'weight': state_dict['P.4.weight'], 'bias': state_dict['P.4.bias']})
                    self.is_dueling = False
                else:
                    raise ValueError(f"Unknown DDQN architecture. Keys: {list(state_dict.keys())}")

                self.network.eval()
                if hasattr(self, 'value_stream'):
                    self.value_stream.eval()
                    self.advantage_stream.eval()

            def select_action(self, state, action_mask=None):
                # Adapt state
                adapted_state = self.state_adapter.transform(state.flatten())

                # Get Q-values
                with torch_lib.no_grad():
                    state_tensor = torch_lib.FloatTensor(adapted_state).unsqueeze(0).to(self.device)

                    if self.is_dueling:
                        features = self.network(state_tensor)
                        value = self.value_stream(features)
                        advantage = self.advantage_stream(features)
                        if advantage.dim() == 1:
                            advantage = advantage.unsqueeze(0)
                            value = value.unsqueeze(0)
                        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
                    elif self.is_c51:
                        # C51 Distributional RL: Convert distribution to Q-values
                        # Output shape: [batch, num_atoms * action_dim]
                        dist_output = self.network(state_tensor)  # [1, 204] for 51 atoms × 4 actions

                        # Reshape to [batch, action_dim, num_atoms]
                        dist_output = dist_output.view(-1, self.action_dim, self.num_atoms)  # [1, 4, 51]

                        # Apply softmax over atoms to get probabilities
                        probs = torch_lib.nn.functional.softmax(dist_output, dim=-1)  # [1, 4, 51]

                        # Compute expected Q-value for each action (assume uniform support [-10, 10])
                        # For simplicity, just take the mean over atoms (equivalent to expected value)
                        q_values = probs.mean(dim=-1)  # [1, 4]
                    else:
                        q_values = self.network(state_tensor)

                # Select best action
                model_action = q_values.argmax().item()

                # Adapt action
                isaac_action = self.action_adapter.map_action(model_action, action_mask=action_mask)
                return isaac_action

        agent = DDQNAdapterAgent(state_dict, model_state_dim, model_action_dim, state_adapter, action_adapter)
        print(f"✅ Loaded DDQN variant with adapters")
        return agent

    else:
        raise ValueError(f"Unknown agent1_type: {agent_type}")


def load_agent2(model_config: dict, grid_size: int, num_cubes: int, cube_spacing: float = 0.13):
    """
    Load Agent 2 (Reshuffling) based on model configuration

    Returns:
        agent: Loaded agent (MASAC or GAT+CVD)
    """
    agent_type = model_config["agent2_type"]

    # Calculate Agent 2 state dimension
    agent2_state_dim = (num_cubes * 3) + 3 + num_cubes + (grid_size * grid_size) + 10

    if agent_type == "gat_cvd":
        # GAT+CVD agent is already loaded in load_agent1
        # Agent 2 uses the same GAT+CVD agent (MASAC component)
        print(f"✅ Using GAT+CVD Agent 2 (MASAC+GAT) - shared with Agent 1")
        return None  # Will be handled by TwoAgentEnv wrapper

    elif agent_type == "masac":
        # Load MASAC agent
        pretrained_path = project_root / "scripts" / "Reinforcement Learning" / "MASAC" / "pretrained_models"

        agent = MASACContinuousWrapper(
            state_dim=agent2_state_dim,
            grid_size=grid_size,
            num_cubes=num_cubes,
            cube_spacing=cube_spacing,
            pretrained_model_path=str(pretrained_path)
        )

        print(f"✅ Loaded MASAC Agent 2 (Reshuffling)")
        return agent

    elif agent_type == "heuristic":
        # Create heuristic agent
        agent = HeuristicAgent2(state_dim=agent2_state_dim, grid_size=grid_size, num_cubes=num_cubes)
        print(f"✅ Created Heuristic Agent 2")
        return agent

    else:
        raise ValueError(f"Unknown agent2_type: {agent_type}")


# Main test loop
def main():
    """Main testing function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Using device: {device}")

    # Create Isaac Sim environment (shared across all tests)
    print("\n" + "="*80)
    print("CREATING ISAAC SIM ENVIRONMENT")
    print("="*80)

    base_env, world, trainer = create_isaacsim_environment(
        grid_size=args.grid_size,
        num_cubes=args.num_cubes,
        max_steps=50
    )

    # Test each model with multiple seeds
    for seed in args.seeds:
        print(f"\n{'='*80}")
        print(f"TESTING WITH SEED: {seed}")
        print(f"{'='*80}")

        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Create logger for this seed
        logger = TwoAgentLogger(
            base_dir=str(project_root / "scripts/experiments/rlmodels/mutliagent/ddqn_gatcvd/gat_cvd_test_results"),
            action_space="discrete",
            seed=seed
        )

        # Test each model
        for model_idx, model_config in enumerate(DISCRETE_MODELS):
            print(f"\n{'-'*80}")
            print(f"Model {model_idx + 1}/{len(DISCRETE_MODELS)}: {model_config['name']}")
            print(f"{'-'*80}")

            try:
                # Load agents
                agent1 = load_agent1(model_config, base_env, args.grid_size, args.num_cubes, trainer, device)
                agent2 = load_agent2(model_config, args.grid_size, args.num_cubes)

                # Initialize variables for GAT+CVD special handling
                gat_cvd_full_agent = None
                gat_cvd_trainer = None

                # Special handling for GAT+CVD (uses same agent for both Agent1 and Agent2)
                if model_config["agent1_type"] == "gat_cvd":
                    # For GAT+CVD, agent1 is the wrapper, we need the underlying agent
                    gat_cvd_agent = agent1.agent

                    # Create TwoAgentEnv with GAT+CVD
                    env = TwoAgentEnv(
                        base_env=base_env,
                        ddqn_agent=agent1,  # Wrapper for DDQN component
                        grid_size=args.grid_size,
                        num_cubes=args.num_cubes,
                        max_reshuffles_per_episode=5,
                        reshuffle_reward_scale=1.0,
                        max_episode_steps=50,
                        verbose=False
                    )

                    # Override the masac_agent with GAT+CVD's MASAC component
                    env.masac_agent = gat_cvd_agent  # Use the full agent for MASAC

                    # Store the underlying agent and trainer for special handling in test loop
                    agent2 = None  # Will be handled specially in test loop
                    gat_cvd_full_agent = gat_cvd_agent
                    gat_cvd_trainer = trainer
                else:
                    # Create TwoAgentEnv with separate agents
                    env = TwoAgentEnv(
                        base_env=base_env,
                        ddqn_agent=agent1,
                        grid_size=args.grid_size,
                        num_cubes=args.num_cubes,
                        max_reshuffles_per_episode=5,
                        reshuffle_reward_scale=1.0,
                        max_episode_steps=50,
                        verbose=False
                    )
                    env.masac_agent = agent2

                # Run episodes (log all episodes including timeouts)
                for episode in range(args.episodes):
                    print(f"\n  Episode {episode + 1}/{args.episodes}")

                    # Randomize cube positions for stochasticity
                    trainer.randomize_cube_positions()

                    # Reset environment
                    obs, info = env.reset()
                    done = False
                    truncated = False
                    step = 0
                    cumulative_reward = 0.0

                    # Episode metrics
                    episode_start_time = time.time()

                    while not (done or truncated):
                        # Calculate valid cubes for action masking
                        valid_cubes = [
                            i for i in range(args.num_cubes)
                            if i not in env.base_env.objects_picked
                            and env.reshuffle_count_per_cube.get(i, 0) < 2
                        ]

                        # Agent 2 selects reshuffling action
                        if model_config["agent2_type"] == "gat_cvd":
                            # Special handling for GAT+CVD - call select_actions() once for both agents
                            from gat_cvd.graph_utils import build_graph

                            # Build graph from observation
                            robot_pos, _ = gat_cvd_trainer.franka.get_world_pose()
                            robot_positions = [robot_pos]
                            object_positions = gat_cvd_trainer.get_cube_positions()
                            obstacle_positions = gat_cvd_trainer.get_obstacle_positions()

                            graph = build_graph(
                                obs=obs,
                                robot_positions=robot_positions,
                                object_positions=object_positions,
                                obstacles=obstacle_positions,
                                edge_threshold=2.0,
                                device=device
                            )

                            # Get both actions at once (efficient!)
                            _, action_masac_int = gat_cvd_full_agent.select_actions(
                                graph,
                                epsilon_ddqn=0.0,  # No exploration
                                epsilon_masac=0.0,  # No exploration
                                action_mask=None
                            )

                            # MASAC returns discrete action integer - decode it
                            # decode_action returns ReshuffleAction object, convert to dict
                            reshuffle_action = env.reshuffle_action_space.decode_action(action_masac_int)
                            action_dict = {
                                'cube_idx': reshuffle_action.cube_idx,
                                'target_grid_x': reshuffle_action.target_grid_x,
                                'target_grid_y': reshuffle_action.target_grid_y
                            }

                        elif model_config["agent2_type"] == "masac":
                            action_dict = agent2.select_action(obs, deterministic=True, valid_cubes=valid_cubes)

                            # Handle case where MASAC returns None (no valid cubes)
                            if action_dict is None:
                                # No valid cubes to reshuffle - use a default action (first cube, center position)
                                action_dict = {
                                    'cube_idx': 0,
                                    'target_grid_x': args.grid_size // 2,
                                    'target_grid_y': args.grid_size // 2
                                }
                        elif model_config["agent2_type"] == "heuristic":
                            action_continuous = agent2.select_action(obs, deterministic=True, valid_cubes=valid_cubes)
                            # Convert continuous to dictionary
                            cube_idx = int((action_continuous[0] + 1) / 2 * (args.num_cubes - 1))
                            grid_x = int((action_continuous[1] + 1) / 2 * (args.grid_size - 1))
                            grid_y = int((action_continuous[2] + 1) / 2 * (args.grid_size - 1))
                            action_dict = {'cube_idx': cube_idx, 'target_grid_x': grid_x, 'target_grid_y': grid_y}

                        # Convert dictionary action to integer action
                        action_int = env.reshuffle_action_space.encode_action(
                            cube_idx=action_dict['cube_idx'],
                            grid_x=action_dict['target_grid_x'],
                            grid_y=action_dict['target_grid_y']
                        )

                        # Take step
                        obs, reward, done, truncated, info = env.step(action_int)

                        cumulative_reward += reward
                        step += 1

                        # Log timestep data (as dictionary)
                        logger.log_timestep({
                            'episode': episode,
                            'step_in_episode': step,
                            'model': model_config["name"],
                            'seed': seed,
                            'reward': reward,
                            'cumulative_reward': cumulative_reward,
                            'reshuffled': info.get('reshuffled_this_step', False),
                            'distance_reduced': info.get('distance_reduced', 0.0),
                            'time_saved': info.get('time_saved', 0.0),
                            'cubes_picked_so_far': info.get('cubes_picked', 0),
                            'done': done,
                            'truncated': truncated
                        })

                    # Episode finished
                    episode_duration = time.time() - episode_start_time

                    # Check if episode timed out (truncated=True means timeout)
                    is_timeout = info.get('timeout', False) or truncated

                    # Calculate efficiency metrics (for all episodes including timeouts)
                    from gat_cvd_test_utils import calculate_efficiency_metrics

                    episode_data = {
                        'total_distance_reduced': info.get('total_distance_reduced', 0.0),
                        'total_distance_traveled': info.get('total_distance_traveled', 0.0),
                        'total_time_saved': info.get('total_time_saved', 0.0),
                        'total_time_taken': info.get('total_time_taken', 0.0),
                        'reshuffles_performed': info.get('reshuffles_performed', 0),
                        'cubes_picked': info.get('cubes_picked', 0),
                        'episode_length': step,
                        'total_cubes': args.num_cubes
                    }

                    efficiency_metrics = calculate_efficiency_metrics(episode_data)

                    # For Heuristic models, rewards don't make sense - set to None
                    is_heuristic = model_config["name"] == "Heuristic"
                    agent1_reward = None if is_heuristic else info.get('total_pick_reward', 0.0)  # Total Agent 1 reward
                    agent2_reward = None if is_heuristic else info.get('total_reshuffle_reward', 0.0)  # Total Agent 2 reward
                    total_reward = None if is_heuristic else cumulative_reward

                    # Log episode data (as dictionary) - LOG ALL EPISODES INCLUDING TIMEOUTS
                    logger.log_episode({
                        'episode': episode,  # 0-based internally, converted to 1-based in logger
                        'model': model_config["name"],
                        'seed': seed,
                        'agent1_reward': agent1_reward,
                        'success': info.get('success', False),
                        'cubes_picked': info.get('cubes_picked', 0),
                        'pick_failures': info.get('pick_failures', 0),
                        'successful_picks': info.get('successful_picks', 0),
                        'unreachable_cubes': info.get('unreachable_cubes', 0),
                        'path_efficiency': info.get('path_efficiency', 0.0),
                        'action_entropy': info.get('action_entropy', 0.0),
                        'agent2_reward': agent2_reward,
                        'reshuffles_performed': info.get('reshuffles_performed', 0),
                        'total_distance_reduced': info.get('total_distance_reduced', 0.0),
                        'total_time_saved': info.get('total_time_saved', 0.0),
                        'total_distance_traveled': info.get('total_distance_traveled', 0.0),
                        'total_time_taken': info.get('total_time_taken', 0.0),
                        'total_reward': total_reward,
                        'episode_length': step,
                        'duration': episode_duration,
                        'timeout': is_timeout,  # NEW: Flag for timeout episodes
                        'planner': "RRT",
                        'grid_size': args.grid_size,
                        'num_cubes': args.num_cubes,
                        # Add efficiency metrics (GAT+CVD specific)
                        'distance_efficiency': efficiency_metrics['distance_efficiency'],
                        'time_efficiency': efficiency_metrics['time_efficiency'],
                        'avg_distance_per_reshuffle': efficiency_metrics['avg_distance_per_reshuffle'],
                        'avg_time_per_reshuffle': efficiency_metrics['avg_time_per_reshuffle'],
                        'success_rate': efficiency_metrics['success_rate'],
                        'steps_per_cube': efficiency_metrics['steps_per_cube']
                    })

                    if is_timeout:
                        print(f"    ⚠️  TIMEOUT: Reward: {cumulative_reward:.2f}, Cubes: {info.get('cubes_picked', 0)}, Steps: {step}")
                    else:
                        print(f"    ✓ Reward: {cumulative_reward:.2f}, Cubes: {info.get('cubes_picked', 0)}, Steps: {step}")

                print(f"\n✅ Completed {args.episodes} episodes for {model_config['name']}")

            except Exception as e:
                print(f"\n❌ Error testing {model_config['name']}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Results are auto-saved by logger during log_episode and log_timestep
        print(f"\n💾 Results saved for seed {seed}")

    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print(f"Results saved to: {project_root / 'scripts/experiments/rlmodels/mutliagent/ddqn_gatcvd/gat_cvd_test_results'}")


if __name__ == "__main__":
    main()

