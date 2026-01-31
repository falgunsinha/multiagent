"""
Train MAPPO Two-Agent System with Isaac Sim RRT Environment

This script trains the MAPPO reshuffling agent (Agent 2) with a frozen DDQN pick agent (Agent 1)
using the actual Isaac Sim RRT environment.

Usage:
    C:\isaacsim\python.bat train_isaacsim_mappo.py --timesteps 50000 --grid_size 4 --num_cubes 9
"""

import argparse
import sys
from pathlib import Path

# Parse command-line arguments BEFORE importing Isaac Sim
parser = argparse.ArgumentParser(description="Train MAPPO two-agent system with Isaac Sim RRT")
parser.add_argument("--timesteps", type=int, default=20000,
                   help="Total training timesteps (default: 20000)")
parser.add_argument("--grid_size", type=int, default=4,
                   help="Grid size (default: 4)")
parser.add_argument("--num_cubes", type=int, default=9,
                   help="Number of cubes (default: 9)")
parser.add_argument("--save_freq", type=int, default=5000,
                   help="Save checkpoint every N steps (default: 5000)")
parser.add_argument("--run_name", type=str, default=None,
                   help="WandB run name (auto-generated if None)")
parser.add_argument("--config_name", type=str, default=None,
                   help="Configuration name for grouping runs")
parser.add_argument("--use_wandb", action="store_true",
                   help="Use Weights & Biases for logging")
parser.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
parser.add_argument("--ddqn_model_path", type=str, default=None,
                   help="Path to pre-trained DDQN model")

# MAPPO hyperparameters
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
parser.add_argument("--clip_param", type=float, default=0.2, help="PPO clip parameter")
parser.add_argument("--value_loss_coef", type=float, default=1.0, help="Value loss coefficient")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm")
parser.add_argument("--buffer_size", type=int, default=256, help="Rollout buffer size (reduced from 2048 to prevent slowdown)")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size (reduced to match buffer size)")
parser.add_argument("--ppo_epochs", type=int, default=10, help="PPO epochs per update")
parser.add_argument("--num_mini_batch", type=int, default=4, help="Number of mini-batches")

# Logging and saving
parser.add_argument("--log_interval", type=int, default=10, help="Log every N episodes")
parser.add_argument("--save_interval", type=int, default=100, help="Save checkpoint every N episodes")

# Isaac Sim specific
parser.add_argument("--execute_picks", action="store_true", help="Execute actual pick-and-place during training")

args = parser.parse_args()

# Create SimulationApp BEFORE importing any Isaac Sim modules
try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})  # Console-only mode for Isaac Sim

import os
import time
import numpy as np
from datetime import datetime
import json
import torch

# Isaac Sim imports
import omni.timeline
import omni.usd
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

# Add project root to path (must be done after SimulationApp is created)
project_root = Path(r"C:\isaacsim\cobotproject")
mappo_root = project_root / "scripts" / "Reinforcement Learning" / "MAPPO"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import DDQN agent and environment (absolute imports from project root)
from src.rl.doubleDQN import DoubleDQNAgent
from src.rl.object_selection_env_rrt import ObjectSelectionEnvRRT
from src.manipulators import SingleManipulator
from src.grippers import ParallelGripper

# Import MAPPO modules using absolute imports from project root
# This avoids relative import issues with Isaac Sim's Python environment
import importlib.util
import sys as sys_module

# Helper function to import module from file path
def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys_module.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import MAPPO modules
MAPPOPolicy = import_module_from_path(
    "mappo_policy",
    mappo_root / "algorithms" / "mappo_policy.py"
).MAPPOPolicy
MAPPO = import_module_from_path(
    "mappo_trainer",
    mappo_root / "algorithms" / "mappo_trainer.py"
).MAPPO
TwoAgentEnv = import_module_from_path(
    "two_agent_env",
    mappo_root / "envs" / "two_agent_env.py"
).TwoAgentEnv
RolloutBuffer = import_module_from_path(
    "replay_buffer",
    mappo_root / "utils" / "replay_buffer.py"
).RolloutBuffer
WandBLogger = import_module_from_path(
    "wandb_config",
    mappo_root / "utils" / "wandb_config.py"
).WandBLogger
DetailedLogger = import_module_from_path(
    "detailed_logger",
    mappo_root / "utils" / "detailed_logger.py"
).DetailedLogger


class FrankaRRTTrainer:
    """
    Franka controller for RRT-based MAPPO training.
    Adapted from train_rrt_isaacsim_ddqn.py
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
        self.obstacles = []  # Track obstacles for re-randomization

        print(f"[TRAINER] Initializing Franka RRT Trainer for MAPPO")
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

    def _create_random_obstacles(self):
        """Create random obstacles in empty grid cells"""
        num_obstacles_map = {
            3: 1,
            4: 2,
            6: np.random.randint(3, 6),
        }
        num_obstacles = num_obstacles_map.get(self.training_grid_size, max(1, self.training_grid_size // 3))

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
            self.obstacles.append(obstacle)

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
        """Re-randomize cube positions without recreating them"""
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

        cube_indices = list(range(len(self.cubes)))
        np.random.shuffle(cube_indices)

        positions = []
        for cell_idx in sorted(selected_indices):
            row = cell_idx // self.training_grid_size
            col = cell_idx % self.training_grid_size

            base_x = start_x + (row * cube_spacing)
            base_y = start_y + (col * cube_spacing)

            offset_x = np.random.uniform(-0.02, 0.02)
            offset_y = np.random.uniform(-0.02, 0.02)

            position = np.array([base_x + offset_x, base_y + offset_y, cube_size / 2.0])
            positions.append(position)

        for i, cube_idx in enumerate(cube_indices):
            cube, cube_name = self.cubes[cube_idx]
            cube.set_world_pose(position=positions[i])
            self.cube_positions[cube_idx] = positions[i]

    def randomize_obstacle_positions(self):
        """Re-randomize obstacle positions without recreating them"""
        num_obstacles_map = {
            3: 1,
            4: 2,
            6: np.random.randint(3, 6),
        }
        num_obstacles = num_obstacles_map.get(self.training_grid_size, max(1, self.training_grid_size // 3))

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

        if num_obstacles == 0 or len(self.obstacles) == 0:
            return

        np.random.shuffle(empty_cells)
        selected_cells = empty_cells[:num_obstacles]

        for idx, (grid_x, grid_y) in enumerate(selected_cells):
            if idx >= len(self.obstacles):
                break

            world_pos = self._grid_to_world(grid_x, grid_y)
            obs_position = np.array([world_pos[0], world_pos[1], 0.055])
            self.obstacles[idx].set_world_pose(position=obs_position)


def main():
    """Main training loop"""
    print("=" * 80)
    print("MAPPO Two-Agent Training with Isaac Sim RRT")
    print("=" * 80)
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"Number of cubes: {args.num_cubes}")
    print(f"Total timesteps: {args.timesteps}")
    print(f"Learning rate: {args.lr}")
    print(f"Buffer size: {args.buffer_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"PPO epochs: {args.ppo_epochs}")
    print("=" * 80)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories (shared across all environment types, like DDQN)
    models_dir = mappo_root / "models"
    logs_dir = mappo_root / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Initialize WandB
    wandb_logger = None
    # Model name includes 'mappo_' prefix for file naming
    # WandB run name matches DDQN exactly (no 'mappo_' prefix) for easy comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"mappo_rrt_isaacsim_grid{args.grid_size}_cubes{args.num_cubes}_{timestamp}"
    wandb_run_name = f"rrt_isaacsim_grid{args.grid_size}_cubes{args.num_cubes}_{timestamp}"

    if args.use_wandb:
        run_name = args.run_name or wandb_run_name
        config_name = args.config_name or f"grid{args.grid_size}x{args.grid_size}_{args.num_cubes}cubes"

        wandb_logger = WandBLogger(
            project_name="ddqn-mappo-object-selection-reshuffling",
            run_name=run_name,
            group=config_name,
            config={
                "environment": "rrt_isaacsim",
                "grid_size": args.grid_size,
                "num_cubes": args.num_cubes,
                "timesteps": args.timesteps,
                "lr": args.lr,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "clip_param": args.clip_param,
                "value_loss_coef": args.value_loss_coef,
                "entropy_coef": args.entropy_coef,
                "max_grad_norm": args.max_grad_norm,
                "buffer_size": args.buffer_size,
                "batch_size": args.batch_size,
                "ppo_epochs": args.ppo_epochs,
                "num_mini_batch": args.num_mini_batch,
            },
            tags=["mappo", "rrt_isaacsim", f"grid{args.grid_size}", f"cubes{args.num_cubes}"],
            notes=f"MAPPO two-agent training with Isaac Sim RRT environment on {args.grid_size}x{args.grid_size} grid with {args.num_cubes} cubes",
        )
        print(f"WandB initialized: {run_name}")

    # Initialize detailed logger
    print("Initializing detailed logger...")
    # Use shared log directory (like DDQN), file names differentiate environment types
    detailed_logger = DetailedLogger(
        log_dir=logs_dir,
        run_name=model_name,  # Use model_name for log files (includes 'mappo_' prefix)
        wandb_logger=wandb_logger,
    )

    # Create FrankaRRTTrainer (Isaac Sim scene with robot and RRT planner)
    print("\nCreating Franka RRT Trainer...")
    franka_trainer = FrankaRRTTrainer(
        num_cubes=args.num_cubes,
        training_grid_size=args.grid_size
    )

    # Setup scene
    franka_trainer.setup_scene()

    # Create Isaac Sim RRT environment
    print("\nCreating Isaac Sim RRT environment...")
    max_objects = args.grid_size * args.grid_size

    base_env = ObjectSelectionEnvRRT(
        franka_controller=franka_trainer,
        max_objects=max_objects,
        max_steps=50,
        num_cubes=args.num_cubes,
        training_grid_size=args.grid_size,
        execute_picks=args.execute_picks,
        rrt_planner=franka_trainer.rrt,
        kinematics_solver=franka_trainer.kinematics_solver,
        articulation_kinematics_solver=franka_trainer.articulation_kinematics_solver,
        franka_articulation=franka_trainer.franka
    )
    print("Isaac Sim environment created with Franka controller and RRT planner")

    # Load frozen DDQN agent (Agent 1)
    print("\nLoading frozen DDQN agent (Agent 1)...")

    # Use provided model path or default
    if args.ddqn_model_path:
        ddqn_model_path = Path(args.ddqn_model_path)
    else:
        ddqn_model_path = project_root / "scripts" / "Reinforcement Learning" / "doubleDQN_script" / "models" / f"ddqn_rrt_isaacsim_grid{args.grid_size}_cubes{args.num_cubes}_final.pt"

    if not ddqn_model_path.exists():
        print(f"ERROR: DDQN model not found at {ddqn_model_path}")
        print("Please provide a valid DDQN model path using --ddqn_model_path")
        simulation_app.close()
        sys.exit(1)

    # Create agent with correct dimensions
    state_dim = max_objects * 6
    action_dim = max_objects

    agent1 = DoubleDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )
    agent1.load(str(ddqn_model_path))

    # Freeze DDQN agent (set to evaluation mode)
    agent1.epsilon = 0.0  # No exploration
    agent1.policy_net.eval()
    agent1.target_net.eval()
    for param in agent1.policy_net.parameters():
        param.requires_grad = False
    for param in agent1.target_net.parameters():
        param.requires_grad = False
    print(f"DDQN agent loaded from {ddqn_model_path} and frozen")

    # Create two-agent environment wrapper
    print("\nCreating two-agent environment...")
    env = TwoAgentEnv(
        base_env=base_env,
        ddqn_agent=agent1,
        grid_size=args.grid_size,
        num_cubes=args.num_cubes,
    )
    print("Two-agent environment created")

    # Create MAPPO policy (Agent 2)
    print("\nCreating MAPPO policy (Agent 2)...")
    obs_dim = env.agent2_obs_dim
    action_dim = env.agent2_action_dim

    policy = MAPPOPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
        lr=args.lr,
        device=device,
    )
    print(f"MAPPO policy created: obs_dim={obs_dim}, action_dim={action_dim}")

    # Create MAPPO trainer
    mappo_trainer = MAPPO(
        policy=policy,
        device=device,
        clip_param=args.clip_param,
        ppo_epoch=args.ppo_epochs,
        num_mini_batch=args.num_mini_batch,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
    )
    print("MAPPO trainer created")

    # Create rollout buffer
    buffer = RolloutBuffer(
        buffer_size=args.buffer_size,
        obs_dim=env.agent2_obs_dim,
        action_dim=env.agent2_action_dim,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        device=device,
    )
    print(f"Rollout buffer created: size={args.buffer_size}")

    print("\nInitialization complete. Starting training...")
    print("=" * 80)

    # Training loop
    total_steps = 0
    episode = 0

    while total_steps < args.timesteps:
        episode += 1
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done and total_steps < args.timesteps:
            # Update observation normalizer statistics
            policy.update_obs_normalizer(obs)

            # Get action mask
            action_mask = env.get_agent2_action_mask()

            # Get action from Agent 2 (MAPPO)
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action_mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(device)
                action, log_prob, value = policy.get_actions(obs_tensor, action_mask_tensor)
                action = action.cpu().item()
                log_prob = log_prob.cpu().item()
                value = value.cpu().item()

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            buffer.add(obs, action, reward, value, log_prob, done, action_mask)

            obs = next_obs
            episode_reward += reward
            episode_length += 1
            total_steps += 1

            # Update policy when buffer is full
            if buffer.ptr >= args.buffer_size:
                # Compute returns and advantages
                with torch.no_grad():
                    next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(device)
                    next_value = policy.critic(next_obs_tensor).cpu().item()
                buffer.finish_path(last_value=next_value)

                # Update policy
                train_info = mappo_trainer.train(buffer)

                # Log training metrics
                if wandb_logger:
                    # Log MAPPO-specific metrics
                    wandb_logger.log_mappo_metrics(train_info, total_steps)

                    # Also log as general training metrics (matches DDQN format)
                    wandb_logger.log_training_metrics({
                        "loss": train_info.get("value_loss", 0) + train_info.get("policy_loss", 0),
                        "value_loss": train_info.get("value_loss", 0),
                        "policy_loss": train_info.get("policy_loss", 0),
                    }, total_steps)

                # Clear buffer
                buffer.clear()

                print(f"Step {total_steps}/{args.timesteps} | Episode {episode} | "
                      f"Reward: {episode_reward:.2f} | Length: {episode_length} | "
                      f"Policy Loss: {train_info['policy_loss']:.4f} | "
                      f"Value Loss: {train_info['value_loss']:.4f}")

        # Log episode metrics
        # Get cube distances for logging
        cube_positions = env.base_env.get_cube_positions()
        robot_pos = env.base_env.get_robot_position()
        cube_distances = {
            i: float(np.linalg.norm(cube_positions[i] - robot_pos))
            for i in range(len(cube_positions))
        }

        # Log detailed episode data (including distance/time metrics)
        detailed_logger.log_episode(
            episode=episode,
            total_reward=episode_reward,
            episode_length=episode_length,
            reshuffles_performed=info.get("reshuffles_performed", 0),
            reshuffle_details=info.get("reshuffle_history", []),
            cube_distances=cube_distances,
            total_distance_reduced=info.get("total_distance_reduced", 0.0),  # NEW: Log to CSV
            total_time_saved=info.get("total_time_saved", 0.0),  # NEW: Log to CSV
        )

        # Log system metrics to WandB (matches DDQN format)
        if wandb_logger:
            wandb_logger.log_system_metrics({
                "total_reward": episode_reward,
                "episode_length": episode_length,
                "cubes_picked": info.get("cubes_picked", 0),
                "reshuffles_performed": info.get("reshuffles_performed", 0),
                "distance_reduced": info.get("total_distance_reduced", 0.0),  # NEW: Distance improvement
                "time_saved": info.get("total_time_saved", 0.0),  # NEW: Time saved
            }, total_steps)

        # Save checkpoint (use DDQN naming convention)
        if episode % (args.save_freq // 100) == 0:  # Approximate episodes
            checkpoint_path = models_dir / f"{model_name}_step_{total_steps}.pt"
            torch.save({
                'step': total_steps,
                'episode': episode,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': policy.optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model (use DDQN naming convention)
    final_model_path = models_dir / f"{model_name}_final.pt"
    torch.save({
        'step': total_steps,
        'episode': episode,
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': policy.optimizer.state_dict(),
    }, final_model_path)
    print(f"\nTraining complete! Final model saved: {final_model_path}")

    # Close loggers
    detailed_logger.close()
    if wandb_logger:
        wandb_logger.finish()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()

