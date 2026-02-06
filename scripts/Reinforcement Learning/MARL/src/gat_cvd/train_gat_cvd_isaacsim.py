import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Train GAT+CVD multi-agent system with Isaac Sim RRT")
parser.add_argument("--timesteps", type=int, default=None,
                   help="Total training timesteps (auto-set based on grid/cubes if not specified)")
parser.add_argument("--grid_size", type=int, default=4,
                   help="Grid size (default: 4)")
parser.add_argument("--num_cubes", type=int, default=9,
                   help="Number of cubes (default: 9)")
parser.add_argument("--save_freq", type=int, default=5000,
                   help="Save checkpoint every N steps (default: 5000)")
parser.add_argument("--config", type=str, default="config_gat_cvd.yaml",
                   help="Path to config file (default: config_gat_cvd.yaml)")
parser.add_argument("--device", type=str, default="cuda",
                   help="Device to use (cuda/cpu, default: cuda)")
parser.add_argument("--use_wandb", action="store_true",
                   help="Use Weights & Biases for logging")
parser.add_argument("--execute_picks", action="store_true",
                   help="Execute actual pick-and-place during training")
parser.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
args = parser.parse_args()

# Create SimulationApp BEFORE importing any Isaac Sim modules
try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

import os
import time
import numpy as np
from datetime import datetime
import json
import yaml
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
from collections import deque
import gc

# Add MARL source to path
marl_src = Path(r"C:\isaacsim\cobotproject\scripts\Reinforcement Learning\MARL\src")
if str(marl_src) not in sys.path:
    sys.path.insert(0, str(marl_src))

from gat_cvd.gat_cvd_agent import GATCVDAgent
from gat_cvd.graph_utils import build_graph, compute_edge_features

from src.rl.object_selection_env_rrt import ObjectSelectionEnvRRT
from src.rl.doubleDQN import DoubleDQNAgent
mappo_path = Path(r"C:\isaacsim\cobotproject\scripts\Reinforcement Learning\MAPPO")
if str(mappo_path) not in sys.path:
    sys.path.insert(0, str(mappo_path))

masac_path = Path(r"C:\isaacsim\cobotproject\scripts\Reinforcement Learning\MASAC")
if str(masac_path) not in sys.path:
    sys.path.insert(0, str(masac_path))

from envs.two_agent_env import TwoAgentEnv
from agents.masac_continuous_wrapper import MASACContinuousWrapper


def clear_console():
    """Clear console output to prevent memory buildup from terminal history"""
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')


class FrankaRRTTrainer:
    """
    Franka controller for RRT-based GAT+CVD training.
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
        self.obstacles = []

        print(f"[TRAINER] Initializing Franka RRT Trainer for GAT+CVD")
        print(f"[TRAINER] Grid: {training_grid_size}x{training_grid_size}, Cubes: {num_cubes}")

    def setup_scene(self):
        """Setup Isaac Sim scene with Franka and cubes"""
        print("[TRAINER] Setting up scene...")

        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        assets_root_path = get_assets_root_path()
        franka_prim_path = "/World/Franka"

        franka_usd_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        robot_prim = add_reference_to_stage(usd_path=franka_usd_path, prim_path=franka_prim_path)
        robot_prim.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
        robot_prim.GetVariantSet("Mesh").SetVariantSelection("Quality")
        self.gripper = ParallelGripper(
            end_effector_prim_path=f"{franka_prim_path}/panda_rightfinger",
            joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
            joint_opened_positions=np.array([0.04, 0.04]),
            joint_closed_positions=np.array([0.0, 0.0]),
            action_deltas=np.array([0.01, 0.01])
        )
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
        print("[TRAINER] Initializing RRT planner...")
        self._setup_rrt_planner()

        print("[TRAINER] Adding container...")
        self._setup_container()

   
        self._spawn_cubes()
        print("[TRAINER] Creating random obstacles in empty cells...")
        self._create_random_obstacles()

    
        self.world.reset()

        print("[TRAINER] Scene setup complete")

    def _setup_rrt_planner(self):
        """Setup RRT path planner"""
        try:
            mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")

            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Navigate to cobotproject root: gat_cvd -> src -> MARL -> Reinforcement Learning -> scripts -> cobotproject
            cobotproject_root = os.path.join(script_dir, "..", "..", "..", "..", "..")
            robot_description_file = os.path.join(cobotproject_root, "assets", "franka_conservative_spheres_robot_description.yaml")
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

    def get_obstacle_positions(self):
        """Get current obstacle positions"""
        positions = []
        for obstacle in self.obstacles:
            pos, _ = obstacle.get_world_pose()
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
    if args.timesteps is None:
        if args.grid_size == 3 and args.num_cubes == 4:
            args.timesteps = 50000  # GAT+CVD with reshuffling needs more steps
        elif args.grid_size == 4 and args.num_cubes == 6:
            args.timesteps = 75000
        elif args.grid_size == 4 and args.num_cubes == 9:
            args.timesteps = 100000  # Harder task with reshuffling
        else:
            args.timesteps = 100000  # Default
        print(f"Auto-set timesteps to {args.timesteps} based on grid_size={args.grid_size}, num_cubes={args.num_cubes}")

    config_path = os.path.join(os.path.dirname(__file__), args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['device'] = args.device

    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project="gat-cvd-object-selection",
                name=f"gat_cvd_isaacsim_grid{args.grid_size}_cubes{args.num_cubes}",
                config={
                    "method": "gat_cvd_isaacsim_reshuffling",
                    "grid_size": args.grid_size,
                    "num_cubes": args.num_cubes,
                    "timesteps": args.timesteps,
                    "execute_picks": args.execute_picks,
                    **config
                }
            )
            print("W&B logging enabled")
        except ImportError:
            print("ERROR: wandb not installed. Install with: C:\\isaacsim\\python.bat -m pip install wandb")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: W&B initialization failed: {e}")
            sys.exit(1)

    print("=" * 60)
    print("GAT+CVD MULTI-AGENT TRAINING - ISAAC SIM (WITH RESHUFFLING)")
    print("=" * 60)
    print(f"Timesteps: {args.timesteps}")
    print(f"Grid: {args.grid_size}x{args.grid_size}")
    print(f"Cubes: {args.num_cubes}")
    print(f"Execute picks: {args.execute_picks}")
    print(f"Device: {args.device}")
    print("=" * 60)

    trainer = FrankaRRTTrainer(
        num_cubes=args.num_cubes,
        training_grid_size=args.grid_size
    )

    trainer.setup_scene()

    print("\n[TRAINER] Creating RRT-based RL environment...")
    max_objects = args.grid_size * args.grid_size

    base_env = ObjectSelectionEnvRRT(
        franka_controller=trainer,
        max_objects=max_objects,
        max_steps=50,
        num_cubes=args.num_cubes,
        training_grid_size=args.grid_size,
        execute_picks=False,  # Don't execute picks during training (only reachability checks)
        rrt_planner=trainer.rrt,
        kinematics_solver=trainer.kinematics_solver,
        articulation_kinematics_solver=trainer.articulation_kinematics_solver,
        franka_articulation=trainer.franka
    )

    config['node_dim'] = 7  # [x, y, z, is_robot, is_obstacle, is_target, object_id]
    config['edge_dim'] = 3  # [distance, reachability, blocking_score]
    config['n_actions_ddqn'] = max_objects
    config['n_actions_masac'] = args.num_cubes * args.grid_size * args.grid_size

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    agent = GATCVDAgent(config, device=device)

    print(f"\n[AGENT] GAT+CVD Agent created:")
    print(f"  - DDQN action space: {config['n_actions_ddqn']} (object selection)")
    print(f"  - MASAC action space: {config['n_actions_masac']} (reshuffling: {args.num_cubes} cubes × {args.grid_size}×{args.grid_size} grid)")
    print(f"  - Device: {device}")
    print(f"  - Training from scratch (no pretrained DDQN)")

    class DDQNAgentWrapper:
        """Wrapper to make GATCVDAgent's DDQN component compatible with TwoAgentEnv"""
        def __init__(self, gat_cvd_agent, device):
            self.gat_cvd_agent = gat_cvd_agent
            self.device = device
            self.epsilon = 0.1  # Will be updated during training

        def select_action(self, obs, action_mask):
            """Select action using DDQN from GATCVDAgent"""
            # Build graph from observation
            # Note: TwoAgentEnv calls this with base_env observation
            # We need to convert it to graph format
            robot_pos, _ = trainer.franka.get_world_pose()
            robot_positions = [robot_pos]
            object_positions = trainer.get_cube_positions()
            obstacle_positions = trainer.get_obstacle_positions()

            graph = build_graph(
                obs=obs,
                robot_positions=robot_positions,
                object_positions=object_positions,
                obstacles=obstacle_positions,
                edge_threshold=config['graph']['edge_threshold'],
                device=self.device
            )

        
            action, _ = self.gat_cvd_agent.select_actions(
                graph,
                epsilon_ddqn=self.epsilon,
                epsilon_masac=0.0,  # Not used here
                action_mask=action_mask
            )
            return action


    ddqn_wrapper = DDQNAgentWrapper(agent, device)
    print(f" Created DDQN wrapper for TwoAgentEnv (uses GAT+CVD's DDQN)")


    print("\n[TRAINER] Creating TwoAgentEnv wrapper for reshuffling...")
    env = TwoAgentEnv(
        base_env=base_env,
        ddqn_agent=ddqn_wrapper,  # Use wrapper that connects to GATCVDAgent's DDQN
        grid_size=args.grid_size,
        num_cubes=args.num_cubes,
        max_reshuffles_per_episode=5,
        reshuffle_reward_scale=1.0,
        max_episode_steps=50,
        verbose=False
    )

    print("[TRAINER] Configuring reshuffling thresholds...")
    env.reshuffle_decision.min_reachable_distance = 0.30
    env.reshuffle_decision.max_reachable_distance = 0.90
    env.reshuffle_decision.path_length_ratio_threshold = 1.5
    env.reshuffle_decision.crowded_threshold = 2
    env.reshuffle_decision.rrt_failure_window = 2
    env.reshuffle_decision.min_clearance = 0.35
    env.reshuffle_decision.far_cube_ratio = 1.1
    env.reshuffle_decision.batch_reshuffle_count = 2
    print("[TRAINER] Reshuffling thresholds configured!")

    print(f" TwoAgentEnv created with reshuffling support!")

    total_steps = 0
    episode = 1  # Start from episode 1, not 0
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"ERROR: Checkpoint not found: {args.resume}")
            simulation_app.close()
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"RESUMING FROM CHECKPOINT")
        print(f"{'='*60}")
        print(f"Loading: {args.resume}")
        agent.load(args.resume)
        total_steps = agent.total_steps
        episode = agent.episodes
        print(f"Resuming from step {total_steps}, episode {episode}")
        print(f"{'='*60}\n")

        checkpoint_name = os.path.basename(args.resume)
        run_name = '_'.join(checkpoint_name.split('_')[:-2])
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"gat_cvd_isaacsim_grid{args.grid_size}_cubes{args.num_cubes}_{timestamp}"

    log_dir = r"C:\isaacsim\cobotproject\scripts\Reinforcement Learning\MARL\src\gat_cvd\logs"
    model_dir = r"C:\isaacsim\cobotproject\scripts\Reinforcement Learning\MARL\src\gat_cvd\models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{run_name}_training.csv")
    episode_log_file = os.path.join(log_dir, f"{run_name}_episodes.csv")

    if args.resume:
        if not os.path.exists(log_file):
            print(f"WARNING: Training log not found, creating new: {log_file}")
            with open(log_file, 'w') as f:
                f.write("step,episode,loss_ddqn,loss_masac,loss_cvd,q_value_ddqn,q_overestimation_ddqn,epsilon_ddqn,epsilon_masac,reward_total,reward_agent1,reward_agent2,episode_reward,episode_length,episode_reshuffles,cubes_picked,avg_reward_100,success_rate\n")
        if not os.path.exists(episode_log_file):
            print(f"WARNING: Episode log not found, creating new: {episode_log_file}")
            with open(episode_log_file, 'w') as f:
                f.write("episode,total_reward,reward_agent1_total,reward_agent2_total,length,cubes_picked,success,reshuffles,distance_reduced,time_saved,avg_q_value,avg_q_overestimation,avg_reward_100,success_rate_100\n")
    else:
        with open(log_file, 'w') as f:
            f.write("step,episode,loss_ddqn,loss_masac,loss_cvd,q_value_ddqn,q_overestimation_ddqn,epsilon_ddqn,epsilon_masac,reward_total,reward_agent1,reward_agent2,episode_reward,episode_length,episode_reshuffles,cubes_picked,avg_reward_100,success_rate\n")

        with open(episode_log_file, 'w') as f:
            f.write("episode,total_reward,reward_agent1_total,reward_agent2_total,length,cubes_picked,success,reshuffles,distance_reduced,time_saved,avg_q_value,avg_q_overestimation,avg_reward_100,success_rate_100\n")

    print("\nStarting training...")
    print(f"Checkpoints will be saved to: {model_dir}/{run_name}")
    print(f"Training log will be saved to: {log_file}")
    print(f"Episode log will be saved to: {episode_log_file}")
    print("=" * 60 + "\n")

  
    episode_rewards = deque(maxlen=1000)
    episode_lengths = deque(maxlen=1000)
    episode_successes = deque(maxlen=1000)
    epsilon_ddqn = config['training']['epsilon_start_ddqn']
    epsilon_masac = config['training']['epsilon_start_masac']
    epsilon_end_ddqn = config['training']['epsilon_end_ddqn']
    epsilon_end_masac = config['training']['epsilon_end_masac']
    epsilon_decay_ddqn = config['training']['epsilon_decay_ddqn']
    epsilon_decay_masac = config['training']['epsilon_decay_masac']

    while total_steps < args.timesteps:
        state, info = env.reset()
        episode_reward = 0
        episode_reward_agent1 = 0  # Agent 1 (DDQN) reward
        episode_reward_agent2 = 0  # Agent 2 (MASAC) reward
        episode_length = 0
        episode_reshuffles = 0
        episode_q_values = []  # Track Q-values for this episode
        episode_returns = []  # Track actual returns for Q-overestimation
        done = False

        while not done and total_steps < args.timesteps:
            robot_pos, _ = trainer.franka.get_world_pose()
            robot_positions = [robot_pos]
            object_positions = trainer.get_cube_positions()
            obstacle_positions = trainer.get_obstacle_positions()
            graph = build_graph(
                obs=state,
                robot_positions=robot_positions,
                object_positions=object_positions,
                obstacles=obstacle_positions,
                edge_threshold=config['graph']['edge_threshold'],
                device=device
            )

            action_mask = info.get('action_mask', env.base_env.action_masks())
            valid_cubes = [
                i for i in range(args.num_cubes)
                if i not in env.base_env.objects_picked
                and env.reshuffle_count_per_cube.get(i, 0) < 2
            ]

            action_ddqn, action_masac = agent.select_actions(
                graph,
                epsilon_ddqn=epsilon_ddqn,
                epsilon_masac=epsilon_masac,
                action_mask=action_mask
            )

            with torch.no_grad():
                q_values = agent.ddqn_policy(graph)  # DQNNetworkGAT expects graph object
                q_value_selected = q_values[0, action_ddqn].item()
                episode_q_values.append(q_value_selected)

            ddqn_wrapper.epsilon = epsilon_ddqn

            next_state, reward, terminated, truncated, info = env.step(action_masac)
            done = terminated or truncated

        
            reward_agent1 = info.get('pick_reward', 0.0)  # Agent 1 (DDQN) reward
            reward_agent2 = info.get('reshuffle_reward', 0.0)  # Agent 2 (MASAC) reward
            cubes_picked_now = info.get('cubes_picked', 0)


            episode_reward_agent1 += reward_agent1
            episode_reward_agent2 += reward_agent2

        
            if info.get('reshuffled_this_step', False):
                episode_reshuffles += 1
            next_action_mask = info.get('action_mask', env.base_env.action_masks())

        
            next_robot_pos, _ = trainer.franka.get_world_pose()
            next_robot_positions = [next_robot_pos]
            next_object_positions = trainer.get_cube_positions()
            next_obstacle_positions = trainer.get_obstacle_positions()

            next_graph = build_graph(
                obs=next_state,
                robot_positions=next_robot_positions,
                object_positions=next_object_positions,
                obstacles=next_obstacle_positions,
                edge_threshold=config['graph']['edge_threshold'],
                device=device
            )

            agent.store_transition(
                graph, action_ddqn, action_masac, reward, next_graph, done,
                action_mask, next_action_mask
            )

            loss_ddqn = agent.train_step_ddqn()
            loss_masac_critic, loss_masac_actor, loss_masac_alpha = agent.train_step_masac()
            loss_cvd = agent.train_step_cvd()
            if loss_masac_critic is not None:
                loss_masac = loss_masac_critic + loss_masac_actor + loss_masac_alpha
            else:
                loss_masac = None

            if agent.can_train():
                agent.soft_update_targets()

            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            episode_returns.append(reward)
            epsilon_ddqn = max(epsilon_end_ddqn, epsilon_ddqn * epsilon_decay_ddqn)
            epsilon_masac = max(epsilon_end_masac, epsilon_masac * epsilon_decay_masac)
            avg_reward_100 = np.mean(list(episode_rewards)[-100:]) if episode_rewards else 0.0
            success_rate = np.mean(list(episode_successes)[-100:]) if episode_successes else 0.0
            q_overestimation = q_value_selected - reward if len(episode_q_values) > 0 else 0.0
            cubes_picked_current = info.get('cubes_picked', 0)
            with open(log_file, 'a') as f:
                loss_ddqn_val = f"{loss_ddqn:.6f}" if loss_ddqn is not None else ""
                loss_masac_val = f"{loss_masac:.6f}" if loss_masac is not None else ""
                loss_cvd_val = f"{loss_cvd:.6f}" if loss_cvd is not None else ""
                f.write(f"{total_steps},{episode},{loss_ddqn_val},{loss_masac_val},{loss_cvd_val},"
                       f"{q_value_selected:.6f},{q_overestimation:.6f},"
                       f"{epsilon_ddqn:.6f},{epsilon_masac:.6f},"
                       f"{reward:.6f},{reward_agent1:.6f},{reward_agent2:.6f},"
                       f"{episode_reward:.6f},{episode_length},{episode_reshuffles},{cubes_picked_current},"
                       f"{avg_reward_100:.6f},{success_rate:.6f}\n")

            if args.use_wandb:
                import wandb
                wandb.log({
                    "global_step": total_steps,
                    "training/loss_ddqn": loss_ddqn if loss_ddqn is not None else 0.0,
                    "training/loss_masac": loss_masac if loss_masac is not None else 0.0,
                    "training/loss_cvd": loss_cvd if loss_cvd is not None else 0.0,
                    "training/q_value_ddqn": q_value_selected,
                    "training/q_overestimation_ddqn": q_overestimation,
                    "training/epsilon_ddqn": epsilon_ddqn,
                    "training/epsilon_masac": epsilon_masac,
                    "training/reward_total": reward,
                    "training/reward_agent1": reward_agent1,
                    "training/reward_agent2": reward_agent2,
                    "training/episode_reward_running": episode_reward,
                    "training/episode_length_running": episode_length,
                    "training/episode_reshuffles_running": episode_reshuffles,
                    "training/cubes_picked": cubes_picked_current,
                    "training/total_distance_reduced": env.total_distance_reduced,
                    "training/total_time_saved": env.total_time_saved,
                })

            if total_steps % 1000 == 0:
                if total_steps % 5000 == 0:
                    clear_console()
                    print("=" * 60)
                    print("GAT+CVD TRAINING IN PROGRESS")
                    print("=" * 60)

                loss_ddqn_str = f"{loss_ddqn:.4f}" if loss_ddqn else "0.0000"
                loss_masac_str = f"{loss_masac:.4f}" if loss_masac else "0.0000"
                loss_cvd_str = f"{loss_cvd:.4f}" if loss_cvd else "0.0000"
                print(f"Steps: {total_steps}/{args.timesteps} | "
                      f"Episode: {episode} | "
                      f"Avg Reward (100 ep): {avg_reward_100:.2f} | "
                      f"Epsilon DDQN: {epsilon_ddqn:.4f} | "
                      f"Loss DDQN: {loss_ddqn_str} | "
                      f"Loss MASAC: {loss_masac_str} | "
                      f"Loss CVD: {loss_cvd_str}")

            if total_steps % args.save_freq == 0:
                checkpoint_path = os.path.join(model_dir, f"{run_name}_step_{total_steps}.pt")
                agent.save(checkpoint_path)

        cubes_picked_final = len(env.base_env.objects_picked)
        episode_success = cubes_picked_final / env.num_cubes
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_successes.append(episode_success)
        avg_reward_100 = np.mean(list(episode_rewards)[-100:])
        success_rate_100 = np.mean(list(episode_successes)[-100:])
        avg_q_value = np.mean(episode_q_values) if episode_q_values else 0.0
        # Calculate Q-overestimation: avg(Q-values) - avg(actual returns)
        avg_return = np.mean(episode_returns) if episode_returns else 0.0
        avg_q_overestimation = avg_q_value - avg_return
        with open(episode_log_file, 'a') as f:
            f.write(f"{episode},{episode_reward:.6f},{episode_reward_agent1:.6f},{episode_reward_agent2:.6f},"
                   f"{episode_length},{cubes_picked_final},{episode_success:.6f},"
                   f"{episode_reshuffles},{env.total_distance_reduced:.6f},{env.total_time_saved:.6f},"
                   f"{avg_q_value:.6f},{avg_q_overestimation:.6f},"
                   f"{avg_reward_100:.6f},{success_rate_100:.6f}\n")

        if args.use_wandb:
            import wandb
            wandb.log({
                "global_step": total_steps,
                "episode/total_reward": episode_reward,
                "episode/reward_agent1": episode_reward_agent1,
                "episode/reward_agent2": episode_reward_agent2,
                "episode/total_length": episode_length,
                "episode/cubes_picked": cubes_picked_final,
                "episode/success": episode_success,
                "episode/reshuffles": episode_reshuffles,
                "episode/distance_reduced": env.total_distance_reduced,
                "episode/time_saved": env.total_time_saved,
                "episode/avg_q_value": avg_q_value,
                "episode/avg_q_overestimation": avg_q_overestimation,
                "episode/avg_reward_100": avg_reward_100,
                "episode/success_rate_100": success_rate_100
            })

        agent.episodes += 1
        episode += 1

        # Periodic garbage collection
        if episode % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    final_path = os.path.join(model_dir, f"{run_name}_final.pt")
    agent.save(final_path)

    metadata = {
        "method": "gat_cvd_isaacsim",
        "algorithm": "gat_cvd",
        "training_grid_size": args.grid_size,
        "num_cubes": args.num_cubes,
        "max_objects": max_objects,
        "max_steps": 50,
        "execute_picks": args.execute_picks,
        "timestamp": run_name.split('_')[-2] + '_' + run_name.split('_')[-1] if not args.resume else "resumed",
        "total_timesteps": args.timesteps,
        "config": config,
        "total_episodes": episode,
        "avg_reward_100": np.mean(list(episode_rewards)[-100:]) if episode_rewards else 0.0,
        "success_rate_100": np.mean(list(episode_successes)[-100:]) if episode_successes else 0.0
    }
    metadata_path = os.path.join(model_dir, f"{run_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total episodes: {episode}")
    print(f"Final model saved to: {final_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Average reward (last 100 ep): {metadata['avg_reward_100']:.2f}")
    print(f"Success rate (last 100 ep): {metadata['success_rate_100']:.2%}")
    print("=" * 60)

    if args.use_wandb:
        import wandb
        wandb.finish()

    simulation_app.close()
    sys.exit(0)


if __name__ == "__main__":
    main()

