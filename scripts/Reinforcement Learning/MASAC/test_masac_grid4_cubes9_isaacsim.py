"""
Test MASAC on Grid 4x4, 9 Cubes ONLY (Isaac Sim RRT)
Simplified version that tests only one configuration with Isaac Sim RRT planner

Usage:
    C:\isaacsim\python.bat test_masac_grid4_cubes9_isaacsim.py --episodes 5
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Add MASAC and MAPPO to path
masac_path = Path(__file__).parent
sys.path.insert(0, str(masac_path))
mappo_path = Path(__file__).parent.parent / "MAPPO"
sys.path.insert(0, str(mappo_path))

# Import Isaac Sim components (must be before other imports)
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

# Now import everything else
import numpy as np
import json
import csv
from datetime import datetime
import argparse

# Import Isaac Sim modules (using new isaacsim imports)
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
import os

from src.rl.doubleDQN import DoubleDQNAgent
from src.rl.object_selection_env_rrt import ObjectSelectionEnvRRT
from agents.masac_continuous_wrapper import MASACContinuousWrapper
from envs.two_agent_env import TwoAgentEnv


class FrankaRRTTrainer:
    """
    Franka controller for RRT-based testing (reachability checks only, no execution).
    Adapted from MAPPO training script.
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

        # Add ground plane
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
        print("[TRAINER] Creating random obstacles...")
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

            print("[TRAINER] RRT planner initialized successfully")

        except Exception as e:
            print(f"[TRAINER ERROR] Failed to initialize RRT planner: {e}")
            raise

    def _setup_container(self):
        """Setup container for cubes"""
        container_prim_path = "/World/Container"
        container_usd_path = f"{get_assets_root_path()}/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Storage/Containers/Container_I/Container_I04_160x120x64cm_PR_V_NVD_01.usd"
        add_reference_to_stage(usd_path=container_usd_path, prim_path=container_prim_path)

        container_position = np.array([0.45, -0.10, 0.0])
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

    def _spawn_cubes(self):
        """Spawn cubes in grid pattern with DDQN spacing (0.13/0.15)"""
        cube_size = 0.0515
        cube_spacing = 0.13 if self.training_grid_size > 3 else 0.15  # Match DDQN training
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
                        size=cube_size,
                        color=np.array([0.0, 0.5, 1.0])
                    )
                )
                self.cubes.append((cube, cube_name))
                self.cube_positions.append(position)
                cube_count += 1

        print(f"[TRAINER] Spawned {cube_count} cubes with spacing {cube_spacing}m")

    def _create_random_obstacles(self):
        """Create random obstacles in empty cells"""
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
            obs_position = np.array([obs_x, obs_y, 0.05])

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

        # Update RRT world to include obstacles
        if self.rrt:
            self.rrt.update_world()
            print(f"[TRAINER] Added {num_obstacles} obstacles to RRT collision checker")

    def randomize_cube_positions(self):
        """Randomize cube positions (for episode variation)"""
        if not self.cubes:
            return

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


def create_isaacsim_environment(grid_size: int, num_cubes: int, max_steps: int = 50):
    """
    Create Isaac Sim environment with Franka robot and RRT planner.
    Uses FrankaRRTTrainer for cube spawning, obstacle registration, and RRT reachability checks.
    """
    print("[ENV] Creating Isaac Sim environment...")
    print(f"[ENV] Grid: {grid_size}x{grid_size}, Cubes: {num_cubes}")

    # Create FrankaRRTTrainer (handles scene setup, cube spawning, RRT initialization)
    trainer = FrankaRRTTrainer(num_cubes=num_cubes, training_grid_size=grid_size)
    trainer.setup_scene()

    # Create environment with FrankaRRTTrainer as controller
    max_objects = grid_size * grid_size
    env = ObjectSelectionEnvRRT(
        franka_controller=trainer,  # ✅ Pass trainer for RRT reachability checks
        max_objects=max_objects,
        max_steps=max_steps,
        num_cubes=num_cubes,
        training_grid_size=grid_size,
        execute_picks=False,  # ❌ Don't execute actual picks (testing only)
        rrt_planner=trainer.rrt,
        kinematics_solver=trainer.kinematics_solver,
        articulation_kinematics_solver=trainer.articulation_kinematics_solver,
        franka_articulation=trainer.franka
    )

    print("[ENV] Isaac Sim environment created successfully")
    return env, trainer.world


def load_ddqn_agent(model_path: str, env) -> DoubleDQNAgent:
    """Load pretrained DDQN agent"""
    import torch
    agent = DoubleDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    agent.load(model_path)
    agent.epsilon = 0.01  # Set to minimum for testing
    print(f"✅ Loaded DDQN model: {model_path}")
    return agent


def save_results(env_type: str, grid_size: int, num_cubes: int, episode_results: list, timestep_results: list,
                 log_dir: str, algorithm: str = "MASAC", scenario: str = None, planner: str = None,
                 seed: int = None, run_id: int = 1):
    """
    Save test results to CSV and JSON with MAPPO-style formatting

    Args:
        env_type: Environment type (e.g., 'rrt_isaacsim', 'astar')
        grid_size: Grid size
        num_cubes: Number of cubes
        episode_results: List of episode-level results
        timestep_results: List of timestep-level results
        log_dir: Directory to save logs
        algorithm: Algorithm name (e.g., 'MASAC', 'MAPPO')
        scenario: Scenario name (e.g., 'grid4_cubes9_rrt_isaacsim')
        planner: Planner name (e.g., 'Isaac Sim RRT')
        seed: Random seed
        run_id: Run identifier
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Save Episode-level CSV
    if not episode_results:
        print("⚠️  Warning: No episode results to save!")
        return

    csv_filename = f"masac_{env_type}_grid{grid_size}_cubes{num_cubes}_{timestamp}_episode_log.csv"
    csv_path = log_path / csv_filename

    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=episode_results[0].keys())
            writer.writeheader()
            writer.writerows(episode_results)
        print(f"✅ Saved episode CSV: {csv_path.name}")
    except Exception as e:
        print(f"❌ Error saving episode CSV: {e}")

    # Save Timestep-level CSV
    if timestep_results:
        timestep_csv_filename = f"masac_{env_type}_grid{grid_size}_cubes{num_cubes}_{timestamp}_timestep_log.csv"
        timestep_csv_path = log_path / timestep_csv_filename

        try:
            with open(timestep_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=timestep_results[0].keys())
                writer.writeheader()
                writer.writerows(timestep_results)
            print(f"✅ Saved timestep CSV: {timestep_csv_path.name}")
        except Exception as e:
            print(f"❌ Error saving timestep CSV: {e}")
    else:
        print("⚠️  Warning: No timestep results to save!")

    # Calculate statistics with variance
    rewards = [r['total_reward'] for r in episode_results]
    reshuffles = [r['reshuffles_performed'] for r in episode_results]
    distances = [r['total_distance_reduced'] for r in episode_results]
    times = [r['total_time_saved'] for r in episode_results]
    episode_lengths = [r['episode_length'] for r in episode_results]
    cubes_picked = [r['cubes_picked'] for r in episode_results]
    success_ratios = [r['success'] for r in episode_results]

    # Calculate average success rate (now it's a ratio, not boolean)
    avg_success_ratio = np.mean(success_ratios) if success_ratios else 0.0
    success_rate_pct = avg_success_ratio * 100  # Convert to percentage

    # Save summary JSON with mean, std, variance, median, min, max (MAPPO-style)
    summary = {
        # Metadata
        'algorithm': algorithm,
        'scenario': scenario if scenario else f"grid{grid_size}_cubes{num_cubes}_{env_type}",
        'planner': planner if planner else env_type,
        'env_type': env_type,
        'grid_size': grid_size,
        'num_cubes': num_cubes,
        'seed': seed if seed is not None else -1,
        'run_id': run_id,
        'num_episodes': len(episode_results),
        'total_timesteps': len(timestep_results),
        'timestamp': timestamp,

        # Reward statistics
        'reward': {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'variance': float(np.var(rewards)),
            'median': float(np.median(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards)),
        },

        # Success rate statistics (NEW - for MAPPO-style tables)
        'success_rate': {
            'mean': float(avg_success_ratio),
            'percentage': float(success_rate_pct),
            'std': float(np.std([float(s) * 100 for s in success_ratios])),
        },

        # Reshuffles statistics
        'reshuffles': {
            'mean': float(np.mean(reshuffles)),
            'std': float(np.std(reshuffles)),
            'variance': float(np.var(reshuffles)),
            'median': float(np.median(reshuffles)),
            'min': int(np.min(reshuffles)),
            'max': int(np.max(reshuffles)),
        },

        # Distance reduced statistics (meters)
        'distance_reduced_m': {
            'mean': float(np.mean(distances)),
            'std': float(np.std(distances)),
            'variance': float(np.var(distances)),
            'median': float(np.median(distances)),
            'min': float(np.min(distances)),
            'max': float(np.max(distances)),
        },

        # Time saved statistics (seconds)
        'time_saved_s': {
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'variance': float(np.var(times)),
            'median': float(np.median(times)),
            'min': float(np.min(times)),
            'max': float(np.max(times)),
        },

        # Episode length statistics
        'episode_length': {
            'mean': float(np.mean(episode_lengths)),
            'std': float(np.std(episode_lengths)),
            'variance': float(np.var(episode_lengths)),
            'median': float(np.median(episode_lengths)),
            'min': int(np.min(episode_lengths)),
            'max': int(np.max(episode_lengths)),
        },

        # Cubes picked statistics
        'cubes_picked': {
            'mean': float(np.mean(cubes_picked)),
            'std': float(np.std(cubes_picked)),
            'variance': float(np.var(cubes_picked)),
            'median': float(np.median(cubes_picked)),
            'min': int(np.min(cubes_picked)),
            'max': int(np.max(cubes_picked)),
        },

        # Units reference
        'units': {
            'distance': 'meters',
            'time': 'seconds',
            'reward': 'dimensionless',
            'success_rate': 'percentage'
        }
    }

    json_filename = f"masac_{env_type}_grid{grid_size}_cubes{num_cubes}_{timestamp}_summary.json"
    json_path = log_path / json_filename

    try:
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Saved JSON summary: {json_path.name}")
    except Exception as e:
        print(f"❌ Error saving JSON summary: {e}")
        import traceback
        traceback.print_exc()


def test_masac_isaacsim_grid4_cubes9(num_episodes: int, log_dir: str, seed: int = None, run_id: int = 1):
    """
    Test MASAC on Grid 4x4, 9 cubes with Isaac Sim RRT

    Args:
        num_episodes: Number of test episodes
        log_dir: Directory to save logs
        seed: Random seed for reproducibility (optional)
        run_id: Unique identifier for this run
    """
    grid_size = 4
    num_cubes = 9
    algorithm = "MASAC"
    scenario = f"grid{grid_size}_cubes{num_cubes}_rrt_isaacsim"
    planner = "Isaac Sim RRT"

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)
        print(f"🎲 Random seed set to: {seed}")

    print(f"\n{'='*80}")
    print(f"Testing {algorithm}: {planner} | Grid {grid_size}x{grid_size} | {num_cubes} cubes")
    print(f"Scenario: {scenario} | Seed: {seed} | Run ID: {run_id}")
    print(f"{'='*80}")

    # DDQN model mapping
    config_key = f"rrt_isaacsim_grid{grid_size}_cubes{num_cubes}"
    ddqn_model_mapping = {
        'rrt_isaacsim_grid4_cubes9': 'ddqn_rrt_isaacsim_grid4_cubes9_20251224_185752_final.pt',
    }

    if config_key not in ddqn_model_mapping:
        print(f"⚠️  No DDQN model mapping for {config_key}")
        return None

    ddqn_model_filename = ddqn_model_mapping[config_key]
    ddqn_models_dir = project_root / "scripts" / "Reinforcement Learning" / "doubleDQN_script" / "models"
    ddqn_model_path = ddqn_models_dir / ddqn_model_filename

    if not ddqn_model_path.exists():
        print(f"⚠️  DDQN model not found: {ddqn_model_path}")
        return None

    # Create Isaac Sim environment with RRT planner
    print("✓ Creating Isaac Sim environment...")
    base_env, world = create_isaacsim_environment(
        grid_size=grid_size,
        num_cubes=num_cubes,
        max_steps=50
    )

    # Load DDQN agent
    print("✓ Loading DDQN agent...")
    ddqn_agent = load_ddqn_agent(str(ddqn_model_path), base_env)

    # Calculate Agent 2 observation dimension
    agent2_state_dim = (num_cubes * 3) + 3 + num_cubes + (grid_size * grid_size) + 10

    # Create MASAC wrapper with dimension adapter
    pretrained_path = project_root / "scripts" / "Reinforcement Learning" / "MASAC" / "pretrained_models"
    cube_spacing = 0.13  # Grid 4x4 uses 0.13

    masac_agent = MASACContinuousWrapper(
        state_dim=agent2_state_dim,
        grid_size=grid_size,
        num_cubes=num_cubes,
        cube_spacing=cube_spacing,
        pretrained_model_path=str(pretrained_path),
        use_dimension_adapter=True,
        memory_size=10000,
        batch_size=64
    )

    # Create two-agent environment
    two_agent_env = TwoAgentEnv(
        base_env=base_env,
        ddqn_agent=ddqn_agent,
        grid_size=grid_size,
        num_cubes=num_cubes,
        max_reshuffles_per_episode=5,
        reshuffle_reward_scale=1.0,
        max_episode_steps=50,
        verbose=False
    )

    # Relax reshuffling thresholds for testing
    print("[TEST] Relaxing reshuffling thresholds for testing...")
    two_agent_env.reshuffle_decision.min_reachable_distance = 0.30
    two_agent_env.reshuffle_decision.max_reachable_distance = 0.90
    two_agent_env.reshuffle_decision.path_length_ratio_threshold = 1.5
    two_agent_env.reshuffle_decision.crowded_threshold = 2
    two_agent_env.reshuffle_decision.rrt_failure_window = 2
    two_agent_env.reshuffle_decision.min_clearance = 0.35
    two_agent_env.reshuffle_decision.far_cube_ratio = 1.1
    two_agent_env.reshuffle_decision.batch_reshuffle_count = 2
    print("[TEST] Reshuffling thresholds relaxed!")

    # Enable test mode to skip expensive reachability checks during PCA fitting
    print("[TEST] Enabling test mode for fast PCA fitting...")
    base_env.test_mode = True
    print("[TEST] Test mode enabled! Reachability checks will be skipped.")

    # Fit PCA dimension adapter
    masac_agent.fit_dimension_adapter(two_agent_env, n_samples=500)

    # Set test mode
    masac_agent.set_test_mode(True)

    # Test episodes
    episode_results = []
    timestep_results = []  # NEW: Track timestep-level data
    global_timestep = 0  # NEW: Global timestep counter across all episodes

    for episode in range(num_episodes):
        # DO NOT randomize cube positions - use fixed positions for consistent testing
        # This ensures all episodes are tested on the same cube configurations
        # if hasattr(base_env, 'franka_controller') and base_env.franka_controller is not None:
        #     base_env.franka_controller.randomize_cube_positions()

        obs, reset_info = two_agent_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        reshuffles_performed = 0

        print(f"\n[Episode {episode+1}] Starting...")

        while not (done or truncated) and episode_length < 50:
            # Calculate valid cubes for action masking
            valid_cubes = [
                i for i in range(num_cubes)
                if i not in two_agent_env.base_env.objects_picked
                and two_agent_env.reshuffle_count_per_cube.get(i, 0) < 2
            ]

            # MASAC selects reshuffling action with action masking
            action_dict = masac_agent.select_action(obs, deterministic=True, valid_cubes=valid_cubes)

            # Skip if no valid action
            if action_dict is None:
                print(f"  [WARNING] No valid cubes to reshuffle, skipping step")
                break

            # Convert dictionary action to integer action for TwoAgentEnv
            action_int = two_agent_env.reshuffle_action_space.encode_action(
                cube_idx=action_dict['cube_idx'],
                grid_x=action_dict['target_grid_x'],
                grid_y=action_dict['target_grid_y']
            )

            # Execute action in environment
            next_obs, reward, done, truncated, info = two_agent_env.step(action_int)

            episode_reward += reward
            episode_length += 1
            global_timestep += 1

            if info.get('reshuffled_this_step', False):
                reshuffles_performed += 1

            # Log timestep-level data with algorithm/scenario metadata
            timestep_data = {
                'global_timestep': global_timestep,
                'episode': episode + 1,
                'step_in_episode': episode_length,
                'reward': float(reward),
                'cumulative_reward': float(episode_reward),
                'reshuffled': info.get('reshuffled_this_step', False),
                'distance_reduced': float(two_agent_env.total_distance_reduced),
                'time_saved': float(two_agent_env.total_time_saved),
                'cubes_picked': len(two_agent_env.base_env.objects_picked),
                'done': done,
                'truncated': truncated,
                'algorithm': algorithm,
                'scenario': scenario,
                'planner': planner,
                'seed': seed if seed is not None else -1,
                'run_id': run_id
            }
            timestep_results.append(timestep_data)

            obs = next_obs

        # Log episode results with success ratio and metadata
        cubes_picked_count = len(two_agent_env.base_env.objects_picked)
        success_ratio = cubes_picked_count / num_cubes  # Success ratio (e.g., 8/9 = 0.89)

        result = {
            'episode': episode + 1,
            'total_reward': float(episode_reward),
            'episode_length': episode_length,
            'reshuffles_performed': reshuffles_performed,
            'total_distance_reduced': float(two_agent_env.total_distance_reduced),
            'total_time_saved': float(two_agent_env.total_time_saved),
            'cubes_picked': cubes_picked_count,
            'success': success_ratio,  # Changed to ratio instead of boolean
            'algorithm': algorithm,
            'scenario': scenario,
            'planner': planner,
            'seed': seed if seed is not None else -1,
            'run_id': run_id
        }
        episode_results.append(result)

        print(f"Episode {episode+1}/{num_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Reshuffles={reshuffles_performed}, "
              f"Distance={two_agent_env.total_distance_reduced:.3f}m, "
              f"Cubes={len(two_agent_env.base_env.objects_picked)}/{num_cubes}")

    # Save results (both episode and timestep level)
    save_results(
        env_type='rrt_isaacsim',
        grid_size=grid_size,
        num_cubes=num_cubes,
        episode_results=episode_results,
        timestep_results=timestep_results,
        log_dir=log_dir,
        algorithm=algorithm,
        scenario=scenario,
        planner=planner,
        seed=seed,
        run_id=run_id
    )

    # Clean up Isaac Sim
    print("\nCleaning up Isaac Sim...")
    from omni.isaac.core.utils.stage import clear_stage
    clear_stage()
    simulation_app.close()

    return episode_results


def main():
    """Test MASAC on Grid 4x4, 9 cubes (Isaac Sim RRT)"""
    parser = argparse.ArgumentParser(description='Test MASAC on Grid 4x4, 9 cubes (Isaac Sim RRT)')
    parser.add_argument('--episodes', type=int, default=5, help='Number of test episodes')
    parser.add_argument('--log_dir', type=str, default='cobotproject/scripts/Reinforcement Learning/MASAC/logs',
                        help='Directory to save logs')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--run_id', type=int, default=1, help='Run identifier for multiple runs')
    args = parser.parse_args()

    print(f"MASAC TESTING - Grid 4x4, 9 Cubes (Isaac Sim RRT)")
    print(f"Episodes: {args.episodes}")
    print(f"Log directory: {args.log_dir}")
    print(f"Seed: {args.seed}")
    print(f"Run ID: {args.run_id}")
    print(f"{'='*80}\n")

    try:
        results = test_masac_isaacsim_grid4_cubes9(
            num_episodes=args.episodes,
            log_dir=args.log_dir,
            seed=args.seed,
            run_id=args.run_id
        )

        if results:
            print(f"\n{'='*80}")
            print("MASAC TESTING COMPLETE - Grid 4x4, 9 Cubes (Isaac Sim RRT)")
            print(f"{'='*80}")
            print(f"Successfully tested Isaac Sim RRT planner")
            print(f"Results saved to: {args.log_dir}")
            print(f"{'='*80}\n")
        else:
            print(f"\n❌ Testing failed")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Always clean up Isaac Sim properly
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


if __name__ == "__main__":
    main()


