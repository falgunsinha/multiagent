"""
Train RL Agent with RRT Path Planning (Option 3)
Uses actual RRT path planning in Isaac Sim for training.
Requires Isaac Sim to be running during training.

This is the most accurate training method but also the slowest.

Usage:
    C:\isaacsim\python.bat train_object_selection_rrt.py --timesteps 50000 --training_grid_size 6
"""

import argparse
import sys
from pathlib import Path

# Parse command-line arguments BEFORE importing Isaac Sim
parser = argparse.ArgumentParser(description="Train object selection RL agent with RRT (Option 3)")
parser.add_argument("--timesteps", type=int, default=10000,
                   help="Total training timesteps (default: 10000, recommended: 10k-20k)")
parser.add_argument("--training_grid_size", type=int, default=3,
                   help="Training grid size (e.g., 3 for 3x3 grid, default: 3)")
parser.add_argument("--num_cubes", type=int, default=4,
                   help="Number of cubes per episode (default: 4)")
parser.add_argument("--save_freq", type=int, default=2000,
                   help="Save checkpoint every N steps (default: 2000)")
parser.add_argument("--learning_rate", type=float, default=3e-4,
                   help="Learning rate for PPO (default: 3e-4)")
parser.add_argument("--execute_picks", action="store_true",
                   help="Actually execute pick-and-place during training (very slow)")
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
import omni.timeline
import omni.usd

# Isaac Sim imports
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api.objects import DynamicCuboid
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

# Stable-Baselines3 imports
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure

# Custom imports
from src.rl.object_selection_env_rrt import ObjectSelectionEnvRRT


class FrankaRRTTrainer:
    """
    Franka controller for RRT-based RL training.
    Simplified version of the standalone script focused on training.
    """

    def __init__(self, num_cubes=4, training_grid_size=6):
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

        print(f"[TRAINER] Initializing Franka RRT Trainer")
        print(f"[TRAINER] Grid: {training_grid_size}x{training_grid_size}, Cubes: {num_cubes}")

    def setup_scene(self):
        """Setup Isaac Sim scene with Franka and cubes"""
        print("[TRAINER] Setting up scene...")

        # Create world
        self.world = World(stage_units_in_meters=1.0)

        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Add Franka robot USD to stage (this creates the prim immediately)
        assets_root_path = get_assets_root_path()
        franka_prim_path = "/World/Franka"

        # Use the same pattern as standalone: franka.usd with variant selection
        franka_usd_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        robot_prim = add_reference_to_stage(usd_path=franka_usd_path, prim_path=franka_prim_path)
        robot_prim.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
        robot_prim.GetVariantSet("Mesh").SetVariantSelection("Quality")

        # Create gripper (prim now exists, no need to reset world first)
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

        # Spawn cubes FIRST (needed to determine empty cells)
        self._spawn_cubes()

        # Create random obstacles in empty cells
        print("[TRAINER] Creating random obstacles in empty cells...")
        self._create_random_obstacles()

        # Reset world after all objects added
        self.world.reset()

        print("[TRAINER] Scene setup complete")

    def _setup_rrt_planner(self):
        """Setup RRT path planner"""
        try:
            mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")

            # Use local robot description file from assets folder
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(script_dir, "..", "..")
            robot_description_file = os.path.join(project_root, "assets", "franka_conservative_spheres_robot_description.yaml")
            robot_description_file = os.path.normpath(robot_description_file)

            urdf_path = os.path.join(mg_extension_path, "motion_policy_configs", "franka", "lula_franka_gen.urdf")
            rrt_config_file = os.path.join(mg_extension_path, "path_planner_configs", "franka", "rrt", "franka_planner_config.yaml")

            # Verify files exist
            if not os.path.exists(robot_description_file):
                print(f"[TRAINER WARNING] Robot description not found: {robot_description_file}")
                print("[TRAINER] Using default RMPflow robot descriptor")
                robot_description_file = os.path.join(mg_extension_path, "motion_policy_configs", "franka", "rmpflow", "robot_descriptor.yaml")

            self.rrt = RRT(
                robot_description_path=robot_description_file,
                urdf_path=urdf_path,
                rrt_config_path=rrt_config_file,
                end_effector_frame_name="right_gripper"
            )
            self.rrt.set_max_iterations(10000)

            # Create path planner visualizer (needed for getting correct joint subset)
            self.path_planner_visualizer = PathPlannerVisualizer(
                robot_articulation=self.franka,
                path_planner=self.rrt
            )

            # Create kinematics solvers (needed for RL environment)
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

        # Add physics to container
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

        # Convert to grid coordinates using ROUNDING (not truncation)
        grid_x = int(round((world_pos[0] - start_x) / cube_spacing))
        grid_y = int(round((world_pos[1] - start_y) / cube_spacing))

        # CLAMP to valid grid bounds [0, grid_size-1]
        # Ensures cubes/obstacles outside grid boundaries map to edge cells
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
        """Create random obstacles in empty grid cells for training"""
        from isaacsim.core.api.objects import FixedCuboid

        # Number of obstacles based on grid size
        num_obstacles_map = {
            3: 1,   # 3x3: 1 obstacle
            4: 2,   # 4x4: 2 obstacles
            6: np.random.randint(3, 6),  # 6x6: 3-5 obstacles (random)
        }
        num_obstacles = num_obstacles_map.get(self.training_grid_size, max(1, self.training_grid_size // 3))

        # Get cube positions to avoid placing obstacles on cubes
        cube_cells = set()
        for i in range(self.num_cubes):
            cube_pos = self.cube_positions[i]
            grid_col, grid_row = self._world_to_grid(cube_pos[:2])
            cube_cells.add((grid_col, grid_row))

        # Get all empty cells (cells without cubes)
        empty_cells = []
        for grid_x in range(self.training_grid_size):
            for grid_y in range(self.training_grid_size):
                if (grid_x, grid_y) not in cube_cells:
                    empty_cells.append((grid_x, grid_y))

        # Randomly select obstacle positions from empty cells
        if len(empty_cells) < num_obstacles:
            print(f"[TRAINER] Not enough empty cells for {num_obstacles} obstacles (only {len(empty_cells)} available)")
            num_obstacles = len(empty_cells)

        if num_obstacles == 0:
            print(f"[TRAINER] No empty cells for obstacles")
            return

        print(f"[TRAINER] Creating {num_obstacles} random obstacles in empty cells for {self.training_grid_size}x{self.training_grid_size} grid")

        # Randomly shuffle and pick first num_obstacles cells
        np.random.shuffle(empty_cells)
        selected_cells = empty_cells[:num_obstacles]

        for idx, (grid_x, grid_y) in enumerate(selected_cells):
            # Convert to world coordinates
            world_pos = self._grid_to_world(grid_x, grid_y)
            obs_position = np.array([world_pos[0], world_pos[1], 0.055])
            obs_name = f"Obstacle_{idx}"

            # Create obstacle as FixedCuboid (static, no PhysX errors)
            obstacle = self.world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/{obs_name}",
                    name=obs_name,
                    position=obs_position,
                    scale=np.array([0.11, 0.11, 0.11]),  # 11cm obstacle
                    color=np.array([1.0, 0.0, 0.0])  # Red color
                )
            )

            print(f"[TRAINER] Created {obs_name} at grid ({grid_x}, {grid_y}) -> world {obs_position[:2]}")

    def _spawn_cubes(self):
        """Spawn cubes in grid pattern"""
        cube_size = 0.0515
        # CRITICAL: cube_spacing MUST match cell_size used by path estimators!
        # Updated from 0.20/0.22 to 0.26/0.28 to ensure gripper (15cm) can fit between objects
        # With 26cm spacing: cube-to-cube gap = 26cm - 5.15cm = 20.85cm (gripper 15cm fits with 5.85cm clearance)
        cube_spacing = 0.26 if self.training_grid_size > 3 else 0.28
        grid_center_x = 0.45
        grid_center_y = -0.10

        # Calculate grid extent
        grid_extent_x = (self.training_grid_size - 1) * cube_spacing
        grid_extent_y = (self.training_grid_size - 1) * cube_spacing
        start_x = grid_center_x - (grid_extent_x / 2.0)
        start_y = grid_center_y - (grid_extent_y / 2.0)

        # Randomly select cells
        total_cells = self.training_grid_size * self.training_grid_size
        selected_indices = np.random.choice(total_cells, size=self.num_cubes, replace=False)
        selected_cells = set(selected_indices)

        cube_count = 0
        for row in range(self.training_grid_size):
            for col in range(self.training_grid_size):
                cell_index = row * self.training_grid_size + col
                if cell_index not in selected_cells:
                    continue

                # Calculate position
                base_x = start_x + (row * cube_spacing)
                base_y = start_y + (col * cube_spacing)

                # Random offset
                offset_x = np.random.uniform(-0.02, 0.02)
                offset_y = np.random.uniform(-0.02, 0.02)

                position = np.array([base_x + offset_x, base_y + offset_y, cube_size / 2.0])

                # Create cube
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
                # Store as tuple (cube, name) to match expected format
                self.cubes.append((cube, cube_name))
                self.cube_positions.append(position)
                cube_count += 1

        print(f"[TRAINER] Spawned {len(self.cubes)} cubes")

    def get_cube_positions(self):
        """Get current cube positions"""
        positions = []
        for cube, cube_name in self.cubes:
            pos, _ = cube.get_world_pose()
            positions.append(pos)
        return positions


def mask_fn(env):
    """Extract action mask from environment for ActionMasker wrapper"""
    return env.action_masks()


def main():
    """Main training loop"""
    print("=" * 60)
    print("OBJECT SELECTION RL TRAINING - RRT METHOD (OPTION 3)")
    print("=" * 60)
    print(f"Timesteps: {args.timesteps}")
    print(f"Training grid: {args.training_grid_size}x{args.training_grid_size}")
    print(f"Cubes per episode: {args.num_cubes}")
    print(f"Execute picks: {args.execute_picks}")
    print(f"Action Masking: ENABLED (invalid picks prevented)")
    print("=" * 60)

    # Create trainer
    trainer = FrankaRRTTrainer(
        num_cubes=args.num_cubes,
        training_grid_size=args.training_grid_size
    )

    # Setup scene
    trainer.setup_scene()

    # Create RL environment
    print("\n[TRAINER] Creating RRT-based RL environment with action masking...")
    max_objects = args.training_grid_size * args.training_grid_size

    def make_env():
        env = ObjectSelectionEnvRRT(
            franka_controller=trainer,
            max_objects=max_objects,
            max_steps=50,
            training_grid_size=args.training_grid_size,
            execute_picks=args.execute_picks,
            rrt_planner=trainer.rrt,
            kinematics_solver=trainer.kinematics_solver,
            articulation_kinematics_solver=trainer.articulation_kinematics_solver,
            franka_articulation=trainer.franka
        )
        # Wrap with ActionMasker for action masking support
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"object_selection_rrt_grid{args.training_grid_size}x{args.training_grid_size}_cubes{args.num_cubes}_{timestamp}"
    log_dir = "logs/object_selection"
    model_dir = "models/object_selection"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Create MaskablePPO model (supports action masking)
    print("\n[TRAINER] Creating MaskablePPO model with action masking...")
    model = MaskablePPO(
        "MlpPolicy",  # Use built-in MlpPolicy (supports action masking)
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=1024,  # Smaller for RRT (slower steps)
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=log_dir,
        seed=42
    )

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=os.path.join(model_dir, run_name),
        name_prefix="ppo_rrt",
        save_vecnormalize=True
    )

    # Configure logger
    logger = configure(os.path.join(log_dir, run_name), ["stdout", "tensorboard"])
    model.set_logger(logger)

    # Train
    print("\n" + "=" * 60)
    print("STARTING RRT-BASED TRAINING")
    print("=" * 60)
    print(f"Monitor training with: tensorboard --logdir {log_dir}")
    print("=" * 60 + "\n")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")

    # Save final model
    final_model_path = os.path.join(model_dir, f"{run_name}_final")
    model.save(final_model_path)
    vec_env.save(os.path.join(model_dir, f"{run_name}_vecnormalize.pkl"))

    # Save metadata
    import json
    metadata = {
        "method": "rrt",
        "training_grid_size": args.training_grid_size,
        "max_objects": max_objects,
        "num_cubes": args.num_cubes,
        "max_steps": 50,
        "timestamp": timestamp,
        "total_timesteps": args.timesteps,
        "learning_rate": args.learning_rate,
        "execute_picks": args.execute_picks,
        "policy": "mlp"
    }
    metadata_path = os.path.join(model_dir, f"{run_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final model saved to: {final_model_path}")
    print(f"VecNormalize stats saved to: {run_name}_vecnormalize.pkl")
    print(f"Metadata saved to: {run_name}_metadata.json")
    print("=" * 60)

    # Close environment
    vec_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()


