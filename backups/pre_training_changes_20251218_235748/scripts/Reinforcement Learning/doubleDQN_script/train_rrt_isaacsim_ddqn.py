"""
Train Double DQN Agent with RRT Path Planning in Isaac Sim
Uses Double DQN algorithm with actual Isaac Sim RRT for training.

Usage:
    C:\isaacsim\python.bat train_rrt_isaacsim_ddqn.py --timesteps 50000 --grid_size 4 --num_cubes 9
"""

import argparse
import sys
from pathlib import Path

# Parse command-line arguments BEFORE importing Isaac Sim
parser = argparse.ArgumentParser(description="Train Double DQN agent with Isaac Sim RRT")
parser.add_argument("--timesteps", type=int, default=50000,
                   help="Total training timesteps (default: 50000)")
parser.add_argument("--grid_size", type=int, default=4,
                   help="Grid size (default: 4)")
parser.add_argument("--num_cubes", type=int, default=9,
                   help="Number of cubes (default: 9)")
parser.add_argument("--save_freq", type=int, default=5000,
                   help="Save checkpoint every N steps (default: 5000)")
parser.add_argument("--learning_rate", type=float, default=1e-3,
                   help="Learning rate (default: 1e-3)")
parser.add_argument("--batch_size", type=int, default=64,
                   help="Batch size (default: 64)")
parser.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor (default: 0.99)")
parser.add_argument("--epsilon_start", type=float, default=1.0,
                   help="Initial epsilon (default: 1.0)")
parser.add_argument("--epsilon_end", type=float, default=0.01,
                   help="Final epsilon (default: 0.01)")
parser.add_argument("--epsilon_decay", type=float, default=0.995,
                   help="Epsilon decay rate (default: 0.995)")
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


def main():
    """Main training loop"""
    print("=" * 60)
    print("DOUBLE DQN TRAINING - RRT ISAAC SIM")
    print("=" * 60)
    print(f"Timesteps: {args.timesteps}")
    print(f"Grid: {args.grid_size}x{args.grid_size}")
    print(f"Cubes: {args.num_cubes}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gamma: {args.gamma}")
    print(f"Epsilon: {args.epsilon_start} -> {args.epsilon_end} (decay: {args.epsilon_decay})")
    print("=" * 60)

    # Create trainer
    trainer = FrankaRRTTrainer(
        num_cubes=args.num_cubes,
        training_grid_size=args.grid_size
    )

    # Setup scene
    trainer.setup_scene()

    # Create RL environment
    print("\n[TRAINER] Creating RRT-based RL environment...")
    max_objects = args.grid_size * args.grid_size

    env = ObjectSelectionEnvRRT(
        franka_controller=trainer,
        max_objects=max_objects,
        max_steps=50,
        training_grid_size=args.grid_size,
        execute_picks=False,
        rrt_planner=trainer.rrt,
        kinematics_solver=trainer.kinematics_solver,
        articulation_kinematics_solver=trainer.articulation_kinematics_solver,
        franka_articulation=trainer.franka
    )

    # Create agent
    state_dim = max_objects * 6
    action_dim = max_objects

    agent = DoubleDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        buffer_capacity=100000,
        target_update_freq=1000
    )

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ddqn_rrt_isaacsim_grid{args.grid_size}_cubes{args.num_cubes}_{timestamp}"
    log_dir = r"C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\logs"
    model_dir = r"C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Create log file
    log_file = os.path.join(log_dir, f"{run_name}_training.csv")
    with open(log_file, 'w') as f:
        f.write("step,episode,loss,reward,epsilon\n")

    print("\nStarting training...")
    print(f"Checkpoints will be saved to: {model_dir}/{run_name}")
    print(f"Training log will be saved to: {log_file}")
    print("=" * 60 + "\n")

    episode_rewards = []
    episode_lengths = []
    total_steps = 0
    episode = 0

    while total_steps < args.timesteps:
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done and total_steps < args.timesteps:
            # Get action mask
            action_mask = info.get('action_mask', env.action_masks())

            # Select action
            action = agent.select_action(state, action_mask)

            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Get next action mask
            next_action_mask = info.get('action_mask', env.action_masks())

            # Store transition
            agent.store_transition(state, action, reward, next_state, done, action_mask, next_action_mask)

            # Train
            loss = agent.train_step()

            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1

            # Log training data
            with open(log_file, 'a') as f:
                loss_val = f"{loss:.6f}" if loss is not None else ""
                f.write(f"{total_steps},{episode},{loss_val},{reward:.6f},{agent.epsilon:.6f}\n")

            # Print progress
            if total_steps % 1000 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                loss_str = f"{loss:.4f}" if loss else "0.0000"
                print(f"Steps: {total_steps}/{args.timesteps} | "
                      f"Episode: {episode} | "
                      f"Avg Reward (100 ep): {avg_reward:.2f} | "
                      f"Epsilon: {agent.epsilon:.4f} | "
                      f"Loss: {loss_str}")

            # Save checkpoint
            if total_steps % args.save_freq == 0:
                checkpoint_path = os.path.join(model_dir, f"{run_name}_step_{total_steps}.pt")
                agent.save(checkpoint_path)

        # Episode finished
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        agent.episodes += 1
        episode += 1

    # Save final model
    final_path = os.path.join(model_dir, f"{run_name}_final.pt")
    agent.save(final_path)

    # Save metadata
    metadata = {
        "method": "rrt",
        "algorithm": "double_dqn",
        "training_grid_size": args.grid_size,
        "num_cubes": args.num_cubes,
        "max_objects": max_objects,
        "max_steps": 50,
        "timestamp": timestamp,
        "total_timesteps": args.timesteps,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay": args.epsilon_decay,
        "final_epsilon": agent.epsilon,
        "total_episodes": episode
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
    print(f"Average reward (last 100 ep): {np.mean(episode_rewards[-100:]):.2f}")
    print("=" * 60)

    simulation_app.close()


if __name__ == "__main__":
    main()

