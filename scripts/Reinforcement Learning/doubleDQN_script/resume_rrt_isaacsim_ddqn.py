import argparse
import sys
from pathlib import Path
import gc
import torch

parser = argparse.ArgumentParser(description="Resume Double DQN agent training with Isaac Sim RRT")
parser.add_argument("--checkpoint_path", type=str,
                   default="auto",
                   help="Path to checkpoint to resume from (default: auto - finds latest checkpoint)")
parser.add_argument("--target_timesteps", type=int, default=50000,
                   help="Target total timesteps (default: 50000)")
parser.add_argument("--save_freq", type=int, default=5000,
                   help="Save checkpoint every N steps (default: 5000)")
args = parser.parse_args()

if args.checkpoint_path == "auto":
    models_dir = Path(r"C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\models")
    # Find latest checkpoint (not final, not emergency)
    checkpoints = list(models_dir.glob("ddqn_rrt_isaacsim_*_step_*.pt"))
    checkpoints = [c for c in checkpoints if "final" not in c.name and "emergency" not in c.name]

    if not checkpoints:
        print("[ERROR] No checkpoints found to resume from!")
        print(f"[ERROR] Searched in: {models_dir}")
        sys.exit(1)

    # Sort by modification time (most recent first)
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    args.checkpoint_path = str(checkpoints[0])
    print(f"[AUTO] Found latest checkpoint: {checkpoints[0].name}")
else:
    args.checkpoint_path = str(Path(args.checkpoint_path))

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("[RESUME] Cleared CUDA cache")

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
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.manipulators import SingleManipulator
from src.grippers import ParallelGripper

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
        """Setup the scene with Franka, cubes, and container"""
        print(f"[TRAINER] Setting up scene...")
        if World.instance() is not None:
            World.clear_instance()
        self.world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
        self.world.scene.add_default_ground_plane()
        franka_prim_path = "/World/Franka"
        franka_usd_path = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        add_reference_to_stage(usd_path=franka_usd_path, prim_path=franka_prim_path)
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

        stage = omni.usd.get_context().get_stage()
        container_prim = stage.GetPrimAtPath(container_prim_path)
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(container_prim)
        rigid_body_api.CreateKinematicEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(container_prim)

    
        cube_size = 0.0515
        if self.num_cubes <= 4:
            cube_spacing = 0.28
        elif self.num_cubes <= 9:
            cube_spacing = 0.26
        else:
            cube_spacing = 0.26

        grid_center_x = 0.45
        grid_center_y = -0.10
        grid_extent_x = (self.training_grid_size - 1) * cube_spacing
        grid_extent_y = (self.training_grid_size - 1) * cube_spacing
        start_x = grid_center_x - (grid_extent_x / 2.0)
        start_y = grid_center_y - (grid_extent_y / 2.0)

        colors = [
            np.array([0, 0, 1]),      # Blue
            np.array([1, 0, 0]),      # Red
            np.array([0, 1, 0]),      # Green
            np.array([1, 1, 0]),      # Yellow
            np.array([1, 0, 1]),      # Magenta
            np.array([0, 1, 1]),      # Cyan
            np.array([1, 0.5, 0]),    # Orange
            np.array([0.5, 0, 1]),    # Purple
            np.array([0.5, 0.5, 0.5]),# Gray
            np.array([1, 0.75, 0.8]), # Pink
        ]

        cube_index = 0
        for row in range(self.training_grid_size):
            for col in range(self.training_grid_size):
                if cube_index >= self.num_cubes:
                    break

                cube_x = start_x + (row * cube_spacing)
                cube_y = start_y + (col * cube_spacing)
                cube_z = cube_size/2.0 + 0.01
                cube_position = np.array([cube_x, cube_y, cube_z])

                color = colors[cube_index % len(colors)]
                cube_number = cube_index + 1
                cube_name = f"Cube_{cube_number}"
                prim_path = f"/World/{cube_name}"

                cube = self.world.scene.add(
                    DynamicCuboid(
                        name=cube_name,
                        position=cube_position,
                        prim_path=prim_path,
                        scale=np.array([cube_size, cube_size, cube_size]),
                        size=1.0,
                        color=color
                    )
                )

                # Store as tuple (cube, cube_name) to match original training script
                self.cubes.append((cube, cube_name))
                self.cube_positions.append(cube_position)
                cube_index += 1

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
            path_planner=self.rrt  # Correct parameter name
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

        # Initialize physics
        self.world.initialize_physics()
        self.world.reset()

        print(f"[TRAINER] Scene setup complete")


def main():
    """Resume training from checkpoint"""
    
    checkpoint_path = Path(args.checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        simulation_app.close()
        return
    
    print("="*80)
    print("RESUMING DOUBLE DQN TRAINING WITH ISAAC SIM RRT")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Target timesteps: {args.target_timesteps}")
    print("="*80)
    
    # Load checkpoint to get metadata
    print(f"\n[RESUME] Loading checkpoint metadata...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    current_steps = checkpoint.get('steps', 0)
    current_episodes = checkpoint.get('episodes', 0)
    current_epsilon = checkpoint.get('epsilon', 0.01)
    
    print(f"[RESUME] Current steps: {current_steps}")
    print(f"[RESUME] Current episodes: {current_episodes}")
    print(f"[RESUME] Current epsilon: {current_epsilon:.6f}")
    
    remaining_steps = args.target_timesteps - current_steps
    print(f"[RESUME] Remaining steps: {remaining_steps}")
    
    if remaining_steps <= 0:
        print(f"[RESUME] Training already complete! ({current_steps} >= {args.target_timesteps})")
        simulation_app.close()
        return
    
 
    filename_parts = checkpoint_path.stem.split('_')
    grid_size = 4  # default
    num_cubes = 9  # default
    timestamp_str = None
    
    for i, part in enumerate(filename_parts):
        if part.startswith('grid') and i+1 < len(filename_parts):
            try:
                grid_size = int(filename_parts[i].replace('grid', ''))
            except:
                pass
        if part.startswith('cubes') and i+1 < len(filename_parts):
            try:
                num_cubes = int(filename_parts[i].replace('cubes', ''))
            except:
                pass
        if part.isdigit() and len(part) >= 14:  # Timestamp format
            timestamp_str = part
    
    print(f"[RESUME] Grid size: {grid_size}")
    print(f"[RESUME] Number of cubes: {num_cubes}")
    
    script_dir = Path(__file__).parent
    models_dir = script_dir / "models"
    logs_dir = script_dir / "logs"

    if timestamp_str:
        model_name = f"ddqn_rrt_isaacsim_grid{grid_size}_cubes{num_cubes}_{timestamp_str}"
    else:
        model_name = f"ddqn_rrt_isaacsim_grid{grid_size}_cubes{num_cubes}_20251216_010838"

    log_file = logs_dir / f"{model_name}_training.csv"

    print(f"[RESUME] Model name: {model_name}")
    print(f"[RESUME] Log file: {log_file}")
    print(f"\n[RESUME] Setting up scene...")
    trainer = FrankaRRTTrainer(num_cubes=num_cubes, training_grid_size=grid_size)
    trainer.setup_scene()

    print(f"[RESUME] Scene setup complete")

    print(f"\n[RESUME] Creating RRT-based RL environment...")
    max_objects = grid_size * grid_size

    env = ObjectSelectionEnvRRT(
        franka_controller=trainer,
        max_objects=max_objects,
        max_steps=50,
        training_grid_size=grid_size,
        execute_picks=False,
        rrt_planner=trainer.rrt,
        kinematics_solver=trainer.kinematics_solver,
        articulation_kinematics_solver=trainer.articulation_kinematics_solver,
        franka_articulation=trainer.franka
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"[RESUME] State dim: {state_dim}, Action dim: {action_dim}")
    print(f"\n[RESUME] Initializing Double DQN agent...")
    agent = DoubleDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=checkpoint.get('learning_rate', 1e-3),
        gamma=checkpoint.get('gamma', 0.99),
        epsilon_start=current_epsilon,  # Start from current epsilon
        epsilon_end=checkpoint.get('epsilon_end', 0.01),
        epsilon_decay=checkpoint.get('epsilon_decay', 0.995),
        batch_size=checkpoint.get('batch_size', 64),
        buffer_capacity=100000,  # Correct parameter name
        target_update_freq=1000
    )
    print(f"\n[RESUME] Loading checkpoint weights...")
    agent.load(str(checkpoint_path))
    print(f"[RESUME] Checkpoint loaded successfully")
    print(f"[RESUME] Agent epsilon: {agent.epsilon:.6f}")
    print(f"[RESUME] Agent steps: {agent.steps}")
    print(f"[RESUME] Agent episodes: {agent.episodes}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[RESUME] Cleared CUDA cache after loading checkpoint")
    print(f"\n{'='*80}")
    print(f"RESUMING TRAINING FROM STEP {current_steps}")
    print(f"{'='*80}\n")

    state, info = env.reset()
    episode_reward = 0
    episode_start_time = time.time()

    step = current_steps
    episode = current_episodes

    try:
        while step < args.target_timesteps:
            action_mask = info.get('action_mask', env.action_masks())
            action = agent.select_action(state, action_mask)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_action_mask = info.get('action_mask', env.action_masks())
            agent.store_transition(state, action, reward, next_state, done, action_mask, next_action_mask)
            loss = agent.train_step()
            if step % 1000 == 0:
                torch.cuda.empty_cache()
            state = next_state
            episode_reward += reward
            step += 1
            with open(log_file, 'a') as f:
                loss_val = f"{loss:.6f}" if loss is not None else ""
                f.write(f"{step},{episode},{loss_val},{reward:.6f},{agent.epsilon:.6f}\n")
            if done:
                episode += 1
                episode_time = time.time() - episode_start_time

                if episode % 10 == 0:
                    print(f"[RESUME] Step {step}/{args.target_timesteps} | Episode {episode} | "
                          f"Reward: {episode_reward:.2f} | Epsilon: {agent.epsilon:.4f} | "
                          f"Time: {episode_time:.1f}s")
                state, info = env.reset()
                episode_reward = 0
                episode_start_time = time.time()

            if step % args.save_freq == 0 and step > current_steps:
                checkpoint_file = models_dir / f"{model_name}_step_{step}.pt"
                agent.save(str(checkpoint_file))
                print(f"\n[RESUME] Checkpoint saved: {checkpoint_file.name}\n")

                # Clear CUDA cache after saving
                torch.cuda.empty_cache()
                gc.collect()

        final_checkpoint = models_dir / f"{model_name}_final.pt"
        agent.save(str(final_checkpoint))
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE!")
        print(f"{'='*80}")
        print(f"Final model saved: {final_checkpoint.name}")
        print(f"Total steps: {step}")
        print(f"Total episodes: {episode}")
        print(f"Final epsilon: {agent.epsilon:.6f}")

    except Exception as e:
        print(f"\n[ERROR] Training interrupted: {e}")
        import traceback
        traceback.print_exc()

        emergency_checkpoint = models_dir / f"{model_name}_step_{step}_emergency.pt"
        agent.save(str(emergency_checkpoint))
        print(f"[RESUME] Emergency checkpoint saved: {emergency_checkpoint.name}")

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()


