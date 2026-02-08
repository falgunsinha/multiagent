import sys
from pathlib import Path
import numpy as np
import json
import csv
from datetime import datetime
import argparse


project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
masac_path = Path(__file__).parent
sys.path.insert(0, str(masac_path))
mappo_path = Path(__file__).parent.parent / "MAPPO"
sys.path.insert(0, str(mappo_path))
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
from isaacsim.core.api import World
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
import numpy as np
from src.rl.doubleDQN import DoubleDQNAgent
from src.rl.object_selection_env_rrt import ObjectSelectionEnvRRT
from agents.masac_continuous_wrapper import MASACContinuousWrapper
from envs.two_agent_env import TwoAgentEnv


class FrankaRRTTrainer:
    """
    Franka controller for RRT-based testing (reachability checks only, no execution).
   
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
        print("[TRAINER] Adding container...")
        self._setup_container()
        self._spawn_cubes()
        print("[TRAINER] Creating random obstacles...")
        self._create_random_obstacles()
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
        from isaacsim.core.prims import SingleXFormPrim

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
        occupied_cells = set()
        for row in range(self.training_grid_size):
            for col in range(self.training_grid_size):
                cell_x = start_x + (row * cube_spacing)
                cell_y = start_y + (col * cube_spacing)
                for cube_pos in self.cube_positions:
                    if np.linalg.norm(cube_pos[:2] - np.array([cell_x, cell_y])) < 0.05:
                        occupied_cells.add((row, col))
                        break
        empty_cells = []
        for row in range(self.training_grid_size):
            for col in range(self.training_grid_size):
                if (row, col) not in occupied_cells:
                    empty_cells.append((row, col))
        if len(empty_cells) < num_obstacles:
            num_obstacles = len(empty_cells)

        selected_cells = np.random.choice(len(empty_cells), size=num_obstacles, replace=False)
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
        total_cells = self.training_grid_size * self.training_grid_size
        selected_indices = np.random.choice(total_cells, size=self.num_cubes, replace=False)
        selected_cells = set(selected_indices)
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
    """
    print("[ENV] Creating Isaac Sim environment...")
    print(f"[ENV] Grid: {grid_size}x{grid_size}, Cubes: {num_cubes}")
    trainer = FrankaRRTTrainer(num_cubes=num_cubes, training_grid_size=grid_size)
    trainer.setup_scene()
    max_objects = grid_size * grid_size
    env = ObjectSelectionEnvRRT(
        franka_controller=trainer,  
        max_objects=max_objects,
        max_steps=max_steps,
        num_cubes=num_cubes,
        training_grid_size=grid_size,
        execute_picks=False, 
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
    print(f" Loaded DDQN model: {model_path}")
    return agent


def test_masac_isaacsim_configuration(
    grid_size: int,
    num_cubes: int,
    num_episodes: int = 5,
    log_dir: str = "cobotproject/scripts/Reinforcement Learning/MASAC/logs"
):
    """Test MASAC on Isaac Sim RRT configuration"""
    
    print(f"\n{'='*80}")
    print(f"Testing MASAC: RRT_ISAACSIM | Grid {grid_size}x{grid_size} | {num_cubes} cubes")
    print(f"{'='*80}")
    
    # DDQN model mapping
    ddqn_model_mapping = {
        "rrt_isaacsim_grid3_cubes4": "ddqn_rrt_isaacsim_grid3_cubes4_20251223_203144_final.pt",
        "rrt_isaacsim_grid4_cubes6": "ddqn_rrt_isaacsim_grid4_cubes6_20251224_122040_final.pt",
        "rrt_isaacsim_grid4_cubes9": "ddqn_rrt_isaacsim_grid4_cubes9_20251224_185752_final.pt",
    }
    
    config_key = f"rrt_isaacsim_grid{grid_size}_cubes{num_cubes}"
    
    if config_key not in ddqn_model_mapping:
        print(f"  No DDQN model mapping for {config_key}")
        return None
    
    ddqn_model_filename = ddqn_model_mapping[config_key]
    ddqn_models_dir = project_root / "scripts" / "Reinforcement Learning" / "doubleDQN_script" / "models"
    ddqn_model_path = ddqn_models_dir / ddqn_model_filename
    
    if not ddqn_model_path.exists():
        print(f" DDQN model not found: {ddqn_model_path}")
        return None
    
    # Create Isaac Sim environment with RRT planner
    print("Creating Isaac Sim environment with RRT planner...")
    base_env, world = create_isaacsim_environment(
        grid_size=grid_size,
        num_cubes=num_cubes,
        max_steps=50
    )
    ddqn_agent = load_ddqn_agent(str(ddqn_model_path), base_env)
    agent2_state_dim = (num_cubes * 3) + 3 + num_cubes + (grid_size * grid_size) + 10
    pretrained_path = project_root / "scripts" / "Reinforcement Learning" / "MASAC" / "pretrained_models"
    cube_spacing = 0.13 if grid_size > 3 else 0.15  # Match DDQN training spacing
    
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
    print("[TEST] Relaxing reshuffling thresholds for testing...")
    two_agent_env.reshuffle_decision.min_reachable_distance = 0.30  # Was 0.35
    two_agent_env.reshuffle_decision.max_reachable_distance = 0.90  # Was 0.85
    two_agent_env.reshuffle_decision.path_length_ratio_threshold = 1.5  # Was 1.8
    two_agent_env.reshuffle_decision.crowded_threshold = 2  # Was 3
    two_agent_env.reshuffle_decision.rrt_failure_window = 2  # Was 3
    two_agent_env.reshuffle_decision.min_clearance = 0.35  # Was 0.3
    two_agent_env.reshuffle_decision.far_cube_ratio = 1.1  # Was 1.2
    two_agent_env.reshuffle_decision.batch_reshuffle_count = 2  # Was 3
    print("[TEST] Reshuffling thresholds relaxed!")

    # Enable test mode to skip expensive reachability checks during PCA fitting
    print("[TEST] Enabling test mode for fast PCA fitting...")
    base_env.test_mode = True
    print("[TEST] Test mode enabled! Reachability checks will be skipped.")
    masac_agent.fit_dimension_adapter(two_agent_env, n_samples=500)
    masac_agent.set_test_mode(True)
    episode_results = []

    for episode in range(num_episodes):
        # Randomize cube positions for each episode (adds spatial variation)
        if hasattr(base_env, 'franka_controller') and base_env.franka_controller is not None:
            base_env.franka_controller.randomize_cube_positions()

        obs, reset_info = two_agent_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        reshuffles_performed = 0

        # DEBUG: Print initial state
        print(f"\n[DEBUG Episode {episode+1}] Reset complete")
        print(f"  Cubes in scene: {reset_info.get('cubes_picked', 'N/A')}")
        cube_positions = two_agent_env.base_env.get_cube_positions()
        robot_pos = two_agent_env.base_env.get_robot_position()
        print(f"  Robot position: {robot_pos}")
        print(f"  Cube positions ({len(cube_positions)} cubes):")
        for i, pos in enumerate(cube_positions):
            dist = np.linalg.norm(pos - robot_pos)
            print(f"    Cube {i}: {pos} (dist={dist:.3f}m)")

        while not (done or truncated) and episode_length < 50:
            # Calculate valid cubes for action masking (not picked, not reshuffled 2+ times)
            valid_cubes = [
                i for i in range(num_cubes)
                if i not in two_agent_env.base_env.objects_picked
                and two_agent_env.reshuffle_count_per_cube.get(i, 0) < 2
            ]
            action_dict = masac_agent.select_action(obs, deterministic=True, valid_cubes=valid_cubes)
            if action_dict is None:
                print(f"  [WARNING] No valid cubes to reshuffle, skipping step")
                break
            action_int = two_agent_env.reshuffle_action_space.encode_action(
                cube_idx=action_dict['cube_idx'],
                grid_x=action_dict['target_grid_x'],
                grid_y=action_dict['target_grid_y']
            )
            print(f"\n[DEBUG Step {episode_length+1}]")
            print(f"  MASAC action: cube={action_dict['cube_idx']}, grid=({action_dict['target_grid_x']}, {action_dict['target_grid_y']})")
            next_obs, reward, done, truncated, info = two_agent_env.step(action_int)
            print(f"  DDQN selected: cube {info.get('agent1_action', 'N/A')}")
            print(f"  Reshuffle decision: {info.get('reshuffle_reason', 'N/A')}")
            print(f"  Reshuffled: {info.get('reshuffled_this_step', False)}")
            print(f"  Reward: {reward:.2f}")
            print(f"  Done: {done}, Truncated: {truncated}")

            episode_reward += reward
            episode_length += 1

            if info.get('reshuffled_this_step', False):
                reshuffles_performed += 1

            obs = next_obs

        # Log episode results
        result = {
            'episode': episode + 1,
            'total_reward': float(episode_reward),
            'episode_length': episode_length,
            'reshuffles_performed': reshuffles_performed,
            'total_distance_reduced': float(two_agent_env.total_distance_reduced),
            'total_time_saved': float(two_agent_env.total_time_saved),
            'cubes_picked': len(two_agent_env.base_env.objects_picked)
        }
        episode_results.append(result)

        print(f"Episode {episode+1}/{num_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Reshuffles={reshuffles_performed}, "
              f"Distance={two_agent_env.total_distance_reduced:.3f}m, "
              f"Cubes={len(two_agent_env.base_env.objects_picked)}/{num_cubes}")
    save_results("rrt_isaacsim", grid_size, num_cubes, episode_results, log_dir)

    return episode_results


def save_results(env_type: str, grid_size: int, num_cubes: int, episode_results: list, timestep_results: list, log_dir: str):
    """Save test results to CSV and JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    csv_filename = f"masac_{env_type}_grid{grid_size}_cubes{num_cubes}_{timestamp}_episode_log.csv"
    csv_path = log_path / csv_filename

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=episode_results[0].keys())
        writer.writeheader()
        writer.writerows(episode_results)

    print(f" Saved episode CSV log: {csv_path}")

    # NEW: Save Timestep-level CSV
    if timestep_results:
        timestep_csv_filename = f"masac_{env_type}_grid{grid_size}_cubes{num_cubes}_{timestamp}_timestep_log.csv"
        timestep_csv_path = log_path / timestep_csv_filename

        with open(timestep_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=timestep_results[0].keys())
            writer.writeheader()
            writer.writerows(timestep_results)

        print(f" Saved timestep CSV log: {timestep_csv_path}")
    summary = {
        'env_type': env_type,
        'grid_size': grid_size,
        'num_cubes': num_cubes,
        'num_episodes': len(episode_results),
        'total_timesteps': len(timestep_results) if timestep_results else 0,
        'avg_reward': np.mean([r['total_reward'] for r in episode_results]),
        'avg_reshuffles': np.mean([r['reshuffles_performed'] for r in episode_results]),
        'avg_distance_reduced': np.mean([r['total_distance_reduced'] for r in episode_results]),
        'avg_cubes_picked': np.mean([r['cubes_picked'] for r in episode_results]),
        'timestamp': timestamp
    }

    json_filename = f"masac_{env_type}_grid{grid_size}_cubes{num_cubes}_{timestamp}_summary.json"
    json_path = log_path / json_filename

    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f" Saved JSON summary: {json_path}")


def main():
    """Test MASAC on Isaac Sim RRT configurations"""
    parser = argparse.ArgumentParser(description='Test MASAC on Isaac Sim RRT')
    parser.add_argument('--episodes', type=int, default=5, help='Number of test episodes per config')
    parser.add_argument('--log_dir', type=str, default='cobotproject/scripts/Reinforcement Learning/MASAC/logs',
                        help='Directory to save logs')
    args = parser.parse_args()

    # Isaac Sim RRT configurations
    configurations = [
        (3, 4),   # Grid 3x3, 4 cubes
        (4, 6),   # Grid 4x4, 6 cubes
        (4, 9),   # Grid 4x4, 9 cubes
    ]

    print(f"MASAC TESTING - ISAAC SIM RRT (3 CONFIGURATIONS)")
    print(f"Episodes per config: {args.episodes}")
    print(f"Log directory: {args.log_dir}")
    print(f"{'='*80}\n")

    all_results = {}

    for grid_size, num_cubes in configurations:
        try:
            results = test_masac_isaacsim_configuration(
                grid_size=grid_size,
                num_cubes=num_cubes,
                num_episodes=args.episodes,
                log_dir=args.log_dir
            )

            if results:
                config_key = f"rrt_isaacsim_grid{grid_size}_cubes{num_cubes}"
                all_results[config_key] = results

            # Clean up scene between configurations
            print("\n[CLEANUP] Clearing scene for next configuration...")
            try:
                from omni.isaac.core.utils.stage import clear_stage
                clear_stage()
                print("[CLEANUP] Scene cleared successfully")
            except Exception as cleanup_error:
                print(f"[CLEANUP] Warning: Could not clear stage: {cleanup_error}")

        except Exception as e:
            print(f" Error testing grid{grid_size} cubes{num_cubes}: {e}")
            import traceback
            traceback.print_exc()

            # Try to clean up even after error
            try:
                from omni.isaac.core.utils.stage import clear_stage
                clear_stage()
                print("[CLEANUP] Scene cleared after error")
            except:
                pass

    print(f"\n{'='*80}")
    print("MASAC TESTING COMPLETE")
    print(f"{'='*80}")
    print(f"Tested {len(all_results)}/{len(configurations)} configurations successfully")
    print(f"Results saved to: {args.log_dir}")
    print(f"{'='*80}\n")

    # Close Isaac Sim properly
    print("Closing Isaac Sim...")
    try:
        simulation_app.close()
        print("Isaac Sim closed successfully")
    except Exception as e:
        print(f"Warning: Error closing Isaac Sim: {e}")

    # Exit with success code
    import sys
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

        # Try to close Isaac Sim even on error
        try:
            simulation_app.close()
        except:
            pass

        # Exit with error code
        import sys
        sys.exit(1)


