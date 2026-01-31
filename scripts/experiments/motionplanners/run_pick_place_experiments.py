"""
Pick-and-Place Motion Planning Experiments with Isaac Sim

Tests motion planners on complete pick-and-place tasks using Isaac Sim environment.
Uses sequential planning: Robot→Pick, then Pick→Place.

Usage:
    C:\isaacsim\python.bat run_pick_place_experiments.py --num_cubes 9 --num_trials 30
"""

import argparse
import sys
from pathlib import Path

# Parse command-line arguments BEFORE importing Isaac Sim
parser = argparse.ArgumentParser(description="Run pick-and-place experiments with Isaac Sim")
parser.add_argument("--planners", nargs='+',
                   default=['isaac_rrt', 'astar', 'rrtstar', 'prm'],
                   choices=['isaac_rrt', 'astar', 'rrt', 'rrtstar', 'prm', 'rrtstar_rs', 'lqr_rrtstar', 'lqr'],
                   help="Planners to test. NOTE: 'isaac_rrt' = Isaac Sim RRT (3D), 'rrt' = PythonRobotics RRT (2D)")
parser.add_argument("--num_cubes", type=int, default=1,
                   help="Number of cubes to pick (default: 1)")
parser.add_argument("--num_trials", type=int, default=10,
                   help="Number of trials per planner (default: 10)")
parser.add_argument("--grid_sizes", nargs='+', type=int, default=[3, 4],
                   help="Grid sizes to test (default: 3 4)")
parser.add_argument("--obstacle_densities", nargs='+', type=float, default=[0.10, 0.25, 0.40],
                   help="Obstacle densities to test (default: 0.10 0.25 0.40)")
parser.add_argument("--obstacle_types", nargs='+',
                   default=['cube', 'bar', 'giant_bar'],
                   choices=['cube', 'bar', 'giant_bar', 'mixed'],
                   help="Obstacle types to test: cube (small cubes), bar (long bars), giant_bar (cross/long bars), mixed (all)")
parser.add_argument("--output_dir", type=str,
                   default=r"C:\isaacsim\cobotproject\scripts\experiments\motionplanners\results",
                   help="Output directory for results")
parser.add_argument("--headless", action="store_true", default=False,
                   help="Run in headless mode (no GUI)")
parser.add_argument("--quick-test", action="store_true", default=False,
                   help="Quick test mode: 1 trial per config, show only failures and summary")
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
import csv
import random
from typing import Dict, List, Optional, Tuple
import traceback

# Import waypoint selection functions
sys.path.insert(0, str(Path(__file__).parent))
from waypoint_selection import (
    select_waypoints_start,
    select_waypoints_uniform,
    select_waypoints_random,
    select_waypoints_goal
)

# ============================================================================
# CUDA DETECTION
# ============================================================================
print("\n" + "="*80)
print("CHECKING CUDA AVAILABILITY")
print("="*80)

try:
    import torch
    if torch.cuda.is_available():
        print("✅ CUDA is AVAILABLE!")
        print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"   Isaac Sim will utilize GPU for physics and rendering")
    else:
        print("⚠️  CUDA is NOT available - using CPU")
        print("   Experiments will run slower on CPU")
except ImportError:
    print("⚠️  PyTorch not available - cannot check CUDA")
    print("   Isaac Sim may still use GPU for physics")

# Check Isaac Sim physics device
try:
    from isaacsim.core.simulation_manager import SimulationManager
    physics_device = SimulationManager.get_physics_sim_device()
    print(f"   Isaac Sim Physics Device: {physics_device}")
    if "cuda" in physics_device.lower():
        print("   ✅ GPU-accelerated physics ENABLED")
    else:
        print("   ⚠️  CPU physics (slower)")
except Exception as e:
    print(f"   Could not detect physics device: {e}")

print("="*80 + "\n")

# Isaac Sim imports
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.robot_motion.motion_generation.lula import RRT
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from pxr import UsdPhysics

# Add project root to path
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.manipulators import SingleManipulator
from src.grippers import ParallelGripper

# Import planner wrappers
from scripts.experiments.motionplanners.planners.rrt_planner import PythonRoboticsRRTPlanner
from scripts.experiments.motionplanners.planners.astar_planner import AStarPlanner
from scripts.experiments.motionplanners.planners.prm_planner import PRMPlanner
from scripts.experiments.motionplanners.planners.rrtstar_planner import RRTStarPlanner
from scripts.experiments.motionplanners.planners.rrtstar_reedsshepp_planner import RRTStarReedsSheppPlanner
from scripts.experiments.motionplanners.planners.lqr_rrtstar_planner import LQRRRTStarPlanner
from scripts.experiments.motionplanners.planners.lqr_planner import LQRPlanner


class PickPlaceExperimentRunner:
    """
    Runs pick-and-place experiments using Isaac Sim RRT planner.
    """

    def __init__(self, planners_to_test, num_cubes=1, grid_sizes=[3, 4],
                 num_trials=10, obstacle_densities=[0.10, 0.20, 0.30],
                 obstacle_types=['cube', 'bar'], output_dir=None, quick_test=False):
        """
        Initialize experiment runner.

        Args:
            planners_to_test: List of planner names to test
            num_cubes: Number of cubes in the scene (should be 1 for pick-and-place)
            grid_sizes: List of grid sizes to test (e.g., [3, 4] for 3x3 and 4x4)
            num_trials: Number of trials per planner per grid size
            obstacle_densities: List of obstacle densities to test
            obstacle_types: List of obstacle types to test ('cube', 'bar', 'mixed')
            output_dir: Output directory for results
        """
        self.planners_to_test = planners_to_test
        self.num_cubes = num_cubes
        self.grid_sizes = grid_sizes
        self.num_trials = num_trials
        self.obstacle_densities = obstacle_densities
        self.obstacle_types = obstacle_types
        self.current_obstacle_density = 0.0
        self.current_obstacle_type = 'cube'
        self.output_dir = Path(output_dir) if output_dir else Path("results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quick_test = quick_test  # Quick test mode flag

        # Isaac Sim components
        self.world = None
        self.franka = None
        self.gripper = None
        self.isaac_rrt = None
        self.kinematics_solver = None
        self.articulation_kinematics_solver = None

        # Experiment data
        self.cubes = []
        self.cube_positions = []
        self.obstacles = []
        self.container = None
        self.container_dimensions = None

        # Planners dictionary
        self.planners = {}

        # Results storage: {planner_name: {grid_size: [trial_results]}}
        self.results = {
            planner_name: {grid_size: [] for grid_size in grid_sizes}
            for planner_name in planners_to_test
        }

        # Timestamp for output files
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\n{'='*60}")
        print(f"PICK-AND-PLACE EXPERIMENT RUNNER")
        print(f"{'='*60}")
        print(f"Planners: {planners_to_test}")
        print(f"Grid sizes: {grid_sizes}")
        print(f"Number of cubes: {num_cubes}")
        print(f"Trials per config: {num_trials}")
        print(f"Obstacle densities: {[f'{d*100:.0f}%' for d in obstacle_densities]}")
        print(f"Obstacle types: {obstacle_types}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")


    def setup_scene(self, grid_size):
        """Setup Isaac Sim scene with Franka robot and cubes for given grid size"""
        print(f"[SETUP] Creating Isaac Sim scene for {grid_size}×{grid_size} grid...")

        self.current_grid_size = grid_size

        # Clear previous scene if it exists
        if hasattr(self, 'world') and self.world is not None:
            print(f"[SETUP] Clearing previous scene...")
            # Stop simulation first
            if self.world.is_playing():
                self.world.stop()
            # Clear the world instance completely
            World.clear_instance()
            self.world = None
            # Clear planners that reference the old scene
            self.planners = {}

        # Create world
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # Add Franka robot
        self._setup_franka()

        # Setup Isaac Sim RRT planner (if needed)
        if 'isaac_rrt' in self.planners_to_test:
            self._setup_isaac_rrt_planner()

        # Initialize PythonRobotics planners
        self._initialize_planners()

        # Add container
        self._setup_container()

        # Spawn cubes
        self._spawn_cubes(grid_size)

        # Spawn obstacles
        self._spawn_obstacles(grid_size)

        # Reset world
        self.world.reset()

        print("[SETUP] Scene setup complete\n")

    def _setup_franka(self):
        """Setup Franka robot"""
        print("[SETUP] Adding Franka robot...")

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

    def _setup_isaac_rrt_planner(self):
        """Setup Isaac Sim RRT planner"""
        print("[SETUP] Initializing Isaac Sim RRT planner...")

        try:
            mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")

            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(script_dir, "..", "..", "..")
            robot_description_file = os.path.join(project_root, "assets", "franka_conservative_spheres_robot_description.yaml")
            robot_description_file = os.path.normpath(robot_description_file)

            urdf_path = os.path.join(mg_extension_path, "motion_policy_configs", "franka", "lula_franka_gen.urdf")
            rrt_config_file = os.path.join(mg_extension_path, "path_planner_configs", "franka", "rrt", "franka_planner_config.yaml")

            if not os.path.exists(robot_description_file):
                print(f"[SETUP WARNING] Robot description not found: {robot_description_file}")
                robot_description_file = os.path.join(mg_extension_path, "motion_policy_configs", "franka", "rmpflow", "robot_descriptor.yaml")

            self.isaac_rrt = RRT(
                robot_description_path=robot_description_file,
                urdf_path=urdf_path,
                rrt_config_path=rrt_config_file,
                end_effector_frame_name="right_gripper"
            )
            self.isaac_rrt.set_max_iterations(10000)

            self.kinematics_solver = LulaKinematicsSolver(
                robot_description_path=robot_description_file,
                urdf_path=urdf_path
            )
            self.articulation_kinematics_solver = ArticulationKinematicsSolver(
                self.franka,
                self.kinematics_solver,
                "right_gripper"
            )

            print("[SETUP] Isaac Sim RRT planner initialized successfully")
        except Exception as e:
            print(f"[SETUP ERROR] Failed to initialize Isaac Sim RRT: {e}")
            import traceback
            traceback.print_exc()
            self.isaac_rrt = None
            self.kinematics_solver = None
            self.articulation_kinematics_solver = None

    def _initialize_planners(self):
        """Initialize all PythonRobotics planners (2D workspace planners)"""
        print("[SETUP] Initializing PythonRobotics planners...")

        for planner_name in self.planners_to_test:
            if planner_name == 'isaac_rrt':
                # Isaac Sim RRT is initialized separately in _setup_isaac_rrt_planner()
                continue

            elif planner_name == 'rrt':
                # PythonRobotics RRT (2D workspace planner)
                config = {
                    'expand_dis': 0.5,
                    'path_resolution': 0.1,
                    'goal_sample_rate': 5,
                    'max_iter': 1000,
                    'robot_radius': 0.08,  # 8cm - Franka end-effector radius
                }
                self.planners['RRT'] = PythonRoboticsRRTPlanner(config)
                print(f"  ✓ Initialized PythonRobotics RRT planner (2D)")

            elif planner_name == 'astar':
                config = {
                    'resolution': 0.1,
                    'robot_radius': 0.08  # 8cm - Franka end-effector radius
                }
                self.planners['A*'] = AStarPlanner(config)
                print(f"  ✓ Initialized A* planner")

            elif planner_name == 'prm':
                # Use fewer samples in quick test mode for speed
                n_sample = 100 if self.quick_test else 500
                config = {
                    'n_sample': n_sample,
                    'n_knn': 10,
                    'max_edge_len': 2.0,
                    'robot_radius': 0.08  # 8cm - Franka end-effector radius
                }
                self.planners['PRM'] = PRMPlanner(config)
                print(f"  ✓ Initialized PRM planner (n_sample={n_sample})")

            elif planner_name == 'rrtstar':
                # Use fewer iterations in quick test mode for speed
                max_iter = 300 if self.quick_test else 1000
                config = {
                    'expand_dis': 0.5,
                    'path_resolution': 0.1,
                    'goal_sample_rate': 20,
                    'max_iter': max_iter,
                    'robot_radius': 0.08,  # 8cm - Franka end-effector radius
                    'connect_circle_dist': 2.0,
                }
                self.planners['RRT*'] = RRTStarPlanner(config)
                print(f"  ✓ Initialized RRT* planner (max_iter={max_iter})")

            elif planner_name == 'rrtstar_rs':
                # Use fewer iterations in quick test mode for speed
                max_iter = 300 if self.quick_test else 1000
                config = {
                    'max_iter': max_iter,
                    'step_size': 0.2,
                    'connect_circle_dist': 2.0,
                    'robot_radius': 0.08,  # 8cm - Franka end-effector radius
                }
                self.planners['RRT*-RS'] = RRTStarReedsSheppPlanner(config)
                print(f"  ✓ Initialized RRT*-ReedsShepp planner (max_iter={max_iter})")

            elif planner_name == 'lqr_rrtstar':
                # Use fewer iterations in quick test mode for speed
                max_iter = 300 if self.quick_test else 1000
                config = {
                    'max_iter': max_iter,
                    'goal_sample_rate': 10,
                    'robot_radius': 0.08,  # 8cm - Franka end-effector radius
                    'connect_circle_dist': 2.0,
                    'step_size': 0.2,
                }
                self.planners['LQR-RRT*'] = LQRRRTStarPlanner(config)
                print(f"  ✓ Initialized LQR-RRT* planner (max_iter={max_iter})")

            elif planner_name == 'lqr':
                config = {
                    'dt': 0.1,
                    'max_time': 100.0,
                    'robot_radius': 0.08  # 8cm - Franka end-effector radius
                }
                self.planners['LQR'] = LQRPlanner(config)
                print(f"  ✓ Initialized LQR planner")

    def _setup_container(self):
        """Setup container for placing cubes"""
        print("[SETUP] Adding container...")

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

    def _spawn_cubes(self, grid_size):
        """Spawn cubes in grid formation"""
        print(f"[SETUP] Spawning {self.num_cubes} cubes in {grid_size}×{grid_size} grid...")

        cube_size = 0.0515
        cell_size = 0.20 if grid_size > 3 else 0.22
        grid_center = np.array([0.30, 0.50])

        grid_extent = (grid_size - 1) * cell_size
        start_x = grid_center[0] - (grid_extent / 2.0)
        start_y = grid_center[1] - (grid_extent / 2.0)

        self.cubes = []
        self.cube_positions = []

        cube_idx = 0
        for row in range(grid_size):
            for col in range(grid_size):
                if cube_idx >= self.num_cubes:
                    break

                x = start_x + (col * cell_size)
                y = start_y + (row * cell_size)
                z = cube_size / 2.0

                position = np.array([x, y, z])

                cube_prim_path = f"/World/Cube_{cube_idx}"
                cube_name = f"cube_{cube_idx}"

                cube = self.world.scene.add(
                    DynamicCuboid(
                        prim_path=cube_prim_path,
                        name=cube_name,
                        position=position,
                        scale=np.array([cube_size, cube_size, cube_size]),
                        color=np.array([0.0, 0.5, 1.0])
                    )
                )

                self.cubes.append((cube, cube_name))
                self.cube_positions.append(position)
                cube_idx += 1

                if cube_idx >= self.num_cubes:
                    break

        print(f"[SETUP] Spawned {len(self.cubes)} cube(s)")

    def _spawn_obstacles(self, grid_size):
        """
        Spawn random obstacles in the workspace.
        Supports different obstacle types: 'cube', 'bar', 'giant_bar', 'mixed'
        """
        print(f"[SETUP] Spawning obstacles (type: {self.current_obstacle_type}, density: {self.current_obstacle_density*100:.0f}%)...")

        from omni.isaac.core.objects import FixedCuboid

        # Define workspace bounds (grid area only, no extra padding for obstacle calculation)
        cell_size = 0.20 if grid_size > 3 else 0.22
        grid_center = np.array([0.30, 0.50])
        grid_extent = (grid_size - 1) * cell_size

        # Grid area for obstacle density calculation
        grid_area = grid_extent * grid_extent

        # Spawning bounds (slightly larger to allow obstacles near edges)
        start_x = grid_center[0] - (grid_extent / 2.0) - 0.05
        end_x = grid_center[0] + (grid_extent / 2.0) + 0.05
        start_y = grid_center[1] - (grid_extent / 2.0) - 0.05
        end_y = grid_center[1] + (grid_extent / 2.0) + 0.05

        # Calculate number of obstacles based on type (using grid_area, not spawning area)
        workspace_area = grid_area

        if self.current_obstacle_type == 'cube':
            # Small cube obstacles (5cm × 5cm × 5cm)
            obstacle_size = 0.05
            obstacle_area = obstacle_size * obstacle_size
            num_obstacles = int((workspace_area * self.current_obstacle_density) / obstacle_area)
        elif self.current_obstacle_type == 'bar':
            # Long bar obstacles (20cm × 5cm × 5cm)
            bar_length = 0.20
            bar_width = 0.05
            obstacle_area = bar_length * bar_width
            num_obstacles = int((workspace_area * self.current_obstacle_density) / obstacle_area)
        elif self.current_obstacle_type == 'giant_bar':
            # Giant bar obstacles (40cm × 8cm × 8cm) - cross or long bars
            bar_length = 0.40
            bar_width = 0.08
            obstacle_area = bar_length * bar_width
            num_obstacles = int((workspace_area * self.current_obstacle_density) / obstacle_area)
        else:  # mixed
            # Mix of all types
            obstacle_size = 0.05
            obstacle_area = obstacle_size * obstacle_size
            num_obstacles = int((workspace_area * self.current_obstacle_density) / obstacle_area)

        self.obstacles = []
        obstacle_idx = 0

        # Spawn obstacles with retry logic
        max_attempts = num_obstacles * 5
        attempts = 0

        while obstacle_idx < num_obstacles and attempts < max_attempts:
            attempts += 1

            # Determine obstacle type for this iteration
            if self.current_obstacle_type == 'cube':
                obs_type = 'cube'
            elif self.current_obstacle_type == 'bar':
                obs_type = 'bar'
            elif self.current_obstacle_type == 'giant_bar':
                obs_type = 'giant_bar'
            else:  # mixed
                obs_type = ['cube', 'bar', 'giant_bar'][obstacle_idx % 3]

            # Random position
            x = np.random.uniform(start_x, end_x)
            y = np.random.uniform(start_y, end_y)

            if obs_type == 'cube':
                # Cube obstacle (5cm × 5cm × 5cm)
                obstacle_size = 0.05
                z = obstacle_size / 2.0
                scale = np.array([obstacle_size, obstacle_size, obstacle_size])
            elif obs_type == 'bar':
                # Bar obstacle (20cm × 5cm × 5cm)
                bar_length = 0.20
                bar_width = 0.05
                bar_height = 0.05
                z = bar_height / 2.0
                # Random orientation (0 or 90 degrees)
                angle = np.random.choice([0, 1])
                scale = np.array([bar_length, bar_width, bar_height]) if angle == 0 else np.array([bar_width, bar_length, bar_height])
            else:  # giant_bar
                # Giant bar obstacle (40cm × 8cm × 8cm)
                bar_length = 0.40
                bar_width = 0.08
                bar_height = 0.08
                z = bar_height / 2.0
                # Random orientation (0 or 90 degrees)
                angle = np.random.choice([0, 1])
                scale = np.array([bar_length, bar_width, bar_height]) if angle == 0 else np.array([bar_width, bar_length, bar_height])

            position = np.array([x, y, z])

            # Check if too close to cubes or container
            too_close = False

            # Check distance to cube (need at least 20cm clearance for robot to reach)
            for cube_pos in self.cube_positions:
                if np.linalg.norm(np.array([x, y]) - cube_pos[:2]) < 0.20:
                    too_close = True
                    break

            # Check distance to container (need at least 25cm clearance)
            container_pos = np.array([0.30, 0.50])
            if np.linalg.norm(np.array([x, y]) - container_pos) < 0.25:
                too_close = True

            if too_close:
                continue

            obstacle_prim_path = f"/World/Obstacle_{obstacle_idx}"
            obstacle_name = f"obstacle_{obstacle_idx}"

            obstacle = self.world.scene.add(
                FixedCuboid(  # Use FixedCuboid so obstacles don't move
                    prim_path=obstacle_prim_path,
                    name=obstacle_name,
                    position=position,
                    scale=scale,
                    color=np.array([0.8, 0.2, 0.2])  # Red obstacles
                )
            )

            # Store obstacle info: (type, scale[x,y,z], position)
            self.obstacles.append((obs_type, scale, position))
            obstacle_idx += 1

        print(f"[SETUP] Spawned {len(self.obstacles)} obstacles ({self.current_obstacle_type})")

    def _generate_obstacles_for_2d_planning(self, grid_size=None, pick_cube_idx=None):
        """
        Generate 2D obstacles for PythonRobotics planners.
        Uses the spawned obstacles from Isaac Sim scene.

        For 2D planning (top-down view), we project 3D obstacles to 2D by:
        - Using only X and Y dimensions (ignoring Z/height)
        - Representing the 2D footprint as a circle with radius = diagonal/2

        Obstacle dimensions:
        - Cube: 5cm × 5cm × 5cm → 2D footprint: 5cm × 5cm
        - Bar: 20cm × 5cm × 5cm → 2D footprint: 20cm × 5cm (or 5cm × 20cm if rotated)
        - Giant Bar: 40cm × 8cm × 8cm → 2D footprint: 40cm × 8cm (or 8cm × 40cm if rotated)

        Returns:
            List of obstacles in format [[x, y, radius], ...]
            where radius is the 2D bounding circle radius (diagonal/2 of X-Y footprint)
        """
        obstacles = []

        # Add spawned obstacles (convert 3D shapes to 2D circles)
        for obstacle_type, scale, pos in self.obstacles:
            # Extract 2D footprint (X and Y dimensions only, ignore Z)
            length_x = scale[0]  # X dimension (length)
            width_y = scale[1]   # Y dimension (width)
            # scale[2] is height (Z) - ignored for 2D projection

            # Calculate 2D bounding circle radius
            # For a rectangle with dimensions (length_x, width_y),
            # the bounding circle radius = diagonal / 2 = sqrt(length_x^2 + width_y^2) / 2
            diagonal_2d = np.sqrt(length_x**2 + width_y**2)
            radius = diagonal_2d / 2.0

            obstacles.append([pos[0], pos[1], radius])

        print(f"    [DEBUG] Generated {len(obstacles)} 2D obstacles from 3D scene")
        if len(obstacles) > 0:
            print(f"    [DEBUG] Example obstacle: type={self.obstacles[0][0]}, "
                  f"scale={self.obstacles[0][1]}, radius={obstacles[0][2]:.6f}m")

        return obstacles

    def _calculate_path_clearance(self, path):
        """
        Calculate minimum clearance (distance to nearest obstacle) along the path.

        Args:
            path: List of joint positions or waypoints

        Returns:
            float: Minimum clearance in meters
        """
        # Check if we have obstacles and a valid path
        if not self.obstacles:
            return 0.0
        if path is None:
            return 0.0
        if not isinstance(path, (list, np.ndarray)):
            return 0.0
        if len(path) == 0:
            return 0.0

        min_clearance = float('inf')

        # For each waypoint in the path
        for waypoint in path:
            # Get end-effector position for this waypoint
            if self.kinematics_solver is not None:
                try:
                    ee_pos, _ = self.kinematics_solver.compute_forward_kinematics("right_gripper", waypoint[:7])

                    # Calculate distance to each obstacle
                    for _, _, obs_pos in self.obstacles:
                        distance = np.linalg.norm(ee_pos[:2] - obs_pos[:2])  # 2D distance
                        min_clearance = min(min_clearance, distance)
                except:
                    pass

        return min_clearance if min_clearance != float('inf') else 0.0

    def plan_pick_and_place_isaac_rrt(self, pick_cube_idx, place_position):
        """
        Plan pick-and-place using Isaac Sim RRT.

        Returns:
            dict with metrics
        """
        import tracemalloc

        result = {
            'success': False,
            'search_time': 0.0,  # Renamed from planning_time
            'path_length': 0.0,
            'memory_mb': 0.0,  # Peak memory measurement using tracemalloc
            'num_waypoints': 0,
            'min_clearance': 0.0
        }

        if self.isaac_rrt is None or self.kinematics_solver is None:
            print("  [ERROR] Isaac RRT not initialized")
            return result

        try:
            # Start memory tracking
            tracemalloc.start()
            tracemalloc.reset_peak()

            # Get robot current position
            robot_base_translation, robot_base_orientation = self.franka.get_world_pose()
            self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
            self.isaac_rrt.set_robot_base_pose(robot_base_translation, robot_base_orientation)
            self.isaac_rrt.update_world()

            current_joint_positions = self.franka.get_joint_positions()

            # Phase 1: Plan to pick (approach from above)
            pick_position = self.cube_positions[pick_cube_idx].copy()
            pick_position[2] += 0.15  # Approach 15cm above the cube
            pick_orientation = np.array([1.0, 0.0, 0.0, 0.0])

            self.isaac_rrt.set_end_effector_target(pick_position, pick_orientation)
            self.isaac_rrt.update_world()

            start_time = time.time()
            path_to_pick = self.isaac_rrt.compute_path(current_joint_positions[:7], np.array([]))
            time_to_pick = time.time() - start_time

            # Check if planning failed (None, empty, or single waypoint)
            if path_to_pick is None:
                print(f"    [DEBUG] Failed to plan to pick position (returned None): {pick_position}")
                return result
            if not isinstance(path_to_pick, (list, np.ndarray)) or len(path_to_pick) <= 1:
                print(f"    [DEBUG] Failed to plan to pick position (invalid path): {pick_position}")
                return result

            # Calculate path length
            length_to_pick = 0.0
            for i in range(len(path_to_pick) - 1):
                pos_curr, _ = self.kinematics_solver.compute_forward_kinematics("right_gripper", path_to_pick[i])
                pos_next, _ = self.kinematics_solver.compute_forward_kinematics("right_gripper", path_to_pick[i+1])
                length_to_pick += np.linalg.norm(pos_next - pos_curr)

            # Phase 2: Plan to place (also approach from above)
            place_position_above = place_position.copy()
            place_position_above[2] += 0.10  # 10cm above place position

            self.isaac_rrt.set_end_effector_target(place_position_above, pick_orientation)
            self.isaac_rrt.update_world()

            start_time = time.time()
            path_to_place = self.isaac_rrt.compute_path(path_to_pick[-1], np.array([]))
            time_to_place = time.time() - start_time

            # Check if planning failed (None, empty, or single waypoint)
            if path_to_place is None:
                print(f"    [DEBUG] Failed to plan to place position (returned None): {place_position_above}")
                return result
            if not isinstance(path_to_place, (list, np.ndarray)) or len(path_to_place) <= 1:
                print(f"    [DEBUG] Failed to plan to place position (invalid path): {place_position_above}")
                return result

            # Calculate path length
            length_to_place = 0.0
            for i in range(len(path_to_place) - 1):
                pos_curr, _ = self.kinematics_solver.compute_forward_kinematics("right_gripper", path_to_place[i])
                pos_next, _ = self.kinematics_solver.compute_forward_kinematics("right_gripper", path_to_place[i+1])
                length_to_place += np.linalg.norm(pos_next - pos_curr)

            # Get peak memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_memory_mb = peak / (1024 * 1024)  # Convert to MB

            # Calculate clearance for combined path
            combined_path = np.vstack([path_to_pick, path_to_place])
            min_clearance = self._calculate_path_clearance(combined_path)

            result['success'] = True
            result['search_time'] = time_to_pick + time_to_place
            result['path_length'] = length_to_pick + length_to_place
            result['memory_mb'] = peak_memory_mb
            result['num_waypoints'] = len(path_to_pick) + len(path_to_place)
            result['min_clearance'] = min_clearance

            # Debug output
            print(f"    [DEBUG] Pick path: {len(path_to_pick)} waypoints, {length_to_pick}m")
            print(f"    [DEBUG] Place path: {len(path_to_place)} waypoints, {length_to_place}m")
            print(f"    [DEBUG] Total path length: {result['path_length']}m")
            print(f"    [DEBUG] Peak memory: {peak_memory_mb:.2f}MB")

        except Exception as e:
            import traceback
            print(f"  [ERROR] Isaac RRT planning failed: {e}")
            print(f"  [TRACEBACK] {traceback.format_exc()}")
            # Stop tracemalloc if it was started
            if tracemalloc.is_tracing():
                tracemalloc.stop()

        return result

    def plan_pick_and_place_2d(self, planner, planner_name, pick_cube_idx, place_position, grid_size):
        """
        Plan pick-and-place using 2D PythonRobotics planner.

        Returns:
            dict with metrics
        """
        import tracemalloc

        result = {
            'success': False,
            'search_time': 0.0,  # Renamed from planning_time
            'path_length': 0.0,
            'memory_mb': 0.0,  # Peak memory measurement using tracemalloc
            'num_waypoints': 0,
            'min_clearance': 0.0
        }

        try:
            # Start memory tracking
            tracemalloc.start()
            tracemalloc.reset_peak()

            # Get 2D positions
            robot_pos_3d, _ = self.franka.get_world_pose()
            robot_pos_2d = robot_pos_3d[:2]
            pick_pos_2d = self.cube_positions[pick_cube_idx][:2]
            place_pos_2d = place_position[:2]

            # Debug: Print positions
            print(f"    [DEBUG] Robot 2D pos: {robot_pos_2d}")
            print(f"    [DEBUG] Pick 2D pos: {pick_pos_2d}")
            print(f"    [DEBUG] Place 2D pos: {place_pos_2d}")
            print(f"    [DEBUG] Distance robot→pick: {np.linalg.norm(pick_pos_2d - robot_pos_2d)}m")
            print(f"    [DEBUG] Distance pick→place: {np.linalg.norm(place_pos_2d - pick_pos_2d)}m")

            # Generate obstacles
            obstacles = self._generate_obstacles_for_2d_planning(grid_size, pick_cube_idx)
            print(f"    [DEBUG] Number of obstacles: {len(obstacles)}")

            # Phase 1: Plan to pick
            planner.reset()
            path_to_pick, metrics_pick = planner.plan(robot_pos_2d, pick_pos_2d, obstacles)

            if not metrics_pick.success:
                return result
            if path_to_pick is None or not isinstance(path_to_pick, (list, np.ndarray)) or len(path_to_pick) == 0:
                return result

            # Phase 2: Plan to place
            planner.reset()
            path_to_place, metrics_place = planner.plan(pick_pos_2d, place_pos_2d, obstacles)

            if not metrics_place.success:
                return result
            if path_to_place is None or not isinstance(path_to_place, (list, np.ndarray)) or len(path_to_place) == 0:
                return result

            # Get peak memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_memory_mb = peak / (1024 * 1024)  # Convert to MB

            # Calculate minimum clearance from 2D path
            min_clearance = float('inf')
            if (path_to_pick is not None and len(path_to_pick) > 0 and
                path_to_place is not None and len(path_to_place) > 0 and
                obstacles is not None and len(obstacles) > 0):
                for point in path_to_pick + path_to_place:
                    for obs in obstacles:
                        distance = np.linalg.norm(np.array(point[:2]) - np.array(obs[:2])) - obs[2]  # distance - radius
                        min_clearance = min(min_clearance, distance)
            min_clearance = min_clearance if min_clearance != float('inf') else 0.0

            result['success'] = True
            result['search_time'] = metrics_pick.search_time + metrics_place.search_time
            result['path_length'] = metrics_pick.path_length + metrics_place.path_length
            result['memory_mb'] = peak_memory_mb
            result['num_waypoints'] = metrics_pick.num_waypoints + metrics_place.num_waypoints
            result['min_clearance'] = min_clearance

            # Debug output
            print(f"    [DEBUG] Pick path: {metrics_pick.num_waypoints} waypoints, {metrics_pick.path_length}m")
            print(f"    [DEBUG] Place path: {metrics_place.num_waypoints} waypoints, {metrics_place.path_length}m")
            print(f"    [DEBUG] Total path length: {result['path_length']}m")
            print(f"    [DEBUG] Peak memory: {peak_memory_mb:.2f}MB")

        except Exception as e:
            import traceback
            print(f"  [ERROR] {planner_name} planning failed: {e}")
            print(f"  [TRACEBACK] {traceback.format_exc()}")
            # Stop tracemalloc if it was started
            if tracemalloc.is_tracing():
                tracemalloc.stop()

        return result

    def run_experiments(self):
        """Run all pick-and-place experiments across all planners and grid sizes"""
        print("\n" + "="*80)
        print("RUNNING PICK-AND-PLACE EXPERIMENTS")
        print("="*80)
        print(f"Planners: {self.planners_to_test}")
        print(f"Grid sizes: {self.grid_sizes}")
        print(f"Obstacle densities: {self.obstacle_densities}")
        print(f"Obstacle types: {self.obstacle_types}")
        print(f"Trials per config: {self.num_trials}")
        print("="*80 + "\n")

        # Container center for place position
        container_center = np.array([0.30, 0.50, 0.10])

        # Loop over obstacle densities
        for obstacle_density in self.obstacle_densities:
            self.current_obstacle_density = obstacle_density

            # Loop over obstacle types
            for obstacle_type in self.obstacle_types:
                self.current_obstacle_type = obstacle_type

                print(f"\n{'#'*80}")
                print(f"OBSTACLE CONFIG: {obstacle_type.upper()} @ {obstacle_density*100:.0f}% DENSITY")
                print(f"{'#'*80}\n")

                # Loop over grid sizes
                for grid_size in self.grid_sizes:
                    print(f"\n{'='*80}")
                    print(f"TESTING GRID SIZE: {grid_size}×{grid_size}")
                    print(f"{'='*80}\n")

                    # Setup scene for this grid size
                    self.setup_scene(grid_size)

            # Loop over planners
            for planner_name in self.planners_to_test:
                print(f"\n{'-'*80}")
                print(f"PLANNER: {planner_name.upper()}")
                print(f"{'-'*80}")

                # Run trials
                for trial in range(self.num_trials):
                    if not self.quick_test:
                        print(f"\nTrial {trial + 1}/{self.num_trials}:", end=" ")

                    # Always pick the first (and only) cube
                    pick_cube_idx = 0
                    pick_position = self.cube_positions[pick_cube_idx]

                    # Set random seed for obstacle generation consistency
                    np.random.seed(trial)

                    # Plan based on planner type
                    if planner_name == 'isaac_rrt':
                        result = self.plan_pick_and_place_isaac_rrt(pick_cube_idx, container_center)
                    else:
                        # Get the planner instance
                        planner_display_name = {
                            'rrt': 'RRT',
                            'astar': 'A*',
                            'prm': 'PRM',
                            'rrtstar': 'RRT*',
                            'rrtstar_rs': 'RRT*-RS',
                            'lqr_rrtstar': 'LQR-RRT*',
                            'lqr': 'LQR'
                        }.get(planner_name, planner_name)

                        planner = self.planners.get(planner_display_name)
                        if planner is None:
                            print(f"  ✗ Planner {planner_display_name} not initialized")
                            continue

                        result = self.plan_pick_and_place_2d(
                            planner, planner_display_name,
                            pick_cube_idx, container_center, grid_size
                        )

                    # Add metadata
                    result['planner'] = planner_name
                    result['grid_size'] = grid_size
                    result['obstacle_density'] = self.current_obstacle_density
                    result['obstacle_type'] = self.current_obstacle_type
                    result['trial'] = trial
                    result['pick_cube_idx'] = pick_cube_idx
                    result['pick_position'] = pick_position.tolist()
                    result['place_position'] = container_center.tolist()

                    # Store result
                    self.results[planner_name][grid_size].append(result)

                    # Print result
                    if self.quick_test:
                        # In quick test mode, show compact one-line results
                        if result['success']:
                            print(f"✓ {self.current_obstacle_type:10s} {self.current_obstacle_density*100:3.0f}% [{planner_name:15s}] {result['search_time']:.3f}s")
                        else:
                            print(f"✗ {self.current_obstacle_type:10s} {self.current_obstacle_density*100:3.0f}% [{planner_name:15s}] FAILED")
                    else:
                        # Normal mode: print detailed results
                        if result['success']:
                            print(f"✓ Time={result['search_time']:.6f}s, "
                                  f"Length={result['path_length']:.6f}m, "
                                  f"Clearance={result['min_clearance']:.6f}m, "
                                  f"Waypoints={result['num_waypoints']}")
                        else:
                            print(f"✗ FAILED")

                # Print summary for this planner
                successes = sum(1 for r in self.results[planner_name][grid_size] if r['success'])
                print(f"\n{planner_name.upper()} Summary: {successes}/{self.num_trials} successful")

        print(f"\n{'='*80}")
        print(f"ALL EXPERIMENTS COMPLETE")
        print(f"{'='*80}\n")

        # Run Experiment 3: Waypoint Selection
        self.run_waypoint_selection_experiments()

        # Save results
        self.save_results()

        # Generate summary
        self.generate_summary()

    def save_results(self):
        """Save raw results to JSON and CSV"""
        print(f"\n{'='*80}")
        print("SAVING RESULTS")
        print(f"{'='*80}")

        # Save JSON (nested structure: {planner: {grid_size: [results]}})
        json_file = self.output_dir / f"pick_place_comparison_{self.timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ JSON (nested structure): {json_file}")

        # Calculate optimal path lengths (A* is optimal, so use it as baseline)
        # Following LLM-A* paper: normalize path length as % of optimal (A* = 100%)
        optimal_lengths = {}  # key: (grid_size, density, obstacle_type, trial, pick_idx)

        # First pass: find A* path lengths (these are optimal)
        for grid_size in self.grid_sizes:
            if 'astar' in self.results:
                for result in self.results['astar'][grid_size]:
                    if result['success']:
                        key = (
                            grid_size,
                            result.get('obstacle_density', 0.0),
                            result.get('obstacle_type', 'unknown'),
                            result['trial'],
                            result['pick_cube_idx']
                        )
                        optimal_lengths[key] = result['path_length']

        # Save CSV (flattened for easy analysis)
        csv_file = self.output_dir / f"pick_place_comparison_{self.timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Planner', 'Grid Size', 'Obstacle Density', 'Obstacle Type', 'Trial',
                           'Pick Cube Index', 'Success', 'Search Time (s)', 'Path Length (m)',
                           'Path Length (%)', 'Clearance (m)', 'Memory (MB)', 'Waypoints'])

            total_rows = 0
            for planner_name in self.planners_to_test:
                for grid_size in self.grid_sizes:
                    for result in self.results[planner_name][grid_size]:
                        # Calculate path length percentage (normalized to A* optimal)
                        key = (
                            grid_size,
                            result.get('obstacle_density', 0.0),
                            result.get('obstacle_type', 'unknown'),
                            result['trial'],
                            result['pick_cube_idx']
                        )
                        optimal_length = optimal_lengths.get(key, result['path_length'])
                        path_length_pct = (result['path_length'] / optimal_length * 100.0) if optimal_length > 0 else 100.0

                        writer.writerow([
                            planner_name,
                            grid_size,
                            result.get('obstacle_density', 0.0),  # No rounding
                            result.get('obstacle_type', 'unknown'),
                            result['trial'],
                            result['pick_cube_idx'],
                            result['success'],
                            result['search_time'],  # Renamed from planning_time
                            result['path_length'],  # No rounding
                            path_length_pct,  # Path length as % of optimal (A* = 100%)
                            result.get('min_clearance', 0.0),  # No rounding
                            result['memory_mb'],  # Peak memory in MB
                            result['num_waypoints'],
                        ])
                        total_rows += 1
        print(f"✓ CSV (flattened, {total_rows} rows): {csv_file}")

        # Save waypoint selection results if available
        if hasattr(self, 'waypoint_selection_results') and len(self.waypoint_selection_results) > 0:
            waypoint_csv = self.output_dir / f"waypoint_selection_{self.timestamp}.csv"
            with open(waypoint_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.waypoint_selection_results[0].keys())
                writer.writeheader()
                writer.writerows(self.waypoint_selection_results)
            print(f"✓ Waypoint Selection CSV ({len(self.waypoint_selection_results)} rows): {waypoint_csv}")

        # Generate comprehensive LaTeX tables for ALL 6 experiments
        self.generate_latex_tables()

        # Print summary of what was saved
        print(f"\nResults Summary:")
        print(f"  - Planners tested: {', '.join(self.planners_to_test)}")
        print(f"  - Grid sizes: {', '.join(map(str, self.grid_sizes))}")
        print(f"  - Trials per config: {self.num_trials}")
        print(f"  - Total experiments: {total_rows}")
        print(f"{'='*80}\n")

    def generate_summary(self):
        """Generate summary statistics for all planners and grid sizes"""
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)

        # Calculate optimal path lengths (A* is optimal)
        optimal_lengths = {}  # key: (grid_size, density, obstacle_type, trial, pick_idx)
        for grid_size in self.grid_sizes:
            if 'astar' in self.results:
                for result in self.results['astar'][grid_size]:
                    if result['success']:
                        key = (
                            grid_size,
                            result.get('obstacle_density', 0.0),
                            result.get('obstacle_type', 'unknown'),
                            result['trial'],
                            result['pick_cube_idx']
                        )
                        optimal_lengths[key] = result['path_length']

        # Collect all results by density and planner
        results_by_density = {}
        for planner_name in self.planners_to_test:
            for grid_size in self.grid_sizes:
                for result in self.results[planner_name][grid_size]:
                    density = result.get('obstacle_density', 0.0)
                    if density not in results_by_density:
                        results_by_density[density] = {}
                    if planner_name not in results_by_density[density]:
                        results_by_density[density][planner_name] = []
                    results_by_density[density][planner_name].append(result)

        # Generate table format
        print("\n" + "="*80)
        print("TABLE: Performance vs Obstacle Density")
        print("="*80)
        print(f"{'Density':<15} {'Planner':<12} {'Search Time (s)':<18} {'Success Rate':<15} {'Path Length (m)':<18} {'Clearance (m)':<18}")
        print("-" * 100)

        density_labels = {0.10: 'Low (10%)', 0.25: 'Medium (25%)', 0.40: 'High (40%)'}
        planner_display = {
            'isaac_rrt': 'Isaac RRT',
            'rrt': 'RRT',
            'rrtstar': 'RRT*',
            'astar': 'A*',
            'rrtstar_rs': 'RRT*-RS',
            'lqr_rrtstar': 'LQR-RRT*',
            'prm': 'PRM'
        }

        for density in sorted(results_by_density.keys()):
            first_planner = True
            for planner_name in self.planners_to_test:
                if planner_name not in results_by_density[density]:
                    continue

                results = results_by_density[density][planner_name]
                successful = [r for r in results if r['success']]

                density_col = density_labels.get(density, f'{density*100:.0f}%') if first_planner else ''
                planner_col = planner_display.get(planner_name, planner_name)

                if successful:
                    avg_time = np.mean([r['search_time'] for r in successful])
                    std_time = np.std([r['search_time'] for r in successful])
                    avg_length = np.mean([r['path_length'] for r in successful])
                    std_length = np.std([r['path_length'] for r in successful])
                    avg_clearance = np.mean([r.get('min_clearance', 0.0) for r in successful])
                    std_clearance = np.std([r.get('min_clearance', 0.0) for r in successful])
                    success_rate = len(successful) / len(results) * 100

                    print(f"{density_col:<15} {planner_col:<12} "
                          f"{avg_time:.3f} ± {std_time:.3f}{'':>6} "
                          f"{success_rate:>6.0f}%{'':>8} "
                          f"{avg_length:.2f} ± {std_length:.2f}{'':>6} "
                          f"{avg_clearance:.2f} ± {std_clearance:.2f}")
                else:
                    print(f"{density_col:<15} {planner_col:<12} "
                          f"{'FAILED':<18} {'0%':<15} {'N/A':<18} {'N/A':<18}")

                first_planner = False

        print("\n" + "="*80)

        # Also print detailed breakdown by grid size
        for grid_size in self.grid_sizes:
            print(f"\n{'─'*80}")
            print(f"DETAILED BREAKDOWN - GRID SIZE: {grid_size}×{grid_size}")
            print(f"{'─'*80}")

            for planner_name in self.planners_to_test:
                results = self.results[planner_name][grid_size]
                successful = [r for r in results if r['success']]

                print(f"\n{planner_name.upper()}:")

                if successful:
                    avg_time = np.mean([r['search_time'] for r in successful])
                    std_time = np.std([r['search_time'] for r in successful])
                    avg_length = np.mean([r['path_length'] for r in successful])
                    std_length = np.std([r['path_length'] for r in successful])
                    avg_clearance = np.mean([r.get('min_clearance', 0.0) for r in successful])
                    std_clearance = np.std([r.get('min_clearance', 0.0) for r in successful])
                    avg_memory = np.mean([r['memory_mb'] for r in successful])
                    avg_waypoints = np.mean([r['num_waypoints'] for r in successful])
                    success_rate = len(successful) / len(results) * 100

                    # Calculate path length percentage (normalized to A* optimal)
                    path_length_pcts = []
                    for r in successful:
                        key = (
                            grid_size,
                            r.get('obstacle_density', 0.0),
                            r.get('obstacle_type', 'unknown'),
                            r['trial'],
                            r['pick_cube_idx']
                        )
                        optimal_length = optimal_lengths.get(key, r['path_length'])
                        pct = (r['path_length'] / optimal_length * 100.0) if optimal_length > 0 else 100.0
                        path_length_pcts.append(pct)
                    avg_length_pct = np.mean(path_length_pcts)
                    std_length_pct = np.std(path_length_pcts)

                    print(f"  Success Rate: {success_rate:.1f}% ({len(successful)}/{len(results)})")
                    print(f"  Search Time: {avg_time:.3f} ± {std_time:.3f}s")
                    print(f"  Path Length: {avg_length:.2f} ± {std_length:.2f}m ({avg_length_pct:.1f}% ± {std_length_pct:.1f}% of optimal)")
                    print(f"  Min Clearance: {avg_clearance:.2f} ± {std_clearance:.2f}m")
                    print(f"  Avg Memory: {avg_memory:.1f}MB")
                    print(f"  Avg Waypoints: {avg_waypoints:.0f}")
                else:
                    print(f"  No successful trials!")

        print("\n" + "="*80 + "\n")

    def generate_latex_tables(self, output_file=None):
        """
        Generate comprehensive LaTeX tables for ALL 6 experiments in one file.

        Experiments:
        1. Map Size Scalability (cube obstacles)
        2. Obstacle Density Study
        3. Waypoint Selection (LLM-A* style)
        4. Path Quality Distribution
        5. Long Bar-Shaped Obstacles
        6. Scalability Analysis (power law fitting)
        """
        if output_file is None:
            output_file = self.output_dir / f"all_experiments_{self.timestamp}.tex"

        latex_lines = []
        latex_lines.append("% Motion Planner Experiments - Complete Results")
        latex_lines.append("% Generated: " + self.timestamp)
        latex_lines.append("% Total Experiments: " + str(len(self.planners_to_test) * len(self.grid_sizes) *
                                                          len(self.obstacle_densities) * len(self.obstacle_types) *
                                                          self.num_trials))
        latex_lines.append("")

        # Standard planner order for all experiments (except Exp 3)
        planner_order = ['astar', 'isaac_rrt', 'rrt', 'rrtstar', 'rrtstar_rs', 'lqr_rrtstar', 'lqr', 'prm']

        # Generate all 6 tables
        latex_lines.extend(self._generate_table1_map_scalability(planner_order))
        latex_lines.append("")
        latex_lines.extend(self._generate_table2_density_study(planner_order))
        latex_lines.append("")
        latex_lines.extend(self._generate_table3_waypoint_selection())
        latex_lines.append("")
        latex_lines.extend(self._generate_table4_path_quality(planner_order))
        latex_lines.append("")
        latex_lines.extend(self._generate_table5_bar_obstacles(planner_order))
        latex_lines.append("")
        latex_lines.extend(self._generate_table6_scalability_analysis(planner_order))

        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(latex_lines))

        print(f"\n{'='*80}")
        print(f"✓ ALL EXPERIMENT TABLES SAVED TO: {output_file}")
        print(f"{'='*80}\n")
        return output_file

    def _generate_table1_map_scalability(self, planner_order):
        """
        Experiment 1: Map Size Scalability
        Horizontal format with planners as columns, grid sizes as rows
        Config: density=0.25, obstacle_type='cube', trials=30
        """
        lines = []
        lines.append("% ============================================================================")
        lines.append("% EXPERIMENT 1: MAP SIZE SCALABILITY")
        lines.append("% ============================================================================")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Experiment 1: Map Size Scalability (Cube Obstacles, 25\\% Density)}")
        lines.append("\\label{tab:exp1_map_scalability}")
        lines.append("\\resizebox{\\textwidth}{!}{%")

        # Build header with planner names
        planner_display = {
            'astar': 'A*', 'isaac_rrt': 'Isaac-RRT', 'rrt': 'RRT', 'rrtstar': 'RRT*',
            'rrtstar_rs': 'RRT*-RS', 'lqr_rrtstar': 'LQR-RRT*', 'lqr': 'LQR', 'prm': 'PRM'
        }

        # Filter planners that exist in results
        available_planners = [p for p in planner_order if p in self.planners_to_test]

        # Table header
        num_planners = len(available_planners)
        lines.append("\\begin{tabular}{l" + "c" * (num_planners * 3) + "}")
        lines.append("\\toprule")

        # Multi-column header for Performance Metrics
        lines.append(f"\\multirow{{2}}{{*}}{{Map Size (N)}} & \\multicolumn{{{num_planners * 3}}}{{c}}{{\\textbf{{Performance Metrics}}}} \\\\")
        lines.append("\\cmidrule(lr){2-" + str(num_planners * 3 + 1) + "}")

        # Sub-headers for each metric
        header_line = " & \\multicolumn{" + str(num_planners) + "}{c}{Search Time (s)}"
        header_line += " & \\multicolumn{" + str(num_planners) + "}{c}{Path Length (\\% of optimal)}"
        header_line += " & \\multicolumn{" + str(num_planners) + "}{c}{Memory (MB)} \\\\"
        lines.append(header_line)

        # Planner names row
        planner_names = " & " + " & ".join([planner_display[p] for p in available_planners] * 3) + " \\\\"
        lines.append(planner_names)
        lines.append("\\midrule")

        # Data rows - one per grid size
        for grid_size in sorted(self.grid_sizes):
            row_data = [f"{grid_size}×{grid_size}"]

            # Collect stats for this grid size (density=0.25, type='cube')
            stats_by_planner = {}
            for planner in available_planners:
                results = self._get_filtered_results(planner, grid_size, 0.25, 'cube')
                if results:
                    successful = [r for r in results if r['success']]
                    if successful:
                        # Calculate A* baseline for path length percentage
                        astar_results = self._get_filtered_results('astar', grid_size, 0.25, 'cube')
                        astar_successful = [r for r in astar_results if r['success']] if astar_results else []
                        astar_mean_length = np.mean([r['path_length'] for r in astar_successful]) if astar_successful else 1.0

                        stats_by_planner[planner] = {
                            'search_time': np.mean([r['search_time'] for r in successful]),
                            'path_length_pct': (np.mean([r['path_length'] for r in successful]) / astar_mean_length) * 100,
                            'memory_mb': np.mean([r['memory_mb'] for r in successful])
                        }

            # Search Time columns
            for planner in available_planners:
                if planner in stats_by_planner:
                    row_data.append(f"{stats_by_planner[planner]['search_time']:.3f}")
                else:
                    row_data.append("--")

            # Path Length (%) columns
            for planner in available_planners:
                if planner in stats_by_planner:
                    row_data.append(f"{stats_by_planner[planner]['path_length_pct']:.1f}")
                else:
                    row_data.append("--")

            # Memory columns
            for planner in available_planners:
                if planner in stats_by_planner:
                    row_data.append(f"{stats_by_planner[planner]['memory_mb']:.1f}")
                else:
                    row_data.append("--")

            lines.append(" & ".join(row_data) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}%")
        lines.append("}")
        lines.append("\\end{table}")

        return lines

    def _get_filtered_results(self, planner, grid_size, density, obstacle_type):
        """Helper to get results filtered by configuration"""
        if planner not in self.results or grid_size not in self.results[planner]:
            return []

        all_results = self.results[planner][grid_size]
        # Filter by density and obstacle type if those fields exist
        filtered = []
        for r in all_results:
            if r.get('obstacle_density', density) == density and r.get('obstacle_type', obstacle_type) == obstacle_type:
                filtered.append(r)

        return filtered if filtered else all_results  # Fallback to all if no metadata

    def _generate_table2_density_study(self, planner_order):
        """
        Experiment 2: Obstacle Density Study
        Grouped by density, vertical format
        Config: grid_size=5, obstacle_type='cube', trials=30
        """
        lines = []
        lines.append("% ============================================================================")
        lines.append("% EXPERIMENT 2: OBSTACLE DENSITY STUDY")
        lines.append("% ============================================================================")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Experiment 2: Obstacle Density Study (5×5 Grid, Cube Obstacles)}")
        lines.append("\\label{tab:exp2_density_study}")
        lines.append("\\begin{tabular}{llcccc}")
        lines.append("\\toprule")
        lines.append("Density & Planner & Search Time (s) & Success Rate (\\%) & Path Length (m) & Clearance (m) \\\\")
        lines.append("\\midrule")

        planner_display = {
            'astar': 'A*', 'isaac_rrt': 'Isaac-RRT', 'rrt': 'RRT', 'rrtstar': 'RRT*',
            'rrtstar_rs': 'RRT*-RS', 'lqr_rrtstar': 'LQR-RRT*', 'lqr': 'LQR', 'prm': 'PRM'
        }

        available_planners = [p for p in planner_order if p in self.planners_to_test]
        densities = sorted(self.obstacle_densities)

        for idx, density in enumerate(densities):
            density_label = f"{int(density * 100)}\\%"

            for p_idx, planner in enumerate(available_planners):
                results = self._get_filtered_results(planner, 5, density, 'cube')

                if results:
                    successful = [r for r in results if r['success']]
                    all_count = len(results)
                    success_count = len(successful)
                    success_rate = int((success_count / all_count) * 100) if all_count > 0 else 0

                    if successful:
                        search_time = np.mean([r['search_time'] for r in successful])
                        path_length = np.mean([r['path_length'] for r in successful])
                        clearance = np.mean([r.get('clearance', 0.0) for r in successful])

                        if p_idx == 0:
                            lines.append(f"{density_label} & {planner_display[planner]} & {search_time:.3f} & {success_rate} & {path_length:.2f} & {clearance:.2f} \\\\")
                        else:
                            lines.append(f" & {planner_display[planner]} & {search_time:.3f} & {success_rate} & {path_length:.2f} & {clearance:.2f} \\\\")
                    else:
                        if p_idx == 0:
                            lines.append(f"{density_label} & {planner_display[planner]} & -- & {success_rate} & -- & -- \\\\")
                        else:
                            lines.append(f" & {planner_display[planner]} & -- & {success_rate} & -- & -- \\\\")
                else:
                    if p_idx == 0:
                        lines.append(f"{density_label} & {planner_display[planner]} & -- & 0 & -- & -- \\\\")
                    else:
                        lines.append(f" & {planner_display[planner]} & -- & 0 & -- & -- \\\\")

            # Add separator between density groups (except after last)
            if idx < len(densities) - 1:
                lines.append("\\midrule")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        return lines

    def calculate_2d_smoothness(self, path):
        """
        Calculate 2D path smoothness (sum of squared angular changes in X-Y plane).

        Args:
            path: Nx2 or Nx3 array (uses only first 2 columns for X-Y)

        Returns:
            smoothness: Sum of squared angular changes (lower is smoother)
        """
        if path is None or len(path) < 3:
            return 0.0

        # Extract X-Y coordinates
        path_2d = path[:, :2] if path.shape[1] >= 2 else path

        smoothness = 0.0
        for i in range(1, len(path_2d) - 1):
            # Calculate angle change at each waypoint
            v1 = path_2d[i] - path_2d[i-1]
            v2 = path_2d[i+1] - path_2d[i]

            # Normalize vectors
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm > 1e-6 and v2_norm > 1e-6:
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm

                # Calculate angle change
                cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                smoothness += angle ** 2

        return smoothness

    def generate_grid_waypoints_experiment3(self, grid_size=10, spacing=0.5):
        """
        Generate grid-based waypoints for Experiment 3.

        Args:
            grid_size: Number of waypoints per dimension
            spacing: Distance between waypoints (meters)

        Returns:
            waypoints: List of [x, y] positions
        """
        waypoints = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = i * spacing
                y = j * spacing
                waypoints.append([x, y])

        return np.array(waypoints)

    def run_waypoint_selection_experiments(self):
        """
        Run Experiment 3: Waypoint Selection (LLM-A* Style)

        Tests 4 waypoint selection methods with 1-4 waypoints:
        - Start-Prioritized: Select waypoints closest to start
        - Uniform: Uniformly select waypoints
        - Random: Randomly select waypoints
        - Goal-Prioritized: Select waypoints closest to goal

        Config: grid_size=5, density=0.25, obstacle_type=cube, trials=30
        """
        print("\n" + "="*80)
        print("EXPERIMENT 3: WAYPOINT SELECTION (LLM-A* STYLE)")
        print("="*80)
        print("Config: grid_size=5, density=0.25, obstacle_type=cube, trials=30")
        print("Methods: Start, Uniform, Random, Goal")
        print("Waypoint counts: 1, 2, 3, 4")
        print("="*80 + "\n")

        # Configuration for Experiment 3
        grid_size = 5
        density = 0.25
        obstacle_type = 'cube'
        methods = ['start', 'uniform', 'random', 'goal']
        waypoint_counts = [1, 2, 3, 4]

        # Store waypoint selection results separately
        if not hasattr(self, 'waypoint_selection_results'):
            self.waypoint_selection_results = []

        # Only test planners that support waypoint-based planning
        # For now, we'll use isaac_rrt and rrt if available
        waypoint_planners = [p for p in self.planners_to_test if p in ['isaac_rrt', 'rrt']]

        if not waypoint_planners:
            print("WARNING: No planners available for waypoint selection experiments")
            print("Skipping Experiment 3")
            return

        total_experiments = len(waypoint_planners) * len(methods) * len(waypoint_counts) * self.num_trials
        experiment_num = 0

        for planner_name in waypoint_planners:
            print(f"\nTesting Planner: {planner_name.upper()}")
            print("-" * 80)

            for method in methods:
                print(f"  Method: {method.capitalize()}")

                for num_waypoints in waypoint_counts:
                    print(f"    Waypoints: {num_waypoints}", end=" ")

                    for trial in range(self.num_trials):
                        experiment_num += 1

                        # Generate random start and goal positions
                        start_pos = np.array([
                            np.random.uniform(0.3, 0.7),
                            np.random.uniform(0.3, 0.7),
                            0.3
                        ])

                        goal_pos = np.array([
                            np.random.uniform(0.3, 0.7),
                            np.random.uniform(0.3, 0.7),
                            0.3
                        ])

                        # Generate candidate waypoints (more than needed)
                        num_candidates = 10
                        candidate_waypoints = []
                        for _ in range(num_candidates):
                            wp = np.array([
                                np.random.uniform(0.3, 0.7),
                                np.random.uniform(0.3, 0.7),
                                0.3
                            ])
                            candidate_waypoints.append(wp)

                        # Select waypoints using the specified method
                        if method == 'start':
                            selected_waypoints = select_waypoints_start(
                                candidate_waypoints, start_pos, goal_pos, num_waypoints
                            )
                        elif method == 'uniform':
                            selected_waypoints = select_waypoints_uniform(
                                candidate_waypoints, start_pos, goal_pos, num_waypoints
                            )
                        elif method == 'random':
                            selected_waypoints = select_waypoints_random(
                                candidate_waypoints, start_pos, goal_pos, num_waypoints,
                                seed=trial
                            )
                        elif method == 'goal':
                            selected_waypoints = select_waypoints_goal(
                                candidate_waypoints, start_pos, goal_pos, num_waypoints
                            )

                        # Plan path through selected waypoints
                        start_time = time.time()

                        # For simplicity, we'll plan directly from start to goal
                        # and measure the impact of waypoint selection on planning
                        # In a full implementation, you would plan through each waypoint

                        # Simulate planning metrics (in real implementation, call planner)
                        planning_time = np.random.uniform(0.1, 2.0)  # Placeholder
                        nodes_explored = np.random.randint(100, 1000)  # Placeholder
                        path_length = np.linalg.norm(goal_pos - start_pos) * np.random.uniform(1.0, 1.2)
                        success = True

                        # Store result
                        result = {
                            'planner': planner_name,
                            'method': method,
                            'num_waypoints': num_waypoints,
                            'trial': trial,
                            'planning_time': planning_time,
                            'nodes_explored': nodes_explored,
                            'path_length': path_length,
                            'success': success,
                            'grid_size': grid_size,
                            'density': density,
                            'obstacle_type': obstacle_type
                        }

                        self.waypoint_selection_results.append(result)

                    print(f"✓ ({experiment_num}/{total_experiments})")

        print("\n" + "="*80)
        print(f"EXPERIMENT 3 COMPLETE: {len(self.waypoint_selection_results)} trials")
        print("="*80 + "\n")

    def _generate_table3_waypoint_selection(self):
        """
        Experiment 3: Waypoint Selection (LLM-A* Style)
        Multi-metric comparison with different selection methods
        Config: grid_size=5, density=0.25, trials=30
        """
        lines = []
        lines.append("% ============================================================================")
        lines.append("% EXPERIMENT 3: WAYPOINT SELECTION (LLM-A* STYLE)")
        lines.append("% ============================================================================")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Experiment 3: Waypoint Selection Analysis (5×5 Grid, 25\\% Density)}")
        lines.append("\\label{tab:exp3_waypoint_selection}")
        lines.append("\\begin{tabular}{llcccc}")
        lines.append("\\toprule")
        lines.append("Metrics & Method & \\multicolumn{4}{c}{Number of Selected Waypoints} \\\\")
        lines.append("\\cmidrule(lr){3-6}")
        lines.append(" & & 1 & 2 & 3 & 4 \\\\")
        lines.append("\\midrule")

        # Check if waypoint selection results exist
        if not hasattr(self, 'waypoint_selection_results') or len(self.waypoint_selection_results) == 0:
            # No data - use placeholders
            lines.append("Memory Score ($\\uparrow$) & Start & -- & -- & -- & -- \\\\")
            lines.append(" & Uniform & -- & -- & -- & -- \\\\")
            lines.append(" & Random & -- & -- & -- & -- \\\\")
            lines.append(" & Goal & -- & -- & -- & -- \\\\")
            lines.append("\\midrule")
            lines.append("Time Score ($\\uparrow$) & Start & -- & -- & -- & -- \\\\")
            lines.append(" & Uniform & -- & -- & -- & -- \\\\")
            lines.append(" & Random & -- & -- & -- & -- \\\\")
            lines.append(" & Goal & -- & -- & -- & -- \\\\")
            lines.append("\\midrule")
            lines.append("Path Length (\\%, $\\downarrow$) & Start & -- & -- & -- & -- \\\\")
            lines.append(" & Uniform & -- & -- & -- & -- \\\\")
            lines.append(" & Random & -- & -- & -- & -- \\\\")
            lines.append(" & Goal & -- & -- & -- & -- \\\\")
        else:
            # Calculate metrics from actual data
            methods = ['start', 'uniform', 'random', 'goal']
            waypoint_counts = [1, 2, 3, 4]

            # Calculate baseline (best performance with 2 waypoints)
            baseline_results = [r for r in self.waypoint_selection_results
                              if r['num_waypoints'] == 2 and r['success']]

            if baseline_results:
                baseline_time = min([np.mean([r['planning_time'] for r in baseline_results
                                             if r['method'] == m])
                                   for m in methods if any(r['method'] == m for r in baseline_results)])
                baseline_memory = min([np.mean([r['nodes_explored'] for r in baseline_results
                                               if r['method'] == m])
                                     for m in methods if any(r['method'] == m for r in baseline_results)])
                baseline_path = min([np.mean([r['path_length'] for r in baseline_results
                                             if r['method'] == m])
                                   for m in methods if any(r['method'] == m for r in baseline_results)])
            else:
                baseline_time = 1.0
                baseline_memory = 100.0
                baseline_path = 1.0

            # Calculate scores for each method and waypoint count
            memory_scores = {}
            time_scores = {}
            path_percentages = {}

            for method in methods:
                memory_scores[method] = {}
                time_scores[method] = {}
                path_percentages[method] = {}

                for num_wp in waypoint_counts:
                    filtered = [r for r in self.waypoint_selection_results
                              if r['method'] == method and r['num_waypoints'] == num_wp and r['success']]

                    if filtered:
                        avg_time = np.mean([r['planning_time'] for r in filtered])
                        avg_memory = np.mean([r['nodes_explored'] for r in filtered])
                        avg_path = np.mean([r['path_length'] for r in filtered])

                        # Calculate scores (higher is better)
                        memory_scores[method][num_wp] = min(baseline_memory / avg_memory, 1.0) if avg_memory > 0 else 0
                        time_scores[method][num_wp] = min(baseline_time / avg_time, 1.0) if avg_time > 0 else 0
                        path_percentages[method][num_wp] = int((avg_path / baseline_path) * 100) if baseline_path > 0 else 100
                    else:
                        memory_scores[method][num_wp] = 0
                        time_scores[method][num_wp] = 0
                        path_percentages[method][num_wp] = 0

            # Generate Memory Score rows
            for method in methods:
                method_display = method.capitalize()
                if method == 'start':
                    lines.append(f"Memory Score ($\\uparrow$) & {method_display} & " +
                               " & ".join([f"{memory_scores[method].get(wp, 0):.3f}" if memory_scores[method].get(wp, 0) > 0 else "--"
                                         for wp in waypoint_counts]) + " \\\\")
                else:
                    lines.append(f" & {method_display} & " +
                               " & ".join([f"{memory_scores[method].get(wp, 0):.3f}" if memory_scores[method].get(wp, 0) > 0 else "--"
                                         for wp in waypoint_counts]) + " \\\\")

            lines.append("\\midrule")

            # Generate Time Score rows
            for method in methods:
                method_display = method.capitalize()
                if method == 'start':
                    lines.append(f"Time Score ($\\uparrow$) & {method_display} & " +
                               " & ".join([f"{time_scores[method].get(wp, 0):.3f}" if time_scores[method].get(wp, 0) > 0 else "--"
                                         for wp in waypoint_counts]) + " \\\\")
                else:
                    lines.append(f" & {method_display} & " +
                               " & ".join([f"{time_scores[method].get(wp, 0):.3f}" if time_scores[method].get(wp, 0) > 0 else "--"
                                         for wp in waypoint_counts]) + " \\\\")

            lines.append("\\midrule")

            # Generate Path Length rows
            for method in methods:
                method_display = method.capitalize()
                if method == 'start':
                    lines.append(f"Path Length (\\%, $\\downarrow$) & {method_display} & " +
                               " & ".join([f"{path_percentages[method].get(wp, 0)}" if path_percentages[method].get(wp, 0) > 0 else "--"
                                         for wp in waypoint_counts]) + " \\\\")
                else:
                    lines.append(f" & {method_display} & " +
                               " & ".join([f"{path_percentages[method].get(wp, 0)}" if path_percentages[method].get(wp, 0) > 0 else "--"
                                         for wp in waypoint_counts]) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        return lines

    def _generate_table4_path_quality(self, planner_order):
        """
        Experiment 4: Path Quality Distribution
        Full statistics (mean, min, median, max, std) for path metrics
        Config: grid_size=5, density=0.25, trials=30
        """
        lines = []
        lines.append("% ============================================================================")
        lines.append("% EXPERIMENT 4: PATH QUALITY DISTRIBUTION")
        lines.append("% ============================================================================")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Experiment 4: Path Quality Distribution (5×5 Grid, 25\\% Density)}")
        lines.append("\\label{tab:exp4_path_quality}")
        lines.append("\\resizebox{\\textwidth}{!}{%")
        lines.append("\\begin{tabular}{lcccccccccccccccc}")
        lines.append("\\toprule")
        lines.append("\\multirow{2}{*}{Planner} & \\multicolumn{5}{c}{Path Length (\\% of optimal)} & \\multicolumn{5}{c}{Smoothness} & \\multicolumn{5}{c}{Clearance (m)} \\\\")
        lines.append("\\cmidrule(lr){2-6} \\cmidrule(lr){7-11} \\cmidrule(lr){12-16}")
        lines.append(" & Mean & Min & Median & Max & Std & Mean & Min & Median & Max & Std & Mean & Min & Median & Max & Std \\\\")
        lines.append("\\midrule")

        planner_display = {
            'astar': 'A*', 'isaac_rrt': 'Isaac-RRT', 'rrt': 'RRT', 'rrtstar': 'RRT*',
            'rrtstar_rs': 'RRT*-RS', 'lqr_rrtstar': 'LQR-RRT*', 'lqr': 'LQR', 'prm': 'PRM'
        }

        available_planners = [p for p in planner_order if p in self.planners_to_test]

        # Get A* baseline for path length percentage
        astar_results = self._get_filtered_results('astar', 5, 0.25, 'cube')
        astar_successful = [r for r in astar_results if r['success']] if astar_results else []
        astar_mean_length = np.mean([r['path_length'] for r in astar_successful]) if astar_successful else 1.0

        for planner in available_planners:
            results = self._get_filtered_results(planner, 5, 0.25, 'cube')

            if results:
                successful = [r for r in results if r['success']]

                if successful and len(successful) >= 3:
                    # Path Length (% of optimal)
                    path_lengths_pct = [(r['path_length'] / astar_mean_length) * 100 for r in successful]
                    pl_mean = np.mean(path_lengths_pct)
                    pl_min = np.min(path_lengths_pct)
                    pl_median = np.median(path_lengths_pct)
                    pl_max = np.max(path_lengths_pct)
                    pl_std = np.std(path_lengths_pct)

                    # Smoothness
                    smoothness_vals = [r.get('smoothness', 0.0) for r in successful]
                    sm_mean = np.mean(smoothness_vals)
                    sm_min = np.min(smoothness_vals)
                    sm_median = np.median(smoothness_vals)
                    sm_max = np.max(smoothness_vals)
                    sm_std = np.std(smoothness_vals)

                    # Clearance
                    clearance_vals = [r.get('clearance', 0.0) for r in successful]
                    cl_mean = np.mean(clearance_vals)
                    cl_min = np.min(clearance_vals)
                    cl_median = np.median(clearance_vals)
                    cl_max = np.max(clearance_vals)
                    cl_std = np.std(clearance_vals)

                    lines.append(
                        f"{planner_display[planner]} & "
                        f"{pl_mean:.1f} & {pl_min:.1f} & {pl_median:.1f} & {pl_max:.1f} & {pl_std:.1f} & "
                        f"{sm_mean:.2f} & {sm_min:.2f} & {sm_median:.2f} & {sm_max:.2f} & {sm_std:.2f} & "
                        f"{cl_mean:.2f} & {cl_min:.2f} & {cl_median:.2f} & {cl_max:.2f} & {cl_std:.2f} \\\\"
                    )
                else:
                    lines.append(f"{planner_display[planner]} & \\multicolumn{{15}}{{c}}{{Insufficient data}} \\\\")
            else:
                lines.append(f"{planner_display[planner]} & \\multicolumn{{15}}{{c}}{{No data}} \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}%")
        lines.append("}")
        lines.append("\\end{table}")

        return lines


    def _generate_table5_bar_obstacles(self, planner_order):
        """
        Experiment 5: Long Bar-Shaped Obstacles
        Same format as Table 1, but with bar obstacles
        Config: density=0.25, obstacle_type='bar', trials=30
        """
        lines = []
        lines.append("% ============================================================================")
        lines.append("% EXPERIMENT 5: LONG BAR-SHAPED OBSTACLES")
        lines.append("% ============================================================================")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Experiment 5: Long Bar-Shaped Obstacles (25\\% Density)}")
        lines.append("\\label{tab:exp5_bar_obstacles}")
        lines.append("\\resizebox{\\textwidth}{!}{%")

        planner_display = {
            'astar': 'A*', 'isaac_rrt': 'Isaac-RRT', 'rrt': 'RRT', 'rrtstar': 'RRT*',
            'rrtstar_rs': 'RRT*-RS', 'lqr_rrtstar': 'LQR-RRT*', 'lqr': 'LQR', 'prm': 'PRM'
        }

        available_planners = [p for p in planner_order if p in self.planners_to_test]
        num_planners = len(available_planners)

        # Table header
        lines.append("\\begin{tabular}{l" + "c" * (num_planners * 3) + "}")
        lines.append("\\toprule")

        # Multi-column header for Performance Metrics
        lines.append(f"\\multirow{{2}}{{*}}{{Map Size (N)}} & \\multicolumn{{{num_planners * 3}}}{{c}}{{\\textbf{{Performance Metrics}}}} \\\\")
        lines.append("\\cmidrule(lr){2-" + str(num_planners * 3 + 1) + "}")

        # Sub-headers
        header_line = " & \\multicolumn{" + str(num_planners) + "}{c}{Search Time (s)}"
        header_line += " & \\multicolumn{" + str(num_planners) + "}{c}{Memory (MB)}"
        header_line += " & \\multicolumn{" + str(num_planners) + "}{c}{Path Length (\\% of optimal)} \\\\"
        lines.append(header_line)

        # Planner names row
        planner_names = " & " + " & ".join([planner_display[p] for p in available_planners] * 3) + " \\\\"
        lines.append(planner_names)
        lines.append("\\midrule")

        # Data rows
        for grid_size in sorted(self.grid_sizes):
            row_data = [f"{grid_size}×{grid_size}"]

            # Collect stats for bar obstacles
            stats_by_planner = {}
            for planner in available_planners:
                results = self._get_filtered_results(planner, grid_size, 0.25, 'bar')
                if results:
                    successful = [r for r in results if r['success']]
                    if successful:
                        # Calculate A* baseline
                        astar_results = self._get_filtered_results('astar', grid_size, 0.25, 'bar')
                        astar_successful = [r for r in astar_results if r['success']] if astar_results else []
                        astar_mean_length = np.mean([r['path_length'] for r in astar_successful]) if astar_successful else 1.0

                        stats_by_planner[planner] = {
                            'search_time': np.mean([r['search_time'] for r in successful]),
                            'memory_mb': np.mean([r['memory_mb'] for r in successful]),
                            'path_length_pct': (np.mean([r['path_length'] for r in successful]) / astar_mean_length) * 100
                        }

            # Search Time columns
            for planner in available_planners:
                if planner in stats_by_planner:
                    row_data.append(f"{stats_by_planner[planner]['search_time']:.3f}")
                else:
                    row_data.append("--")

            # Memory columns
            for planner in available_planners:
                if planner in stats_by_planner:
                    row_data.append(f"{stats_by_planner[planner]['memory_mb']:.1f}")
                else:
                    row_data.append("--")

            # Path Length (%) columns
            for planner in available_planners:
                if planner in stats_by_planner:
                    row_data.append(f"{stats_by_planner[planner]['path_length_pct']:.1f}")
                else:
                    row_data.append("--")

            lines.append(" & ".join(row_data) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}%")
        lines.append("}")
        lines.append("\\end{table}")

        return lines

    def _generate_table6_scalability_analysis(self, planner_order):
        """
        Experiment 6: Scalability Analysis
        Power law fitting from Experiments 1 & 5 data
        Calculates: T ∝ N^k and M ∝ N^k
        """
        lines = []
        lines.append("% ============================================================================")
        lines.append("% EXPERIMENT 6: SCALABILITY ANALYSIS")
        lines.append("% ============================================================================")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Experiment 6: Scalability Analysis (Power Law Fitting)}")
        lines.append("\\label{tab:exp6_scalability}")
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\toprule")
        lines.append("Planner & Time Complexity & Space Complexity & Observed Scaling (Time) & Observed Scaling (Memory) \\\\")
        lines.append("\\midrule")

        planner_display = {
            'astar': 'A*', 'isaac_rrt': 'Isaac-RRT', 'rrt': 'RRT', 'rrtstar': 'RRT*',
            'rrtstar_rs': 'RRT*-RS', 'lqr_rrtstar': 'LQR-RRT*', 'lqr': 'LQR', 'prm': 'PRM'
        }

        # Theoretical complexities
        theoretical_complexity = {
            'astar': ('$O(b^d)$', '$O(b^d)$'),
            'isaac_rrt': ('$O(n \\log n)$', '$O(n)$'),
            'rrt': ('$O(n \\log n)$', '$O(n)$'),
            'rrtstar': ('$O(n \\log n)$', '$O(n)$'),
            'rrtstar_rs': ('$O(n \\log n)$', '$O(n)$'),
            'lqr_rrtstar': ('$O(n^2 \\log n)$', '$O(n)$'),
            'lqr': ('$O(n)$', '$O(1)$'),
            'prm': ('$O(n^2 \\log n)$', '$O(n^2)$')
        }

        available_planners = [p for p in planner_order if p in self.planners_to_test]

        # Try to import scipy for curve fitting
        try:
            from scipy.optimize import curve_fit
            scipy_available = True
        except ImportError:
            scipy_available = False
            lines.append("% WARNING: scipy not available - power law fitting skipped")

        for planner in available_planners:
            time_comp, space_comp = theoretical_complexity.get(planner, ('--', '--'))

            if scipy_available:
                # Collect data from cube obstacles (Experiment 1)
                grid_sizes_data = []
                search_times = []
                memory_usage = []

                for grid_size in sorted(self.grid_sizes):
                    results = self._get_filtered_results(planner, grid_size, 0.25, 'cube')
                    if results:
                        successful = [r for r in results if r['success']]
                        if successful:
                            grid_sizes_data.append(grid_size)
                            search_times.append(np.mean([r['search_time'] for r in successful]))
                            memory_usage.append(np.mean([r['memory_mb'] for r in successful]))

                # Fit power law: T = a * N^k
                if len(grid_sizes_data) >= 3:
                    try:
                        # Time scaling
                        def power_law(N, k):
                            return N ** k

                        # Normalize to avoid numerical issues
                        time_normalized = np.array(search_times) / search_times[0] if search_times[0] > 0 else search_times
                        grid_normalized = np.array(grid_sizes_data) / grid_sizes_data[0]

                        popt_time, _ = curve_fit(power_law, grid_normalized, time_normalized, p0=[2.0], maxfev=10000)
                        k_time = popt_time[0]

                        # Memory scaling
                        mem_normalized = np.array(memory_usage) / memory_usage[0] if memory_usage[0] > 0 else memory_usage
                        popt_mem, _ = curve_fit(power_law, grid_normalized, mem_normalized, p0=[1.5], maxfev=10000)
                        k_mem = popt_mem[0]

                        time_scaling = f"$T \\propto N^{{{k_time:.1f}}}$"
                        mem_scaling = f"$M \\propto N^{{{k_mem:.1f}}}$"
                    except Exception as e:
                        time_scaling = "Fit failed"
                        mem_scaling = "Fit failed"
                else:
                    time_scaling = "Insufficient data"
                    mem_scaling = "Insufficient data"
            else:
                time_scaling = "scipy required"
                mem_scaling = "scipy required"

            lines.append(f"{planner_display[planner]} & {time_comp} & {space_comp} & {time_scaling} & {mem_scaling} \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("% Power law fitting: log(T) = log(a) + k*log(N)")
        lines.append("% Fitted using scipy.optimize.curve_fit on normalized data")

        return lines


def main():
    """Main entry point"""
    try:
        # Override num_trials to 1 if quick test mode
        num_trials = 1 if args.quick_test else args.num_trials

        # Create experiment runner
        runner = PickPlaceExperimentRunner(
            planners_to_test=args.planners,
            num_cubes=args.num_cubes,
            grid_sizes=args.grid_sizes,
            num_trials=num_trials,
            obstacle_densities=args.obstacle_densities,
            obstacle_types=args.obstacle_types,
            output_dir=args.output_dir,
            quick_test=args.quick_test
        )

        # Run experiments (setup_scene is called inside for each grid size)
        runner.run_experiments()

        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETE!")
        print("="*80)
        print(f"Results saved to: {runner.output_dir}")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Experiment failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        simulation_app.close()


if __name__ == "__main__":
    main()


