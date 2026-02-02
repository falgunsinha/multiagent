"""
Franka RRT Pick and Place - STANDALONE VERSION with RL Object Selection
RRT path planning with obstacle avoidance, conservative collision spheres,
dynamic grid configuration, pick retry logic, return to home after each cube.
Uses PhysX Lidar - Rotating and Intel RealSense D455 depth sensor for obstacle detection.

"""

import argparse
import sys
from pathlib import Path

# Parse command-line arguments BEFORE importing Isaac Sim
parser = argparse.ArgumentParser(description="Franka RRT Pick-and-Place with RL Object Selection (Standalone)")
parser.add_argument("--rl_model", type=str, default=None,
                   help="Path to trained RL model (PPO: .zip, DDQN: .pt)")
parser.add_argument("--num_cubes", type=int, default=None,
                   help="Number of cubes to spawn (default: auto-detect from model metadata)")
parser.add_argument("--training_grid_size", type=int, default=None,
                   help="Fixed training grid size (default: auto-detect from model metadata)")
args, unknown = parser.parse_known_args()

# Auto-detect parameters from model metadata if RL model is provided
if args.rl_model:
    import json
    import os

    # Determine model type and metadata path
    model_path = args.rl_model
    if 'ddqn' in model_path.lower():
        # DDQN model
        if not model_path.endswith('.pt') and not model_path.endswith('.zip'):
            model_path += '.pt'
        metadata_path = model_path.replace('.pt', '_metadata.json')
    else:
        # PPO model
        if not model_path.endswith('.zip') and not model_path.endswith('.pt'):
            model_path += '.zip'
        metadata_path = model_path.replace('_final.zip', '_metadata.json')
        if not os.path.exists(metadata_path):
            metadata_path = model_path.replace('.zip', '_metadata.json')

    # Load metadata if exists
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

                # Auto-detect training_grid_size
                if args.training_grid_size is None:
                    args.training_grid_size = metadata.get("training_grid_size", 6)
                    print(f"[AUTO-DETECT] training_grid_size = {args.training_grid_size} (from metadata)")

                # Auto-detect num_cubes (use max from training if available)
                if args.num_cubes is None:
                    # Try to get from metadata, otherwise use grid_size - 1
                    args.num_cubes = metadata.get("num_cubes", args.training_grid_size * args.training_grid_size - 1)
                    print(f"[AUTO-DETECT] num_cubes = {args.num_cubes} (from metadata)")
        except Exception as e:
            print(f"[WARNING] Could not load metadata from {metadata_path}: {e}")

# Set defaults if still None
if args.training_grid_size is None:
    args.training_grid_size = 6
if args.num_cubes is None:
    args.num_cubes = 4

# Create SimulationApp BEFORE importing any Isaac Sim modules
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# NOW import Isaac Sim modules (after SimulationApp is created)
import asyncio
import time
import numpy as np
import os
import omni.ui as ui
from omni.kit.async_engine import run_coroutine
import omni.timeline
import omni.usd
import omni.usd.editor

# Isaac Sim imports
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid, FixedCuboid
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import PathPlannerVisualizer, ArticulationTrajectory
from isaacsim.robot_motion.motion_generation.lula import RRT
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from isaacsim.robot_motion.motion_generation.lula.trajectory_generator import LulaCSpaceTrajectoryGenerator
from pxr import UsdPhysics, PhysxSchema, Gf, UsdGeom, Sdf
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
import omni.isaac.core.utils.prims as prim_utils
import carb

# PhysX Lidar imports
from isaacsim.sensors.physx import RotatingLidarPhysX

# Depth Camera imports
from isaacsim.sensors.camera import SingleViewDepthSensor

# Add project root to path for local imports
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.manipulators import SingleManipulator
from src.grippers import ParallelGripper
from src.rl.visual_grid import create_visual_grid

# Import RL libraries only if model path is provided
RL_AVAILABLE = False
if args.rl_model:
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from src.rl.doubleDQN.double_dqn_agent import DoubleDQNAgent  # For DDQN models
        from src.rl.object_selection_env import ObjectSelectionEnv
        import torch
        RL_AVAILABLE = True
        print(f"[RL] Model path provided: {args.rl_model}")
        print("[RL] RL libraries loaded successfully (PPO and DDQN support)")
    except ImportError as e:
        print(f"[RL] Warning: Could not import RL libraries: {e}")
        print("[RL] Falling back to greedy baseline")
        RL_AVAILABLE = False


class PerformanceTracker:
    """Track and compare performance metrics for different RL methods"""

    def __init__(self):
        self.start_time = None
        self.pick_times = []
        self.pick_order = []
        self.successes = []
        self.failures = []
        self.total_distance = 0.0
        self.method_name = "Unknown"

    def start(self, method_name="Greedy"):
        """Start tracking a new run"""
        self.start_time = time.time()
        self.pick_times = []
        self.pick_order = []
        self.successes = []
        self.failures = []
        self.total_distance = 0.0
        self.method_name = method_name

    def record_pick(self, cube_index, success, pick_time, distance):
        """Record a pick attempt"""
        self.pick_order.append(cube_index)
        self.pick_times.append(pick_time)
        if success:
            self.successes.append(cube_index)
        else:
            self.failures.append(cube_index)
        self.total_distance += distance

    def get_summary(self):
        """Get performance summary"""
        if self.start_time is None:
            return {}

        total_time = time.time() - self.start_time
        avg_pick_time = np.mean(self.pick_times) if self.pick_times else 0.0
        success_rate = len(self.successes) / (len(self.successes) + len(self.failures)) if (len(self.successes) + len(self.failures)) > 0 else 0.0

        return {
            "method": self.method_name,
            "total_time": total_time,
            "avg_pick_time": avg_pick_time,
            "pick_order": self.pick_order,
            "successes": len(self.successes),
            "failures": len(self.failures),
            "success_rate": success_rate * 100,
            "total_distance": self.total_distance,
            "avg_distance": self.total_distance / len(self.pick_times) if self.pick_times else 0.0
        }

    def print_summary(self):
        """Print performance summary"""
        summary = self.get_summary()
        if not summary:
            return

        print("\n" + "=" * 60)
        print(f"PERFORMANCE SUMMARY - {summary['method']}")
        print("=" * 60)
        print(f"Total time: {summary['total_time']:.2f} seconds")
        print(f"Average pick time: {summary['avg_pick_time']:.2f} seconds")
        print(f"Pick order: {summary['pick_order']}")
        print(f"Successes: {summary['successes']}/{summary['successes'] + summary['failures']} ({summary['success_rate']:.1f}%)")
        print(f"Total distance traveled: {summary['total_distance']:.2f} meters")
        print(f"Average distance per pick: {summary['avg_distance']:.2f} meters")
        print("=" * 60 + "\n")


class FrankaRRTDynamicGrid:
    """PyGame-style pick and place with RRT + RL object selection (Standalone)"""

    def __init__(self, num_cubes=4, training_grid_size=6):
        self.window = None
        self.world = None
        self.franka = None
        self.gripper = None
        self.container = None
        self.rrt = None
        self.path_planner_visualizer = None
        self.cspace_trajectory_generator = None

        # Kinematics solvers
        self.kinematics_solver = None
        self.articulation_kinematics_solver = None

        # Dynamic cube list
        self.cubes = []  # Will store (cube_object, cube_name) tuples

        # PyGame-style grid parameters
        self.num_cubes = num_cubes  # Actual number of cubes to spawn (e.g., 4, 6, 9, 16)
        self.training_grid_size = training_grid_size  # Configurable training grid (e.g., 4x4, 6x6)

        # Validate: num_cubes cannot exceed grid capacity
        max_capacity = self.training_grid_size * self.training_grid_size
        if self.num_cubes > max_capacity:
            print(f"[WARNING] num_cubes ({self.num_cubes}) exceeds grid capacity ({max_capacity})")
            print(f"[WARNING] Clamping to max capacity: {max_capacity}")
            self.num_cubes = max_capacity

        # Container dimensions (will be calculated after loading)
        self.container_dimensions = None  # [length, width, height] in meters

        # Obstacle management
        self.obstacles = {}  # Dictionary to store obstacles {name: obstacle_object}
        self.obstacle_counter = 0  # Counter for unique obstacle names

        # Cube obstacle tracking (for dynamic obstacle avoidance)
        self.cube_obstacles_enabled = False  # Track if cubes are currently added as obstacles

        # Obstacle_1 automatic movement with PhysX Force API (acceleration mode)
        self.obstacle_1_moving = False
        self.obstacle_1_acceleration = 6.0  # Acceleration magnitude in m/s²
        self.obstacle_1_min_x = 0.2  # Left boundary (0.2m)
        self.obstacle_1_max_x = 0.63  # Right boundary (0.63m) - 0.43m total travel
        self.obstacle_1_force_api_applied = False  # Track if Force API has been applied

        # PhysX Lidar sensor
        self.lidar = None
        self.lidar_detected_obstacles = {}  # Dictionary to store dynamically detected obstacles

        # Depth Camera sensor (SingleViewDepthSensor)
        self.depth_camera = None  # Depth camera sensor

        # Timeline
        self.timeline = omni.timeline.get_timeline_interface()

        # Task state
        self.is_picking = False
        self.placed_count = 0
        self.current_cube_index = 0  # Track which cube we're currently working on

        # RL model (optional)
        self.rl_model = None
        self.rl_model_path = args.rl_model
        self.rl_model_type = None  # Will be 'ppo' or 'ddqn'

        # Detect model type and add appropriate extension if missing
        if self.rl_model_path:
            # Check if it's a DDQN model (contains 'ddqn' in filename)
            if 'ddqn' in self.rl_model_path.lower():
                self.rl_model_type = 'ddqn'
                # Auto-add .pt extension if missing
                if not self.rl_model_path.endswith('.pt') and not self.rl_model_path.endswith('.zip'):
                    self.rl_model_path += '.pt'
            else:
                # Default to PPO
                self.rl_model_type = 'ppo'
                # Auto-add .zip extension if missing
                if not self.rl_model_path.endswith('.zip') and not self.rl_model_path.endswith('.pt'):
                    self.rl_model_path += '.zip'

        self.use_rl = RL_AVAILABLE and args.rl_model is not None

        # Performance tracking
        self.performance_tracker = PerformanceTracker()

        # UI elements
        self.load_btn = None
        self.pick_btn = None
        self.reset_btn = None
        self.add_obstacle_btn = None
        self.remove_obstacle_btn = None
        self.status_label = None
        self.num_cubes_field = None
        self.training_grid_field = None

        self.build_ui()
    
    def build_ui(self):
        """Build UI"""
        self.window = ui.Window("Cobot - Grasping", width=450, height=500)

        with self.window.frame:
            with ui.VStack(spacing=10):
                ui.Label("Cobot - Pick and Place",
                        alignment=ui.Alignment.CENTER,
                        style={"font_size": 18})

                ui.Spacer(height=10)

                # Grid Configuration Section
                with ui.CollapsableFrame("Grid Configuration", height=0):
                    with ui.VStack(spacing=5):
                        # Training grid size
                        with ui.HStack(spacing=10):
                            ui.Label("Training Grid Size:", width=150)
                            training_grid_model = ui.SimpleIntModel(self.training_grid_size)
                            self.training_grid_field = ui.IntField(height=25, model=training_grid_model)

                        # Number of cubes
                        with ui.HStack(spacing=10):
                            ui.Label("Number of Cubes:", width=150)
                            num_cubes_model = ui.SimpleIntModel(self.num_cubes)
                            self.num_cubes_field = ui.IntField(height=25, model=num_cubes_model)

                        # Info label
                        ui.Label("Max cubes = Grid Size × Grid Size",
                                alignment=ui.Alignment.CENTER,
                                style={"color": 0xFF888888, "font_size": 12})

                ui.Spacer(height=10)

                # Main Buttons
                self.load_btn = ui.Button("Load Scene", height=40, clicked_fn=self._on_load)
                self.pick_btn = ui.Button("Start Pick and Place", height=40, clicked_fn=self._on_pick, enabled=False)
                self.reset_btn = ui.Button("Reset Scene", height=40, clicked_fn=self._on_reset, enabled=False)

                ui.Spacer(height=10)

                # Obstacle Management Section
                with ui.HStack(spacing=10):
                    self.add_obstacle_btn = ui.Button("Add Obstacle", height=35, clicked_fn=self._on_add_obstacle, enabled=False)
                    self.remove_obstacle_btn = ui.Button("Remove Obstacle", height=35, clicked_fn=self._on_remove_obstacle, enabled=False)

                ui.Spacer(height=10)

                # Status
                self.status_label = ui.Label("Ready - Configure grid and click 'Load Scene'",
                                            alignment=ui.Alignment.CENTER)
    
    def _update_status(self, message):
        """Update status label"""
        if self.status_label:
            self.status_label.text = f"Status: {message}"
    
    def _on_load(self):
        """Load scene button callback"""
        self._update_status("Loading scene...")
        run_coroutine(self._load_scene())
    
    async def _load_scene(self):
        """Load the scene with Franka, PyGame-style cube placement, and container"""
        try:
            # Get parameters from UI
            self.training_grid_size = int(self.training_grid_field.model.get_value_as_int())
            self.num_cubes = int(self.num_cubes_field.model.get_value_as_int())

            # Validate training grid size
            if self.training_grid_size < 1:
                self._update_status("Error: Training grid size must be at least 1")
                return
            if self.training_grid_size > 10:
                self._update_status("Error: Training grid size too large (max 10)")
                return

            # Validate number of cubes
            if self.num_cubes < 1:
                self._update_status("Error: Number of cubes must be at least 1")
                return

            # Check against grid capacity
            max_capacity = self.training_grid_size * self.training_grid_size
            if self.num_cubes > max_capacity:
                self._update_status(f"Error: No. of cubes are more than grid size. Max {max_capacity} for {self.training_grid_size}x{self.training_grid_size} grid")
                return

            self.timeline.stop()
            World.clear_instance()

            # Single update after cleanup
            await omni.kit.app.get_app().next_update_async()

            self.world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
            self.world.scene.add_default_ground_plane()

            # Add camera by copying from source USD file (no wait needed)
            from pxr import Usd, UsdGeom, Sdf
            stage = omni.usd.get_context().get_stage()
            camera_usd_path = "C:/isaacsim/cobotproject/assets/Main_camera.usd"

            # Open the camera USD file
            camera_stage = Usd.Stage.Open(camera_usd_path)
            if camera_stage:
                # Get the default prim or find the camera prim
                default_prim = camera_stage.GetDefaultPrim()
                if default_prim and default_prim.IsValid():
                    # Copy the entire prim hierarchy to /World/Main_Camera
                    Sdf.CopySpec(camera_stage.GetRootLayer(), default_prim.GetPath(),
                                stage.GetRootLayer(), Sdf.Path("/World/Main_Camera"))
                else:
                    # If no default prim, find the first camera
                    for prim in camera_stage.Traverse():
                        if prim.IsA(UsdGeom.Camera):
                            Sdf.CopySpec(camera_stage.GetRootLayer(), prim.GetPath(),
                                        stage.GetRootLayer(), Sdf.Path("/World/Main_Camera"))
                            break

            # Single update after world setup
            await omni.kit.app.get_app().next_update_async()

            franka_name = f"franka_{int(time.time() * 1000)}"
            franka_prim_path = f"/World/Franka_{int(time.time() * 1000)}"

            franka_usd_path = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
            robot_prim = add_reference_to_stage(usd_path=franka_usd_path, prim_path=franka_prim_path)
            robot_prim.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
            robot_prim.GetVariantSet("Mesh").SetVariantSelection("Quality")

            # Create gripper (wider opening to avoid pushing cubes)
            # Cube is 5.15cm wide, so open to 8cm (4cm per finger) for clearance
            self.gripper = ParallelGripper(
                end_effector_prim_path=f"{franka_prim_path}/panda_rightfinger",
                joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
                joint_opened_positions=np.array([0.04, 0.04]),  # 8cm total opening (4cm per finger)
                joint_closed_positions=np.array([0.0, 0.0]),  # Fully closed for better grip
                action_deltas=np.array([0.01, 0.01])
            )

            # Add manipulator
            self.franka = self.world.scene.add(
                SingleManipulator(
                    prim_path=franka_prim_path,
                    name=franka_name,
                    end_effector_prim_path=f"{franka_prim_path}/panda_rightfinger",
                    gripper=self.gripper,
                    position=np.array([0.0, 0.0, 0.0]),
                    orientation=np.array([1.0, 0.0, 0.0, 0.0])
                )
            )

            # Add PhysX Lidar - Rotating sensor attached to Franka
            # Position it to detect obstacles (obstacles are centered at ~10cm height, 30cm tall)
            # Obstacle range: -5cm (bottom) to 25cm (top)
            # Place Lidar at 15cm to be in middle of obstacle height range
            # Attach to robot base for stable scanning
            lidar_prim_path = f"{franka_prim_path}/lidar_sensor"

            # Create PhysX Rotating Lidar
            # Position relative to robot base: at 15cm height to detect obstacles at 10cm center height
            lidar_translation = np.array([0.0, 0.0, 0.15])  # 15cm above robot base

            self.lidar = self.world.scene.add(
                RotatingLidarPhysX(
                    prim_path=lidar_prim_path,
                    name="franka_lidar",
                    translation=lidar_translation,
                    rotation_frequency=20.0,  # 20 Hz rotation
                    fov=(360.0, 30.0),  # 360 degrees horizontal, 30 degrees vertical
                    resolution=(1.0, 1.0),  # 1 degree resolution
                    valid_range=(0.4, 100.0)  # 0.4m to 100m range
                )
            )

            # Initialize Depth Camera attached to panda hand
            depth_camera_prim_path = f"{franka_prim_path}/panda_hand/depth_camera"

            # Position relative to panda hand
            # Translation: 5cm forward along hand's local axis
            # Orientation: looking forward/down from the hand
            position = np.array([0.0, 0.0, 0.05])  # 5cm above hand center
            orientation = euler_angles_to_quats(np.array([90.0, 0.0, 0.0]))  # Pitch down 90 degrees

            # Create SingleViewDepthSensor with minimal parameters
            # Using 512x512 for square aspect ratio to avoid aperture warnings
            self.depth_camera = SingleViewDepthSensor(
                prim_path=depth_camera_prim_path,
                name="depth_camera",
                translation=position,
                orientation=orientation,
                resolution=(512, 512),  # Square resolution to match aperture aspect ratio
                frequency=10  # 10 Hz update rate
            )

            # Add to world scene
            self.world.scene.add(self.depth_camera)

            # Set camera properties via USD API to avoid aperture warnings
            stage = omni.usd.get_context().get_stage()
            camera_prim = stage.GetPrimAtPath(depth_camera_prim_path)
            if camera_prim:
                # Set focal length and aperture for square pixels
                camera_prim.GetAttribute("focalLength").Set(24.0)
                camera_prim.GetAttribute("horizontalAperture").Set(20.955)
                camera_prim.GetAttribute("verticalAperture").Set(20.955)  # Same as horizontal for square
                camera_prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.01, 10000.0))

            # Single update after robot, Lidar, and Depth Camera setup
            await omni.kit.app.get_app().next_update_async()

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

            # Add physics to container (no wait needed)
            from omni.isaac.core.utils.stage import get_current_stage
            stage = get_current_stage()
            container_prim = stage.GetPrimAtPath(container_prim_path)
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(container_prim)
            rigid_body_api.CreateKinematicEnabledAttr(True)
            UsdPhysics.CollisionAPI.Apply(container_prim)

            cube_size = 0.0515
            # Reduced grid size to avoid Obstacle_1 collisions
            # 45cm x 45cm grid = better clearance from Obstacle_1 + all cubes reachable
            if self.num_cubes <= 4:
                cube_spacing = 0.16  # Slightly reduced
            elif self.num_cubes <= 9:
                cube_spacing = 0.1125  # 4x4 grid = 45cm (11.25cm spacing between cubes)
            else:
                cube_spacing = 0.10  # Reduced from 0.11

            self.cubes = []
            colors = [
                (np.array([0, 0, 1]), "Blue"),
                (np.array([1, 0, 0]), "Red"),
                (np.array([0, 1, 0]), "Green"),
                (np.array([1, 1, 0]), "Yellow"),
                (np.array([1, 0, 1]), "Magenta"),
                (np.array([0, 1, 1]), "Cyan"),
                (np.array([1, 0.5, 0]), "Orange"),
                (np.array([0.5, 0, 1]), "Purple"),
                (np.array([0.5, 0.5, 0.5]), "Gray"),
                (np.array([1, 0.75, 0.8]), "Pink"),
            ]

            # Reduced 45cm x 45cm grid - moved 7cm further from robot base AND 1cm away from container
            # Container is at [0.30, 0.50], Obstacle_1 is at [0.27, 0.17]
            # 4x4 grid with 11.25cm spacing spans 45cm (Obstacle_1 will be closer to test detection)
            # Grid center: [0.52, -0.11] - moved 7cm away from robot base (X: 0.45 -> 0.52) and 1cm away from container (Y: -0.10 -> -0.11)
            # Grid bounds with 11.25cm spacing:
            #   X: [0.52 - 0.225, 0.52 + 0.225] = [0.295, 0.745] (within Franka reach ~0.80m)
            #   Y: [-0.11 - 0.225, -0.11 + 0.225] = [-0.335, 0.115] (within Franka reach)
            # Far corner distance: sqrt(0.745² + 0.335²) = 0.817m (at reach limit)
            grid_center_x = 0.52
            grid_center_y = -0.11
            # Use FIXED training grid size for grid extent
            # Grid extent = num_cells * cell_size (NOT (num_cells - 1) * cell_size)
            grid_extent_x = self.training_grid_size * cube_spacing
            grid_extent_y = self.training_grid_size * cube_spacing
            start_x = grid_center_x - (grid_extent_x / 2.0)
            start_y = grid_center_y - (grid_extent_y / 2.0)

            # Create visual grid on ground plane (shows FULL training grid)
            create_visual_grid(start_x, start_y, grid_extent_x, grid_extent_y, cube_spacing, self.training_grid_size, self.training_grid_size)

            # PyGame-style random placement in FIXED training grid
            random_offset_range = 0.0  # NO random offset - place exactly at cell center
            total_cells = self.training_grid_size * self.training_grid_size

            # Randomly select which grid cells to fill (no duplicates)
            selected_indices = np.random.choice(total_cells, size=self.num_cubes, replace=False)
            selected_cells = set(selected_indices)

            cube_index = 0
            for row in range(self.training_grid_size):
                for col in range(self.training_grid_size):
                    # Calculate cell index
                    cell_index = row * self.training_grid_size + col

                    # Skip this cell if not selected
                    if cell_index not in selected_cells:
                        continue

                    # CELL-CENTERED placement (matching PyGame convention)
                    # PyGame convention (from pyGame_Grid.py line 234-235):
                    #   cx = c * CELL + CELL // 2  → col controls X
                    #   cy = r * CELL + CELL // 2  → row controls Y
                    # Therefore:
                    #   - row → Y-axis
                    #   - col → X-axis
                    # Cell boundaries:
                    #   X: [start_x + col*spacing, start_x + (col+1)*spacing]
                    #   Y: [start_y + row*spacing, start_y + (row+1)*spacing]
                    # Cell center:
                    #   X: start_x + col*spacing + spacing/2
                    #   Y: start_y + row*spacing + spacing/2
                    cell_center_x = start_x + (col * cube_spacing) + (cube_spacing / 2.0)
                    cell_center_y = start_y + (row * cube_spacing) + (cube_spacing / 2.0)

                    # Add random offset within cell (±3cm from cell center)
                    random_offset_x = np.random.uniform(-random_offset_range, random_offset_range)
                    random_offset_y = np.random.uniform(-random_offset_range, random_offset_range)

                    # Final cube position: cell center + random offset
                    cube_x = cell_center_x + random_offset_x
                    cube_y = cell_center_y + random_offset_y
                    cube_z = cube_size/2.0 + 0.01
                    cube_position = np.array([cube_x, cube_y, cube_z])

                    color, color_name = colors[cube_index % len(colors)]
                    timestamp = int(time.time() * 1000) + cube_index
                    cube_number = cube_index + 1
                    cube_name = f"Cube_{cube_number}"
                    prim_path = f"/World/Cube_{cube_number}"

                    cube = self.world.scene.add(
                        DynamicCuboid(
                            name=f"cube_{timestamp}",
                            position=cube_position,
                            prim_path=prim_path,
                            scale=np.array([cube_size, cube_size, cube_size]),
                            size=1.0,
                            color=color
                        )
                    )

                    stage = omni.usd.get_context().get_stage()
                    prim = stage.GetPrimAtPath(prim_path)
                    if prim:
                        display_name = f"Cube {cube_number} (t{timestamp})"
                        omni.usd.editor.set_display_name(prim, display_name)

                    self.cubes.append((cube, f"{cube_name} ({color_name})"))
                    cube_index += 1

            # Single update after all cubes created
            await omni.kit.app.get_app().next_update_async()

            # Initialize physics and reset (batch updates)
            self.world.initialize_physics()
            self.world.reset()

            # Initialize PhysX Lidar after world reset
            self.lidar.add_depth_data_to_frame()
            self.lidar.add_point_cloud_data_to_frame()
            self.lidar.enable_visualization()

            # Configure rendering settings for depth sensor (disable DLSS/DLAA to avoid warnings)
            settings = carb.settings.get_settings()

            # Disable DLSS/DLAA for depth sensor compatibility
            settings.set("/rtx/post/dlss/execMode", 0)  # 0 = Off
            settings.set("/rtx/post/aa/op", 0)  # 0 = Disabled

            # Configure depth sensor schema settings
            settings.set("/exts/omni.usd.schema.render_settings/rtx/renderSettings/apiSchemas/autoApply", None)
            settings.set("/exts/omni.usd.schema.render_settings/rtx/camera/apiSchemas/autoApply", None)
            settings.set("/exts/omni.usd.schema.render_settings/rtx/renderProduct/apiSchemas/autoApply", None)



            # Initialize Depth Camera after world reset
            # NOTE: Initialize with attach_rgb_annotator=False for better performance
            self.depth_camera.initialize(attach_rgb_annotator=False)

            # Configure depth sensor parameters AFTER initialize() but BEFORE attach_annotator()
            # Following the official Isaac Sim example pattern from camera_stereoscopic_depth.py

            # Camera lens parameters
            self.depth_camera.set_focal_length(24.0)  # 24mm focal length
            self.depth_camera.set_focus_distance(400.0)  # Focus distance

            # Depth sensor parameters (following official example values)
            self.depth_camera.set_baseline_mm(55.0)  # 55mm baseline (standard stereo)
            self.depth_camera.set_focal_length_pixel(256.0)  # Focal length in pixels (512/2)
            self.depth_camera.set_sensor_size_pixel(1280.0)  # Standard depth sensor size (NOT resolution!)
            self.depth_camera.set_max_disparity_pixel(110.0)  # Max disparity
            self.depth_camera.set_confidence_threshold(0.99)  # High confidence
            self.depth_camera.set_noise_mean(0.1)  # Low noise mean
            self.depth_camera.set_noise_sigma(0.5)  # Low noise sigma
            self.depth_camera.set_noise_downscale_factor_pixel(1.0)  # No downscale
            self.depth_camera.set_min_distance(0.1)  # Minimum distance: 10cm
            self.depth_camera.set_max_distance(2.0)  # Maximum distance: 2m

            print(f"[DEPTH CAMERA] Depth sensor parameters configured")

            # Attach multiple annotators for comprehensive depth data
            # 1. DepthSensorDistance - provides distance measurements
            self.depth_camera.attach_annotator("DepthSensorDistance")

            # 2. DepthSensorPointCloudPosition - provides 3D point cloud positions
            self.depth_camera.attach_annotator("DepthSensorPointCloudPosition")

            # 3. DepthSensorPointCloudColor - provides color for point cloud
            self.depth_camera.attach_annotator("DepthSensorPointCloudColor")

            print("[DEPTH CAMERA] Depth camera initialized on panda hand")
            print("[DEPTH CAMERA] Attached annotators:")
            print("  - DepthSensorDistance (distance measurements)")
            print("  - DepthSensorPointCloudPosition (3D point cloud)")
            print("  - DepthSensorPointCloudColor (point cloud colors)")
            print("[DEPTH CAMERA] Position: 5cm above panda hand, looking forward/down")
            print("[DEPTH CAMERA] Resolution: 512x512 (square), Frequency: 10 Hz")
            print("[DEPTH CAMERA] Depth range: 0.1m - 2.0m, Baseline: 55mm")
            print("[DEPTH CAMERA] Sensor size: 1280 pixels (standard depth sensor parameter)")



            # Configure robot (no waits needed for parameter setting)
            self.franka.disable_gravity()
            articulation_controller = self.franka.get_articulation_controller()
            kp_gains = 1e15 * np.ones(9)
            kd_gains = 1e13 * np.ones(9)
            articulation_controller.set_gains(kp_gains, kd_gains)

            default_joint_positions = np.array([0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75, 0.0, 0.0])
            self.franka.set_joints_default_state(positions=default_joint_positions)
            self.gripper.set_default_state(self.gripper.joint_closed_positions)

            articulation_controller = self.franka.get_articulation_controller()
            close_action = ArticulationAction(
                joint_positions=self.gripper.joint_closed_positions,
                joint_indices=np.array([7, 8])
            )
            articulation_controller.apply_action(close_action)

            # Wait for robot to settle
            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            self._setup_rrt()

            # Add container as a permanent obstacle for collision avoidance
            self._add_container_as_obstacle()

            print("Scene loaded successfully!")

            # Load RL model if provided
            if self.use_rl:
                self._load_rl_model()

            # Enable buttons
            self.pick_btn.enabled = True
            self.reset_btn.enabled = True
            self.add_obstacle_btn.enabled = True
            self.remove_obstacle_btn.enabled = True

            status_msg = "Scene loaded - Ready to pick and place"
            if self.use_rl and self.rl_model:
                status_msg += " (RL mode)"
            elif self.use_rl:
                status_msg += " (Greedy mode - RL failed)"
            else:
                status_msg += " (Greedy mode)"
            self._update_status(status_msg)

        except Exception as e:
            self._update_status(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def _setup_rrt(self):
        """Setup RRT motion planner and kinematics solvers"""
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")

        # Use local robot description file from assets folder (no extension dependency)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, "..", "..")
        robot_description_file = os.path.join(project_root, "assets", "franka_conservative_spheres_robot_description.yaml")
        robot_description_file = os.path.normpath(robot_description_file)

        urdf_path = os.path.join(mg_extension_path, "motion_policy_configs", "franka", "lula_franka_gen.urdf")
        rrt_config_file = os.path.join(mg_extension_path, "path_planner_configs", "franka", "rrt", "franka_planner_config.yaml")

        print(f"[RRT] Config files:")
        print(f"  Robot description: {robot_description_file}")
        print(f"  URDF: {urdf_path}")
        print(f"  RRT config: {rrt_config_file}")

        # Verify files exist
        if not os.path.exists(robot_description_file):
            raise FileNotFoundError(f"Robot description not found: {robot_description_file}")
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        if not os.path.exists(rrt_config_file):
            raise FileNotFoundError(f"RRT config not found: {rrt_config_file}")

        self.rrt = RRT(robot_description_path=robot_description_file, urdf_path=urdf_path,
                       rrt_config_path=rrt_config_file, end_effector_frame_name="right_gripper")
        self.rrt.set_max_iterations(10000)

        self.path_planner_visualizer = PathPlannerVisualizer(robot_articulation=self.franka, path_planner=self.rrt)
        self.kinematics_solver = LulaKinematicsSolver(robot_description_path=robot_description_file, urdf_path=urdf_path)
        self.articulation_kinematics_solver = ArticulationKinematicsSolver(
            self.franka, self.kinematics_solver, "right_gripper")

        self.cspace_trajectory_generator = LulaCSpaceTrajectoryGenerator(robot_description_file, urdf_path)

    def _add_container_as_obstacle(self):
        """Add container as obstacle for RRT collision avoidance"""
        try:
            if self.container is None:
                return

            # Simply add the container itself as an obstacle
            # No need for separate walls - the container already has collision geometry
            self.rrt.add_obstacle(self.container, static=True)

        except Exception as e:
            print(f"[ERROR] Failed to add container as obstacle: {e}")
            import traceback
            traceback.print_exc()

    def _add_other_cubes_as_obstacles(self, current_cube_index, target_position):
        """
        Add ONLY nearby cubes as temporary obstacles (within potential collision zone).
        This prevents the robot from colliding with other cubes during motion,
        while avoiding excessive obstacles that cause RRT failures.

        Args:
            current_cube_index: Index of the cube being picked (this cube will NOT be added as obstacle)
            target_position: Position of the target cube (to determine which cubes are nearby)

        Returns:
            int: Number of nearby cubes added as obstacles
        """
        if self.cube_obstacles_enabled:
            return 0

        try:
            added_count = 0
            collision_radius = 0.15  # 15cm radius (only adjacent cubes, excludes diagonals at 15.9cm)

            for i, (cube, cube_name) in enumerate(self.cubes):
                # Skip the current target cube
                if i == current_cube_index:
                    continue

                # Get cube position
                cube_pos, _ = cube.get_world_pose()

                # Calculate distance to target
                distance = np.linalg.norm(cube_pos[:2] - target_position[:2])  # XY distance only

                # Only add as obstacle if within collision radius
                if distance < collision_radius:
                    self.rrt.add_obstacle(cube, static=False)
                    added_count += 1

            if added_count > 0:
                print(f"  Added {added_count} nearby cubes as obstacles")
            self.cube_obstacles_enabled = True

            return added_count

        except Exception as e:
            print(f"[ERROR] Failed to add cubes as obstacles: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def _remove_all_cube_obstacles(self):
        """
        Remove all cubes from RRT obstacle list.
        Called after pick/place operations are complete.
        """
        if not self.cube_obstacles_enabled:
            return

        try:
            for cube, _ in self.cubes:
                try:
                    self.rrt.remove_obstacle(cube)
                except Exception:
                    # Cube might not be in obstacle list (e.g., it was the target cube)
                    pass

            self.cube_obstacles_enabled = False

        except Exception as e:
            print(f"[ERROR] Failed to remove cube obstacles: {e}")
            import traceback
            traceback.print_exc()

    def _process_lidar_data(self):
        """
        Process PhysX Lidar point cloud data to detect obstacles in real-time.
        Returns list of detected obstacle positions in world coordinates.
        """
        if self.lidar is None:
            print("[LIDAR] Lidar sensor not initialized")
            return []

        try:
            # Get current frame data from PhysX Lidar
            lidar_data = self.lidar.get_current_frame()

            if lidar_data is None or "point_cloud" not in lidar_data:
                return []

            point_cloud_data = lidar_data["point_cloud"]

            # PhysX Lidar returns point cloud as tensor/array
            # Convert to numpy array if needed
            if point_cloud_data is None:
                return []

            # Handle tensor data (convert to numpy)
            if hasattr(point_cloud_data, 'cpu'):
                points = point_cloud_data.cpu().numpy()
            elif hasattr(point_cloud_data, 'numpy'):
                points = point_cloud_data.numpy()
            elif isinstance(point_cloud_data, np.ndarray):
                points = point_cloud_data
            else:
                return []

            # Reshape if needed: (N, 1, 3) -> (N, 3)
            if len(points.shape) == 3 and points.shape[1] == 1:
                points = points.reshape(-1, 3)

            if len(points) == 0:
                return []

            # Validate shape
            if points.ndim != 2 or points.shape[1] != 3:
                return []

            # Transform points from sensor-local to world coordinates
            lidar_world_pos, lidar_world_rot = self.lidar.get_world_pose()
            from scipy.spatial.transform import Rotation as R
            rot_matrix = R.from_quat([lidar_world_rot[1], lidar_world_rot[2], lidar_world_rot[3], lidar_world_rot[0]]).as_matrix()

            # Transform all points to world coordinates
            points_world = (rot_matrix @ points.T).T + lidar_world_pos

            # Filter by height (world coordinates) - only 5cm to 40cm height
            # This automatically filters out cubes (at ~3.5cm height) while keeping obstacles (at ~11cm height)
            # Cubes: cube_size/2 + 0.01 = 2.575cm + 1cm = 3.575cm ✅ Filtered out (below 5cm)
            # Obstacles: center at 11cm ✅ Detected (within 5-40cm range)
            valid_points = points_world[(points_world[:, 2] > 0.05) & (points_world[:, 2] < 0.40)]

            # Filter by distance from robot base - STRICT workspace limits
            robot_pos, _ = self.franka.get_world_pose()
            distances_from_robot = np.linalg.norm(valid_points[:, :2] - robot_pos[:2], axis=1)
            valid_points = valid_points[(distances_from_robot > 0.30) & (distances_from_robot < 0.90)]  # 30cm-90cm only

            # Filter out cube pickup region (adjusted for 45cm x 45cm grid with 11.25cm spacing)
            # Grid center: [0.52, -0.11], Grid extent: 45cm (moved 7cm away from robot base and 1cm away from container)
            cube_grid_center = np.array([0.52, -0.11])
            cube_grid_margin = 0.28  # 22.5cm grid radius + 5.5cm buffer = 28cm margin
            cube_region_mask = ~((np.abs(valid_points[:, 0] - cube_grid_center[0]) < cube_grid_margin) &
                                 (np.abs(valid_points[:, 1] - cube_grid_center[1]) < cube_grid_margin))
            valid_points = valid_points[cube_region_mask]

            # Filter out container/placement region (tighter bounds to avoid blocking obstacles)
            if self.container_dimensions is not None:
                container_pos = np.array([0.30, 0.50, 0.0])
                container_margin = 0.08  # Reduced from 0.15 to 0.08 for tighter filtering
                container_half_dims = self.container_dimensions / 2.0
                container_region_mask = ~((np.abs(valid_points[:, 0] - container_pos[0]) < (container_half_dims[0] + container_margin)) &
                                          (np.abs(valid_points[:, 1] - container_pos[1]) < (container_half_dims[1] + container_margin)))
                valid_points = valid_points[container_region_mask]

            # Filter out robot base and arm region
            robot_base_pos = np.array([0.0, 0.0])
            robot_arm_radius = 0.55
            robot_region_mask = np.linalg.norm(valid_points[:, :2] - robot_base_pos, axis=1) > robot_arm_radius
            valid_points = valid_points[robot_region_mask]

            detected_obstacles = []

            if len(valid_points) > 10:
                # Grid-based clustering: 10cm grid cells
                grid_size = 0.1
                grid_points = np.round(valid_points / grid_size) * grid_size
                unique_cells, counts = np.unique(grid_points, axis=0, return_counts=True)
                obstacle_cells = unique_cells[counts > 5]
                detected_obstacles = obstacle_cells.tolist()

                # Merge nearby obstacles - use XY distance only (ignore Z for vertical stacking)
                # This prevents same obstacle being detected at multiple heights or positions
                if len(detected_obstacles) > 1:
                    merged_obstacles = []
                    used = set()
                    for i, obs1 in enumerate(detected_obstacles):
                        if i in used:
                            continue
                        cluster = [obs1]
                        for j, obs2 in enumerate(detected_obstacles[i+1:], start=i+1):
                            if j in used:
                                continue
                            # Only check XY distance (ignore Z)
                            dist_xy = np.linalg.norm(np.array(obs1[:2]) - np.array(obs2[:2]))
                            # Increased threshold to 25cm to merge detections from same obstacle
                            # (obstacles are 20cm wide, so detections can be up to 20cm apart)
                            if dist_xy < 0.25:  # Same obstacle if within 25cm in XY plane
                                cluster.append(obs2)
                                used.add(j)
                        # Use center position (average XY) and lowest Z
                        cluster_array = np.array(cluster)
                        merged_pos = [
                            np.mean(cluster_array[:, 0]),  # Average X (center)
                            np.mean(cluster_array[:, 1]),  # Average Y (center)
                            np.min(cluster_array[:, 2])    # Lowest Z (base)
                        ]
                        merged_obstacles.append(merged_pos)
                    detected_obstacles = merged_obstacles

                # Log detected obstacles with detailed information
                if len(detected_obstacles) > 0:
                    print(f"\n[LIDAR] PhysX Lidar - Rotating Detection Report:")
                    print(f"[LIDAR] Total point cloud points: {len(valid_points)}")
                    print(f"[LIDAR] Detected obstacles: {len(detected_obstacles)}")
                    print(f"[LIDAR] ----------------------------------------")

                    for i, obs_pos in enumerate(detected_obstacles):
                        # Get actual obstacle name and details from stage
                        obs_name = "Unknown"
                        obs_type = "Unknown"
                        obs_dimensions = "Unknown"
                        point_count = 0

                        from omni.isaac.core.utils.stage import get_current_stage
                        stage = get_current_stage()

                        # Check all prims under /World for obstacles
                        for prim in stage.Traverse():
                            prim_path = str(prim.GetPath())
                            if "/World/Obstacle_" in prim_path or "/World/LidarObstacle_" in prim_path:
                                # Get the XFormPrim to check position
                                try:
                                    from omni.isaac.core.prims import XFormPrim
                                    xform = XFormPrim(prim_path)
                                    prim_pos, _ = xform.get_world_pose()
                                    # Check XY distance only
                                    dist_xy = np.linalg.norm(np.array(obs_pos[:2]) - prim_pos[:2])
                                    if dist_xy < 0.20:  # Within 20cm in XY
                                        obs_name = prim_path.split('/')[-1]

                                        # Get obstacle type
                                        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                                            obs_type = "DynamicCuboid (Rigid Body)"
                                        else:
                                            obs_type = "FixedCuboid (Static)"

                                        # Get dimensions from scale
                                        if prim.GetAttribute("xformOp:scale"):
                                            scale = prim.GetAttribute("xformOp:scale").Get()
                                            obs_dimensions = f"{scale[0]:.2f}m x {scale[1]:.2f}m x {scale[2]:.2f}m"

                                        # Count points near this obstacle
                                        for pt in valid_points:
                                            pt_dist = np.linalg.norm(np.array(pt[:2]) - prim_pos[:2])
                                            if pt_dist < 0.20:
                                                point_count += 1

                                        break
                                except:
                                    pass

                        print(f"[LIDAR] Obstacle #{i+1}:")
                        print(f"[LIDAR]   Name: {obs_name}")
                        print(f"[LIDAR]   Type: {obs_type}")
                        print(f"[LIDAR]   Dimensions: {obs_dimensions}")
                        print(f"[LIDAR]   Position: ({obs_pos[0]:.3f}, {obs_pos[1]:.3f}, {obs_pos[2]:.3f})m")
                        print(f"[LIDAR]   Point cloud hits: {point_count} points")

                    print(f"[LIDAR] ----------------------------------------\n")

                # PhysX Lidar has built-in visualization - no custom debug draw needed

            return detected_obstacles

        except Exception as e:
            carb.log_warn(f"[LIDAR ERROR] Error processing Lidar data: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _process_depth_camera_data(self):
        """
        Process depth camera data to detect obstacles in real-time.
        Returns list of detected obstacle positions in world coordinates.
        """
        if self.depth_camera is None:
            return []

        try:
            # Get current frame data from depth camera
            depth_frame = self.depth_camera.get_current_frame()

            if depth_frame is None:
                print("[DEPTH CAMERA] No frame data available")
                return []

            # Get depth data from the frame (DepthSensorDistance annotator)
            depth_data = None
            if "data" in depth_frame:
                depth_data = depth_frame["data"]
                print(f"[DEPTH CAMERA] Distance data shape: {depth_data.shape}, dtype: {depth_data.dtype}")
            elif "distance" in depth_frame:
                depth_data = depth_frame["distance"]
                print(f"[DEPTH CAMERA] Distance data shape: {depth_data.shape}, dtype: {depth_data.dtype}")
            elif "depth" in depth_frame:
                depth_data = depth_frame["depth"]
                print(f"[DEPTH CAMERA] Depth data shape: {depth_data.shape}, dtype: {depth_data.dtype}")

            if depth_data is None or len(depth_data) == 0:
                return []

            # Print depth statistics
            valid_depths = depth_data[depth_data > 0]
            if len(valid_depths) > 0:
                pass  # Depth data available

            # Check for point cloud data
            if "point_cloud_position" in depth_frame:
                pc_data = depth_frame["point_cloud_position"]
                print(f"[DEPTH CAMERA] Point cloud position shape: {pc_data.shape}")

            if "point_cloud_color" in depth_frame:
                pc_color = depth_frame["point_cloud_color"]
                print(f"[DEPTH CAMERA] Point cloud color shape: {pc_color.shape}")

            # Convert depth image to point cloud in camera frame
            # Depth data is in meters, shape (height, width)
            height, width = depth_data.shape

            # Camera intrinsics (for 512x512 resolution)
            fx = fy = 256.0  # Focal length in pixels (512/2)
            cx, cy = width / 2.0, height / 2.0  # Principal point (center of image)

            # Sample points from depth image (every 20 pixels for performance)
            detected_obstacles = []
            point_count = 0

            for v in range(0, height, 20):  # Sample every 20 rows
                for u in range(0, width, 20):  # Sample every 20 columns
                    depth = depth_data[v, u]

                    # Filter by depth range (0.1m to 1.5m)
                    if depth < 0.1 or depth > 1.5:
                        continue

                    point_count += 1

                    # Convert pixel + depth to 3D point in camera frame
                    x_cam = (u - cx) * depth / fx
                    y_cam = (v - cy) * depth / fy
                    z_cam = depth

                    # Transform to world frame
                    camera_pos, camera_quat = self.depth_camera.get_world_pose()

                    # Point in camera frame (camera looks along +Z, +X is right, +Y is down)
                    # Rotate to match world frame using quaternion rotation
                    point_cam = np.array([x_cam, y_cam, z_cam])

                    # Quaternion rotation: v' = q * v * q^-1
                    # Using simplified rotation for performance
                    from isaacsim.core.utils.rotations import quat_to_rot_matrix
                    rot_matrix = quat_to_rot_matrix(camera_quat)
                    point_world = rot_matrix @ point_cam + camera_pos

                    # Filter by height (only obstacles at reasonable height)
                    if point_world[2] < 0.05 or point_world[2] > 0.5:
                        continue

                    detected_obstacles.append(point_world.tolist())

            # Cluster nearby points (merge detections within 10cm)
            if len(detected_obstacles) > 0:
                merged_obstacles = []
                used = set()

                for i, obs1 in enumerate(detected_obstacles):
                    if i in used:
                        continue
                    used.add(i)
                    cluster = [obs1]
                    for j, obs2 in enumerate(detected_obstacles[i+1:], start=i+1):
                        if j in used:
                            continue
                        dist = np.linalg.norm(np.array(obs1) - np.array(obs2))
                        if dist < 0.10:  # 10cm clustering
                            cluster.append(obs2)
                            used.add(j)

                    # Use cluster center
                    cluster_array = np.array(cluster)
                    merged_pos = np.mean(cluster_array, axis=0).tolist()
                    merged_obstacles.append(merged_pos)

                detected_obstacles = merged_obstacles

            # Log detected obstacles
            if len(detected_obstacles) > 0:
                print(f"\n[DEPTH CAMERA] Intel RealSense D455 Detection Report:")
                print(f"[DEPTH CAMERA] Total depth points sampled: {point_count}")
                print(f"[DEPTH CAMERA] Detected obstacles: {len(detected_obstacles)}")
                print(f"[DEPTH CAMERA] ----------------------------------------")

                pass  # Obstacles detected

            return detected_obstacles

        except Exception as e:
            carb.log_warn(f"[DEPTH CAMERA ERROR] Error processing depth camera data: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _clear_lidar_obstacles(self):
        """
        Temporarily clear all Lidar-detected obstacles from RRT.
        Used during retreat planning to avoid detecting target cube as obstacle.
        """
        if self.rrt is None:
            return

        try:
            for obs_name, obs_obj in list(self.lidar_detected_obstacles.items()):
                try:
                    self.rrt.remove_obstacle(obs_obj)
                except Exception:
                    pass
            # Don't clear the dictionary - obstacles will be re-added on next update
        except Exception as e:
            print(f"[ERROR] Failed to clear Lidar obstacles: {e}")

    def _update_dynamic_obstacles(self):
        """
        Update RRT planner with dynamically detected obstacles from Lidar and Depth Camera.
        Also ensures RRT knows current Obstacle_1 position.

        OPTIMIZED: Instead of deleting and recreating obstacles, we:
        1. Reuse existing obstacle prims by moving them
        2. Only create new prims if we need more
        3. Only delete prims if we have too many
        This avoids constant stage updates and maintains 60 FPS
        """
        if self.lidar is None or self.rrt is None:
            return

        try:
            # CRITICAL: Update RRT world to get current Obstacle_1 position
            # Obstacle_1 is moved by physics callback, so we just need to update RRT's knowledge
            if self.obstacle_1_moving and "obstacle_1" in self.obstacles:
                # Get current Obstacle_1 position for debugging
                obs_pos, _ = self.obstacles["obstacle_1"].get_world_pose()
                # Update RRT's internal representation with current obstacle positions
                self.rrt.update_world()

            # Get detected obstacles from Lidar
            detected_positions = self._process_lidar_data()

            # Get detected obstacles from Depth Camera (if available)
            # NOTE: Disabled for performance - enable when needed for obstacle detection
            # if self.depth_camera is not None:
            #     depth_obstacles = self._process_depth_camera_data()
            #     if depth_obstacles is not None and len(depth_obstacles) > 0:
            #         # Merge depth camera obstacles with Lidar obstacles
            #         detected_positions.extend(depth_obstacles)

            # Limit to 10 obstacles for performance
            detected_positions = detected_positions[:10]

            num_detected = len(detected_positions)
            num_current = len(self.lidar_detected_obstacles)

            # Case 1: Update existing obstacles by moving them (NO deletion/creation)
            existing_obstacles = list(self.lidar_detected_obstacles.items())
            for i in range(min(num_detected, num_current)):
                obs_name, obs_obj = existing_obstacles[i]
                new_pos = np.array(detected_positions[i])

                # Check if position changed (5cm precision)
                current_pos, _ = obs_obj.get_world_pose()
                if np.linalg.norm(new_pos - current_pos) > 0.05:
                    # Move obstacle to new position (fast, no stage update)
                    obs_obj.set_world_pose(position=new_pos)
                    # Update RRT's internal representation
                    self.rrt.update_world()

            # Case 2: Need more obstacles - create new ones (INVISIBLE for performance)
            if num_detected > num_current:
                for i in range(num_current, num_detected):
                    obs_name = f"lidar_obstacle_{i}"
                    obs_prim_path = f"/World/LidarObstacle_{i}"

                    obstacle = self.world.scene.add(
                        FixedCuboid(
                            name=obs_name,
                            prim_path=obs_prim_path,
                            position=np.array(detected_positions[i]),
                            size=1.0,
                            scale=np.array([0.15, 0.15, 0.15]),
                            color=np.array([1.0, 0.0, 0.0]),
                            visible=False  # INVISIBLE - only for collision detection
                        )
                    )
                    # Ensure visibility is off at USD level
                    obstacle.set_visibility(False)
                    self.rrt.add_obstacle(obstacle, static=False)
                    self.lidar_detected_obstacles[obs_name] = obstacle

            # Case 3: Have too many obstacles - remove extras
            elif num_detected < num_current:
                for i in range(num_detected, num_current):
                    obs_name = f"lidar_obstacle_{i}"
                    if obs_name in self.lidar_detected_obstacles:
                        obs_obj = self.lidar_detected_obstacles[obs_name]
                        try:
                            self.rrt.remove_obstacle(obs_obj)
                            self.world.scene.remove_object(obs_name)
                        except:
                            pass
                        del self.lidar_detected_obstacles[obs_name]

        except Exception as e:
            carb.log_warn(f"[RRT ERROR] Error updating dynamic obstacles: {e}")
            import traceback
            traceback.print_exc()

    def _physics_step_callback(self, step_size):
        """
        Physics step callback for continuous sensor updates and obstacle movement.
        Called every physics step (60 Hz).
        """
        # Move Obstacle_1 if enabled
        if self.obstacle_1_moving:
            self._move_obstacle()

        # Update sensors and log data periodically (every 30 frames = 0.5 seconds at 60 Hz)
        if not hasattr(self, '_sensor_log_counter'):
            self._sensor_log_counter = 0

        self._sensor_log_counter += 1
        if self._sensor_log_counter >= 30:  # Log every 0.5 seconds
            self._sensor_log_counter = 0

            # Process and log Lidar data
            if self.lidar is not None:
                self._process_lidar_data()

            # Process and log Depth Camera data
            if self.depth_camera is not None:
                self._process_depth_camera_data()

    def _move_obstacle(self):
        """
        Move Obstacle_1 automatically using PhysX Force API (acceleration mode).
        Applies continuous acceleration to move obstacle back and forth between min_x and max_x.

        Logic:
        - Starts at x=0.63m (right boundary)
        - When x <= 0.2m (left boundary): acceleration = +6 m/s² (move right)
        - When x >= 0.63m (right boundary): acceleration = -6 m/s² (move left)
        """
        if "obstacle_1" not in self.obstacles:
            return

        try:
            obstacle = self.obstacles["obstacle_1"]
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(obstacle.prim_path)

            if not prim or not prim.IsValid():
                return

            # Apply and configure PhysX Force API once
            if not self.obstacle_1_force_api_applied:
                # Apply PhysxForceAPI if not already applied
                if not prim.HasAPI(PhysxSchema.PhysxForceAPI):
                    PhysxSchema.PhysxForceAPI.Apply(prim)

                force_api = PhysxSchema.PhysxForceAPI(prim)

                # Configure Force API attributes (set once)
                force_api.CreateForceEnabledAttr().Set(True)  # Enable force
                force_api.CreateWorldFrameEnabledAttr().Set(True)  # Use world frame
                force_api.CreateModeAttr().Set("acceleration")  # Use "acceleration" mode

                self.obstacle_1_force_api_applied = True

            # Get current position to check boundaries
            current_pos, _ = obstacle.get_world_pose()

            # Determine acceleration based on position
            # When at left boundary (x <= 0.2): apply +6 m/s² to move right
            # When at right boundary (x >= 0.63): apply -6 m/s² to move left
            if current_pos[0] <= self.obstacle_1_min_x:
                # At or past left boundary - accelerate right
                acceleration = self.obstacle_1_acceleration  # +6 m/s²
                if not hasattr(self, '_last_accel') or self._last_accel != acceleration:
                    self._last_accel = acceleration
            elif current_pos[0] >= self.obstacle_1_max_x:
                # At or past right boundary - accelerate left
                acceleration = -self.obstacle_1_acceleration  # -6 m/s²
                if not hasattr(self, '_last_accel') or self._last_accel != acceleration:
                    self._last_accel = acceleration
            else:
                # Between boundaries - maintain current acceleration
                if not hasattr(self, '_last_accel'):
                    # First time in middle - start moving left from initial position
                    acceleration = -self.obstacle_1_acceleration  # -6 m/s²
                    self._last_accel = acceleration
                else:
                    acceleration = self._last_accel

            # Apply acceleration every physics step (acceleration is consumed each step)
            force_api = PhysxSchema.PhysxForceAPI(prim)
            acceleration_vector = Gf.Vec3f(
                acceleration,  # X-axis acceleration (±6 m/s²)
                0.0,  # No Y-axis acceleration
                0.0   # No Z-axis acceleration
            )
            force_api.GetForceAttr().Set(acceleration_vector)

            # CRITICAL: Update RRT's internal representation after moving obstacle
            # This ensures RRT knows the current position of Obstacle_1 for collision avoidance
            if self.rrt is not None:
                self.rrt.update_world()

        except Exception as e:
            carb.log_warn(f"[OBSTACLE] Error moving Obstacle_1: {e}")
            import traceback
            traceback.print_exc()

    def _load_rl_model(self):
        """Load trained RL model for object selection (supports both PPO and DDQN)"""
        if not self.use_rl or not self.rl_model_path:
            return

        try:
            print(f"\n[RL] Loading {self.rl_model_type.upper()} model from: {self.rl_model_path}")

            # Try to load metadata to get training_grid_size
            if self.rl_model_type == 'ppo':
                metadata_path = self.rl_model_path.replace("_final.zip", "_metadata.json")
            else:  # ddqn
                metadata_path = self.rl_model_path.replace(".pt", "_metadata.json")

            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    model_grid_size = metadata.get("training_grid_size", self.training_grid_size)

                    # Warn if mismatch
                    if model_grid_size != self.training_grid_size:
                        self.training_grid_size = model_grid_size

            # Create dummy environment for loading
            max_objects = self.training_grid_size * self.training_grid_size
            dummy_env = ObjectSelectionEnv(
                franka_controller=self,
                max_objects=max_objects,
                max_steps=50,
                training_grid_size=self.training_grid_size
            )

            # Load model based on type
            if self.rl_model_type == 'ppo':
                # Wrap with ActionMasker for action masking support (PPO only)
                def mask_fn(env):
                    return env.action_masks()
                dummy_env = ActionMasker(dummy_env, mask_fn)
                vec_env = DummyVecEnv([lambda: dummy_env])

                # Load VecNormalize if exists
                vecnorm_path = self.rl_model_path.replace("_final.zip", "_vecnormalize.pkl")
                if Path(vecnorm_path).exists():
                    vec_env = VecNormalize.load(vecnorm_path, vec_env)
                    vec_env.training = False
                    vec_env.norm_reward = False

                # Load PPO model
                self.rl_model = MaskablePPO.load(self.rl_model_path, env=vec_env)
                print("[RL] PPO model loaded successfully!")

            else:  # ddqn
                # DDQN uses custom DoubleDQNAgent (not Stable-Baselines3)
                # Load checkpoint to get model parameters
                checkpoint = torch.load(self.rl_model_path, map_location='cpu')

                # Create agent with saved parameters
                self.rl_model = DoubleDQNAgent(
                    state_dim=checkpoint['state_dim'],
                    action_dim=checkpoint['action_dim'],
                    gamma=checkpoint['gamma'],
                    epsilon_start=checkpoint['epsilon'],  # Use saved epsilon
                    epsilon_end=checkpoint['epsilon_end'],
                    epsilon_decay=checkpoint['epsilon_decay'],
                    batch_size=checkpoint['batch_size'],
                    target_update_freq=checkpoint['target_update_freq']
                )

                # Load network weights
                self.rl_model.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.rl_model.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                self.rl_model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.rl_model.epsilon = checkpoint['epsilon']
                self.rl_model.steps = checkpoint['steps']
                self.rl_model.episodes = checkpoint['episodes']

                # Set to evaluation mode
                self.rl_model.policy_net.eval()
                self.rl_model.target_net.eval()

                print("[RL] DDQN model loaded successfully!")
                print(f"[RL] Loaded from step {checkpoint['steps']}, episode {checkpoint['episodes']}, epsilon {checkpoint['epsilon']:.4f}")

        except Exception as e:
            print(f"[RL] Error loading model: {e}")
            import traceback
            traceback.print_exc()
            print("[RL] Falling back to greedy baseline")
            self.rl_model = None
            self.use_rl = False

    def _get_observation(self, picked_indices=None):
        """
        Get observation for RL model (same format as training environment).

        UPDATED: 6 values per object (added picked flag)
        1. Distance to robot EE: 1 value
        2. Distance to container: 1 value
        3. Obstacle proximity score: 1 value
        4. Reachability flag: 1 value
        5. Path clearance score: 1 value
        6. Picked flag: 1 value (0.0 = available, 1.0 = already picked)

        Args:
            picked_indices: List of already picked object indices

        Returns:
            Observation array (flattened, shape: (max_objects * 6,))
        """
        if picked_indices is None:
            picked_indices = []

        # Use training grid size for observation space (e.g., 4x4 = 16)
        max_objects = self.training_grid_size * self.training_grid_size
        obs = np.zeros((max_objects, 6), dtype=np.float32)

        # Get robot EE and container positions
        ee_pos, _ = self.franka.end_effector.get_world_pose()
        container_pos, _ = self.container.get_world_pose()

        for i, (cube, _) in enumerate(self.cubes):
            cube_pos, _ = cube.get_world_pose()

            # 1. Distance to EE (1 value)
            dist_to_ee = np.linalg.norm(cube_pos - ee_pos)
            obs[i, 0] = dist_to_ee

            # 2. Distance to container (1 value)
            obs[i, 1] = np.linalg.norm(cube_pos - container_pos)

            # 3. Obstacle proximity score (1 value) - simplified, no obstacles
            obs[i, 2] = 0.0

            # 4. Reachability flag (1 value)
            # Franka workspace: ~30cm to 90cm from base
            reachable = 1.0 if (0.3 <= dist_to_ee <= 0.9) else 0.0
            obs[i, 3] = reachable

            # 5. Path clearance score (1 value) - simplified
            # For now, assume clear path (no obstacles)
            obs[i, 4] = 1.0

            # 6. Picked flag (1 value) - NEW
            # 0.0 = available to pick, 1.0 = already picked
            picked_flag = 1.0 if i in picked_indices else 0.0
            obs[i, 5] = picked_flag

        return obs.flatten()

    def _get_rl_pick_order(self):
        """
        Get optimal picking order using RL model or greedy baseline.

        Returns:
            List of cube indices in optimal picking order
        """
        if not self.cubes:
            return []

        # Get end effector position
        ee_pos, _ = self.franka.end_effector.get_world_pose()

        # Calculate distances for all cubes
        distances = []
        for i, (cube, cube_name) in enumerate(self.cubes):
            cube_pos, _ = cube.get_world_pose()
            dist = np.linalg.norm(cube_pos - ee_pos)
            distances.append((i, dist, cube_name))

        # Use RL model if available
        if self.use_rl and self.rl_model is not None:
            try:
                print("\n[RL] Using RL model for object selection...")

                # Use RL model to predict picking order
                pick_order = []
                picked_indices = []
                total_objects = len(self.cubes)

                for step in range(total_objects):
                    # Get current observation
                    obs = self._get_observation(picked_indices=picked_indices)

                    # Get action mask (prevent selecting already-picked objects)
                    action_mask = np.zeros(self.training_grid_size * self.training_grid_size, dtype=bool)
                    for idx in range(total_objects):
                        if idx not in picked_indices:
                            action_mask[idx] = True

                    # Predict next action based on model type
                    if self.rl_model_type == 'ppo':
                        # PPO uses .predict() with action_masks parameter
                        action, _ = self.rl_model.predict(obs, action_masks=action_mask, deterministic=True)
                        action = int(action)
                    else:  # ddqn
                        # DDQN uses .select_action() with epsilon=0 for greedy (deterministic) selection
                        # Flatten observation for DDQN (expects 1D state vector)
                        obs_flat = obs.flatten()
                        # Convert to tensor and move to same device as model
                        obs_tensor = torch.FloatTensor(obs_flat).to(self.rl_model.device)
                        action = self.rl_model.policy_net.get_action(obs_tensor, epsilon=0.0, action_mask=action_mask)

                    # Validate action (should never happen with action masking, but keep as safety check)
                    if action in picked_indices:
                        print(f"  [RL] Warning: Model predicted already-picked object {action} (action masking failed!)")
                        # Fallback: pick closest unpicked object
                        unpicked = [idx for idx, _, _ in distances if idx not in picked_indices]
                        if unpicked:
                            action = min(unpicked, key=lambda idx: distances[idx][1])

                    pick_order.append(action)
                    picked_indices.append(action)

                    # Debug output
                    _, dist, name = distances[action]
                    print(f"  Step {step+1}: Pick {name} (index {action}, distance: {dist:.3f}m)")

                print(f"[RL] Final pick order: {pick_order}")
                return pick_order

            except Exception as e:
                print(f"[RL] Error using RL model: {e}")
                import traceback
                traceback.print_exc()
                print("[RL] Falling back to greedy baseline")
                pick_order = [idx for idx, _, _ in sorted(distances, key=lambda x: x[1])]
        else:
            # Greedy baseline: pick closest objects first
            pick_order = [idx for idx, _, _ in sorted(distances, key=lambda x: x[1])]
            print(f"\n[GREEDY] Pick order (closest first): {pick_order}")
            for idx, dist, name in sorted(distances, key=lambda x: x[1]):
                print(f"  {idx}: {name} (distance: {dist:.3f}m)")

        return pick_order

    def _on_pick(self):
        """Pick and place button callback"""
        if not self.world or not self.rrt:
            self._update_status("Load scene first!")
            return

        if self.is_picking:
            # Pause
            self.is_picking = False
            self._update_status("Paused")
            self.timeline.pause()
        else:
            # Start or Resume
            self.is_picking = True

            # If starting fresh (current_cube_index is 0), reset placed_count
            if self.current_cube_index == 0:
                self.placed_count = 0
                self._update_status("Starting...")

            run_coroutine(self._pick_place_loop())

    async def _pick_place_loop(self):
        """Main pick and place loop"""
        try:
            self.timeline.play()

            # Add physics callback now that timeline is playing (physics context is ready)
            if not hasattr(self, '_physics_callback_added'):
                try:
                    self.world.add_physics_callback("sensor_and_obstacle_update", self._physics_step_callback)
                    self._physics_callback_added = True
                except Exception as e:
                    print(f"[PHYSICS] Warning: Could not add physics callback: {e}")
                    self._physics_callback_added = False

            # Start Obstacle_1 automatic movement
            self.obstacle_1_moving = True

            # IMPROVED: Better stabilization before first pick
            # Wait for robot to fully settle in default position
            for _ in range(25):  # Reduced for better FPS
                await omni.kit.app.get_app().next_update_async()

            # Initialize gripper to closed position at start
            articulation_controller = self.franka.get_articulation_controller()
            articulation_controller.apply_action(ArticulationAction(
                joint_positions=self.gripper.joint_closed_positions, joint_indices=np.array([7, 8])))
            for _ in range(8):  # Reduced for better FPS
                await omni.kit.app.get_app().next_update_async()

            # IMPROVED: Explicitly set robot to default position before starting
            # This ensures RRT starts from a known, stable configuration
            default_joint_positions = np.array([0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75, 0.0, 0.0])
            articulation_controller.apply_action(ArticulationAction(joint_positions=default_joint_positions))
            for _ in range(8):  # Reduced for better FPS
                await omni.kit.app.get_app().next_update_async()

            cubes = self.cubes
            total_cubes = len(cubes)

            # Get optimal picking order (RL or greedy)
            if self.current_cube_index == 0:
                # Only compute pick order at the start
                pick_order = self._get_rl_pick_order()
                self.pick_order = pick_order  # Store for resume
            else:
                # Resume from stored pick order
                pick_order = self.pick_order if hasattr(self, 'pick_order') else list(range(total_cubes))

            # Pick cubes in optimal order
            for order_idx in range(self.current_cube_index, total_cubes):
                try:
                    cube_idx = pick_order[order_idx]
                    cube, cube_name = cubes[cube_idx]
                    cube_number = order_idx + 1
                    print(f"\n[{cube_number}/{total_cubes}] Picking {cube_name} (index {cube_idx})")

                    # Update current cube index (this is the actual cube index in self.cubes array)
                    self.current_cube_index = cube_idx

                    # Call pick and place (retry logic is now INSIDE the function)
                    success, error_msg = await self._pick_and_place_cube(cube, cube_name.split()[1])  # Extract "1", "2", etc.

                    if success:
                        self.placed_count += 1
                        print(f"OK")
                    else:
                        print(f"SKIP: {error_msg}")
                    self._update_status(f"{self.placed_count}/{total_cubes} placed")

                    self.current_cube_index += 1

                except Exception as cube_error:
                    # If there's an error with this cube, skip it and continue to next
                    print(f"ERROR: {str(cube_error)}")
                    self._update_status(f"{self.placed_count}/{total_cubes} placed")
                    self.current_cube_index += 1
                    continue

            print(f"Done: {self.placed_count}/{total_cubes}")
            self._update_status(f"Done: {self.placed_count}/{total_cubes} placed")
            self.is_picking = False
            self.current_cube_index = 0  # Reset for next run

            # Stop Obstacle_1 automatic movement
            self.obstacle_1_moving = False

            # Clear selection to avoid "invalid prim" errors when stopping
            selection = omni.usd.get_context().get_selection()
            selection.clear_selected_prim_paths()

            # Small delay before stopping to allow cleanup
            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            self.timeline.stop()

        except Exception as e:
            self._update_status(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.is_picking = False

            # Stop Obstacle_1 automatic movement on error
            self.obstacle_1_moving = False

            self.timeline.stop()

    async def _pick_and_place_cube(self, cube, cube_name):
        """Pick and place cube using RRT (8 phases: pick with retry, place, return home)"""
        try:
            total_cubes = self.num_cubes  # Actual number of cubes spawned
            cube_size = 0.0515
            cube_half = cube_size / 2.0
            orientation = euler_angles_to_quats(np.array([np.pi, 0, 0]))
            max_pick_attempts = 3  # Retry up to 3 times for pick failures (gripper didn't grab cube)
            pick_success = False

            # Get current cube position for obstacle detection
            cube_pos_initial, _ = cube.get_world_pose()

            # PHASE 0: Add NEARBY cubes as obstacles (not the target cube)
            # Only cubes within collision radius will be added to avoid RRT failures
            nearby_obstacle_count = self._add_other_cubes_as_obstacles(self.current_cube_index, cube_pos_initial)

            # CRITICAL FIX: Always reset to safe config before picking
            # This ensures:
            # 1. Robot starts from known-good configuration (valid even with obstacles)
            # 2. Consistent approach angles for all cubes
            # 3. No awkward rotations from arbitrary previous positions
            if nearby_obstacle_count >= 3:
                print(f"  {nearby_obstacle_count} nearby obstacles detected - resetting to safe config first")
            await self._reset_to_safe_config()

            for pick_attempt in range(1, max_pick_attempts + 1):
                if pick_attempt > 1:
                    print(f"  Retry {pick_attempt}/{max_pick_attempts} (pick failed)")
                    # Reset to safe config on retry to ensure valid starting configuration
                    await self._reset_to_safe_config()

                cube_pos_current, _ = cube.get_world_pose()

                # Phase 1: Open gripper fully before approaching
                articulation_controller = self.franka.get_articulation_controller()
                articulation_controller.apply_action(ArticulationAction(
                    joint_positions=self.gripper.joint_opened_positions, joint_indices=np.array([7, 8])))
                for _ in range(5):  # Increased from 2 to ensure gripper is fully open
                    await omni.kit.app.get_app().next_update_async()

                # Phase 2: Move to high waypoint (37cm above cube)
                # Let RRT find the path directly - if configuration is invalid, RRT will fail here
                high_waypoint = np.array([cube_pos_current[0], cube_pos_current[1], 0.37])
                success = await self._move_to_target_rrt(high_waypoint, orientation, skip_factor=6)

                if not success:
                    # RRT failure - skip to next cube immediately (no retries)
                    print(f"  RRT failed to reach above cube - skipping to next cube")
                    self._remove_all_cube_obstacles()
                    await self._reset_to_safe_config()
                    return False, f"RRT failed to reach above {cube_name}"

                # Extra stabilization after reaching high waypoint
                for _ in range(5):
                    await omni.kit.app.get_app().next_update_async()

                # Phase 3: Pick approach (descend straight down to cube center for proper grip)
                cube_pos_realtime, _ = cube.get_world_pose()
                # Target cube center (3.5cm) so gripper fingers straddle the middle
                pick_pos = np.array([cube_pos_realtime[0], cube_pos_realtime[1], 0.035])

                # Slow descent for precision
                success = await self._move_to_target_rrt(pick_pos, orientation, skip_factor=3)
                if not success:
                    # RRT failure - skip to next cube immediately (no retries)
                    print(f"  RRT failed pick approach - skipping to next cube")
                    self._remove_all_cube_obstacles()
                    await self._reset_to_safe_config()
                    return False, f"RRT failed pick approach for {cube_name}"

                for _ in range(5):  # Pick stabilization
                    await omni.kit.app.get_app().next_update_async()

                # CRITICAL: Check if retreat is possible BEFORE closing gripper
                # This prevents getting stuck in invalid configuration after gripper closes
                # IMPORTANT: Skip Lidar obstacle updates to avoid detecting target cube as obstacle
                retreat_pos = np.array([cube_pos_realtime[0], cube_pos_realtime[1], 0.37])

                # Clear Lidar-detected obstacles temporarily (they interfere with retreat planning)
                self._clear_lidar_obstacles()

                # Plan retreat WITHOUT updating dynamic obstacles (update_obstacles=False)
                retreat_plan = self._plan_to_target(retreat_pos, orientation, update_obstacles=False)

                if retreat_plan is None:
                    # Retreat planning failed - skip this cube without closing gripper
                    print(f"  RRT cannot plan retreat from this position - skipping to next cube")
                    self._remove_all_cube_obstacles()
                    await self._reset_to_safe_config()
                    return False, f"RRT cannot plan retreat for {cube_name}"

                # Retreat is possible - safe to close gripper
                articulation_controller.apply_action(ArticulationAction(
                    joint_positions=self.gripper.joint_closed_positions, joint_indices=np.array([7, 8])))
                for _ in range(15):  # Gripper close
                    await omni.kit.app.get_app().next_update_async()

                # Wait for cube to settle in gripper before retreat
                for _ in range(7):  # Cube settling stabilization
                    await omni.kit.app.get_app().next_update_async()

                # Phase 4: Pick retreat - Go straight up from current cube position (not EE position)
                # This prevents rotation by maintaining XY position
                # We already have the retreat plan, so just execute it
                success = await self._move_to_target_rrt(retreat_pos, orientation, skip_factor=5)
                if not success:
                    # RRT failure - skip to next cube immediately (no retries)
                    print(f"  RRT failed pick retreat - skipping to next cube")
                    self.franka.gripper.open()
                    for _ in range(5):
                        await omni.kit.app.get_app().next_update_async()
                    self._remove_all_cube_obstacles()
                    await self._reset_to_safe_config()
                    return False, f"RRT failed pick retreat for {cube_name}"

                # Phase 5: Verify pick
                cube_pos_after_pick, _ = cube.get_world_pose()
                height_lifted = cube_pos_after_pick[2] - cube_pos_realtime[2]

                if height_lifted > 0.05:
                    print(f"  Pick OK ({height_lifted*100:.1f}cm)")
                    pick_success = True
                    break
                else:
                    print(f"  Pick fail ({height_lifted*100:.1f}cm)")
                    if pick_attempt < max_pick_attempts:
                        articulation_controller.apply_action(ArticulationAction(
                            joint_positions=self.gripper.joint_opened_positions, joint_indices=np.array([7, 8])))
                        for _ in range(3):
                            await omni.kit.app.get_app().next_update_async()
                        # Reset to safe position before retry
                        await self._reset_to_safe_config()
                        continue
                    else:
                        # Clean up before returning
                        self._remove_all_cube_obstacles()
                        # Reset to safe position before moving to next cube
                        await self._reset_to_safe_config()
                        return False, f"Failed to pick {cube_name}"

            # If we get here and pick didn't succeed, return failure
            if not pick_success:
                # Remove cube obstacles before returning
                self._remove_all_cube_obstacles()
                # Reset to safe position before moving to next cube
                await self._reset_to_safe_config()
                return False, f"Failed to pick {cube_name}"

            # PICK SUCCESSFUL - Remove cube obstacles temporarily
            # The picked cube is now held by gripper, and we'll re-add remaining cubes before place
            self._remove_all_cube_obstacles()

            container_center = np.array([0.30, 0.50, 0.0])
            # Container dimensions: [0.48m (X-length), 0.36m (Y-width), 0.128m (Z-height)]
            container_length = self.container_dimensions[0]  # X-axis: 0.48m
            container_width = self.container_dimensions[1]   # Y-axis: 0.36m

            # Calculate grid size from ACTUAL number of cubes (not training grid size)
            # This ensures compact placement regardless of training grid size
            # e.g., 9 cubes -> 3x3, 4 cubes -> 2x2, 6 cubes -> 3x2
            place_grid_size = int(np.ceil(np.sqrt(total_cubes)))
            place_row = self.placed_count // place_grid_size
            place_col = self.placed_count % place_grid_size

            # IMPROVED: Increased margins to prevent gripper collision with container walls
            # Larger margins on all sides for safety
            if total_cubes <= 4:
                edge_margin_left = 0.13   # Increased from 0.11 to prevent gripper collision
                edge_margin_right = 0.11  # Increased from 0.09 for safety
                edge_margin_width = 0.11  # Increased from 0.09 for safety
            elif total_cubes <= 9:
                edge_margin_left = 0.13   # Increased from 0.11 to prevent gripper collision
                edge_margin_right = 0.11  # Increased from 0.09 for safety
                edge_margin_width = 0.11  # Increased from 0.09 for safety
            else:
                edge_margin_left = 0.13   # Increased from 0.11 to prevent gripper collision
                edge_margin_right = 0.11  # Increased from 0.09 for safety
                edge_margin_width = 0.11  # Increased from 0.09 for safety

            # Calculate usable space with asymmetric margins
            usable_length = container_length - edge_margin_left - edge_margin_right
            usable_width = container_width - (2 * edge_margin_width)
            spacing_length = usable_length / (place_grid_size - 1) if place_grid_size > 1 else 0.0
            spacing_width = usable_width / (place_grid_size - 1) if place_grid_size > 1 else 0.0

            # Start from left edge with larger margin
            start_x = container_center[0] - (container_length / 2.0) + edge_margin_left
            start_y = container_center[1] - (container_width / 2.0) + edge_margin_width
            cube_x = start_x + (place_row * spacing_length)
            cube_y = start_y + (place_col * spacing_width)

            # Verify placement is within container bounds
            container_x_min = container_center[0] - container_length / 2.0
            container_x_max = container_center[0] + container_length / 2.0
            container_y_min = container_center[1] - container_width / 2.0
            container_y_max = container_center[1] + container_width / 2.0

            if cube_x < container_x_min or cube_x > container_x_max or cube_y < container_y_min or cube_y > container_y_max:
                print(f"  WARNING: Cube {self.placed_count + 1} placement outside container!")
                print(f"    Position: ({cube_x:.3f}, {cube_y:.3f})")
                print(f"    Container X: [{container_x_min:.3f}, {container_x_max:.3f}]")
                print(f"    Container Y: [{container_y_min:.3f}, {container_y_max:.3f}]")

            place_height = cube_half + 0.005
            place_pos = np.array([cube_x, cube_y, place_height])

            # PLACE PHASE - Add NEARBY unpicked cubes as obstacles
            # Only add cubes within collision radius of the place position
            # Reduced radius to minimize RRT failures
            collision_radius = 0.15  # 15cm radius (only adjacent cubes, excludes diagonals)
            for i in range(self.current_cube_index + 1, len(self.cubes)):
                other_cube, _ = self.cubes[i]
                other_pos, _ = other_cube.get_world_pose()
                distance = np.linalg.norm(other_pos[:2] - place_pos[:2])
                if distance < collision_radius:
                    try:
                        self.rrt.add_obstacle(other_cube, static=False)
                        self.cube_obstacles_enabled = True
                    except Exception:
                        pass

            # Use higher waypoints to provide clearance for held cube
            pre_place_pos = place_pos + np.array([0.0, 0.0, 0.28])

            # Phase 5: Pre-place - Add intermediate waypoint for smoother approach
            # This prevents complex RRT paths and ensures straight descent
            via_point = np.array([0.35, 0.30, 0.40])
            await self._move_to_target_rrt(via_point, orientation, skip_factor=5)

            # Approach from directly above the place position for straight descent
            above_place = np.array([place_pos[0], place_pos[1], 0.35])
            await self._move_to_target_rrt(above_place, orientation, skip_factor=5)
            await self._move_to_target_rrt(pre_place_pos, orientation, skip_factor=4)

            # Phase 6: Place approach (slightly slower for precision)
            release_height = place_pos + np.array([0.0, 0.0, 0.08])
            await self._move_to_target_rrt(release_height, orientation, skip_factor=3)
            for _ in range(1):  # Place stabilization
                await omni.kit.app.get_app().next_update_async()

            articulation_controller = self.franka.get_articulation_controller()
            articulation_controller.apply_action(ArticulationAction(
                joint_positions=self.gripper.joint_opened_positions, joint_indices=np.array([7, 8])))
            for _ in range(12):  # Gripper open (increased from 10 to prevent cube throw)
                await omni.kit.app.get_app().next_update_async()

            # PLACE COMPLETE - Remove cube obstacles
            self._remove_all_cube_obstacles()

            # Verify placement
            cube_pos_final, _ = cube.get_world_pose()
            xy_distance = np.linalg.norm(cube_pos_final[:2] - place_pos[:2])
            placement_successful = (xy_distance < 0.15) and (cube_pos_final[2] < 0.15)

            if placement_successful:
                print(f"  Place OK ({xy_distance*100:.1f}cm)")
            else:
                print(f"  Place fail ({xy_distance*100:.1f}cm)")

            # Phase 7: Place retreat (faster with skip_factor=6)
            current_ee_pos, _ = self.franka.end_effector.get_world_pose()
            retreat_pos = current_ee_pos + np.array([0.0, 0.0, 0.15])  # Retreat up
            await self._move_to_target_rrt(retreat_pos, orientation, skip_factor=6)

            articulation_controller.apply_action(ArticulationAction(
                joint_positions=self.gripper.joint_closed_positions, joint_indices=np.array([7, 8])))
            for _ in range(2):  # Gripper close
                await omni.kit.app.get_app().next_update_async()

            # Move to a safe intermediate position using RRT to avoid obstacles
            # Reduced height from 0.50 to 0.35 (15cm lower)
            safe_ee_position = np.array([0.40, 0.0, 0.35])  # Centered position, lower height
            safe_success = await self._move_to_target_rrt(safe_ee_position, orientation, skip_factor=5)

            if not safe_success:
                # If RRT fails to reach safe position, use direct reset as last resort
                # This may hit obstacles but ensures robot doesn't stay in invalid config
                await self._reset_to_safe_config()

            return (True, "") if placement_successful else (False, f"{cube_name} not in container")

        except Exception as e:
            # Clean up cube obstacles on error
            self._remove_all_cube_obstacles()

            error_msg = f"Error picking/placing Cube {cube_name}: {str(e)}"
            import traceback
            traceback.print_exc()
            return False, error_msg  # Failed

    async def _move_to_target_ik(self, target_position, target_orientation, num_steps=8):
        """
        Move to target using IK directly (for simple, straight-line movements)
        This is faster and more predictable than RRT for vertical descents

        Args:
            target_position: Target end effector position
            target_orientation: Target end effector orientation
            num_steps: Number of interpolation steps (default: 8, reduced for FPS)

        Returns:
            bool: True if successful, False if IK failed
        """
        # Update robot base pose
        robot_base_translation, robot_base_orientation = self.franka.get_world_pose()
        self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        # Compute IK solution
        ik_action, ik_success = self.articulation_kinematics_solver.compute_inverse_kinematics(
            target_position, target_orientation
        )

        if not ik_success:
            return False

        # Get current joint positions
        current_positions = self.franka.get_joint_positions()[:7]  # First 7 joints (arm only)
        target_positions = ik_action.joint_positions[:7]

        # Interpolate from current to target
        articulation_controller = self.franka.get_articulation_controller()
        for i in range(num_steps):
            alpha = (i + 1) / num_steps
            interpolated_positions = current_positions + alpha * (target_positions - current_positions)

            # Create action for arm joints only (indices 0-6)
            action = ArticulationAction(
                joint_positions=interpolated_positions,
                joint_indices=np.array([0, 1, 2, 3, 4, 5, 6])
            )
            articulation_controller.apply_action(action)
            await omni.kit.app.get_app().next_update_async()

        return True

    async def _move_to_target_rrt(self, target_position, target_orientation, skip_factor=3):
        """
        Move to target using RRT (for long-distance collision-free motion)

        Args:
            target_position: Target end effector position
            target_orientation: Target end effector orientation
            skip_factor: Frame skip factor for execution speed (default=3 for 60 FPS)

        Returns:
            bool: True if successful, False if planning/execution failed
        """
        plan = self._plan_to_target(target_position, target_orientation)
        if plan is None:
            return False
        await self._execute_plan(plan, skip_factor=skip_factor)

        # CRITICAL: Wait for robot to stabilize after motion
        # Without this, next RRT planning may see invalid joint configuration
        for _ in range(10):  # 10 frames @ 60Hz = ~167ms stabilization
            await omni.kit.app.get_app().next_update_async()

        return True

    async def _reset_to_safe_config(self):
        """
        Reset robot to a known safe configuration using smooth RRT motion.
        This ensures the robot starts from a valid configuration that works even with obstacles.
        """
        # First, open gripper to prepare for next pick
        articulation_controller = self.franka.get_articulation_controller()
        articulation_controller.apply_action(ArticulationAction(
            joint_positions=self.gripper.joint_opened_positions, joint_indices=np.array([7, 8])))
        for _ in range(3):  # Reduced from 5 to 3 for faster operation
            await omni.kit.app.get_app().next_update_async()

        # Move to safe end-effector position using RRT (smooth motion)
        # This position corresponds to the safe joint configuration
        # Safe config: [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]
        # Corresponding EE position: approximately [0.40, 0.0, 0.35]
        safe_ee_position = np.array([0.40, 0.0, 0.35])
        orientation = euler_angles_to_quats(np.array([np.pi, 0, 0]))

        # Use RRT for smooth motion to safe position with faster skip_factor
        await self._move_to_target_rrt(safe_ee_position, orientation, skip_factor=6)

        # CRITICAL: Longer stabilization to ensure joint positions are fully updated
        # This prevents "Invalid configuration" errors in subsequent RRT planning
        for _ in range(30):  # Increased from 10 to 30 frames @ 60Hz = ~500ms stabilization
            await omni.kit.app.get_app().next_update_async()

    def _plan_to_target(self, target_position, target_orientation, update_obstacles=True):
        """Plan path to target using RRT with smooth trajectory generation

        Args:
            target_position: Target end-effector position
            target_orientation: Target end-effector orientation
            update_obstacles: If True, update dynamic obstacles from Lidar (default: True)
                             Set to False when planning retreat to avoid detecting target cube
        """
        # Update dynamic obstacles from Lidar before planning (real-time detection)
        if update_obstacles:
            self._update_dynamic_obstacles()

        robot_base_translation, robot_base_orientation = self.franka.get_world_pose()
        self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        self.rrt.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        self.rrt.update_world()

        _, ik_success = self.articulation_kinematics_solver.compute_inverse_kinematics(target_position, target_orientation)
        if not ik_success:
            carb.log_warn(f"IK failed for {target_position}")
            return None

        self.rrt.set_end_effector_target(target_position, target_orientation)
        self.rrt.update_world()

        active_joints = self.path_planner_visualizer.get_active_joints_subset()
        start_pos = active_joints.get_joint_positions()

        # Validate current robot configuration (only check for NaN and extreme values)
        if np.any(np.isnan(start_pos)) or np.any(np.abs(start_pos) > 10.0):
            carb.log_error(f"Invalid robot config: {start_pos}")
            return None

        # Check if robot is near Obstacle_1 (moving obstacle)
        near_obstacle_1 = False
        if self.obstacle_1_moving and "obstacle_1" in self.obstacles:
            obs_pos, _ = self.obstacles["obstacle_1"].get_world_pose()
            ee_pos, _ = self.franka.end_effector.get_world_pose()
            distance_to_obstacle = np.linalg.norm(ee_pos[:2] - obs_pos[:2])  # XY distance only
            near_obstacle_1 = distance_to_obstacle < 0.30  # Within 30cm of Obstacle_1

        has_obstacles = self.obstacle_counter > 0

        # Reduced max_iterations for faster failure detection (1-2 seconds)
        # If RRT can't find path quickly, cube is likely unreachable
        if near_obstacle_1:
            max_iterations = 3000  # Fast failure near Obstacle_1
        elif has_obstacles:
            max_iterations = 5000  # Fast failure with static obstacles
        else:
            max_iterations = 4000  # Fast failure without obstacles

        self.rrt.set_max_iterations(max_iterations)

        rrt_plan = self.rrt.compute_path(start_pos, np.array([]))

        if rrt_plan is None or len(rrt_plan) <= 1:
            carb.log_warn(f"RRT failed for {target_position}")
            return None

        return self._convert_rrt_plan_to_trajectory(rrt_plan)

    def _convert_rrt_plan_to_trajectory(self, rrt_plan):
        """Convert RRT waypoints to smooth trajectory"""
        interpolated_path = self.path_planner_visualizer.interpolate_path(rrt_plan, 0.015)
        trajectory = self.cspace_trajectory_generator.compute_c_space_trajectory(interpolated_path)
        art_trajectory = ArticulationTrajectory(self.franka, trajectory, 1.0 / 60.0)
        return art_trajectory.get_action_sequence()

    async def _execute_plan(self, action_sequence, skip_factor=3):
        """Execute trajectory action sequence

        Args:
            action_sequence: Sequence of actions to execute
            skip_factor: Number of frames to skip (higher = faster, default=3 for 60 FPS)
        """
        if action_sequence is None or len(action_sequence) == 0:
            return False

        # Skip frames for faster motion (skip_factor=3 is optimal for 60 FPS)
        for i, action in enumerate(action_sequence):
            if i % skip_factor == 0:
                self.franka.apply_action(action)
                await omni.kit.app.get_app().next_update_async()

        return True

    def compute_forward_kinematics(self):
        """
        Compute forward kinematics to get current end effector pose
        Returns: (position, rotation_matrix) or (None, None) if solver not initialized
        """
        if self.articulation_kinematics_solver is None:
            carb.log_warn("Articulation kinematics solver not initialized")
            return None, None

        # Update robot base pose
        robot_base_translation, robot_base_orientation = self.franka.get_world_pose()
        self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        # Compute end effector pose
        ee_position, ee_rot_mat = self.articulation_kinematics_solver.compute_end_effector_pose()

        return ee_position, ee_rot_mat

    def _on_reset(self):
        """Reset button callback - Delete all prims from stage"""
        try:
            self.is_picking = False
            self.obstacle_1_moving = False

            # Stop timeline first
            self.timeline.stop()

            # STEP 1: Clear World instance first
            if self.world is not None:
                try:
                    World.clear_instance()
                except Exception as e:
                    print(f"[RESET] Error clearing World instance: {e}")

            # STEP 2: Delete ALL prims from stage
            import omni.usd
            import omni.kit.commands
            from omni.isaac.core.utils.stage import clear_stage

            stage = omni.usd.get_context().get_stage()

            if stage:
                # Get all top-level prims under /World
                world_prim = stage.GetPrimAtPath("/World")
                if world_prim.IsValid():
                    # Collect all children paths
                    children_paths = [str(child.GetPath()) for child in world_prim.GetAllChildren()]
                    if children_paths:
                        omni.kit.commands.execute('DeletePrims', paths=children_paths)

                # Also delete /World itself and recreate it
                if world_prim.IsValid():
                    omni.kit.commands.execute('DeletePrims', paths=['/World'])

            # Clear the entire stage
            clear_stage()

            # STEP 3: Reset all state variables
            self.world = None
            self.franka = None
            self.gripper = None
            self.container = None
            self.rrt = None
            self.path_planner_visualizer = None
            self.cspace_trajectory_generator = None
            self.kinematics_solver = None
            self.articulation_kinematics_solver = None
            self.cubes = []
            self.obstacles = {}
            self.obstacle_counter = 0
            self.lidar = None
            self.lidar_detected_obstacles = {}
            self.depth_camera = None
            self.placed_count = 0
            self.current_cube_index = 0
            self.is_picking = False
            self.obstacle_1_moving = False
            self.obstacle_1_force_api_applied = False
            if hasattr(self, '_last_accel'):
                delattr(self, '_last_accel')
            self._physics_callback_added = False

            # Reset UI
            self.load_btn.enabled = True
            self.pick_btn.enabled = False
            self.reset_btn.enabled = False
            self.add_obstacle_btn.enabled = False
            self.remove_obstacle_btn.enabled = False

            print("[RESET] Reset complete")
            self._update_status("Reset complete - stage cleared")

        except Exception as e:
            self._update_status(f"Error resetting: {e}")
            import traceback
            traceback.print_exc()

    def _on_add_obstacle(self):
        """Add obstacle button callback"""
        if not self.world or not self.rrt:
            self._update_status("Load scene first!")
            return

        try:
            # Generate unique obstacle name
            self.obstacle_counter += 1
            obstacle_name = f"obstacle_{self.obstacle_counter}"

            # Generate unique prim path
            obstacle_prim_path = find_unique_string_name(
                initial_name=f"/World/Obstacle_{self.obstacle_counter}",
                is_unique_fn=lambda x: not is_prim_path_valid(x)
            )

            # Calculate obstacle position based on number of existing obstacles (not counter)
            # This ensures positions cycle correctly even when obstacles are removed
            # Position 1: [0.63, 0.26, 0.11] (original position)
            # Position 2: [0.63, -0.39, 0.11] (opposite side of grid)
            # Position 3+: Each 8cm away from position 2 at same side

            num_existing_obstacles = len(self.obstacles)

            if num_existing_obstacles == 0:
                # First obstacle - original position
                obstacle_position = np.array([0.63, 0.26, 0.11])
            elif num_existing_obstacles == 1:
                # Second obstacle - opposite side of grid
                obstacle_position = np.array([0.63, -0.39, 0.11])
            else:
                # Third and subsequent obstacles - 8cm away from previous obstacle
                # Each obstacle is 8cm further in Y direction (Y decreases)
                offset = (num_existing_obstacles - 1) * 0.08  # 8cm spacing from position 2
                obstacle_position = np.array([0.63, -0.39 - offset, 0.11])

            obstacle_size = np.array([0.20, 0.05, 0.22])  # [length, width, height]

            # Create cube prim using prim_utils (creates Xform with Cube mesh as child)
            cube_prim = prim_utils.create_prim(
                prim_path=obstacle_prim_path,
                prim_type="Cube",
                position=obstacle_position,
                orientation=euler_angles_to_quats(np.array([0, 0, 0])),
                scale=obstacle_size,
                attributes={"size": 1.0}
            )

            # Set color (blue)
            stage = omni.usd.get_context().get_stage()
            cube_geom = UsdGeom.Cube.Get(stage, obstacle_prim_path)
            if cube_geom:
                cube_geom.CreateDisplayColorAttr([(0.0, 0.0, 1.0)])

            # Apply RigidBodyAPI for physics (dynamic, not kinematic)
            if not cube_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI.Apply(cube_prim)

            rigid_body_api = UsdPhysics.RigidBodyAPI(cube_prim)

            # CRITICAL: Set kinematic to False (dynamic rigid body that responds to acceleration)
            rigid_body_api.CreateKinematicEnabledAttr().Set(False)

            # Apply CollisionAPI
            if not cube_prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(cube_prim)

            # Set collision approximation to convex hull
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(cube_prim)
            mesh_collision_api.GetApproximationAttr().Set("convexHull")

            # Create FixedCuboid wrapper for RRT planner (doesn't create physics view, safe to add during simulation)
            obstacle = FixedCuboid(
                name=obstacle_name,
                prim_path=obstacle_prim_path,
                size=1.0,
                scale=obstacle_size
            )

            # Add to world scene
            self.world.scene.add(obstacle)

            # Add obstacle to RRT planner (static=False for dynamic obstacles)
            self.rrt.add_obstacle(obstacle, static=False)

            # Store obstacle reference
            self.obstacles[obstacle_name] = obstacle

            self._update_status(f"Obstacle added ({len(self.obstacles)})")

        except Exception as e:
            self._update_status(f"Error adding obstacle: {e}")
            import traceback
            traceback.print_exc()

    def _on_remove_obstacle(self):
        """Remove obstacle button callback - removes last added obstacle"""
        if not self.world or not self.rrt:
            self._update_status("Load scene first!")
            return

        if len(self.obstacles) == 0:
            self._update_status("No obstacles to remove!")
            return

        try:
            # Get the last added obstacle (highest counter number)
            obstacle_name = list(self.obstacles.keys())[-1]
            obstacle = self.obstacles[obstacle_name]

            # Get the prim path for USD deletion
            obstacle_prim_path = obstacle.prim_path

            print(f"Removing obstacle: {obstacle_name} at {obstacle_prim_path}")

            # CRITICAL: Remove from RRT planner first (before deleting prim)
            try:
                self.rrt.remove_obstacle(obstacle)
                print(f"  Removed from RRT planner")
            except Exception as e:
                print(f"  Warning: Could not remove from RRT planner: {e}")

            # Remove from our tracking dictionary BEFORE deleting prim
            del self.obstacles[obstacle_name]
            print(f"  Removed from tracking dictionary")

            # NOTE: Do NOT decrement obstacle_counter - keep incrementing to ensure unique names
            # Even if we remove obstacles, the world scene registry keeps track of used names
            # So we must always use new unique names for new obstacles

            # Delete the USD prim from stage using omni.kit.commands (safer than direct RemovePrim)
            import omni.kit.commands

            # Use DeletePrimsCommand for safe deletion
            omni.kit.commands.execute('DeletePrims', paths=[obstacle_prim_path])
            print(f"  Deleted USD prim from stage")

            print(f"Obstacle removed successfully: {obstacle_name} (Remaining: {len(self.obstacles)})")
            self._update_status(f"Obstacle removed ({len(self.obstacles)} remaining)")

        except Exception as e:
            self._update_status(f"Error removing obstacle: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function for standalone execution"""
    # Create application
    app = FrankaRRTDynamicGrid(num_cubes=args.num_cubes, training_grid_size=args.training_grid_size)

    # Keep simulation running
    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

