"""
Franka RRT Pick and Place - STANDALONE VERSION with RL Object Selection
RRT path planning with obstacle avoidance, conservative collision spheres,
dynamic grid configuration, pick retry logic, return to home after each cube.
Uses PhysX Lidar - Rotating and depth sensor for obstacle detection.

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
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder, VisualCuboid, FixedCuboid
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import PathPlannerVisualizer, ArticulationTrajectory
from isaacsim.robot_motion.motion_generation.lula import RRT
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from isaacsim.robot_motion.motion_generation.lula.trajectory_generator import LulaCSpaceTrajectoryGenerator
from pxr import UsdPhysics, PhysxSchema, Gf, UsdGeom, Sdf, Usd, UsdShade
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
from src.grippers import ParallelGripper, GraspConfig
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
        print("[RL] RL libraries loaded successfully")
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

        # Grasp configuration
        self.grasp_config = None
        self.current_grasp_name = "grasp_0"  # Default grasp (matches franka_rrt_9cylinders_v1.5.py)

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

        # Robot stand and table
        self.robot_stand = None
        self.table = None

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

            # Add textured table as robot stand
            assets_root_path = get_assets_root_path()
            stand_prim_path = "/World/RobotStand"
            stand_usd_path = assets_root_path + "/Isaac/Props/Mounts/textured_table.usd"
            add_reference_to_stage(usd_path=stand_usd_path, prim_path=stand_prim_path)

            # Position stand at 0.6m height with -90° Z rotation (quaternion)
            stand_translation = np.array([-0.1, 0.0, 0.6])  # Updated X to -0.1
            stand_orientation = np.array([0.7071, 0.0, 0.0, -0.7071])  # [w, x, y, z] for -90° Z rotation
            stand_scale = np.array([0.5, 0.4, 0.6])  # Added scale
            self.robot_stand = self.world.scene.add(
                SingleXFormPrim(
                    prim_path=stand_prim_path,
                    name="robot_stand",
                    translation=stand_translation,
                    orientation=stand_orientation,
                    scale=stand_scale
                )
            )

            # Make stand a static collider (no rigid body, no joints)
            stand_prim = stage.GetPrimAtPath(stand_prim_path)
            if stand_prim.IsValid():
                # Remove any rigid body API if it exists
                if stand_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    stand_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                # Add collision API for static collider
                if not stand_prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(stand_prim)



            # Add table for cylinders and container on ground plane
            table_prim_path = "/World/Table"
            table_usd_path = assets_root_path + "/Isaac/Environments/Simple_Room/Props/table_low.usd"
            add_reference_to_stage(usd_path=table_usd_path, prim_path=table_prim_path)

            # Position table in front of robot on ground plane with quaternion orientation
            table_top_height = 0.75  # Table top is at 75cm height
            table_orientation = np.array([0.7071, 0.0, 0.0, 0.7071])  # [w, x, y, z] for 90° Z rotation

            self.table = self.world.scene.add(
                SingleXFormPrim(
                    prim_path=table_prim_path,
                    name="table",
                    translation=np.array([0.5, 0.1, 0.0]),  # X: 0.5, Y: 0.1
                    orientation=table_orientation,
                    scale=np.array([0.9, 0.7, 0.9])
                )
            )

            # Make table static by setting it as a collider without rigid body dynamics
            table_prim = stage.GetPrimAtPath(table_prim_path)
            if table_prim:
                # Remove any rigid body API if it exists
                if table_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    table_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                # Ensure it has collision API for objects to rest on it
                if not table_prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(table_prim)

            # Apply material to table
            material_prim_path = "/World/Table/Looks/MI_Table"
            material_usd_path = assets_root_path + "/Isaac/Environments/Simple_Room/Materials/MI_Table.mdl"

            # Bind material to table_low prim
            table_low_prim_path = "/World/Table/table_low"
            table_low_prim = stage.GetPrimAtPath(table_low_prim_path)
            if table_low_prim:
                # Create material if it doesn't exist
                material = UsdShade.Material.Get(stage, material_prim_path)
                if not material:
                    material = UsdShade.Material.Define(stage, material_prim_path)
                    # Set MDL shader
                    mdl_shader = UsdShade.Shader.Define(stage, material_prim_path + "/Shader")
                    mdl_shader.CreateIdAttr("mdlMaterial")
                    mdl_shader.SetSourceAsset(material_usd_path, "mdl")
                    mdl_shader.SetSourceAssetSubIdentifier("MI_Table", "mdl")
                    material.CreateSurfaceOutput("mdl").ConnectToSource(mdl_shader.ConnectableAPI(), "out")

                # Bind material to table_low
                UsdShade.MaterialBindingAPI(table_low_prim).Bind(material)

            franka_name = f"franka_{int(time.time() * 1000)}"
            franka_prim_path = f"/World/Franka_{int(time.time() * 1000)}"

            franka_usd_path = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
            robot_prim = add_reference_to_stage(usd_path=franka_usd_path, prim_path=franka_prim_path)
            robot_prim.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
            robot_prim.GetVariantSet("Mesh").SetVariantSelection("Quality")

            # Store paths for later gripper/manipulator creation (after cylinders)
            robot_translation = stand_translation  # Same as stand (-0.1, 0, 0.6)
            self._franka_prim_path = franka_prim_path
            self._franka_name = franka_name
            self._end_effector_prim_path = f"{franka_prim_path}/panda_rightfinger"
            self._robot_translation = robot_translation



            # Add PhysX Lidar - Rotating sensor attached to Franka
            # Robot base now at Z=0.6m (stand height)
            # Cylinders on table at Z=0.75m (table top height)
            # Position Lidar relative to robot base to detect table-level obstacles
            # Attach to robot base for stable scanning
            lidar_prim_path = f"{franka_prim_path}/lidar_sensor"

            # Create PhysX Rotating Lidar
            # Position relative to robot base: at 15cm height to detect table-level obstacles
            lidar_translation = np.array([0.0, 0.0, 0.15])  # 15cm above robot base (0.6 + 0.15 = 0.75m world)

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
            print("[LIDAR] PhysX Rotating Lidar initialized: 20 Hz rotation, 360°×30° FOV, 1° resolution, point cloud range 0.4-100m")

            # Initialize Depth Camera at world level (not attached to panda hand)
            print("[DEPTH CAMERA] Initializing depth camera at world level...")
            depth_camera_prim_path = "/World/depth_camera"

            # Position in world coordinates
            # Orientation: XYZ Euler angles (0.0, 30.0, 90.0) in degrees
            # Depth camera position and orientation (user specified)
            # Position: (0.75, 0.1, 1.9) - high above workspace for better view
            # Orientation: (5.0, 0.0, 90.0) - Euler angles in degrees (XYZ)
            # We'll set orientation via USD API after creation to ensure correct axis
            position = np.array([0.75, 0.1, 1.9])  # World coordinates (X, Y, Z)
            # Use identity quaternion for now, will set rotation via USD API
            orientation = euler_angles_to_quats(np.array([0.0, 0.0, 0.0]), degrees=True)

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
            # The depth_camera_prim_path points to an Xform, the actual Camera is a child prim
            xform_prim = stage.GetPrimAtPath(depth_camera_prim_path)
            if xform_prim:
                # Set scale to 0.3 using Xformable API on the Xform prim
                xformable = UsdGeom.Xformable(xform_prim)
                # Get existing scale op or create new one
                scale_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]
                if scale_ops:
                    # Use existing scale op
                    scale_ops[0].Set(Gf.Vec3d(0.3, 0.3, 0.3))
                else:
                    # Create new scale op with double precision to match existing
                    xform_op_scale = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
                    xform_op_scale.Set(Gf.Vec3d(0.3, 0.3, 0.3))

                # Fix orientation: Modify existing Orient transform to (5.0, 0.0, 90.0) degrees
                # SingleViewDepthSensor creates an "orient" op, we need to update it
                orient_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpName() == "xformOp:orient"]
                if orient_ops:
                    # Modify existing orient op - convert (5.0, 0.0, 90.0) degrees to quaternion
                    # Using euler_angles_to_quats for accurate conversion
                    quat_orientation = euler_angles_to_quats(np.array([5.0, 0.0, 90.0]), degrees=True)
                    # Convert numpy array [w, x, y, z] to Gf.Quatd (w, x, y, z)
                    orient_ops[0].Set(Gf.Quatd(float(quat_orientation[0]), float(quat_orientation[1]),
                                               float(quat_orientation[2]), float(quat_orientation[3])))
                else:
                    # If no orient op exists, create rotateXYZ op
                    xform_op_rotate = xformable.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble)
                    xform_op_rotate.Set(Gf.Vec3d(5.0, 0.0, 90.0))

                # Find the Camera child prim (usually named "Camera" or similar)
                camera_prim = None
                for child in xform_prim.GetChildren():
                    if child.GetTypeName() == "Camera":
                        camera_prim = child
                        break

                if camera_prim:
                    # Set focal length and aperture for square pixels on the Camera prim
                    # Isaac Sim UI displays focal length in tenths of mm
                    # To show 13.0 in UI, we need to set 1.3mm (1.3mm × 10 = 13.0 display)
                    focal_length_attr = camera_prim.GetAttribute("focalLength")
                    if focal_length_attr:
                        focal_length_attr.Set(1.3)  # 1.3mm (displays as 13.0 in UI)
                        print(f"[DEPTH CAMERA] Focal length set to 1.3mm (displays as 13.0 in UI)")
                    else:
                        print(f"[DEPTH CAMERA] Warning: focalLength attribute not found")

                    camera_prim.GetAttribute("horizontalAperture").Set(20.955)
                    camera_prim.GetAttribute("verticalAperture").Set(20.955)  # Same as horizontal for square
                    camera_prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.01, 10000.0))
                else:
                    pass  # Camera child prim not found, but depth camera still works

            # Single update after robot, Lidar, and Depth Camera setup
            await omni.kit.app.get_app().next_update_async()

            # Add container on table
            container_prim_path = "/World/Container"
            container_usd_path = f"{get_assets_root_path()}/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Storage/Containers/Container_I/Container_I04_160x120x64cm_PR_V_NVD_01.usd"
            add_reference_to_stage(usd_path=container_usd_path, prim_path=container_prim_path)

            # Container position on table - positioned away from grid for clearance
            # Y=0.5 provides good separation from grid area
            # Distance from robot [-0.1, 0.0]: √((0.35+0.1)² + (0.5-0)²) = √(0.2025 + 0.25) = 0.67m
            container_position = np.array([0.35, 0.5, 0.6])  # Y=0.5 for clearance from grid
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

            # Create cylinder USD prims BEFORE world.reset()
            # Base dimensions (will be scaled by USD scale property)
            cylinder_radius = 0.0258  # ~5.15cm diameter (same as cube size)
            cylinder_height = 0.0515  # Same height as cube size

            # Calculate total_cubes for spacing logic
            total_cubes = self.num_cubes

            # OPTIMIZED spacing for better reachability
            # Reduced spacing to bring cylinders closer to robot
            # Robot base at [-0.1, 0.0], Franka reach ~0.855m
            if total_cubes <= 4:
                cylinder_spacing = 0.14  # Reduced from 0.18
            elif total_cubes <= 9:
                cylinder_spacing = 0.12  # Reduced from 0.15
            else:
                cylinder_spacing = 0.10  # Reduced from 0.13

            # OPTIMIZED grid center - moved away from robot base for better clearance
            # Grid positioned at comfortable distance from robot base [-0.1, 0.0]
            # This provides good reach while avoiding base collision
            grid_center_x = 0.40  # Positioned at comfortable distance
            grid_center_y = -0.08  # Centered in workspace
            # Use FIXED training grid size for grid extent
            # Grid extent = num_cells * cell_size (NOT (num_cells - 1) * cell_size)
            grid_extent_x = self.training_grid_size * cylinder_spacing
            grid_extent_y = self.training_grid_size * cylinder_spacing
            start_x = grid_center_x - (grid_extent_x / 2.0)
            start_y = grid_center_y - (grid_extent_y / 2.0)

            # Create visual grid at robot stand height (shows FULL training grid)
            # Z=0.61 is slightly above robot stand for better visibility
            create_visual_grid(start_x, start_y, grid_extent_x, grid_extent_y, cylinder_spacing, self.training_grid_size, self.training_grid_size, z_height=0.61)

            # PyGame-style random placement in FIXED training grid
            random_offset_range = 0.0  # NO random offset - place exactly at cell center
            total_cells = self.training_grid_size * self.training_grid_size

            # Randomly select which grid cells to fill (no duplicates)
            selected_indices = np.random.choice(total_cells, size=self.num_cubes, replace=False)
            selected_cells = set(selected_indices)

            # Create cylinders using DynamicCylinder
            self.cubes = []  # Still called cubes for compatibility
            cylinder_index = 0

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
                    cell_center_x = start_x + (col * cylinder_spacing) + (cylinder_spacing / 2.0)
                    cell_center_y = start_y + (row * cylinder_spacing) + (cylinder_spacing / 2.0)

                    # Add random offset within cell (±3cm from cell center)
                    random_offset_x = np.random.uniform(-random_offset_range, random_offset_range)
                    random_offset_y = np.random.uniform(-random_offset_range, random_offset_range)

                    # Final cylinder position: cell center + random offset
                    cylinder_x = cell_center_x + random_offset_x
                    cylinder_y = cell_center_y + random_offset_y
                    # Place cylinders on table at Z=0.64
                    # After 1.4x scale, cylinder height will be ~7.21cm, so center at 0.64 + half_height
                    cylinder_scale = np.array([1.4, 1.4, 1.4])
                    scaled_cylinder_height = cylinder_height * cylinder_scale[2]
                    cylinder_z = 0.64 + scaled_cylinder_height/2.0

                    cylinder_number = cylinder_index + 1
                    cylinder_name = f"Cylinder_{cylinder_number}"
                    prim_path = f"/World/Cylinder_{cylinder_number}"
                    timestamp = int(time.time() * 1000) + cylinder_index

                    # Create DynamicCylinder object (has get_world_pose() method for RRT)
                    cylinder = self.world.scene.add(
                        DynamicCylinder(
                            name=f"cylinder_{timestamp}",
                            position=np.array([cylinder_x, cylinder_y, cylinder_z]),
                            prim_path=prim_path,
                            radius=cylinder_radius,
                            height=cylinder_height,
                            color=np.array([0.0, 0.0, 0.0]),  # Dark black
                            scale=cylinder_scale
                        )
                    )

                    # Set display name
                    prim = stage.GetPrimAtPath(prim_path)
                    if prim:
                        display_name = f"Cylinder {cylinder_number} (t{timestamp})"
                        omni.usd.editor.set_display_name(prim, display_name)

                    # Store cylinder object and name (removed "Dark Blue" from display)
                    self.cubes.append((cylinder, cylinder_name))
                    cylinder_index += 1

            # Wait for stage to update before applying shaders
            await omni.kit.app.get_app().next_update_async()

            # Apply emissive shader to all shared materials at /World/Looks/
            # Materials are: visual_material, visual_material_1, Visual_Material_2, Visual_Material_3, etc.
            shader_count = 0

            # Find all material prims under /World/Looks/
            looks_prim = stage.GetPrimAtPath("/World/Looks")
            if looks_prim.IsValid():
                for child_prim in looks_prim.GetChildren():
                    material_name = child_prim.GetName()
                    # Check if it's a visual material (visual_material, visual_material_1, Visual_Material_2, etc.)
                    if "visual" in material_name.lower() and "material" in material_name.lower():
                        # Try both "shader" and "Shader" as child names
                        for shader_name in ["shader", "Shader"]:
                            shader_prim_path = f"/World/Looks/{material_name}/{shader_name}"
                            shader_prim = stage.GetPrimAtPath(shader_prim_path)
                            if shader_prim.IsValid():
                                shader = UsdShade.Shader(shader_prim)
                                # Set emissive color to black: #000000
                                emissive_color = Gf.Vec3f(0.0, 0.0, 0.0)
                                emissive_input = shader.GetInput("emissiveColor")
                                if not emissive_input:
                                    emissive_input = shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
                                emissive_input.Set(emissive_color)
                                # Set metallic
                                metallic_input = shader.GetInput("metallic")
                                if not metallic_input:
                                    metallic_input = shader.CreateInput("metallic", Sdf.ValueTypeNames.Float)
                                metallic_input.Set(0.6)
                                shader_count += 1
                                break

            # Single update after applying shaders
            await omni.kit.app.get_app().next_update_async()

            # Create gripper and manipulator AFTER cylinders but BEFORE world.reset() (like UR10e)

            # Create gripper (wider opening to avoid pushing cylinders)
            # Cylinder is ~7.23cm diameter (1.4x scale), so open to 10cm (5cm per finger) for clearance
            self.gripper = ParallelGripper(
                end_effector_prim_path=self._end_effector_prim_path,
                joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
                joint_opened_positions=np.array([0.05, 0.05]),  # 10cm total opening (5cm per finger)
                joint_closed_positions=np.array([0.0, 0.0]),  # Fully closed for better grip
                action_deltas=np.array([0.01, 0.01])
            )
            print(f"  [OK] Created gripper with end effector: {self._end_effector_prim_path}")

            # Add manipulator at stand height (no fixed joints)
            self.franka = self.world.scene.add(
                SingleManipulator(
                    prim_path=self._franka_prim_path,
                    name=self._franka_name,
                    end_effector_prim_path=self._end_effector_prim_path,
                    gripper=self.gripper,
                    position=np.array([-0.1, 0.0, 0.6]),  # Fixed position (-0.1, 0, 0.6)
                    orientation=np.array([1.0, 0.0, 0.0, 0.0])
                )
            )
            print(f"  [OK] Franka positioned at (-0.1, 0.0, 0.6) (on stand, no fixed joints)")

            # Load grasp configuration for cylinders
            try:
                grasp_file = os.path.join(
                    r"C:\isaacsim\cobotproject\src\grippers",
                    "franka_cylinder_grasp.yaml"
                )
                self.grasp_config = GraspConfig(grasp_file)

                if self.grasp_config.is_loaded():
                    available_grasps = self.grasp_config.get_available_grasps()
                    print(f"[GRASP CONFIG] Loaded {len(available_grasps)} grasps: {available_grasps}")
                    print(f"[GRASP CONFIG] Using default grasp: {self.current_grasp_name}")
                    confidence = self.grasp_config.get_grasp_confidence(self.current_grasp_name)
                    print(f"[GRASP CONFIG] Grasp confidence: {confidence}")
                else:
                    print("[GRASP CONFIG] Warning: Grasp configuration not loaded, using default gripper positions")
            except Exception as e:
                print(f"[GRASP CONFIG] Error loading grasp configuration: {e}")
                self.grasp_config = None

            # Add Obstacle_1 and Obstacle_2 automatically when scene loads
            self._add_initial_obstacles()

            # Initialize physics and reset (batch updates)
            self.world.initialize_physics()
            self.world.reset()

            # Set initial obstacle positions after world reset
            if hasattr(self, '_initial_obstacle_positions'):
                for obstacle_name, position in self._initial_obstacle_positions.items():
                    if obstacle_name in self.obstacles:
                        self.obstacles[obstacle_name].set_world_pose(position=position)

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
            # Isaac Sim UI displays focal length in tenths of mm
            # Set to 1.3mm to display as 13.0 in UI
            self.depth_camera.set_focal_length(1.3)  # 1.3mm (displays as 13.0 in UI)
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



            # Wait for render variables to be ready before attaching annotators
            # This prevents "invalid input resource for renderVar" warnings
            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            # Attach multiple annotators for comprehensive depth data
            # 1. DepthSensorDistance - provides distance measurements
            self.depth_camera.attach_annotator("DepthSensorDistance")

            # 2. DepthSensorPointCloudPosition - provides 3D point cloud positions
            self.depth_camera.attach_annotator("DepthSensorPointCloudPosition")

            # 3. DepthSensorPointCloudColor - provides color for point cloud
            self.depth_camera.attach_annotator("DepthSensorPointCloudColor")





            # Configure robot (no waits needed for parameter setting)
            self.franka.disable_gravity()
            articulation_controller = self.franka.get_articulation_controller()
            # Increased gripper gains (joints 7, 8) for maximum grip force
            # 20x higher gains to prevent any slipping
            kp_gains = np.array([1e15, 1e15, 1e15, 1e15, 1e15, 1e15, 1e15, 2e16, 2e16])  # 20x higher kp for gripper
            kd_gains = np.array([1e13, 1e13, 1e13, 1e13, 1e13, 1e13, 1e13, 2e14, 2e14])  # 20x higher kd for gripper
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

            # Add initial obstacles (Obstacle_1 and Obstacle_2) to RRT planner
            self._add_initial_obstacles_to_rrt()

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

                # Check if cylinder prim still exists and is valid (not picked yet)
                if not cube.prim or not cube.prim.IsValid():
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
                print(f"  Added {added_count} nearby cylinders as obstacles")
            self.cube_obstacles_enabled = True

            return added_count

        except Exception as e:
            print(f"[ERROR] Failed to add cylinders as obstacles: {e}")
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

            # Filter by height (world coordinates) - 59cm to 80cm height (cylinder level on table)
            # Robot base at Z=0.6m, table top at Z=0.75m, cylinders at Z=0.64 + scaled_height/2 (~0.676m)
            # Lidar at Z=0.75m (0.6 + 0.15)
            # Cylinders (1.4x scale): ~0.64-0.71m ✅ Detected (within 0.59-0.80m range)
            # Obstacles: center at table level ✅ Detected (within 0.59-0.80m range)
            valid_points = points_world[(points_world[:, 2] > 0.59) & (points_world[:, 2] < 0.80)]

            # Filter by distance from robot base - STRICT workspace limits
            robot_pos, _ = self.franka.get_world_pose()
            distances_from_robot = np.linalg.norm(valid_points[:, :2] - robot_pos[:2], axis=1)
            valid_points = valid_points[(distances_from_robot > 0.30) & (distances_from_robot < 0.90)]  # 30cm-90cm only

            # Filter out cylinder pickup region (grid center: [0.40, -0.08])
            # Grid extent depends on training_grid_size and cylinder_spacing
            cube_grid_center = np.array([0.40, -0.08])  # Updated to match new grid center
            cube_grid_margin = 0.28  # 28cm margin to cover grid area
            cube_region_mask = ~((np.abs(valid_points[:, 0] - cube_grid_center[0]) < cube_grid_margin) &
                                 (np.abs(valid_points[:, 1] - cube_grid_center[1]) < cube_grid_margin))
            valid_points = valid_points[cube_region_mask]

            # Filter out container/placement region (container at [0.35, 0.5, 0.6])
            if self.container_dimensions is not None:
                container_pos = np.array([0.35, 0.5])  # XY only (Z is 0.6 on table), Y=0.5
                container_margin = 0.08  # 8cm margin
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

                # Verbose logging removed - LIDAR detection works silently in background

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
        # Move Obstacle_1 if enabled (COMMENTED OUT - no automatic movement)
        # if self.obstacle_1_moving:
        #     self._move_obstacle()

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
                # Set weights_only=False since this is our own trusted checkpoint
                checkpoint = torch.load(self.rl_model_path, map_location='cpu', weights_only=False)

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

        # Get all obstacle positions (including LIDAR-detected obstacles)
        obstacle_positions = []
        if hasattr(self, 'obstacles') and self.obstacles:
            for obs_name, obs_obj in self.obstacles.items():
                try:
                    obs_pos, _ = obs_obj.get_world_pose()
                    obstacle_positions.append(obs_pos)
                except Exception:
                    pass  # Skip invalid obstacles

        # Add LIDAR-detected obstacles
        if hasattr(self, 'lidar_detected_obstacles') and self.lidar_detected_obstacles:
            for lidar_obs_name, lidar_obs_obj in self.lidar_detected_obstacles.items():
                try:
                    lidar_obs_pos, _ = lidar_obs_obj.get_world_pose()
                    obstacle_positions.append(lidar_obs_pos)
                except Exception:
                    pass  # Skip invalid obstacles

        for i, (cube, _) in enumerate(self.cubes):
            # Validate cube before getting position
            if not cube.prim or not cube.prim.IsValid():
                # Cube already picked/deleted - mark as picked
                obs[i, 0] = 0.0  # Distance to EE
                obs[i, 1] = 0.0  # Distance to container
                obs[i, 2] = 0.0  # Obstacle proximity
                obs[i, 3] = 0.0  # Not reachable
                obs[i, 4] = 0.0  # No path clearance
                obs[i, 5] = 1.0  # Marked as picked
                continue

            cube_pos, _ = cube.get_world_pose()

            # 1. Distance to EE (1 value)
            dist_to_ee = np.linalg.norm(cube_pos - ee_pos)
            obs[i, 0] = dist_to_ee

            # 2. Distance to container (1 value)
            obs[i, 1] = np.linalg.norm(cube_pos - container_pos)

            # 3. Obstacle proximity score (1 value) - REAL calculation
            # MUST MATCH TRAINING: _calculate_obstacle_score_with_unpicked_cubes()
            # Calculate minimum distance to any obstacle (including unpicked cubes)
            min_obstacle_dist = float('inf')

            # Check distance to static obstacles (2D distance only, matching training)
            if obstacle_positions:
                for obs_pos in obstacle_positions:
                    dist = np.linalg.norm(cube_pos[:2] - obs_pos[:2])  # 2D distance (X, Y only)
                    min_obstacle_dist = min(min_obstacle_dist, dist)

            # Check distance to unpicked cubes (except current cube) - matching training
            for j, (other_cube, _) in enumerate(self.cubes):
                # Skip if same cube or already picked
                if j == i or j in picked_indices:
                    continue

                # Skip if other cube is invalid
                if not other_cube.prim or not other_cube.prim.IsValid():
                    continue

                try:
                    other_pos, _ = other_cube.get_world_pose()
                    dist = np.linalg.norm(cube_pos[:2] - other_pos[:2])  # 2D distance (X, Y only)
                    min_obstacle_dist = min(min_obstacle_dist, dist)
                except Exception:
                    pass

            # Convert distance to score (EXACT MATCH to training formula)
            # 0.0 = far from obstacles (> 30cm) → SAFE
            # 1.0 = very close to obstacles (< 10cm) → DANGEROUS
            if min_obstacle_dist < 0.10:
                obs[i, 2] = 1.0  # Very close
            elif min_obstacle_dist > 0.30:
                obs[i, 2] = 0.0  # Far away (safe)
            else:
                # Linear interpolation between 10cm and 30cm
                obs[i, 2] = 1.0 - (min_obstacle_dist - 0.10) / 0.20

            # 4. Reachability flag (1 value)
            # Franka workspace: ~30cm to 90cm from base
            reachable = 1.0 if (0.3 <= dist_to_ee <= 0.9) else 0.0
            obs[i, 3] = reachable

            # 5. Path clearance score (1 value) - REAL calculation
            # Check if there are other unpicked cubes blocking the path to this cube
            path_clear = 1.0  # Assume clear by default

            # Check collision with other unpicked cubes (treat them as obstacles)
            collision_radius = 0.15  # 15cm radius (gripper + cube size)
            for j, (other_cube, _) in enumerate(self.cubes):
                # Skip if same cube or already picked
                if j == i or j in picked_indices:
                    continue

                # Skip if other cube is invalid
                if not other_cube.prim or not other_cube.prim.IsValid():
                    continue

                try:
                    other_pos, _ = other_cube.get_world_pose()
                    dist_to_other = np.linalg.norm(cube_pos - other_pos)

                    # If another cube is too close, path is blocked
                    if dist_to_other < collision_radius:
                        path_clear = 0.0
                        break
                except Exception:
                    pass

            # Also check proximity to obstacles
            if obstacle_positions and path_clear > 0.0:
                for obs_pos in obstacle_positions:
                    dist_to_obs = np.linalg.norm(cube_pos - obs_pos)
                    # If obstacle is very close (within gripper reach), path is risky
                    if dist_to_obs < 0.20:  # 20cm threshold
                        path_clear = 0.5  # Partial clearance (risky but possible)
                        break

            obs[i, 4] = path_clear

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

            # Start Obstacle_1 automatic movement (COMMENTED OUT - no automatic movement)
            # self.obstacle_1_moving = True

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

                    # Validate cube index is within bounds
                    if cube_idx < 0 or cube_idx >= total_cubes:
                        print(f"\n[{order_idx + 1}/{total_cubes}] ERROR: Invalid cube index {cube_idx} (valid range: 0-{total_cubes-1})")
                        print(f"  This may indicate RL model was trained on different grid size")
                        self.current_cube_index += 1
                        continue

                    cube, cube_name = cubes[cube_idx]
                    cube_number = order_idx + 1
                    print(f"\n[{cube_number}/{total_cubes}] Picking {cube_name} (index {cube_idx})")

                    # Update current cube index (this is the actual cube index in self.cubes array)
                    self.current_cube_index = cube_idx

                    # Call pick and place (retry logic is now INSIDE the function)
                    # Extract number from "Cylinder_6" -> "6" (split by underscore, not space)
                    success, error_msg = await self._pick_and_place_cube(cube, cube_name.split('_')[1])

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
                    import traceback
                    traceback.print_exc()
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
        """Pick and place cylinder using RRT (8 phases: pick with retry, place, return home)"""
        try:
            total_cubes = self.num_cubes  # Actual number of cubes spawned
            # Cylinders are scaled 1.4x uniformly, so actual height is ~7.21cm
            cylinder_base_height = 0.0515
            cylinder_scale_z = 1.4  # Updated to 1.4
            cylinder_height = cylinder_base_height * cylinder_scale_z  # After scaling = 0.0721m
            cylinder_half = cylinder_height / 2.0

            # Safe height for waypoints (above all objects)
            # Robot base now at Z=0.6m (stand height)
            # Cylinders at Z=0.64m, cylinder tops at ~0.71725m (0.64 + 0.07725)
            # Franka max reach is ~0.855m from base, so max reachable Z ≈ 0.6 + 0.855 = 1.455m
            # Using 0.94m safe height gives 0.22275m (22.3cm) clearance above cylinder tops
            safe_height = 0.94  # Safe height within Franka reach

            orientation = euler_angles_to_quats(np.array([np.pi, 0, 0]))
            max_pick_attempts = 3  # Retry up to 3 times for pick failures (gripper didn't grab cube)
            pick_success = False

            # Validate cube object
            if cube is None or not hasattr(cube, 'get_world_pose'):
                raise ValueError(f"Invalid cube object for {cube_name}")

            # Check if cube prim is valid
            if not cube.prim or not cube.prim.IsValid():
                raise ValueError(f"Cube prim is invalid or already deleted for {cube_name}")

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

                # Phase 2: Move directly above target at safe height
                # Cylinders are at ~0.71725m, use safe_height (0.94m) for clearance
                # Add 4mm offset for better gripper alignment
                high_waypoint = np.array([
                    cube_pos_current[0] + 0.003,  # 4mm offset in X
                    cube_pos_current[1] + 0.003,  # 4mm offset in Y
                    safe_height
                ])
                success = await self._move_to_target_rrt(high_waypoint, orientation, skip_factor=4)

                if not success:
                    # RRT failure - skip to next cube immediately (no retries)
                    print(f"  RRT failed to reach above cube - skipping to next cube")
                    self._remove_all_cube_obstacles()
                    await self._reset_to_safe_config()
                    return False, f"RRT failed to reach above {cube_name}"

                # Extra stabilization after reaching high waypoint
                for _ in range(3):  # Reduced from 5
                    await omni.kit.app.get_app().next_update_async()

                # Phase 3: Pick approach (descend to exact cylinder center for grasping)
                cube_pos_realtime, _ = cube.get_world_pose()
                # Add 4mm offset for better gripper alignment
                pick_pos = np.array([
                    cube_pos_realtime[0] + 0.004,  # 4mm offset in X
                    cube_pos_realtime[1] + 0.004,  # 4mm offset in Y
                    cube_pos_realtime[2]
                ])

                # Slow descent for precision
                success = await self._move_to_target_rrt(pick_pos, orientation, skip_factor=2)
                if not success:
                    # RRT failure - skip to next cube immediately (no retries)
                    print(f"  RRT failed pick approach - skipping to next cube")
                    self._remove_all_cube_obstacles()
                    await self._reset_to_safe_config()
                    return False, f"RRT failed pick approach for {cube_name}"

                for _ in range(3):  # Pick stabilization (reduced from 5)
                    await omni.kit.app.get_app().next_update_async()

                # CRITICAL: Check if retreat is possible BEFORE closing gripper
                # This prevents getting stuck in invalid configuration after gripper closes
                # IMPORTANT: Skip Lidar obstacle updates to avoid detecting target cube as obstacle
                retreat_pos = np.array([
                    cube_pos_realtime[0] + 0.004,  # 4mm offset in X
                    cube_pos_realtime[1] + 0.004,  # 4mm offset in Y
                    safe_height
                ])

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
                # Close gripper to grasp cylinder at center
                # Use grasp configuration if available for optimal finger positions
                if self.grasp_config and self.grasp_config.is_loaded():
                    grasp_positions = self.grasp_config.get_gripper_joint_positions(self.current_grasp_name)
                    if grasp_positions:
                        # Use optimized joint positions from grasp file
                        joint_positions = np.array([
                            grasp_positions.get("panda_finger_joint1", 0.037),
                            grasp_positions.get("panda_finger_joint2", 0.037)
                        ])
                        print(f"[GRASP] Using optimized positions: {joint_positions}")
                    else:
                        # Fallback to default closed positions
                        joint_positions = self.gripper.joint_closed_positions
                else:
                    # No grasp config - use default closed positions
                    joint_positions = self.gripper.joint_closed_positions

                articulation_controller.apply_action(ArticulationAction(
                    joint_positions=joint_positions, joint_indices=np.array([7, 8])))
                for _ in range(20):  # Gripper close (increased for stronger grip)
                    await omni.kit.app.get_app().next_update_async()

                # Wait for cube to settle in gripper before retreat
                for _ in range(15):  # Cube settling stabilization (increased to ensure firm grip)
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

            # SAME margins as franka_rrt_2cubes_v1.5
            if total_cubes <= 4:
                edge_margin_left = 0.11
                edge_margin_right = 0.11
                edge_margin_width = 0.10
            elif total_cubes <= 9:
                edge_margin_left = 0.11
                edge_margin_right = 0.11
                edge_margin_width = 0.10
            else:
                edge_margin_left = 0.09
                edge_margin_right = 0.09
                edge_margin_width = 0.07

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

            # Place cylinders in container
            # Container center Z = 0.6m, container height = 0.128m (scaled)
            # Container half-height = 0.064m
            # Container bottom Z = 0.6 - 0.064 = 0.536m
            # Cylinder half height (1.4x scale) = 0.0721/2 = 0.03605m
            # Place cylinder center at: container_bottom + cylinder_half_height + small_offset
            container_center_z = 0.6
            container_half_height = self.container_dimensions[2] / 2.0  # 0.128 / 2 = 0.064m
            container_bottom_z = container_center_z - container_half_height  # 0.536m

            # No clearance offset - place directly on container bottom for full height visibility
            clearance_offset = 0.0
            place_height = container_bottom_z + cylinder_half + clearance_offset
            place_pos = np.array([cube_x, cube_y, place_height])

            # PLACE PHASE - Add NEARBY unpicked cubes as obstacles
            # Only add cubes within collision radius of the place position
            # Reduced radius to minimize RRT failures (matching franka_rrt_9cylinders_v1.5.py)
            collision_radius = 0.20  # 20cm radius (only nearby cubes)
            for i in range(self.current_cube_index + 1, len(self.cubes)):
                other_cube, _ = self.cubes[i]

                # Check if cube prim still exists (not picked yet)
                if not other_cube.prim or not other_cube.prim.IsValid():
                    continue

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

            # Phase 5: Pre-place - Go directly above place position
            # Straight vertical descent without intermediate waypoints
            above_place = np.array([place_pos[0], place_pos[1], safe_height])
            await self._move_to_target_rrt(above_place, orientation, skip_factor=3)  # Slower approach
            await self._move_to_target_rrt(pre_place_pos, orientation, skip_factor=3)  # Slower descent

            # Phase 6: Place approach (slower to prevent cylinder swing/hitting container)
            # Release height: 16cm above place position
            release_height = place_pos + np.array([0.0, 0.0, 0.16])
            await self._move_to_target_rrt(release_height, orientation, skip_factor=3)  # Slower for stability
            for _ in range(3):  # Place stabilization (increased from 1 to prevent cylinder throw)
                await omni.kit.app.get_app().next_update_async()

            articulation_controller = self.franka.get_articulation_controller()
            articulation_controller.apply_action(ArticulationAction(
                joint_positions=self.gripper.joint_opened_positions, joint_indices=np.array([7, 8])))
            for _ in range(12):  # Gripper open (increased from 10 to prevent cylinder throw)
                await omni.kit.app.get_app().next_update_async()

            # PLACE COMPLETE - Remove cube obstacles
            self._remove_all_cube_obstacles()

            # Verify placement
            cube_pos_final, _ = cube.get_world_pose()
            xy_distance = np.linalg.norm(cube_pos_final[:2] - place_pos[:2])
            # Cylinders should be in container at Z > 0.6
            placement_successful = (xy_distance < 0.15) and (cube_pos_final[2] > 0.6)

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
            # Use safe_height for clearance above table and cylinders
            safe_ee_position = np.array([0.40, 0.0, safe_height])  # Centered position at safe height
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
        for _ in range(5):  # 5 frames @ 60Hz = ~83ms stabilization (reduced from 10)
            await omni.kit.app.get_app().next_update_async()

        return True

    async def _reset_to_safe_config(self):
        """Reset robot to a known safe configuration using direct joint control"""
        # Franka has 9 DOF: 7 arm joints + 2 gripper joints
        # Safe config: arm joints + gripper closed positions
        safe_arm_joints = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])
        safe_joint_positions = np.concatenate([safe_arm_joints, self.gripper.joint_closed_positions])

        articulation_controller = self.franka.get_articulation_controller()
        articulation_controller.apply_action(ArticulationAction(joint_positions=safe_joint_positions))
        for _ in range(8):  # 8 frames (reduced from 15)
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

        # Increased max_iterations for better obstacle avoidance
        # More iterations = better path quality around obstacles
        if near_obstacle_1:
            max_iterations = 10000  # More iterations near moving Obstacle_1 (increased from 7000)
        elif has_obstacles:
            max_iterations = 8000   # More iterations with static obstacles (increased from 6000)
        else:
            max_iterations = 5000   # Standard iterations without obstacles (increased from 4000)

        self.rrt.set_max_iterations(max_iterations)

        rrt_plan = self.rrt.compute_path(start_pos, np.array([]))

        if rrt_plan is None or len(rrt_plan) <= 1:
            # Enhanced logging for RRT failures
            ee_pos, _ = self.franka.end_effector.get_world_pose()
            carb.log_warn(f"RRT failed: target={target_position}, current_ee={ee_pos}, obstacles={self.obstacle_counter}, iterations={max_iterations}")
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

    def _add_initial_obstacles(self):
        """Add Obstacle_1 and Obstacle_2 automatically when scene loads"""
        try:
            # Add Obstacle_1 at position [0.55, 0.26, 0.7] (on table surface)
            # Positioned between grid and container for testing obstacle avoidance
            self.obstacle_counter += 1
            obstacle_1_name = f"obstacle_{self.obstacle_counter}"
            obstacle_1_prim_path = f"/World/Obstacle_{self.obstacle_counter}"
            obstacle_1_position = np.array([0.55, 0.26, 0.7])  # Table surface height, Y=0.26
            obstacle_size = np.array([0.20, 0.05, 0.22])  # [length, width, height]

            # Create Obstacle_1
            cube_prim_1 = prim_utils.create_prim(
                prim_path=obstacle_1_prim_path,
                prim_type="Cube",
                position=obstacle_1_position,
                orientation=euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
                scale=obstacle_size,
                attributes={"size": 1.0}
            )

            # Set color (blue)
            stage = omni.usd.get_context().get_stage()
            cube_geom_1 = UsdGeom.Cube.Get(stage, obstacle_1_prim_path)
            if cube_geom_1:
                cube_geom_1.CreateDisplayColorAttr([(0.0, 0.0, 1.0)])

            # Apply physics to Obstacle_1 (KINEMATIC - fixed in place, not affected by gravity)
            if not cube_prim_1.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI.Apply(cube_prim_1)
            rigid_body_api_1 = UsdPhysics.RigidBodyAPI(cube_prim_1)
            rigid_body_api_1.CreateKinematicEnabledAttr().Set(True)  # Kinematic - fixed in place

            if not cube_prim_1.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(cube_prim_1)
            mesh_collision_api_1 = UsdPhysics.MeshCollisionAPI.Apply(cube_prim_1)
            mesh_collision_api_1.GetApproximationAttr().Set("convexHull")

            # Create FixedCuboid wrapper for Obstacle_1
            obstacle_1 = FixedCuboid(
                name=obstacle_1_name,
                prim_path=obstacle_1_prim_path,
                size=1.0,
                scale=obstacle_size
            )
            self.world.scene.add(obstacle_1)
            self.obstacles[obstacle_1_name] = obstacle_1

            # Store position to set after physics initialization
            if not hasattr(self, '_initial_obstacle_positions'):
                self._initial_obstacle_positions = {}
            self._initial_obstacle_positions[obstacle_1_name] = obstacle_1_position



            # Add Obstacle_2 at position [0.55, -0.45, 0.7] (on table surface)
            # Positioned on right side, below grid for testing obstacle avoidance
            self.obstacle_counter += 1
            obstacle_2_name = f"obstacle_{self.obstacle_counter}"
            obstacle_2_prim_path = f"/World/Obstacle_{self.obstacle_counter}"
            obstacle_2_position = np.array([0.55, -0.45, 0.7])  # Table surface height, Y=-0.45
            obstacle_2_size = np.array([0.20, 0.05, 0.22])  # Same as Obstacle_1

            # Create Obstacle_2 with 90-degree Z rotation (Euler angles from UI in degrees)
            cube_prim_2 = prim_utils.create_prim(
                prim_path=obstacle_2_prim_path,
                prim_type="Cube",
                position=obstacle_2_position,
                orientation=euler_angles_to_quats(np.array([0, 0, 90]), degrees=True),
                scale=obstacle_2_size,
                attributes={"size": 1.0}
            )

            # Set color (blue)
            cube_geom_2 = UsdGeom.Cube.Get(stage, obstacle_2_prim_path)
            if cube_geom_2:
                cube_geom_2.CreateDisplayColorAttr([(0.0, 0.0, 1.0)])

            # Apply physics to Obstacle_2 (KINEMATIC - fixed in place, not affected by gravity)
            if not cube_prim_2.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI.Apply(cube_prim_2)
            rigid_body_api_2 = UsdPhysics.RigidBodyAPI(cube_prim_2)
            rigid_body_api_2.CreateKinematicEnabledAttr().Set(True)  # Kinematic - fixed in place

            if not cube_prim_2.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(cube_prim_2)
            mesh_collision_api_2 = UsdPhysics.MeshCollisionAPI.Apply(cube_prim_2)
            mesh_collision_api_2.GetApproximationAttr().Set("convexHull")

            # Create FixedCuboid wrapper for Obstacle_2
            obstacle_2 = FixedCuboid(
                name=obstacle_2_name,
                prim_path=obstacle_2_prim_path,
                size=1.0,
                scale=obstacle_2_size
            )
            self.world.scene.add(obstacle_2)
            self.obstacles[obstacle_2_name] = obstacle_2

            # Store position to set after physics initialization
            self._initial_obstacle_positions[obstacle_2_name] = obstacle_2_position



        except Exception as e:
            print(f"[ERROR] Failed to add initial obstacles: {e}")
            import traceback
            traceback.print_exc()

    def _add_initial_obstacles_to_rrt(self):
        """Add initial obstacles (Obstacle_1 and Obstacle_2) to RRT planner after RRT is initialized"""
        try:
            if not self.rrt:
                print("[ERROR] RRT planner not initialized")
                return

            # Add all obstacles to RRT planner
            for obstacle_name, obstacle in self.obstacles.items():
                # Obstacle_1 is dynamic (will move), Obstacle_2 is static
                is_static = obstacle_name != "obstacle_1"
                self.rrt.add_obstacle(obstacle, static=is_static)

        except Exception as e:
            print(f"[ERROR] Failed to add initial obstacles to RRT: {e}")
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
            # Position 1: [0.55, 0.26, 0.75] (on table surface, between grid and container)
            # Position 2: [0.55, -0.45, 0.75] (on table surface, below grid)
            # Position 3+: Each 8cm away from position 2 at same side

            num_existing_obstacles = len(self.obstacles)

            if num_existing_obstacles == 0:
                # First obstacle - on table surface, between grid and container
                obstacle_position = np.array([0.55, 0.26, 0.7])
                obstacle_orientation = np.array([0, 0, 0])
            elif num_existing_obstacles == 1:
                # Second obstacle - on table surface, below grid (rotated 90 degrees)
                obstacle_position = np.array([0.55, -0.45, 0.7])
                obstacle_orientation = np.array([0, 0, 90])
            else:
                # Third and subsequent obstacles - 8cm away from previous obstacle, on table
                # Each obstacle is 8cm further in Y direction (Y decreases)
                offset = (num_existing_obstacles - 1) * 0.08  # 8cm spacing from position 2
                obstacle_position = np.array([0.55, -0.45 - offset, 0.7])
                obstacle_orientation = np.array([0, 0, 0])

            obstacle_size = np.array([0.20, 0.05, 0.22])  # [length, width, height]

            # Create cube prim using prim_utils (creates Xform with Cube mesh as child)
            # Euler angles are in degrees (matching UI display)
            cube_prim = prim_utils.create_prim(
                prim_path=obstacle_prim_path,
                prim_type="Cube",
                position=obstacle_position,
                orientation=euler_angles_to_quats(obstacle_orientation, degrees=True),
                scale=obstacle_size,
                attributes={"size": 1.0}
            )

            # Set color (blue)
            stage = omni.usd.get_context().get_stage()
            cube_geom = UsdGeom.Cube.Get(stage, obstacle_prim_path)
            if cube_geom:
                cube_geom.CreateDisplayColorAttr([(0.0, 0.0, 1.0)])

            # Apply RigidBodyAPI for physics (KINEMATIC - fixed in place)
            if not cube_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI.Apply(cube_prim)

            rigid_body_api = UsdPhysics.RigidBodyAPI(cube_prim)

            # CRITICAL: Set kinematic to True (fixed in place, not affected by gravity/forces)
            rigid_body_api.CreateKinematicEnabledAttr().Set(True)

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

