"""
Franka MPC Pick and Place - Dynamic Grid with Kinematics Solver
MPC (Model Predictive Control) path planning with MPPI optimization for obstacle avoidance,
dynamic grid configuration, pick retry logic, return to home after each cube.
Uses PhysX Lidar - Rotating and depth sensor for obstacle detection.
"""

import asyncio
import time
import numpy as np
import os
from pathlib import Path
import sys
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
# NOTE: Not using Isaac Sim's IK solvers - using cuRobo-style batched IK instead
# from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from pxr import UsdPhysics, PhysxSchema, Gf, UsdGeom, Sdf
from omni.isaac.franka import Franka  # cuRobo uses this official Franka class
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

# Clear module cache to force reload of MPC modules
import importlib
import importlib.util

# Remove cached MPC modules if they exist
modules_to_clear = [name for name in sys.modules.keys() if 'mpc' in name.lower() or 'gpu_mpc' in name]
for module_name in modules_to_clear:
    del sys.modules[module_name]
    print(f"[CACHE] Cleared cached module: {module_name}")

# Import URDF-based Batched IK using direct file loading
# First, load urdf_kinematics module
urdf_kinematics_path = project_root / "src" / "mpc" / "urdf_kinematics.py"
spec_urdf = importlib.util.spec_from_file_location("urdf_kinematics", urdf_kinematics_path)
urdf_kinematics_module = importlib.util.module_from_spec(spec_urdf)
sys.modules['urdf_kinematics'] = urdf_kinematics_module  # Register in sys.modules
spec_urdf.loader.exec_module(urdf_kinematics_module)

# Then, load batched_ik_urdf module
batched_ik_urdf_path = project_root / "src" / "mpc" / "batched_ik_urdf.py"
spec = importlib.util.spec_from_file_location("batched_ik_urdf", batched_ik_urdf_path)
batched_ik_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batched_ik_module)
BatchedIKSolverURDF = batched_ik_module.BatchedIKSolverURDF
print("[CACHE] URDF-based Batched IK modules loaded fresh (cache cleared)")


class FrankaMPCDynamicGrid:
    """Dynamic grid pick and place with Batched Collision-Free IK (cuRobo-style)"""

    def __init__(self):
        self.window = None
        self.world = None
        self.franka = None
        self.gripper = None
        self.container = None
        self.mpc = None  # MPC planner (replaces RRT)

        # NOTE: Not using Isaac Sim's IK solvers - using cuRobo-style batched IK instead
        # self.kinematics_solver = None
        # self.articulation_kinematics_solver = None

        # Dynamic cube list
        self.cubes = []  # Will store (cube_object, cube_name) tuples

        # Grid parameters
        self.grid_length = 2  # Default: 2 rows
        self.grid_width = 2   # Default: 2 columns

        # Container dimensions (will be calculated after loading)
        self.container_dimensions = None  # [length, width, height] in meters

        # Obstacle management
        self.obstacles = {}  # Dictionary to store obstacles {name: obstacle_object}
        self.obstacle_counter = 0  # Counter for unique obstacle names

        # Obstacle_1 automatic movement with PhysX Force API (acceleration mode)
        self.obstacle_1_moving = False
        self.obstacle_1_acceleration = 6.0  # Acceleration magnitude in m/s²
        self.obstacle_1_min_x = 0.2  # Left boundary (0.2m)
        self.obstacle_1_max_x = 0.63  # Right boundary (0.63m) - 0.43m total travel
        self.obstacle_1_force_api_applied = False  # Track if Force API has been applied

        # PhysX Lidar sensor
        self.lidar = None
        self.lidar_detected_obstacles = {}  # Dictionary to store dynamically detected obstacles
        self.dynamic_obstacles = {}  # Dictionary for MPC obstacle tracking

        # Depth Camera sensor (SingleViewDepthSensor)
        self.depth_camera = None  # Depth camera sensor

        # Object attachment
        self._attached_cube = None  # Currently attached cube (for manual position updates)

        # Timeline
        self.timeline = omni.timeline.get_timeline_interface()

        # Task state
        self.is_picking = False
        self.placed_count = 0
        self.current_cube_index = 0  # Track which cube we're currently working on

        # UI elements
        self.load_btn = None
        self.pick_btn = None
        self.reset_btn = None
        self.add_obstacle_btn = None
        self.remove_obstacle_btn = None
        self.status_label = None
        self.length_field = None
        self.width_field = None

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
                        # Length (rows)
                        with ui.HStack(spacing=10):
                            ui.Label("Grid Length (rows):", width=150)
                            # Create IntField with SimpleIntModel initialized to default value
                            length_model = ui.SimpleIntModel(2)
                            self.length_field = ui.IntField(height=25, model=length_model)

                        # Width (columns)
                        with ui.HStack(spacing=10):
                            ui.Label("Grid Width (columns):", width=150)
                            # Create IntField with SimpleIntModel initialized to default value
                            width_model = ui.SimpleIntModel(2)
                            self.width_field = ui.IntField(height=25, model=width_model)

                        # Info label
                        ui.Label("Total cubes will be: Length × Width",
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
        """Load the scene with Franka, dynamic grid of cubes, and container"""
        try:
            # Get grid parameters from UI
            self.grid_length = int(self.length_field.model.get_value_as_int())
            self.grid_width = int(self.width_field.model.get_value_as_int())

            # Validate grid parameters
            if self.grid_length < 1 or self.grid_width < 1:
                self._update_status("Error: Grid dimensions must be at least 1x1")
                return
            if self.grid_length > 10 or self.grid_width > 10:
                self._update_status("Error: Grid dimensions too large (max 10x10)")
                return

            total_cubes = self.grid_length * self.grid_width
            print(f"Loading {self.grid_length}x{self.grid_width} grid ({total_cubes} cubes)")

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

            # Set Main_Camera scale to (0.3, 0.3, 0.3)
            main_camera_prim = stage.GetPrimAtPath("/World/Main_Camera")
            if main_camera_prim and main_camera_prim.IsValid():
                # Check if scale attribute already exists
                scale_attr = main_camera_prim.GetAttribute("xformOp:scale")
                if scale_attr:
                    # Use existing scale attribute
                    scale_attr.Set(Gf.Vec3d(0.3, 0.3, 0.3))
                else:
                    # Create new scale operation
                    xformable = UsdGeom.Xformable(main_camera_prim)
                    scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
                    scale_op.Set(Gf.Vec3d(0.3, 0.3, 0.3))
                print("[MAIN CAMERA] Set scale to (0.3, 0.3, 0.3)")

            # Single update after world setup
            await omni.kit.app.get_app().next_update_async()

            franka_name = f"franka_{int(time.time() * 1000)}"
            franka_prim_path = f"/World/Franka_{int(time.time() * 1000)}"

            # Use Isaac Sim's official Franka class (same as cuRobo)
            # This ensures USD and URDF kinematics match!
            self.franka = self.world.scene.add(
                Franka(
                    prim_path=franka_prim_path,
                    name=franka_name,
                    end_effector_prim_name="panda_hand"  # cuRobo uses panda_hand
                )
            )

            # Get the gripper from the Franka robot
            self.gripper = self.franka.gripper

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

            # Initialize Depth Camera at World level (detached from robot)
            print("[DEPTH CAMERA] Initializing depth camera at World level...")
            depth_camera_prim_path = "/World/Depth_Camera"

            # World position and orientation
            # Translation: (1.1, 0.6, 0.9)
            # Orientation: (-29, 29, 144) - these values are ALREADY in quaternion format (w, x, y, z)
            position = np.array([1.1, 0.6, 0.9])
            orientation = np.array([-29.0, 29.0, 144.0, 1.0])  # (x, y, z, w) quaternion

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

            # Set camera properties via USD API
            stage = omni.usd.get_context().get_stage()
            camera_prim = stage.GetPrimAtPath(depth_camera_prim_path)
            if camera_prim:
                # Set focal length and aperture for square pixels
                camera_prim.GetAttribute("focalLength").Set(14.9)
                camera_prim.GetAttribute("horizontalAperture").Set(20.955)
                camera_prim.GetAttribute("verticalAperture").Set(20.955)  # Same as horizontal for square
                camera_prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.01, 10000.0))

                # Update transform operations (Translation, Orientation, Scale)
                xformable = UsdGeom.Xformable(camera_prim)

                # Update Translation
                translate_attr = camera_prim.GetAttribute("xformOp:translate")
                if translate_attr:
                    translate_attr.Set(Gf.Vec3d(1.1, 0.6, 0.9))
                else:
                    translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
                    translate_op.Set(Gf.Vec3d(1.1, 0.6, 0.9))

                # Update Orientation (values are already quaternion: x, y, z, w)
                orient_attr = camera_prim.GetAttribute("xformOp:orient")
                # Quaternion values: (-29, 29, 144, 1) in (x, y, z, w) format
                # USD expects (w, x, y, z) format
                quat_xyzw = np.array([-29.0, 29.0, 144.0, 1.0])
                if orient_attr:
                    # Set quaternion directly (w, x, y, z)
                    orient_attr.Set(Gf.Quatd(quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]))
                else:
                    orient_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
                    orient_op.Set(Gf.Quatd(quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]))

                # Update Scale
                scale_attr = camera_prim.GetAttribute("xformOp:scale")
                if scale_attr:
                    scale_attr.Set(Gf.Vec3d(0.3, 0.3, 0.3))
                else:
                    scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
                    scale_op.Set(Gf.Vec3d(0.3, 0.3, 0.3))

            print(f"[DEPTH CAMERA] Depth camera created at {depth_camera_prim_path} (World level)")
            print(f"[DEPTH CAMERA] Position: (1.1, 0.6, 0.9), Orientation: (-29.0, 29.0, 144.0)")
            print(f"[DEPTH CAMERA] Scale: (0.3, 0.3, 0.3), Focal Length: 14.9")

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
            if total_cubes <= 4:
                cube_spacing = 0.18
            elif total_cubes <= 9:
                cube_spacing = 0.15
            else:
                cube_spacing = 0.13

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

            grid_center_x = 0.45
            grid_center_y = -0.10
            grid_extent_x = (self.grid_length - 1) * cube_spacing
            grid_extent_y = (self.grid_width - 1) * cube_spacing
            start_x = grid_center_x - (grid_extent_x / 2.0)
            start_y = grid_center_y - (grid_extent_y / 2.0)

            cube_index = 0
            for row in range(self.grid_length):
                for col in range(self.grid_width):
                    cube_x = start_x + (row * cube_spacing)
                    cube_y = start_y + (col * cube_spacing)
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
            await omni.kit.app.get_app().next_update_async()
            self.world.reset()
            await omni.kit.app.get_app().next_update_async()

            # Set physics solver parameters (like cuRobo does)
            # Must be done AFTER world.reset() when physics context is initialized
            print("[PHYSICS] Configuring physics solver settings (like cuRobo)...")
            self.franka.set_solver_velocity_iteration_count(4)
            self.franka.set_solver_position_iteration_count(124)

            # Access physics context via USD API directly (more reliable)
            from pxr import PhysxSchema
            stage = omni.usd.get_context().get_stage()
            physics_scene_path = "/physicsScene"
            physics_scene_prim = stage.GetPrimAtPath(physics_scene_path)

            if physics_scene_prim and physics_scene_prim.IsValid():
                physx_scene_api = PhysxSchema.PhysxSceneAPI.Get(stage, physics_scene_path)
                if not physx_scene_api:
                    physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)

                # Set solver type to TGS
                physx_scene_api.CreateSolverTypeAttr("TGS")
                print(f"[PHYSICS] Solver type: {physx_scene_api.GetSolverTypeAttr().Get()}")
                print(f"[PHYSICS] Velocity iterations: 4, Position iterations: 124")
            else:
                print("[PHYSICS] Warning: Could not find physics scene to set solver type")

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

            print("[DEPTH CAMERA] Rendering settings configured (DLSS/DLAA disabled)")

            # Initialize Depth Camera after world reset
            # NOTE: Initialize with attach_rgb_annotator=False for better performance
            self.depth_camera.initialize(attach_rgb_annotator=False)

            # Attach multiple annotators for comprehensive depth data
            # 1. DepthSensorDistance - provides distance measurements
            self.depth_camera.attach_annotator("DepthSensorDistance")

            # 2. DepthSensorPointCloudPosition - provides 3D point cloud positions
            self.depth_camera.attach_annotator("DepthSensorPointCloudPosition")

            # 3. DepthSensorPointCloudColor - provides color for point cloud
            self.depth_camera.attach_annotator("DepthSensorPointCloudColor")

            # Configure depth sensor parameters
            self.depth_camera.set_enabled(enabled=True)
            self.depth_camera.set_min_distance(0.1)  # Minimum distance: 10cm
            self.depth_camera.set_max_distance(2.0)  # Maximum distance: 2m
            self.depth_camera.set_baseline_mm(55.0)  # Baseline: 55mm (standard stereo)
            self.depth_camera.set_focal_length_pixel(256.0)  # Focal length in pixels (512/2)
            self.depth_camera.set_confidence_threshold(0.95)  # High confidence threshold

            print("[DEPTH CAMERA] Depth camera initialized at World level")
            print("[DEPTH CAMERA] Attached annotators:")
            print("  - DepthSensorDistance (distance measurements)")
            print("  - DepthSensorPointCloudPosition (3D point cloud)")
            print("  - DepthSensorPointCloudColor (point cloud colors)")
            print("[DEPTH CAMERA] Position: (1.1, 0.6, 0.9), Orientation: (-29.0, 29.0, 144.0)")
            print("[DEPTH CAMERA] Scale: (0.3, 0.3, 0.3), Focal Length: 14.9")
            print("[DEPTH CAMERA] Resolution: 512x512 (square), Frequency: 10 Hz")
            print("[DEPTH CAMERA] Depth range: 0.1m - 2.0m, Baseline: 55mm")

            # Configure robot (no waits needed for parameter setting)
            self.franka.disable_gravity()

            # CRITICAL FIX: Use proper gains for accurate position tracking
            # Previous gains (kp=1e15, kd=1e13) were WAY too high, causing instability
            # Recommended gains from Isaac Sim examples:
            # - test_articulation_determinism.py: kp=1e4, kd=1e3
            # - franka_helpers.py: stiffness=400-2000, damping=40-50
            articulation_controller = self.franka.get_articulation_controller()
            kp_gains = 1e4 * np.ones(9)  # Reduced from 1e15 to 1e4
            kd_gains = 1e3 * np.ones(9)  # Reduced from 1e13 to 1e3
            articulation_controller.set_gains(kp_gains, kd_gains)

            # Increase solver iterations for better convergence (like cuRobo)
            self.franka.set_solver_position_iteration_count(64)  # Increased from default
            self.franka.set_solver_velocity_iteration_count(64)  # Increased from default

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

            self._setup_mpc()

            # DO NOT add cubes as obstacles - they are targets to pick, not obstacles to avoid
            # Only manual obstacles will be added to MPC for collision avoidance

            print("Scene loaded successfully!")

            # Enable buttons
            self.pick_btn.enabled = True
            self.reset_btn.enabled = True
            self.add_obstacle_btn.enabled = True
            self.remove_obstacle_btn.enabled = True
            self._update_status("Scene loaded - Ready to pick and place")

        except Exception as e:
            self._update_status(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def _setup_mpc(self):
        """Setup MPC motion planner and kinematics solvers"""
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_id = ext_manager.get_enabled_extension_id("isaacsim.examples.interactive")
        examples_extension_path = ext_manager.get_extension_path(ext_id)

        robot_description_file = os.path.join(
            examples_extension_path, "isaacsim", "examples", "interactive", "path_planning",
            "path_planning_example_assets", "franka_conservative_spheres_robot_description.yaml")
        # NOTE: Not using Isaac Sim's Lula IK solver - using cuRobo-style batched IK instead
        # urdf_path = os.path.join(mg_extension_path, "motion_policy_configs", "franka", "lula_franka_gen.urdf")
        # self.kinematics_solver = LulaKinematicsSolver(robot_description_path=robot_description_file, urdf_path=urdf_path)
        # self.articulation_kinematics_solver = ArticulationKinematicsSolver(
        #     self.franka, self.kinematics_solver, "right_gripper")

        # Initialize GPU-accelerated URDF-based Batched IK solver (cuRobo-style from scratch)
        # Load YML configuration (like cuRobo does) - contains URDF path + collision spheres
        yml_path = str(project_root / "assets" / "franka_curobo.yml")

        self.mpc = BatchedIKSolverURDF(
            yml_path=yml_path,  # Load from YML (cuRobo-style)
            num_seeds=50,  # 50 random seeds optimized in parallel on GPU (reduced for speed)
            position_threshold=0.01,  # 1cm position accuracy (practical for pick-and-place)
            rotation_threshold=0.01,  # Rotation threshold (linear metric: 1 - dot_product)
            mppi_iterations=10,  # MPPI exploration iterations (reduced for speed)
            lbfgs_iterations=20,  # L-BFGS refinement iterations (reduced for speed)
            device="cuda:0"  # Use GPU for acceleration
        )

        print("[BatchedIK] GPU-accelerated Batched IK solver initialized (cuRobo-style)")
        print(f"[BatchedIK] Loaded configuration from: {yml_path}")

    def _print_camera_distances(self):
        """
        Print actual depth data from Depth_Camera showing what it sees.
        Only shows cubes and obstacles visible to the camera.
        """
        if self.depth_camera is None:
            print("[CAMERA DISTANCES] Depth camera not initialized")
            return

        try:
            # Get camera position
            camera_pos, _ = self.depth_camera.get_world_pose()

            print("\n" + "="*80)
            print("[DEPTH CAMERA] Actual depth data from camera:")
            print(f"[DEPTH CAMERA] Camera position: ({camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f})")
            print("="*80)

            # Get depth sensor data
            depth_data = self.depth_camera.get_current_frame()

            if depth_data is None:
                print("[DEPTH CAMERA] No depth data available yet (get_current_frame returned None).")
                print("[DEPTH CAMERA] Camera may need to warm up or annotators not initialized.")
                print("="*80 + "\n")
                return

            # Debug: Print available keys
            print(f"[DEPTH CAMERA DEBUG] Available data keys: {list(depth_data.keys())}")

            # The correct key is 'DepthSensorDistance', not 'distance_to_camera'
            if "DepthSensorDistance" not in depth_data:
                print("[DEPTH CAMERA] 'DepthSensorDistance' key not found in depth data.")
                print("[DEPTH CAMERA] This annotator may not be attached or enabled.")
                print("="*80 + "\n")
                return

            # Get distance data (2D array of distances in meters)
            distance_array = depth_data["DepthSensorDistance"]

            # Get point cloud data if available (correct key is 'DepthSensorPointCloudPosition')
            point_cloud_data = depth_data.get("DepthSensorPointCloudPosition", None)

            # Statistics about what the camera sees
            valid_depths = distance_array[distance_array > 0]  # Filter out invalid/infinite depths

            if len(valid_depths) == 0:
                print("[DEPTH CAMERA] No valid depth measurements detected")
                print("="*80 + "\n")
                return

            print(f"\n[DEPTH STATISTICS]")
            print(f"  Valid depth pixels: {len(valid_depths):,} / {distance_array.size:,}")
            print(f"  Min distance: {np.min(valid_depths):.3f}m")
            print(f"  Max distance: {np.max(valid_depths):.3f}m")
            print(f"  Mean distance: {np.mean(valid_depths):.3f}m")
            print(f"  Median distance: {np.median(valid_depths):.3f}m")

            # Detect objects in depth image by clustering similar depths
            # Group depths into bins to identify distinct objects
            print(f"\n[DETECTED OBJECTS IN CAMERA VIEW]")

            # Create depth histogram to find object clusters
            depth_bins = np.linspace(np.min(valid_depths), np.max(valid_depths), 20)
            hist, bin_edges = np.histogram(valid_depths, bins=depth_bins)

            # Find significant peaks in histogram (potential objects)
            threshold = len(valid_depths) * 0.01  # Objects must occupy at least 1% of valid pixels
            significant_bins = np.where(hist > threshold)[0]

            if len(significant_bins) > 0:
                for i, bin_idx in enumerate(significant_bins, 1):
                    bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
                    pixel_count = hist[bin_idx]
                    percentage = (pixel_count / len(valid_depths)) * 100
                    print(f"  Object {i}: Distance ~{bin_center:.3f}m ({pixel_count:,} pixels, {percentage:.1f}%)")
            else:
                print("  No distinct objects detected")

            # Try to match detected depths with known cubes and obstacles
            print(f"\n[MATCHING WITH KNOWN OBJECTS]")

            # Check cubes
            if self.cubes:
                print("\n  Cubes:")
                for cube, cube_name in self.cubes:
                    cube_pos, _ = cube.get_world_pose()
                    expected_distance = np.linalg.norm(cube_pos - camera_pos)

                    # Check if this distance appears in the depth data
                    # Allow 10cm tolerance
                    tolerance = 0.1
                    matching_pixels = np.sum((valid_depths >= expected_distance - tolerance) &
                                            (valid_depths <= expected_distance + tolerance))

                    if matching_pixels > 100:  # At least 100 pixels
                        visibility = "VISIBLE"
                        percentage = (matching_pixels / len(valid_depths)) * 100
                        print(f"    {cube_name}: {expected_distance:.3f}m [{visibility}] ({matching_pixels} pixels, {percentage:.1f}%)")
                    else:
                        print(f"    {cube_name}: {expected_distance:.3f}m [NOT VISIBLE or OCCLUDED]")

            # Check obstacles
            if self.dynamic_obstacles:
                print("\n  Obstacles:")
                for obs_name, obs_data in self.dynamic_obstacles.items():
                    obs_pos = obs_data['position']
                    expected_distance = np.linalg.norm(obs_pos - camera_pos)

                    # Check if this distance appears in the depth data
                    tolerance = 0.1
                    matching_pixels = np.sum((valid_depths >= expected_distance - tolerance) &
                                            (valid_depths <= expected_distance + tolerance))

                    if matching_pixels > 100:
                        visibility = "VISIBLE"
                        percentage = (matching_pixels / len(valid_depths)) * 100
                        print(f"    {obs_name}: {expected_distance:.3f}m [{visibility}] ({matching_pixels} pixels, {percentage:.1f}%)")
                    else:
                        print(f"    {obs_name}: {expected_distance:.3f}m [NOT VISIBLE or OCCLUDED]")

            print("="*80 + "\n")

        except Exception as e:
            print(f"[DEPTH CAMERA] Error processing depth data: {e}")
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
            # Debug: Print once to confirm this method is being called
            if not hasattr(self, '_lidar_debug_printed'):
                print("[LIDAR DEBUG] _process_lidar_data() is being called")
                self._lidar_debug_printed = True

            # Get current frame data from PhysX Lidar
            lidar_data = self.lidar.get_current_frame()

            # Debug: Check what data is returned
            if not hasattr(self, '_lidar_data_debug_printed'):
                if lidar_data is None:
                    print("[LIDAR DEBUG] get_current_frame() returned None")
                else:
                    print(f"[LIDAR DEBUG] get_current_frame() returned data with keys: {list(lidar_data.keys())}")
                self._lidar_data_debug_printed = True

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

            if len(points) == 0:
                return []

            # Validate and fix shape
            if points.ndim == 1:
                if len(points) == 3:
                    return []  # Single point - skip
                if len(points) % 3 == 0:
                    points = points.reshape(-1, 3)
                else:
                    return []
            elif points.ndim != 2 or points.shape[1] != 3:
                return []

            # Transform points from sensor-local to world coordinates
            lidar_world_pos, lidar_world_rot = self.lidar.get_world_pose()
            from scipy.spatial.transform import Rotation as R
            rot_matrix = R.from_quat([lidar_world_rot[1], lidar_world_rot[2], lidar_world_rot[3], lidar_world_rot[0]]).as_matrix()

            # Transform all points to world coordinates
            points_world = (rot_matrix @ points.T).T + lidar_world_pos

            # Filter by height (world coordinates) - only 5cm to 40cm height
            valid_points = points_world[(points_world[:, 2] > 0.05) & (points_world[:, 2] < 0.40)]

            # Filter by distance from robot base - STRICT workspace limits
            robot_pos, _ = self.franka.get_world_pose()
            distances_from_robot = np.linalg.norm(valid_points[:, :2] - robot_pos[:2], axis=1)
            valid_points = valid_points[(distances_from_robot > 0.30) & (distances_from_robot < 0.90)]  # 30cm-90cm only

            # Filter out cube pickup region (tighter bounds to avoid blocking obstacles)
            cube_grid_center = np.array([0.45, -0.10])
            cube_grid_margin = 0.30  # Reduced from 0.35 to 0.30 for tighter filtering
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

            # Print all available keys in the frame
            print(f"[DEPTH CAMERA] Frame keys: {list(depth_frame.keys())}")

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
                print("[DEPTH CAMERA] No depth data in frame")
                return []

            # Print depth statistics
            valid_depths = depth_data[depth_data > 0]
            if len(valid_depths) > 0:
                print(f"[DEPTH CAMERA] Depth range: min={valid_depths.min():.3f}m, max={valid_depths.max():.3f}m, mean={valid_depths.mean():.3f}m")
                print(f"[DEPTH CAMERA] Valid depth pixels: {len(valid_depths)} / {depth_data.size}")

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

                for i, obs_pos in enumerate(detected_obstacles):
                    print(f"[DEPTH CAMERA] Obstacle #{i+1}:")
                    print(f"[DEPTH CAMERA]   Position: ({obs_pos[0]:.3f}, {obs_pos[1]:.3f}, {obs_pos[2]:.3f})m")
                    print(f"[DEPTH CAMERA]   Distance from camera: {np.linalg.norm(np.array(obs_pos) - camera_pos):.3f}m")

                print(f"[DEPTH CAMERA] ----------------------------------------\n")

            return detected_obstacles

        except Exception as e:
            carb.log_warn(f"[DEPTH CAMERA ERROR] Error processing depth camera data: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _update_dynamic_obstacles(self):
        """
        Update MPC planner with dynamically detected obstacles from Lidar.
        Also moves Obstacle_1 automatically.

        For MPC: We only need obstacle positions (not prims) since MPC uses
        cost-based obstacle avoidance rather than collision checking.
        """
        if self.lidar is None or self.mpc is None:
            return

        try:
            # Move Obstacle_1 automatically (if enabled)
            if self.obstacle_1_moving:
                self._move_obstacle()

            # Get detected obstacles from Lidar
            detected_positions = self._process_lidar_data()

            # Limit to 10 obstacles for performance
            detected_positions = detected_positions[:10]

            # Convert to obstacle dictionaries for Batched IK
            obstacle_list = []
            for pos in detected_positions:
                obstacle_list.append({
                    'position': np.array(pos),
                    'radius': 0.1  # Default radius for detected obstacles
                })

            # Update Batched IK solver with obstacle list (only if obstacles exist)
            if len(obstacle_list) > 0:
                self.mpc.set_obstacles(obstacle_list)

            # Store for visualization/debugging and for _plan_to_target
            self.dynamic_obstacles = {}
            for i, obs in enumerate(obstacle_list):
                self.dynamic_obstacles[f"lidar_obstacle_{i}"] = obs

            self.lidar_detected_obstacles = {
                f"lidar_obstacle_{i}": obs['position']
                for i, obs in enumerate(obstacle_list)
            }

        except Exception as e:
            carb.log_warn(f"[BatchedIK ERROR] Error updating dynamic obstacles: {e}")
            import traceback
            traceback.print_exc()

    def _physics_step_callback(self, step_size):
        """
        Physics step callback for continuous sensor updates and obstacle movement.
        Called every physics step (60 Hz).
        """
        # Debug: Print callback status once
        if not hasattr(self, '_callback_debug_printed'):
            print("[PHYSICS] Physics step callback is running!")
            self._callback_debug_printed = True

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

        except Exception as e:
            carb.log_warn(f"[OBSTACLE] Error moving Obstacle_1: {e}")
            import traceback
            traceback.print_exc()

    def _on_pick(self):
        """Pick and place button callback"""
        if not self.world or not self.mpc:
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

            # Print depth camera data when starting pick and place
            if self.current_cube_index == 0:
                # Wait longer for depth camera to warm up (30 frames = 0.5 seconds at 60 FPS)
                for _ in range(30):
                    await omni.kit.app.get_app().next_update_async()

                # Print depth camera distances
                self._print_camera_distances()

            # Add physics callback now that timeline is playing (physics context is ready)
            if not hasattr(self, '_physics_callback_added'):
                try:
                    self.world.add_physics_callback("sensor_and_obstacle_update", self._physics_step_callback)
                    self._physics_callback_added = True
                    print("[PHYSICS] Physics step callback added successfully")
                except Exception as e:
                    print(f"[PHYSICS] Warning: Could not add physics callback: {e}")
                    self._physics_callback_added = False

            # Start Obstacle_1 automatic movement (only if obstacle exists)
            if "obstacle_1" in self.obstacles:
                self.obstacle_1_moving = True
                print("[OBSTACLE] Obstacle_1 automatic movement started")
            else:
                self.obstacle_1_moving = False

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
            # This ensures MPC starts from a known, stable configuration
            default_joint_positions = np.array([0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75, 0.0, 0.0])
            articulation_controller.apply_action(ArticulationAction(joint_positions=default_joint_positions))
            for _ in range(8):  # Reduced for better FPS
                await omni.kit.app.get_app().next_update_async()

            cubes = self.cubes
            total_cubes = len(cubes)

            for i in range(self.current_cube_index, total_cubes):
                try:
                    cube, cube_name = cubes[i]
                    cube_number = i + 1
                    print(f"[{cube_number}/{total_cubes}] {cube_name}")

                    # Update current cube index
                    self.current_cube_index = i

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

            # Stop Obstacle_1 automatic movement (only if obstacle exists)
            if "obstacle_1" in self.obstacles:
                self.obstacle_1_moving = False
                print("[OBSTACLE] Obstacle_1 automatic movement stopped")
            else:
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
        """Pick and place cube using MPC (8 phases: pick with retry, place, return home)"""
        try:
            total_cubes = self.grid_length * self.grid_width
            cube_size = 0.0515
            cube_half = cube_size / 2.0
            orientation = euler_angles_to_quats(np.array([np.pi, 0, 0]))
            max_pick_attempts = 3
            pick_success = False

            for pick_attempt in range(1, max_pick_attempts + 1):
                if pick_attempt > 1:
                    print(f"  Retry {pick_attempt}/{max_pick_attempts}")
                # Don't reset to home on retry - just try again from current position
                # This avoids unnecessary double movements

                # Cubes are NOT obstacles - no need to disable/enable
                cube_pos_current, _ = cube.get_world_pose()

                # Use same pre-pick height as v1.5 (works well with proper gains)
                pre_pick_height = 0.10 if total_cubes <= 4 else (0.12 if total_cubes <= 9 else 0.15)
                pre_pick_pos = cube_pos_current + np.array([0.0, 0.0, pre_pick_height])

                # Phase 1: Pre-pick (open gripper BEFORE approaching) - FASTER with skip_factor=4
                articulation_controller = self.franka.get_articulation_controller()
                articulation_controller.apply_action(ArticulationAction(
                    joint_positions=self.gripper.joint_opened_positions, joint_indices=np.array([7, 8])))
                for _ in range(2):  # Gripper open (reduced from 3)
                    await omni.kit.app.get_app().next_update_async()

                # NOTE: Removed Lula IK reachability check - batched IK will handle this
                # The batched IK solver will return None if position is unreachable

                success = await self._move_to_target_rrt(pre_pick_pos, orientation, skip_factor=4)
                if not success:
                    if pick_attempt < max_pick_attempts:
                        continue
                    return False, f"Failed pre-pick for {cube_name}"

                # Phase 2: Pick approach (gripper already open)
                cube_pos_realtime, _ = cube.get_world_pose()

                # CRITICAL FIX: Go directly to cube position (like v1.5)!
                # The gripper fingers close AROUND the cube at the cube's height
                # NOT above it! The 0.092m offset is from EE to gripper fingers,
                # but we command the EE position such that fingers reach the cube.
                # v1.5 goes directly to cube_pos, and it works!
                pick_pos = np.array([
                    cube_pos_realtime[0],
                    cube_pos_realtime[1],
                    cube_pos_realtime[2]  # Go directly to cube center height
                ])

                print(f"  [DEBUG] Cube position: {cube_pos_realtime}")
                print(f"  [DEBUG] Pick target (at cube height): {pick_pos}")

                # Moderate speed approach with skip_factor=3 for good balance
                success = await self._move_to_target_rrt(pick_pos, orientation, skip_factor=3)
                if not success:
                    if pick_attempt < max_pick_attempts:
                        continue
                    return False, f"Failed pick approach for {cube_name}"

                # Check actual end-effector position after movement
                actual_ee_pos, actual_ee_ori = self.franka.end_effector.get_world_pose()
                ee_error = np.linalg.norm(actual_ee_pos - pick_pos)

                # DEBUG: Check actual joint positions
                actual_joints = self.franka.get_joint_positions()[:7]  # First 7 joints only
                print(f"  [DEBUG] Actual joints after execution: {actual_joints}")
                print(f"  [DEBUG] EE position: {actual_ee_pos}")
                print(f"  [DEBUG] Target position: {pick_pos}")
                print(f"  [DEBUG] EE error: {ee_error*100:.2f}cm")

                for _ in range(10):  # Pick stabilization (increased for better alignment)
                    await omni.kit.app.get_app().next_update_async()

                articulation_controller.apply_action(ArticulationAction(
                    joint_positions=self.gripper.joint_closed_positions, joint_indices=np.array([7, 8])))
                for _ in range(20):  # Gripper close (increased for better grip)
                    await omni.kit.app.get_app().next_update_async()

                # Attach cube to gripper (like cuRobo does)
                self._attach_cube_to_gripper(cube)

                # Wait a few frames for attachment to take effect
                for _ in range(5):
                    await omni.kit.app.get_app().next_update_async()

                # Phase 3: Pick retreat (faster with skip_factor=5)
                current_ee_pos, _ = self.franka.end_effector.get_world_pose()
                retreat_height = 0.15 if total_cubes <= 4 else (0.18 if total_cubes <= 9 else 0.20)
                retreat_pos = current_ee_pos + np.array([0.0, 0.0, retreat_height])

                success = await self._move_to_target_rrt(retreat_pos, orientation, skip_factor=5)
                if not success:
                    self.franka.gripper.open()
                    for _ in range(5):
                        await omni.kit.app.get_app().next_update_async()
                    if pick_attempt < max_pick_attempts:
                        continue
                    return False, f"Failed pick retreat for {cube_name}"

                # Phase 4: Verify pick
                cube_pos_after_pick, _ = cube.get_world_pose()
                height_lifted = cube_pos_after_pick[2] - cube_pos_realtime[2]

                print(f"  [DEBUG] Cube before pick: Z={cube_pos_realtime[2]:.4f}m")
                print(f"  [DEBUG] Cube after pick: Z={cube_pos_after_pick[2]:.4f}m")
                print(f"  [DEBUG] Height lifted: {height_lifted*100:.2f}cm (need >5cm)")

                if height_lifted > 0.05:
                    print(f"  Pick OK ({height_lifted*100:.1f}cm)")
                    pick_success = True
                    break
                else:
                    print(f"  Pick fail ({height_lifted*100:.1f}cm)")
                    if pick_attempt < max_pick_attempts:
                        print(f"  Retry {pick_attempt+1}/{max_pick_attempts}")

                        # Open gripper
                        articulation_controller.apply_action(ArticulationAction(
                            joint_positions=self.gripper.joint_opened_positions, joint_indices=np.array([7, 8])))
                        for _ in range(10):
                            await omni.kit.app.get_app().next_update_async()

                        # Move to safe height (20cm above current position)
                        current_ee_pos, _ = self.franka.end_effector.get_world_pose()
                        safe_height_pos = current_ee_pos + np.array([0.0, 0.0, 0.20])
                        print(f"  Moving to safe height: {safe_height_pos}")
                        await self._move_to_target_rrt(safe_height_pos, orientation, skip_factor=5)

                        # Move back to pre-pick position before retrying
                        cube_pos_retry, _ = cube.get_world_pose()
                        pre_pick_pos_retry = cube_pos_retry + np.array([0.0, 0.0, pre_pick_height])
                        print(f"  Moving to pre-pick position: {pre_pick_pos_retry}")
                        await self._move_to_target_rrt(pre_pick_pos_retry, orientation, skip_factor=4)

                        continue
                    else:
                        return False, f"Failed to pick {cube_name}"

            # If we get here and pick didn't succeed, return failure
            if not pick_success:
                return False, f"Failed to pick {cube_name}"

            # PLACE PHASE - Cube remains DISABLED (not an obstacle during place)
            # Conservative collision spheres + higher waypoints provide clearance for held cube

            container_center = np.array([0.30, 0.50, 0.0])
            # Container dimensions: [0.48m (X-length), 0.36m (Y-width), 0.128m (Z-height)]
            container_length = self.container_dimensions[0]  # X-axis: 0.48m
            container_width = self.container_dimensions[1]   # Y-axis: 0.36m

            # Use same grid dimensions as pickup grid (like working version)
            # This ensures proper spacing and placement
            place_row = self.placed_count // self.grid_width
            place_col = self.placed_count % self.grid_width

            # IMPROVED: Asymmetric margins for container (from working version)
            # Larger left margin (start side) to prevent gripper collision on cube 1
            # Smaller right margin (end side) to maximize usable space
            if total_cubes <= 4:
                edge_margin_left = 0.11   # Prevents gripper collision on cube 1
                edge_margin_right = 0.09  # Smaller to maximize space
                edge_margin_width = 0.09
            elif total_cubes <= 9:
                edge_margin_left = 0.11   # Prevents gripper collision on cube 1
                edge_margin_right = 0.09  # Smaller to maximize space
                edge_margin_width = 0.09
            else:
                edge_margin_left = 0.11   # Prevents gripper collision on cube 1
                edge_margin_right = 0.09  # Smaller to maximize space
                edge_margin_width = 0.09

            # Calculate usable space with asymmetric margins
            usable_length = container_length - edge_margin_left - edge_margin_right
            usable_width = container_width - (2 * edge_margin_width)
            spacing_length = usable_length / (self.grid_length - 1) if self.grid_length > 1 else 0.0
            spacing_width = usable_width / (self.grid_width - 1) if self.grid_width > 1 else 0.0

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

            # Use higher waypoints to provide clearance for held cube (cube size ~5cm)
            # Reduced height to avoid IK failures for far positions
            pre_place_pos = place_pos + np.array([0.0, 0.0, 0.25])  # Reduced from 0.35 to 0.25

            # Phase 5: Pre-place (faster with skip_factor=5)
            # Add via point to ensure consistent approach direction (prevents base rotation)
            via_point = np.array([0.35, 0.25, 0.40])  # Reduced from 0.45 to 0.40
            await self._move_to_target_rrt(via_point, orientation, skip_factor=5)
            await self._move_to_target_rrt(pre_place_pos, orientation, skip_factor=5)

            # Phase 6: Place approach (moderate speed with skip_factor=4)
            release_height = place_pos + np.array([0.0, 0.0, 0.08])
            await self._move_to_target_rrt(release_height, orientation, skip_factor=4)
            for _ in range(3):  # Place stabilization (increased from 1 to prevent cube throw)
                await omni.kit.app.get_app().next_update_async()

            # Detach cube from gripper before opening (like cuRobo does)
            self._detach_cube_from_gripper(cube)

            # Wait a few frames for detachment to take effect
            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            articulation_controller = self.franka.get_articulation_controller()
            articulation_controller.apply_action(ArticulationAction(
                joint_positions=self.gripper.joint_opened_positions, joint_indices=np.array([7, 8])))
            for _ in range(12):  # Gripper open (increased from 10 to prevent cube throw)
                await omni.kit.app.get_app().next_update_async()

            # Cubes are NOT obstacles - no need to re-enable

            # Verify placement
            cube_pos_final, _ = cube.get_world_pose()
            xy_distance = np.linalg.norm(cube_pos_final[:2] - place_pos[:2])
            placement_successful = (xy_distance < 0.15) and (cube_pos_final[2] < 0.15)

            if placement_successful:
                print(f"  Place OK ({xy_distance*100:.1f}cm)")
            else:
                print(f"  Place fail ({xy_distance*100:.1f}cm)")

            # Phase 7: Place retreat (faster with skip_factor=6)
            # CRITICAL FIX: Account for container height when retreating!
            # Container height = 0.128m (from container_dimensions[2])
            # Need to retreat above container walls to avoid collision
            container_height = self.container_dimensions[2]  # 0.128m
            retreat_height = container_height + 0.20  # 20cm above container top
            retreat_pos = np.array([cube_x, cube_y, retreat_height])
            retreat_success = await self._move_to_target_rrt(retreat_pos, orientation, skip_factor=6)

            articulation_controller.apply_action(ArticulationAction(
                joint_positions=self.gripper.joint_closed_positions, joint_indices=np.array([7, 8])))
            for _ in range(2):  # Gripper close
                await omni.kit.app.get_app().next_update_async()

            # Move to a safe intermediate position using MPC to avoid obstacles
            # Reduced height from 0.50 to 0.35 (15cm lower)
            safe_ee_position = np.array([0.40, 0.0, 0.35])  # Centered position, lower height
            safe_success = await self._move_to_target_rrt(safe_ee_position, orientation, skip_factor=5)

            if not safe_success:
                # If MPC fails to reach safe position, use direct reset as last resort
                # This may hit obstacles but ensures robot doesn't stay in invalid config
                await self._reset_to_safe_config()

            return (True, "") if placement_successful else (False, f"{cube_name} not in container")

        except Exception as e:
            error_msg = f"Error picking/placing Cube {cube_name}: {str(e)}"
            import traceback
            traceback.print_exc()
            return False, error_msg  # Failed

    async def _move_to_target_ik(self, target_position, target_orientation, num_steps=8):
        """
        DEPRECATED: This function used Lula IK solver which has been removed.
        Use _move_to_target_rrt() with batched IK instead.
        """
        print("[WARNING] _move_to_target_ik is deprecated - use _move_to_target_rrt instead")
        return False

        # NOTE: Old Lula IK code commented out - not used anymore
        # # Update robot base pose
        # robot_base_translation, robot_base_orientation = self.franka.get_world_pose()
        # self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        #
        # # Compute IK solution
        # ik_action, ik_success = self.articulation_kinematics_solver.compute_inverse_kinematics(
        #     target_position, target_orientation
        # )
        #
        # if not ik_success:
        #     return False

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
        Move to target using BATCHED COLLISION-FREE IK (cuRobo-style)

        This uses GPU-accelerated batched IK optimization:
        - Batched IK solving: ~0.05-0.1s (100 seeds in parallel)
        - Collision-free solutions
        - Smooth joint interpolation (no jerky motion)
        - No simulation freezing (non-blocking)

        Much faster than MPC (0.2-0.3s) and more robust than simple IK!

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
        return True

    async def _reset_to_safe_config(self):
        """Reset robot to a known safe configuration using direct joint control"""
        # Franka has 9 DOF: 7 arm joints + 2 gripper joints
        # Safe config: arm joints + gripper closed positions
        safe_arm_joints = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])
        safe_joint_positions = np.concatenate([safe_arm_joints, self.gripper.joint_closed_positions])

        articulation_controller = self.franka.get_articulation_controller()
        articulation_controller.apply_action(ArticulationAction(joint_positions=safe_joint_positions))
        for _ in range(30):
            await omni.kit.app.get_app().next_update_async()

    async def _taskspace_straight_line(self, start_pos, end_pos, orientation, num_waypoints=8):
        """
        DEPRECATED: This function used Lula IK solver which has been removed.
        Use _move_to_target_rrt() with batched IK instead.
        """
        print("[WARNING] _taskspace_straight_line is deprecated - use _move_to_target_rrt instead")
        return False

        # NOTE: Old Lula IK code commented out - not used anymore
        # # Update robot base pose for kinematics solver
        # robot_base_translation, robot_base_orientation = self.franka.get_world_pose()
        # self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        #
        # # Get current joint positions as starting point (7 arm joints only)
        # current_joint_positions = self.franka.get_joint_positions()[:7]
        #
        # # Compute IK for end position
        # ik_action, ik_success = self.articulation_kinematics_solver.compute_inverse_kinematics(
        #     end_pos, orientation
        # )
        #
        # if not ik_success:
        #     return False

        # Extract joint positions from ArticulationAction (7 arm joints only)
        end_joint_positions = ik_action.joint_positions[:7]

        # Interpolate in joint space (smooth motion, maintains orientation)
        for i in range(1, num_waypoints + 1):
            alpha = i / num_waypoints
            interpolated_joints = current_joint_positions + alpha * (end_joint_positions - current_joint_positions)

            # Apply joint positions to arm only (indices 0-6)
            action = ArticulationAction(
                joint_positions=interpolated_joints,
                joint_indices=np.array([0, 1, 2, 3, 4, 5, 6])
            )
            self.franka.get_articulation_controller().apply_action(action)

            # Wait for physics update
            await omni.kit.app.get_app().next_update_async()

        # No settling time for faster motion
        return True

    def _plan_to_target(self, target_position, target_orientation):
        """
        Plan path to target using BATCHED COLLISION-FREE IK (cuRobo-style)

        This uses GPU-accelerated batched IK optimization:
        1. Generate 100 random seed configurations
        2. Stage 1: MPPI optimization (5-10 iterations) - fast global search
        3. Stage 2: Gradient descent (10-20 iterations) - precise refinement
        4. Select best collision-free solution

        Much faster than MPC and more robust than simple IK!
        Time: ~0.05-0.1s (vs MPC: 0.2-0.3s, Simple IK: <0.01s)
        """
        # Update dynamic obstacles from Lidar before planning (real-time detection)
        self._update_dynamic_obstacles()

        # NOTE: Removed Lula IK base pose setting - batched IK doesn't need it
        # (Batched IK uses URDF-based FK which assumes robot base at origin)

        # Get current joint positions
        current_joints = self.franka.get_joint_positions()
        current_arm_joints = current_joints[:7]  # First 7 joints (arm only)
        gripper_joints = current_joints[7:9]  # Gripper joints

        if np.any(np.isnan(current_arm_joints)) or np.any(np.abs(current_arm_joints) > 10.0):
            print(f"[BatchedIK] Invalid robot config: {current_arm_joints}")
            return None

        print(f"[BatchedIK] Planning from {current_arm_joints[:3]}")
        print(f"[BatchedIK] Target position: {target_position}")

        # Update obstacles for batched IK (only if obstacles exist)
        if len(self.dynamic_obstacles) > 0:
            obstacle_list = []
            for obs_name, obs_data in self.dynamic_obstacles.items():
                obstacle_list.append({
                    'position': obs_data['position'],
                    'radius': obs_data['radius']
                })
            self.mpc.set_obstacles(obstacle_list)

        # Convert orientation to quaternion if needed
        target_quat = None
        if target_orientation is not None:
            # Isaac Sim uses [w, x, y, z] quaternion format
            target_quat = np.array(target_orientation)

        # Use batched IK solver (cuRobo-style with URDF-based FK)
        try:
            start_time = time.time()
            goal_joints, success, info = self.mpc.solve(
                target_position=target_position,
                target_orientation=target_quat,
                current_joints=current_arm_joints,
                retract_config=None,
                return_all_solutions=False
            )
            solve_time = time.time() - start_time

            if not success or goal_joints is None:
                print(f"[BatchedIK] ✗ No solution found (time: {solve_time:.3f}s)")
                return None

            # Extract error information
            pos_err = info.get('position_error', 0.0)
            rot_err = info.get('rotation_error', 0.0)

            print(f"[BatchedIK] ✓ Solution found: pos_err={pos_err*100:.2f}cm, rot_err={rot_err:.3f} (time: {solve_time:.3f}s)")

            # DEBUG: Verify FK of solution matches target
            import torch
            goal_joints_tensor = torch.tensor(goal_joints, device='cuda:0', dtype=torch.float32).unsqueeze(0)
            fk_pos, fk_quat = self.mpc.fk.forward_kinematics(goal_joints_tensor)
            fk_pos_np = fk_pos[0].cpu().numpy()
            fk_quat_np = fk_quat[0].cpu().numpy()
            print(f"  [DEBUG-FK] IK solution joints: {goal_joints}")
            print(f"  [DEBUG-FK] FK of solution: pos={fk_pos_np}, quat={fk_quat_np}")
            print(f"  [DEBUG-FK] Target: pos={target_position}, quat={target_quat}")
            print(f"  [DEBUG-FK] FK error: {np.linalg.norm(fk_pos_np - target_position)*100:.2f}cm")

            # Generate smooth joint-space trajectory (linear interpolation)
            num_waypoints = 20
            trajectory_actions = []

            for i in range(num_waypoints + 1):
                alpha = i / num_waypoints
                # Smooth interpolation with ease-in-out
                alpha_smooth = alpha * alpha * (3.0 - 2.0 * alpha)

                # Interpolate joint positions
                interpolated_joints = current_arm_joints + alpha_smooth * (goal_joints - current_arm_joints)

                # Combine with gripper joints
                full_joint_positions = np.concatenate([interpolated_joints, gripper_joints])
                action = ArticulationAction(joint_positions=full_joint_positions)
                trajectory_actions.append(action)

            print(f"[BatchedIK] Generated smooth trajectory with {len(trajectory_actions)} waypoints")
            print(f"  [DEBUG-TRAJ] First waypoint joints: {trajectory_actions[0].joint_positions[:7]}")
            print(f"  [DEBUG-TRAJ] Last waypoint joints: {trajectory_actions[-1].joint_positions[:7]}")
            return trajectory_actions

        except Exception as e:
            print(f"[BatchedIK] Planning failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _execute_plan(self, action_sequence, skip_factor=3):
        """Execute trajectory action sequence

        Args:
            action_sequence: Sequence of actions to execute
            skip_factor: Number of frames to skip (higher = faster, default=3 for 60 FPS)
        """
        if action_sequence is None or len(action_sequence) == 0:
            return False

        # Use articulation controller for better position control
        articulation_controller = self.franka.get_articulation_controller()

        # Apply each waypoint multiple times (like cuRobo's wait_steps)
        # This ensures the robot actually reaches each waypoint before proceeding
        # With proper gains (kp=1e4, kd=1e3), we need fewer steps
        steps_per_waypoint = 5  # Reduced from 10 (proper gains allow faster convergence)

        for i, action in enumerate(action_sequence):
            for _ in range(steps_per_waypoint):
                articulation_controller.apply_action(action)
                await omni.kit.app.get_app().next_update_async()

        # Apply final action even more times to ensure convergence
        final_action = action_sequence[-1]
        for _ in range(30):  # Reduced from 50 (proper gains allow faster convergence)
            articulation_controller.apply_action(final_action)
            await omni.kit.app.get_app().next_update_async()

        return True

    def _attach_cube_to_gripper(self, cube):
        """Attach cube to gripper using fixed joint

        Creates a USD fixed joint between the gripper and cube.
        IMPORTANT: Cube must be positioned close to gripper before calling this!
        """
        try:
            from pxr import UsdPhysics, Gf

            # Get the stage
            stage = omni.usd.get_context().get_stage()

            # Get cube prim
            cube_prim_path = cube.prim_path

            # Get gripper prim path - use the actual Franka prim path from self.franka
            # The Franka robot prim path is dynamic (e.g., /World/Franka_1764603325298)
            gripper_prim_path = f"{self.franka.prim_path}/panda_hand"

            # Verify gripper prim exists
            gripper_prim = stage.GetPrimAtPath(gripper_prim_path)
            if not gripper_prim or not gripper_prim.IsValid():
                print(f"  [ATTACH] ERROR: Gripper prim not found at {gripper_prim_path}")
                print(f"  [ATTACH] Franka prim path: {self.franka.prim_path}")
                return False

            # CRITICAL FIX: Position cube at gripper location BEFORE creating joint
            # This prevents "disjointed body transforms" warning
            ee_pos, ee_rot = self.franka.end_effector.get_world_pose()

            # Position cube below gripper (at gripper contact point)
            cube_half = 0.02575  # Half of cube size (0.0515m)
            gripper_offset = 0.092  # Distance from EE to gripper fingers
            cube_offset = gripper_offset - cube_half  # Net offset below EE

            # Set cube position to be exactly at gripper contact point
            cube_attach_pos = ee_pos + np.array([0.0, 0.0, -cube_offset])
            cube.set_world_pose(cube_attach_pos, ee_rot)

            print(f"  [ATTACH] Positioned cube at gripper: {cube_attach_pos}")

            # Create a fixed joint between gripper and cube
            joint_path = f"{cube_prim_path}/attachment_joint"

            # Remove old joint if it exists
            if stage.GetPrimAtPath(joint_path):
                stage.RemovePrim(joint_path)

            # Create fixed joint
            fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
            fixed_joint.CreateBody0Rel().SetTargets([gripper_prim_path])
            fixed_joint.CreateBody1Rel().SetTargets([cube_prim_path])

            # Set joint to be enabled
            fixed_joint.CreateJointEnabledAttr(True)

            # Set local poses for the joint (relative to bodies)
            # This ensures smooth attachment without snapping
            fixed_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, -cube_offset))
            fixed_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))

            # Store the cube reference
            self._attached_cube = cube

            print(f"  [ATTACH] Cube attached to gripper via fixed joint")
            print(f"  [ATTACH] Gripper path: {gripper_prim_path}")
            return True

        except Exception as e:
            print(f"  [ATTACH] Failed to attach cube: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _detach_cube_from_gripper(self, cube):
        """Detach cube from gripper by removing the fixed joint"""
        try:
            # Get the stage
            stage = omni.usd.get_context().get_stage()

            # Remove the fixed joint
            joint_path = f"{cube.prim_path}/attachment_joint"
            if stage.GetPrimAtPath(joint_path):
                stage.RemovePrim(joint_path)
                print(f"  [DETACH] Fixed joint removed")

            # Clear the attached cube reference
            self._attached_cube = None

            return True

        except Exception as e:
            print(f"  [DETACH] Failed to detach cube: {e}")
            import traceback
            traceback.print_exc()
            return False



    def compute_forward_kinematics(self):
        """
        DEPRECATED: This function used Lula IK solver which has been removed.
        Use self.mpc.fk.forward_kinematics() for batched FK instead.
        """
        print("[WARNING] compute_forward_kinematics is deprecated - use self.mpc.fk.forward_kinematics instead")
        return None, None

        # NOTE: Old Lula FK code commented out - not used anymore
        # if self.articulation_kinematics_solver is None:
        #     carb.log_warn("Articulation kinematics solver not initialized")
        #     return None, None
        #
        # # Update robot base pose
        # robot_base_translation, robot_base_orientation = self.franka.get_world_pose()
        # self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        #
        # # Compute end effector pose
        # ee_position, ee_rot_mat = self.articulation_kinematics_solver.compute_end_effector_pose()
        #
        # return ee_position, ee_rot_mat

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
            self.mpc = None
            # NOTE: Lula IK solvers removed - using batched IK instead
            # self.kinematics_solver = None
            # self.articulation_kinematics_solver = None
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
        if not self.world or not self.mpc:
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

            # Default obstacle position and size
            obstacle_position = np.array([0.63, 0.26, 0.11])
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

            # Create FixedCuboid wrapper for MPC planner (doesn't create physics view, safe to add during simulation)
            obstacle = FixedCuboid(
                name=obstacle_name,
                prim_path=obstacle_prim_path,
                size=1.0,
                scale=obstacle_size
            )

            # Add to world scene
            self.world.scene.add(obstacle)

            # Store obstacle reference
            self.obstacles[obstacle_name] = obstacle

            # Update MPC with new obstacle (convert to dictionary format)
            obstacle_list = []
            for obs in self.obstacles.values():
                pos, _ = obs.get_world_pose()
                obstacle_list.append({
                    'position': np.array(pos),
                    'radius': 0.15  # Approximate radius for cuboid obstacles
                })
            self.mpc.update_obstacles(obstacle_list)

            self._update_status(f"Obstacle added ({len(self.obstacles)})")

        except Exception as e:
            self._update_status(f"Error adding obstacle: {e}")
            import traceback
            traceback.print_exc()

    def _on_remove_obstacle(self):
        """Remove obstacle button callback"""
        if not self.world or not self.mpc:
            self._update_status("Load scene first!")
            return

        if len(self.obstacles) == 0:
            self._update_status("No obstacles to remove!")
            return

        try:
            # Get the last added obstacle
            obstacle_name = list(self.obstacles.keys())[-1]

            # Remove from scene
            self.world.scene.remove_object(obstacle_name)

            # Remove from our tracking dictionary
            del self.obstacles[obstacle_name]

            # Update MPC with remaining obstacles
            obstacle_positions = [np.array(obs.get_world_pose()[0]) for obs in self.obstacles.values()]
            self.mpc.update_obstacles(obstacle_positions)

            print(f"Obstacle removed: {obstacle_name} (Remaining: {len(self.obstacles)})")
            self._update_status(f"Obstacle removed ({len(self.obstacles)})")

        except Exception as e:
            self._update_status(f"Error removing obstacle: {e}")
            import traceback
            traceback.print_exc()


# Create and show UI
app = FrankaMPCDynamicGrid()

