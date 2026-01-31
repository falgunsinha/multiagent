"""
Franka RRT Pick and Place - Dynamic Grid with Kinematics Solver
RRT path planning with obstacle avoidance, conservative collision spheres,
dynamic grid configuration, pick retry logic, return to home after each cube.
Uses PhysX Lidar - Rotating and Intel RealSense D455 depth sensor for obstacle detection.
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


class FrankaRRTDynamicGrid:
    """Dynamic grid pick and place with RRT"""

    def __init__(self):
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
        self.lidar_point_cloud_buffer = []  # Accumulate point clouds over multiple frames
        self.lidar_buffer_max_frames = 90  # Accumulate 90 frames (1.5 seconds at 60 Hz)

        # Depth Camera sensor (SingleViewDepthSensor)
        self.depth_camera = None  # Depth camera sensor

        # Physics callback subscription
        self._physics_callback_subscription = None

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
                    valid_range=(0.2, 100.0)  # 0.2m to 100m range (reduced from 0.4m to detect closer obstacles)
                )
            )

            # Initialize Depth Camera attached to panda hand
            print("[DEPTH CAMERA] Initializing depth camera on panda hand...")
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

            print(f"[DEPTH CAMERA] Depth camera created at {depth_camera_prim_path} (attached to panda hand)")

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

            print("[DEPTH CAMERA] Depth camera initialized on panda hand")
            print("[DEPTH CAMERA] Attached annotators:")
            print("  - DepthSensorDistance (distance measurements)")
            print("  - DepthSensorPointCloudPosition (3D point cloud)")
            print("  - DepthSensorPointCloudColor (point cloud colors)")
            print("[DEPTH CAMERA] Position: 5cm above panda hand, looking forward/down")
            print("[DEPTH CAMERA] Resolution: 512x512 (square), Frequency: 10 Hz")
            print("[DEPTH CAMERA] Depth range: 0.1m - 2.0m, Baseline: 55mm")

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

            # DO NOT add cubes as obstacles - they are targets to pick, not obstacles to avoid
            # Only manual obstacles will be added to RRT for collision avoidance

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

    def _setup_rrt(self):
        """Setup RRT motion planner and kinematics solvers"""
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_id = ext_manager.get_enabled_extension_id("isaacsim.examples.interactive")
        examples_extension_path = ext_manager.get_extension_path(ext_id)

        robot_description_file = os.path.join(
            examples_extension_path, "isaacsim", "examples", "interactive", "path_planning",
            "path_planning_example_assets", "franka_conservative_spheres_robot_description.yaml")
        urdf_path = os.path.join(mg_extension_path, "motion_policy_configs", "franka", "lula_franka_gen.urdf")
        rrt_config_file = os.path.join(mg_extension_path, "path_planner_configs", "franka", "rrt", "franka_planner_config.yaml")

        self.rrt = RRT(robot_description_path=robot_description_file, urdf_path=urdf_path,
                       rrt_config_path=rrt_config_file, end_effector_frame_name="right_gripper")
        self.rrt.set_max_iterations(10000)

        self.path_planner_visualizer = PathPlannerVisualizer(robot_articulation=self.franka, path_planner=self.rrt)
        self.kinematics_solver = LulaKinematicsSolver(robot_description_path=robot_description_file, urdf_path=urdf_path)
        self.articulation_kinematics_solver = ArticulationKinematicsSolver(
            self.franka, self.kinematics_solver, "right_gripper")

        self.cspace_trajectory_generator = LulaCSpaceTrajectoryGenerator(robot_description_file, urdf_path)

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

            # Add current points to buffer
            self.lidar_point_cloud_buffer.append(points)

            # Keep only the last N frames (sliding window)
            if len(self.lidar_point_cloud_buffer) > self.lidar_buffer_max_frames:
                self.lidar_point_cloud_buffer.pop(0)

            # Need at least 90 frames for reliable detection
            if len(self.lidar_point_cloud_buffer) < 90:
                return []

            # Concatenate all buffered point clouds
            points = np.vstack(self.lidar_point_cloud_buffer)

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

            # Filter by height (world coordinates) - only 8cm to 18cm height
            # Cubes: cube_size/2 + 0.01 = 2.575cm + 1cm = 3.575cm ✅ Filtered out (below 8cm)
            # Container: at 0cm height ✅ Filtered out (below 8cm)
            # Obstacles: center at ~11cm ✅ Detected (within 8-18cm range)
            valid_points = points_world[(points_world[:, 2] > 0.08) & (points_world[:, 2] < 0.18)]

            # Filter by distance from robot base - STRICT workspace limits
            robot_pos, _ = self.franka.get_world_pose()
            distances_from_robot = np.linalg.norm(valid_points[:, :2] - robot_pos[:2], axis=1)
            valid_points = valid_points[(distances_from_robot > 0.30) & (distances_from_robot < 0.90)]  # 30cm-90cm only

            # Filter out robot base and arm region
            robot_base_pos = np.array([0.0, 0.0])
            robot_arm_radius = 0.55
            robot_region_mask = np.linalg.norm(valid_points[:, :2] - robot_base_pos, axis=1) > robot_arm_radius
            valid_points = valid_points[robot_region_mask]

            # Filter out cube positions (exclude points near cube locations)
            # Cubes are targets to pick, not obstacles to avoid
            cube_exclusion_radius = 0.08  # 8cm radius around each cube
            cube_mask = np.ones(len(valid_points), dtype=bool)
            for cube, _ in self.cubes:
                try:
                    cube_pos, _ = cube.get_world_pose()
                    distances_to_cube = np.linalg.norm(valid_points[:, :2] - cube_pos[:2], axis=1)
                    cube_mask &= (distances_to_cube > cube_exclusion_radius)
                except:
                    pass  # Skip if cube position unavailable
            valid_points = valid_points[cube_mask]

            # Filter out container region (exclude points near container)
            # Container is at (0.30, 0.50, 0.0) with dimensions ~0.48m x 0.36m
            container_center = np.array([0.30, 0.50])
            container_half_size = np.array([0.30, 0.24])  # Half of container size + margin
            container_mask = (
                (np.abs(valid_points[:, 0] - container_center[0]) > container_half_size[0]) |
                (np.abs(valid_points[:, 1] - container_center[1]) > container_half_size[1])
            )
            valid_points = valid_points[container_mask]

            detected_obstacles = []

            if len(valid_points) > 10:
                # SIMPLIFIED APPROACH: Direct clustering without grid downsampling
                # This preserves full point cloud precision for accurate bounding boxes

                # Use DBSCAN-like clustering with distance threshold
                clusters = []
                unassigned = set(range(len(valid_points)))

                while unassigned:
                    # Start new cluster with first unassigned point
                    seed_idx = next(iter(unassigned))
                    cluster = [seed_idx]
                    unassigned.remove(seed_idx)

                    # Grow cluster by finding nearby points
                    to_check = [seed_idx]
                    while to_check:
                        current_idx = to_check.pop()
                        current_point = valid_points[current_idx]

                        # Find all unassigned points within 15cm (XY distance only)
                        for idx in list(unassigned):
                            point = valid_points[idx]
                            dist_xy = np.linalg.norm(current_point[:2] - point[:2])
                            if dist_xy < 0.15:  # 15cm clustering threshold
                                cluster.append(idx)
                                unassigned.remove(idx)
                                to_check.append(idx)

                    # Only keep clusters with enough points
                    if len(cluster) > 10:  # Require at least 10 points for valid obstacle
                        clusters.append(valid_points[cluster])

                # Calculate bounding boxes for each cluster
                for cluster_points in clusters:
                    # Calculate axis-aligned bounding box (AABB)
                    min_bounds = np.min(cluster_points, axis=0)
                    max_bounds = np.max(cluster_points, axis=0)

                    # Center position
                    center_pos = (min_bounds + max_bounds) / 2.0

                    # Size (dimensions) from bounding box
                    size = max_bounds - min_bounds

                    # Add 10cm safety margin to each dimension (5cm on each side)
                    size_with_margin = size + 0.10

                    # Ensure minimum size of 10cm per dimension
                    size_with_margin = np.maximum(size_with_margin, 0.10)

                    # Store as dict with position, size, and point count
                    detected_obstacles.append({
                        'position': center_pos.tolist(),
                        'size': size_with_margin.tolist(),
                        'point_count': len(cluster_points)
                    })

                # Log detected obstacles (throttled - only when count changes or every 10 seconds)
                current_time = time.time()
                last_log_time = getattr(self, '_last_obstacle_log_time', 0)
                last_count = getattr(self, '_last_logged_obstacle_count', -1)
                should_log = (len(detected_obstacles) != last_count) or (current_time - last_log_time > 10.0)

                if len(detected_obstacles) > 0 and should_log:
                    print(f"[LIDAR] Detected {len(detected_obstacles)} obstacle(s)")
                    self._last_obstacle_log_time = current_time
                    self._last_logged_obstacle_count = len(detected_obstacles)

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
                return []

            # Get depth data from the frame (DepthSensorDistance annotator)
            depth_data = None
            if "data" in depth_frame:
                depth_data = depth_frame["data"]
            elif "distance" in depth_frame:
                depth_data = depth_frame["distance"]
            elif "depth" in depth_frame:
                depth_data = depth_frame["depth"]

            if depth_data is None or len(depth_data) == 0:
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
        Update RRT planner with dynamically detected obstacles from Lidar and Depth Camera.
        Also moves Obstacle_1 automatically.

        OPTIMIZED: Instead of deleting and recreating obstacles, we:
        1. Reuse existing obstacle prims by moving them
        2. Only create new prims if we need more
        3. Only delete prims if we have too many
        This avoids constant stage updates and maintains 60 FPS
        """
        if self.lidar is None or self.rrt is None:
            return

        try:
            # Move Obstacle_1 automatically (if enabled)
            if self.obstacle_1_moving:
                self._move_obstacle()

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

            # IMPROVED: Match detected obstacles to existing obstacles by position (nearest neighbor)
            # This prevents obstacles from "jumping" between slots when clustering order changes
            if num_current > 0 and num_detected > 0:
                # Get current obstacle positions
                existing_obstacles = list(self.lidar_detected_obstacles.items())
                current_positions = []
                for obs_name, obs_obj in existing_obstacles:
                    pos, _ = obs_obj.get_world_pose()
                    current_positions.append(pos[:2])  # XY only
                current_positions = np.array(current_positions)

                # Get detected obstacle positions
                detected_obstacle_positions = np.array([obs['position'][:2] for obs in detected_positions])

                # Match using greedy nearest neighbor (simple but effective)
                matched_pairs = []
                used_detected = set()
                used_current = set()

                # For each current obstacle, find nearest detected obstacle
                for curr_idx in range(num_current):
                    if curr_idx in used_current:
                        continue

                    best_det_idx = None
                    best_dist = float('inf')

                    for det_idx in range(num_detected):
                        if det_idx in used_detected:
                            continue

                        dist = np.linalg.norm(current_positions[curr_idx] - detected_obstacle_positions[det_idx])
                        if dist < best_dist and dist < 0.30:  # Max 30cm matching threshold
                            best_dist = dist
                            best_det_idx = det_idx

                    if best_det_idx is not None:
                        matched_pairs.append((curr_idx, best_det_idx))
                        used_current.add(curr_idx)
                        used_detected.add(best_det_idx)

                # Update matched obstacles
                for curr_idx, det_idx in matched_pairs:
                    obs_name, obs_obj = existing_obstacles[curr_idx]
                    obs_data = detected_positions[det_idx]
                    new_pos = np.array(obs_data['position'])
                    new_size = np.array(obs_data['size'])

                    # Always update position and size (no threshold check for fast-moving obstacles)
                    obs_obj.set_world_pose(position=new_pos)
                    obs_obj.set_local_scale(new_size)

                # Update RRT once after all obstacles are updated
                self.rrt.update_world()

                # Handle unmatched detected obstacles (new obstacles appeared)
                unmatched_detected = [i for i in range(num_detected) if i not in used_detected]
                for det_idx in unmatched_detected:
                    obs_data = detected_positions[det_idx]
                    obs_pos = np.array(obs_data['position'])
                    obs_size = np.array(obs_data['size'])

                    # Find available slot
                    obs_idx = len(self.lidar_detected_obstacles)
                    obs_name = f"lidar_obstacle_{obs_idx}"
                    obs_prim_path = f"/World/LidarObstacle_{obs_idx}"

                    # Create new obstacle
                    obstacle = self.world.scene.add(
                        FixedCuboid(
                            name=obs_name,
                            prim_path=obs_prim_path,
                            position=obs_pos,
                            size=1.0,
                            scale=obs_size,
                            color=np.array([1.0, 0.0, 0.0]),
                            visible=False
                        )
                    )
                    obstacle.set_visibility(False)
                    self.rrt.add_obstacle(obstacle, static=False)
                    self.lidar_detected_obstacles[obs_name] = obstacle

                # Handle unmatched current obstacles (obstacles disappeared)
                unmatched_current = [i for i in range(num_current) if i not in used_current]
                for curr_idx in unmatched_current:
                    obs_name, obs_obj = existing_obstacles[curr_idx]
                    try:
                        self.rrt.remove_obstacle(obs_obj)
                        self.world.scene.remove_object(obs_name)
                        del self.lidar_detected_obstacles[obs_name]
                    except:
                        pass

            elif num_detected > 0 and num_current == 0:
                # First time detecting obstacles - create all
                for i, obs_data in enumerate(detected_positions):
                    obs_name = f"lidar_obstacle_{i}"
                    obs_prim_path = f"/World/LidarObstacle_{i}"
                    obs_pos = np.array(obs_data['position'])
                    obs_size = np.array(obs_data['size'])

                    obstacle = self.world.scene.add(
                        FixedCuboid(
                            name=obs_name,
                            prim_path=obs_prim_path,
                            position=obs_pos,
                            size=1.0,
                            scale=obs_size,
                            color=np.array([1.0, 0.0, 0.0]),
                            visible=False
                        )
                    )
                    obstacle.set_visibility(False)
                    self.rrt.add_obstacle(obstacle, static=False)
                    self.lidar_detected_obstacles[obs_name] = obstacle

                self.rrt.update_world()

            elif num_detected == 0 and num_current > 0:
                # All obstacles disappeared - remove all
                for obs_name, obs_obj in list(self.lidar_detected_obstacles.items()):
                    try:
                        self.rrt.remove_obstacle(obs_obj)
                        self.world.scene.remove_object(obs_name)
                    except:
                        pass
                self.lidar_detected_obstacles.clear()

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

        # Process Lidar data EVERY physics step to accumulate point clouds
        if self.lidar is not None:
            self._process_lidar_data()

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

            # CRITICAL: Update RRT's world representation after moving obstacle
            # This ensures RRT knows the obstacle's current position for path planning
            if self.rrt is not None:
                self.rrt.update_world()

        except Exception as e:
            carb.log_warn(f"[OBSTACLE] Error moving Obstacle_1: {e}")
            import traceback
            traceback.print_exc()

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

    async def _wait_for_lidar_buffer(self, required_frames=90, timeout_seconds=5):
        """Wait for Lidar buffer to accumulate enough frames for obstacle detection

        Args:
            required_frames: Minimum number of frames needed (default: 90 = 1.5 seconds at 60 Hz)
            timeout_seconds: Maximum time to wait (default: 5 seconds)

        Returns:
            bool: True if buffer filled, False if timeout
        """
        if self.lidar is None:
            return True  # No Lidar, skip waiting

        print(f"[LIDAR] Waiting for buffer ({required_frames} frames, ~{required_frames/60:.1f}s)...")
        start_time = time.time()

        while len(self.lidar_point_cloud_buffer) < required_frames:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                print(f"[LIDAR] Timeout (only {len(self.lidar_point_cloud_buffer)}/{required_frames} frames)")
                return False

            await omni.kit.app.get_app().next_update_async()

        print(f"[LIDAR] Buffer ready ({len(self.lidar_point_cloud_buffer)} frames)")
        return True

    async def _pick_place_loop(self):
        """Main pick and place loop"""
        try:
            # Wait for Lidar buffer to fill before starting (only on first cube)
            if self.current_cube_index == 0 and len(self.obstacles) > 0:
                self._update_status("Waiting for Lidar...")
                await self._wait_for_lidar_buffer(required_frames=90, timeout_seconds=5)
                self._update_status("Lidar ready!")
        except Exception as e:
            print(f"[ERROR] Lidar wait failed: {e}")

        try:
            self.timeline.play()

            # Add physics callback using PhysX interface directly (more reliable than World.add_physics_callback)
            if not hasattr(self, '_physics_callback_subscription') or self._physics_callback_subscription is None:
                try:
                    import omni.physx
                    physx_interface = omni.physx.get_physx_interface()
                    if physx_interface is not None:
                        self._physics_callback_subscription = physx_interface.subscribe_physics_step_events(
                            self._physics_step_callback
                        )
                        print("[PHYSICS] Physics step callback subscribed successfully")
                    else:
                        print("[PHYSICS] Warning: PhysX interface not available")
                        self._physics_callback_subscription = None
                except Exception as e:
                    print(f"[PHYSICS] Warning: Could not subscribe to physics callback: {e}")
                    self._physics_callback_subscription = None

            # Start Obstacle_1 automatic movement
            self.obstacle_1_moving = True
            print("[OBSTACLE] Obstacle_1 automatic movement started")

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

            # Stop Obstacle_1 automatic movement
            self.obstacle_1_moving = False
            print("[OBSTACLE] Obstacle_1 automatic movement stopped")

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

                # DEBUG: Print cube position and distance from robot
                robot_pos, _ = self.franka.get_world_pose()
                dist_from_robot = np.linalg.norm(cube_pos_current[:2] - robot_pos[:2])
                if pick_attempt == 1:
                    print(f"  Cube position: ({cube_pos_current[0]:.3f}, {cube_pos_current[1]:.3f}, {cube_pos_current[2]:.3f})")
                    print(f"  Distance from robot: {dist_from_robot:.3f}m")

                    # DEBUG: Show obstacle detection status
                    num_obstacles = len(self.lidar_detected_obstacles)
                    print(f"  [OBSTACLE DEBUG] Currently tracking {num_obstacles} obstacles in RRT")
                    if num_obstacles > 0:
                        for obs_name, obs_obj in self.lidar_detected_obstacles.items():
                            obs_pos, _ = obs_obj.get_world_pose()
                            obs_scale = obs_obj.get_local_scale()
                            print(f"    - {obs_name}: pos=({obs_pos[0]:.3f}, {obs_pos[1]:.3f}, {obs_pos[2]:.3f}), size=({obs_scale[0]:.3f}, {obs_scale[1]:.3f}, {obs_scale[2]:.3f})")

                pre_pick_height = 0.10 if total_cubes <= 4 else (0.12 if total_cubes <= 9 else 0.15)
                pre_pick_pos = cube_pos_current + np.array([0.0, 0.0, pre_pick_height])

                # Phase 1: Pre-pick (open gripper BEFORE approaching) - FASTER with skip_factor=4
                articulation_controller = self.franka.get_articulation_controller()
                articulation_controller.apply_action(ArticulationAction(
                    joint_positions=self.gripper.joint_opened_positions, joint_indices=np.array([7, 8])))
                for _ in range(2):  # Gripper open (reduced from 3)
                    await omni.kit.app.get_app().next_update_async()

                robot_base_translation, robot_base_orientation = self.franka.get_world_pose()
                self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
                _, ik_success = self.articulation_kinematics_solver.compute_inverse_kinematics(pre_pick_pos, orientation)

                if not ik_success:
                    print(f"  ❌ IK FAILED for pre-pick position: ({pre_pick_pos[0]:.3f}, {pre_pick_pos[1]:.3f}, {pre_pick_pos[2]:.3f})")
                    if pick_attempt < max_pick_attempts:
                        continue
                    return False, f"{cube_name} out of reach (IK failed)"

                success = await self._move_to_target_rrt(pre_pick_pos, orientation, skip_factor=4)
                if not success:
                    print(f"  ❌ RRT FAILED for pre-pick position: ({pre_pick_pos[0]:.3f}, {pre_pick_pos[1]:.3f}, {pre_pick_pos[2]:.3f})")
                    print(f"  ⚠️ Obstacle may be blocking path - skipping this cube")
                    if pick_attempt < max_pick_attempts:
                        continue
                    # Don't abort - skip to next cube instead
                    return False, f"SKIP: RRT failed (obstacle blocking)"

                # Phase 2: Pick approach (gripper already open)
                cube_pos_realtime, _ = cube.get_world_pose()
                pick_pos = np.array([cube_pos_realtime[0], cube_pos_realtime[1], cube_pos_realtime[2]])

                # Moderate speed approach with skip_factor=3 for good balance
                success = await self._move_to_target_rrt(pick_pos, orientation, skip_factor=3)
                if not success:
                    print(f"  ❌ RRT FAILED for pick position: ({pick_pos[0]:.3f}, {pick_pos[1]:.3f}, {pick_pos[2]:.3f})")
                    print(f"  ⚠️ Obstacle may be blocking path - skipping this cube")
                    if pick_attempt < max_pick_attempts:
                        continue
                    # Don't abort - skip to next cube instead
                    return False, f"SKIP: RRT failed (obstacle blocking)"

                for _ in range(5):  # Pick stabilization (increased from 3 for better alignment)
                    await omni.kit.app.get_app().next_update_async()

                articulation_controller.apply_action(ArticulationAction(
                    joint_positions=self.gripper.joint_closed_positions, joint_indices=np.array([7, 8])))
                for _ in range(15):  # Gripper close (increased from 12 for better grip)
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
            place_success = await self._move_to_target_rrt(release_height, orientation, skip_factor=4)

            # DEBUG: Check where robot actually is before releasing
            actual_ee_pos, _ = self.franka.end_effector.get_world_pose()
            ee_error = np.linalg.norm(actual_ee_pos[:2] - release_height[:2])
            print(f"  [RELEASE DEBUG] Target: ({release_height[0]:.3f}, {release_height[1]:.3f}, {release_height[2]:.3f})")
            print(f"  [RELEASE DEBUG] Actual EE: ({actual_ee_pos[0]:.3f}, {actual_ee_pos[1]:.3f}, {actual_ee_pos[2]:.3f})")
            print(f"  [RELEASE DEBUG] XY Error: {ee_error*100:.1f}cm, RRT Success: {place_success}")

            for _ in range(3):  # Place stabilization (increased from 1 to prevent cube throw)
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

            # Check if cube is actually in container
            in_container_x = (cube_pos_final[0] >= container_x_min) and (cube_pos_final[0] <= container_x_max)
            in_container_y = (cube_pos_final[1] >= container_y_min) and (cube_pos_final[1] <= container_y_max)
            in_container = in_container_x and in_container_y

            if placement_successful:
                print(f"  Place OK ({xy_distance*100:.1f}cm)")
            else:
                print(f"  Place fail ({xy_distance*100:.1f}cm)")

            print(f"  [PLACE RESULT] Final position: ({cube_pos_final[0]:.3f}, {cube_pos_final[1]:.3f}, {cube_pos_final[2]:.3f})")
            if in_container:
                print(f"  ✅ Cube IS in container")
            else:
                print(f"  ❌ Cube is OUTSIDE container!")
                if not in_container_x:
                    print(f"     X out of bounds: {cube_pos_final[0]:.3f} not in [{container_x_min:.3f}, {container_x_max:.3f}]")
                if not in_container_y:
                    print(f"     Y out of bounds: {cube_pos_final[1]:.3f} not in [{container_y_min:.3f}, {container_y_max:.3f}]")

            # Phase 7: Place retreat (faster with skip_factor=6)
            current_ee_pos, _ = self.franka.end_effector.get_world_pose()
            retreat_pos = current_ee_pos + np.array([0.0, 0.0, 0.15])  # Retreat up
            retreat_success = await self._move_to_target_rrt(retreat_pos, orientation, skip_factor=6)

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
        Execute straight-line motion using IK + joint-space interpolation
        This avoids RRT's orientation changes by directly interpolating in joint space

        Args:
            start_pos: Starting position (not used, kept for compatibility)
            end_pos: Target end effector position
            orientation: Target end effector orientation
            num_waypoints: Number of interpolation steps (optimized for speed)

        Returns:
            bool: True if successful, False if IK failed
        """
        # Update robot base pose for kinematics solver
        robot_base_translation, robot_base_orientation = self.franka.get_world_pose()
        self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        # Get current joint positions as starting point (7 arm joints only)
        # Franka has 7 active joints in cspace, gripper is controlled separately
        current_joint_positions = self.franka.get_joint_positions()[:7]

        # Compute IK for end position
        ik_action, ik_success = self.articulation_kinematics_solver.compute_inverse_kinematics(
            end_pos, orientation
        )

        if not ik_success:
            return False

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
        """Plan path to target using RRT with smooth trajectory generation"""
        # Update dynamic obstacles from Lidar before planning (real-time detection)
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

        if np.any(np.isnan(start_pos)) or np.any(np.abs(start_pos) > 10.0):
            carb.log_error(f"Invalid robot config: {start_pos}")
            return None

        total_cubes = len(self.cubes)
        has_obstacles = self.obstacle_counter > 0
        max_iterations = 15000 if has_obstacles else (8000 if total_cubes <= 4 else 12000)
        self.rrt.set_max_iterations(max_iterations)

        # DEBUG: Show RRT planning parameters
        num_obstacles = len(self.lidar_detected_obstacles)
        if num_obstacles > 0:
            print(f"  [RRT DEBUG] Planning with {num_obstacles} obstacles, max_iterations={max_iterations}")

        rrt_plan = self.rrt.compute_path(start_pos, np.array([]))

        if rrt_plan is None or len(rrt_plan) <= 1:
            carb.log_warn(f"RRT failed for {target_position}")
            if num_obstacles > 0:
                print(f"  [RRT DEBUG] RRT failed - likely obstacle blocking path")
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

        # Execute trajectory with frame skipping for speed
        # IMPORTANT: Always execute the LAST waypoint to ensure target is reached
        for i, action in enumerate(action_sequence):
            is_last = (i == len(action_sequence) - 1)
            if i % skip_factor == 0 or is_last:
                self.franka.apply_action(action)
                await omni.kit.app.get_app().next_update_async()

        # Add extra settling frames for final position accuracy
        for _ in range(3):
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
            self.lidar_point_cloud_buffer = []
            self.depth_camera = None
            self.placed_count = 0
            self.current_cube_index = 0
            self.is_picking = False
            self.obstacle_1_moving = False
            self.obstacle_1_force_api_applied = False
            if hasattr(self, '_last_accel'):
                delattr(self, '_last_accel')

            # Unsubscribe from physics callback
            if hasattr(self, '_physics_callback_subscription') and self._physics_callback_subscription is not None:
                self._physics_callback_subscription = None

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

    def _add_obstacle_internal(self):
        """Internal function to add an obstacle (called by button or automatically)"""
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

            # Create FixedCuboid wrapper for RRT planner (doesn't create physics view, safe to add during simulation)
            obstacle = FixedCuboid(
                name=obstacle_name,
                prim_path=obstacle_prim_path,
                size=1.0,
                scale=obstacle_size
            )

            # Add to world scene
            self.world.scene.add(obstacle)

            # DO NOT add ANY manually-created obstacles to RRT planner here!
            #
            # Reason: Lidar will detect ALL obstacles (Obstacle_1, Obstacle_2, etc.) and add them dynamically.
            # Adding them here would create DUPLICATE obstacles in RRT:
            #   - One static obstacle at creation position (this line)
            #   - One dynamic obstacle at detected position (via Lidar)
            #
            # This causes two problems:
            #   1. For MOVING obstacles (Obstacle_1): RRT has old position + current position
            #   2. For STATIC obstacles (Obstacle_2, etc.): RRT has duplicate at same position
            #
            # Solution: Let Lidar handle ALL obstacles for consistent, real-time tracking.
            # Lidar updates every physics step (60 Hz), ensuring RRT always has current positions.
            #
            # NOTE: If you add obstacles OUTSIDE Lidar's detection range, you'll need to
            # manually add them to RRT here with: self.rrt.add_obstacle(obstacle, static=True)

            # Store obstacle reference
            self.obstacles[obstacle_name] = obstacle

            # CRITICAL: Reinitialize Lidar to detect newly added obstacles
            # PhysX Lidar builds collision mesh during initialization
            # Objects added after initialization are not detected unless we reinitialize
            if self.lidar is not None:
                print(f"[LIDAR] Reinitializing Lidar to detect {obstacle_name}...")
                # Clear point cloud buffer
                self.lidar_point_cloud_buffer = []
                # Reinitialize Lidar data streams
                self.lidar.add_depth_data_to_frame()
                self.lidar.add_point_cloud_data_to_frame()
                print(f"[LIDAR] Lidar reinitialized successfully")

            self._update_status(f"Obstacle added ({len(self.obstacles)})")

        except Exception as e:
            self._update_status(f"Error adding obstacle: {e}")
            import traceback
            traceback.print_exc()

    def _on_add_obstacle(self):
        """Add obstacle button callback"""
        if not self.world or not self.rrt:
            self._update_status("Load scene first!")
            return

        self._add_obstacle_internal()

    def _on_remove_obstacle(self):
        """Remove obstacle button callback"""
        if not self.world or not self.rrt:
            self._update_status("Load scene first!")
            return

        if len(self.obstacles) == 0:
            self._update_status("No obstacles to remove!")
            return

        try:
            # Get the last added obstacle
            obstacle_name = list(self.obstacles.keys())[-1]
            obstacle = self.obstacles[obstacle_name]

            # Remove from RRT planner
            self.rrt.remove_obstacle(obstacle)

            # Remove from scene
            self.world.scene.remove_object(obstacle_name)

            # Remove from our tracking dictionary
            del self.obstacles[obstacle_name]

            print(f"Obstacle removed: {obstacle_name} (Remaining: {len(self.obstacles)})")
            self._update_status(f"Obstacle removed ({len(self.obstacles)})")

        except Exception as e:
            self._update_status(f"Error removing obstacle: {e}")
            import traceback
            traceback.print_exc()


# Create and show UI
app = FrankaRRTDynamicGrid()

