"""
Franka RRT Pick and Place - Dynamic Grid with Kinematics Solver + YOLOv8 3D Object Detection

Features:
- RRT path planning with obstacle avoidance
- PhysX Lidar - Rotating for real-time obstacle detection
- YOLOv8 + Depth Camera for 3D object detection with size estimation
- Custom Collision Checker Wrapper for unified obstacle management
- Dynamic obstacle registration with RRT
- Automatic size estimation (no hardcoded dimensions)

Sensors:
- Main_Camera: (0.4, 0.2, 1.5) - RGB for YOLOv8 detection
- Depth_Camera: (1.0, 0.1, 0.4) - Depth for 3D position/size estimation
- PhysX Lidar: Rotating sensor for obstacle detection

Collision Checker System:
1. Real-time sensor querying (Lidar + YOLOv8)
2. Automatic obstacle registration with RRT
3. Dynamic RRT updates (add/remove/move obstacles)
4. Uses detected sizes instead of hardcoded dimensions
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

# Camera imports (REMOVED: Camera, SingleViewDepthSensorAsset - now using Replicator API)

# Add project root to path for local imports
# This is the recommended way for Isaac Sim Script Editor
project_root = Path(r"C:\isaacsim\cobotproject")
sys.path.append(str(project_root))

from src.manipulators import SingleManipulator
from src.grippers import ParallelGripper
# Roboflow API - using direct HTTP requests (no local installation needed)
from src.sensors.physx_lidar_sensor import PhysXLidarSensor
from src.sensors.collision_checker import CollisionChecker
# Note: DepthCameraSensor not needed - using Femto camera directly


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
        self.lidar_sensor = None  # PhysXLidarSensor wrapper
        self.lidar_detected_obstacles = {}  # Dictionary to store dynamically detected obstacles

        # Simple RGB Camera for object detection (2D only)
        # RGB data obtained via Replicator annotator
        self.rgb_annotator = None  # Replicator RGB annotator
        self.render_product = None  # Replicator render product

        # Collision Checker
        self.collision_checker = None  # CollisionChecker wrapper

        # Roboflow API settings (direct HTTP requests, no local installation)
        self.roboflow_api_key = None
        self.roboflow_model_id = None
        self.roboflow_api_url = None
        self.roboflow_confidence = 15
        self.roboflow_overlap = 50

        self.detected_targets = []  # List of detected target cubes/cuboids
        self.detected_obstacles = []  # List of detected obstacles from vision
        self.detected_containers = []  # List of detected containers
        self.detection_enabled = False  # Flag to enable/disable object detection

        # Container position (can be detected or hardcoded)
        self.container_position = np.array([0.30, 0.50, 0.0])



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
                            length_model = ui.SimpleIntModel(2)
                            self.length_field = ui.IntField(height=25, model=length_model)

                        # Width (columns)
                        with ui.HStack(spacing=10):
                            ui.Label("Grid Width (columns):", width=150)
                            width_model = ui.SimpleIntModel(2)
                            self.width_field = ui.IntField(height=25, model=width_model)



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
            print("YOLOv8 will automatically classify which cubes to pick")

            self.timeline.stop()
            World.clear_instance()

            # Single update after cleanup
            await omni.kit.app.get_app().next_update_async()

            self.world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
            self.world.scene.add_default_ground_plane()

            # Single update after world setup
            await omni.kit.app.get_app().next_update_async()

            franka_name = f"franka_{int(time.time() * 1000)}"
            franka_prim_path = f"/World/Franka_{int(time.time() * 1000)}"

            franka_usd_path = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
            robot_prim = add_reference_to_stage(usd_path=franka_usd_path, prim_path=franka_prim_path)
            robot_prim.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
            robot_prim.GetVariantSet("Mesh").SetVariantSelection("Quality")

            # Add simple RGB camera for object detection
            from pxr import UsdGeom, Gf

            stage = omni.usd.get_context().get_stage()

            # Create camera prim
            camera_prim_path = "/World/RGB_Camera"
            camera_prim = UsdGeom.Camera.Define(stage, camera_prim_path)

            # Set camera properties
            camera_prim.GetFocalLengthAttr().Set(12.8)
            camera_prim.GetHorizontalApertureAttr().Set(20.955)
            camera_prim.GetVerticalApertureAttr().Set(15.2908)
            camera_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 10000.0))

            # Position camera to view the workspace
            camera_xform = UsdGeom.Xformable(camera_prim)
            camera_xform.ClearXformOpOrder()

            translate_op = camera_xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(0.8, 0.6, 1.3))

            rotate_op = camera_xform.AddRotateXYZOp()
            rotate_op.Set(Gf.Vec3f(0.0, 0.0, 88.0))

            print(f"[CAMERA] RGB camera created at: {camera_prim_path}")
            print(f"[CAMERA] Focal Length: 12.8")
            print(f"[CAMERA] Position: (0.8, 0.6, 1.3)")
            print(f"[CAMERA] Rotation XYZ: (0.0, 0.0, 88.0)")

            await omni.kit.app.get_app().next_update_async()

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

            # NOTE: Depth camera is now part of Femto camera (camera_tof_nfov)
            # No need to create separate SingleViewDepthSensor
            # The Femto camera provides both RGB (camera_rgb) and Depth (camera_tof_nfov)

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

            # Setup RGB camera with Replicator (for object detection)
            rgb_camera_prim_path = "/World/RGB_Camera"

            import omni.replicator.core as rep
            render_product = rep.create.render_product(rgb_camera_prim_path, resolution=(640, 640))

            self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self.rgb_annotator.attach([render_product])

            print(f"[RGB CAMERA] Replicator annotator attached (640x640)")
            self.render_product = render_product

            # No depth camera needed for 2D detection
            self.depth_annotator = None

            await omni.kit.app.get_app().next_update_async()

            # Initialize Roboflow API settings (direct HTTP requests)
            print("\n" + "="*60)
            print("ROBOFLOW API CONFIGURATION")
            print("="*60)
            print("Using direct API calls (no local installation)")

            self.roboflow_api_key = "WF1HIzXyqs1Ioxdsldgc"  # Default private API key
            self.roboflow_model_id = "finalshapesegment/1"
            self.roboflow_api_url = "https://detect.roboflow.com"
            self.roboflow_confidence = 15  # 15% minimum confidence
            self.roboflow_overlap = 50  # 50% overlap threshold for NMS

            print(f"[ROBOFLOW] API Key: {self.roboflow_api_key[:10]}...")
            print(f"[ROBOFLOW] Model: {self.roboflow_model_id}")
            print(f"[ROBOFLOW] Confidence threshold: {self.roboflow_confidence}%")

            self.detection_enabled = True

            # Initialize PhysXLidarSensor wrapper
            self.lidar_sensor = PhysXLidarSensor(
                lidar_sensor=self.lidar,
                robot_articulation=self.franka,
                container_dimensions=self.container_dimensions,
                verbose=True  # Enable detailed logging
            )

            # Initialize CollisionChecker wrapper
            # Femto camera position and orientation (from Femto Rotate transform)
            self.collision_checker = CollisionChecker(
                depth_camera_position=np.array([0.8, 0.0, 0.8]),
                depth_camera_orientation=np.array([-67.8, -0.1, 90.5]),
                verbose=False
            )

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

            # Add one obstacle automatically for testing YOLOv8 obstacle detection
            await self._add_obstacle_to_scene()

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

        # Use local robot description file from assets folder (absolute path)
        robot_description_file = r"C:\isaacsim\cobotproject\assets\franka_conservative_spheres_robot_description.yaml"

        urdf_path = os.path.join(mg_extension_path, "motion_policy_configs", "franka", "lula_franka_gen.urdf")
        rrt_config_file = os.path.join(mg_extension_path, "path_planner_configs", "franka", "rrt", "franka_planner_config.yaml")

        # Verify files exist
        if not os.path.exists(robot_description_file):
            raise FileNotFoundError(f"Robot description file not found: {robot_description_file}")

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
        Uses PhysXLidarSensor wrapper module.
        Returns list of detected obstacle positions in world coordinates.
        """
        if self.lidar_sensor is None:
            return []

        try:
            # Use PhysXLidarSensor wrapper to process point cloud
            detected_obstacles = self.lidar_sensor.process_point_cloud()
            return detected_obstacles

        except Exception as e:
            carb.log_warn(f"[LIDAR ERROR] Error processing Lidar data: {e}")
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

            # Get detected obstacles from YOLOv8 3D detection (if available)
            # Merge YOLOv8-detected obstacles with Lidar obstacles
            yolov8_obstacles = self._get_yolov8_detected_obstacles()
            if yolov8_obstacles is not None and len(yolov8_obstacles) > 0:
                detected_positions.extend(yolov8_obstacles)

            # Limit to 10 obstacles for performance
            detected_positions = detected_positions[:10]

            num_detected = len(detected_positions)
            num_current = len(self.lidar_detected_obstacles)

            # Case 1: Update existing obstacles by moving them (NO deletion/creation)
            existing_obstacles = list(self.lidar_detected_obstacles.items())
            for i in range(min(num_detected, num_current)):
                obs_name, obs_obj = existing_obstacles[i]
                obs_data = detected_positions[i]

                # Extract position and size from obstacle data
                if isinstance(obs_data, dict):
                    new_pos = np.array(obs_data['position'])
                    new_size = np.array(obs_data.get('size', [0.15, 0.15, 0.15]))
                else:
                    new_pos = np.array(obs_data)
                    new_size = np.array([0.15, 0.15, 0.15])

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

                    obs_data = detected_positions[i]

                    # Extract position and size from obstacle data
                    if isinstance(obs_data, dict):
                        obs_pos = np.array(obs_data['position'])
                        obs_size = np.array(obs_data.get('size', [0.15, 0.15, 0.15]))
                    else:
                        obs_pos = np.array(obs_data)
                        obs_size = np.array([0.15, 0.15, 0.15])

                    obstacle = self.world.scene.add(
                        FixedCuboid(
                            name=obs_name,
                            prim_path=obs_prim_path,
                            position=obs_pos,
                            size=1.0,
                            scale=obs_size,  # Use detected size
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

    def _get_yolov8_detected_obstacles(self):
        """
        Get obstacles detected by YOLOv8 3D detection.
        Uses CollisionChecker wrapper module.
        Returns list of obstacle data with position and size.
        """
        if self.collision_checker is None:
            return []

        try:
            if not hasattr(self, 'detected_obstacles') or self.detected_obstacles is None:
                return []

            # Use CollisionChecker wrapper to get YOLOv8 obstacles
            obstacles = self.collision_checker.get_owlv2_obstacles(self.detected_obstacles)
            return obstacles

        except Exception as e:
            carb.log_warn(f"[COLLISION CHECKER] Error getting YOLOv8 obstacles: {e}")
            return []

    # OLD METHODS - Moved to sensor modules
    # _camera_to_world_transform -> CollisionChecker.camera_to_world_transform
    # check_collision_at_config -> CollisionChecker.check_collision_at_config (placeholder)

    def _get_main_camera_rgb(self):
        """Get RGB image from simple RGB camera using Replicator annotator"""
        if not hasattr(self, 'rgb_annotator') or self.rgb_annotator is None:
            return None

        try:
            rgb_data = self.rgb_annotator.get_data()
            if rgb_data is None or len(rgb_data) == 0:
                return None

            # Convert RGBA to RGB if needed
            if len(rgb_data.shape) == 3 and rgb_data.shape[2] == 4:
                rgb_data = rgb_data[:, :, :3]

            return rgb_data

        except Exception as e:
            print(f"[RGB CAMERA ERROR] {e}")
            return None

    def _call_roboflow_api(self, rgb_image):
        """
        Call Roboflow API directly via HTTP POST request.
        No local installation needed - uses only standard libraries.

        Args:
            rgb_image: RGB image as numpy array (H, W, 3)

        Returns:
            List of detections with bbox, class, confidence, type
        """
        import base64
        import io
        import requests
        import urllib3
        from PIL import Image

        # Suppress SSL warnings (corporate firewall)
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(rgb_image)

            # Convert to JPEG bytes
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=95)
            img_bytes = buffer.getvalue()

            # Encode to base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

            # Build API URL
            api_url = f"{self.roboflow_api_url}/{self.roboflow_model_id}"

            params = {
                "api_key": self.roboflow_api_key,
                "confidence": self.roboflow_confidence,
                "overlap": self.roboflow_overlap
            }

            # Send POST request (SSL verification disabled for corporate firewall)
            print(f"[ROBOFLOW API] Sending request to {api_url}")
            response = requests.post(
                api_url,
                params=params,
                data=img_base64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                verify=False,  # Bypass SSL certificate verification
                timeout=10
            )

            if response.status_code != 200:
                print(f"[ROBOFLOW API ERROR] Status {response.status_code}: {response.text}")
                return []

            result = response.json()
            predictions = result.get('predictions', [])

            print(f"[ROBOFLOW API] Received {len(predictions)} predictions")

            # Parse predictions into standard format
            detections = []
            for pred in predictions:
                class_name = pred.get('class', 'unknown').lower()
                confidence = pred.get('confidence', 0.0)
                x = pred.get('x', 0)
                y = pred.get('y', 0)
                width = pred.get('width', 0)
                height = pred.get('height', 0)

                # Convert center coordinates to bbox (x1, y1, x2, y2)
                x1 = int(x - width / 2)
                y1 = int(y - height / 2)
                x2 = int(x + width / 2)
                y2 = int(y + height / 2)

                # Classify object type
                if class_name in ['cube', 'cylinder']:
                    obj_type = 'target'
                elif class_name in ['cuboid', 'sphere', 'cone']:
                    obj_type = 'obstacle'
                elif class_name == 'container':
                    obj_type = 'container'
                else:
                    obj_type = 'unknown'

                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'center': (int(x), int(y)),
                    'class': class_name,
                    'confidence': confidence,
                    'type': obj_type
                }

                detections.append(detection)

            return detections

        except Exception as e:
            print(f"[ROBOFLOW API ERROR] {e}")
            import traceback
            traceback.print_exc()
            return []

    async def _detect_objects_roboflow_async(self):
        """Async version of Roboflow 2D detection with Replicator step"""
        if not self.detection_enabled or self.roboflow_api_key is None:
            return

        try:
            import omni.replicator.core as rep
            await rep.orchestrator.step_async(rt_subframes=4)

            rgb_frame = self._get_main_camera_rgb()
            if rgb_frame is None:
                return

            # Call Roboflow API directly (2D detection only)
            detections = self._call_roboflow_api(rgb_frame)

            # Print all detections for debugging
            print(f"[ROBOFLOW DEBUG] Total detections: {len(detections)}")
            for i, det in enumerate(detections):
                print(f"  Detection {i+1}: class={det['class']}, type={det['type']}, conf={det['confidence']:.2%}")

            # Separate by type
            self.detected_targets = [d for d in detections if d['type'] == 'target']
            self.detected_obstacles = [d for d in detections if d['type'] == 'obstacle']
            self.detected_containers = [d for d in detections if d['type'] == 'container']

            if len(self.detected_targets) > 0 or len(self.detected_obstacles) > 0:
                print(f"\n[DETECTION] Found {len(self.detected_targets)} targets, {len(self.detected_obstacles)} obstacles")
                print(f"NOTE: 2D detection only - robot uses ground truth positions for picking")
                for i, target in enumerate(self.detected_targets):
                    bbox = target['bbox']
                    print(f"  Target {i+1}: {target['class']} ({target['confidence']:.1%}), bbox={bbox}")
                for i, obs in enumerate(self.detected_obstacles):
                    bbox = obs['bbox']
                    print(f"  Obstacle {i+1}: {obs['class']} ({obs['confidence']:.1%}), bbox={bbox}")



            self._save_detection_images(rgb_frame, self.detected_targets, self.detected_obstacles, self.detected_containers)

        except Exception as e:
            carb.log_warn(f"[ROBOFLOW ERROR] Detection failed: {e}")
            import traceback
            traceback.print_exc()

    def _detect_objects_roboflow(self):
        """Synchronous wrapper for initial detection (called from async context)"""
        asyncio.ensure_future(self._detect_objects_roboflow_async())

    def _save_detection_images(self, rgb_frame, targets, obstacles, containers=None):
        """
        Save input image and output image with bounding boxes.
        """
        try:
            import cv2
            from pathlib import Path

            # Create output directory
            output_dir = Path("C:/isaacsim/cobotproject/models/roboflow_detections")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save input image (raw camera capture)
            input_path = output_dir / "input_image.png"
            bgr_input = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(input_path), bgr_input)

            print(f"[IMAGE SAVE] Input image saved: {input_path}")
            print(f"[IMAGE SAVE] Image shape: {rgb_frame.shape}")
            print(f"[IMAGE SAVE] Targets to draw: {len(targets)}")
            print(f"[IMAGE SAVE] Obstacles to draw: {len(obstacles)}")
            print(f"[IMAGE SAVE] Containers to draw: {len(containers) if containers else 0}")

            # Create output image with bounding boxes
            output_image = rgb_frame.copy()

            # Draw bounding boxes for targets (green)
            for target in targets:
                bbox = target['bbox']
                cv2.rectangle(output_image,
                            (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                            (0, 255, 0), 2)
                label = f"{target['class']} {target['confidence']:.1%}"
                cv2.putText(output_image, label,
                          (bbox[0], bbox[1] - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw bounding boxes for obstacles (red)
            for obs in obstacles:
                bbox = obs['bbox']
                cv2.rectangle(output_image,
                            (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                            (255, 0, 0), 2)
                label = f"{obs['class']} {obs['confidence']:.1%}"
                cv2.putText(output_image, label,
                          (bbox[0], bbox[1] - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Draw bounding boxes for containers (blue)
            if containers:
                for container in containers:
                    bbox = container['bbox']
                    cv2.rectangle(output_image,
                                (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                (0, 0, 255), 2)
                    label = f"{container['class']} {container['confidence']:.1%}"
                    cv2.putText(output_image, label,
                              (bbox[0], bbox[1] - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Save output image
            output_path = output_dir / "output_with_bboxes.png"
            bgr_output = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), bgr_output)

            print(f"[IMAGE SAVE] Output image with bboxes saved: {output_path}")

        except Exception as e:
            print(f"[DETECTION ERROR] Failed to save images: {e}")
            import traceback
            traceback.print_exc()

    # REMOVED: _project_world_to_image() - No longer needed with simplified detection logic

    def _is_cube_detected_as_target(self, cube_position):
        """
        Check if a cube at the given position is detected as a target by YOLOv8.

        SIMPLIFIED: Since Femto camera has complex angle, we just check if targets were detected.
        If yes, all small cubes are considered targets (YOLOv8 filters out obstacles already).

        Args:
            cube_position: np.array([x, y, z]) - world position of the cube

        Returns:
            bool: True if cube is detected as a target, False otherwise
        """
        if not self.detection_enabled:
            return True  # In manual mode, pick all cubes

        # Simplified: If YOLOv8 detected any targets, assume all small cubes are targets
        # YOLOv8 already filters out obstacles (cuboid) vs targets (cube)
        if len(self.detected_targets) > 0:
            return True
        else:
            print(f"[MATCH DEBUG] No targets detected by YOLOv8, skipping all cubes")
            return False



    def _physics_step_callback(self, step_size):
        """Physics step callback for sensor updates and obstacle movement"""
        if not hasattr(self, '_callback_debug_printed'):
            self._callback_debug_printed = True

        if self.obstacle_1_moving:
            self._move_obstacle()

        if not hasattr(self, '_sensor_log_counter'):
            self._sensor_log_counter = 0

        self._sensor_log_counter += 1
        if self._sensor_log_counter >= 60:
            self._sensor_log_counter = 0
            if self.lidar is not None:
                self._process_lidar_data()

        if not hasattr(self, '_detection_counter'):
            self._detection_counter = 0

        self._detection_counter += 1
        if self._detection_counter >= 120:
            self._detection_counter = 0
            if self.detection_enabled:
                asyncio.ensure_future(self._detect_objects_roboflow_async())

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

            # OBJECT DETECTION MODE: Use detected positions directly
            if self.detection_enabled:
                print("\n[OBJECT DETECTION MODE] Using Roboflow 2D object detection")

                # Run initial detection (await async version)
                await self._detect_objects_roboflow_async()

                if len(self.detected_targets) == 0:
                    print("[DETECTION] No targets detected - stopping")
                    return

                # Pick detected targets using ground truth positions
                # (2D detection only classifies objects, actual positions from USD prims)
                total_targets = len(self.detected_targets)
                total_cubes = len(self.cubes)  # Use ground truth cube count

                print(f"\n[PICKING STRATEGY] Using ground truth positions for {total_cubes} cubes")
                print(f"[DETECTION] Detected {total_targets} targets via Roboflow")

                for i in range(self.current_cube_index, total_cubes):
                    try:
                        # Get ground truth cube position
                        cube_obj, cube_name = self.cubes[i]
                        cube_position, _ = cube_obj.get_world_pose()
                        target_number = i + 1

                        # Update current cube index
                        self.current_cube_index = i

                        print(f"[{target_number}/{total_cubes}] Picking {cube_name}")
                        print(f"  Ground truth pos: {cube_position}")

                        # Call pick and place with ground truth position
                        success, error_msg = await self._pick_and_place_cube(cube_obj, cube_name)

                        if success:
                            self.placed_count += 1
                            print(f"OK")
                        else:
                            print(f"SKIP: {error_msg}")
                        self._update_status(f"{self.placed_count}/{total_cubes} placed")

                        self.current_cube_index += 1

                    except Exception as target_error:
                        # If there's an error with this target, skip it and continue to next
                        print(f"ERROR: {str(target_error)}")
                        import traceback
                        traceback.print_exc()
                        self._update_status(f"{self.placed_count}/{total_cubes} placed")
                        self.current_cube_index += 1
                        continue
            else:
                # MANUAL MODE: Pick all cubes using ground truth
                print("\n[MANUAL MODE] No object detection - picking all cubes")
                cubes = self.cubes
                total_cubes = len(cubes)

                for i in range(self.current_cube_index, total_cubes):
                    try:
                        cube, cube_name = cubes[i]
                        cube_number = i + 1

                        # Update current cube index
                        self.current_cube_index = i

                        print(f"[{cube_number}/{total_cubes}] {cube_name}")

                        # Call pick and place (retry logic is now INSIDE the function)
                        success, error_msg = await self._pick_and_place_cube(cube, cube_name.split()[1])

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

    async def _pick_and_place_detected_target(self, world_pos, target_name):
        """Pick and place using detected position (no ground truth cube object)"""
        try:
            total_cubes = self.grid_length * self.grid_width

            # Use detected position directly
            cube_size = 0.0515
            cube_half = cube_size / 2.0
            orientation = euler_angles_to_quats(np.array([np.pi, 0, 0]))
            max_pick_attempts = 3
            pick_success = False

            for pick_attempt in range(1, max_pick_attempts + 1):
                if pick_attempt > 1:
                    print(f"  Retry {pick_attempt}/{max_pick_attempts}")

                # Use detected world position
                cube_pos_current = world_pos

                pre_pick_height = 0.10 if total_cubes <= 4 else (0.12 if total_cubes <= 9 else 0.15)
                pre_pick_pos = cube_pos_current + np.array([0.0, 0.0, pre_pick_height])

                # Phase 1: Pre-pick (open gripper BEFORE approaching)
                articulation_controller = self.franka.get_articulation_controller()
                articulation_controller.apply_action(ArticulationAction(
                    joint_positions=self.gripper.joint_opened_positions, joint_indices=np.array([7, 8])))
                for _ in range(2):
                    await omni.kit.app.get_app().next_update_async()

                robot_base_translation, robot_base_orientation = self.franka.get_world_pose()
                self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
                _, ik_success = self.articulation_kinematics_solver.compute_inverse_kinematics(pre_pick_pos, orientation)

                if not ik_success:
                    if pick_attempt < max_pick_attempts:
                        continue
                    return False, f"{target_name} out of reach"

                success = await self._move_to_target_rrt(pre_pick_pos, orientation, skip_factor=4)
                if not success:
                    if pick_attempt < max_pick_attempts:
                        continue
                    return False, f"Failed pre-pick for {target_name}"

                # Phase 2: Pick approach (gripper already open)
                pick_pos = cube_pos_current + np.array([0.0, 0.0, cube_half])
                success = await self._move_to_target_rrt(pick_pos, orientation, skip_factor=6)
                if not success:
                    if pick_attempt < max_pick_attempts:
                        continue
                    return False, f"Failed pick approach for {target_name}"

                # Phase 3: Close gripper
                articulation_controller.apply_action(ArticulationAction(
                    joint_positions=self.gripper.joint_closed_positions, joint_indices=np.array([7, 8])))
                for _ in range(8):
                    await omni.kit.app.get_app().next_update_async()

                # Phase 4: Lift cube
                lift_pos = pick_pos + np.array([0.0, 0.0, 0.15])
                success = await self._move_to_target_rrt(lift_pos, orientation, skip_factor=6)
                if not success:
                    if pick_attempt < max_pick_attempts:
                        continue
                    return False, f"Failed lift for {target_name}"

                # Check if cube was picked (simplified - no ground truth check)
                pick_success = True
                break

            if not pick_success:
                return False, f"Failed to pick {target_name}"

            # Phase 5: Place approach
            container_pos, _ = self.container.get_world_pose()
            place_height = 0.15
            place_approach_pos = container_pos + np.array([0.0, 0.0, place_height])

            success = await self._move_to_target_rrt(place_approach_pos, orientation, skip_factor=4)
            if not success:
                return False, f"Failed place approach for {target_name}"

            # Phase 6: Place down
            place_pos = container_pos + np.array([0.0, 0.0, 0.10])
            success = await self._move_to_target_rrt(place_pos, orientation, skip_factor=6)
            if not success:
                return False, f"Failed place for {target_name}"

            # Open gripper to release
            articulation_controller.apply_action(ArticulationAction(
                joint_positions=self.gripper.joint_opened_positions, joint_indices=np.array([7, 8])))
            for _ in range(8):
                await omni.kit.app.get_app().next_update_async()

            # Phase 7: Place retreat
            current_ee_pos, _ = self.franka.end_effector.get_world_pose()
            retreat_pos = current_ee_pos + np.array([0.0, 0.0, 0.15])
            retreat_success = await self._move_to_target_rrt(retreat_pos, orientation, skip_factor=6)

            articulation_controller.apply_action(ArticulationAction(
                joint_positions=self.gripper.joint_closed_positions, joint_indices=np.array([7, 8])))
            for _ in range(2):
                await omni.kit.app.get_app().next_update_async()

            return True, None

        except Exception as e:
            print(f"ERROR in pick_and_place_detected_target: {e}")
            import traceback
            traceback.print_exc()
            return False, f"Exception: {str(e)}"

    async def _pick_and_place_cube(self, cube, cube_name):
        """Pick and place cube using RRT (8 phases: pick with retry, place, return home)"""
        try:
            total_cubes = self.grid_length * self.grid_width

            # TODO: Get cube size from YOLOv8 3D detection when depth camera is working
            # For now, using hardcoded size (0.0515m = 5.15cm)
            # When depth is enabled, use: detected_targets[i]['size'] from YOLOv8
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
                    if pick_attempt < max_pick_attempts:
                        continue
                    return False, f"{cube_name} out of reach"

                success = await self._move_to_target_rrt(pre_pick_pos, orientation, skip_factor=4)
                if not success:
                    if pick_attempt < max_pick_attempts:
                        continue
                    return False, f"Failed pre-pick for {cube_name}"

                # Phase 2: Pick approach (gripper already open)
                cube_pos_realtime, _ = cube.get_world_pose()
                pick_pos = np.array([cube_pos_realtime[0], cube_pos_realtime[1], cube_pos_realtime[2]])

                # Moderate speed approach with skip_factor=3 for good balance
                success = await self._move_to_target_rrt(pick_pos, orientation, skip_factor=3)
                if not success:
                    if pick_attempt < max_pick_attempts:
                        continue
                    return False, f"Failed pick approach for {cube_name}"

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
                    print(f" Pick OK ({height_lifted*100:.1f}cm)")
                    pick_success = True
                    break
                else:
                    print(f" Pick fail ({height_lifted*100:.1f}cm)")
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
            await self._move_to_target_rrt(release_height, orientation, skip_factor=4)
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

            if placement_successful:
                print(f" Place OK ({xy_distance*100:.1f}cm)")
            else:
                print(f" Place fail ({xy_distance*100:.1f}cm)")

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


            self._update_status("Reset complete - stage cleared")

        except Exception as e:
            self._update_status(f"Error resetting: {e}")
            import traceback
            traceback.print_exc()

    async def _add_obstacle_to_scene(self):
        """Add obstacle to scene (async version for scene loading)"""
        if not self.world or not self.rrt:
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

            # Default obstacle position and size (tall cuboid for YOLOv8 testing)
            obstacle_position = np.array([0.63, 0.26, 0.11])
            obstacle_size = np.array([0.20, 0.05, 0.22])  # [length, width, height] - tall cuboid

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

            print(f"[OBSTACLE] Added obstacle_1 for YOLOv8 testing")

            # Wait for obstacle to settle
            await omni.kit.app.get_app().next_update_async()

        except Exception as e:
            print(f"[OBSTACLE] Error adding obstacle: {e}")
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

