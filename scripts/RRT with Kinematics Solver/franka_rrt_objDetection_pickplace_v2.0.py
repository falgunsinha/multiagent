"""
Franka RRT Pick and Place with Object Detection - v2.0 (Modular Architecture)

Features:
- Modular camera setup (FemtoCamera)
- Modular pick and place controller (RRTPickPlaceController)
- OwlV2 object detection for target classification
- PhysX Lidar for obstacle detection
- RRT path planning with obstacle avoidance
- Clean separation of concerns

Architecture:
- src/sensors/femto_camera.py: Camera setup and data acquisition
- src/sensors/maskrcnn_detector.py: Object detection using MaskRCNN
- src/sensors/physx_lidar_sensor.py: Lidar processing
- src/sensors/collision_checker.py: Collision checking
- src/controllers/rrt_pick_place_controller.py: Pick and place logic
- src/manipulators/single_manipulator.py: Robot control
- src/grippers/parallel_gripper.py: Gripper control
"""

import time
import numpy as np
import os
from pathlib import Path
import sys
import omni.ui as ui
from omni.kit.async_engine import run_coroutine
import omni.timeline
import omni.usd

# Isaac Sim imports
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import PathPlannerVisualizer
from isaacsim.robot_motion.motion_generation.lula import RRT
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from isaacsim.robot_motion.motion_generation.lula.trajectory_generator import LulaCSpaceTrajectoryGenerator
from pxr import UsdPhysics, UsdGeom
from isaacsim.sensors.physx import RotatingLidarPhysX
import carb

# Add project root to path
project_root = Path(r"C:\isaacsim\cobotproject")
sys.path.append(str(project_root))

# Import modular components
from src.manipulators import SingleManipulator
from src.grippers import ParallelGripper
from src.sensors import FemtoCamera, PhysXLidarSensor, CollisionChecker
from src.sensors.custom_maskrcnn_detector import CustomMaskRCNNDetector  # Custom trained detector
from src.controllers.rrt_pick_place_controller import RRTPickPlaceController


class FrankaObjectDetectionPickPlace:
    """Franka pick and place with object detection and modular architecture"""

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

        # Modular components
        self.femto_camera = None  # FemtoCamera module
        self.maskrcnn_detector = None  # MaskRCNNDetector module
        self.lidar = None  # PhysX Lidar sensor
        self.lidar_sensor = None  # PhysXLidarSensor module
        self.collision_checker = None  # CollisionChecker module
        self.pick_place_controller = None  # RRTPickPlaceController module

        # Scene objects
        self.cubes = []  # List of cube objects for ground truth
        self.obstacles = {}  # Dictionary of obstacles
        self.obstacle_counter = 0

        # Grid parameters
        self.grid_length = 2
        self.grid_width = 2

        # Container
        self.container_position = np.array([0.30, 0.50, 0.0])
        self.container_dimensions = None

        # Detection results
        self.detected_targets = []
        self.detected_obstacles = []
        self.lidar_detected_obstacles = {}  # Dictionary to store dynamically detected obstacles
        self.obstacles = {}  # Dictionary to store manually added obstacles
        self.obstacle_counter = 0  # Counter for obstacle naming
        self.detection_enabled = True

        # Timeline
        self.timeline = omni.timeline.get_timeline_interface()

        # Task state
        self.is_picking = False
        self.placed_count = 0
        self.current_target_index = 0

        # UI elements
        self.load_btn = None
        self.pick_btn = None
        self.reset_btn = None
        self.status_label = None
        self.length_field = None
        self.width_field = None

        self.build_ui()

    def build_ui(self):
        """Build UI"""
        self.window = ui.Window("Franka Object Detection Pick and Place", width=450, height=400)

        with self.window.frame:
            with ui.VStack(spacing=10):
                ui.Label("Franka - Object Detection Pick and Place",
                        alignment=ui.Alignment.CENTER,
                        style={"font_size": 18})

                ui.Spacer(height=10)

                # Grid Configuration
                with ui.CollapsableFrame("Grid Configuration", height=0):
                    with ui.VStack(spacing=5):
                        with ui.HStack(spacing=10):
                            ui.Label("Grid Length (rows):", width=150)
                            length_model = ui.SimpleIntModel(2)
                            self.length_field = ui.IntField(height=25, model=length_model)

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

                # Obstacle Buttons
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
        """Load the scene with Franka, cubes, container, and sensors"""
        try:
            # Get grid parameters
            self.grid_length = int(self.length_field.model.get_value_as_int())
            self.grid_width = int(self.width_field.model.get_value_as_int())

            # Validate grid
            if self.grid_length < 1 or self.grid_width < 1:
                self._update_status("Error: Grid dimensions must be at least 1x1")
                return
            if self.grid_length > 10 or self.grid_width > 10:
                self._update_status("Error: Grid dimensions too large (max 10x10)")
                return

            total_cubes = self.grid_length * self.grid_width
            print(f"\n[SETUP] Loading {self.grid_length}x{self.grid_width} grid ({total_cubes} cubes)")

            # Clear existing world
            self.timeline.stop()
            World.clear_instance()
            await omni.kit.app.get_app().next_update_async()

            # Create world
            self.world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
            self.world.scene.add_default_ground_plane()

            # Get stage reference for obstacle creation
            from omni.isaac.core.utils.stage import get_current_stage
            self.stage = get_current_stage()

            await omni.kit.app.get_app().next_update_async()

            # Load Franka robot
            franka_name = f"franka_{int(time.time() * 1000)}"
            franka_prim_path = f"/World/Franka_{int(time.time() * 1000)}"
            franka_usd_path = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
            robot_prim = add_reference_to_stage(usd_path=franka_usd_path, prim_path=franka_prim_path)
            robot_prim.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
            robot_prim.GetVariantSet("Mesh").SetVariantSelection("Quality")

            # Setup Femto camera (modular)
            # Camera position: overhead view of workspace
            print("[SETUP] Setting up Femto camera...")
            self.femto_camera = FemtoCamera(
                prim_path="/World/Femto",
                position=[1.06843, 0.18415, 0.34743],  # Overhead position
                rotation=[-45.0, 0.0, 90.0],  # Looking down at workspace
                scale=[0.01, 0.01, 0.01]
            )
            self.femto_camera.setup(resolution=(640, 640), focal_length=2.3)
            await omni.kit.app.get_app().next_update_async()

            # Get actual RGB camera world position AND orientation for collision checker
            rgb_camera_path = self.femto_camera.rgb_camera_prim_path
            print(f"[SETUP DEBUG] RGB camera prim path: {rgb_camera_path}")
            rgb_camera_prim = self.stage.GetPrimAtPath(rgb_camera_path)
            print(f"[SETUP DEBUG] RGB camera prim valid: {rgb_camera_prim.IsValid()}")
            if rgb_camera_prim.IsValid():
                from pxr import UsdGeom, Gf
                from scipy.spatial.transform import Rotation

                xformable = UsdGeom.Xformable(rgb_camera_prim)
                transform_matrix = xformable.ComputeLocalToWorldTransform(0)

                # Get world position
                rgb_camera_world_pos = transform_matrix.ExtractTranslation()
                rgb_camera_position = np.array([rgb_camera_world_pos[0], rgb_camera_world_pos[1], rgb_camera_world_pos[2]])

                # Get world orientation (rotation matrix)
                rotation_matrix_gf = transform_matrix.ExtractRotationMatrix()
                # Convert Gf.Matrix3d to numpy array
                rotation_matrix_np = np.array([
                    [rotation_matrix_gf[0][0], rotation_matrix_gf[0][1], rotation_matrix_gf[0][2]],
                    [rotation_matrix_gf[1][0], rotation_matrix_gf[1][1], rotation_matrix_gf[1][2]],
                    [rotation_matrix_gf[2][0], rotation_matrix_gf[2][1], rotation_matrix_gf[2][2]]
                ])

                # Convert rotation matrix to Euler angles (XYZ order, degrees)
                rot = Rotation.from_matrix(rotation_matrix_np)
                rgb_camera_orientation = rot.as_euler('xyz', degrees=True)

                print(f"[SETUP] RGB camera world position: {rgb_camera_position}")
                print(f"[SETUP] RGB camera world orientation (XYZ Euler): {rgb_camera_orientation}")
                print(f"[SETUP] RGB camera rotation matrix:\n{rotation_matrix_np}")
            else:
                # Fallback to Femto parent position and orientation
                rgb_camera_position = np.array([1.06843, 0.18415, 0.34743])
                rgb_camera_orientation = np.array([-45.0, 0.0, 90.0])
                print(f"[SETUP WARNING] Could not get RGB camera world transform, using Femto parent values")

            # Create gripper
            self.gripper = ParallelGripper(
                end_effector_prim_path=f"{franka_prim_path}/panda_rightfinger",
                joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
                joint_opened_positions=np.array([0.04, 0.04]),
                joint_closed_positions=np.array([0.0, 0.0]),
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

            # Add PhysX Lidar
            print("[SETUP] Setting up PhysX Lidar...")
            lidar_prim_path = f"{franka_prim_path}/lidar_sensor"
            lidar_translation = np.array([0.0, 0.0, 0.15])

            self.lidar = self.world.scene.add(
                RotatingLidarPhysX(
                    prim_path=lidar_prim_path,
                    name="franka_lidar",
                    translation=lidar_translation,
                    rotation_frequency=20.0,
                    fov=(360.0, 30.0),
                    resolution=(1.0, 1.0),
                    valid_range=(0.4, 100.0)
                )
            )

            await omni.kit.app.get_app().next_update_async()

            # Add container
            print("[SETUP] Adding container...")
            container_prim_path = "/World/Container"
            container_usd_path = f"{get_assets_root_path()}/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Storage/Containers/Container_I/Container_I04_160x120x64cm_PR_V_NVD_01.usd"
            add_reference_to_stage(usd_path=container_usd_path, prim_path=container_prim_path)

            scale = np.array([0.3, 0.3, 0.2])
            original_size = np.array([1.60, 1.20, 0.64])
            self.container_dimensions = original_size * scale

            self.container = self.world.scene.add(
                SingleXFormPrim(
                    prim_path=container_prim_path,
                    name="container",
                    translation=self.container_position,
                    scale=scale
                )
            )

            # Add physics to container
            stage = omni.usd.get_context().get_stage()
            container_prim = stage.GetPrimAtPath(container_prim_path)
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(container_prim)
            rigid_body_api.CreateKinematicEnabledAttr(True)
            UsdPhysics.CollisionAPI.Apply(container_prim)

            # Add cubes in grid
            print(f"[SETUP] Adding {total_cubes} cubes...")
            await self._add_cubes_grid()

            await omni.kit.app.get_app().next_update_async()

            # Initialize physics
            self.world.initialize_physics()
            self.world.reset()

            # Initialize Lidar
            self.lidar.add_depth_data_to_frame()
            self.lidar.add_point_cloud_data_to_frame()
            self.lidar.enable_visualization()

            await omni.kit.app.get_app().next_update_async()

            # Setup RRT and kinematics
            print("[SETUP] Setting up RRT planner...")
            self._setup_rrt()

            # Initialize Custom MaskRCNN detector (trained on cubes/cylinders)
            print("[SETUP] Initializing Custom MaskRCNN detector...")

            # Path to trained weights (absolute path)
            import os
            weights_path = os.path.abspath(
                os.path.join(
                    "C:/isaacsim/cobotproject/models",
                    "mask_rcnn_cube_cylinder_pytorch.pth"
                )
            )

            camera_params = self.femto_camera.get_camera_params()
            self.maskrcnn_detector = CustomMaskRCNNDetector(
                weights_path=weights_path,
                confidence_threshold=0.30,  # Lowered to 30% to detect all cubes
                device="auto",
                camera_params=camera_params,
                num_classes=4,  # background + cube + cylinder + cuboid
                save_detections=True,
                output_dir="C:/isaacsim/cobotproject/detection_results",
                femto_camera=self.femto_camera  # Pass camera for built-in transformations
            )
            self.detection_enabled = True

            # Initialize PhysXLidarSensor wrapper (modular)
            print("[SETUP] Initializing Lidar sensor wrapper...")
            self.lidar_sensor = PhysXLidarSensor(
                lidar_sensor=self.lidar,
                robot_articulation=self.franka,
                container_dimensions=self.container_dimensions,
                verbose=True
            )

            # Initialize CollisionChecker wrapper (modular)
            # Use actual RGB camera world position AND orientation (not Femto parent values)
            print("[SETUP] Initializing collision checker...")
            self.collision_checker = CollisionChecker(
                depth_camera_position=rgb_camera_position,
                depth_camera_orientation=rgb_camera_orientation,  # Use actual RGB camera world orientation
                verbose=False
            )

            # Initialize RRTPickPlaceController (modular)
            print("[SETUP] Initializing pick and place controller...")
            self.pick_place_controller = RRTPickPlaceController(
                manipulator=self.franka,
                gripper=self.gripper,
                rrt_planner=self.rrt,
                kinematics_solver=self.kinematics_solver,
                articulation_kinematics_solver=self.articulation_kinematics_solver,
                cspace_trajectory_generator=self.cspace_trajectory_generator,
                container_position=self.container_position,
                container_dimensions=self.container_dimensions,
                obstacle_update_callback=self._update_dynamic_obstacles  # Connect obstacle detection
            )

            # Configure robot
            print("[SETUP] Configuring robot...")
            self.franka.disable_gravity()
            articulation_controller = self.franka.get_articulation_controller()
            kp_gains = 1e15 * np.ones(9)
            kd_gains = 1e13 * np.ones(9)
            articulation_controller.set_gains(kp_gains, kd_gains)

            default_joint_positions = np.array([0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75, 0.0, 0.0])
            self.franka.set_joints_default_state(positions=default_joint_positions)
            self.gripper.set_default_state(self.gripper.joint_closed_positions)

            articulation_controller.apply_action(ArticulationAction(
                joint_positions=self.gripper.joint_closed_positions,
                joint_indices=np.array([7, 8])
            ))

            # Wait for robot to settle
            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            print("[SETUP] Scene loaded successfully!")

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

    async def _add_cubes_grid(self):
        """Add cubes in grid pattern"""
        cube_size = 0.0515
        total_cubes = self.grid_length * self.grid_width

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

                from isaacsim.core.api.objects import DynamicCuboid
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

                self.cubes.append((cube, f"{cube_name} ({color_name})"))
                cube_index += 1

    def _setup_rrt(self):
        """Setup RRT motion planner and kinematics solvers"""
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")

        robot_description_file = r"C:\isaacsim\cobotproject\assets\franka_conservative_spheres_robot_description.yaml"
        urdf_path = os.path.join(mg_extension_path, "motion_policy_configs", "franka", "lula_franka_gen.urdf")
        rrt_config_file = os.path.join(mg_extension_path, "path_planner_configs", "franka", "rrt", "franka_planner_config.yaml")

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

    async def _detect_objects(self):
        """Perform object detection using MaskRCNN"""
        if not self.detection_enabled or self.maskrcnn_detector is None:
            return

        try:
            # Step replicator for fresh frame
            import omni.replicator.core as rep
            await rep.orchestrator.step_async(rt_subframes=4)

            # Get RGB and depth data from Femto camera
            rgb_frame = self.femto_camera.get_rgb_data()
            if rgb_frame is None:
                print("[DETECTION] No RGB data available")
                return

            depth_frame = self.femto_camera.get_depth_data()

            # Perform 3D detection with MaskRCNN
            if depth_frame is not None:
                detections = self.maskrcnn_detector.detect_3d(rgb_frame, depth_frame, verbose=False)
            else:
                detections = self.maskrcnn_detector.detect(rgb_frame, verbose=False)

            # DEBUG: Get actual cube positions from scene for comparison
            print("\n[DEBUG] Actual cube positions in scene:")
            from omni.isaac.core.utils.stage import get_current_stage
            from pxr import UsdGeom
            stage = get_current_stage()
            for i in range(1, 10):  # Check up to 9 cubes
                cube_path = f"/World/Cube_{i}"
                cube_prim = stage.GetPrimAtPath(cube_path)
                if cube_prim.IsValid():
                    xformable = UsdGeom.Xformable(cube_prim)
                    transform_matrix = xformable.ComputeLocalToWorldTransform(0)
                    position = transform_matrix.ExtractTranslation()
                    print(f"  Cube_{i}: world_pos=[{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")

            # Classify detections into targets and obstacles
            self.detected_targets = []
            self.detected_obstacles = []

            for det in detections:
                # Filter by size - only small cubes (exclude container and obstacles)
                # Small cubes are ~5cm (0.05m), container is ~20cm (0.20m)
                if 'pos_3d' in det and 'size_3d' in det:
                    size = det['size_3d']
                    max_size = max(size)

                    # Keep only objects smaller than 12cm (0.12m)
                    if max_size < 0.12:
                        target = {
                            'class': det['label'],
                            'confidence': det['confidence'],
                            'position_3d': det['pos_3d'],
                            'size_3d': det.get('size_3d', None),
                            'bbox': det['bbox'],
                            'center': det['center']
                        }
                        self.detected_targets.append(target)
                    else:
                        print(f"[DETECTION] Filtered out large object: {det['label']} ({det['confidence']:.1%}), size={size}, max={max_size:.3f}m")

            if len(self.detected_targets) > 0:
                print(f"\n[DETECTION] Found {len(self.detected_targets)} targets")
                for i, target in enumerate(self.detected_targets):
                    cam_pos = target['position_3d']
                    pos_str = f", cam_pos={cam_pos}"
                    size_str = f", size={target['size_3d']}" if target['size_3d'] is not None else ""
                    print(f"  Target {i+1}: {target['class']} ({target['confidence']:.1%}){pos_str}{size_str}")

        except Exception as e:
            carb.log_warn(f"[DETECTION ERROR] {e}")
            import traceback
            traceback.print_exc()

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
        Update RRT planner with dynamically detected obstacles from Lidar.

        OPTIMIZED: Instead of deleting and recreating obstacles, we:
        1. Reuse existing obstacle prims by moving them
        2. Only create new prims if we need more
        3. Only delete prims if we have too many
        This avoids constant stage updates and maintains 60 FPS
        """
        if self.lidar is None or self.rrt is None:
            return

        try:
            # Get detected obstacles from Lidar
            detected_positions = self._process_lidar_data()

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

                    from omni.isaac.core.objects import FixedCuboid
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

            if self.current_target_index == 0:
                self.placed_count = 0
                self._update_status("Starting...")

            run_coroutine(self._pick_place_loop())

    async def _pick_place_loop(self):
        """Main pick and place loop using modular controller"""
        try:
            self.timeline.play()

            # Wait for robot to settle
            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()

            # Perform object detection
            print("\n[DETECTION] Running object detection...")
            await self._detect_objects()

            if len(self.detected_targets) == 0:
                print("[DETECTION] No targets detected - stopping")
                print("[NOTE] Custom MaskRCNN is trained to detect: cube, cylinder, cuboid")
                print("[NOTE] Check camera view and object visibility")
                print("[NOTE] Try lowering confidence threshold if objects are visible")
                self._update_status("No targets detected")
                self.is_picking = False
                return

            total_targets = len(self.detected_targets)
            print(f"\n[PICK AND PLACE] Starting pick and place for {total_targets} targets")

            # Pick and place each detected target
            for i in range(self.current_target_index, total_targets):
                if not self.is_picking:
                    break

                try:
                    target = self.detected_targets[i]
                    target_number = i + 1
                    self.current_target_index = i

                    # Get world position (already transformed by Isaac Sim Camera)
                    world_pos = np.array(target['position_3d'])

                    print(f"\n[{target_number}/{total_targets}] Target {target_number} ({target['class']} {target['confidence']*100:.1f}%)")
                    print(f"  World pos: {world_pos}")

                    # Update dynamic obstacles from Lidar before planning (real-time detection)
                    self._update_dynamic_obstacles()

                    # Use modular pick and place controller
                    success = await self.pick_place_controller.pick_and_place_object(
                        target_position=world_pos,
                        target_name=f"Target_{target_number}"
                    )

                    if success:
                        self.placed_count += 1
                        print(f"  OK - Placed successfully")
                    else:
                        print(f"  SKIP - Failed to pick/place")

                    self._update_status(f"{self.placed_count}/{total_targets} placed")
                    self.current_target_index += 1

                except Exception as target_error:
                    print(f"  ERROR: {str(target_error)}")
                    import traceback
                    traceback.print_exc()
                    self.current_target_index += 1
                    continue

            print(f"\n[DONE] Placed {self.placed_count}/{total_targets} targets")
            self._update_status(f"Done: {self.placed_count}/{total_targets} placed")
            self.is_picking = False
            self.current_target_index = 0

            # Return to home
            await self.pick_place_controller.return_to_home()

            # Small delay before stopping
            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            self.timeline.stop()

        except Exception as e:
            self._update_status(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.is_picking = False
            self.timeline.stop()

    def _on_reset(self):
        """Reset scene button callback"""
        self._update_status("Resetting...")
        run_coroutine(self._reset_scene())

    async def _reset_scene(self):
        """Reset the scene"""
        try:
            if self.world:
                self.timeline.stop()
                await omni.kit.app.get_app().next_update_async()
                self.world.reset()
                self.placed_count = 0
                self.current_target_index = 0
                self.is_picking = False
                self._update_status("Scene reset - Ready to pick and place")
        except Exception as e:
            self._update_status(f"Reset error: {e}")
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
            from isaacsim.core.utils.prims import is_prim_path_valid
            from isaacsim.core.utils.string import find_unique_string_name
            obstacle_prim_path = find_unique_string_name(
                initial_name=f"/World/Obstacle_{self.obstacle_counter}",
                is_unique_fn=lambda x: not is_prim_path_valid(x)
            )

            # Default obstacle position and size
            obstacle_position = np.array([0.63, 0.26, 0.11])
            obstacle_size = np.array([0.20, 0.05, 0.22])  # [length, width, height]

            # Create cube prim using prim_utils
            from isaacsim.core.utils import prims as prim_utils
            from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
            cube_prim = prim_utils.create_prim(
                prim_path=obstacle_prim_path,
                prim_type="Cube",
                position=obstacle_position,
                orientation=euler_angles_to_quats(np.array([0, 0, 0])),
                scale=obstacle_size,
                attributes={"size": 1.0}
            )

            # Set color (red)
            stage = omni.usd.get_context().get_stage()
            cube_geom = UsdGeom.Cube.Get(stage, obstacle_prim_path)
            if cube_geom:
                cube_geom.CreateDisplayColorAttr([(1.0, 0.0, 0.0)])

            # Apply RigidBodyAPI for physics (dynamic, not kinematic)
            from pxr import UsdPhysics
            if not cube_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI.Apply(cube_prim)

            rigid_body_api = UsdPhysics.RigidBodyAPI(cube_prim)

            # Set kinematic to False (dynamic rigid body)
            rigid_body_api.CreateKinematicEnabledAttr().Set(False)

            # Apply CollisionAPI
            if not cube_prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(cube_prim)

            # Set collision approximation to convex hull
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(cube_prim)
            mesh_collision_api.GetApproximationAttr().Set("convexHull")

            # Wrap in FixedCuboid for RRT
            from omni.isaac.core.objects import FixedCuboid
            obstacle = FixedCuboid(
                prim_path=obstacle_prim_path,
                name=obstacle_name
            )
            self.world.scene.add(obstacle)

            # Add to RRT planner
            self.rrt.add_obstacle(obstacle)
            self.rrt.update_world()

            # Store obstacle reference
            self.obstacles[obstacle_name] = obstacle

            print(f"[OBSTACLE] Added {obstacle_name} at {obstacle_position}")
            self._update_status(f"Added obstacle ({len(self.obstacles)} total)")

        except Exception as e:
            self._update_status(f"Add obstacle error: {e}")
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

            print(f"[OBSTACLE] Removed {obstacle_name} (Remaining: {len(self.obstacles)})")
            self._update_status(f"Obstacle removed ({len(self.obstacles)} remaining)")

        except Exception as e:
            self._update_status(f"Error removing obstacle: {e}")
            import traceback
            traceback.print_exc()


# Main entry point
if __name__ == "__main__":
    app = FrankaObjectDetectionPickPlace()

