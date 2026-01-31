"""
Franka RRT Pick and Place - Dynamic Grid with Kinematics Solver
RRT path planning with obstacle avoidance, conservative collision spheres,
dynamic grid configuration, pick retry logic, return to home after each cube.
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
from pxr import UsdPhysics, PhysxSchema, Gf
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
import carb

# RTX Lidar imports
from isaacsim.sensors.rtx import LidarRtx

# Add project root to path for local imports
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.manipulators import SingleManipulator
from src.grippers import ParallelGripper
from src.sensors import RTXLidarSensor
from src.sensors.depth_sensor import DepthCameraSensor


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

        # Obstacle_1 automatic movement with force-based physics
        self.obstacle_1_moving = False
        self.obstacle_1_force = -8.0  # -8 N on x axis (moving left initially)
        self.obstacle_1_target_velocity = -0.2  # Target velocity: -0.2 m/s (left), +0.2 m/s (right)
        self.obstacle_1_min_x = 0.2  # Range: 0.2m to 0.6m (0.4m total travel)
        self.obstacle_1_max_x = 0.6

        # RTX Lidar sensor (will be wrapped in RTXLidarSensor)
        self.lidar = None
        self.lidar_sensor = None  # RTXLidarSensor wrapper

        # Depth Camera sensor (Intel RealSense D455 on Franka hand)
        self.depth_camera = None  # DepthCameraSensor wrapper

        # Timeline
        self.timeline = omni.timeline.get_timeline_interface()

        # Task state
        self.is_picking = False
        self.placed_count = 0
        self.current_cube_index = 0  # Track which cube we're currently working on
        self.pick_place_task = None  # Async task for pick-and-place operation

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

            # Proper cleanup to avoid OmniGraph errors
            try:
                # Clear Replicator/Orchestrator if it exists
                import omni.graph.core as og
                stage = omni.usd.get_context().get_stage()
                if stage and stage.GetPrimAtPath("/Orchestrator"):
                    stage.RemovePrim("/Orchestrator")
            except:
                pass  # Ignore if Replicator is not loaded

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

            # Add RTX Lidar sensor attached to Franka
            # Position it to detect obstacles (obstacles are centered at ~10cm height, 30cm tall)
            # Obstacle range: -5cm (bottom) to 25cm (top)
            # Place Lidar at 15cm to be in middle of obstacle height range
            # Attach to robot base for stable scanning
            lidar_prim_path = f"{franka_prim_path}/lidar_sensor"

            # Create Ouster OS1 Lidar
            # Position relative to robot base: at 15cm height to detect obstacles at 10cm center height
            lidar_translation = np.array([0.0, 0.0, 0.15])  # 15cm above robot base
            lidar_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # No rotation

            self.lidar = self.world.scene.add(
                LidarRtx(
                    prim_path=lidar_prim_path,
                    name="franka_lidar",
                    translation=lidar_translation,
                    orientation=lidar_orientation,
                    config_file_name="XT32_SD10"  # HESAI XT32 (32 channels, good for indoor)
                )
            )

            # Initialize lidar and attach annotator for point cloud data
            self.lidar.initialize()
            self.lidar.attach_annotator("IsaacExtractRTXSensorPointCloudNoAccumulator")

            # Enable visualization (built-in ray visualization)
            self.lidar.enable_visualization()

            # Create RTXLidarSensor wrapper for obstacle detection and visualization
            self.lidar_sensor = RTXLidarSensor(
                lidar=self.lidar,
                robot_prim=self.franka,
                world=self.world,
                rrt_planner=self.rrt,
                enable_visualization=True,
                visual_update_interval=3
            )

            # Container dimensions will be set after container is created (see below)

            # Initialize Depth Camera on Franka hand (Intel RealSense D455)
            print("[DEPTH CAMERA] Initializing Intel RealSense D455 on Franka hand...")
            depth_camera_parent_path = f"{franka_prim_path}/panda_hand"  # Use dynamic Franka path
            print(f"[DEPTH CAMERA] Parent path: {depth_camera_parent_path}")

            self.depth_camera = DepthCameraSensor(
                world=self.world,
                parent_prim_path=depth_camera_parent_path,  # Attach to Franka hand (dynamic path)
                rrt_planner=self.rrt,
                use_asset=True,
                asset_name="rsd455",
                position=np.array([0.0, 0.0, 0.05]),  # 5cm above hand center
                orientation=np.array([0.0, 90.0, 0.0]),  # Point forward from hand
                resolution=(640, 480),
                frequency=10,  # 10 Hz update rate
                min_distance=0.1,  # 10cm minimum
                max_distance=1.5,  # 1.5m maximum
                enable_obstacle_detection=True
            )

            # Initialize the depth camera
            if self.depth_camera.initialize():
                print("[DEPTH CAMERA] Intel RealSense D455 initialized successfully")
                print(f"[DEPTH CAMERA] Info: {self.depth_camera.get_sensor_info()}")

                # Attach additional annotators for visualization
                self.depth_camera.attach_additional_annotators()
                print("[DEPTH CAMERA] Attached additional annotators (rgb, normals, distance_to_image_plane)")

                # Show live depth view in Isaac Sim viewport
                self.depth_camera.show_in_viewport()
                print("[DEPTH CAMERA] Live viewport display enabled")

                # Create debug window for depth visualization
                self.depth_camera.create_debug_window()
                print("[DEPTH CAMERA] Debug window created")

            else:
                print("[DEPTH CAMERA] Failed to initialize depth camera")
                self.depth_camera = None

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

            # Now set container dimensions for Lidar filtering (after container is created)
            if self.lidar_sensor is not None:
                self.lidar_sensor.set_container_dimensions(self.container_dimensions)
                print(f"[LIDAR] Container dimensions set: {self.container_dimensions}")

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

            # Single update after physics initialization
            await omni.kit.app.get_app().next_update_async()

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

            # Wait for robot to settle (reduced from 10 to 5 frames)
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
        Process Lidar point cloud data to detect obstacles in real-time.
        Returns list of detected obstacle positions in world coordinates.

        This is a wrapper around RTXLidarSensor.process_point_cloud()
        """
        if self.lidar_sensor is None:
            return []

        try:
            return self.lidar_sensor.process_point_cloud()
        except Exception as e:
            carb.log_warn(f"[LIDAR ERROR] Error processing Lidar data: {e}")
            return []

    # OLD LIDAR PROCESSING CODE REMOVED - Now handled by RTXLidarSensor module
    # See src/sensors/rtx_lidar.py for the implementation

    def _update_lidar_visuals_only(self):
        """
        Update ONLY Lidar visuals (rays, points) without affecting RRT planner.
        This is for real-time visual feedback during non-planning phases.

        Wrapper around RTXLidarSensor.update_visuals_only()
        """
        if self.lidar_sensor is None:
            return

        self.lidar_sensor.update_visuals_only()

    def _periodic_visual_update(self):
        """
        Periodically update visuals based on frame counter.
        Call this during every frame to ensure smooth visual updates.

        Wrapper around RTXLidarSensor.periodic_visual_update()
        """
        if self.lidar_sensor is None:
            return

        self.lidar_sensor.periodic_visual_update()

    def _move_obstacle(self):
        """
        Move Obstacle_1 back and forth between x=-0.6 and x=0.6 using force-based control.
        Uses Physics Force API with F = ±29.7 N on x axis.
        Target velocity: ±5.5 m/s, reverses at boundaries.
        This is called during simulation updates for dynamic obstacle testing.
        """
        if not self.obstacle_1_moving:
            return

        # Find Obstacle_1 in obstacles dictionary
        obstacle_1 = None
        for name, obs in self.obstacles.items():
            if "obstacle_1" in name.lower() or obs.prim_path.endswith("Obstacle_1"):
                obstacle_1 = obs
                break

        if obstacle_1 is None:
            return

        try:
            # Get current position
            current_pos, _ = obstacle_1.get_world_pose()

            # Get current velocity
            try:
                current_velocity = obstacle_1.get_linear_velocity()
            except:
                current_velocity = np.array([0.0, 0.0, 0.0])

            # Get USD prim for force application
            stage = omni.usd.get_context().get_stage()
            obstacle_prim = stage.GetPrimAtPath(obstacle_1.prim_path)

            # Check boundaries and reverse force if needed
            if current_pos[0] >= self.obstacle_1_max_x and current_velocity[0] > 0:
                # Hit right boundary, reverse to move left
                self.obstacle_1_force = -8.0  # Force in -X direction
                self.obstacle_1_target_velocity = -0.2
                carb.log_info(f"[OBSTACLE MOVEMENT] Obstacle_1 hit right boundary at x={current_pos[0]:.2f}, reversing to left")

            elif current_pos[0] <= self.obstacle_1_min_x and current_velocity[0] < 0:
                # Hit left boundary, reverse to move right
                self.obstacle_1_force = 8.0  # Force in +X direction
                self.obstacle_1_target_velocity = 0.2
                carb.log_info(f"[OBSTACLE MOVEMENT] Obstacle_1 hit left boundary at x={current_pos[0]:.2f}, reversing to right")

            # Apply force using PhysX API
            if obstacle_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_body_api = UsdPhysics.RigidBodyAPI(obstacle_prim)

                # Apply constant force in X direction
                force_vector = Gf.Vec3f(self.obstacle_1_force, 0.0, 0.0)

                # Get or create PhysxForceAPI
                if not obstacle_prim.HasAPI(PhysxSchema.PhysxForceAPI):
                    force_api = PhysxSchema.PhysxForceAPI.Apply(obstacle_prim)
                else:
                    force_api = PhysxSchema.PhysxForceAPI(obstacle_prim)

                # Set force attribute
                force_api.GetForceAttr().Set(force_vector)
                force_api.GetForceEnabledAttr().Set(True)

        except Exception as e:
            carb.log_warn(f"[OBSTACLE MOVEMENT] Error moving Obstacle_1: {e}")

    def _update_dynamic_obstacles(self):
        """
        Update RRT planner with dynamically detected obstacles from Lidar and Depth Camera.
        Also moves Obstacle_1 if automatic movement is enabled.

        Wrapper around RTXLidarSensor.update_rrt_obstacles() and DepthCameraSensor.process_frame()
        """
        if self.rrt is None:
            return

        try:
            # Move Obstacle_1 if automatic movement is enabled
            self._move_obstacle()

            # Update RRT obstacles from Lidar
            if self.lidar_sensor is not None:
                self.lidar_sensor.update_rrt_obstacles()

            # Update RRT obstacles from Depth Camera
            if self.depth_camera is not None:
                self.depth_camera.process_frame()
                # Update debug window visualization
                self.depth_camera.update_debug_window()

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

            # If starting fresh (current_cube_index is 0), reset placed_count
            if self.current_cube_index == 0:
                self.placed_count = 0
                self._update_status("Starting...")

            # Store the task reference so we can check if it's running
            self.pick_place_task = asyncio.ensure_future(self._pick_place_loop())

    async def _pick_place_loop(self):
        """Main pick and place loop"""
        try:
            self.timeline.play()

            # Start automatic obstacle movement for Obstacle_1 (runs independently)
            self.obstacle_1_moving = True
            print(f"[OBSTACLE MOVEMENT] Obstacle_1 automatic movement started (x={self.obstacle_1_min_x} to x={self.obstacle_1_max_x}, "
                  f"force={self.obstacle_1_force}N, target_velocity={self.obstacle_1_target_velocity}m/s, Force-based)")

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

                    # Update Lidar visuals before each cube (for real-time visual feedback)
                    self._process_lidar_data()

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

            # Stop automatic obstacle movement
            self.obstacle_1_moving = False
            print("[OBSTACLE MOVEMENT] Obstacle_1 automatic movement stopped")

            # Clear selection to avoid "invalid prim" errors when stopping
            selection = omni.usd.get_context().get_selection()
            selection.clear_selected_prim_paths()

            # Small delay before stopping to allow cleanup
            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            # Clean up Orchestrator to prevent OmniGraph errors
            try:
                stage = omni.usd.get_context().get_stage()
                if stage and stage.GetPrimAtPath("/Orchestrator"):
                    stage.RemovePrim("/Orchestrator")
            except:
                pass

            self.timeline.stop()

        except Exception as e:
            self._update_status(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.is_picking = False

            # Stop obstacle movement on error
            self.obstacle_1_moving = False

            # Clean up Orchestrator even on error
            try:
                stage = omni.usd.get_context().get_stage()
                if stage and stage.GetPrimAtPath("/Orchestrator"):
                    stage.RemovePrim("/Orchestrator")
            except:
                pass

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
                    # On retry, move to safe joint configuration first to escape invalid configuration
                    # Use DIRECT JOINT CONTROL (not RRT) to escape collision state
                    try:
                        print(f"  Moving to safe joint configuration before retry...")
                        # Safe joint configuration: arm up and away from obstacles
                        safe_joint_positions = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785])  # Home-like position

                        articulation_controller = self.franka.get_articulation_controller()
                        for _ in range(30):  # Move slowly to safe position (30 frames)
                            articulation_controller.apply_action(ArticulationAction(
                                joint_positions=safe_joint_positions, joint_indices=np.array([0, 1, 2, 3, 4, 5, 6])))
                            await omni.kit.app.get_app().next_update_async()
                    except Exception as e:
                        print(f"  Safe joint move failed: {e}")
                        pass  # If safe move fails, continue anyway

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
                        # Try to recover by moving to intermediate safe position
                        print(f"  Pre-pick failed, attempting recovery...")
                        try:
                            intermediate_pos = np.array([0.35, cube_pos_current[1], 0.35])
                            await self._move_to_target_rrt(intermediate_pos, orientation, skip_factor=5)
                        except:
                            pass
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

                for i in range(5):  # Pick stabilization (increased from 3 for better alignment)
                    self._periodic_visual_update()  # Update visuals every frame
                    await omni.kit.app.get_app().next_update_async()

                articulation_controller.apply_action(ArticulationAction(
                    joint_positions=self.gripper.joint_closed_positions, joint_indices=np.array([7, 8])))
                for i in range(15):  # Gripper close (increased from 12 for better grip)
                    self._periodic_visual_update()  # Update visuals every frame
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
            await self._move_to_target_rrt(release_height, orientation, skip_factor=4)
            for _ in range(3):  # Place stabilization (increased from 1 to prevent cube throw)
                self._periodic_visual_update()  # Update visuals every frame
                await omni.kit.app.get_app().next_update_async()

            articulation_controller = self.franka.get_articulation_controller()
            articulation_controller.apply_action(ArticulationAction(
                joint_positions=self.gripper.joint_opened_positions, joint_indices=np.array([7, 8])))
            for _ in range(12):  # Gripper open (increased from 10 to prevent cube throw)
                self._periodic_visual_update()  # Update visuals every frame
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
            current_ee_pos, _ = self.franka.end_effector.get_world_pose()
            retreat_pos = current_ee_pos + np.array([0.0, 0.0, 0.15])  # Retreat up
            retreat_success = await self._move_to_target_rrt(retreat_pos, orientation, skip_factor=6)

            articulation_controller.apply_action(ArticulationAction(
                joint_positions=self.gripper.joint_closed_positions, joint_indices=np.array([7, 8])))
            for _ in range(2):  # Gripper close
                await omni.kit.app.get_app().next_update_async()

            # Move to a safe intermediate position using RRT to avoid obstacles
            # Safe height: 0.45m (high enough to clear obstacles and container)
            safe_ee_position = np.array([0.40, 0.0, 0.45])  # Centered position, safe height
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
        num_lidar_obstacles = len(self.lidar_sensor.detected_obstacles) if self.lidar_sensor else 0
        has_obstacles = self.obstacle_counter > 0 or num_lidar_obstacles > 0
        # Increase max iterations significantly when obstacles detected
        max_iterations = 20000 if has_obstacles else (8000 if total_cubes <= 4 else 12000)
        self.rrt.set_max_iterations(max_iterations)

        rrt_plan = self.rrt.compute_path(start_pos, np.array([]))

        if rrt_plan is None or len(rrt_plan) <= 1:
            carb.log_warn(f"RRT failed for {target_position} (obstacles: {num_lidar_obstacles})")
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
                # Update visuals periodically during motion for real-time feedback
                self._periodic_visual_update()
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
        """Reset button callback"""
        try:
            self.is_picking = False

            # Stop obstacle movement
            self.obstacle_1_moving = False

            # Stop timeline first
            self.timeline.stop()

            # STEP 1: Cleanup Lidar sensor (uses RTXLidarSensor.cleanup())
            if self.lidar_sensor is not None:
                try:
                    self.lidar_sensor.cleanup()
                    print("[RESET] Cleaned up Lidar sensor and obstacles")
                except Exception as e:
                    print(f"[RESET] Error cleaning up Lidar sensor: {e}")

            # STEP 2: Properly cleanup Lidar hardware (child to parent order)
            import omni.usd
            import omni.kit.commands
            stage = omni.usd.get_context().get_stage()

            # Clear Lidar hardware AFTER sensor cleanup
            if self.lidar is not None:
                try:
                    lidar_prim_path = self.lidar.prim_path

                    # Disable visualization first
                    try:
                        self.lidar.disable_visualization()
                    except:
                        pass

                    # Detach annotators
                    try:
                        self.lidar.detach_all_annotators()
                    except:
                        pass

                    # Remove from scene
                    if self.world is not None and self.world.scene is not None:
                        try:
                            self.world.scene.remove_object("franka_lidar")
                            print("[RESET] Removed Lidar from scene")
                        except Exception as e:
                            print(f"[RESET] Error removing Lidar from scene: {e}")

                    # Delete Lidar USD prim and all children (camera, etc.)
                    if stage and stage.GetPrimAtPath(lidar_prim_path).IsValid():
                        lidar_prim = stage.GetPrimAtPath(lidar_prim_path)
                        # Collect all child prims first
                        child_paths = [str(child.GetPath()) for child in lidar_prim.GetAllChildren()]
                        # Delete children first, then parent
                        if child_paths:
                            omni.kit.commands.execute('DeletePrims', paths=child_paths)
                            print(f"[RESET] Deleted {len(child_paths)} Lidar child prims")
                        # Now delete parent
                        omni.kit.commands.execute('DeletePrims', paths=[lidar_prim_path])
                        print(f"[RESET] Deleted Lidar prim: {lidar_prim_path}")

                    # Set to None to release reference
                    self.lidar = None
                    self.lidar_sensor = None
                except Exception as e:
                    print(f"[RESET] Error cleaning up Lidar: {e}")

            # STEP 3: Clear World instance
            if self.world is not None:
                World.clear_instance()

            # Clear stage manually
            from omni.isaac.core.utils.stage import clear_stage

            # Delete any remaining prims under /World
            if stage:
                world_prim = stage.GetPrimAtPath("/World")
                if world_prim.IsValid():
                    children_paths = [str(child.GetPath()) for child in world_prim.GetAllChildren()]
                    if children_paths:
                        omni.kit.commands.execute('DeletePrims', paths=children_paths)

            clear_stage()

            # Reset state
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
            self.lidar_sensor = None
            self.placed_count = 0
            self.current_cube_index = 0  # Reset cube index

            # Reset UI
            self.load_btn.enabled = True
            self.pick_btn.enabled = False
            self.reset_btn.enabled = False
            self.add_obstacle_btn.enabled = False
            self.remove_obstacle_btn.enabled = False

            print("[RESET] Reset complete - all sensors and cameras cleared")
            self._update_status("Reset complete")

        except Exception as e:
            self._update_status(f"Error resetting: {e}")
            import traceback
            traceback.print_exc()

    def _on_add_obstacle(self):
        """Add obstacle button callback - launches async coroutine"""
        if not self.world or not self.rrt:
            self._update_status("Load scene first!")
            return

        # Check if pick-and-place is running
        if self.pick_place_task and not self.pick_place_task.done():
            self._update_status("Cannot add obstacle while pick-and-place is running! Stop first.")
            carb.log_warn("[OBSTACLE] Cannot add obstacle during pick-and-place operation")
            return

        # Run async version
        run_coroutine(self._add_obstacle_async())

    async def _add_obstacle_async(self):
        """Add obstacle asynchronously to avoid UI freeze"""
        try:
            self._update_status("Adding obstacle...")

            # Stop timeline if running to avoid physics invalidation
            timeline = omni.timeline.get_timeline_interface()
            was_playing = timeline.is_playing()
            if was_playing:
                timeline.stop()
                # Wait for physics to settle
                for _ in range(5):
                    await omni.kit.app.get_app().next_update_async()

            # Generate unique obstacle name
            self.obstacle_counter += 1
            obstacle_name = f"obstacle_{self.obstacle_counter}"

            # Generate unique prim path
            obstacle_prim_path = find_unique_string_name(
                initial_name=f"/World/Obstacle_{self.obstacle_counter}",
                is_unique_fn=lambda x: not is_prim_path_valid(x)
            )

            # Default obstacle position (y changed from 0.3 to 0.27)
            # Scale: width=0.20, depth=0.05, height=0.22 (reverted to original depth)
            obstacle_position = np.array([0.60, 0.27, 0.11])  # Z adjusted for new height (0.22/2)
            obstacle_orientation = euler_angles_to_quats(np.array([0.0, 0.0, 0.0]))
            obstacle_scale = np.array([0.20, 0.05, 0.22])  # Width=0.20, Depth=0.05, Height=0.22

            # Create obstacle as a DynamicCuboid (Rigid Body with collider)
            # No mass specified - will use force-based movement instead
            obstacle = self.world.scene.add(
                DynamicCuboid(
                    name=obstacle_name,
                    prim_path=obstacle_prim_path,
                    position=obstacle_position,
                    orientation=obstacle_orientation,
                    size=1.0,
                    scale=obstacle_scale,
                    color=np.array([0.0, 0.0, 1.0]),
                    linear_velocity=np.array([0.0, 0.0, 0.0])  # Start stationary
                )
            )

            # Set physics properties: Rigid Body with Convex Hull collider
            stage = omni.usd.get_context().get_stage()
            obstacle_prim = stage.GetPrimAtPath(obstacle_prim_path)

            # Apply Rigid Body API
            if not obstacle_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(obstacle_prim)
            else:
                rigid_body_api = UsdPhysics.RigidBodyAPI(obstacle_prim)

            # Remove mass attribute - let PhysX compute from density for force-based control
            mass_api = UsdPhysics.MassAPI(obstacle_prim)
            if mass_api:
                # Clear the mass attribute if it exists
                mass_attr = mass_api.GetMassAttr()
                if mass_attr and mass_attr.IsValid():
                    mass_attr.Clear()
                    carb.log_info("[OBSTACLE CREATION] Cleared mass attribute for force-based control")

            # Set collider to Convex Hull (not Triangle Mesh)
            if not obstacle_prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(obstacle_prim)

            # Apply PhysX collision API for convex hull
            if not obstacle_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(obstacle_prim)
                # Set approximation to Convex Hull
                physx_collision_api.GetCollisionEnabledAttr().Set(True)

            # Set mesh collision to convex hull
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(obstacle_prim)
            mesh_collision_api.GetApproximationAttr().Set("convexHull")

            # Apply PhysX Force API for force-based movement (but don't enable yet)
            if not obstacle_prim.HasAPI(PhysxSchema.PhysxForceAPI):
                force_api = PhysxSchema.PhysxForceAPI.Apply(obstacle_prim)
            else:
                force_api = PhysxSchema.PhysxForceAPI(obstacle_prim)

            # Set initial force to 0 (will be enabled when "Start Pick and Place" is clicked)
            initial_force = Gf.Vec3f(0.0, 0.0, 0.0)
            force_api.GetForceAttr().Set(initial_force)
            force_api.GetForceEnabledAttr().Set(False)  # Disabled until movement starts

            carb.log_info("[OBSTACLE CREATION] Applied PhysX Force API (disabled until movement starts)")

            # Reset the world to initialize physics for new obstacle
            carb.log_info("[OBSTACLE] Resetting world to initialize physics for new obstacle...")

            # Make sure timeline is stopped before reset
            timeline.stop()
            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            # Reset world
            self.world.reset()

            # Wait for reset to complete and physics to reinitialize
            for _ in range(30):  # Increased wait time for physics to fully reinitialize
                await omni.kit.app.get_app().next_update_async()

            carb.log_info("[OBSTACLE] World reset complete, physics reinitialized")

            # Add obstacle to RRT planner (after reset)
            self.rrt.add_obstacle(obstacle, static=False)

            # Store obstacle reference
            self.obstacles[obstacle_name] = obstacle

            print(f"Obstacle added: {obstacle_name} (Total: {len(self.obstacles)})")
            self._update_status(f"Obstacle added ({len(self.obstacles)})")

            # Restart timeline if it was playing
            if was_playing:
                timeline.play()
                # Wait for timeline to start and physics to stabilize
                for _ in range(10):
                    await omni.kit.app.get_app().next_update_async()

            carb.log_info(f"[OBSTACLE] Successfully added {obstacle_name}. Ready for pick-and-place.")

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

