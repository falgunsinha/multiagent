"""
UR10e RRT Pick and Place - Dynamic Grid with Kinematics Solver
RRT path planning with obstacle avoidance, conservative collision spheres,
dynamic grid configuration, pick retry logic, return to home after each cube.
Uses PhysX Lidar - Rotating and depth sensor for obstacle detection.

NEW FEATURES:
- UR10e robot with Robotiq gripper mounted on stand
- Cubes and container on table
- Smart cube obstacle avoidance: Only nearby cubes (within 30cm) are treated as obstacles
  to prevent RRT failures while still avoiding collisions
- Container collision avoidance: Container added as static obstacle to RRT
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
from isaacsim.core.api.objects import DynamicCylinder, VisualCuboid, FixedCuboid
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


class UR10eRRTDynamicGrid:
    """Dynamic grid pick and place with RRT for UR10e robot"""

    def __init__(self):
        self.window = None
        self.world = None
        self.ur10e = None
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

        # UI elements
        self.load_btn = None
        self.pick_btn = None
        self.reset_btn = None
        self.add_obstacle_btn = None
        self.remove_obstacle_btn = None
        self.status_label = None
        self.length_field = None
        self.width_field = None

        # Robot stand and table
        self.robot_stand = None
        self.table = None

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
        """Load the scene with UR10e on stand, dynamic grid of cubes on table, and container"""
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

            # Add robot stand on ground plane
            assets_root_path = get_assets_root_path()
            stand_prim_path = "/World/RobotStand"
            stand_usd_path = assets_root_path + "/Isaac/Props/Mounts/Stand/stand_instanceable.usd"
            add_reference_to_stage(usd_path=stand_usd_path, prim_path=stand_prim_path)

            # Position stand for UR10e
            # Decreased to half: Z=0.75 (was 1.5), Z scale=1.3 (was 2.6)
            stand_translation = np.array([0.0, 0.0, 0.75])
            stand_scale = np.array([1.8, 1.8, 1.3])
            self.robot_stand = self.world.scene.add(
                SingleXFormPrim(
                    prim_path=stand_prim_path,
                    name="robot_stand",
                    translation=stand_translation,
                    scale=stand_scale
                )
            )

            # Enable physics on the stand so it can be used in fixed joint
            # The stand needs to be a rigid body for the fixed joint to work
            stand_prim = stage.GetPrimAtPath(stand_prim_path)
            if stand_prim.IsValid():
                # Apply RigidBodyAPI to make it a rigid body
                if not stand_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI.Apply(stand_prim)
                    # Make it static (kinematic) so it doesn't fall
                    rigid_body = UsdPhysics.RigidBodyAPI(stand_prim)
                    rigid_body.CreateKinematicEnabledAttr(True)

                # Add collision API to the stand
                if not stand_prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(stand_prim)

                print("  [OK] Stand configured as static rigid body with collision")

            # Create fixed joint between World and RobotStand to anchor it
            # This ensures the stand stays at its specified position
            world_stand_joint_path = "/World/world_stand_fixed_joint"
            world_stand_joint = UsdPhysics.FixedJoint.Define(stage, world_stand_joint_path)
            # For World-to-object joints, only set body1 (the object)
            world_stand_joint.CreateBody1Rel().SetTargets([stand_prim_path])
            world_stand_joint.CreateBreakForceAttr().Set(float('inf'))
            world_stand_joint.CreateBreakTorqueAttr().Set(float('inf'))

            print(f"  [OK] Created fixed joint between /World and {stand_prim_path}")

            # Add table for cubes and container on ground plane
            table_prim_path = "/World/Table"
            table_usd_path = assets_root_path + "/Isaac/Environments/Simple_Room/Props/table_low.usd"
            add_reference_to_stage(usd_path=table_usd_path, prim_path=table_prim_path)

            # Position table in front of robot on ground plane with quaternion orientation
            # Use XFormPrim and disable physics to make it static
            table_top_height = 0.75  # Table top is at 75cm height

            # Quaternion for 90 degrees rotation around Z-axis: [w, x, y, z]
            # For 90 deg Z rotation: w=cos(45°)=0.7071, z=sin(45°)=0.7071, x=0, y=0
            table_orientation = np.array([0.7071, 0.0, 0.0, 0.7071])  # [w, x, y, z] for 90° Z rotation

            self.table = self.world.scene.add(
                SingleXFormPrim(
                    prim_path=table_prim_path,
                    name="table",
                    translation=np.array([0.6, 0.0, 0.0]),  # 60cm in front of robot, on ground
                    orientation=table_orientation,  # 90 degrees quaternion
                    scale=np.array([0.9, 0.9, 0.9])  # Updated scale
                )
            )

            # Make table static by setting it as a collider without rigid body dynamics
            stage = omni.usd.get_context().get_stage()
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
                from pxr import UsdShade
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

            # Add UR10e robot with Robotiq 2F-140 gripper
            # Using the same USD as the official Isaac Sim tutorial
            print("=== Adding UR10e with Robotiq 2F-140 Gripper ===")
            ur10e_prim_path = "/World/ur10e"
            ur10e_name = "ur10e_robot"

            # Use the official tutorial USD that has the gripper built-in
            # This is different from the variant-based USD
            ur10e_usd_path = assets_root_path + "/Isaac/Samples/Rigging/Manipulator/configure_manipulator/ur10e/ur/ur_gripper.usd"
            add_reference_to_stage(usd_path=ur10e_usd_path, prim_path=ur10e_prim_path)

            print(f"  [OK] Loaded UR10e from: {ur10e_usd_path}")

            # Set UR10e position BEFORE creating fixed joint
            # Robot base should have SAME translation as stand (they're connected via fixed joint)
            robot_translation = stand_translation  # Same as stand

            ur10e_prim = stage.GetPrimAtPath(ur10e_prim_path)
            if ur10e_prim.IsValid():
                xformable = UsdGeom.Xformable(ur10e_prim)
                # Don't clear xform ops - just set the values on existing ops or add if needed

                # Check if translate op exists, if not add it
                translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
                if translate_ops:
                    translate_ops[0].Set(Gf.Vec3d(robot_translation[0], robot_translation[1], robot_translation[2]))
                else:
                    translate_op = xformable.AddTranslateOp()
                    translate_op.Set(Gf.Vec3d(robot_translation[0], robot_translation[1], robot_translation[2]))

                # Check if orient op exists, if not add it with correct precision
                orient_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient]
                if orient_ops:
                    orient_ops[0].Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))  # w, x, y, z
                else:
                    orient_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
                    orient_op.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))  # w, x, y, z

                print(f"  [OK] Set UR10e position to {robot_translation}")

            # Define end effector path (same as official examples)
            end_effector_prim_path = f"{ur10e_prim_path}/ee_link/robotiq_arg2f_base_link"

            # Store paths for later gripper/manipulator creation
            self._ur10e_prim_path = ur10e_prim_path
            self._ur10e_name = ur10e_name
            self._end_effector_prim_path = end_effector_prim_path

            print(f"  [OK] UR10e USD loaded and positioned, gripper will be created after world setup")

            # Create fixed joint between UR10e base and RobotStand (like Franka example)
            # This physically mounts the robot to the stand
            ur10e_base_link_path = f"{ur10e_prim_path}/world/base_link"
            fixed_joint_path = f"{ur10e_prim_path}/world/stand_fixed_joint"

            # Create FixedJoint
            fixed_joint = UsdPhysics.FixedJoint.Define(stage, fixed_joint_path)
            fixed_joint.CreateBody0Rel().SetTargets([stand_prim_path])  # Stand is body0
            fixed_joint.CreateBody1Rel().SetTargets([ur10e_base_link_path])  # Robot base is body1
            fixed_joint.CreateBreakForceAttr().Set(float('inf'))  # Unbreakable joint
            fixed_joint.CreateBreakTorqueAttr().Set(float('inf'))

            print(f"  [OK] Created fixed joint between {stand_prim_path} and {ur10e_base_link_path}")

            # Add PhysX Lidar - Rotating sensor attached to UR10e
            # Position it to detect obstacles (obstacles are centered at ~10cm height on table, 30cm tall)
            # Obstacle range: table_height + (-5cm to 25cm)
            # Place Lidar at table height + 15cm to be in middle of obstacle height range
            # Attach to robot base for stable scanning
            lidar_prim_path = f"{ur10e_prim_path}/lidar_sensor"

            # Create PhysX Rotating Lidar
            # Position relative to robot base: at stand_height + 15cm to detect obstacles on table
            lidar_translation = np.array([0.0, 0.0, 0.15])  # 15cm above robot base (which is on stand)

            self.lidar = self.world.scene.add(
                RotatingLidarPhysX(
                    prim_path=lidar_prim_path,
                    name="ur10e_lidar",
                    translation=lidar_translation,
                    rotation_frequency=20.0,  # 20 Hz rotation
                    fov=(360.0, 30.0),  # 360 degrees horizontal, 30 degrees vertical
                    resolution=(1.0, 1.0),  # 1 degree resolution
                    valid_range=(0.4, 100.0)  # 0.4m to 100m range
                )
            )

            # Initialize Depth Camera attached to UR10e end effector
            print("[DEPTH CAMERA] Initializing depth camera on UR10e end effector...")
            depth_camera_prim_path = f"{ur10e_prim_path}/ee_link/depth_camera"

            # Position relative to ee_link
            # Translation: 5cm forward along end effector's local axis
            # Orientation: looking forward/down from the end effector
            position = np.array([0.0, 0.0, 0.05])  # 5cm above ee_link center
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

            print(f"[DEPTH CAMERA] Depth camera created at {depth_camera_prim_path} (attached to UR10e end effector)")

            # Single update after robot, Lidar, and Depth Camera setup
            await omni.kit.app.get_app().next_update_async()

            # Add container on table
            container_prim_path = "/World/Container"
            container_usd_path = f"{get_assets_root_path()}/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Storage/Containers/Container_I/Container_I04_160x120x64cm_PR_V_NVD_01.usd"
            add_reference_to_stage(usd_path=container_usd_path, prim_path=container_prim_path)

            # Position container on ground plane
            container_position = np.array([0.60, 0.40, 0.6])  # On ground plane at Z=0.6
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

            # Adjust spacing for scaled cylinders (1.5x scale will be applied via USD)
            if total_cubes <= 4:
                cylinder_spacing = 0.18 * 1.5
            elif total_cubes <= 9:
                cylinder_spacing = 0.15 * 1.5
            else:
                cylinder_spacing = 0.13 * 1.5

            # Position cylinders on table (table is at x=0.6, table top at ~0.75m)
            grid_center_x = 0.60  # Center on table
            grid_center_y = -0.15  # Offset to left side of table
            grid_extent_x = (self.grid_length - 1) * cylinder_spacing
            grid_extent_y = (self.grid_width - 1) * cylinder_spacing
            start_x = grid_center_x - (grid_extent_x / 2.0)
            start_y = grid_center_y - (grid_extent_y / 2.0)

            # Create cylinders using DynamicCylinder (like Franka uses DynamicCuboid)
            print("=== Creating Cylinders ===")
            self.cubes = []  # Still called cubes for compatibility
            cylinder_index = 0

            for row in range(self.grid_length):
                for col in range(self.grid_width):
                    cylinder_x = start_x + (row * cylinder_spacing)
                    cylinder_y = start_y + (col * cylinder_spacing)
                    # Place cylinders on ground plane at Z=0.65
                    # After 3.0x scale, cylinder height will be ~15.45cm, so center at 0.65 + half_height
                    cylinder_scale = np.array([2.0, 2.0, 3.0])
                    scaled_cylinder_height = cylinder_height * cylinder_scale[2]
                    cylinder_z = 0.65 + scaled_cylinder_height/2.0

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
                            color=np.array([0.0, 0.0, 1.0]),  # Blue
                            scale=cylinder_scale
                        )
                    )

                    # Set display name and apply metallic shader
                    prim = stage.GetPrimAtPath(prim_path)
                    if prim:
                        display_name = f"Cylinder {cylinder_number} (t{timestamp})"
                        omni.usd.editor.set_display_name(prim, display_name)

                        # Apply metallic shader to Looks/Visual_Material
                        material_prim_path = f"{prim_path}/Looks/Visual_Material"
                        material_prim = stage.GetPrimAtPath(material_prim_path)
                        if material_prim.IsValid():
                            # Get the shader
                            shader_prim_path = f"{material_prim_path}/Shader"
                            shader_prim = stage.GetPrimAtPath(shader_prim_path)
                            if shader_prim.IsValid():
                                shader = UsdShade.Shader(shader_prim)
                                # Set metallic value to 0.6
                                metallic_input = shader.CreateInput("metallic", Sdf.ValueTypeNames.Float)
                                metallic_input.Set(0.6)

                    # Store cylinder object and name
                    self.cubes.append((cylinder, f"{cylinder_name} (Blue)"))
                    cylinder_index += 1

            print(f"  [OK] Created {len(self.cubes)} cylinders")

            # Single update after creating cylinders
            await omni.kit.app.get_app().next_update_async()

            # Create gripper and manipulator BEFORE world.reset() (like Franka script)
            print("=== Creating Gripper and Manipulator ===")
            # Use src classes (same as Franka script - these work in Script Editor)
            from src.grippers import ParallelGripper
            from src.manipulators import SingleManipulator

            self.gripper = ParallelGripper(
                end_effector_prim_path=self._end_effector_prim_path,
                joint_prim_names=["finger_joint"],
                joint_opened_positions=np.array([0]),
                joint_closed_positions=np.array([40]),  # 40 degrees (NOT radians)
                action_deltas=np.array([-40]),  # Negative to close
                use_mimic_joints=True,
            )
            print(f"  [OK] Created gripper with end effector: {self._end_effector_prim_path}")

            self.ur10e = self.world.scene.add(
                SingleManipulator(
                    prim_path=self._ur10e_prim_path,
                    name=self._ur10e_name,
                    end_effector_prim_path=self._end_effector_prim_path,
                    gripper=self.gripper
                    # Position already set via USD (before fixed joint creation)
                )
            )
            print(f"  [OK] UR10e added: {self._ur10e_name} with Robotiq 2F-140 gripper")

            # Initialize physics and reset AFTER adding all objects (like official examples)
            self.world.reset()

            # Wait for physics to be fully ready after world.reset()
            await omni.kit.app.get_app().next_update_async()

            # Configure UR10e robot (after world.reset() initialization)
            print("=== Configuring UR10e ===")
            self.ur10e.disable_gravity()
            articulation_controller = self.ur10e.get_articulation_controller()

            # UR10e with Robotiq gripper has 12 total joints (6 arm + 6 gripper mimic joints)
            # Set gains for all joints
            num_joints = self.ur10e.num_dof
            kp_gains = 1e15 * np.ones(num_joints)
            kd_gains = 1e13 * np.ones(num_joints)
            articulation_controller.set_gains(kp_gains, kd_gains)

            # UR10e default joint positions for all 12 DOF (6 arm + 6 gripper mimic)
            # Arm joints: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
            # Gripper joints: handled by mimic joints (set to 0 for now)
            default_joint_positions = np.array([0.0, -1.2, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.ur10e.set_joints_default_state(positions=default_joint_positions)

            # Close gripper using gripper controller
            self.gripper.close()

            # Wait for robot to settle
            for _ in range(10):  # Increased wait time for robot to settle
                await omni.kit.app.get_app().next_update_async()

            print("  [OK] UR10e configured with gains and default positions")

            # Initialize PhysX Lidar after world reset
            self.lidar.add_depth_data_to_frame()
            self.lidar.add_point_cloud_data_to_frame()
            self.lidar.enable_visualization()

            # Configure rendering settings for depth sensor (disable DLSS/DLAA to avoid warnings)
            settings = carb.settings.get_settings()

            # Completely disable DLSS
            settings.set("/rtx/post/dlss/execMode", 0)  # 0 = Off
            settings.set("/rtx/post/dlss/enabled", False)

            # Disable all anti-aliasing
            settings.set("/rtx/post/aa/op", 0)  # 0 = Disabled
            settings.set("/rtx/post/aa/enabled", False)

            # Disable TAA (Temporal Anti-Aliasing)
            settings.set("/rtx/post/taa/enabled", False)

            # Configure depth sensor schema settings
            settings.set("/exts/omni.usd.schema.render_settings/rtx/renderSettings/apiSchemas/autoApply", None)
            settings.set("/exts/omni.usd.schema.render_settings/rtx/camera/apiSchemas/autoApply", None)
            settings.set("/exts/omni.usd.schema.render_settings/rtx/renderProduct/apiSchemas/autoApply", None)

            print("[DEPTH CAMERA] Rendering settings configured (DLSS/DLAA/TAA disabled)")

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

            print("[DEPTH CAMERA] Depth camera initialized on UR10e end effector")
            print("[DEPTH CAMERA] Attached annotators:")
            print("  - DepthSensorDistance (distance measurements)")
            print("  - DepthSensorPointCloudPosition (3D point cloud)")
            print("  - DepthSensorPointCloudColor (point cloud colors)")
            print("[DEPTH CAMERA] Position: 5cm above ee_link, looking forward/down")
            print("[DEPTH CAMERA] Resolution: 512x512 (square), Frequency: 10 Hz")
            print("[DEPTH CAMERA] Depth range: 0.1m - 2.0m, Baseline: 55mm")
            print("[DEPTH CAMERA] Sensor size: 1280 pixels (standard depth sensor parameter)")

            # Setup RRT after robot is configured
            self._setup_rrt()

            # Initialize robot base pose for kinematics solver AFTER robot has settled
            robot_base_translation, robot_base_orientation = self.ur10e.get_world_pose()
            self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
            self.rrt.set_robot_base_pose(robot_base_translation, robot_base_orientation)
            print(f"  [OK] Initialized kinematics solver with robot base at {robot_base_translation}")

            # Add container as a permanent obstacle for collision avoidance
            self._add_container_as_obstacle()

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
        """Setup RRT motion planner and kinematics solvers for UR10e"""
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")

        # Use standalone example's robot_descriptor.yaml which has collision spheres for gripper
        # This is similar to Franka using conservative_spheres_robot_description.yaml
        script_dir = os.path.dirname(os.path.abspath(__file__))
        standalone_examples_path = os.path.join(script_dir, "..", "..", "..", "standalone_examples", "api", "isaacsim.robot.manipulators", "ur10e", "rmpflow")

        # UR10e configuration files (like Franka pattern)
        robot_description_file = os.path.join(standalone_examples_path, "robot_descriptor.yaml")
        urdf_path = os.path.join(standalone_examples_path, "ur10e.urdf")
        rrt_config_file = os.path.join(mg_extension_path, "path_planner_configs", "universal_robots", "ur10e", "rrt", "ur10e_planner_config.yaml")

        # Verify files exist
        if not os.path.exists(robot_description_file):
            print(f"[WARNING] robot_descriptor.yaml not found at {robot_description_file}")
            print(f"[WARNING] Falling back to default ur10e_robot_description.yaml")
            robot_description_file = os.path.join(mg_extension_path, "motion_policy_configs", "universal_robots", "ur10e", "rmpflow", "ur10e_robot_description.yaml")
            urdf_path = os.path.join(mg_extension_path, "motion_policy_configs", "universal_robots", "ur10e", "ur10e.urdf")

        # Create RRT planner with UR10e configuration
        self.rrt = RRT(robot_description_path=robot_description_file, urdf_path=urdf_path,
                       rrt_config_path=rrt_config_file, end_effector_frame_name="tool0")
        self.rrt.set_max_iterations(10000)  # Balanced for quality vs speed

        self.path_planner_visualizer = PathPlannerVisualizer(robot_articulation=self.ur10e, path_planner=self.rrt)
        self.kinematics_solver = LulaKinematicsSolver(robot_description_path=robot_description_file, urdf_path=urdf_path)
        self.articulation_kinematics_solver = ArticulationKinematicsSolver(
            self.ur10e, self.kinematics_solver, "tool0")

        self.cspace_trajectory_generator = LulaCSpaceTrajectoryGenerator(robot_description_file, urdf_path)

        # NOTE: Robot base pose will be initialized after robot settles (in _load_scene)

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
        """
        if self.cube_obstacles_enabled:
            return

        try:
            added_count = 0
            # Reduced collision radius to allow RRT to plan around obstacles better
            # Only add cubes very close to target (within gripper path)
            collision_radius = 0.20  # Reduced from 0.30 to 0.20 (20cm)

            for i, (cube, _) in enumerate(self.cubes):
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

            self.cube_obstacles_enabled = True

        except Exception as e:
            print(f"[ERROR] Failed to add cubes as obstacles: {e}")
            import traceback
            traceback.print_exc()

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
            robot_pos, _ = self.ur10e.get_world_pose()
            distances_from_robot = np.linalg.norm(valid_points[:, :2] - robot_pos[:2], axis=1)
            valid_points = valid_points[(distances_from_robot > 0.30) & (distances_from_robot < 0.90)]  # 30cm-90cm only

            # Filter out cube pickup region on table (tighter bounds to avoid blocking obstacles)
            cube_grid_center = np.array([0.60, -0.15])  # Updated to match table position
            cube_grid_margin = 0.30  # Reduced from 0.35 to 0.30 for tighter filtering
            cube_region_mask = ~((np.abs(valid_points[:, 0] - cube_grid_center[0]) < cube_grid_margin) &
                                 (np.abs(valid_points[:, 1] - cube_grid_center[1]) < cube_grid_margin))
            valid_points = valid_points[cube_region_mask]

            # Filter out container/placement region on table (tighter bounds to avoid blocking obstacles)
            if self.container_dimensions is not None:
                container_pos = np.array([0.60, 0.40])  # Updated to match table position
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

            # CRITICAL: Update RRT's internal representation after moving obstacle
            # This ensures RRT knows the current position of Obstacle_1 for collision avoidance
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

    async def _pick_place_loop(self):
        """Main pick and place loop"""
        try:
            self.timeline.play()

            # Add physics callback now that timeline is playing (physics context is ready)
            if not hasattr(self, '_physics_callback_added'):
                try:
                    self.world.add_physics_callback("sensor_and_obstacle_update", self._physics_step_callback)
                    self._physics_callback_added = True
                    print("[PHYSICS] Physics step callback added successfully")
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
            self.gripper.close()
            for _ in range(8):  # Reduced for better FPS
                await omni.kit.app.get_app().next_update_async()

            # IMPROVED: Explicitly set robot to default position before starting
            # This ensures RRT starts from a known, stable configuration
            default_joint_positions = np.array([0.0, -1.2, 1.1, 0.0, 0.0, 0.0])  # UR10e 6 arm joints
            articulation_controller = self.ur10e.get_articulation_controller()
            articulation_controller.apply_action(ArticulationAction(
                joint_positions=default_joint_positions,
                joint_indices=np.array([0, 1, 2, 3, 4, 5])
            ))
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

    def _get_prim_world_pose(self, prim_path):
        """Get world pose of a prim from USD"""
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])

        xform = UsdGeom.Xformable(prim)
        world_transform = xform.ComputeLocalToWorldTransform(0)
        translation = world_transform.ExtractTranslation()
        rotation = world_transform.ExtractRotationQuat()

        position = np.array([translation[0], translation[1], translation[2]])
        # USD quaternion is (real, i, j, k), Isaac Sim uses (w, x, y, z)
        orientation = np.array([rotation.GetReal(), rotation.GetImaginary()[0],
                               rotation.GetImaginary()[1], rotation.GetImaginary()[2]])

        return position, orientation

    async def _pick_and_place_cube(self, cube, cube_name):
        """Pick and place cylinder using RRT (8 phases: pick with retry, place, return home)"""
        try:
            total_cubes = self.grid_length * self.grid_width
            # Cylinders are scaled 3.0x in Z, so actual height is ~15.45cm
            cylinder_base_height = 0.0515
            cylinder_radius = 0.0258  # Base radius
            cylinder_scale_z = 3.0
            cylinder_height = cylinder_base_height * cylinder_scale_z  # After scaling = 0.1545m
            cylinder_half = cylinder_height / 2.0

            # Safe height for waypoints (above all objects)
            # Robot base now at Z=0.75m (half of previous 1.5m)
            # Cylinder tops are at ~0.8045m
            # UR10e max reach is ~1.3m, so safe height must be within reach
            # Robot base at 0.75m, so max reachable Z ≈ 0.75 + 1.3 = 2.05m
            # Using 1.0m safe height gives 0.1955m (19.5cm) clearance above cylinder tops
            safe_height = 1.0  # Safe height within UR10e reach

            # End effector offset for horizontal side grasping
            # For side approach (along Y-axis), offset should be along Y, not Z
            # The tool0 frame is offset from the actual gripper fingers
            # For horizontal grasping, we need Y-offset to account for gripper length
            # Setting to zero for now - will tune based on actual gripper geometry
            end_effector_offset = np.array([0, 0, 0])

            # UR10e with Robotiq gripper orientation for SIDE grasping (horizontal approach)
            # Isaac Sim viewport: Z-axis is UP (vertical), Y-axis is green (left/right), X-axis is red (forward/back)
            #
            # Goal: Gripper approaches HORIZONTALLY from the SIDE (parallel to Y-axis)
            #       and grasps cylinder at middle of its body (like grasping from side)
            #
            # Default orientation [0, 0, 0]: gripper points down along -Z
            # We need: gripper points horizontally along Y-axis, fingers vertical (parallel to Z)
            #
            # Solution: Pitch 90° around X-axis, then Yaw 90° around Z-axis
            # [π/2, 0, π/2] = gripper approaches along Y-axis, fingers wrap around cylinder vertically
            orientation = euler_angles_to_quats(np.array([np.pi / 2.0, 0, np.pi / 2.0]))

            max_pick_attempts = 3
            pick_success = False

            # Get current cube position for obstacle detection
            cube_pos_initial, _ = cube.get_world_pose()

            # PHASE 0: Add NEARBY cubes as obstacles (not the target cube)
            # Only cubes within collision radius will be added to avoid RRT failures
            self._add_other_cubes_as_obstacles(self.current_cube_index, cube_pos_initial)

            for pick_attempt in range(1, max_pick_attempts + 1):
                if pick_attempt > 1:
                    print(f"  Retry {pick_attempt}/{max_pick_attempts}")
                # Don't reset to home on retry - just try again from current position
                # This avoids unnecessary double movements

                cube_pos_current, _ = cube.get_world_pose()

                # Phase 1: Open gripper fully before approaching
                self.gripper.open()
                for _ in range(5):  # Increased from 2 to ensure gripper is fully open
                    await omni.kit.app.get_app().next_update_async()

                # Phase 2: Move to side approach position (offset along Y-axis)
                # Approach horizontally from the side (parallel to Y-axis)
                # Position gripper 20cm away from cylinder along Y-axis at cylinder's center height
                side_offset = 0.20  # 20cm offset along Y-axis for side approach
                high_waypoint = np.array([
                    cube_pos_current[0],
                    cube_pos_current[1] + side_offset,  # Offset along Y-axis (approach from side)
                    cube_pos_current[2]  # At cylinder center height (middle of body)
                ])
                # Apply end effector offset
                high_waypoint = high_waypoint + end_effector_offset

                success = await self._move_to_target_rrt(high_waypoint, orientation, skip_factor=5)
                if not success:
                    if pick_attempt < max_pick_attempts:
                        # Reset to safe position before retry
                        await self._reset_to_safe_config()
                        continue
                    # Clean up before returning
                    self._remove_all_cube_obstacles()
                    # Reset to safe position before moving to next cube
                    await self._reset_to_safe_config()
                    return False, f"Failed to reach side approach {cube_name}"

                # Phase 3: Pick approach (move horizontally toward cylinder center)
                cube_pos_realtime, _ = cube.get_world_pose()
                # Grasp at cylinder center (middle of height) - approach horizontally along Y-axis
                # Gripper moves from side_offset position to cylinder center
                pick_pos = np.array([
                    cube_pos_realtime[0],
                    cube_pos_realtime[1],  # Move to cylinder's Y position (no offset)
                    cube_pos_realtime[2]   # At cylinder center height
                ])
                # Apply end effector offset
                pick_pos = pick_pos + end_effector_offset

                # Slow descent for precision
                success = await self._move_to_target_rrt(pick_pos, orientation, skip_factor=3)
                if not success:
                    if pick_attempt < max_pick_attempts:
                        # Reset to safe position before retry
                        await self._reset_to_safe_config()
                        continue
                    # Clean up before returning
                    self._remove_all_cube_obstacles()
                    # Reset to safe position before moving to next cube
                    await self._reset_to_safe_config()
                    return False, f"Failed pick approach for {cube_name}"

                for _ in range(5):  # Pick stabilization
                    await omni.kit.app.get_app().next_update_async()

                self.gripper.close()
                for _ in range(15):  # Gripper close
                    await omni.kit.app.get_app().next_update_async()

                # Phase 4: Pick retreat - Go straight up from current cube position (not EE position)
                # This prevents rotation by maintaining XY position
                retreat_pos = np.array([cube_pos_realtime[0], cube_pos_realtime[1], safe_height])

                success = await self._move_to_target_rrt(retreat_pos, orientation, skip_factor=5)
                if not success:
                    self.ur10e.gripper.open()
                    for _ in range(5):
                        await omni.kit.app.get_app().next_update_async()
                    if pick_attempt < max_pick_attempts:
                        # Reset to safe position before retry
                        await self._reset_to_safe_config()
                        continue
                    # Clean up before returning
                    self._remove_all_cube_obstacles()
                    # Reset to safe position before moving to next cube
                    await self._reset_to_safe_config()
                    return False, f"Failed pick retreat for {cube_name}"

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
                        self.gripper.open()
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

            # Use same grid dimensions as pickup grid (like working version)
            # This ensures proper spacing and placement
            place_row = self.placed_count // self.grid_width
            place_col = self.placed_count % self.grid_width

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

            # Place position: XY from grid calculation, Z at cylinder center height (like Franka)
            place_height = cylinder_half + 0.005  # Slightly above ground
            place_pos = np.array([cube_x, cube_y, place_height])
            # Apply end effector offset
            place_pos = place_pos + end_effector_offset

            # PLACE PHASE - Add NEARBY unpicked cubes as obstacles
            # Only add cubes within collision radius of the place position
            # Reduced radius to minimize RRT failures
            collision_radius = 0.20  # Reduced from 0.30 to 0.20 (20cm)
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
            via_point = np.array([0.35, 0.30, safe_height])
            await self._move_to_target_rrt(via_point, orientation, skip_factor=5)

            # Approach from directly above the place position for straight descent
            above_place = np.array([place_pos[0], place_pos[1], safe_height])
            await self._move_to_target_rrt(above_place, orientation, skip_factor=5)
            await self._move_to_target_rrt(pre_place_pos, orientation, skip_factor=4)

            # Phase 6: Place approach (slower for precision)
            release_height = place_pos + np.array([0.0, 0.0, 0.08])
            await self._move_to_target_rrt(release_height, orientation, skip_factor=3)
            for _ in range(3):  # Place stabilization (increased from 1 to prevent cube throw)
                await omni.kit.app.get_app().next_update_async()

            self.gripper.open()
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
            current_ee_pos, _ = self.ur10e.end_effector.get_world_pose()
            retreat_pos = current_ee_pos + np.array([0.0, 0.0, 0.15])  # Retreat up
            await self._move_to_target_rrt(retreat_pos, orientation, skip_factor=6)

            self.gripper.close()
            for _ in range(2):  # Gripper close
                await omni.kit.app.get_app().next_update_async()

            # Move to a safe intermediate position using RRT to avoid obstacles
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
        robot_base_translation, robot_base_orientation = self.ur10e.get_world_pose()
        self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        # Compute IK solution
        ik_action, ik_success = self.articulation_kinematics_solver.compute_inverse_kinematics(
            target_position, target_orientation
        )

        if not ik_success:
            return False

        # Get current joint positions (UR10e has 6 arm joints)
        current_positions = self.ur10e.get_joint_positions()[:6]  # First 6 joints (arm only)
        target_positions = ik_action.joint_positions[:6]

        # Interpolate from current to target
        articulation_controller = self.ur10e.get_articulation_controller()
        for i in range(num_steps):
            alpha = (i + 1) / num_steps
            interpolated_positions = current_positions + alpha * (target_positions - current_positions)

            # Create action for arm joints only (indices 0-6)
            action = ArticulationAction(
                joint_positions=interpolated_positions,
                joint_indices=np.array([0, 1, 2, 3, 4, 5])  # UR10e has 6 arm joints
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
        # UR10e has 6 arm joints - set them to safe configuration
        # Safe config: arm joints only, gripper controlled separately
        safe_arm_joints = np.array([0.0, -1.2, 1.1, 0.0, 0.0, 0.0])

        articulation_controller = self.ur10e.get_articulation_controller()
        articulation_controller.apply_action(ArticulationAction(
            joint_positions=safe_arm_joints,
            joint_indices=np.array([0, 1, 2, 3, 4, 5])
        ))
        self.gripper.close()
        for _ in range(15):
            await omni.kit.app.get_app().next_update_async()

    async def _taskspace_straight_line(self, end_pos, orientation, num_waypoints=8):
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
        robot_base_translation, robot_base_orientation = self.ur10e.get_world_pose()
        self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        # Get current joint positions as starting point (6 arm joints only)
        # UR10e has 6 active joints in cspace, gripper is controlled separately
        current_joint_positions = self.ur10e.get_joint_positions()[:6]

        # Compute IK for end position
        ik_action, ik_success = self.articulation_kinematics_solver.compute_inverse_kinematics(
            end_pos, orientation
        )

        if not ik_success:
            return False

        # Extract joint positions from ArticulationAction (6 arm joints only for UR10e)
        end_joint_positions = ik_action.joint_positions[:6]

        # Interpolate in joint space (smooth motion, maintains orientation)
        for i in range(1, num_waypoints + 1):
            alpha = i / num_waypoints
            interpolated_joints = current_joint_positions + alpha * (end_joint_positions - current_joint_positions)

            # Apply joint positions to arm only (indices 0-6)
            action = ArticulationAction(
                joint_positions=interpolated_joints,
                joint_indices=np.array([0, 1, 2, 3, 4, 5])
            )
            self.ur10e.get_articulation_controller().apply_action(action)

            # Wait for physics update
            await omni.kit.app.get_app().next_update_async()

        # No settling time for faster motion
        return True

    def _plan_to_target(self, target_position, target_orientation):
        """Plan path to target using RRT with smooth trajectory generation"""
        # Update dynamic obstacles from Lidar before planning (real-time detection)
        self._update_dynamic_obstacles()

        robot_base_translation, robot_base_orientation = self.ur10e.get_world_pose()
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
        # Balanced iterations: enough for quality paths but fast failures when blocked
        # Increased slightly from previous values for better success rate
        max_iterations = 12000 if has_obstacles else (8000 if total_cubes <= 4 else 10000)
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
        art_trajectory = ArticulationTrajectory(self.ur10e, trajectory, 1.0 / 60.0)
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
                self.ur10e.apply_action(action)
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
        robot_base_translation, robot_base_orientation = self.ur10e.get_world_pose()
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
            self.ur10e = None
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
app = UR10eRRTDynamicGrid()

