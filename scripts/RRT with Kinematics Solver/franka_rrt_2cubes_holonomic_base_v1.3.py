"""
Franka RRT Pick and Place - Dynamic Grid with Kinematics Solver + Holonomic Base
RRT path planning with obstacle avoidance, conservative collision spheres,
dynamic grid configuration, pick retry logic, return to home after each cube.
Now includes holonomic mobile base for the Franka robot.
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
import omni.kit.commands

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
from pxr import UsdPhysics
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
import carb

# Holonomic robot imports
from isaacsim.robot.wheeled_robots.robots import WheeledRobot, HolonomicRobotUsdSetup
from isaacsim.robot.wheeled_robots.controllers import HolonomicController

# URDF import
from isaacsim.asset.importer.urdf import _urdf

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

        # Holonomic base
        self.holonomic_base = None
        self.holonomic_controller = None
        self.base_prim_path = "/World/HolonomicBase"

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

        # Holonomic base control UI
        self.move_forward_btn = None
        self.move_backward_btn = None
        self.move_left_btn = None
        self.move_right_btn = None
        self.rotate_left_btn = None
        self.rotate_right_btn = None
        self.stop_base_btn = None

        self.build_ui()
    
    def build_ui(self):
        """Build UI"""
        self.window = ui.Window("Cobot - Grasping with Holonomic Base", width=450, height=700)

        with self.window.frame:
            with ui.VStack(spacing=10):
                ui.Label("Cobot - Pick and Place with Mobile Base",
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

                # Holonomic Base Control Section
                with ui.CollapsableFrame("Holonomic Base Control", height=0, collapsed=False):
                    with ui.VStack(spacing=5):
                        ui.Label("Move the mobile base:", alignment=ui.Alignment.CENTER)

                        # Movement controls in grid layout
                        with ui.HStack(spacing=5):
                            ui.Spacer()
                            self.move_forward_btn = ui.Button("Forward", height=30, width=80,
                                                             clicked_fn=lambda: self._move_base(0.2, 0.0, 0.0), enabled=False)
                            ui.Spacer()

                        with ui.HStack(spacing=5):
                            self.move_left_btn = ui.Button("Left", height=30, width=80,
                                                          clicked_fn=lambda: self._move_base(0.0, 0.2, 0.0), enabled=False)
                            self.stop_base_btn = ui.Button("STOP", height=30, width=80,
                                                          clicked_fn=lambda: self._move_base(0.0, 0.0, 0.0), enabled=False)
                            self.move_right_btn = ui.Button("Right", height=30, width=80,
                                                           clicked_fn=lambda: self._move_base(0.0, -0.2, 0.0), enabled=False)

                        with ui.HStack(spacing=5):
                            ui.Spacer()
                            self.move_backward_btn = ui.Button("Backward", height=30, width=80,
                                                              clicked_fn=lambda: self._move_base(-0.2, 0.0, 0.0), enabled=False)
                            ui.Spacer()

                        ui.Spacer(height=5)

                        # Rotation controls
                        with ui.HStack(spacing=5):
                            self.rotate_left_btn = ui.Button("Rotate Left", height=30,
                                                            clicked_fn=lambda: self._move_base(0.0, 0.0, 0.5), enabled=False)
                            self.rotate_right_btn = ui.Button("Rotate Right", height=30,
                                                             clicked_fn=lambda: self._move_base(0.0, 0.0, -0.5), enabled=False)

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
        """Load the scene with Franka on holonomic base, dynamic grid of cubes, and container"""
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
            await omni.kit.app.get_app().next_update_async()

            World.clear_instance()
            await omni.kit.app.get_app().next_update_async()

            self.world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
            await omni.kit.app.get_app().next_update_async()

            self.world.scene.add_default_ground_plane()
            await omni.kit.app.get_app().next_update_async()

            # Load holonomic base from URDF
            print("Loading holonomic base from URDF...")
            await self._load_holonomic_base()
            await omni.kit.app.get_app().next_update_async()

            # Add Franka as a child of the holonomic base
            # This ensures Franka moves with the base automatically
            print("=== Adding Franka on holonomic base ===")
            franka_name = f"franka_{int(time.time() * 1000)}"

            # Mount Franka on the base_link of the holonomic base
            # This makes it move with the base automatically via USD hierarchy
            franka_prim_path = f"{self.base_prim_path}/base_link/Franka"

            franka_usd_path = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
            robot_prim = add_reference_to_stage(usd_path=franka_usd_path, prim_path=franka_prim_path)
            robot_prim.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
            robot_prim.GetVariantSet("Mesh").SetVariantSelection("Quality")

            await omni.kit.app.get_app().next_update_async()

            print(f"Franka loaded at: {franka_prim_path}")

            # Position Franka on top of the holonomic base
            # The base_link is at the center of the chassis
            # After scaling the base by 2x, the chassis height is 0.03m * 2 = 0.06m
            # The chassis center is at z=0.1m * 2 = 0.2m (after scaling)
            # So we need to position Franka above the chassis top
            from pxr import Gf, UsdGeom
            stage = omni.usd.get_context().get_stage()
            franka_xform = UsdGeom.Xformable(robot_prim)

            # Clear any existing transforms and set new position
            # Position Franka at the top of the chassis
            # Chassis center is at 0.2m, half-height is 0.03m, so top is at 0.23m
            # Add small clearance and position Franka
            franka_xform.ClearXformOpOrder()
            translate_op = franka_xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.25))  # 25cm above base_link origin

            print(f"Franka positioned 25cm above base_link (on top of chassis)")

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

            # Add physics to container
            from omni.isaac.core.utils.stage import get_current_stage
            stage = get_current_stage()
            container_prim = stage.GetPrimAtPath(container_prim_path)
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(container_prim)
            rigid_body_api.CreateKinematicEnabledAttr(True)
            UsdPhysics.CollisionAPI.Apply(container_prim)

            await omni.kit.app.get_app().next_update_async()

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

            await omni.kit.app.get_app().next_update_async()

            self.world.initialize_physics()
            for _ in range(3):
                await omni.kit.app.get_app().next_update_async()

            self.world.reset()
            for _ in range(3):
                await omni.kit.app.get_app().next_update_async()

            self.franka.disable_gravity()
            articulation_controller = self.franka.get_articulation_controller()
            kp_gains = 1e15 * np.ones(9)
            kd_gains = 1e13 * np.ones(9)
            articulation_controller.set_gains(kp_gains, kd_gains)

            default_joint_positions = np.array([0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75, 0.0, 0.0])
            self.franka.set_joints_default_state(positions=default_joint_positions)
            self.gripper.set_default_state(self.gripper.joint_closed_positions)

            for _ in range(3):
                await omni.kit.app.get_app().next_update_async()

            articulation_controller = self.franka.get_articulation_controller()
            close_action = ArticulationAction(
                joint_positions=self.gripper.joint_closed_positions,
                joint_indices=np.array([7, 8])
            )
            articulation_controller.apply_action(close_action)

            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()

            self._setup_rrt()

            # DO NOT add cubes as obstacles - they are targets to pick, not obstacles to avoid
            # Only manual obstacles will be added to RRT for collision avoidance

            print("Scene loaded")

            # Enable buttons
            self.pick_btn.enabled = True
            self.reset_btn.enabled = True
            self.add_obstacle_btn.enabled = True
            self.remove_obstacle_btn.enabled = True

            # Enable holonomic base control buttons
            self.move_forward_btn.enabled = True
            self.move_backward_btn.enabled = True
            self.move_left_btn.enabled = True
            self.move_right_btn.enabled = True
            self.rotate_left_btn.enabled = True
            self.rotate_right_btn.enabled = True
            self.stop_base_btn.enabled = True

            self._update_status("Scene loaded! Ready to pick and place")

        except Exception as e:
            self._update_status(f"Error: {e}")
            import traceback
            traceback.print_exc()

    async def _load_holonomic_base(self):
        """Load holonomic robot base from URDF (converted from xacro)"""
        try:
            # Paths for xacro/URDF import
            root_path = str(project_root / "holonomic" / "description")
            xacro_file = str(project_root / "holonomic" / "description" / "robot_core.xacro")

            print(f"Loading holonomic base from xacro...")
            print(f"Xacro file: {xacro_file}")
            print(f"Root path: {root_path}")

            # Try to use xacro to convert to URDF
            urdf_content = None
            try:
                # Try importing xacro as a Python library
                print("Attempting to convert xacro to URDF using xacro library...")
                import xacro

                # Process the xacro file
                urdf_content = xacro.process_file(xacro_file).toxml()
                print("Successfully converted xacro to URDF using xacro library")

            except ImportError:
                print("xacro library not installed")
                print("Falling back to using pre-generated URDF with STL meshes...")

                # Fall back to the high-res URDF with STL meshes
                urdf_path = str(project_root / "holonomic" / "description" / "holonomic_robot3.urdf")
                with open(urdf_path, 'r') as f:
                    urdf_content = f.read()

                # Replace package://holonomic_robot3/meshes/ with ../meshes/
                urdf_content = urdf_content.replace(
                    'package://holonomic_robot3/meshes/',
                    '../meshes/'
                )
            except Exception as e:
                print(f"Could not use xacro library: {e}")
                print("Falling back to using pre-generated URDF with STL meshes...")

                # Fall back to the high-res URDF with STL meshes
                urdf_path = str(project_root / "holonomic" / "description" / "holonomic_robot3.urdf")
                with open(urdf_path, 'r') as f:
                    urdf_content = f.read()

                # Replace package://holonomic_robot3/meshes/ with ../meshes/
                urdf_content = urdf_content.replace(
                    'package://holonomic_robot3/meshes/',
                    '../meshes/'
                )

            # Write modified URDF to a temporary file in the same directory
            # This ensures relative paths work correctly
            import tempfile
            import os
            temp_urdf = tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.urdf',
                delete=False,
                dir=root_path  # Create temp file in same directory as original
            )
            temp_urdf.write(urdf_content)
            temp_urdf.close()
            temp_file_name = os.path.basename(temp_urdf.name)

            # Create import configuration
            import_config = _urdf.ImportConfig()
            import_config.merge_fixed_joints = False
            import_config.fix_base = False  # Mobile base should not be fixed
            import_config.make_default_prim = False
            import_config.create_physics_scene = False  # Already created by World
            import_config.import_inertia_tensor = True
            import_config.self_collision = False
            import_config.distance_scale = 1.0
            import_config.density = 0.0
            import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_VELOCITY

            # Use the URDF interface directly (like in the examples)
            urdf_interface = _urdf.acquire_urdf_interface()

            print(f"Parsing URDF...")

            # First parse the URDF file to get a UrdfRobot object
            robot_model = urdf_interface.parse_urdf(root_path, temp_file_name, import_config)

            if not robot_model:
                raise Exception("Failed to parse URDF file")

            print(f"URDF parsed successfully, importing to stage...")

            # Import robot using the interface with root_path and filename
            # This allows the interface to resolve relative mesh paths correctly
            prim_path = urdf_interface.import_robot(
                root_path,
                temp_file_name,
                robot_model,  # Pass the parsed robot model
                import_config,
                "",  # Empty dest_path means import to current stage
                False  # get_articulation_root
            )

            # Clean up temporary file
            try:
                os.unlink(temp_urdf.name)
            except:
                pass

            if not prim_path:
                raise Exception("Failed to import holonomic base URDF")

            print(f"Holonomic base imported at: {prim_path}")

            # The robot is imported at the root, we need to move it to our desired path
            if prim_path != self.base_prim_path:
                print(f"Moving robot from {prim_path} to {self.base_prim_path}")
                omni.kit.commands.execute(
                    "MovePrim",
                    path_from=prim_path,
                    path_to=self.base_prim_path
                )
                prim_path = self.base_prim_path

            print(f"Holonomic base ready at: {prim_path}")

            # Wait for import to complete
            await omni.kit.app.get_app().next_update_async()

            # Scale and position the holonomic base
            from pxr import Gf, UsdGeom, Sdf, Usd, UsdPhysics
            stage = omni.usd.get_context().get_stage()
            base_prim = stage.GetPrimAtPath(self.base_prim_path)

            if base_prim.IsValid():
                base_xform = UsdGeom.Xformable(base_prim)

                # Get existing xform ops (URDF importer already created them)
                xform_ops = base_xform.GetOrderedXformOps()

                # Find and modify the scale and translate ops
                for xform_op in xform_ops:
                    op_name = xform_op.GetOpName()

                    if op_name == "xformOp:scale":
                        # Scale up the base (it's currently too small - 0.5m x 0.5m chassis)
                        # Scale by 2x to make it 1m x 1m, more appropriate for a mobile base
                        xform_op.Set(Gf.Vec3d(2.0, 2.0, 2.0))
                        print(f"Set holonomic base scale to 2x")

                    elif op_name == "xformOp:translate":
                        # Position the base so it sits on the ground plane
                        xform_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
                        print(f"Set holonomic base position to ground level")

            await omni.kit.app.get_app().next_update_async()

            # Debug: Print the USD hierarchy to find the actual joint paths

            print("\n=== Inspecting USD hierarchy ===")
            base_prim = stage.GetPrimAtPath(self.base_prim_path)
            if base_prim.IsValid():
                print(f"Base prim found at: {self.base_prim_path}")
                # List all children
                for child in base_prim.GetChildren():
                    print(f"  Child: {child.GetPath()}")
                    # Check if it's a joint
                    if child.HasAPI(UsdPhysics.RevoluteJoint):
                        print(f"    -> This is a RevoluteJoint!")
                    # List grandchildren
                    for grandchild in child.GetChildren():
                        print(f"    Grandchild: {grandchild.GetPath()}")
                        if grandchild.HasAPI(UsdPhysics.RevoluteJoint):
                            print(f"      -> This is a RevoluteJoint!")
            print("=== End hierarchy ===\n")

            # Find all wheel joints - they are under /joints/ folder
            wheel_joint_names = ["wheel1_joint", "wheel2_joint", "wheel3_joint", "wheel4_joint"]
            wheel_joints_found = []

            for joint_name in wheel_joint_names:
                # Joints are under the /joints/ subfolder
                joint_path = f"{self.base_prim_path}/joints/{joint_name}"
                joint_prim = stage.GetPrimAtPath(joint_path)

                if joint_prim.IsValid():
                    wheel_joints_found.append((joint_path, joint_name))
                    print(f"Found wheel joint: {joint_path}")
                else:
                    print(f"Warning: Joint {joint_name} not found at {joint_path}")

            if len(wheel_joints_found) != 4:
                print(f"Warning: Expected 4 wheel joints, found {len(wheel_joints_found)}")

            # Add custom attributes for holonomic wheels to the wheel joints
            # For omnidirectional wheels (not mecanum), the angle is 0 degrees
            mecanum_angles = [0.0, 0.0, 0.0, 0.0]  # All 0 for omnidirectional wheels
            wheel_radius = 0.08  # 8cm radius from xacro file

            print("Adding holonomic wheel attributes to joints...")
            for i, (joint_path, joint_name) in enumerate(wheel_joints_found[:4]):  # Take first 4
                joint_prim = stage.GetPrimAtPath(joint_path)

                if joint_prim.IsValid():
                    # Add custom attributes for omnidirectional wheel
                    if not joint_prim.HasAttribute("isaacmecanumwheel:angle"):
                        joint_prim.CreateAttribute("isaacmecanumwheel:angle", Sdf.ValueTypeNames.Float).Set(mecanum_angles[i])
                    if not joint_prim.HasAttribute("isaacmecanumwheel:radius"):
                        joint_prim.CreateAttribute("isaacmecanumwheel:radius", Sdf.ValueTypeNames.Float).Set(wheel_radius)
                    print(f"  Added attributes to {joint_name}: angle={mecanum_angles[i]}, radius={wheel_radius}")
                else:
                    print(f"  Warning: Joint {joint_name} not valid at {joint_path}")

            # Create WheeledRobot wrapper for the holonomic base
            # The holonomic robot has 4 continuous wheel joints
            self.holonomic_base = self.world.scene.add(
                WheeledRobot(
                    prim_path=self.base_prim_path,
                    name="holonomic_base",
                    wheel_dof_names=wheel_joint_names,
                    position=np.array([0.0, 0.0, 0.0]),
                )
            )

            await omni.kit.app.get_app().next_update_async()

            # Setup holonomic controller
            holonomic_setup = HolonomicRobotUsdSetup(
                robot_prim_path=self.base_prim_path,
                com_prim_path=f"{self.base_prim_path}/base_link"
            )

            (
                wheel_radius,
                wheel_positions,
                wheel_orientations,
                mecanum_angles,
                wheel_axis,
                up_axis,
            ) = holonomic_setup.get_holonomic_controller_params()

            self.holonomic_controller = HolonomicController(
                name="holonomic_controller",
                wheel_radius=wheel_radius,
                wheel_positions=wheel_positions,
                wheel_orientations=wheel_orientations,
                mecanum_angles=mecanum_angles,
                wheel_axis=wheel_axis,
                up_axis=up_axis,
            )

            print("Holonomic base and controller setup complete")

        except Exception as e:
            print(f"Error loading holonomic base: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _move_base(self, linear_x, linear_y, angular_z):
        """Move the holonomic base

        Args:
            linear_x: Forward/backward velocity (m/s)
            linear_y: Left/right velocity (m/s)
            angular_z: Rotation velocity (rad/s)
        """
        if self.holonomic_base is None or self.holonomic_controller is None:
            print("Holonomic base not initialized")
            return

        try:
            # Create command array [vx, vy, omega]
            command = np.array([linear_x, linear_y, angular_z])

            # Get wheel velocities from controller
            action = self.holonomic_controller.forward(command)

            # Apply action to the wheeled robot
            self.holonomic_base.apply_wheel_actions(action)

            print(f"Base command: vx={linear_x:.2f}, vy={linear_y:.2f}, omega={angular_z:.2f}")

        except Exception as e:
            print(f"Error moving base: {e}")
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

            # IMPROVED: Better stabilization before first pick
            # Wait for robot to fully settle in default position
            for _ in range(35):  # Optimized settling time
                await omni.kit.app.get_app().next_update_async()

            # Initialize gripper to closed position at start
            articulation_controller = self.franka.get_articulation_controller()
            articulation_controller.apply_action(ArticulationAction(
                joint_positions=self.gripper.joint_closed_positions, joint_indices=np.array([7, 8])))
            for _ in range(12):  # Optimized gripper close wait
                await omni.kit.app.get_app().next_update_async()

            # IMPROVED: Explicitly set robot to default position before starting
            # This ensures RRT starts from a known, stable configuration
            default_joint_positions = np.array([0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75, 0.0, 0.0])
            articulation_controller.apply_action(ArticulationAction(joint_positions=default_joint_positions))
            for _ in range(10):  # Reduced settling time
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
            self.timeline.stop()

        except Exception as e:
            self._update_status(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.is_picking = False
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

                pre_pick_height = 0.10 if total_cubes <= 4 else (0.12 if total_cubes <= 9 else 0.15)
                pre_pick_pos = cube_pos_current + np.array([0.0, 0.0, pre_pick_height])

                # Phase 1: Pre-pick (open gripper BEFORE approaching)
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

                success = await self._move_to_target_rrt(pre_pick_pos, orientation)
                if not success:
                    if pick_attempt < max_pick_attempts:
                        continue
                    return False, f"Failed pre-pick for {cube_name}"

                # Phase 2: Pick approach (gripper already open)
                cube_pos_realtime, _ = cube.get_world_pose()
                pick_pos = np.array([cube_pos_realtime[0], cube_pos_realtime[1], cube_pos_realtime[2]])

                # Faster approach with skip_factor=4
                success = await self._move_to_target_rrt(pick_pos, orientation, skip_factor=4)
                if not success:
                    if pick_attempt < max_pick_attempts:
                        continue
                    return False, f"Failed pick approach for {cube_name}"

                for _ in range(2):  # Pick stabilization (reduced from 3)
                    await omni.kit.app.get_app().next_update_async()

                articulation_controller.apply_action(ArticulationAction(
                    joint_positions=self.gripper.joint_closed_positions, joint_indices=np.array([7, 8])))
                for _ in range(10):  # Gripper close (reduced from 12)
                    await omni.kit.app.get_app().next_update_async()

                # Phase 3: Pick retreat
                current_ee_pos, _ = self.franka.end_effector.get_world_pose()
                retreat_height = 0.15 if total_cubes <= 4 else (0.18 if total_cubes <= 9 else 0.20)
                retreat_pos = current_ee_pos + np.array([0.0, 0.0, retreat_height])

                success = await self._move_to_target_rrt(retreat_pos, orientation)
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
            container_length, container_width = self.container_dimensions[0], self.container_dimensions[1]
            place_row = self.placed_count // self.grid_width
            place_col = self.placed_count % self.grid_width

            # IMPROVED: Asymmetric margins for container
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

            place_height = cube_half + 0.005
            place_pos = np.array([cube_x, cube_y, place_height])

            # Use higher waypoints to provide clearance for held cube (cube size ~5cm)
            # Reduced height to avoid IK failures for far positions
            pre_place_pos = place_pos + np.array([0.0, 0.0, 0.25])  # Reduced from 0.35 to 0.25

            # Phase 5: Pre-place (cube disabled, using high waypoints for clearance)
            # Add via point to ensure consistent approach direction (prevents base rotation)
            via_point = np.array([0.35, 0.25, 0.40])  # Reduced from 0.45 to 0.40
            await self._move_to_target_rrt(via_point, orientation)
            await self._move_to_target_rrt(pre_place_pos, orientation)

            # Phase 6: Place approach
            release_height = place_pos + np.array([0.0, 0.0, 0.08])
            await self._move_to_target_rrt(release_height, orientation)
            for _ in range(1):  # Place stabilization (reduced from 2)
                await omni.kit.app.get_app().next_update_async()

            articulation_controller = self.franka.get_articulation_controller()
            articulation_controller.apply_action(ArticulationAction(
                joint_positions=self.gripper.joint_opened_positions, joint_indices=np.array([7, 8])))
            for _ in range(10):  # Gripper open (reduced from 12)
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

            # Phase 7: Place retreat - retreat vertically then move to safe position
            # First retreat vertically (faster with skip_factor=4)
            current_ee_pos, _ = self.franka.end_effector.get_world_pose()
            retreat_pos = current_ee_pos + np.array([0.0, 0.0, 0.15])  # Retreat up
            retreat_success = await self._move_to_target_rrt(retreat_pos, orientation, skip_factor=4)

            articulation_controller.apply_action(ArticulationAction(
                joint_positions=self.gripper.joint_closed_positions, joint_indices=np.array([7, 8])))
            for _ in range(2):  # Gripper close
                await omni.kit.app.get_app().next_update_async()

            # Move to a safe intermediate position using RRT to avoid obstacles
            # This position is high and centered, away from obstacles
            safe_ee_position = np.array([0.40, 0.0, 0.50])  # High, centered position
            safe_success = await self._move_to_target_rrt(safe_ee_position, orientation, skip_factor=4)

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

    async def _move_to_target_ik(self, target_position, target_orientation, num_steps=12):
        """
        Move to target using IK directly (for simple, straight-line movements)
        This is faster and more predictable than RRT for vertical descents

        Args:
            target_position: Target end effector position
            target_orientation: Target end effector orientation
            num_steps: Number of interpolation steps (default: 12, optimized for speed)

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
            skip_factor: Frame skip factor for execution speed (default=3)

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

    async def _taskspace_straight_line(self, start_pos, end_pos, orientation, num_waypoints=12):
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
            skip_factor: Number of frames to skip (higher = faster, default=3)
        """
        if action_sequence is None or len(action_sequence) == 0:
            return False

        # Skip frames for faster motion
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
        """Reset button callback"""
        try:
            self.is_picking = False

            # Stop timeline first
            self.timeline.stop()

            # Clear World instance (this deletes all objects from stage)
            if self.world is not None:
                World.clear_instance()

            # Also clear stage manually to ensure everything is deleted
            from omni.isaac.core.utils.stage import clear_stage
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
            self.holonomic_base = None
            self.holonomic_controller = None
            self.cubes = []
            self.obstacles = {}
            self.obstacle_counter = 0
            self.placed_count = 0
            self.current_cube_index = 0  # Reset cube index

            # Reset UI
            self.load_btn.enabled = True
            self.pick_btn.enabled = False
            self.reset_btn.enabled = False
            self.add_obstacle_btn.enabled = False
            self.remove_obstacle_btn.enabled = False

            # Disable holonomic base control buttons
            self.move_forward_btn.enabled = False
            self.move_backward_btn.enabled = False
            self.move_left_btn.enabled = False
            self.move_right_btn.enabled = False
            self.rotate_left_btn.enabled = False
            self.rotate_right_btn.enabled = False
            self.stop_base_btn.enabled = False

            print("Reset complete")
            self._update_status("Reset complete")

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

            # Default obstacle position - user can move it in the scene
            # RRT will dynamically detect and avoid obstacles
            obstacle_position = np.array([0.45, 0.20, 0.10])
            obstacle_orientation = euler_angles_to_quats(np.array([0.0, 0.0, 0.0]))  # No rotation
            obstacle_scale = np.array([0.20, 0.05, 0.30])  # Default: 20x5x30cm

            # Create obstacle as a FixedCuboid (has collision geometry for RRT)
            obstacle = self.world.scene.add(
                FixedCuboid(
                    name=obstacle_name,
                    prim_path=obstacle_prim_path,
                    position=obstacle_position,
                    orientation=obstacle_orientation,
                    size=1.0,
                    scale=obstacle_scale,
                    color=np.array([0.0, 0.0, 1.0])  # Blue color
                )
            )

            # Add obstacle to RRT planner for collision avoidance
            # IMPORTANT: Add as DYNAMIC obstacle (static=False) so position updates when moved in scene
            self.rrt.add_obstacle(obstacle, static=False)

            # Store obstacle reference
            self.obstacles[obstacle_name] = obstacle

            print(f"Obstacle added: {obstacle_name} (Total: {len(self.obstacles)})")
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

