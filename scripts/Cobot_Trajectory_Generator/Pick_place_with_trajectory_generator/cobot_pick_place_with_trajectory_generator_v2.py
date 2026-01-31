"""
UR10 Pick and Place with Surface Gripper and Task-Space Trajectory
Supports variable number of cubes in a grid pattern
"""

import asyncio
import time
import carb
import numpy as np
import os
from pathlib import Path
import sys
import omni.ui as ui
from omni.kit.async_engine import run_coroutine
import omni.timeline

# Isaac Sim Core APIs
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.prims import delete_prim, get_prim_at_path
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.types import ArticulationAction
from pxr import UsdPhysics

# Motion generation imports
from isaacsim.robot_motion.motion_generation import (
    LulaTaskSpaceTrajectoryGenerator,
    LulaKinematicsSolver,
    ArticulationTrajectory
)

# Surface Gripper imports
import usd.schema.isaac.robot_schema as robot_schema
from isaacsim.core.utils.stage import get_current_stage

# Add project root to path for local imports
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.manipulators import SingleManipulator
from src.grippers import SurfaceGripper


class UR10PickPlaceTaskSpace:
    """UR10 Pick and Place with Surface Gripper using Task-Space Trajectory"""

    def __init__(self):
        self.window = None
        self.world = None
        self.ur10 = None
        self.gripper = None
        self.container = None
        self.taskspace_trajectory_generator = None
        self.kinematics_solver = None

        # Dynamic cube list
        self.cubes = []  # Will store (cube_object, cube_name) tuples

        # Grid parameters
        self.grid_length = 2  # Default: 2 rows
        self.grid_width = 2   # Default: 2 columns

        # Timeline
        self.timeline = omni.timeline.get_timeline_interface()

        # Task state
        self.is_picking = False
        self.placed_count = 0

        # UI elements
        self.load_btn = None
        self.pick_btn = None
        self.reset_btn = None
        self.status_label = None
        self.length_field = None
        self.width_field = None

        # End effector name for UR10
        self._end_effector_name = "ee_link"

        self.build_ui()
    
    def build_ui(self):
        """Build UI"""
        self.window = ui.Window("UR10 Pick and Place - Task-Space Trajectory", width=450, height=400)

        with self.window.frame:
            with ui.VStack(spacing=10):
                ui.Label("UR10 - Pick and Place with Surface Gripper",
                        alignment=ui.Alignment.CENTER,
                        style={"font_size": 18})

                ui.Spacer(height=10)

                # Grid Configuration Section
                with ui.CollapsableFrame("Grid Configuration", height=0):
                    with ui.VStack(spacing=5):
                        # Length (rows)
                        with ui.HStack(spacing=10):
                            ui.Label("Grid Length (rows):", width=150)
                            self.length_field = ui.IntField(height=25)
                            self.length_field.model.set_value(2)  # Default: 2 rows

                        # Width (columns)
                        with ui.HStack(spacing=10):
                            ui.Label("Grid Width (columns):", width=150)
                            self.width_field = ui.IntField(height=25)
                            self.width_field.model.set_value(2)  # Default: 2 columns

                        # Info label
                        ui.Label("Total cubes will be: Length × Width",
                                alignment=ui.Alignment.CENTER,
                                style={"color": 0xFF888888, "font_size": 12})

                ui.Spacer(height=10)

                # Buttons
                self.load_btn = ui.Button("Load Scene", height=40, clicked_fn=self._on_load)
                self.pick_btn = ui.Button("Start Pick and Place", height=40, clicked_fn=self._on_pick, enabled=False)
                self.reset_btn = ui.Button("Reset Scene", height=40, clicked_fn=self._on_reset, enabled=False)

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
        """Load the scene with UR10, dynamic grid of cubes, and container"""
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

            print("\n" + "="*60)
            print(f"LOADING SCENE - {self.grid_length}x{self.grid_width} GRID ({total_cubes} CUBES)")
            print("="*60)
            
            # Stop timeline
            self.timeline.stop()
            await omni.kit.app.get_app().next_update_async()
            
            # Clear existing world
            World.clear_instance()
            await omni.kit.app.get_app().next_update_async()
            
            # Create world
            print("\n=== Creating World ===")
            self.world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
            await omni.kit.app.get_app().next_update_async()
            
            # Add ground
            print("=== Adding Ground ===")
            self.world.scene.add_default_ground_plane()
            await omni.kit.app.get_app().next_update_async()
            
            # Add UR10 with Surface Gripper
            print("=== Adding UR10 with Surface Gripper ===")
            ur10_name = f"ur10_{int(time.time() * 1000)}"
            ur10_prim_path = f"/World/UR10_{int(time.time() * 1000)}"
            
            ur10_usd_path = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
            robot_prim = add_reference_to_stage(usd_path=ur10_usd_path, prim_path=ur10_prim_path)
            
            # Set gripper variant to Short_Suction (includes surface gripper)
            robot_prim.GetVariantSet("Gripper").SetVariantSelection("Short_Suction")
            
            await omni.kit.app.get_app().next_update_async()
            
            # Configure Surface Gripper properties
            print("=== Configuring Surface Gripper ===")
            stage = get_current_stage()
            gripper_prim_path = f"{ur10_prim_path}/ee_link/SurfaceGripper"
            gripper_prim = stage.GetPrimAtPath(gripper_prim_path)

            if gripper_prim.IsValid():
                # Set gripper properties for better grasping
                gripper_prim.GetAttribute(robot_schema.Attributes.MAX_GRIP_DISTANCE.name).Set(0.05)  # 5cm grip distance
                gripper_prim.GetAttribute(robot_schema.Attributes.COAXIAL_FORCE_LIMIT.name).Set(100.0)  # Strong axial force
                gripper_prim.GetAttribute(robot_schema.Attributes.SHEAR_FORCE_LIMIT.name).Set(100.0)  # Strong shear force
                gripper_prim.GetAttribute(robot_schema.Attributes.RETRY_INTERVAL.name).Set(2.0)  # Try for 2 seconds
                print(f"Surface gripper properties configured at: {gripper_prim_path}")
            else:
                print(f"Warning: Surface gripper prim not found at {gripper_prim_path}")

            # Create Surface Gripper wrapper
            self.gripper = SurfaceGripper(
                end_effector_prim_path=f"{ur10_prim_path}/ee_link",
                surface_gripper_path=gripper_prim_path
            )
            
            # Add manipulator
            self.ur10 = self.world.scene.add(
                SingleManipulator(
                    prim_path=ur10_prim_path,
                    name=ur10_name,
                    end_effector_prim_path=f"{ur10_prim_path}/ee_link",
                    gripper=self.gripper,
                    position=np.array([0.0, 0.0, 0.0]),
                    orientation=np.array([1.0, 0.0, 0.0, 0.0])
                )
            )
            
            print(f"UR10 added: {ur10_name}")
            await omni.kit.app.get_app().next_update_async()
            
            # Add container IN FRONT of robot (positive X direction)
            print("=== Adding Container ===")
            container_prim_path = "/World/Container"
            container_usd_path = f"{get_assets_root_path()}/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Storage/Containers/Container_I/Container_I04_160x120x64cm_PR_V_NVD_01.usd"
            
            add_reference_to_stage(usd_path=container_usd_path, prim_path=container_prim_path)
            
            # Container position: IN FRONT (positive X), to the right (positive Y), farther away
            container_position = np.array([0.7, 0.3, 0.0])  # Moved farther: X from 0.55 to 0.7
            
            # Scale: X=0.3, Y=0.3, Z=0.2 (reduced height to lower rim)
            self.container = self.world.scene.add(
                SingleXFormPrim(
                    prim_path=container_prim_path,
                    name="container",
                    translation=container_position,
                    scale=np.array([0.3, 0.3, 0.2])
                )
            )
            
            # Add physics to container
            stage = get_current_stage()
            container_prim = stage.GetPrimAtPath(container_prim_path)
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(container_prim)
            rigid_body_api.CreateKinematicEnabledAttr(True)
            UsdPhysics.CollisionAPI.Apply(container_prim)
            
            print(f"Container added at: {container_position}")
            await omni.kit.app.get_app().next_update_async()

            # Add cubes dynamically based on grid parameters
            print(f"=== Adding {total_cubes} Cubes ({self.grid_length}x{self.grid_width} Grid) ===")
            cube_size = 0.0515
            cube_spacing = 0.10  # 10cm spacing between cubes

            # Clear previous cubes list
            self.cubes = []

            # Define color palette for cubes
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

            # Calculate grid center position (in front of robot, farther away)
            grid_center_x = 0.6  # Moved from 0.4 to 0.6 (20cm farther)
            grid_center_y = -0.3  # Moved to the left side

            # Calculate starting position (top-left of grid)
            start_x = grid_center_x - ((self.grid_length - 1) * cube_spacing) / 2.0
            start_y = grid_center_y - ((self.grid_width - 1) * cube_spacing) / 2.0

            cube_index = 0
            for row in range(self.grid_length):
                for col in range(self.grid_width):
                    # Calculate position for this cube
                    cube_x = start_x + (row * cube_spacing)
                    cube_y = start_y + (col * cube_spacing)
                    cube_z = cube_size/2.0 + 0.01  # Slightly above ground
                    cube_position = np.array([cube_x, cube_y, cube_z])

                    # Get color for this cube (cycle through color palette)
                    color, color_name = colors[cube_index % len(colors)]

                    # Create cube
                    timestamp = int(time.time() * 1000) + cube_index
                    cube_name = f"Cube_{row+1}_{col+1}"
                    cube = self.world.scene.add(
                        DynamicCuboid(
                            name=f"cube_{timestamp}",
                            position=cube_position,
                            prim_path=f"/World/Cube_{timestamp}",
                            scale=np.array([cube_size, cube_size, cube_size]),
                            size=1.0,
                            color=color
                        )
                    )

                    # Store cube with its name
                    self.cubes.append((cube, f"{cube_name} ({color_name})"))
                    print(f"  {cube_name} ({color_name}) added at: [{cube_x:.3f}, {cube_y:.3f}, {cube_z:.3f}]")

                    cube_index += 1

            await omni.kit.app.get_app().next_update_async()

            # Initialize physics
            print("=== Initializing Physics ===")
            self.world.initialize_physics()

            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            # Set UR10 default joint positions BEFORE reset
            print("=== Setting UR10 Default Joint Positions ===")
            # Start with robot in a good "home" position
            # Joint order: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
            # This pose puts the end effector at approximately [0.5, 0.0, 0.4] pointing down
            default_joint_positions = np.array([0.0, -2.0, 1.5, -1.07, -1.57, 0.0])
            self.ur10.set_joints_default_state(positions=default_joint_positions)

            # Set gripper default state (opened=True means gripper starts open)
            self.gripper.set_default_state(opened=True)

            # Reset world (this applies the default joint positions)
            print("=== Resetting World ===")
            self.world.reset()

            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()

            # Verify robot is in correct position
            print("=== Verifying Robot Position ===")
            current_joints = self.ur10.get_joint_positions()
            print(f"Current joint positions: {current_joints}")
            print(f"Expected joint positions: {default_joint_positions}")

            # Explicitly set joint positions if they don't match
            if not np.allclose(current_joints, default_joint_positions, atol=0.1):
                print("Joint positions don't match, setting explicitly...")
                self.ur10.set_joint_positions(default_joint_positions)

                for _ in range(20):
                    await omni.kit.app.get_app().next_update_async()

                current_joints = self.ur10.get_joint_positions()
                print(f"Updated joint positions: {current_joints}")

            # Explicitly open gripper to ensure it starts open
            print("=== Opening gripper (initial state) ===")
            self.gripper.open()

            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

            # Setup Task-Space Trajectory Generator
            print("=== Setting up Task-Space Trajectory Generator ===")
            self._setup_trajectory_generator()

            # Verify end effector position
            print("=== Verifying End Effector Position ===")
            current_joints = self.ur10.get_joint_positions()
            ee_pos, ee_rot = self.kinematics_solver.compute_forward_kinematics(
                self._end_effector_name, current_joints
            )
            print(f"End effector position: {ee_pos}")
            print(f"End effector should be around: [0.5, 0.25, 0.4] (approximate)")

            print("\n" + "="*60)
            print("SCENE LOADED SUCCESSFULLY!")
            print("="*60 + "\n")

            # Enable buttons
            self.pick_btn.enabled = True
            self.reset_btn.enabled = True
            self._update_status("Scene loaded! Ready to pick and place")

        except Exception as e:
            self._update_status(f"Error: {e}")
            print(f"Error loading scene: {e}")
            import traceback
            traceback.print_exc()

    def _setup_trajectory_generator(self):
        """Setup Task-Space Trajectory Generator"""
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

        self.taskspace_trajectory_generator = LulaTaskSpaceTrajectoryGenerator(
            robot_description_path=os.path.join(rmp_config_dir, "universal_robots/ur10/rmpflow/ur10_robot_description.yaml"),
            urdf_path=os.path.join(rmp_config_dir, "universal_robots/ur10/ur10_robot.urdf")
        )

        self.kinematics_solver = LulaKinematicsSolver(
            robot_description_path=os.path.join(rmp_config_dir, "universal_robots/ur10/rmpflow/ur10_robot_description.yaml"),
            urdf_path=os.path.join(rmp_config_dir, "universal_robots/ur10/ur10_robot.urdf")
        )

        print("Task-Space Trajectory Generator initialized successfully!")

    def _on_pick(self):
        """Pick and place button callback"""
        if not self.world or not self.taskspace_trajectory_generator:
            self._update_status("Load scene first!")
            return

        if self.is_picking:
            # Pause
            self.is_picking = False
            self._update_status("Paused")
            self.timeline.pause()
        else:
            # Start
            self.is_picking = True
            self._update_status("Starting pick and place...")
            run_coroutine(self._pick_place_loop())

    async def _pick_place_loop(self):
        """Main pick and place loop"""
        try:
            # Start timeline
            self.timeline.play()

            # Minimal wait for timeline to start
            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            print("=== Starting pick and place ===\n")

            # Robot should already be at home position from initialization
            # Wait for robot to settle
            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()

            # Use the dynamically created cubes list
            cubes = self.cubes

            # Get total number of cubes
            total_cubes = len(cubes)
            print(f"Total cubes to pick and place: {total_cubes}\n")

            # Pick and place each cube iteratively
            for i, (cube, cube_name) in enumerate(cubes, 1):
                print(f"\n>>> Picking {cube_name} ({i}/{total_cubes})")
                is_last = (i == total_cubes)  # Check if this is the last cube
                success = await self._pick_and_place_cube(cube, cube_name, is_last)
                if success:
                    self.placed_count += 1
                else:
                    self._update_status(f"Failed to pick {cube_name}")
                    self.is_picking = False
                    return

            print(f"\n=== ALL {total_cubes} CUBES PLACED! ===\n")
            self._update_status(f"All {total_cubes} cubes placed successfully!")
            self.is_picking = False
            self.timeline.stop()

        except Exception as e:
            self._update_status(f"Error: {e}")
            print(f"Error in pick and place: {e}")
            import traceback
            traceback.print_exc()
            self.is_picking = False
            self.timeline.stop()

    async def _pick_and_place_cube(self, cube, cube_name, is_last_cube=False):
        """
        Pick and place a single cube using task-space trajectory

        Args:
            cube: The cube object to pick and place
            cube_name: Name of the cube for logging
            is_last_cube: True if this is the last cube in the sequence
        """
        try:
            # Get current cube position
            cube_pos, _ = cube.get_world_pose()
            print(f"\n=== {cube_name} at position: {cube_pos} ===")

            # Calculate positions
            cube_size = 0.0515
            cube_half = cube_size / 2.0

            # Pick approach: 10cm above cube
            pick_approach = cube_pos + np.array([0.0, 0.0, 0.10])

            # Pick position: VERY close to cube top surface for surface gripper to work
            # Surface gripper needs to be within max_grip_distance (5cm) of the object
            # Cube top is at cube_pos[2] + cube_half, we want to be just above it
            pick_height = cube_pos[2] + cube_half + 0.005  # 5mm above cube top
            pick_pos = np.array([cube_pos[0], cube_pos[1], pick_height])

            # Calculate placement position inside container
            container_center = np.array([0.7, 0.3, 0.0])  # Match container position

            # Spacing between cubes in container
            place_spacing = 0.06  # 6cm spacing in container

            # Calculate which row and column this cube should be placed in
            place_row = self.placed_count // self.grid_width
            place_col = self.placed_count % self.grid_width

            # Calculate offset from container center
            offset_x = (place_row - (self.grid_length - 1) / 2.0) * place_spacing
            offset_y = (place_col - (self.grid_width - 1) / 2.0) * place_spacing
            offset = np.array([offset_x, offset_y, 0.0])

            # Place position: neighboring position at bottom of container
            container_floor_z = 0.0
            place_height = container_floor_z + cube_half + 0.005  # Cube bottom at 0.5cm above floor
            place_pos = container_center + offset + np.array([0.0, 0.0, place_height])

            # Safe transit position: Directly above the PICK position
            safe_transit_height = 0.30
            safe_transit_pos = np.array([cube_pos[0], cube_pos[1], safe_transit_height])

            # Place approach: 15cm above the place position
            place_approach = place_pos + np.array([0.0, 0.0, 0.15])

            # Debug output
            print(f"  → Placing at grid position: Row {place_row + 1}, Column {place_col + 1}")
            print(f"  → Place position: {place_pos} (offset: [{offset[0]:.3f}, {offset[1]:.3f}])")
            print(f"  → Cube bottom will be at Z={place_height - cube_half:.4f}m (container floor at Z=0.0m)")

            # Orientation (gripper pointing down)
            orientation = euler_angles_to_quats(np.array([np.pi, 0, 0]))

            # Home position (safe position between picks)
            home_pos = np.array([0.5, 0.0, 0.4])  # Center, elevated, farther from base

            # Step 1: Move to pick approach
            print(f"Step 1: Moving to pick approach: {pick_approach}")
            success = await self._move_to_target(pick_approach, orientation)
            if not success:
                print("Failed to move to pick approach")
                return False

            # Step 2: Open gripper
            print("Step 2: Opening gripper")
            self.gripper.open()
            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

            # Step 3: Descend to pick position
            print(f"Step 3: Descending to pick: {pick_pos}")
            success = await self._move_to_target(pick_pos, orientation)
            if not success:
                print("Failed to descend to pick")
                return False

            # Wait for robot to settle
            for _ in range(20):
                await omni.kit.app.get_app().next_update_async()

            # Verify end effector is close to cube
            current_joints = self.ur10.get_joint_positions()
            ee_pos, _ = self.kinematics_solver.compute_forward_kinematics(
                self._end_effector_name, current_joints
            )
            distance_to_cube = np.linalg.norm(ee_pos - pick_pos)
            print(f"  End effector at: {ee_pos}")
            print(f"  Target pick pos: {pick_pos}")
            print(f"  Distance to target: {distance_to_cube:.4f}m")

            if distance_to_cube > 0.05:  # More than 5cm away
                print(f"  WARNING: End effector is {distance_to_cube:.4f}m from target (should be < 0.05m)")

            # Step 4: Close gripper
            print("Step 4: Closing gripper to grasp cube")
            self.gripper.close()

            # Wait for gripper to close (retry_interval is 2.0 seconds, so wait at least that long)
            # At 60 FPS, 120 frames = 2 seconds
            for _ in range(150):  # Wait 2.5 seconds
                await omni.kit.app.get_app().next_update_async()

            # Check gripper status
            gripper_status = self.gripper.is_closed()
            print(f"Gripper status after closing: {'Closed' if gripper_status else 'Not Closed'}")

            # Verify cube was picked
            cube_pos_after_pick, _ = cube.get_world_pose()
            z_change = abs(cube_pos_after_pick[2] - cube_pos[2])
            if z_change < 0.001:
                print(f"WARNING: Cube may not have been picked! Position unchanged: {cube_pos_after_pick}")
                print(f"  Original Z: {cube_pos[2]:.4f}, Current Z: {cube_pos_after_pick[2]:.4f}, Change: {z_change:.6f}")
            else:
                print(f"Cube picked successfully! Z changed by {z_change:.4f}m (from {cube_pos[2]:.4f} to {cube_pos_after_pick[2]:.4f})")

            # Step 5: Ascend from pick
            print(f"Step 5: Ascending from pick: {pick_approach}")
            success = await self._move_to_target(pick_approach, orientation)
            if not success:
                print("Failed to ascend from pick")
                return False

            # Step 6: Move to safe transit position
            print(f"Step 6: Moving to safe transit position: {safe_transit_pos}")
            success = await self._move_to_target(safe_transit_pos, orientation)
            if not success:
                print("Failed to move to safe transit position")
                return False

            # Step 7: Descend to place approach
            print(f"Step 7: Descending to place approach: {place_approach}")
            success = await self._move_to_target(place_approach, orientation)
            if not success:
                print("Failed to descend to place approach")
                return False

            # Step 8: Descend to place position
            print(f"Step 8: Descending to place: {place_pos}")
            success = await self._move_to_target(place_pos, orientation)
            if not success:
                print("Failed to descend to place")
                return False

            # Wait for robot to settle
            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

            # Step 9: Open gripper (release cube)
            print("Step 9: Opening gripper to release cube")
            self.gripper.open()
            for _ in range(40):
                await omni.kit.app.get_app().next_update_async()

            # Step 10: Ascend from place
            print(f"Step 10: Ascending from place: {place_approach}")
            success = await self._move_to_target(place_approach, orientation)
            if not success:
                print("Failed to ascend from place")
                return False

            # Check if this is the last cube
            if is_last_cube:
                print("  → Last cube placed, returning to home position")

                # Step 11: Move to safe intermediate position
                safe_intermediate = np.array([0.5, 0.2, 0.35])
                print(f"Step 11: Moving to safe intermediate position: {safe_intermediate}")
                try:
                    success = await self._move_to_target(safe_intermediate, orientation)
                    if not success:
                        print("Warning: Could not move to intermediate position, trying home directly")
                except Exception as e:
                    print(f"Warning: Error moving to intermediate position: {e}")

                # Step 12: Return to home position
                print(f"Step 12: Returning to home position: {home_pos}")
                try:
                    success = await self._move_to_target(home_pos, orientation)
                    if not success:
                        print("Warning: Failed to move to home position, continuing anyway")
                except Exception as e:
                    print(f"Warning: Could not return to home position: {e}")

                # Step 13: Close gripper to reset
                print("Step 13: Closing gripper")
                self.gripper.close()
                for _ in range(50):
                    await omni.kit.app.get_app().next_update_async()
            else:
                print("  → More cubes to pick, moving directly to next cube")

                # Step 11: Close gripper (prepare for next pick)
                print("Step 11: Closing gripper (prepare for next pick)")
                self.gripper.close()
                for _ in range(30):
                    await omni.kit.app.get_app().next_update_async()

            print(f">>> {cube_name} placed successfully!\n")
            return True

        except Exception as e:
            print(f"Error picking {cube_name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _move_to_target(self, target_position, target_orientation, use_intermediate=True):
        """
        Move to target using task-space trajectory

        Args:
            target_position: Target position (3D numpy array)
            target_orientation: Target orientation (quaternion)
            use_intermediate: If True, add intermediate waypoint for smoother motion

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get current end effector pose
            current_joint_positions = self.ur10.get_joint_positions()
            current_ee_position, current_ee_rotation = self.kinematics_solver.compute_forward_kinematics(
                self._end_effector_name, current_joint_positions
            )

            # Convert rotation matrix to quaternion
            from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats
            current_ee_orientation = rot_matrices_to_quats(current_ee_rotation)

            # Check if we need an intermediate waypoint
            distance = np.linalg.norm(target_position - current_ee_position)

            if use_intermediate and distance > 0.2:  # If moving more than 20cm
                # Add intermediate waypoint at midpoint with elevated Z
                mid_position = (current_ee_position + target_position) / 2.0
                mid_position[2] = max(current_ee_position[2], target_position[2]) + 0.1  # Elevate by 10cm

                # Generate trajectory with 3 waypoints: current → mid → target
                positions = np.array([current_ee_position, mid_position, target_position])
                orientations = np.array([current_ee_orientation, target_orientation, target_orientation])

                print(f"  Using intermediate waypoint at {mid_position} (distance: {distance:.3f}m)")
            else:
                # Generate task-space trajectory with 2 waypoints: current → target
                positions = np.array([current_ee_position, target_position])
                orientations = np.array([current_ee_orientation, target_orientation])

            trajectory = self.taskspace_trajectory_generator.compute_task_space_trajectory_from_points(
                positions, orientations, self._end_effector_name
            )

            if trajectory is None:
                print(f"Failed to compute trajectory to {target_position}")
                print(f"  Current position: {current_ee_position}")
                print(f"  Target position: {target_position}")
                print(f"  Distance: {distance:.3f}m")
                return False

            # Convert to articulation actions
            physics_dt = 1 / 60
            articulation_trajectory = ArticulationTrajectory(self.ur10, trajectory, physics_dt)
            action_sequence = articulation_trajectory.get_action_sequence()

            # Execute trajectory
            for action in action_sequence:
                self.ur10.apply_action(action)
                await omni.kit.app.get_app().next_update_async()

            return True

        except Exception as e:
            print(f"Error moving to target {target_position}: {e}")
            import traceback
            traceback.print_exc()
            return False

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
            from isaacsim.core.utils.stage import clear_stage
            clear_stage()

            # Reset state
            self.world = None
            self.ur10 = None
            self.gripper = None
            self.container = None
            self.taskspace_trajectory_generator = None
            self.kinematics_solver = None
            self.cubes = []
            self.placed_count = 0

            # Reset UI
            self.load_btn.enabled = True
            self.pick_btn.enabled = False
            self.reset_btn.enabled = False

            self._update_status("Reset complete - Stage cleared")
            print("\n=== RESET COMPLETE - STAGE CLEARED ===\n")

        except Exception as e:
            self._update_status(f"Error resetting: {e}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


# Create and show UI
print("\n" + "="*60)
print("UR10 Pick and Place - Task-Space Trajectory with Surface Gripper")
print("="*60 + "\n")

app = UR10PickPlaceTaskSpace()
print("UI loaded! Configure grid and use the buttons to control the robot.")
print("="*60 + "\n")

