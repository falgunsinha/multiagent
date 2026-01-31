"""
Franka RRT Pick and Place - Dynamic Grid
Supports variable number of cubes in a grid pattern
Container placed in front for easy reach
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

# Isaac Sim imports
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import PathPlannerVisualizer
from isaacsim.robot_motion.motion_generation.lula import RRT
from pxr import UsdPhysics

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

        self.build_ui()
    
    def build_ui(self):
        """Build UI"""
        self.window = ui.Window("Franka RRT - Dynamic Grid", width=450, height=400)

        with self.window.frame:
            with ui.VStack(spacing=10):
                ui.Label("Franka RRT Pick and Place - Dynamic Grid",
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
            
            # Add Franka
            print("=== Adding Franka ===")
            franka_name = f"franka_{int(time.time() * 1000)}"
            franka_prim_path = f"/World/Franka_{int(time.time() * 1000)}"
            
            franka_usd_path = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
            robot_prim = add_reference_to_stage(usd_path=franka_usd_path, prim_path=franka_prim_path)
            robot_prim.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
            robot_prim.GetVariantSet("Mesh").SetVariantSelection("Quality")
            
            await omni.kit.app.get_app().next_update_async()
            
            # Create gripper (tighter close for better grip)
            self.gripper = ParallelGripper(
                end_effector_prim_path=f"{franka_prim_path}/panda_rightfinger",
                joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
                joint_opened_positions=np.array([0.05, 0.05]),
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
            
            print(f"Franka added: {franka_name}")
            await omni.kit.app.get_app().next_update_async()
            
            # Add container IN FRONT of robot (positive X direction)
            print("=== Adding Container ===")
            container_prim_path = "/World/Container"
            container_usd_path = f"{get_assets_root_path()}/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Storage/Containers/Container_I/Container_I04_160x120x64cm_PR_V_NVD_01.usd"
            
            add_reference_to_stage(usd_path=container_usd_path, prim_path=container_prim_path)
            
            # Container position: IN FRONT (positive X), to the right (positive Y)
            # Reduced X from 0.6 to 0.55 and Y from 0.5 to 0.4 to be within robot's comfortable reach
            container_position = np.array([0.55, 0.4, 0.0])
            
            # Scale: X=0.3, Y=0.3, Z=0.2 (reduced height to lower rim)
            # Original: 160x120x64cm → Scaled: 48x36x12.8cm (instead of 19.2cm)
            self.container = self.world.scene.add(
                SingleXFormPrim(
                    prim_path=container_prim_path,
                    name="container",
                    translation=container_position,
                    scale=np.array([0.3, 0.3, 0.2])  # Reduced Z from 0.3 to 0.2
                )
            )
            
            # Add physics to container
            from omni.isaac.core.utils.stage import get_current_stage
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

            # Calculate grid center position (in front of robot)
            # Grid will be centered around X=0.4, Y=0.0
            grid_center_x = 0.4
            grid_center_y = 0.0

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
            
            # Reset world
            print("=== Resetting World ===")
            self.world.reset()
            
            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()
            
            # Set Franka default joint positions
            print("=== Initializing Franka ===")
            # Start with gripper CLOSED (ready to open when approaching cube)
            default_joint_positions = np.array([0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75, 0.0, 0.0])
            self.franka.set_joints_default_state(positions=default_joint_positions)
            self.gripper.set_default_state(self.gripper.joint_closed_positions)

            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            # Explicitly close gripper to ensure it starts closed
            print("=== Closing gripper (initial state) ===")
            articulation_controller = self.franka.get_articulation_controller()
            close_action = ArticulationAction(
                joint_positions=self.gripper.joint_closed_positions,
                joint_indices=np.array([7, 8])
            )
            articulation_controller.apply_action(close_action)

            for _ in range(20):
                await omni.kit.app.get_app().next_update_async()
            
            # Setup RRT
            print("=== Setting up RRT ===")
            self._setup_rrt()
            
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

    def _setup_rrt(self):
        """Setup RRT motion planner"""
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")

        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs", "franka", "rmpflow")
        robot_description_file = os.path.join(rmp_config_dir, "robot_descriptor.yaml")
        urdf_path = os.path.join(mg_extension_path, "motion_policy_configs", "franka", "lula_franka_gen.urdf")
        rrt_config_file = os.path.join(mg_extension_path, "path_planner_configs", "franka", "rrt", "franka_planner_config.yaml")

        self.rrt = RRT(
            robot_description_path=robot_description_file,
            urdf_path=urdf_path,
            rrt_config_path=rrt_config_file,
            end_effector_frame_name="right_gripper"
        )

        self.rrt.set_max_iterations(10000)

        self.path_planner_visualizer = PathPlannerVisualizer(
            robot_articulation=self.franka,
            path_planner=self.rrt
        )

        print("RRT initialized successfully!")

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

            # Use the dynamically created cubes list
            cubes = self.cubes

            # Get total number of cubes
            total_cubes = len(cubes)
            print(f"Total cubes to pick and place: {total_cubes}\n")

            # Pick and place each cube iteratively
            for i, (cube, cube_name) in enumerate(cubes, 1):
                print(f"\n>>> Picking {cube_name} ({i}/{total_cubes})")
                is_last = (i == total_cubes)  # Check if this is the last cube
                success = await self._pick_and_place_cube(cube, cube_name.split()[1], is_last)  # Extract "1", "2", etc.
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
        Pick and place a single cube

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

            # NOTE: RRT uses the "right_gripper" frame from URDF, which already accounts for
            # the end effector position. We should NOT add manual offsets here.
            # The "right_gripper" frame is different from "panda_rightfinger" USD prim.

            # Pick approach: 15cm above cube
            pick_approach = cube_pos + np.array([0.0, 0.0, 0.15])

            # Pick position: slightly below cube center for better gripping
            # Cube is 5.15cm tall, center is at 2.575cm from bottom
            # Pick at 2cm height (0.575cm below center) so gripper fingers wrap around cube better
            pick_pos = np.array([cube_pos[0], cube_pos[1], 0.02])

            # Calculate placement position inside container
            # Container: 48cm x 36cm x 12.8cm (scale 0.3, 0.3, 0.2 from 160x120x64cm)
            # Container center: [0.55, 0.4, 0.0]
            container_center = np.array([0.55, 0.4, 0.0])

            # Calculate placement position dynamically based on grid
            # Place cubes in same grid pattern inside container

            # Spacing between cubes in container (slightly smaller than pick grid)
            place_spacing = 0.06  # 6cm spacing in container

            # Calculate which row and column this cube should be placed in
            # Use same grid dimensions as pick grid
            place_row = self.placed_count // self.grid_width
            place_col = self.placed_count % self.grid_width

            # Calculate offset from container center
            # Center the grid inside the container
            offset_x = (place_row - (self.grid_length - 1) / 2.0) * place_spacing
            offset_y = (place_col - (self.grid_width - 1) / 2.0) * place_spacing
            offset = np.array([offset_x, offset_y, 0.0])

            # Place position: neighboring position at bottom of container
            # Container bottom is at Z=0, place cube so its bottom sits on container floor
            # Add small clearance (0.5cm) to avoid collision with container floor
            container_floor_z = 0.0
            place_height = container_floor_z + cube_half + 0.005  # Cube bottom at 0.5cm above floor
            place_pos = container_center + offset + np.array([0.0, 0.0, place_height])

            # Safe transit position: Directly above the PICK position (not container)
            # This ensures robot lifts straight up after picking, then moves horizontally
            # Use 30cm height to clear container rim (12.8cm) with good margin
            safe_transit_height = 0.30
            safe_transit_pos = np.array([cube_pos[0], cube_pos[1], safe_transit_height])

            # Place approach: 15cm above the place position (but still need safe transit first)
            place_approach = place_pos + np.array([0.0, 0.0, 0.15])

            # Debug output
            print(f"  → Placing at grid position: Row {place_row + 1}, Column {place_col + 1}")
            print(f"  → Place position: {place_pos} (offset: [{offset[0]:.3f}, {offset[1]:.3f}])")
            print(f"  → Cube bottom will be at Z={place_height - cube_half:.4f}m (container floor at Z=0.0m)")

            # Orientation (gripper pointing down)
            orientation = euler_angles_to_quats(np.array([np.pi, 0, 0]))

            # Home position (safe position between picks)
            home_pos = np.array([0.4, 0.0, 0.3])  # Center, elevated

            # Step 1: Move to pick approach
            print(f"Step 1: Moving to pick approach: {pick_approach}")
            plan = self._plan_to_target(pick_approach, orientation)
            if plan is None:
                print("Failed to plan to pick approach")
                return False
            await self._execute_plan(plan)

            # Step 2: Open gripper (set to fully open position)
            print("Step 2: Opening gripper fully")
            articulation_controller = self.franka.get_articulation_controller()
            # Apply open position directly to gripper joints (indices 7 and 8)
            open_action = ArticulationAction(
                joint_positions=self.gripper.joint_opened_positions,
                joint_indices=np.array([7, 8])
            )
            articulation_controller.apply_action(open_action)
            # Wait longer for gripper to fully open before descending
            print("  Waiting for gripper to fully open...")
            for _ in range(50):
                await omni.kit.app.get_app().next_update_async()

            # Step 3: Descend to pick position
            print(f"Step 3: Descending to pick: {pick_pos}")
            plan = self._plan_to_target(pick_pos, orientation)
            if plan is None:
                print("Failed to plan descent to pick")
                return False
            await self._execute_plan(plan)

            # Wait for robot to settle at pick position
            print("  Waiting for robot to settle at pick position...")
            for _ in range(20):
                await omni.kit.app.get_app().next_update_async()

            # Step 4: Close gripper (set to fully closed position)
            print("Step 4: Closing gripper to grasp cube")
            articulation_controller = self.franka.get_articulation_controller()
            close_action = ArticulationAction(
                joint_positions=self.gripper.joint_closed_positions,
                joint_indices=np.array([7, 8])
            )
            articulation_controller.apply_action(close_action)
            # Wait for gripper to close and grip cube
            for _ in range(50):
                await omni.kit.app.get_app().next_update_async()

            # Verify cube was picked by checking if it moved
            cube_pos_after_pick, _ = cube.get_world_pose()
            if abs(cube_pos_after_pick[2] - cube_pos[2]) < 0.001:
                print(f"WARNING: Cube may not have been picked! Position unchanged: {cube_pos_after_pick}")
            else:
                print(f"Cube picked successfully! New Z position: {cube_pos_after_pick[2]:.4f}")

            # Step 5: Ascend from pick
            print(f"Step 5: Ascending from pick: {pick_approach}")
            plan = self._plan_to_target(pick_approach, orientation)
            if plan is None:
                print("Failed to plan ascent from pick")
                return False
            await self._execute_plan(plan)

            # Step 6: Move to safe transit position (high above container to avoid collision)
            print(f"Step 6: Moving to safe transit position: {safe_transit_pos} (Z={safe_transit_height}m)")
            plan = self._plan_to_target(safe_transit_pos, orientation)
            if plan is None:
                print("Failed to plan to safe transit position")
                return False
            await self._execute_plan(plan)

            # Step 7: Descend to place approach (15cm above final position)
            print(f"Step 7: Descending to place approach: {place_approach}")
            plan = self._plan_to_target(place_approach, orientation)
            if plan is None:
                print("Failed to plan to place approach")
                return False
            await self._execute_plan(plan)

            # Step 8: Descend to place position
            print(f"Step 8: Descending to place: {place_pos}")
            plan = self._plan_to_target(place_pos, orientation)
            if plan is None:
                print("Failed to plan descent to place")
                return False
            await self._execute_plan(plan)

            # Wait for robot to settle at place position before releasing
            print("  Waiting for robot to settle at place position...")
            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

            # Step 9: Open gripper (release cube)
            print("Step 9: Opening gripper to release cube")
            articulation_controller = self.franka.get_articulation_controller()
            open_action = ArticulationAction(
                joint_positions=self.gripper.joint_opened_positions,
                joint_indices=np.array([7, 8])
            )
            articulation_controller.apply_action(open_action)

            # Wait for gripper to open and cube to settle
            print("  Waiting for cube to settle...")
            for _ in range(40):
                await omni.kit.app.get_app().next_update_async()

            # Step 10: Ascend from place
            print(f"Step 10: Ascending from place: {place_approach}")
            plan = self._plan_to_target(place_approach, orientation)
            if plan is None:
                print("Failed to plan ascent from place")
                return False
            print(f"  Plan has {len(plan)} actions")
            await self._execute_plan(plan)
            # Wait for robot to settle
            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()

            # Check if this is the last cube (passed as parameter)
            if is_last_cube:
                # Only return to home after placing the LAST cube
                print("  → Last cube placed, returning to home position")

                # Step 11: Move to safe intermediate position (helps avoid invalid configurations)
                # Position between container and home, at safe height
                # Adjusted to match new container position [0.55, 0.4, 0.0]
                safe_intermediate = np.array([0.5, 0.2, 0.35])
                print(f"Step 11: Moving to safe intermediate position: {safe_intermediate}")
                try:
                    plan = self._plan_to_target(safe_intermediate, orientation)
                    if plan is not None:
                        print(f"  Plan has {len(plan)} actions")
                        await self._execute_plan(plan)
                        # Wait for robot to settle
                        for _ in range(10):
                            await omni.kit.app.get_app().next_update_async()
                    else:
                        print("Warning: Could not plan to intermediate position, trying home directly")
                except Exception as e:
                    print(f"Warning: Error moving to intermediate position: {e}")

                # Step 12: Return to home position (safe configuration for next pick)
                print(f"Step 12: Returning to home position: {home_pos}")
                try:
                    plan = self._plan_to_target(home_pos, orientation)
                    if plan is None:
                        print("Warning: Failed to plan to home position, continuing anyway")
                    else:
                        print(f"  Plan has {len(plan)} actions")
                        await self._execute_plan(plan)
                        # Wait for robot to settle at home
                        for _ in range(10):
                            await omni.kit.app.get_app().next_update_async()
                except Exception as e:
                    print(f"Warning: Could not return to home position: {e}")
                    print("Continuing anyway...")

                # Step 13: Close gripper to reset
                print("Step 13: Closing gripper")
                articulation_controller = self.franka.get_articulation_controller()
                close_action = ArticulationAction(
                    joint_positions=self.gripper.joint_closed_positions,
                    joint_indices=np.array([7, 8])
                )
                articulation_controller.apply_action(close_action)
                # Wait for gripper to close
                for _ in range(50):
                    await omni.kit.app.get_app().next_update_async()
            else:
                # For cubes 1-3, just close gripper and move directly to next pick
                print("  → More cubes to pick, moving directly to next cube")

                # Step 11: Close gripper (prepare for next pick)
                print("Step 11: Closing gripper (prepare for next pick)")
                articulation_controller = self.franka.get_articulation_controller()
                close_action = ArticulationAction(
                    joint_positions=self.gripper.joint_closed_positions,
                    joint_indices=np.array([7, 8])
                )
                articulation_controller.apply_action(close_action)
                # Wait for gripper to close
                for _ in range(30):  # Shorter wait since we're continuing
                    await omni.kit.app.get_app().next_update_async()

            print(f">>> {cube_name} placed successfully!\n")
            return True

        except Exception as e:
            print(f"Error picking {cube_name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _plan_to_target(self, target_position, target_orientation):
        """Plan path to target using RRT"""
        self.rrt.set_end_effector_target(target_position, target_orientation)
        self.rrt.update_world()
        # Use moderate resolution (0.02) for balance between performance and reliability
        plan = self.path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=0.02)
        return plan

    async def _execute_plan(self, plan):
        """Execute a plan"""
        if plan is None:
            return False

        # Execute plan - wait every frame for reliable motion
        for action in plan:
            self.franka.apply_action(action)
            await omni.kit.app.get_app().next_update_async()

        return True

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
            self.cube1 = None
            self.cube2 = None
            self.cube1_picked = False
            self.cube2_picked = False
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
print("Franka RRT - Dynamic Grid Pick and Place")
print("="*60 + "\n")

app = FrankaRRTDynamicGrid()
print("UI loaded! Configure grid and use the buttons to control the robot.")
print("="*60 + "\n")

