"""
UR10 Pick and Place with C-Space Trajectory + IK
Uses LulaKinematicsSolver (IK) + LulaCSpaceTrajectoryGenerator
Simple and reliable approach for pick-and-place
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

# Isaac Sim imports
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.types import ArticulationAction
from pxr import UsdPhysics
import usd.schema.isaac.robot_schema as robot_schema

# Motion generation imports
from isaacsim.robot_motion.motion_generation import (
    LulaCSpaceTrajectoryGenerator,
    LulaKinematicsSolver,
    ArticulationTrajectory
)

# Add project root to path for local imports
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.manipulators import SingleManipulator
from src.grippers import SurfaceGripper


class UR10CSpaceIK:
    """UR10 Pick and Place using C-Space Trajectory + IK"""

    def __init__(self):
        self.window = None
        self.world = None
        self.ur10 = None
        self.gripper = None
        self.container = None
        self.cspace_generator = None
        self.kinematics_solver = None

        # Dynamic cube list
        self.cubes = []

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

        # End effector name
        self._end_effector_name = "ee_link"

        self.build_ui()
    
    def build_ui(self):
        """Build UI"""
        self.window = ui.Window("UR10 - C-Space + IK Pick & Place", width=450, height=400)

        with self.window.frame:
            with ui.VStack(spacing=10):
                ui.Label("UR10 - C-Space Trajectory + IK",
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

                ui.Spacer(height=10)

                # Control Buttons
                with ui.VStack(spacing=5):
                    self.load_btn = ui.Button("Load Scene", height=40, clicked_fn=self._on_load)
                    self.pick_btn = ui.Button("Start Pick and Place", height=40, clicked_fn=self._on_pick, enabled=False)
                    self.reset_btn = ui.Button("Reset Scene", height=40, clicked_fn=self._on_reset)

                ui.Spacer(height=10)

                # Status
                with ui.CollapsableFrame("Status", collapsed=False):
                    with ui.VStack(spacing=5):
                        self.status_label = ui.Label("Ready", word_wrap=True)

    def _update_status(self, message):
        """Update status label"""
        if self.status_label:
            self.status_label.text = message
        print(f"[STATUS] {message}")

    def _on_load(self):
        """Load scene button callback"""
        self._update_status("Loading scene...")
        run_coroutine(self._load_scene())

    def _on_pick(self):
        """Pick and place button callback"""
        if not self.world or not self.cspace_generator:
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

    def _on_reset(self):
        """Reset button callback"""
        try:
            self.is_picking = False

            # Stop timeline first
            self.timeline.stop()

            # Clear World instance
            if self.world is not None:
                World.clear_instance()
                self.world = None

            # Delete all prims from stage
            stage = omni.usd.get_context().get_stage()
            if stage:
                world_prim = stage.GetPrimAtPath("/World")
                if world_prim.IsValid():
                    stage.RemovePrim("/World")

            # Reset state
            self.ur10 = None
            self.gripper = None
            self.container = None
            self.cspace_generator = None
            self.kinematics_solver = None
            self.cubes = []
            self.placed_count = 0

            # Update UI
            if self.pick_btn:
                self.pick_btn.enabled = False

            self._update_status("Scene reset complete")

        except Exception as e:
            self._update_status(f"Reset error: {e}")
            import traceback
            traceback.print_exc()

    async def _load_scene(self):
        """Load the scene with UR10, cubes, and container"""
        try:
            # Get grid dimensions from UI
            self.grid_length = int(self.length_field.model.get_value_as_int())
            self.grid_width = int(self.width_field.model.get_value_as_int())

            # Validate grid dimensions
            if self.grid_length < 1 or self.grid_width < 1:
                self._update_status("Grid dimensions must be at least 1x1")
                return
            if self.grid_length > 5 or self.grid_width > 5:
                self._update_status("Grid dimensions cannot exceed 5x5")
                return

            # Create World
            print("=== Creating World ===")
            self.world = World()
            await self.world.initialize_simulation_context_async()

            # Add ground plane
            print("=== Adding Ground ===")
            self.world.scene.add_default_ground_plane()

            # Add UR10 with Surface Gripper
            print("=== Adding UR10 with Surface Gripper ===")
            assets_root_path = get_assets_root_path()
            ur10_usd_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"

            # Generate unique name with timestamp
            ur10_name = f"ur10_{int(time.time() * 1000)}"
            ur10_prim_path = f"/World/{ur10_name}"

            # Add UR10 to stage
            add_reference_to_stage(usd_path=ur10_usd_path, prim_path=ur10_prim_path)

            # Set gripper variant to Short_Suction
            stage = get_current_stage()
            robot_prim = stage.GetPrimAtPath(ur10_prim_path)
            robot_prim.GetVariantSet("Gripper").SetVariantSelection("Short_Suction")

            # Configure Surface Gripper properties
            print("=== Configuring Surface Gripper ===")
            gripper_prim_path = f"{ur10_prim_path}/ee_link/SurfaceGripper"
            gripper_prim = stage.GetPrimAtPath(gripper_prim_path)
            
            if gripper_prim.IsValid():
                gripper_prim.GetAttribute(robot_schema.Attributes.MAX_GRIP_DISTANCE.name).Set(0.05)
                gripper_prim.GetAttribute(robot_schema.Attributes.COAXIAL_FORCE_LIMIT.name).Set(100.0)
                gripper_prim.GetAttribute(robot_schema.Attributes.SHEAR_FORCE_LIMIT.name).Set(100.0)
                gripper_prim.GetAttribute(robot_schema.Attributes.RETRY_INTERVAL.name).Set(2.0)
                print(f"Surface gripper configured at: {gripper_prim_path}")
            
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

            # Add container
            print("=== Adding Container ===")
            container_position = np.array([0.9, 0.4, 0.0])
            self.container = self._create_container(container_position)
            print(f"Container added at: {container_position}")

            # Add cubes in grid
            print(f"=== Adding {self.grid_length * self.grid_width} Cubes ({self.grid_length}x{self.grid_width} Grid) ===")
            self._create_cube_grid()

            # Initialize physics
            print("=== Initializing Physics ===")
            await self.world.reset_async()

            # Set UR10 default joint positions
            print("=== Setting UR10 Default Joint Positions ===")
            default_joint_positions = np.array([0.0, -2.0, 1.5, -1.07, -1.57, 0.0])
            self.ur10.set_joints_default_state(positions=default_joint_positions)
            self.gripper.set_default_state(opened=True)

            # Reset world
            print("=== Resetting World ===")
            await self.world.reset_async()

            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()

            # Open gripper
            print("=== Opening gripper (initial state) ===")
            self.gripper.open()

            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()
            
            # Setup C-Space Trajectory Generator + IK
            print("=== Setting up C-Space Trajectory Generator + IK ===")
            self._setup_motion_generation()

            print("\n" + "="*60)
            print("SCENE LOADED SUCCESSFULLY!")
            print("="*60 + "\n")

            # Enable pick button
            if self.pick_btn:
                self.pick_btn.enabled = True

            self._update_status(f"Scene loaded! {self.grid_length}x{self.grid_width} grid ready")

        except Exception as e:
            self._update_status(f"Load error: {e}")
            import traceback
            traceback.print_exc()

    def _setup_motion_generation(self):
        """Setup C-Space Trajectory Generator and IK Solver"""
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

        robot_description_path = os.path.join(rmp_config_dir, "universal_robots/ur10/rmpflow/ur10_robot_description.yaml")
        urdf_path = os.path.join(rmp_config_dir, "universal_robots/ur10/ur10_robot.urdf")

        # Create C-Space Trajectory Generator
        self.cspace_generator = LulaCSpaceTrajectoryGenerator(
            robot_description_path=robot_description_path,
            urdf_path=urdf_path
        )

        # Create IK Solver
        self.kinematics_solver = LulaKinematicsSolver(
            robot_description_path=robot_description_path,
            urdf_path=urdf_path
        )

        print("C-Space Trajectory Generator + IK Solver initialized!")

    def _create_container(self, position):
        """Create container for placing cubes"""
        container_size = np.array([0.15, 0.15, 0.05])  # 15cm x 15cm x 5cm
        container = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Container",
                name="Container",
                position=position,
                scale=container_size,
                color=np.array([0.5, 0.5, 0.5]),  # Gray
                mass=100.0  # Heavy so it doesn't move
            )
        )
        return container

    def _create_cube_grid(self):
        """Create cubes in a grid pattern"""
        cube_size = 0.0515  # 5.15cm cubes
        cube_spacing = 0.10  # 10cm spacing between cubes

        # Grid center position
        grid_center_x = 0.7
        grid_center_y = -0.4

        # Color palette
        colors = [
            ("Blue", np.array([0.0, 0.0, 1.0])),
            ("Red", np.array([1.0, 0.0, 0.0])),
            ("Green", np.array([0.0, 1.0, 0.0])),
            ("Yellow", np.array([1.0, 1.0, 0.0])),
            ("Magenta", np.array([1.0, 0.0, 1.0])),
            ("Cyan", np.array([0.0, 1.0, 1.0])),
            ("Orange", np.array([1.0, 0.5, 0.0])),
            ("Purple", np.array([0.5, 0.0, 0.5])),
            ("Gray", np.array([0.5, 0.5, 0.5])),
            ("Pink", np.array([1.0, 0.75, 0.8]))
        ]

        # Calculate grid start position
        grid_start_x = grid_center_x - ((self.grid_length - 1) * cube_spacing) / 2.0
        grid_start_y = grid_center_y - ((self.grid_width - 1) * cube_spacing) / 2.0

        # Create cubes
        cube_index = 0
        for row in range(self.grid_length):
            for col in range(self.grid_width):
                # Calculate position
                x = grid_start_x + (row * cube_spacing)
                y = grid_start_y + (col * cube_spacing)
                z = cube_size / 2.0 + 0.01  # Half cube height + 1cm above ground

                # Get color
                color_name, color_rgb = colors[cube_index % len(colors)]

                # Create cube
                cube_name = f"Cube_{row+1}_{col+1}"
                cube = self.world.scene.add(
                    DynamicCuboid(
                        prim_path=f"/World/{cube_name}",
                        name=cube_name,
                        position=np.array([x, y, z]),
                        scale=np.array([cube_size, cube_size, cube_size]),
                        color=color_rgb,
                        mass=0.05  # 50g
                    )
                )

                # Store cube with metadata
                self.cubes.append({
                    'object': cube,
                    'name': cube_name,
                    'color': color_name,
                    'row': row + 1,
                    'col': col + 1
                })

                print(f"  {cube_name} ({color_name}) added at: [{x:.3f}, {y:.3f}, {z:.3f}]")

                cube_index += 1

    async def _pick_place_loop(self):
        """Main pick and place loop"""
        try:
            # Start timeline
            self.timeline.play()

            # Wait for timeline to start
            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            print("=== Starting pick and place ===\n")

            # Wait for robot to settle
            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()

            # Use the dynamically created cubes list
            cubes = self.cubes
            total_cubes = len(cubes)
            print(f"Total cubes to pick and place: {total_cubes}\n")

            # Pick and place each cube
            for idx, cube_data in enumerate(cubes):
                if not self.is_picking:
                    print("Pick and place paused by user")
                    break

                cube = cube_data['object']
                cube_name = cube_data['name']
                color_name = cube_data['color']
                is_last_cube = (idx == total_cubes - 1)

                print(f"\n>>> Picking {cube_name} ({color_name}) ({idx+1}/{total_cubes})\n")

                success = await self._pick_and_place_cube(cube, cube_name, color_name, idx, is_last_cube)

                if not success:
                    print(f"Failed to pick and place {cube_name}")
                    break

                print(f">>> {cube_name} ({color_name}) placed successfully!\n")

            if self.is_picking:
                print("\n" + "="*60)
                print(f"=== ALL {total_cubes} CUBES PLACED! ===")
                print("="*60 + "\n")
                self._update_status(f"All {total_cubes} cubes placed!")
                self.is_picking = False

        except Exception as e:
            print(f"Error in pick and place loop: {e}")
            import traceback
            traceback.print_exc()
            self.is_picking = False
            self._update_status(f"Error: {e}")

    async def _pick_and_place_cube(self, cube, cube_name, color_name, cube_index, is_last_cube=False):
        """Pick and place a single cube using C-Space Trajectory + IK"""
        try:
            # Get cube position
            cube_pos, _ = cube.get_world_pose()
            print(f"=== {cube_name} ({color_name}) at position: {cube_pos} ===")

            # Calculate place position in container
            cube_size = 0.0515
            container_pos, _ = self.container.get_world_pose()
            container_size = np.array([0.15, 0.15, 0.05])

            # Calculate grid position in container
            cubes_per_row = 2
            row = cube_index // cubes_per_row
            col = cube_index % cubes_per_row

            # Spacing between cubes in container
            place_spacing = 0.06

            # Calculate offsets from container center
            offset_x = (col - 0.5) * place_spacing
            offset_y = (row - 0.5) * place_spacing

            # Place position (on top of container)
            place_pos = container_pos + np.array([offset_x, offset_y, container_size[2]/2 + cube_size/2 + 0.005])

            print(f"  → Placing at grid position: Row {row+1}, Column {col+1}")
            print(f"  → Place position: {place_pos}")

            # Calculate positions
            cube_half = cube_size / 2.0

            # Pick approach: 10cm above cube
            pick_approach = cube_pos + np.array([0.0, 0.0, 0.10])

            # Pick position: 5mm above cube top
            pick_height = cube_pos[2] + cube_half + 0.005
            pick_pos = np.array([cube_pos[0], cube_pos[1], pick_height])

            # Place approach: 15cm above place position
            place_approach = place_pos + np.array([0.0, 0.0, 0.15])

            # Gripper orientation (pointing down)
            orientation = euler_angles_to_quats(np.array([np.pi, 0, 0]))

            # Step 1: Move to pick approach using C-Space + IK
            print(f"Step 1: Moving to pick approach: {pick_approach}")
            success = await self._move_to_target_cspace(pick_approach, orientation)
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
            success = await self._move_to_target_cspace(pick_pos, orientation)
            if not success:
                print("Failed to descend to pick position")
                return False

            # Wait for robot to settle
            for _ in range(20):
                await omni.kit.app.get_app().next_update_async()

            # Step 4: Close gripper
            print("Step 4: Closing gripper to grasp cube")
            self.gripper.close()

            # Wait for gripper to close
            for _ in range(150):
                await omni.kit.app.get_app().next_update_async()

            # Check gripper status
            gripper_status = self.gripper.is_closed()
            print(f"Gripper status: {'Closed' if gripper_status else 'Not Closed'}")

            # Step 5: Ascend from pick
            print(f"Step 5: Ascending from pick: {pick_approach}")
            success = await self._move_to_target_cspace(pick_approach, orientation)
            if not success:
                print("Failed to ascend from pick")
                return False

            # Step 6: Move to place approach
            print(f"Step 6: Moving to place approach: {place_approach}")
            success = await self._move_to_target_cspace(place_approach, orientation)
            if not success:
                print("Failed to move to place approach")
                return False

            # Step 7: Descend to place position
            print(f"Step 7: Descending to place: {place_pos}")
            success = await self._move_to_target_cspace(place_pos, orientation)
            if not success:
                print("Failed to descend to place position")
                return False

            # Wait for robot to settle
            for _ in range(20):
                await omni.kit.app.get_app().next_update_async()

            # Step 8: Open gripper (release)
            print("Step 8: Opening gripper to release cube")
            self.gripper.open()
            for _ in range(50):
                await omni.kit.app.get_app().next_update_async()

            # Step 9: Ascend from place
            print(f"Step 9: Ascending from place: {place_approach}")
            success = await self._move_to_target_cspace(place_approach, orientation)
            if not success:
                print("Failed to ascend from place")
                return False

            # Step 10: Return to home (only for last cube)
            if is_last_cube:
                print("  → Last cube placed, returning to home position")
                home_pos = np.array([0.6, 0.0, 0.5])
                print(f"Step 10: Returning to home position: {home_pos}")
                success = await self._move_to_target_cspace(home_pos, orientation)
                if not success:
                    print("Warning: Failed to return to home position")

                # Close gripper
                print("Step 11: Closing gripper")
                self.gripper.close()
                for _ in range(30):
                    await omni.kit.app.get_app().next_update_async()

            return True

        except Exception as e:
            print(f"Error picking {cube_name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _move_to_target_cspace(self, target_position, target_orientation):
        """
        Move to target using C-Space Trajectory + IK

        Args:
            target_position: Target position (3D numpy array)
            target_orientation: Target orientation (quaternion)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Step 1: Use IK to compute target joint positions
            target_joints, success = self.kinematics_solver.compute_inverse_kinematics(
                self._end_effector_name,
                target_position,
                target_orientation
            )

            if not success:
                print(f"  IK failed for target: {target_position}")
                return False

            print(f"  IK solved! Target joints: {target_joints}")

            # Step 2: Get current joint positions
            current_joints = self.ur10.get_joint_positions()

            # Step 3: Generate C-Space trajectory from current to target
            trajectory = self.cspace_generator.compute_c_space_trajectory(
                current_joints,
                target_joints
            )

            # Step 4: Convert to ArticulationTrajectory
            articulation_trajectory = ArticulationTrajectory(
                robot_articulation=self.ur10,
                trajectory=trajectory
            )

            # Step 5: Execute trajectory
            print(f"  Executing trajectory ({len(articulation_trajectory.get_action_sequence())} actions)...")

            actions = articulation_trajectory.get_action_sequence()
            for action in actions:
                self.ur10.apply_action(action)
                await omni.kit.app.get_app().next_update_async()

            # Wait for robot to settle
            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()

            # Verify we reached the target
            final_joints = self.ur10.get_joint_positions()
            joint_error = np.linalg.norm(final_joints - target_joints)
            print(f"  Joint error: {joint_error:.4f} rad")

            if joint_error > 0.1:  # 0.1 rad tolerance
                print(f"  WARNING: Large joint error! Expected: {target_joints}, Got: {final_joints}")
                return False

            return True

        except Exception as e:
            print(f"Error moving to target {target_position}: {e}")
            import traceback
            traceback.print_exc()
            return False


# Main entry point
if __name__ == "__main__":
    app = UR10CSpaceIK()
    print("UR10 C-Space + IK Pick and Place UI loaded!")
    print("Click 'Load Scene' to start")

