"""
Franka RRT Pick and Place - 4 Cuboids
Version for testing with 4 cubes in front of robot
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


class FrankaRRT4Cubes:
    """4-cube pick and place with RRT"""

    def __init__(self):
        self.window = None
        self.world = None
        self.franka = None
        self.gripper = None
        self.container = None
        self.rrt = None
        self.path_planner_visualizer = None

        # Cubes
        self.cube1 = None
        self.cube2 = None
        self.cube3 = None
        self.cube4 = None

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
        
        self.build_ui()
    
    def build_ui(self):
        """Build UI"""
        self.window = ui.Window("Franka RRT - 4 Cubes", width=400, height=300)

        with self.window.frame:
            with ui.VStack(spacing=10):
                ui.Label("Franka RRT Pick and Place - 4 Cubes",
                        alignment=ui.Alignment.CENTER,
                        style={"font_size": 18})
                
                ui.Spacer(height=10)
                
                # Buttons
                self.load_btn = ui.Button("Load Scene", height=40, clicked_fn=self._on_load)
                self.pick_btn = ui.Button("Pick and Place", height=40, clicked_fn=self._on_pick, enabled=False)
                self.reset_btn = ui.Button("Reset", height=40, clicked_fn=self._on_reset, enabled=False)
                
                ui.Spacer(height=10)
                
                # Status
                self.status_label = ui.Label("Ready", alignment=ui.Alignment.CENTER)
    
    def _update_status(self, message):
        """Update status label"""
        if self.status_label:
            self.status_label.text = f"Status: {message}"
    
    def _on_load(self):
        """Load scene button callback"""
        self._update_status("Loading scene...")
        run_coroutine(self._load_scene())
    
    async def _load_scene(self):
        """Load the scene with Franka, 2 cubes, and container"""
        try:
            print("\n" + "="*60)
            print("LOADING SCENE - 2 CUBES")
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
            
            # Add 4 cubes IN FRONT of robot in a 2x2 grid
            print("=== Adding 4 Cubes ===")
            cube_size = 0.0515
            cube_spacing = 0.15  # 15cm spacing between cubes

            # Cube 1: Front-left (Blue)
            cube1_position = np.array([0.5, -0.1, cube_size/2.0 + 0.01])
            self.cube1 = self.world.scene.add(
                DynamicCuboid(
                    name=f"cube1_{int(time.time() * 1000)}",
                    position=cube1_position,
                    prim_path=f"/World/Cube1_{int(time.time() * 1000)}",
                    scale=np.array([cube_size, cube_size, cube_size]),
                    size=1.0,
                    color=np.array([0, 0, 1]),  # Blue
                )
            )
            print(f"Cube 1 (Blue) added at: {cube1_position}")

            # Cube 2: Front-right (Red)
            cube2_position = np.array([0.5, 0.1, cube_size/2.0 + 0.01])
            self.cube2 = self.world.scene.add(
                DynamicCuboid(
                    name=f"cube2_{int(time.time() * 1000)}",
                    position=cube2_position,
                    prim_path=f"/World/Cube2_{int(time.time() * 1000)}",
                    scale=np.array([cube_size, cube_size, cube_size]),
                    size=1.0,
                    color=np.array([1, 0, 0]),  # Red
                )
            )
            print(f"Cube 2 (Red) added at: {cube2_position}")

            # Cube 3: Back-left (Green)
            cube3_position = np.array([0.35, -0.1, cube_size/2.0 + 0.01])
            self.cube3 = self.world.scene.add(
                DynamicCuboid(
                    name=f"cube3_{int(time.time() * 1000)}",
                    position=cube3_position,
                    prim_path=f"/World/Cube3_{int(time.time() * 1000)}",
                    scale=np.array([cube_size, cube_size, cube_size]),
                    size=1.0,
                    color=np.array([0, 1, 0]),  # Green
                )
            )
            print(f"Cube 3 (Green) added at: {cube3_position}")

            # Cube 4: Back-right (Yellow)
            cube4_position = np.array([0.35, 0.1, cube_size/2.0 + 0.01])
            self.cube4 = self.world.scene.add(
                DynamicCuboid(
                    name=f"cube4_{int(time.time() * 1000)}",
                    position=cube4_position,
                    prim_path=f"/World/Cube4_{int(time.time() * 1000)}",
                    scale=np.array([cube_size, cube_size, cube_size]),
                    size=1.0,
                    color=np.array([1, 1, 0]),  # Yellow
                )
            )
            print(f"Cube 4 (Yellow) added at: {cube4_position}")
            
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

            # List of cubes to pick and place
            cubes = [
                (self.cube1, "Cube 1 (Blue)"),
                (self.cube2, "Cube 2 (Red)"),
                (self.cube3, "Cube 3 (Green)"),
                (self.cube4, "Cube 4 (Yellow)")
            ]

            # Pick and place each cube iteratively
            for i, (cube, cube_name) in enumerate(cubes, 1):
                print(f"\n>>> Picking {cube_name} ({i}/4)")
                success = await self._pick_and_place_cube(cube, cube_name.split()[1])  # Extract "1", "2", etc.
                if success:
                    self.placed_count += 1
                else:
                    self._update_status(f"Failed to pick {cube_name}")
                    self.is_picking = False
                    return

            print("\n=== ALL 4 CUBES PLACED! ===\n")
            self._update_status("All cubes placed successfully!")
            self.is_picking = False
            self.timeline.stop()

        except Exception as e:
            self._update_status(f"Error: {e}")
            print(f"Error in pick and place: {e}")
            import traceback
            traceback.print_exc()
            self.is_picking = False
            self.timeline.stop()

    async def _pick_and_place_cube(self, cube, cube_name):
        """Pick and place a single cube"""
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

            # Pick position: at cube center for better gripping
            # Cube is 5.15cm tall, center is at 2.575cm from bottom
            # The gripper fingers will wrap around the cube at its center
            pick_pos = np.array([cube_pos[0], cube_pos[1], cube_pos[2]])

            # Place approach: 15cm above container center
            # Container is at [0.55, 0.4, 0.0] with scale 0.3 (moved closer to robot)
            # Original container is 160x120x64cm, scaled to 48x36x19.2cm
            container_center = np.array([0.55, 0.4, 0.0])
            place_height = cube_half + 0.12 + (self.placed_count * cube_size)
            place_approach = container_center + np.array([0.0, 0.0, place_height + 0.15])

            # Place position: final position (centered in container)
            place_pos = container_center + np.array([0.0, 0.0, place_height])

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
            print("Step 2: Opening gripper")
            articulation_controller = self.franka.get_articulation_controller()
            # Apply open position directly to gripper joints (indices 7 and 8)
            open_action = ArticulationAction(
                joint_positions=self.gripper.joint_opened_positions,
                joint_indices=np.array([7, 8])
            )
            articulation_controller.apply_action(open_action)
            # Wait for gripper to open
            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

            # Step 3: Descend to pick position
            print(f"Step 3: Descending to pick: {pick_pos}")
            plan = self._plan_to_target(pick_pos, orientation)
            if plan is None:
                print("Failed to plan descent to pick")
                return False
            await self._execute_plan(plan)

            # Step 4: Close gripper (set to fully closed position)
            print("Step 4: Closing gripper")
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

            # Step 6: Move to place approach
            print(f"Step 6: Moving to place approach: {place_approach}")
            plan = self._plan_to_target(place_approach, orientation)
            if plan is None:
                print("Failed to plan to place approach")
                return False
            await self._execute_plan(plan)

            # Step 7: Descend to place
            print(f"Step 7: Descending to place: {place_pos}")
            plan = self._plan_to_target(place_pos, orientation)
            if plan is None:
                print("Failed to plan descent to place")
                return False
            await self._execute_plan(plan)

            # Step 8: Open gripper (release cube)
            print("Step 8: Opening gripper")
            articulation_controller = self.franka.get_articulation_controller()
            open_action = ArticulationAction(
                joint_positions=self.gripper.joint_opened_positions,
                joint_indices=np.array([7, 8])
            )
            articulation_controller.apply_action(open_action)
            # Wait for gripper to open
            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

            # Step 9: Ascend from place
            print(f"Step 9: Ascending from place: {place_approach}")
            plan = self._plan_to_target(place_approach, orientation)
            if plan is None:
                print("Failed to plan ascent from place")
                return False
            print(f"  Plan has {len(plan)} actions")
            await self._execute_plan(plan)
            # Wait for robot to settle
            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()

            # Step 10: Move to safe intermediate position (helps avoid invalid configurations)
            # Position between container and home, at safe height
            # Adjusted to match new container position [0.55, 0.4, 0.0]
            safe_intermediate = np.array([0.5, 0.2, 0.35])
            print(f"Step 10: Moving to safe intermediate position: {safe_intermediate}")
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

            # Step 11: Return to home position (safe configuration for next pick)
            print(f"Step 11: Returning to home position: {home_pos}")
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

            # Step 12: Close gripper to reset for next pick
            print("Step 12: Closing gripper (reset for next pick)")
            articulation_controller = self.franka.get_articulation_controller()
            close_action = ArticulationAction(
                joint_positions=self.gripper.joint_closed_positions,
                joint_indices=np.array([7, 8])
            )
            articulation_controller.apply_action(close_action)
            # Wait for gripper to close
            for _ in range(50):
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
print("Franka RRT - 4 Cubes Pick and Place")
print("="*60 + "\n")

app = FrankaRRT4Cubes()
print("UI loaded! Use the buttons to control the robot.")
print("="*60 + "\n")

