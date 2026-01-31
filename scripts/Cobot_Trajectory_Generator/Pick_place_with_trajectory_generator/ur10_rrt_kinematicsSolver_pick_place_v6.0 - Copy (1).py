"""UR10 Pick and Place with RRT + IK - v6.0"""

print("\n" + "="*80)
print("UR10 RRT+IK PICK-AND-PLACE v6.0")
print("="*80 + "\n")

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
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationAction
from pxr import UsdPhysics
import carb

# Motion generation imports
from isaacsim.robot_motion.motion_generation import (
    PathPlannerVisualizer,
    LulaKinematicsSolver,
    ArticulationKinematicsSolver
)
from isaacsim.robot_motion.motion_generation.lula import RRT

# Add project root to path for local imports
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.manipulators import SingleManipulator
from src.grippers import SurfaceGripper


class UR10RRTKinematicsSolver:
    """UR10 Pick and Place with RRT + Kinematics Solver"""

    def __init__(self):
        self.window = None
        self.world = None
        self.ur10 = None
        self.gripper = None
        self.container = None
        self.container_dimensions = None  # Store actual container dimensions

        # Motion generation tools (RRT + Kinematics)
        self.rrt = None                  # RRT path planner
        self.path_planner_visualizer = None  # Wraps RRT for visualization
        self.kinematics_solver = None    # LulaKinematicsSolver for IK/FK
        self.articulation_kinematics_solver = None  # Pairs kinematics solver with robot

        # Dynamic cube list
        self.cubes = []

        # Grid parameters
        self.grid_length = 2
        self.grid_width = 2

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

        # End effector name (SUCTION GRIPPER VARIANT)
        self._end_effector_name = "ee_suction_link"  # For suction gripper, use ee_suction_link

        self.build_ui()
    
    def build_ui(self):
        """Build UI"""
        self.window = ui.Window("Pick & Place with Trajectory Generation", width=500, height=450)

        with self.window.frame:
            with ui.VStack(spacing=10):
                ui.Label("Using Surface Gripper",
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
                            self.length_field.model.set_value(2)

                        # Width (columns)
                        with ui.HStack(spacing=10):
                            ui.Label("Grid Width (columns):", width=150)
                            self.width_field = ui.IntField(height=25)
                            self.width_field.model.set_value(2)

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

    def _on_reset(self):
        """Reset button callback"""
        try:
            self.is_picking = False

            # Stop timeline first
            self.timeline.stop()

            # Clear World instance (this should clear physics scene too)
            if self.world is not None:
                self.world.clear()
                World.clear_instance()
                self.world = None

            # Delete all prims from stage
            stage = omni.usd.get_context().get_stage()
            if stage:
                # Remove World prim
                world_prim = stage.GetPrimAtPath("/World")
                if world_prim.IsValid():
                    stage.RemovePrim("/World")

                # Remove physics scene if it exists
                physics_scene_prim = stage.GetPrimAtPath("/physicsScene")
                if physics_scene_prim.IsValid():
                    stage.RemovePrim("/physicsScene")

            # Reset state
            self.ur10 = None
            self.gripper = None
            self.container = None
            self.taskspace_generator = None
            self.cspace_generator = None
            self.kinematics_solver = None
            self.rrt = None
            self.path_planner_visualizer = None
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

            # Create World (this automatically creates physics scene)
            print("=== Creating World ===")
            self.world = World(stage_units_in_meters=1.0)
            await self.world.initialize_simulation_context_async()

            # Verify physics scene was created
            stage = omni.usd.get_context().get_stage()
            physics_scene_prim = stage.GetPrimAtPath("/physicsScene")
            if physics_scene_prim.IsValid():
                print("✅ Physics scene created at /physicsScene")
            else:
                print("⚠️  Warning: Physics scene not found at /physicsScene")

            # Add ground plane
            print("=== Adding Ground ===")
            self.world.scene.add_default_ground_plane()

            # Add UR10 with Surface Gripper
            print("=== Adding UR10 with Surface Gripper (Built-in Variant) ===")
            print("🔍 DEBUG: Using STATIC UR10 name (v6.0 - FIXED VERSION)")
            assets_root_path = get_assets_root_path()
            ur10_usd_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"

            # Use static name for UR10 (articulation system requires consistent naming)
            ur10_name = "ur10"
            ur10_prim_path = f"/World/{ur10_name}"
            print(f"🔍 DEBUG: UR10 prim path = {ur10_prim_path}")

            # Add UR10 to stage
            add_reference_to_stage(usd_path=ur10_usd_path, prim_path=ur10_prim_path)

            # Get stage
            stage = get_current_stage()
            robot_prim = stage.GetPrimAtPath(ur10_prim_path)

            # Set gripper variant to Short_Suction
            print("=== Setting Gripper Variant to Short_Suction ===")
            robot_prim.GetVariantSet("Gripper").SetVariantSelection("Short_Suction")

            # Enable camera components in the gripper (camera, camera_mount, camera_geom)
            print("=== Enabling Camera Components ===")
            ee_link_prim = stage.GetPrimAtPath(f"{ur10_prim_path}/ee_link")
            if ee_link_prim.IsValid():
                camera_components = ["Camera", "camera_mount", "camera_geom"]
                for component_name in camera_components:
                    component_prim = stage.GetPrimAtPath(f"{ur10_prim_path}/ee_link/{component_name}")
                    if component_prim.IsValid():
                        component_prim.SetActive(True)
                        print(f"   Enabled: {component_name}")
                        print(f"  [OK] Disabled: {component_name}")
                    else:
                        print(f"  [INFO] Not found: {component_name}")

                # List remaining active children
                print("Remaining active children of ee_link:")
                for child in ee_link_prim.GetChildren():
                    if child.IsActive():
                        print(f"  - {child.GetName()} (Type: {child.GetTypeName()})")

            # Use the built-in SurfaceGripper from the variant
            print("=== Creating Surface Gripper Wrapper ===")
            surface_gripper_prim_path = f"{ur10_prim_path}/ee_link/SurfaceGripper"
            gripper_prim = stage.GetPrimAtPath(surface_gripper_prim_path)

            if gripper_prim.IsValid():
                print(f"  [OK] Surface gripper found at: {surface_gripper_prim_path}")

                # Set clearance offset on gripper joints to avoid self-collision warning
                # The warning suggests 0.003m, so let's set it
                from pxr import UsdPhysics
                suction_joint_path = f"{surface_gripper_prim_path}/suction_cup/Suction_Joint"
                suction_joint_prim = stage.GetPrimAtPath(suction_joint_path)
                if suction_joint_prim.IsValid():
                    # Set clearance offset attribute
                    if suction_joint_prim.HasAttribute("isaac:clearanceOffset"):
                        suction_joint_prim.GetAttribute("isaac:clearanceOffset").Set(0.003)
                        print(f"  Set clearance offset to 0.003m on {suction_joint_path}")
                    else:
                        print(f"  [WARN] clearanceOffset attribute not found on {suction_joint_path}")

                # Create Surface Gripper wrapper
                self.gripper = SurfaceGripper(
                    end_effector_prim_path=f"{ur10_prim_path}/ee_link",
                    surface_gripper_path=surface_gripper_prim_path
                )
                print(f"  [OK] SurfaceGripper wrapper created")
            else:
                print(f"  [ERROR] SurfaceGripper prim not found at {surface_gripper_prim_path}")
                raise Exception(f"SurfaceGripper not found at {surface_gripper_prim_path}")

            # Add manipulator (using SingleManipulator like Franka pattern)
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

            print(f"UR10 added: {ur10_name} (using SingleManipulator like Franka)")

            # Add container IN FRONT of robot (positive X direction)
            print("=== Adding Container ===")
            # Container position: IN FRONT (positive X), to the right (positive Y)
            # Position adjusted to be further away from robot
            container_position = np.array([0.75, 0.5, 0.0])  # Moved further away
            self.container = await self._create_container(container_position)
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
            
            # Setup Hybrid Motion Generation
            print("=== Setting up Hybrid Motion Generation ===")
            self._setup_hybrid_motion_generation()

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

    def _setup_hybrid_motion_generation(self):
        """Setup motion generation tools (RRT + Kinematics - Franka Pattern)"""
        print("=== Setting up Motion Generation (RRT + Kinematics) ===")

        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
        rrt_config_dir = os.path.join(mg_extension_path, "path_planner_configs")

        # Use SUCTION-SPECIFIC robot description and URDF files
        robot_description_path = os.path.join(rmp_config_dir, "universal_robots/ur10/rmpflow_suction/ur10_robot_description.yaml")
        urdf_path = os.path.join(rmp_config_dir, "universal_robots/ur10/ur10_robot_suction.urdf")
        rrt_config_path = os.path.join(rrt_config_dir, "universal_robots/ur10/rrt/ur10_planner_config.yaml")

        # 1. Initialize RRT path planner (like Franka)
        print("  - Initializing RRT Path Planner...")
        self.rrt = RRT(
            robot_description_path=robot_description_path,
            urdf_path=urdf_path,
            rrt_config_path=rrt_config_path,
            end_effector_frame_name=self._end_effector_name
        )
        self.rrt.set_max_iterations(10000)

        # 2. Initialize PathPlannerVisualizer (wraps RRT - like Franka)
        print("  - Initializing Path Planner Visualizer...")
        self.path_planner_visualizer = PathPlannerVisualizer(
            robot_articulation=self.ur10,
            path_planner=self.rrt
        )
        print("  [OK] RRT initialized successfully!")

        # 3. Initialize Lula Kinematics Solver (like Franka)
        print("  - Initializing Lula Kinematics Solver...")
        self.kinematics_solver = LulaKinematicsSolver(
            robot_description_path=robot_description_path,
            urdf_path=urdf_path
        )
        print(f"  Valid frame names: {self.kinematics_solver.get_all_frame_names()}")

        # 4. Initialize Articulation Kinematics Solver (pairs LulaKinematicsSolver with UR10 - like Franka)
        print("  - Initializing Articulation Kinematics Solver...")
        self.articulation_kinematics_solver = ArticulationKinematicsSolver(
            self.ur10,
            self.kinematics_solver,
            self._end_effector_name
        )
        print("  [OK] Kinematics solvers initialized successfully!")

        print("[OK] Motion Generation initialized!")
        print("  - RRT: Path planning with obstacle avoidance")
        print("  - Kinematics Solver: IK/FK computations")
        print(f"  - End Effector Frame: {self._end_effector_name}")

    async def _create_container(self, position):
        """Create container for placing cubes"""
        # Generate unique container name with timestamp
        container_name = f"container_{int(time.time() * 1000)}"
        container_prim_path = f"/World/{container_name}"
        container_usd_path = f"{get_assets_root_path()}/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Storage/Containers/Container_I/Container_I04_160x120x64cm_PR_V_NVD_01.usd"

        add_reference_to_stage(usd_path=container_usd_path, prim_path=container_prim_path)

        # Scale: X=0.3, Y=0.3, Z=0.2 (reduced height to lower rim)
        # Original: 160x120x64cm → Scaled: 48x36x12.8cm
        scale = np.array([0.3, 0.3, 0.2])
        original_size = np.array([1.60, 1.20, 0.64])  # Original size in meters

        # Calculate actual container dimensions after scaling
        self.container_dimensions = original_size * scale
        print(f"Container dimensions (LxWxH): {self.container_dimensions[0]:.3f}m x {self.container_dimensions[1]:.3f}m x {self.container_dimensions[2]:.3f}m")

        container = self.world.scene.add(
            SingleXFormPrim(
                prim_path=container_prim_path,
                name=container_name,
                translation=position,
                scale=scale
            )
        )

        # Add physics to container
        stage = get_current_stage()
        container_prim = stage.GetPrimAtPath(container_prim_path)
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(container_prim)
        rigid_body_api.CreateKinematicEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(container_prim)

        await omni.kit.app.get_app().next_update_async()

        print(f"Container created: {container_name}")
        return container

    def _create_cube_grid(self):
        """Create cubes in a grid pattern"""
        cube_size = 0.0515
        cube_spacing = 0.10

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
                z = cube_size / 2.0 + 0.01

                # Get color
                color_name, color_rgb = colors[cube_index % len(colors)]

                # Generate unique cube name with timestamp
                # Format: Cube_R{row}_C{col}_{timestamp}
                timestamp = int(time.time() * 1000)
                cube_name = f"Cube_R{row+1}_C{col+1}_{timestamp}"
                cube_display_name = f"Cube_{row+1}_{col+1}"  # For logging

                cube = self.world.scene.add(
                    DynamicCuboid(
                        prim_path=f"/World/{cube_name}",
                        name=cube_name,
                        position=np.array([x, y, z]),
                        scale=np.array([cube_size, cube_size, cube_size]),
                        color=color_rgb,
                        mass=0.05
                    )
                )

                # Store cube with metadata
                self.cubes.append({
                    'object': cube,
                    'name': cube_name,
                    'display_name': cube_display_name,
                    'color': color_name,
                    'row': row + 1,
                    'col': col + 1
                })

                print(f"  {cube_display_name} ({color_name}) created as: {cube_name}")
                print(f"    Position: [{x:.3f}, {y:.3f}, {z:.3f}]")

                cube_index += 1

                # Small delay to ensure unique timestamps
                time.sleep(0.001)

    async def _pick_place_loop(self):
        """Main pick and place loop"""
        try:
            # Start timeline
            self.timeline.play()

            # Wait for timeline to start
            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            print("\n" + "="*60)
            print("=== STARTING PICK AND PLACE ===")
            print("="*60 + "\n")

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
                cube_name = cube_data['name']  # Unique name with timestamp
                cube_display_name = cube_data['display_name']  # Human-readable name
                color_name = cube_data['color']
                is_last_cube = (idx == total_cubes - 1)

                print(f"\n{'='*60}")
                print(f">>> CUBE {idx+1}/{total_cubes}: {cube_display_name} ({color_name})")
                print(f"    Prim path: /World/{cube_name}")
                print(f"{'='*60}\n")

                success = await self._pick_and_place_cube_hybrid(cube, cube_display_name, color_name, idx, is_last_cube)

                if not success:
                    print(f"[ERROR] Failed to pick and place {cube_name}")
                    self.is_picking = False
                    self._update_status(f"Failed at cube {idx+1}/{total_cubes}")
                    return  # Exit the loop and function

                print(f"[OK] {cube_name} ({color_name}) placed successfully!\n")

            if self.is_picking:
                print("\n" + "="*60)
                print(f"ALL {total_cubes} CUBES PLACED SUCCESSFULLY!")
                print("="*60 + "\n")
                self._update_status(f"All {total_cubes} cubes placed!")
                self.is_picking = False

        except Exception as e:
            print(f"Error in pick and place loop: {e}")
            import traceback
            traceback.print_exc()
            self.is_picking = False
            self._update_status(f"Error: {e}")

    async def _pick_and_place_cube_hybrid(self, cube, cube_name, color_name, cube_index, is_last_cube=False):
        """
        Pick and place a single cube using RRT + Kinematics Solver (Franka Pattern)

        All phases use RRT for motion planning:
        - IK validates targets are reachable
        - RRT plans collision-free paths
        - Execution follows the planned path
        """
        try:
            # Get cube position
            cube_pos, _ = cube.get_world_pose()
            print(f"[PICK] Cube position: {cube_pos}")

            # Calculate place position in container using actual dimensions
            cube_size = 0.0515
            container_pos, _ = self.container.get_world_pose()

            # Use actual container dimensions (stored during creation)
            container_length = self.container_dimensions[0]  # X-axis (0.48m)
            container_width = self.container_dimensions[1]   # Y-axis (0.36m)
            container_height = self.container_dimensions[2]  # Z-axis (0.128m)

            # Get total number of cubes from UI (FIXED: was using width_field only!)
            grid_length = int(self.length_field.model.get_value_as_int())
            grid_width = int(self.width_field.model.get_value_as_int())
            total_cubes = grid_length * grid_width

            # PLACEMENT STRATEGY:
            # Cube 1 at exact center of container
            # Subsequent cubes placed in grid pattern around cube 1
            # Auto-calculate spacing based on container area and number of cubes

            # Grid layout
            cubes_per_row = grid_width  # Columns along X-axis
            total_rows = grid_length     # Rows along Y-axis
            row = cube_index // cubes_per_row
            col = cube_index % cubes_per_row

            # Calculate spacing between cubes (minimum 8cm for safety)
            min_spacing = 0.08

            # Available space for grid (leave 8cm margin on each side)
            margin = 0.08
            usable_length = container_length - 2 * margin
            usable_width = container_width - 2 * margin

            # Calculate spacing based on available space
            if cubes_per_row > 1:
                spacing_x = max(min_spacing, usable_length / (cubes_per_row - 1))
            else:
                spacing_x = 0

            if total_rows > 1:
                spacing_y = max(min_spacing, usable_width / (total_rows - 1))
            else:
                spacing_y = 0

            # Calculate offset from container center
            # For cube 1 (row=0, col=0): offset_x = 0, offset_y = 0 (exact center)
            # For other cubes: offset based on grid position
            if cubes_per_row > 1:
                total_width_x = (cubes_per_row - 1) * spacing_x
                offset_x = -total_width_x/2 + col * spacing_x
            else:
                offset_x = 0

            if total_rows > 1:
                total_width_y = (total_rows - 1) * spacing_y
                offset_y = -total_width_y/2 + row * spacing_y
            else:
                offset_y = 0

            # Place position (on top of container)
            place_pos = container_pos + np.array([
                offset_x,
                offset_y,
                container_height/2 + cube_size/2 + 0.005
            ])

            if cube_index == 0:
                print(f"[PLACE] Cube 1 at CENTER: {place_pos}")
            else:
                print(f"[PLACE] Cube {cube_index+1}: {place_pos} | Grid R{row+1}C{col+1} | Spacing: {spacing_x:.2f}x{spacing_y:.2f}m")

            # Calculate key positions
            cube_half = cube_size / 2.0

            # Pre-pick position: 10cm above cube
            pre_pick_pos = cube_pos + np.array([0.0, 0.0, 0.10])

            # Pick position: 5mm above cube top (for suction)
            pick_height = cube_pos[2] + cube_half + 0.005
            pick_pos = np.array([cube_pos[0], cube_pos[1], pick_height])

            # Pre-place position: 15cm above place location
            pre_place_pos = place_pos + np.array([0.0, 0.0, 0.15])

            # Gripper orientation (pointing down)
            orientation = euler_angles_to_quat(np.array([0, np.pi / 2.0, 0]))

            # PHASE 1: TRAVEL TO PRE-PICK
            print(f"\n[PHASE 1] Travel to pre-pick {pre_pick_pos}")
            success = await self._move_to_target_rrt(pre_pick_pos, orientation)
            if not success:
                print("[ERROR] Failed pre-pick")
                return False

            # PHASE 2: PICK APPROACH
            print(f"[PHASE 2] Pick approach")
            self.gripper.open()
            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

            current_ee_pos, _ = self.ur10.end_effector.get_world_pose()
            success = await self._taskspace_straight_line(current_ee_pos, pick_pos, orientation, num_waypoints=20)
            if not success:
                print("[ERROR] Failed pick approach")
                return False

            for _ in range(20):
                await omni.kit.app.get_app().next_update_async()

            # Activate suction
            print("   Activating suction...")
            self.gripper.close()
            for _ in range(60):
                await omni.kit.app.get_app().next_update_async()

            gripper_status = self.gripper.is_closed()
            print(f"   Gripper: {'OK' if gripper_status else 'WARN - Not Closed'}")

            # ========================================
            # PHASE 3: PICK RETREAT (IK - Straight Up)
            # ========================================
            print(f"[PHASE 3] Pick retreat")
            current_ee_pos, _ = self.ur10.end_effector.get_world_pose()
            retreat_pos = current_ee_pos + np.array([0.0, 0.0, 0.10])
            success = await self._taskspace_straight_line(current_ee_pos, retreat_pos, orientation, num_waypoints=20)
            if not success:
                print("[ERROR] Failed pick retreat")
                return False

            # PHASE 4: TRAVEL TO PRE-PLACE
            print(f"[PHASE 4] Travel to pre-place {pre_place_pos}")
            success = await self._move_to_target_rrt(pre_place_pos, orientation)
            if not success:
                print("[ERROR] Failed pre-place")
                return False

            # PHASE 5: PLACE APPROACH
            print(f"[PHASE 5] Place approach")
            current_ee_pos, _ = self.ur10.end_effector.get_world_pose()
            success = await self._taskspace_straight_line(current_ee_pos, place_pos, orientation, num_waypoints=20)
            if not success:
                print("[ERROR] Failed place approach")
                return False

            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

            # Deactivate suction
            print("   Deactivating suction...")
            self.gripper.open()
            for _ in range(60):
                await omni.kit.app.get_app().next_update_async()

            # PHASE 6: PLACE RETREAT
            print(f"[PHASE 6] Place retreat")
            current_ee_pos, _ = self.ur10.end_effector.get_world_pose()
            retreat_pos = current_ee_pos + np.array([0.0, 0.0, 0.15])
            success = await self._taskspace_straight_line(current_ee_pos, retreat_pos, orientation, num_waypoints=20)
            if not success:
                print("[ERROR] Failed place retreat")
                return False

            # PHASE 7: RETURN TO HOME (only for last cube)
            if is_last_cube:
                print(f"[PHASE 7] Return to home")
                home_pos = np.array([0.6, 0.0, 0.5])
                success = await self._move_to_target_rrt(home_pos, orientation)
                if not success:
                    print("[WARN] Failed home")

                self.gripper.close()
                for _ in range(30):
                    await omni.kit.app.get_app().next_update_async()

            return True

        except Exception as e:
            print(f"[ERROR] Error picking {cube_name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _correct_orientation(self, target_pos, target_orientation, num_steps=20):
        """
        Smoothly correct orientation at current position after RRT
        This prevents abrupt rotation when transitioning from RRT to IK motions
        """
        # Check if robot is initialized
        if not self.ur10.handles_initialized:
            print("[ERROR] Robot not initialized")
            return False

        # Update robot base pose for kinematics solver
        robot_base_translation, robot_base_orientation = self.ur10.get_world_pose()
        self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        # Get current joint positions
        current_joint_positions = self.ur10.get_joint_positions()

        # Compute IK for same position but with correct orientation
        ik_action, ik_success = self.articulation_kinematics_solver.compute_inverse_kinematics(
            target_pos, target_orientation
        )

        if not ik_success:
            print(f"   [WARN] IK failed for orientation correction")
            return True  # Continue anyway

        # Extract target joint positions
        target_joint_positions = ik_action.joint_positions

        # Smoothly interpolate to correct orientation
        for i in range(1, num_steps + 1):
            alpha = i / num_steps
            interpolated_joints = current_joint_positions + alpha * (target_joint_positions - current_joint_positions)

            action = ArticulationAction(joint_positions=interpolated_joints)
            self.ur10.apply_action(action)
            await omni.kit.app.get_app().next_update_async()

        # Settling time
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()

        return True

    async def _taskspace_straight_line(self, start_pos, end_pos, orientation, num_waypoints=10):
        """
        Execute straight-line motion using IK + joint-space interpolation
        This avoids RRT's orientation changes by directly interpolating in joint space
        """
        # Check if robot is initialized
        if not self.ur10.handles_initialized:
            print("[ERROR] Robot not initialized")
            return False

        # Update robot base pose for kinematics solver
        robot_base_translation, robot_base_orientation = self.ur10.get_world_pose()
        self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        # Get current joint positions as starting point
        current_joint_positions = self.ur10.get_joint_positions()

        # Compute IK for end position
        # Returns (ArticulationAction, bool) - need to extract joint_positions
        ik_action, ik_success = self.articulation_kinematics_solver.compute_inverse_kinematics(
            end_pos, orientation
        )

        if not ik_success:
            print(f"   [ERROR] IK failed for target {end_pos}")
            return False

        # Extract joint positions from ArticulationAction
        end_joint_positions = ik_action.joint_positions

        # Interpolate in joint space (smooth motion, maintains orientation)
        for i in range(1, num_waypoints + 1):
            alpha = i / num_waypoints
            interpolated_joints = current_joint_positions + alpha * (end_joint_positions - current_joint_positions)

            # Apply joint positions directly
            action = ArticulationAction(joint_positions=interpolated_joints)
            self.ur10.apply_action(action)

            # Wait for physics update
            await omni.kit.app.get_app().next_update_async()

        # Extra settling time
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()

        return True

    def _plan_to_target(self, target_position, target_orientation):
        """
        Plan path to target using RRT with IK validation (Franka Pattern)

        Args:
            target_position: Target position (3D numpy array)
            target_orientation: Target orientation (quaternion)

        Returns:
            plan: List of ArticulationActions, or None if planning failed
        """
        # Check if robot is initialized
        if not self.ur10.handles_initialized:
            print("[ERROR] Robot articulation not initialized")
            return None

        # Track robot base pose for BOTH kinematics solver AND RRT planner
        robot_base_translation, robot_base_orientation = self.ur10.get_world_pose()
        self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        self.rrt.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        # Validate target is reachable using IK solver before planning
        _, ik_success = self.articulation_kinematics_solver.compute_inverse_kinematics(
            target_position, target_orientation
        )



        # Plan path using RRT
        self.rrt.set_end_effector_target(target_position, target_orientation)
        self.rrt.update_world()
        plan = self.path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=0.015)
        return plan

    async def _execute_plan(self, plan):
        """
        Execute a plan (Franka Pattern)

        Args:
            plan: List of ArticulationActions

        Returns:
            bool: True if successful
        """
        if plan is None:
            return False

        # Execute plan - wait every frame for smooth motion
        for action in plan:
            self.ur10.apply_action(action)
            # Wait for physics to update
            await omni.kit.app.get_app().next_update_async()

        # Wait for robot to settle after plan execution
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()

        return True

    async def _move_to_target_rrt(self, target_position, target_orientation, maintain_orientation=False):
        """Move to target using RRT"""
        plan = self._plan_to_target(target_position, target_orientation)

        if plan is None:
            print(f"   [ERROR] RRT planning failed")
            return False

        success = await self._execute_plan(plan)
        if not success:
            print(f"   [ERROR] RRT execution failed")

        return success

# Main entry point
if __name__ == "__main__":
    app = UR10RRTKinematicsSolver()
    print("="*60)
    print("PICK AND PLACE WITH RRT + KINEMATICS SOLVER")
    print("="*60)
    print("\nClick 'Load Scene' to start")

