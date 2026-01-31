"""
Franka RRT Pick and Place - Dynamic Grid with Kinematics Solver
Supports variable number of cubes in a grid pattern
Container placed in front for easy reach

Features:
- RRT path planning for collision-free motion with obstacle avoidance
- LulaKinematicsSolver paired with ArticulationKinematicsSolver for IK/FK
- IK validation before RRT planning to check target reachability
- Dynamic grid configuration via UI
- Proper grid placement with edge margins from container walls
- Add/Remove obstacles dynamically during operation

Version 1.2 Updates:
- Pick grid shifted to Y=-0.15 (towards -Y) to separate pick and place areas
- Container at original position [0.5, 0.4, 0.0] for comfortable reach
- Edge margin increased to 9cm (provides 6.4cm clearance from cube edge to container wall)
- Immediate ascent after gripper closes (15cm) to avoid hitting nearby cubes
- Fixed IK failure for far cubes: validates reachability and uses adjusted intermediate approach
- No RRT fallback for place operations - consistent IK approach for all cubes
- Release gripper at 8cm height to prevent fingers from hitting container walls
- Reduced wait times for better FPS (20-30 FPS expected)
- Optimized IK interpolation steps for faster motion
- Use central safe position for transit (handles all cube positions)
- Improved error handling with specific failure messages
- Final message shows actual placed count vs total count
- Added detailed debug output for placement positions and clearances
- OBSTACLE AVOIDANCE: Add/Remove obstacles with UI buttons, RRT automatically avoids them
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
from pxr import UsdPhysics
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
import carb

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
            
            print(f"Franka added: {franka_name}")
            await omni.kit.app.get_app().next_update_async()
            
            # Add container IN FRONT of robot (positive X direction)
            print("=== Adding Container ===")
            container_prim_path = "/World/Container"
            container_usd_path = f"{get_assets_root_path()}/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Storage/Containers/Container_I/Container_I04_160x120x64cm_PR_V_NVD_01.usd"
            
            add_reference_to_stage(usd_path=container_usd_path, prim_path=container_prim_path)
            
            # Container position: IN FRONT (positive X), to the right (positive Y)
            # Positioned for comfortable reach: not too close, not too far
            # X=0.40 for better reachability (closer to robot)
            # Y=0.45 provides good space between column 3 cubes and container
            container_position = np.array([0.40, 0.45, 0.0])

            # Scale: X=0.3, Y=0.3, Z=0.2 (reduced height to lower rim)
            # Original: 160x120x64cm -> Scaled: 48x36x12.8cm
            scale = np.array([0.3, 0.3, 0.2])
            original_size = np.array([1.60, 1.20, 0.64])  # Original size in meters

            # Calculate actual container dimensions after scaling
            self.container_dimensions = original_size * scale
            print(f"Container dimensions (LxWxH): {self.container_dimensions[0]:.3f}m x {self.container_dimensions[1]:.3f}m x {self.container_dimensions[2]:.3f}m")

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
            
            print(f"Container added at: {container_position}")
            await omni.kit.app.get_app().next_update_async()
            
            # Add cubes dynamically based on grid parameters
            print(f"=== Adding {total_cubes} Cubes ({self.grid_length}x{self.grid_width} Grid) ===")
            cube_size = 0.0515
            cube_spacing = 0.15  # 15cm spacing between cubes for better picking reliability and obstacle placement

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
            # Grid centered at X=0.5, Y=0.0 (original position)
            grid_center_x = 0.5
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
                    cube_number = cube_index + 1  # 1-based numbering
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

                    # Set display name with cube number and timestamp
                    stage = omni.usd.get_context().get_stage()
                    prim = stage.GetPrimAtPath(prim_path)
                    if prim:
                        display_name = f"Cube {cube_number} (t{timestamp})"
                        omni.usd.editor.set_display_name(prim, display_name)

                    # Store cube with its name
                    self.cubes.append((cube, f"{cube_name} ({color_name})"))
                    print(f"  {cube_name} ({color_name}) added at: [{cube_x:.3f}, {cube_y:.3f}, {cube_z:.3f}] - Display: {display_name}")

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
            self.add_obstacle_btn.enabled = True
            self.remove_obstacle_btn.enabled = True
            self._update_status("Scene loaded! Ready to pick and place")
            
        except Exception as e:
            self._update_status(f"Error: {e}")
            print(f"Error loading scene: {e}")
            import traceback
            traceback.print_exc()

    def _setup_rrt(self):
        """Setup RRT motion planner and kinematics solvers"""
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")

        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs", "franka", "rmpflow")
        robot_description_file = os.path.join(rmp_config_dir, "robot_descriptor.yaml")
        urdf_path = os.path.join(mg_extension_path, "motion_policy_configs", "franka", "lula_franka_gen.urdf")
        rrt_config_file = os.path.join(mg_extension_path, "path_planner_configs", "franka", "rrt", "franka_planner_config.yaml")

        # Initialize RRT path planner
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

        # Initialize Lula Kinematics Solver
        print("=== Setting up Kinematics Solvers ===")
        self.kinematics_solver = LulaKinematicsSolver(
            robot_description_path=robot_description_file,
            urdf_path=urdf_path
        )

        print("Valid frame names at which to compute kinematics:", self.kinematics_solver.get_all_frame_names())

        # Initialize Articulation Kinematics Solver (pairs LulaKinematicsSolver with Articulation)
        end_effector_name = "right_gripper"
        self.articulation_kinematics_solver = ArticulationKinematicsSolver(
            self.franka,
            self.kinematics_solver,
            end_effector_name
        )

        print("Kinematics solvers initialized successfully!")

        # Initialize CSpace Trajectory Generator for smooth motion
        print("=== Setting up Trajectory Generator ===")
        self.cspace_trajectory_generator = LulaCSpaceTrajectoryGenerator(
            robot_description_file,
            urdf_path
        )

        # Verify that the robot description includes acceleration and jerk limits
        # These are important for smooth trajectory generation
        for i in range(len(self.rrt.get_active_joints())):
            if not self.cspace_trajectory_generator._lula_kinematics.has_c_space_acceleration_limit(i):
                carb.log_warn(f"Joint {i} missing acceleration limit in robot description")
            if not self.cspace_trajectory_generator._lula_kinematics.has_c_space_jerk_limit(i):
                carb.log_warn(f"Joint {i} missing jerk limit in robot description")

        print("Trajectory generator initialized successfully!")

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
                self._update_status("Starting pick and place...")
            else:
                self._update_status(f"Resuming from cube {self.current_cube_index + 1}...")

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

            # Pick and place each cube iteratively, starting from current_cube_index
            for i in range(self.current_cube_index, total_cubes):
                cube, cube_name = cubes[i]
                cube_number = i + 1  # 1-based for display

                print(f"\n{'='*60}")
                print(f"Starting {cube_name} ({cube_number}/{total_cubes})")
                print(f"{'='*60}")
                is_last = (cube_number == total_cubes)  # Check if this is the last cube

                # Update current cube index
                self.current_cube_index = i

                # Call pick and place (retry logic is now INSIDE the function)
                success, error_msg = await self._pick_and_place_cube(cube, cube_name.split()[1], is_last)  # Extract "1", "2", etc.

                if success:
                    self.placed_count += 1
                    print(f"\n=== {self.placed_count} of {total_cubes} CUBES PLACED! ===")
                    print(f"[SUCCESS] {cube_name} placed successfully!\n")
                    self._update_status(f"{self.placed_count} of {total_cubes} cubes placed successfully!")
                else:
                    print(f"\n[SKIPPING] {cube_name}: {error_msg}")
                    print(f"   Current progress: {self.placed_count}/{total_cubes} cubes placed")
                    self._update_status(f"{self.placed_count} of {total_cubes} cubes placed ({cube_name} skipped)")

                # Move to next cube
                self.current_cube_index += 1

            print(f"\n=== COMPLETE: {self.placed_count} of {total_cubes} CUBES PLACED! ===\n")
            self._update_status(f"Complete: {self.placed_count} of {total_cubes} cubes placed successfully!")
            self.is_picking = False
            self.current_cube_index = 0  # Reset for next run
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
        Pick and place a single cube using state machine with FULL OBSTACLE AVOIDANCE

        NEW STATE MACHINE (4 RRT phases for full obstacle avoidance):

        PICK PHASE (with retry):
        - PHASE 1: HOME → PRE-PICK (RRT with obstacle avoidance)
        - PHASE 2: PRE-PICK → PICK (IK straight down, no obstacles)
        - PHASE 3: PICK → POST-PICK (IK straight up, no obstacles)
        - PHASE 4: POST-PICK → HOME (RRT with obstacle avoidance) - NEW!
        - Verify pick success, retry up to 3 times if failed

        PLACE PHASE (no retry):
        - PHASE 5: HOME → PRE-PLACE (RRT with obstacle avoidance)
        - PHASE 6: PRE-PLACE → PLACE (IK straight down, no obstacles)
        - PHASE 7: PLACE → POST-PLACE (IK straight up, no obstacles)
        - PHASE 8: POST-PLACE → HOME (RRT with obstacle avoidance) - ALWAYS!

        Key Improvement: Robot always returns to safe HOME position between pick and place,
        ensuring obstacle avoidance for ALL horizontal travel while keeping fast IK for vertical motions.

        Args:
            cube: The cube object to pick and place
            cube_name: Name of the cube for logging
            is_last_cube: Not used anymore (always return to home)

        Returns:
            tuple: (success: bool, error_message: str)
                - success: True if both pick and place succeeded
                - error_message: Description of failure
        """
        try:
            # Calculate key positions
            cube_size = 0.0515
            cube_half = cube_size / 2.0

            # Gripper orientation (pointing down)
            orientation = euler_angles_to_quats(np.array([np.pi, 0, 0]))

            # HOME position (safe position for obstacle avoidance)
            home_pos = np.array([0.35, 0.0, 0.5])

            # ========================================
            # PICK PHASE WITH RETRY LOOP
            # ========================================
            max_pick_attempts = 3
            pick_success = False

            for pick_attempt in range(1, max_pick_attempts + 1):
                if pick_attempt > 1:
                    print(f"\n{'*'*60}")
                    print(f"PICK RETRY {pick_attempt}/{max_pick_attempts} for Cube {cube_name}")
                    print(f"{'*'*60}")

                # Get CURRENT cube position for this pick attempt
                cube_pos_current, _ = cube.get_world_pose()
                print(f"\n[PICK ATTEMPT {pick_attempt}] Cube {cube_name} position: [{cube_pos_current[0]:.4f}, {cube_pos_current[1]:.4f}, {cube_pos_current[2]:.4f}]")

                # Pre-pick position: 12cm above cube's current position
                pre_pick_pos = cube_pos_current + np.array([0.0, 0.0, 0.12])

                # ========================================
                # PHASE 1: TRAVEL TO PRE-PICK (RRT)
                # ========================================
                print(f"[PHASE 1] Travel to pre-pick [{pre_pick_pos[0]:.4f}, {pre_pick_pos[1]:.4f}, {pre_pick_pos[2]:.4f}]")
                success = await self._move_to_target_rrt(pre_pick_pos, orientation)
                if not success:
                    print(f"[ERROR] Failed to reach pre-pick position")
                    if pick_attempt < max_pick_attempts:
                        print(f"   Retrying pick... ({max_pick_attempts - pick_attempt} attempts remaining)")
                        continue
                    else:
                        print(f"[FAILED] All {max_pick_attempts} pick attempts failed")
                        return False, f"Failed to reach pre-pick position for Cube {cube_name}"

                # ========================================
                # PHASE 2: PICK APPROACH
                # ========================================
                print(f"[PHASE 2] Pick approach")

                # READ REAL-TIME CUBE POSITION NOW (just before pick)
                cube_pos_realtime, _ = cube.get_world_pose()
                print(f"   Real-time cube position: [{cube_pos_realtime[0]:.4f}, {cube_pos_realtime[1]:.4f}, {cube_pos_realtime[2]:.4f}]")

                # Calculate pick position based on REAL-TIME position
                pick_pos = np.array([cube_pos_realtime[0], cube_pos_realtime[1], 0.02])
                print(f"   Target pick position: [{pick_pos[0]:.4f}, {pick_pos[1]:.4f}, {pick_pos[2]:.4f}]")

                # Open gripper
                articulation_controller = self.franka.get_articulation_controller()
                open_action = ArticulationAction(
                    joint_positions=self.gripper.joint_opened_positions,
                    joint_indices=np.array([7, 8])
                )
                articulation_controller.apply_action(open_action)
                for _ in range(30):
                    await omni.kit.app.get_app().next_update_async()

                # Descend to pick position (IK - short vertical motion with more steps for accuracy)
                current_ee_pos, _ = self.franka.end_effector.get_world_pose()
                success = await self._move_to_target_ik(pick_pos, orientation, num_steps=30)
                if not success:
                    print(f"[ERROR] Failed pick approach (IK descent)")
                    if pick_attempt < max_pick_attempts:
                        print(f"   Retrying pick... ({max_pick_attempts - pick_attempt} attempts remaining)")
                        continue
                    else:
                        print(f"[FAILED] All {max_pick_attempts} pick attempts failed")
                        return False, f"Failed to descend to pick position for Cube {cube_name}"

                # Wait longer for robot to stabilize at pick position
                for _ in range(30):
                    await omni.kit.app.get_app().next_update_async()

                # Close gripper
                print("   Closing gripper...")
                close_action = ArticulationAction(
                    joint_positions=self.gripper.joint_closed_positions,
                    joint_indices=np.array([7, 8])
                )
                articulation_controller.apply_action(close_action)
                # Wait longer for gripper to fully close and grasp cube
                for _ in range(60):
                    await omni.kit.app.get_app().next_update_async()

                # ========================================
                # PHASE 3: PICK RETREAT (IK - short vertical motion)
                # ========================================
                print(f"[PHASE 3] Pick retreat")
                current_ee_pos, _ = self.franka.end_effector.get_world_pose()
                retreat_pos = current_ee_pos + np.array([0.0, 0.0, 0.18])
                success = await self._move_to_target_ik(retreat_pos, orientation, num_steps=20)
                if not success:
                    print(f"[ERROR] Failed pick retreat")
                    if pick_attempt < max_pick_attempts:
                        print(f"   Retrying pick... ({max_pick_attempts - pick_attempt} attempts remaining)")
                        continue
                    else:
                        print(f"[FAILED] All {max_pick_attempts} pick attempts failed")
                        return False, f"Failed pick retreat for Cube {cube_name}"

                # ========================================
                # VERIFY PICK SUCCESS
                # ========================================
                cube_pos_after_pick, _ = cube.get_world_pose()
                height_lifted = cube_pos_after_pick[2] - cube_pos_realtime[2]

                if height_lifted > 0.05:  # Cube lifted at least 5cm
                    print(f"   [PICK SUCCESS] Cube lifted {height_lifted*100:.1f}cm")
                    pick_success = True
                    break  # Exit pick retry loop
                else:
                    print(f"   [PICK FAILED] Cube only lifted {height_lifted*100:.1f}cm (need >5cm)")
                    if pick_attempt < max_pick_attempts:
                        print(f"   Retrying pick... ({max_pick_attempts - pick_attempt} attempts remaining)")
                        # Open gripper before retry
                        open_action = ArticulationAction(
                            joint_positions=self.gripper.joint_opened_positions,
                            joint_indices=np.array([7, 8])
                        )
                        articulation_controller.apply_action(open_action)
                        for _ in range(10):
                            await omni.kit.app.get_app().next_update_async()
                        continue
                    else:
                        print(f"[FAILED] All {max_pick_attempts} pick attempts failed")
                        return False, f"Failed to pick Cube {cube_name} after {max_pick_attempts} attempts"

            # If we get here and pick didn't succeed, return failure
            if not pick_success:
                return False, f"Failed to pick Cube {cube_name}"

            # ========================================
            # PHASE 4: RETURN TO HOME AFTER PICK (RRT - NEW!)
            # ========================================
            print(f"\n[PHASE 4] Return to HOME after pick (with cube)")
            print(f"   Target: [{home_pos[0]:.4f}, {home_pos[1]:.4f}, {home_pos[2]:.4f}]")
            success = await self._move_to_target_rrt(home_pos, orientation)
            if not success:
                print("[ERROR] Failed to return to HOME after pick")
                return False, f"Failed to return to HOME after picking Cube {cube_name}"

            print(f"   [OK] Reached HOME safely with cube")

            # ========================================
            # PLACE PHASE (NO RETRY) - Starting from HOME
            # ========================================
            print(f"\n[PLACE] Starting place phase for Cube {cube_name} (from HOME)")

            # Calculate placement position inside container with proper grid spacing
            # Container dimensions: self.container_dimensions = [length, width, height]
            # Container center: [0.45, 0.40, 0.0] (moved closer to robot for better reach)
            container_center = np.array([0.45, 0.40, 0.0])

            # Get container dimensions
            container_length = self.container_dimensions[0]  # X dimension (48cm = 0.48m)
            container_width = self.container_dimensions[1]   # Y dimension (36cm = 0.36m)

            # Edge margin: Distance from container edge to cube CENTER
            # Cube size is 5.15cm, so cube extends 2.575cm from center
            # Need at least 2.575cm + safety margin (8.5cm) = 11cm from edge to cube center
            # This provides ~8.4cm clearance from cube edge to container wall
            # Extra margin accounts for cube bounce/shift when dropped from 8cm height
            edge_margin = 0.11  # 11cm in meters (ensures cube edges stay 8.4cm away from container walls)

            # Calculate usable space inside container (excluding margins)
            usable_length = container_length - (2 * edge_margin)  # 0.48 - 0.22 = 0.26m
            usable_width = container_width - (2 * edge_margin)    # 0.36 - 0.22 = 0.14m

            # Calculate which row and column this cube should be placed in
            # Use same grid dimensions as pick grid
            place_row = self.placed_count // self.grid_width
            place_col = self.placed_count % self.grid_width

            # Calculate spacing between cubes (center to center)
            # For N cubes, we need (N-1) gaps. If N=1, spacing doesn't matter (single cube centered)
            if self.grid_length > 1:
                spacing_length = usable_length / (self.grid_length - 1)
            else:
                spacing_length = 0.0  # Single row, no spacing needed

            if self.grid_width > 1:
                spacing_width = usable_width / (self.grid_width - 1)
            else:
                spacing_width = 0.0  # Single column, no spacing needed

            # Calculate starting position (accounting for edge margin)
            # Start from the edge of usable space (container_center - half_dimension + margin)
            start_x = container_center[0] - (container_length / 2.0) + edge_margin
            start_y = container_center[1] - (container_width / 2.0) + edge_margin

            # Calculate cube position in container
            cube_x = start_x + (place_row * spacing_length)
            cube_y = start_y + (place_col * spacing_width)

            # Place position: at bottom of container
            # Container bottom is at Z=0, place cube so its bottom sits on container floor
            # Add small clearance (0.5cm) to avoid collision with container floor
            container_floor_z = 0.0
            place_height = container_floor_z + cube_half + 0.005  # Cube bottom at 0.5cm above floor
            place_pos = np.array([cube_x, cube_y, place_height])

            # Safe transit position: Use a central, reachable position instead of directly above pick
            # For far cubes (row 2 in 3x3 grid), position directly above pick may be out of reach
            # Use a central position that's always reachable: slightly in front of robot, centered, high
            safe_transit_pos = np.array([0.35, 0.0, 0.35])  # Central, high, always reachable

            # Place approach: 15cm above the place position (but still need safe transit first)
            place_approach = place_pos + np.array([0.0, 0.0, 0.15])

            # Calculate clearances from container walls
            container_left_edge = container_center[0] - container_length/2
            container_right_edge = container_center[0] + container_length/2
            container_front_edge = container_center[1] - container_width/2
            container_back_edge = container_center[1] + container_width/2

            clearance_left = (cube_x - cube_half) - container_left_edge
            clearance_right = container_right_edge - (cube_x + cube_half)
            clearance_front = (cube_y - cube_half) - container_front_edge
            clearance_back = container_back_edge - (cube_y + cube_half)

            # Debug output
            print(f"  - Placing at grid position: Row {place_row + 1}, Column {place_col + 1}")
            print(f"  - Place position: [{place_pos[0]:.4f}, {place_pos[1]:.4f}, {place_pos[2]:.4f}]")
            print(f"  - Spacing: Length={spacing_length:.4f}m, Width={spacing_width:.4f}m")
            print(f"  - Edge margin: {edge_margin:.4f}m, Start: X={start_x:.4f}m, Y={start_y:.4f}m")
            print(f"  - Container edges: X=[{container_left_edge:.4f}, {container_right_edge:.4f}], Y=[{container_front_edge:.4f}, {container_back_edge:.4f}]")
            print(f"  - Cube will extend to: X=[{cube_x - cube_half:.4f}, {cube_x + cube_half:.4f}], Y=[{cube_y - cube_half:.4f}, {cube_y + cube_half:.4f}]")
            print(f"  - Clearance from walls: Left={clearance_left*100:.2f}cm, Right={clearance_right*100:.2f}cm, Front={clearance_front*100:.2f}cm, Back={clearance_back*100:.2f}cm")
            print(f"  - Cube bottom will be at Z={place_height - cube_half:.4f}m (container floor at Z=0.0m)")

            # Pre-place position: 30cm above place location (increased to avoid container rim)
            pre_place_pos = place_pos + np.array([0.0, 0.0, 0.30])

            # ========================================
            # PHASE 5: HOME → PRE-PLACE (RRT with obstacle avoidance)
            # ========================================
            print(f"\n[PHASE 5] Travel HOME → PRE-PLACE (with cube)")
            print(f"   From: [{home_pos[0]:.4f}, {home_pos[1]:.4f}, {home_pos[2]:.4f}]")
            print(f"   To:   [{pre_place_pos[0]:.4f}, {pre_place_pos[1]:.4f}, {pre_place_pos[2]:.4f}]")
            success = await self._move_to_target_rrt(pre_place_pos, orientation)
            if not success:
                print("[ERROR] Failed to reach pre-place position")
                return False, f"Failed to reach pre-place for Cube {cube_name}"

            # ========================================
            # PHASE 6: PLACE APPROACH (IK - short vertical motion)
            # ========================================
            print(f"[PHASE 6] Place approach")

            # Special handling for first cube to avoid rim bounce
            # First cube often hits container rim and bounces out
            # Solution: Place first cube more towards CENTER of container (away from rim)
            if self.placed_count == 0:
                print("   [FIRST CUBE] Adjusting position towards container center to avoid rim")
                # Move first cube 3cm towards container center (X direction)
                adjusted_place_pos = place_pos + np.array([0.03, 0.0, 0.0])
                release_height = adjusted_place_pos + np.array([0.0, 0.0, 0.10])
                print(f"   Original: [{place_pos[0]:.4f}, {place_pos[1]:.4f}, {place_pos[2]:.4f}]")
                print(f"   Adjusted: [{adjusted_place_pos[0]:.4f}, {adjusted_place_pos[1]:.4f}, {adjusted_place_pos[2]:.4f}]")
            else:
                # Normal placement for subsequent cubes
                release_height = place_pos + np.array([0.0, 0.0, 0.10])

            # Descend to 10cm above final position (safe clearance, avoids gripper hitting rim)
            success = await self._move_to_target_ik(release_height, orientation, num_steps=40)
            if not success:
                print("[WARN] Failed place approach - continuing anyway")

            # Wait for robot to stabilize at release position
            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

            # Open gripper to release cube
            print("   Opening gripper...")
            articulation_controller = self.franka.get_articulation_controller()
            open_action = ArticulationAction(
                joint_positions=self.gripper.joint_opened_positions,
                joint_indices=np.array([7, 8])
            )
            articulation_controller.apply_action(open_action)

            # Wait for gripper to fully open
            for _ in range(50):
                await omni.kit.app.get_app().next_update_async()

            # Wait for cube to settle after release
            for _ in range(80):
                await omni.kit.app.get_app().next_update_async()

            # ========================================
            # FINAL VERIFICATION: Check if cube is in container
            # ========================================
            cube_pos_final, _ = cube.get_world_pose()

            # Check if cube is in the container area (within 15cm of target XY)
            xy_distance = np.linalg.norm(cube_pos_final[:2] - place_pos[:2])
            cube_in_container = xy_distance < 0.15  # Within 15cm of target

            # Check if cube is at reasonable height (not still in gripper)
            cube_at_place_height = cube_pos_final[2] < 0.15  # Below 15cm height

            if cube_in_container and cube_at_place_height:
                print(f"   [PLACE SUCCESS] Cube in container! Position: [{cube_pos_final[0]:.3f}, {cube_pos_final[1]:.3f}, {cube_pos_final[2]:.3f}], distance: {xy_distance*100:.1f}cm")
                placement_successful = True
            else:
                print(f"   [PLACE FAILED] Cube NOT in container!")
                print(f"   Final position: [{cube_pos_final[0]:.3f}, {cube_pos_final[1]:.3f}, {cube_pos_final[2]:.3f}]")
                print(f"   Target position: [{place_pos[0]:.3f}, {place_pos[1]:.3f}, {place_pos[2]:.3f}]")
                print(f"   Distance from target: {xy_distance*100:.1f}cm")
                placement_successful = False

            # ========================================
            # PHASE 7: PLACE RETREAT (IK - short vertical motion)
            # ========================================
            print(f"[PHASE 7] Place retreat")
            current_ee_pos, _ = self.franka.end_effector.get_world_pose()
            retreat_pos = current_ee_pos + np.array([0.0, 0.0, 0.20])
            success = await self._move_to_target_ik(retreat_pos, orientation, num_steps=20)
            if not success:
                print("[WARN] Failed place retreat - continuing anyway")

            # ========================================
            # PHASE 8: POST-PLACE → HOME (RRT - ALWAYS!)
            # ========================================
            print(f"\n[PHASE 8] Return POST-PLACE → HOME (obstacle avoidance)")
            print(f"   Target: [{home_pos[0]:.4f}, {home_pos[1]:.4f}, {home_pos[2]:.4f}]")
            success = await self._move_to_target_rrt(home_pos, orientation)
            if not success:
                print("[ERROR] Failed to return to HOME after place")
                return False, f"Failed to return to HOME after placing Cube {cube_name}"

            # Close gripper at home
            close_action = ArticulationAction(
                joint_positions=self.gripper.joint_closed_positions,
                joint_indices=np.array([7, 8])
            )
            articulation_controller.apply_action(close_action)
            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

            print(f"   [OK] Reached HOME safely")

            # ========================================
            # RETURN RESULT BASED ON FINAL VERIFICATION
            # ========================================
            if placement_successful:
                print(f"Cube {cube_name} placed successfully!\n")
                return True, ""  # Success
            else:
                print(f"Cube {cube_name} placement FAILED\n")
                return False, f"Cube {cube_name} not in container"  # Place failed (pick succeeded)

        except Exception as e:
            error_msg = f"Error picking/placing Cube {cube_name}: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return False, error_msg  # Failed

    async def _move_to_target_ik(self, target_position, target_orientation, num_steps=20):
        """
        Move to target using IK directly (for simple, straight-line movements)
        This is faster and more predictable than RRT for vertical descents

        Args:
            target_position: Target end effector position
            target_orientation: Target end effector orientation
            num_steps: Number of interpolation steps (default: 20, reduced for better FPS)

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

    async def _move_to_target_rrt(self, target_position, target_orientation):
        """
        Move to target using RRT (for long-distance collision-free motion)

        Args:
            target_position: Target end effector position
            target_orientation: Target end effector orientation

        Returns:
            bool: True if successful, False if planning/execution failed
        """
        plan = self._plan_to_target(target_position, target_orientation)
        if plan is None:
            return False
        await self._execute_plan(plan)
        return True

    async def _taskspace_straight_line(self, start_pos, end_pos, orientation, num_waypoints=20):
        """
        Execute straight-line motion using IK + joint-space interpolation
        This avoids RRT's orientation changes by directly interpolating in joint space

        Args:
            start_pos: Starting position (not used, kept for compatibility)
            end_pos: Target end effector position
            orientation: Target end effector orientation
            num_waypoints: Number of interpolation steps

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
            print(f"   [ERROR] IK failed for target {end_pos}")
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

        # Minimal settling time
        for _ in range(2):
            await omni.kit.app.get_app().next_update_async()

        return True

    def _plan_to_target(self, target_position, target_orientation):
        """
        Plan path to target using RRT with smooth trajectory generation

        This method follows the example pattern:
        1. Validate target with IK
        2. Compute RRT path (waypoints)
        3. Interpolate waypoints densely
        4. Generate smooth trajectory with LulaCSpaceTrajectoryGenerator
        5. Convert to ArticulationAction sequence
        """
        # Track robot base pose for BOTH kinematics solver AND RRT planner
        robot_base_translation, robot_base_orientation = self.franka.get_world_pose()
        self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        self.rrt.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        # Validate target is reachable using IK solver before planning
        _, ik_success = self.articulation_kinematics_solver.compute_inverse_kinematics(
            target_position, target_orientation
        )

        if not ik_success:
            carb.log_warn(f"IK did not converge for target position {target_position}. RRT may fail.")
            print(f"  WARNING: IK validation failed for target {target_position}")
        else:
            print(f"  [OK] IK validation successful for target {target_position}")

        # Set target for RRT planner
        self.rrt.set_end_effector_target(target_position, target_orientation)
        self.rrt.update_world()

        # Get current joint positions as start state
        active_joints = self.path_planner_visualizer.get_active_joints_subset()
        start_pos = active_joints.get_joint_positions()

        # Validate current robot configuration before planning
        # Check for NaN or extreme joint values that indicate invalid state
        if np.any(np.isnan(start_pos)) or np.any(np.abs(start_pos) > 10.0):
            carb.log_error(f"Invalid robot joint configuration detected: {start_pos}")
            print(f"  [ERROR] Robot in invalid configuration - cannot plan")
            return None

        # Compute RRT path (returns waypoints in C-space)
        # Significantly increase iterations for complex obstacle avoidance scenarios
        # Default config has 50,000 - we use that for better success rate
        self.rrt.set_max_iterations(50000)

        print(f"  [RRT] Planning with {50000} max iterations...")
        print(f"  [RRT] Start config: {start_pos}")
        print(f"  [RRT] Target position: {target_position}")

        rrt_plan = self.rrt.compute_path(start_pos, np.array([]))

        if rrt_plan is None or len(rrt_plan) <= 1:
            carb.log_warn(f"No plan could be generated to target pose: {target_position}")
            print(f"  [ERROR] RRT planning failed after 50000 iterations")
            print(f"  [ERROR] This usually means obstacle is blocking all paths")
            print(f"  [ERROR] Consider repositioning obstacle or increasing workspace")
            return None

        print(f"  [OK] RRT found path with {len(rrt_plan)} waypoints")

        # Convert RRT waypoints to smooth trajectory using LulaCSpaceTrajectoryGenerator
        # This provides better motion quality with proper acceleration/jerk limits
        action_sequence = self._convert_rrt_plan_to_trajectory(rrt_plan)

        return action_sequence

    def _convert_rrt_plan_to_trajectory(self, rrt_plan):
        """
        Convert RRT waypoints to smooth trajectory using LulaCSpaceTrajectoryGenerator

        This follows the example pattern from path_planning_controller.py:
        - Interpolate RRT waypoints densely (max distance 0.01)
        - Use trajectory generator to create smooth spline-based trajectory
        - Convert to ArticulationAction sequence with proper timing

        Args:
            rrt_plan: List of waypoints from RRT planner

        Returns:
            List of ArticulationAction for execution
        """
        # Interpolate path for smooth trajectory generation
        # Increased from 0.01 to 0.1 to significantly reduce computation and prevent freezing
        # Larger value = fewer waypoints = faster execution
        rrt_interpolation_max_dist = 0.1
        interpolated_path = self.path_planner_visualizer.interpolate_path(rrt_plan, rrt_interpolation_max_dist)

        print(f"  [TRAJECTORY] Interpolated to {len(interpolated_path)} waypoints")

        # Generate smooth C-space trajectory with acceleration/jerk limits
        trajectory = self.cspace_trajectory_generator.compute_c_space_trajectory(interpolated_path)

        # Convert to ArticulationTrajectory with physics timestep
        physics_dt = 1.0 / 60.0  # Match world physics timestep
        art_trajectory = ArticulationTrajectory(self.franka, trajectory, physics_dt)

        # Get action sequence for execution
        action_sequence = art_trajectory.get_action_sequence()

        print(f"  [TRAJECTORY] Generated {len(action_sequence)} actions for smooth execution")

        return action_sequence

    async def _execute_plan(self, action_sequence):
        """
        Execute a smooth trajectory action sequence

        Args:
            action_sequence: List of ArticulationActions from trajectory generator

        Returns:
            bool: True if successful, False if plan is None
        """
        if action_sequence is None or len(action_sequence) == 0:
            return False

        # Execute action sequence frame by frame
        # The trajectory generator has already computed proper timing
        for action in action_sequence:
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

            self._update_status("Reset complete - Stage cleared")
            print("\n=== RESET COMPLETE - STAGE CLEARED ===\n")

        except Exception as e:
            self._update_status(f"Error resetting: {e}")
            print(f"Error: {e}")
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

            # Obstacle position: Strategic placement for RRT horizontal travel avoidance
            # Full obstacle avoidance: RRT for all horizontal travel (4 phases)
            # Position obstacle to challenge RRT paths and demonstrate avoidance
            # X=0.27: Between robot base and cubes (forces RRT to plan around it)
            # Y=0.17: Between cube grid and container (challenges paths)
            # Z=0.19: Mid-height to demonstrate avoidance
            obstacle_position = np.array([0.27, 0.17, 0.19])

            # Create obstacle as a FixedCuboid (has collision geometry for RRT)
            # Use FixedCuboid instead of VisualCuboid to ensure proper collision detection
            # Parallel to X-axis: rotate 90 degrees around Z-axis
            # Updated scale as requested by user
            obstacle = self.world.scene.add(
                FixedCuboid(
                    name=obstacle_name,
                    prim_path=obstacle_prim_path,
                    position=obstacle_position,
                    orientation=euler_angles_to_quats(np.array([0, 0, np.pi / 2])),  # Parallel to X-axis
                    size=1.0,
                    scale=np.array([0.05, 0.25, 0.25]),  # User-specified scale
                    color=np.array([0.0, 0.0, 1.0])  # Blue color
                )
            )

            # Add obstacle to RRT planner for collision avoidance
            self.rrt.add_obstacle(obstacle)

            # Store obstacle reference
            self.obstacles[obstacle_name] = obstacle

            print(f"\n=== OBSTACLE ADDED ===")
            print(f"Name: {obstacle_name}")
            print(f"Type: FixedCuboid (with collision geometry)")
            print(f"Position: {obstacle_position} (X, Y, Z)")
            print(f"  - X=0.27: Between robot base and cubes")
            print(f"  - Y=0.17: Between cube grid and container")
            print(f"  - Z=0.19: Mid-height")
            print(f"Scale: [0.05, 0.25, 0.25] (thickness, width, height)")
            print(f"Orientation: Parallel to X-axis")
            print(f"Total obstacles: {len(self.obstacles)}")
            print(f"RRT max iterations: 50000 (increased for complex obstacle avoidance)")
            print(f"RRT interpolation: 0.1 (optimized for performance)")
            print("\nNEW STATE MACHINE - FULL OBSTACLE AVOIDANCE:")
            print("  PICK PHASE:")
            print("    - Phase 1: HOME → PRE-PICK (RRT - obstacle avoidance)")
            print("    - Phase 2: PRE-PICK → PICK (IK - straight down)")
            print("    - Phase 3: PICK → POST-PICK (IK - straight up)")
            print("    - Phase 4: POST-PICK → HOME (RRT - obstacle avoidance) [NEW!]")
            print("  PLACE PHASE:")
            print("    - Phase 5: HOME → PRE-PLACE (RRT - obstacle avoidance)")
            print("    - Phase 6: PRE-PLACE → PLACE (IK - straight down)")
            print("    - Phase 7: PLACE → POST-PLACE (IK - straight up)")
            print("    - Phase 8: POST-PLACE → HOME (RRT - obstacle avoidance) [ALWAYS!]")
            print("\nRobot ALWAYS returns to safe HOME between operations!")
            print("4 RRT phases ensure obstacle avoidance for ALL horizontal travel!\n")

            self._update_status(f"Obstacle added! Total: {len(self.obstacles)}")

        except Exception as e:
            self._update_status(f"Error adding obstacle: {e}")
            print(f"Error adding obstacle: {e}")
            import traceback
            traceback.print_exc()

    def _on_remove_obstacle(self):
        """Remove obstacle button callback"""
        if not self.world or not self.rrt:
            self._update_status("Load scene first!")
            return

        if len(self.obstacles) == 0:
            self._update_status("No obstacles to remove!")
            print("No obstacles to remove")
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

            print(f"\n=== OBSTACLE REMOVED ===")
            print(f"Name: {obstacle_name}")
            print(f"Remaining obstacles: {len(self.obstacles)}\n")

            self._update_status(f"Obstacle removed! Remaining: {len(self.obstacles)}")

        except Exception as e:
            self._update_status(f"Error removing obstacle: {e}")
            print(f"Error removing obstacle: {e}")
            import traceback
            traceback.print_exc()


# Create and show UI
print("\n" + "="*60)
print("Cobot - Grasping")
print("="*60 + "\n")

app = FrankaRRTDynamicGrid()
print("UI loaded! Configure grid and use the buttons to control the robot.")
print("="*60 + "\n")

