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

from isaacsim.robot_motion.motion_generation import (
    LulaTaskSpaceTrajectoryGenerator,
    LulaKinematicsSolver,
    ArticulationTrajectory
)

import usd.schema.isaac.robot_schema as robot_schema
from isaacsim.core.utils.stage import get_current_stage

current_file = Path(__file__).resolve()
project_root = None
for parent in current_file.parents:
    if parent.name == "multiagent":
        project_root = parent
        break

if project_root is None:
    raise RuntimeError("Could not find 'multiagent' folder in parent directories")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.manipulators import SingleManipulator
from src.grippers import SurfaceGripper


class PickPlaceTaskSpace:
    """Pick and Place with Surface Gripper using Task-Space Trajectory"""

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

                with ui.CollapsableFrame("Grid Configuration", height=0):
                    with ui.VStack(spacing=5):
                        with ui.HStack(spacing=10):
                            ui.Label("Grid Length (rows):", width=150)
                            self.length_field = ui.IntField(height=25)
                            self.length_field.model.set_value(2)

                        with ui.HStack(spacing=10):
                            ui.Label("Grid Width (columns):", width=150)
                            self.width_field = ui.IntField(height=25)
                            self.width_field.model.set_value(2)

                        ui.Label("Total cubes will be: Length × Width",
                                alignment=ui.Alignment.CENTER,
                                style={"color": 0xFF888888, "font_size": 12})

                ui.Spacer(height=10)

                self.load_btn = ui.Button("Load Scene", height=40, clicked_fn=self._on_load)
                self.pick_btn = ui.Button("Start Pick and Place", height=40, clicked_fn=self._on_pick, enabled=False)
                self.reset_btn = ui.Button("Reset Scene", height=40, clicked_fn=self._on_reset, enabled=False)

                ui.Spacer(height=10)

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
            self.grid_length = int(self.length_field.model.get_value_as_int())
            self.grid_width = int(self.width_field.model.get_value_as_int())

            if self.grid_length < 1 or self.grid_width < 1:
                self._update_status("Error: Grid dimensions must be at least 1x1")
                return
            if self.grid_length > 10 or self.grid_width > 10:
                self._update_status("Error: Grid dimensions too large (max 10x10)")
                return

            total_cubes = self.grid_length * self.grid_width

            self.timeline.stop()
            await omni.kit.app.get_app().next_update_async()

            World.clear_instance()
            await omni.kit.app.get_app().next_update_async()

            self.world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
            await omni.kit.app.get_app().next_update_async()

            self.world.scene.add_default_ground_plane()
            await omni.kit.app.get_app().next_update_async()

            ur10_name = f"ur10_{int(time.time() * 1000)}"
            ur10_prim_path = f"/World/UR10_{int(time.time() * 1000)}"

            ur10_usd_path = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
            robot_prim = add_reference_to_stage(usd_path=ur10_usd_path, prim_path=ur10_prim_path)

            robot_prim.GetVariantSet("Gripper").SetVariantSelection("Short_Suction")

            await omni.kit.app.get_app().next_update_async()

            stage = get_current_stage()
            gripper_prim_path = f"{ur10_prim_path}/ee_link/SurfaceGripper"
            gripper_prim = stage.GetPrimAtPath(gripper_prim_path)

            if gripper_prim.IsValid():
                gripper_prim.GetAttribute(robot_schema.Attributes.MAX_GRIP_DISTANCE.name).Set(0.05)
                gripper_prim.GetAttribute(robot_schema.Attributes.COAXIAL_FORCE_LIMIT.name).Set(100.0)
                gripper_prim.GetAttribute(robot_schema.Attributes.SHEAR_FORCE_LIMIT.name).Set(100.0)
                gripper_prim.GetAttribute(robot_schema.Attributes.RETRY_INTERVAL.name).Set(2.0)
            else:
                carb.log_warn(f"Surface gripper prim not found at {gripper_prim_path}")

            self.gripper = SurfaceGripper(
                end_effector_prim_path=f"{ur10_prim_path}/ee_link",
                surface_gripper_path=gripper_prim_path
            )

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

            await omni.kit.app.get_app().next_update_async()

            container_prim_path = "/World/Container"
            container_usd_path = f"{get_assets_root_path()}/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Storage/Containers/Container_I/Container_I04_160x120x64cm_PR_V_NVD_01.usd"

            add_reference_to_stage(usd_path=container_usd_path, prim_path=container_prim_path)

            container_position = np.array([0.7, 0.3, 0.0])

            self.container = self.world.scene.add(
                SingleXFormPrim(
                    prim_path=container_prim_path,
                    name="container",
                    translation=container_position,
                    scale=np.array([0.3, 0.3, 0.2])
                )
            )

            stage = get_current_stage()
            container_prim = stage.GetPrimAtPath(container_prim_path)
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(container_prim)
            rigid_body_api.CreateKinematicEnabledAttr(True)
            UsdPhysics.CollisionAPI.Apply(container_prim)

            await omni.kit.app.get_app().next_update_async()

            cube_size = 0.0515
            cube_spacing = 0.10

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

            grid_center_x = 0.6
            grid_center_y = -0.3

            start_x = grid_center_x - ((self.grid_length - 1) * cube_spacing) / 2.0
            start_y = grid_center_y - ((self.grid_width - 1) * cube_spacing) / 2.0

            cube_index = 0
            for row in range(self.grid_length):
                for col in range(self.grid_width):
                    cube_x = start_x + (row * cube_spacing)
                    cube_y = start_y + (col * cube_spacing)
                    cube_z = cube_size/2.0 + 0.01
                    cube_position = np.array([cube_x, cube_y, cube_z])

                    color, color_name = colors[cube_index % len(colors)]

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

                    self.cubes.append((cube, f"{cube_name} ({color_name})"))

                    cube_index += 1

            await omni.kit.app.get_app().next_update_async()

            self.world.initialize_physics()

            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            default_joint_positions = np.array([0.0, -2.0, 1.5, -1.07, -1.57, 0.0])
            self.ur10.set_joints_default_state(positions=default_joint_positions)

            self.gripper.set_default_state(opened=True)

            self.world.reset()

            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()

            current_joints = self.ur10.get_joint_positions()

            if not np.allclose(current_joints, default_joint_positions, atol=0.1):
                self.ur10.set_joint_positions(default_joint_positions)

                for _ in range(20):
                    await omni.kit.app.get_app().next_update_async()

                current_joints = self.ur10.get_joint_positions()

            self.gripper.open()

            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

            self._setup_trajectory_generator()

            self.pick_btn.enabled = True
            self.reset_btn.enabled = True
            self._update_status("Scene loaded! Ready to pick and place")

        except Exception as e:
            self._update_status(f"Error: {e}")
            carb.log_error(f"Error loading scene: {e}")
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

    def _on_pick(self):
        """Pick and place button callback"""
        if not self.world or not self.taskspace_trajectory_generator:
            self._update_status("Load scene first!")
            return

        if self.is_picking:
            self.is_picking = False
            self._update_status("Paused")
            self.timeline.pause()
        else:
            self.is_picking = True
            self._update_status("Starting pick and place...")
            run_coroutine(self._pick_place_loop())

    async def _pick_place_loop(self):
        """Main pick and place loop"""
        try:
            self.timeline.play()

            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()

            cubes = self.cubes

            total_cubes = len(cubes)

            for i, (cube, cube_name) in enumerate(cubes, 1):
                is_last = (i == total_cubes)
                success = await self._pick_and_place_cube(cube, cube_name, is_last)
                if success:
                    self.placed_count += 1
                else:
                    self._update_status(f"Failed to pick {cube_name}")
                    self.is_picking = False
                    return

            self._update_status(f"All {total_cubes} cubes placed successfully!")
            self.is_picking = False
            self.timeline.stop()

        except Exception as e:
            self._update_status(f"Error: {e}")
            carb.log_error(f"Error in pick and place: {e}")
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
            cube_pos, _ = cube.get_world_pose()

            cube_size = 0.0515
            cube_half = cube_size / 2.0

            pick_approach = cube_pos + np.array([0.0, 0.0, 0.10])

            pick_height = cube_pos[2] + cube_half + 0.005
            pick_pos = np.array([cube_pos[0], cube_pos[1], pick_height])

            container_center = np.array([0.7, 0.3, 0.0])

            place_spacing = 0.06

            place_row = self.placed_count // self.grid_width
            place_col = self.placed_count % self.grid_width

            offset_x = (place_row - (self.grid_length - 1) / 2.0) * place_spacing
            offset_y = (place_col - (self.grid_width - 1) / 2.0) * place_spacing
            offset = np.array([offset_x, offset_y, 0.0])

            container_floor_z = 0.0
            place_height = container_floor_z + cube_half + 0.005
            place_pos = container_center + offset + np.array([0.0, 0.0, place_height])

            safe_transit_height = 0.30
            safe_transit_pos = np.array([cube_pos[0], cube_pos[1], safe_transit_height])

            place_approach = place_pos + np.array([0.0, 0.0, 0.15])

            orientation = euler_angles_to_quats(np.array([np.pi, 0, 0]))

            home_pos = np.array([0.5, 0.0, 0.4])

            success = await self._move_to_target(pick_approach, orientation)
            if not success:
                return False

            self.gripper.open()
            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

            success = await self._move_to_target(pick_pos, orientation)
            if not success:
                return False

            for _ in range(20):
                await omni.kit.app.get_app().next_update_async()

            current_joints = self.ur10.get_joint_positions()
            ee_pos, _ = self.kinematics_solver.compute_forward_kinematics(
                self._end_effector_name, current_joints
            )
            distance_to_cube = np.linalg.norm(ee_pos - pick_pos)

            if distance_to_cube > 0.05:
                carb.log_warn(f"End effector is {distance_to_cube:.4f}m from target (should be < 0.05m)")

            self.gripper.close()

            for _ in range(150):
                await omni.kit.app.get_app().next_update_async()

            cube_pos_after_pick, _ = cube.get_world_pose()
            z_change = abs(cube_pos_after_pick[2] - cube_pos[2])
            if z_change < 0.001:
                carb.log_warn(f"Cube may not have been picked! Position unchanged")

            success = await self._move_to_target(pick_approach, orientation)
            if not success:
                return False

            success = await self._move_to_target(safe_transit_pos, orientation)
            if not success:
                return False

            success = await self._move_to_target(place_approach, orientation)
            if not success:
                return False

            success = await self._move_to_target(place_pos, orientation)
            if not success:
                return False

            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

            self.gripper.open()
            for _ in range(40):
                await omni.kit.app.get_app().next_update_async()

            success = await self._move_to_target(place_approach, orientation)
            if not success:
                return False

            if is_last_cube:
                safe_intermediate = np.array([0.5, 0.2, 0.35])
                try:
                    success = await self._move_to_target(safe_intermediate, orientation)
                    if not success:
                        carb.log_warn("Could not move to intermediate position, trying home directly")
                except Exception as e:
                    carb.log_warn(f"Error moving to intermediate position: {e}")

                try:
                    success = await self._move_to_target(home_pos, orientation)
                    if not success:
                        carb.log_warn("Failed to move to home position, continuing anyway")
                except Exception as e:
                    carb.log_warn(f"Could not return to home position: {e}")

                self.gripper.close()
                for _ in range(50):
                    await omni.kit.app.get_app().next_update_async()
            else:
                self.gripper.close()
                for _ in range(30):
                    await omni.kit.app.get_app().next_update_async()

            return True

        except Exception as e:
            carb.log_error(f"Error picking {cube_name}: {e}")
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
            current_joint_positions = self.ur10.get_joint_positions()
            current_ee_position, current_ee_rotation = self.kinematics_solver.compute_forward_kinematics(
                self._end_effector_name, current_joint_positions
            )

            from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats
            current_ee_orientation = rot_matrices_to_quats(current_ee_rotation)

            distance = np.linalg.norm(target_position - current_ee_position)

            if use_intermediate and distance > 0.2:
                mid_position = (current_ee_position + target_position) / 2.0
                mid_position[2] = max(current_ee_position[2], target_position[2]) + 0.1

                positions = np.array([current_ee_position, mid_position, target_position])
                orientations = np.array([current_ee_orientation, target_orientation, target_orientation])
            else:
                positions = np.array([current_ee_position, target_position])
                orientations = np.array([current_ee_orientation, target_orientation])

            trajectory = self.taskspace_trajectory_generator.compute_task_space_trajectory_from_points(
                positions, orientations, self._end_effector_name
            )

            if trajectory is None:
                carb.log_error(f"Failed to compute trajectory to {target_position}")
                return False

            physics_dt = 1 / 60
            articulation_trajectory = ArticulationTrajectory(self.ur10, trajectory, physics_dt)
            action_sequence = articulation_trajectory.get_action_sequence()

            for action in action_sequence:
                self.ur10.apply_action(action)
                await omni.kit.app.get_app().next_update_async()

            return True

        except Exception as e:
            carb.log_error(f"Error moving to target {target_position}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _on_reset(self):
        """Reset button callback"""
        try:
            self.is_picking = False

            self.timeline.stop()

            if self.world is not None:
                World.clear_instance()

            from isaacsim.core.utils.stage import clear_stage
            clear_stage()

            self.world = None
            self.ur10 = None
            self.gripper = None
            self.container = None
            self.taskspace_trajectory_generator = None
            self.kinematics_solver = None
            self.cubes = []
            self.placed_count = 0

            self.load_btn.enabled = True
            self.pick_btn.enabled = False
            self.reset_btn.enabled = False

            self._update_status("Reset complete - Stage cleared")

        except Exception as e:
            self._update_status(f"Error resetting: {e}")
            carb.log_error(f"Error: {e}")
            import traceback
            traceback.print_exc()


app = PickPlaceTaskSpace()

