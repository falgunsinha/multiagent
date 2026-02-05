
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

from isaacsim.robot_motion.motion_generation import (
    LulaCSpaceTrajectoryGenerator,
    LulaKinematicsSolver,
    ArticulationTrajectory
)

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


class CSpaceIK:
    """Pick and Place using C-Space Trajectory + IK"""

    def __init__(self):
        self.window = None
        self.world = None
        self.ur10 = None
        self.gripper = None
        self.container = None
        self.cspace_generator = None
        self.kinematics_solver = None
        self.cubes = []
        self.grid_length = 2
        self.grid_width = 2
        self.timeline = omni.timeline.get_timeline_interface()
        self.is_picking = False
        self.placed_count = 0
        self.load_btn = None
        self.pick_btn = None
        self.reset_btn = None
        self.status_label = None
        self.length_field = None
        self.width_field = None
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

                ui.Spacer(height=10)

                with ui.VStack(spacing=5):
                    self.load_btn = ui.Button("Load Scene", height=40, clicked_fn=self._on_load)
                    self.pick_btn = ui.Button("Start Pick and Place", height=40, clicked_fn=self._on_pick, enabled=False)
                    self.reset_btn = ui.Button("Reset Scene", height=40, clicked_fn=self._on_reset)

                ui.Spacer(height=10)

                with ui.CollapsableFrame("Status", collapsed=False):
                    with ui.VStack(spacing=5):
                        self.status_label = ui.Label("Ready", word_wrap=True)

    def _update_status(self, message):
        """Update status label"""
        if self.status_label:
            self.status_label.text = message

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
            self.is_picking = False
            self._update_status("Paused")
            self.timeline.pause()
        else:
            self.is_picking = True
            self._update_status("Starting pick and place...")
            run_coroutine(self._pick_place_loop())

    def _on_reset(self):
        """Reset button callback"""
        try:
            self.is_picking = False
            self.timeline.stop()

            if self.world is not None:
                World.clear_instance()
                self.world = None

            stage = omni.usd.get_context().get_stage()
            if stage:
                world_prim = stage.GetPrimAtPath("/World")
                if world_prim.IsValid():
                    stage.RemovePrim("/World")

            self.ur10 = None
            self.gripper = None
            self.container = None
            self.cspace_generator = None
            self.kinematics_solver = None
            self.cubes = []
            self.placed_count = 0

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
            self.grid_length = int(self.length_field.model.get_value_as_int())
            self.grid_width = int(self.width_field.model.get_value_as_int())

            if self.grid_length < 1 or self.grid_width < 1:
                self._update_status("Grid dimensions must be at least 1x1")
                return
            if self.grid_length > 5 or self.grid_width > 5:
                self._update_status("Grid dimensions cannot exceed 5x5")
                return

            self.world = World()
            await self.world.initialize_simulation_context_async()

            self.world.scene.add_default_ground_plane()

            assets_root_path = get_assets_root_path()
            ur10_usd_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"

            ur10_name = f"ur10_{int(time.time() * 1000)}"
            ur10_prim_path = f"/World/{ur10_name}"

            add_reference_to_stage(usd_path=ur10_usd_path, prim_path=ur10_prim_path)

            stage = get_current_stage()
            robot_prim = stage.GetPrimAtPath(ur10_prim_path)
            robot_prim.GetVariantSet("Gripper").SetVariantSelection("Short_Suction")

            gripper_prim_path = f"{ur10_prim_path}/ee_link/SurfaceGripper"
            gripper_prim = stage.GetPrimAtPath(gripper_prim_path)

            if gripper_prim.IsValid():
                gripper_prim.GetAttribute(robot_schema.Attributes.MAX_GRIP_DISTANCE.name).Set(0.05)
                gripper_prim.GetAttribute(robot_schema.Attributes.COAXIAL_FORCE_LIMIT.name).Set(100.0)
                gripper_prim.GetAttribute(robot_schema.Attributes.SHEAR_FORCE_LIMIT.name).Set(100.0)
                gripper_prim.GetAttribute(robot_schema.Attributes.RETRY_INTERVAL.name).Set(2.0)

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

            container_position = np.array([0.9, 0.4, 0.0])
            self.container = self._create_container(container_position)

            self._create_cube_grid()

            await self.world.reset_async()

            default_joint_positions = np.array([0.0, -2.0, 1.5, -1.07, -1.57, 0.0])
            self.ur10.set_joints_default_state(positions=default_joint_positions)
            self.gripper.set_default_state(opened=True)

            await self.world.reset_async()

            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()

            self.gripper.open()

            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

            self._setup_motion_generation()

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

        self.cspace_generator = LulaCSpaceTrajectoryGenerator(
            robot_description_path=robot_description_path,
            urdf_path=urdf_path
        )

        self.kinematics_solver = LulaKinematicsSolver(
            robot_description_path=robot_description_path,
            urdf_path=urdf_path
        )

    def _create_container(self, position):
        """Create container for placing cubes"""
        container_size = np.array([0.15, 0.15, 0.05])
        container = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Container",
                name="Container",
                position=position,
                scale=container_size,
                color=np.array([0.5, 0.5, 0.5]),
                mass=100.0
            )
        )
        return container

    def _create_cube_grid(self):
        """Create cubes in a grid pattern"""
        cube_size = 0.0515
        cube_spacing = 0.10

        grid_center_x = 0.7
        grid_center_y = -0.4

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

        grid_start_x = grid_center_x - ((self.grid_length - 1) * cube_spacing) / 2.0
        grid_start_y = grid_center_y - ((self.grid_width - 1) * cube_spacing) / 2.0

        cube_index = 0
        for row in range(self.grid_length):
            for col in range(self.grid_width):
                x = grid_start_x + (row * cube_spacing)
                y = grid_start_y + (col * cube_spacing)
                z = cube_size / 2.0 + 0.01

                color_name, color_rgb = colors[cube_index % len(colors)]

                cube_name = f"Cube_{row+1}_{col+1}"
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

                self.cubes.append({
                    'object': cube,
                    'name': cube_name,
                    'color': color_name,
                    'row': row + 1,
                    'col': col + 1
                })

                cube_index += 1

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

            for idx, cube_data in enumerate(cubes):
                if not self.is_picking:
                    break

                cube = cube_data['object']
                cube_name = cube_data['name']
                color_name = cube_data['color']
                is_last_cube = (idx == total_cubes - 1)

                success = await self._pick_and_place_cube(cube, cube_name, color_name, idx, is_last_cube)

                if not success:
                    break

            if self.is_picking:
                self._update_status(f"All {total_cubes} cubes placed!")
                self.is_picking = False

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.is_picking = False
            self._update_status(f"Error: {e}")

    async def _pick_and_place_cube(self, cube, cube_name, color_name, cube_index, is_last_cube=False):
        """Pick and place a single cube using C-Space Trajectory + IK"""
        try:
            cube_pos, _ = cube.get_world_pose()

            cube_size = 0.0515
            container_pos, _ = self.container.get_world_pose()
            container_size = np.array([0.15, 0.15, 0.05])

            cubes_per_row = 2
            row = cube_index // cubes_per_row
            col = cube_index % cubes_per_row

            place_spacing = 0.06

            offset_x = (col - 0.5) * place_spacing
            offset_y = (row - 0.5) * place_spacing

            place_pos = container_pos + np.array([offset_x, offset_y, container_size[2]/2 + cube_size/2 + 0.005])

            cube_half = cube_size / 2.0

            pick_approach = cube_pos + np.array([0.0, 0.0, 0.10])

            pick_height = cube_pos[2] + cube_half + 0.005
            pick_pos = np.array([cube_pos[0], cube_pos[1], pick_height])

            place_approach = place_pos + np.array([0.0, 0.0, 0.15])

            orientation = euler_angles_to_quats(np.array([np.pi, 0, 0]))

            success = await self._move_to_target_cspace(pick_approach, orientation)
            if not success:
                return False

            self.gripper.open()
            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

            success = await self._move_to_target_cspace(pick_pos, orientation)
            if not success:
                return False

            for _ in range(20):
                await omni.kit.app.get_app().next_update_async()

            self.gripper.close()

            for _ in range(150):
                await omni.kit.app.get_app().next_update_async()

            success = await self._move_to_target_cspace(pick_approach, orientation)
            if not success:
                return False

            success = await self._move_to_target_cspace(place_approach, orientation)
            if not success:
                return False

            success = await self._move_to_target_cspace(place_pos, orientation)
            if not success:
                return False

            for _ in range(20):
                await omni.kit.app.get_app().next_update_async()

            self.gripper.open()
            for _ in range(50):
                await omni.kit.app.get_app().next_update_async()

            success = await self._move_to_target_cspace(place_approach, orientation)
            if not success:
                return False

            if is_last_cube:
                home_pos = np.array([0.6, 0.0, 0.5])
                success = await self._move_to_target_cspace(home_pos, orientation)

                self.gripper.close()
                for _ in range(30):
                    await omni.kit.app.get_app().next_update_async()

            return True

        except Exception as e:
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
            target_joints, success = self.kinematics_solver.compute_inverse_kinematics(
                self._end_effector_name,
                target_position,
                target_orientation
            )

            if not success:
                return False

            current_joints = self.ur10.get_joint_positions()

            trajectory = self.cspace_generator.compute_c_space_trajectory(
                current_joints,
                target_joints
            )

            articulation_trajectory = ArticulationTrajectory(
                robot_articulation=self.ur10,
                trajectory=trajectory
            )

            actions = articulation_trajectory.get_action_sequence()
            for action in actions:
                self.ur10.apply_action(action)
                await omni.kit.app.get_app().next_update_async()

            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()

            final_joints = self.ur10.get_joint_positions()
            joint_error = np.linalg.norm(final_joints - target_joints)

            if joint_error > 0.1:
                return False

            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    app = CSpaceIK()
