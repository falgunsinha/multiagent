
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
from isaacsim.core.api.objects import DynamicCylinder
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationAction
from pxr import UsdPhysics
import carb

from isaacsim.robot_motion.motion_generation import (
    PathPlannerVisualizer,
    LulaKinematicsSolver,
    ArticulationKinematicsSolver
)
from isaacsim.robot_motion.motion_generation.lula import RRT

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


class RRTKinematicsSolver:
    """Pick and Place with RRT + Kinematics Solver"""

    def __init__(self):
        self.window = None
        self.world = None
        self.ur10 = None
        self.gripper = None
        self.container = None
        self.container_dimensions = None
        self.rrt = None
        self.path_planner_visualizer = None
        self.kinematics_solver = None
        self.articulation_kinematics_solver = None
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
        self._end_effector_name = "ee_suction_link"
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
        if not self.world or not self.rrt:
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
                self.world.clear()
                World.clear_instance()
                self.world = None

            stage = omni.usd.get_context().get_stage()
            if stage:
                world_prim = stage.GetPrimAtPath("/World")
                if world_prim.IsValid():
                    stage.RemovePrim("/World")

                physics_scene_prim = stage.GetPrimAtPath("/physicsScene")
                if physics_scene_prim.IsValid():
                    stage.RemovePrim("/physicsScene")

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

            self.world = World(stage_units_in_meters=1.0)
            await self.world.initialize_simulation_context_async()

            self.world.scene.add_default_ground_plane()

            assets_root_path = get_assets_root_path()
            ur10_usd_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"

            ur10_name = "ur10"
            ur10_prim_path = f"/World/{ur10_name}"

            add_reference_to_stage(usd_path=ur10_usd_path, prim_path=ur10_prim_path)

            stage = get_current_stage()
            robot_prim = stage.GetPrimAtPath(ur10_prim_path)

            robot_prim.GetVariantSet("Gripper").SetVariantSelection("Short_Suction")

            ee_link_prim = stage.GetPrimAtPath(f"{ur10_prim_path}/ee_link")
            if ee_link_prim.IsValid():
                camera_components = ["Camera", "camera_mount", "camera_geom"]
                for component_name in camera_components:
                    component_prim = stage.GetPrimAtPath(f"{ur10_prim_path}/ee_link/{component_name}")
                    if component_prim.IsValid():
                        component_prim.SetActive(True)

            surface_gripper_prim_path = f"{ur10_prim_path}/ee_link/SurfaceGripper"
            gripper_prim = stage.GetPrimAtPath(surface_gripper_prim_path)

            if gripper_prim.IsValid():
                from pxr import UsdPhysics
                suction_joint_path = f"{surface_gripper_prim_path}/suction_cup/Suction_Joint"
                suction_joint_prim = stage.GetPrimAtPath(suction_joint_path)
                if suction_joint_prim.IsValid():
                    if suction_joint_prim.HasAttribute("isaac:clearanceOffset"):
                        suction_joint_prim.GetAttribute("isaac:clearanceOffset").Set(0.003)

                self.gripper = SurfaceGripper(
                    end_effector_prim_path=f"{ur10_prim_path}/ee_link",
                    surface_gripper_path=surface_gripper_prim_path
                )
            else:
                raise Exception(f"SurfaceGripper not found at {surface_gripper_prim_path}")

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

            container_position = np.array([0.75, 0.5, 0.0])
            self.container = await self._create_container(container_position)

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

            self._setup_hybrid_motion_generation()

            if self.pick_btn:
                self.pick_btn.enabled = True

            self._update_status(f"Scene loaded! {self.grid_length}x{self.grid_width} grid ready")

        except Exception as e:
            self._update_status(f"Load error: {e}")
            import traceback
            traceback.print_exc()

    def _setup_hybrid_motion_generation(self):
        """Setup motion generation tools (RRT + Kinematics)"""
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
        rrt_config_dir = os.path.join(mg_extension_path, "path_planner_configs")

        robot_description_path = os.path.join(rmp_config_dir, "universal_robots/ur10/rmpflow_suction/ur10_robot_description.yaml")
        urdf_path = os.path.join(rmp_config_dir, "universal_robots/ur10/ur10_robot_suction.urdf")
        rrt_config_path = os.path.join(rrt_config_dir, "universal_robots/ur10/rrt/ur10_planner_config.yaml")

        self.rrt = RRT(
            robot_description_path=robot_description_path,
            urdf_path=urdf_path,
            rrt_config_path=rrt_config_path,
            end_effector_frame_name=self._end_effector_name
        )
        self.rrt.set_max_iterations(10000)

        self.path_planner_visualizer = PathPlannerVisualizer(
            robot_articulation=self.ur10,
            path_planner=self.rrt
        )

        self.kinematics_solver = LulaKinematicsSolver(
            robot_description_path=robot_description_path,
            urdf_path=urdf_path
        )

        self.articulation_kinematics_solver = ArticulationKinematicsSolver(
            self.ur10,
            self.kinematics_solver,
            self._end_effector_name
        )

    async def _create_container(self, position):
        """Create container for placing cubes"""
        container_name = f"container_{int(time.time() * 1000)}"
        container_prim_path = f"/World/{container_name}"
        container_usd_path = f"{get_assets_root_path()}/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Storage/Containers/Container_I/Container_I04_160x120x64cm_PR_V_NVD_01.usd"

        add_reference_to_stage(usd_path=container_usd_path, prim_path=container_prim_path)

        scale = np.array([0.4, 0.4, 0.25])
        original_size = np.array([1.60, 1.20, 0.64])

        self.container_dimensions = original_size * scale

        container = self.world.scene.add(
            SingleXFormPrim(
                prim_path=container_prim_path,
                name=container_name,
                translation=position,
                scale=scale
            )
        )

        stage = get_current_stage()
        container_prim = stage.GetPrimAtPath(container_prim_path)
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(container_prim)
        rigid_body_api.CreateKinematicEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(container_prim)

        await omni.kit.app.get_app().next_update_async()

        return container

    def _create_cube_grid(self):
        """Create cylinders in a grid pattern"""
        cylinder_radius = 0.04
        cylinder_height = 0.10
        cylinder_spacing = 0.15

        grid_center_x = 0.7
        grid_center_y = -0.4

        cylinder_color = np.array([1.0, 1.0, 1.0])

        grid_start_x = grid_center_x - ((self.grid_length - 1) * cylinder_spacing) / 2.0
        grid_start_y = grid_center_y - ((self.grid_width - 1) * cylinder_spacing) / 2.0

        cube_index = 0
        for row in range(self.grid_length):
            for col in range(self.grid_width):
                x = grid_start_x + (row * cylinder_spacing)
                y = grid_start_y + (col * cylinder_spacing)
                z = cylinder_height / 2.0 + 0.01

                timestamp = int(time.time() * 1000)
                cube_name = f"Cylinder_R{row+1}_C{col+1}_{timestamp}"
                cube_display_name = f"Cylinder_{row+1}_{col+1}"

                cube = self.world.scene.add(
                    DynamicCylinder(
                        prim_path=f"/World/{cube_name}",
                        name=cube_name,
                        position=np.array([x, y, z]),
                        radius=cylinder_radius,
                        height=cylinder_height,
                        color=cylinder_color,
                        mass=0.08
                    )
                )

                self.cubes.append({
                    'object': cube,
                    'name': cube_name,
                    'display_name': cube_display_name,
                    'color': 'White',
                    'row': row + 1,
                    'col': col + 1
                })

                cube_index += 1
                time.sleep(0.001)

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
                cube_display_name = cube_data['display_name']
                color_name = cube_data['color']
                is_last_cube = (idx == total_cubes - 1)

                success = await self._pick_and_place_cube_hybrid(cube, cube_display_name, color_name, idx, is_last_cube)

                if not success:
                    self.is_picking = False
                    self._update_status(f"Failed at cube {idx+1}/{total_cubes}")
                    return

            if self.is_picking:
                self._update_status(f"All {total_cubes} cubes placed!")
                self.is_picking = False

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.is_picking = False
            self._update_status(f"Error: {e}")

    async def _pick_and_place_cube_hybrid(self, cube, cube_name, color_name, cube_index, is_last_cube=False):
        """Pick and place a single cube using RRT + Kinematics Solver"""
        try:
            cube_pos, _ = cube.get_world_pose()

            cylinder_height = 0.10
            container_pos, _ = self.container.get_world_pose()

            container_length = self.container_dimensions[0]
            container_width = self.container_dimensions[1]
            container_height = self.container_dimensions[2]

            grid_length = int(self.length_field.model.get_value_as_int())
            grid_width = int(self.width_field.model.get_value_as_int())

            cubes_per_row = grid_width
            total_rows = grid_length
            row = cube_index // cubes_per_row
            col = cube_index % cubes_per_row

            margin = 0.20

            available_length = container_length - 2 * margin
            available_width = container_width - 2 * margin

            if cubes_per_row > 1:
                spacing_x = available_length / (cubes_per_row - 1)
            else:
                spacing_x = 0

            if total_rows > 1:
                spacing_y = available_width / (total_rows - 1)
            else:
                spacing_y = 0

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

            place_pos = container_pos + np.array([
                offset_x,
                offset_y,
                container_height/2 + cylinder_height/2 + 0.030
            ])

            cylinder_half_height = cylinder_height / 2.0

            pre_pick_pos = cube_pos + np.array([0.0, 0.0, 0.12])

            pick_height = cube_pos[2] + cylinder_half_height + 0.003
            pick_pos = np.array([cube_pos[0], cube_pos[1], pick_height])

            pre_place_pos = place_pos + np.array([0.0, 0.0, 0.25])

            orientation = euler_angles_to_quat(np.array([0, np.pi / 2.0, 0]))

            success = await self._move_to_target_rrt(pre_pick_pos, orientation)
            if not success:
                return False

            self.gripper.open()
            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()

            current_ee_pos, _ = self.ur10.end_effector.get_world_pose()
            success = await self._taskspace_straight_line(current_ee_pos, pick_pos, orientation, num_waypoints=30)
            if not success:
                return False

            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()

            self.gripper.close()
            for _ in range(15):
                await omni.kit.app.get_app().next_update_async()

            current_ee_pos, _ = self.ur10.end_effector.get_world_pose()
            retreat_pos = current_ee_pos + np.array([0.0, 0.0, 0.08])
            success = await self._taskspace_straight_line(current_ee_pos, retreat_pos, orientation, num_waypoints=30)
            if not success:
                return False

            for _ in range(15):
                await omni.kit.app.get_app().next_update_async()

            success = await self._move_to_target_rrt(pre_place_pos, orientation)
            if not success:
                return False

            current_ee_pos, _ = self.ur10.end_effector.get_world_pose()
            success = await self._taskspace_straight_line(current_ee_pos, place_pos, orientation, num_waypoints=30)
            if not success:
                return False

            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            self.gripper.open()
            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()

            current_ee_pos, _ = self.ur10.end_effector.get_world_pose()
            retreat_pos = current_ee_pos + np.array([0.0, 0.0, 0.10])
            success = await self._taskspace_straight_line(current_ee_pos, retreat_pos, orientation, num_waypoints=30)
            if not success:
                return False

            if is_last_cube:
                home_pos = np.array([0.6, 0.0, 0.5])
                success = await self._move_to_target_rrt(home_pos, orientation)

                self.gripper.close()
                for _ in range(30):
                    await omni.kit.app.get_app().next_update_async()

            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False

    async def _correct_orientation(self, target_pos, target_orientation, num_steps=20):
        """Smoothly correct orientation at current position after RRT"""
        if not self.ur10.handles_initialized:
            return False

        robot_base_translation, robot_base_orientation = self.ur10.get_world_pose()
        self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        current_joint_positions = self.ur10.get_joint_positions()

        ik_action, ik_success = self.articulation_kinematics_solver.compute_inverse_kinematics(
            target_pos, target_orientation
        )

        if not ik_success:
            return True

        target_joint_positions = ik_action.joint_positions

        for i in range(1, num_steps + 1):
            alpha = i / num_steps
            interpolated_joints = current_joint_positions + alpha * (target_joint_positions - current_joint_positions)

            action = ArticulationAction(joint_positions=interpolated_joints)
            self.ur10.apply_action(action)
            await omni.kit.app.get_app().next_update_async()

        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()

        return True

    async def _taskspace_straight_line(self, start_pos, end_pos, orientation, num_waypoints=10):
        """Execute straight-line motion using IK + joint-space interpolation"""
        if not self.ur10.handles_initialized:
            return False

        robot_base_translation, robot_base_orientation = self.ur10.get_world_pose()
        self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        current_joint_positions = self.ur10.get_joint_positions()

        ik_action, ik_success = self.articulation_kinematics_solver.compute_inverse_kinematics(
            end_pos, orientation
        )

        if not ik_success:
            return False

        end_joint_positions = ik_action.joint_positions

        for i in range(1, num_waypoints + 1):
            alpha = i / num_waypoints
            interpolated_joints = current_joint_positions + alpha * (end_joint_positions - current_joint_positions)

            action = ArticulationAction(joint_positions=interpolated_joints)
            self.ur10.apply_action(action)

            await omni.kit.app.get_app().next_update_async()

        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()

        return True

    def _plan_to_target(self, target_position, target_orientation):
        """Plan path to target using RRT with IK validation"""
        if not self.ur10.handles_initialized:
            return None

        robot_base_translation, robot_base_orientation = self.ur10.get_world_pose()
        self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        self.rrt.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        _, ik_success = self.articulation_kinematics_solver.compute_inverse_kinematics(
            target_position, target_orientation
        )

        self.rrt.set_end_effector_target(target_position, target_orientation)
        self.rrt.update_world()
        plan = self.path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=0.015)
        return plan

    async def _execute_plan(self, plan):
        """Execute a plan"""
        if plan is None:
            return False

        for action in plan:
            self.ur10.apply_action(action)
            await omni.kit.app.get_app().next_update_async()

        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()

        return True

    async def _move_to_target_rrt(self, target_position, target_orientation, maintain_orientation=False):
        """Move to target using RRT"""
        plan = self._plan_to_target(target_position, target_orientation)

        if plan is None:
            return False

        success = await self._execute_plan(plan)

        return success

if __name__ == "__main__":
    app = RRTKinematicsSolver()

