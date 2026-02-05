import asyncio
import time
import numpy as np
import os
from pathlib import Path
import sys
import omni.ui as ui
from omni.kit.async_engine import run_coroutine
import omni.timeline

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
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from pxr import UsdPhysics
import carb

project_root = None

try:
    if '__file__' in globals():
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
except:
    pass

if project_root is None or not project_root.exists():
    try:
        cwd = Path(os.getcwd())
        if cwd.name == "multiagent":
            project_root = cwd
        else:
            for parent in [cwd] + list(cwd.parents):
                if parent.name == "multiagent":
                    project_root = parent
                    break
    except:
        pass

if project_root is None:
    project_root = Path.cwd()

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.manipulators import SingleManipulator
from src.grippers import ParallelGripper


class TrajectoryGeneration:
    def __init__(self):
        self.window = None
        self.world = None
        self.franka = None
        self.gripper = None
        self.container = None
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

        self.build_ui()

    def build_ui(self):
        """Build UI"""
        self.window = ui.Window("Cobot - Grasping", width=450, height=400)

        with self.window.frame:
            with ui.VStack(spacing=10):
                ui.Label("Cobot - Pick and Place",
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
        """Load the scene with Franka, dynamic grid of cubes, and container"""
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

            franka_name = f"franka_{int(time.time() * 1000)}"
            franka_prim_path = f"/World/Franka_{int(time.time() * 1000)}"

            franka_usd_path = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
            robot_prim = add_reference_to_stage(usd_path=franka_usd_path, prim_path=franka_prim_path)
            robot_prim.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
            robot_prim.GetVariantSet("Mesh").SetVariantSelection("Quality")

            await omni.kit.app.get_app().next_update_async()

            self.gripper = ParallelGripper(
                end_effector_prim_path=f"{franka_prim_path}/panda_rightfinger",
                joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
                joint_opened_positions=np.array([0.04, 0.04]),
                joint_closed_positions=np.array([0.0, 0.0]),
                action_deltas=np.array([0.01, 0.01])
            )

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

            container_position = np.array([0.55, 0.4, 0.0])

            self.container = self.world.scene.add(
                SingleXFormPrim(
                    prim_path=container_prim_path,
                    name="container",
                    translation=container_position,
                    scale=np.array([0.3, 0.3, 0.2])
                )
            )

            from omni.isaac.core.utils.stage import get_current_stage
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

            grid_center_x = 0.4
            grid_center_y = 0.0

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

            self.world.reset()

            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            default_joint_positions = np.array([0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75, 0.0, 0.0])
            self.franka.set_joints_default_state(positions=default_joint_positions)
            self.gripper.set_default_state(self.gripper.joint_closed_positions)

            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            articulation_controller = self.franka.get_articulation_controller()
            close_action = ArticulationAction(
                joint_positions=self.gripper.joint_closed_positions,
                joint_indices=np.array([7, 8])
            )
            articulation_controller.apply_action(close_action)

            for _ in range(20):
                await omni.kit.app.get_app().next_update_async()

            self._setup_rrt()

            self.pick_btn.enabled = True
            self.reset_btn.enabled = True
            self._update_status("Scene loaded! Ready to pick and place")

        except Exception as e:
            self._update_status(f"Error: {e}")
            carb.log_error(f"Error loading scene: {e}")
            import traceback
            traceback.print_exc()

    def _setup_rrt(self):
        """Setup RRT motion planner and kinematics solvers"""
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

        self.kinematics_solver = LulaKinematicsSolver(
            robot_description_path=robot_description_file,
            urdf_path=urdf_path
        )

        end_effector_name = "right_gripper"
        self.articulation_kinematics_solver = ArticulationKinematicsSolver(
            self.franka,
            self.kinematics_solver,
            end_effector_name
        )

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

    async def _pick_place_loop(self):
        """Main pick and place loop"""
        try:
            self.timeline.play()

            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            cubes = self.cubes

            total_cubes = len(cubes)

            for i, (cube, cube_name) in enumerate(cubes, 1):
                is_last = (i == total_cubes)
                success = await self._pick_and_place_cube(cube, cube_name.split()[1], is_last)
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
        """Pick and place a single cube"""
        try:
            cube_pos, _ = cube.get_world_pose()

            cube_size = 0.0515
            cube_half = cube_size / 2.0

            pick_approach = cube_pos + np.array([0.0, 0.0, 0.15])

            pick_pos = np.array([cube_pos[0], cube_pos[1], 0.02])

            container_center = np.array([0.55, 0.4, 0.0])

            place_spacing = 0.10

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

            home_pos = np.array([0.4, 0.0, 0.3])

            plan = self._plan_to_target(pick_approach, orientation)
            if plan is None:
                return False
            await self._execute_plan(plan)

            articulation_controller = self.franka.get_articulation_controller()
            open_action = ArticulationAction(
                joint_positions=self.gripper.joint_opened_positions,
                joint_indices=np.array([7, 8])
            )
            articulation_controller.apply_action(open_action)
            for _ in range(60):
                await omni.kit.app.get_app().next_update_async()

            success = await self._move_to_target_ik(pick_pos, orientation)
            if not success:
                plan = self._plan_to_target(pick_pos, orientation)
                if plan is None:
                    return False
                await self._execute_plan(plan)

            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

            articulation_controller = self.franka.get_articulation_controller()
            close_action = ArticulationAction(
                joint_positions=self.gripper.joint_closed_positions,
                joint_indices=np.array([7, 8])
            )
            articulation_controller.apply_action(close_action)
            for _ in range(50):
                await omni.kit.app.get_app().next_update_async()

            cube_pos_after_pick, _ = cube.get_world_pose()
            if abs(cube_pos_after_pick[2] - cube_pos[2]) < 0.001:
                carb.log_warn(f"Cube may not have been picked! Position unchanged: {cube_pos_after_pick}")

            plan = self._plan_to_target(pick_approach, orientation)
            if plan is None:
                return False
            await self._execute_plan(plan)

            plan = self._plan_to_target(safe_transit_pos, orientation)
            if plan is None:
                return False
            await self._execute_plan(plan)

            place_high = place_pos + np.array([0.0, 0.0, 0.25])
            plan = self._plan_to_target(place_high, orientation)
            if plan is None:
                return False
            await self._execute_plan(plan)

            success = await self._move_to_target_ik(place_approach, orientation, num_steps=20)
            if not success:
                plan = self._plan_to_target(place_approach, orientation)
                if plan is not None:
                    await self._execute_plan(plan)

            success = await self._move_to_target_ik(place_pos, orientation)
            if not success:
                plan = self._plan_to_target(place_pos, orientation)
                if plan is None:
                    return False
                await self._execute_plan(plan)

            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

            articulation_controller = self.franka.get_articulation_controller()
            open_action = ArticulationAction(
                joint_positions=self.gripper.joint_opened_positions,
                joint_indices=np.array([7, 8])
            )
            articulation_controller.apply_action(open_action)

            for _ in range(40):
                await omni.kit.app.get_app().next_update_async()

            place_clearance = place_pos + np.array([0.0, 0.0, 0.05])
            success = await self._move_to_target_ik(place_clearance, orientation, num_steps=15)
            if not success:
                plan = self._plan_to_target(place_clearance, orientation)
                if plan is not None:
                    await self._execute_plan(plan)

            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()

            plan = self._plan_to_target(place_high, orientation)
            if plan is None:
                return False
            await self._execute_plan(plan)
            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()

            if is_last_cube:
                safe_intermediate = np.array([0.5, 0.2, 0.35])
                try:
                    plan = self._plan_to_target(safe_intermediate, orientation)
                    if plan is not None:
                        await self._execute_plan(plan)
                        for _ in range(10):
                            await omni.kit.app.get_app().next_update_async()
                except Exception as e:
                    carb.log_warn(f"Error moving to intermediate position: {e}")

                try:
                    plan = self._plan_to_target(home_pos, orientation)
                    if plan is not None:
                        await self._execute_plan(plan)
                        for _ in range(10):
                            await omni.kit.app.get_app().next_update_async()
                except Exception as e:
                    carb.log_warn(f"Could not return to home position: {e}")

                articulation_controller = self.franka.get_articulation_controller()
                close_action = ArticulationAction(
                    joint_positions=self.gripper.joint_closed_positions,
                    joint_indices=np.array([7, 8])
                )
                articulation_controller.apply_action(close_action)
                for _ in range(50):
                    await omni.kit.app.get_app().next_update_async()
            else:
                safe_intermediate = np.array([0.45, 0.0, 0.35])
                try:
                    plan = self._plan_to_target(safe_intermediate, orientation)
                    if plan is not None:
                        await self._execute_plan(plan)
                        for _ in range(10):
                            await omni.kit.app.get_app().next_update_async()
                except Exception as e:
                    carb.log_warn(f"Error moving to intermediate position: {e}")

                articulation_controller = self.franka.get_articulation_controller()
                close_action = ArticulationAction(
                    joint_positions=self.gripper.joint_closed_positions,
                    joint_indices=np.array([7, 8])
                )
                articulation_controller.apply_action(close_action)
                for _ in range(30):
                    await omni.kit.app.get_app().next_update_async()

            return True

        except Exception as e:
            carb.log_error(f"Error picking {cube_name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _move_to_target_ik(self, target_position, target_orientation, num_steps=30):
        """Move to target using IK directly for simple, straight-line movements"""
        robot_base_translation, robot_base_orientation = self.franka.get_world_pose()
        self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        ik_action, ik_success = self.articulation_kinematics_solver.compute_inverse_kinematics(
            target_position, target_orientation
        )

        if not ik_success:
            return False

        current_positions = self.franka.get_joint_positions()[:7]
        target_positions = ik_action.joint_positions[:7]

        articulation_controller = self.franka.get_articulation_controller()
        for i in range(num_steps):
            alpha = (i + 1) / num_steps
            interpolated_positions = current_positions + alpha * (target_positions - current_positions)

            action = ArticulationAction(
                joint_positions=interpolated_positions,
                joint_indices=np.array([0, 1, 2, 3, 4, 5, 6])
            )
            articulation_controller.apply_action(action)
            await omni.kit.app.get_app().next_update_async()

        return True

    def _plan_to_target(self, target_position, target_orientation):
        """Plan path to target using RRT with IK validation"""
        robot_base_translation, robot_base_orientation = self.franka.get_world_pose()
        self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        self.rrt.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        _, ik_success = self.articulation_kinematics_solver.compute_inverse_kinematics(
            target_position, target_orientation
        )

        if not ik_success:
            carb.log_warn(f"IK did not converge for target position {target_position}. RRT may fail.")

        self.rrt.set_end_effector_target(target_position, target_orientation)
        self.rrt.update_world()
        plan = self.path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=0.02)
        return plan

    async def _execute_plan(self, plan):
        """Execute a plan"""
        if plan is None:
            return False

        for action in plan:
            self.franka.apply_action(action)
            await omni.kit.app.get_app().next_update_async()

        return True

    def compute_forward_kinematics(self):
        """Compute forward kinematics to get current end effector pose"""
        if self.articulation_kinematics_solver is None:
            carb.log_warn("Articulation kinematics solver not initialized")
            return None, None

        robot_base_translation, robot_base_orientation = self.franka.get_world_pose()
        self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        ee_position, ee_rot_mat = self.articulation_kinematics_solver.compute_end_effector_pose()

        return ee_position, ee_rot_mat

    def _on_reset(self):
        """Reset button callback"""
        try:
            self.is_picking = False

            self.timeline.stop()

            if self.world is not None:
                World.clear_instance()

            from omni.isaac.core.utils.stage import clear_stage
            clear_stage()

            self.world = None
            self.franka = None
            self.gripper = None
            self.container = None
            self.rrt = None
            self.path_planner_visualizer = None
            self.kinematics_solver = None
            self.articulation_kinematics_solver = None
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


app = TrajectoryGeneration()

