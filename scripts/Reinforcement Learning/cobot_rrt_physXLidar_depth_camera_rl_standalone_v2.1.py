
import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Franka RRT Pick-and-Place with RL Object Selection (Standalone)")
parser.add_argument("--rl_model", type=str, default=None,
                   help="Path to trained RL model (PPO: .zip, DDQN: .pt)")
parser.add_argument("--num_cubes", type=int, default=None,
                   help="Number of cubes to spawn (default: auto-detect from model metadata)")
parser.add_argument("--training_grid_size", type=int, default=None,
                   help="Fixed training grid size (default: auto-detect from model metadata)")
args, unknown = parser.parse_known_args()

if args.rl_model:
    import json
    import os

    model_path = args.rl_model
    if 'ddqn' in model_path.lower():
        if not model_path.endswith('.pt') and not model_path.endswith('.zip'):
            model_path += '.pt'
        metadata_path = model_path.replace('.pt', '_metadata.json')
    else:
        if not model_path.endswith('.zip') and not model_path.endswith('.pt'):
            model_path += '.zip'
        metadata_path = model_path.replace('_final.zip', '_metadata.json')
        if not os.path.exists(metadata_path):
            metadata_path = model_path.replace('.zip', '_metadata.json')

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

                if args.training_grid_size is None:
                    args.training_grid_size = metadata.get("training_grid_size", 6)

                if args.num_cubes is None:
                    args.num_cubes = metadata.get("num_cubes", args.training_grid_size * args.training_grid_size - 1)
        except Exception as e:
            pass

if args.training_grid_size is None:
    args.training_grid_size = 6
if args.num_cubes is None:
    args.num_cubes = 4

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
import asyncio
import time
import numpy as np
import os
import omni.ui as ui
from omni.kit.async_engine import run_coroutine
import omni.timeline
import omni.usd
import omni.usd.editor

from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder, VisualCuboid, FixedCuboid
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import PathPlannerVisualizer, ArticulationTrajectory
from isaacsim.robot_motion.motion_generation.lula import RRT
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from isaacsim.robot_motion.motion_generation.lula.trajectory_generator import LulaCSpaceTrajectoryGenerator
from pxr import UsdPhysics, PhysxSchema, Gf, UsdGeom, Sdf, Usd, UsdShade
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
import omni.isaac.core.utils.prims as prim_utils
import carb

from isaacsim.sensors.physx import RotatingLidarPhysX

from isaacsim.sensors.camera import SingleViewDepthSensor

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
from src.grippers import ParallelGripper, GraspConfig
from src.rl.visual_grid import create_visual_grid

RL_AVAILABLE = False
if args.rl_model:
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from src.rl.doubleDQN.double_dqn_agent import DoubleDQNAgent
        from src.rl.object_selection_env import ObjectSelectionEnv
        import torch
        RL_AVAILABLE = True
    except ImportError as e:
        RL_AVAILABLE = False


class PerformanceTracker:
    """Track and compare performance metrics for different RL methods"""

    def __init__(self):
        self.start_time = None
        self.pick_times = []
        self.pick_order = []
        self.successes = []
        self.failures = []
        self.total_distance = 0.0
        self.method_name = "Unknown"

    def start(self, method_name="Greedy"):
        """Start tracking a new run"""
        self.start_time = time.time()
        self.pick_times = []
        self.pick_order = []
        self.successes = []
        self.failures = []
        self.total_distance = 0.0
        self.method_name = method_name

    def record_pick(self, cube_index, success, pick_time, distance):
        """Record a pick attempt"""
        self.pick_order.append(cube_index)
        self.pick_times.append(pick_time)
        if success:
            self.successes.append(cube_index)
        else:
            self.failures.append(cube_index)
        self.total_distance += distance

    def get_summary(self):
        """Get performance summary"""
        if self.start_time is None:
            return {}

        total_time = time.time() - self.start_time
        avg_pick_time = np.mean(self.pick_times) if self.pick_times else 0.0
        success_rate = len(self.successes) / (len(self.successes) + len(self.failures)) if (len(self.successes) + len(self.failures)) > 0 else 0.0

        return {
            "method": self.method_name,
            "total_time": total_time,
            "avg_pick_time": avg_pick_time,
            "pick_order": self.pick_order,
            "successes": len(self.successes),
            "failures": len(self.failures),
            "success_rate": success_rate * 100,
            "total_distance": self.total_distance,
            "avg_distance": self.total_distance / len(self.pick_times) if self.pick_times else 0.0
        }

    def print_summary(self):
        """Print performance summary"""
        summary = self.get_summary()
        if not summary:
            return

        print("\n" + "=" * 60)
        print(f"PERFORMANCE SUMMARY - {summary['method']}")
        print("=" * 60)
        print(f"Total time: {summary['total_time']:.2f} seconds")
        print(f"Average pick time: {summary['avg_pick_time']:.2f} seconds")
        print(f"Pick order: {summary['pick_order']}")
        print(f"Successes: {summary['successes']}/{summary['successes'] + summary['failures']} ({summary['success_rate']:.1f}%)")
        print(f"Total distance traveled: {summary['total_distance']:.2f} meters")
        print(f"Average distance per pick: {summary['avg_distance']:.2f} meters")
        print("=" * 60 + "\n")


class FrankaRRTDynamicGrid:
    """PyGame-style pick and place with RRT + RL object selection (Standalone)"""

    def __init__(self, num_cubes=4, training_grid_size=6):
        self.window = None
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

        self.num_cubes = num_cubes
        self.training_grid_size = training_grid_size

        max_capacity = self.training_grid_size * self.training_grid_size
        if self.num_cubes > max_capacity:
            self.num_cubes = max_capacity

        self.container_dimensions = None

        self.obstacles = {}
        self.obstacle_counter = 0

        self.cube_obstacles_enabled = False

        self.grasp_config = None
        self.current_grasp_name = "grasp_0"

        self.obstacle_1_moving = False
        self.obstacle_1_acceleration = 6.0
        self.obstacle_1_min_x = 0.2
        self.obstacle_1_max_x = 0.63
        self.obstacle_1_force_api_applied = False

        self.lidar = None
        self.lidar_detected_obstacles = {}

        self.depth_camera = None

        self.timeline = omni.timeline.get_timeline_interface()

        self.is_picking = False
        self.placed_count = 0
        self.current_cube_index = 0

        self.rl_model = None
        self.rl_model_path = args.rl_model
        self.rl_model_type = None

        if self.rl_model_path:
            if 'ddqn' in self.rl_model_path.lower():
                self.rl_model_type = 'ddqn'
                if not self.rl_model_path.endswith('.pt') and not self.rl_model_path.endswith('.zip'):
                    self.rl_model_path += '.pt'
            else:
                self.rl_model_type = 'ppo'
                if not self.rl_model_path.endswith('.zip') and not self.rl_model_path.endswith('.pt'):
                    self.rl_model_path += '.zip'

        self.use_rl = RL_AVAILABLE and args.rl_model is not None

        self.performance_tracker = PerformanceTracker()

        self.robot_stand = None
        self.table = None

        self.load_btn = None
        self.pick_btn = None
        self.reset_btn = None
        self.add_obstacle_btn = None
        self.remove_obstacle_btn = None
        self.status_label = None
        self.num_cubes_field = None
        self.training_grid_field = None

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

                with ui.CollapsableFrame("Grid Configuration", height=0):
                    with ui.VStack(spacing=5):
                        with ui.HStack(spacing=10):
                            ui.Label("Training Grid Size:", width=150)
                            training_grid_model = ui.SimpleIntModel(self.training_grid_size)
                            self.training_grid_field = ui.IntField(height=25, model=training_grid_model)

                        with ui.HStack(spacing=10):
                            ui.Label("Number of Cubes:", width=150)
                            num_cubes_model = ui.SimpleIntModel(self.num_cubes)
                            self.num_cubes_field = ui.IntField(height=25, model=num_cubes_model)



                ui.Spacer(height=10)

                self.load_btn = ui.Button("Load Scene", height=40, clicked_fn=self._on_load)
                self.pick_btn = ui.Button("Start Pick and Place", height=40, clicked_fn=self._on_pick, enabled=False)
                self.reset_btn = ui.Button("Reset Scene", height=40, clicked_fn=self._on_reset, enabled=False)

                ui.Spacer(height=10)

                with ui.HStack(spacing=10):
                    self.add_obstacle_btn = ui.Button("Add Obstacle", height=35, clicked_fn=self._on_add_obstacle, enabled=False)
                    self.remove_obstacle_btn = ui.Button("Remove Obstacle", height=35, clicked_fn=self._on_remove_obstacle, enabled=False)

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
        """Load the scene with Franka, PyGame-style cube placement, and container"""
        try:
            self.training_grid_size = int(self.training_grid_field.model.get_value_as_int())
            self.num_cubes = int(self.num_cubes_field.model.get_value_as_int())

            if self.training_grid_size < 1:
                self._update_status("Error: Training grid size must be at least 1")
                return
            if self.training_grid_size > 10:
                self._update_status("Error: Training grid size too large (max 10)")
                return

            if self.num_cubes < 1:
                self._update_status("Error: Number of cubes must be at least 1")
                return

            max_capacity = self.training_grid_size * self.training_grid_size
            if self.num_cubes > max_capacity:
                self._update_status(f"Error: No. of cubes are more than grid size. Max {max_capacity} for {self.training_grid_size}x{self.training_grid_size} grid")
                return

            self.timeline.stop()
            World.clear_instance()

            await omni.kit.app.get_app().next_update_async()

            self.world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
            self.world.scene.add_default_ground_plane()

            from pxr import Usd, UsdGeom, Sdf
            stage = omni.usd.get_context().get_stage()
            camera_usd_path = "C:/isaacsim/cobotproject/assets/Main_camera.usd"

            camera_stage = Usd.Stage.Open(camera_usd_path)
            if camera_stage:
                default_prim = camera_stage.GetDefaultPrim()
                if default_prim and default_prim.IsValid():
                    Sdf.CopySpec(camera_stage.GetRootLayer(), default_prim.GetPath(),
                                stage.GetRootLayer(), Sdf.Path("/World/Main_Camera"))
                else:
                    for prim in camera_stage.Traverse():
                        if prim.IsA(UsdGeom.Camera):
                            Sdf.CopySpec(camera_stage.GetRootLayer(), prim.GetPath(),
                                        stage.GetRootLayer(), Sdf.Path("/World/Main_Camera"))
                            break

            await omni.kit.app.get_app().next_update_async()

            assets_root_path = get_assets_root_path()
            stand_prim_path = "/World/RobotStand"
            stand_usd_path = assets_root_path + "/Isaac/Props/Mounts/textured_table.usd"
            add_reference_to_stage(usd_path=stand_usd_path, prim_path=stand_prim_path)

            stand_translation = np.array([-0.1, 0.0, 0.6])
            stand_orientation = np.array([0.7071, 0.0, 0.0, -0.7071])
            stand_scale = np.array([0.5, 0.4, 0.6])
            self.robot_stand = self.world.scene.add(
                SingleXFormPrim(
                    prim_path=stand_prim_path,
                    name="robot_stand",
                    translation=stand_translation,
                    orientation=stand_orientation,
                    scale=stand_scale
                )
            )

            stand_prim = stage.GetPrimAtPath(stand_prim_path)
            if stand_prim.IsValid():
                if stand_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    stand_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                if not stand_prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(stand_prim)

            table_prim_path = "/World/Table"
            table_usd_path = assets_root_path + "/Isaac/Environments/Simple_Room/Props/table_low.usd"
            add_reference_to_stage(usd_path=table_usd_path, prim_path=table_prim_path)

            table_top_height = 0.75
            table_orientation = np.array([0.7071, 0.0, 0.0, 0.7071])

            self.table = self.world.scene.add(
                SingleXFormPrim(
                    prim_path=table_prim_path,
                    name="table",
                    translation=np.array([0.5, 0.1, 0.0]),
                    orientation=table_orientation,
                    scale=np.array([0.9, 0.7, 0.9])
                )
            )

            table_prim = stage.GetPrimAtPath(table_prim_path)
            if table_prim:
                if table_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    table_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                if not table_prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(table_prim)

            material_prim_path = "/World/Table/Looks/MI_Table"
            material_usd_path = assets_root_path + "/Isaac/Environments/Simple_Room/Materials/MI_Table.mdl"

            table_low_prim_path = "/World/Table/table_low"
            table_low_prim = stage.GetPrimAtPath(table_low_prim_path)
            if table_low_prim:
                material = UsdShade.Material.Get(stage, material_prim_path)
                if not material:
                    material = UsdShade.Material.Define(stage, material_prim_path)
                    mdl_shader = UsdShade.Shader.Define(stage, material_prim_path + "/Shader")
                    mdl_shader.CreateIdAttr("mdlMaterial")
                    mdl_shader.SetSourceAsset(material_usd_path, "mdl")
                    mdl_shader.SetSourceAssetSubIdentifier("MI_Table", "mdl")
                    material.CreateSurfaceOutput("mdl").ConnectToSource(mdl_shader.ConnectableAPI(), "out")

                UsdShade.MaterialBindingAPI(table_low_prim).Bind(material)

            franka_name = f"franka_{int(time.time() * 1000)}"
            franka_prim_path = f"/World/Franka_{int(time.time() * 1000)}"

            franka_usd_path = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
            robot_prim = add_reference_to_stage(usd_path=franka_usd_path, prim_path=franka_prim_path)
            robot_prim.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
            robot_prim.GetVariantSet("Mesh").SetVariantSelection("Quality")

            robot_translation = stand_translation
            self._franka_prim_path = franka_prim_path
            self._franka_name = franka_name
            self._end_effector_prim_path = f"{franka_prim_path}/panda_rightfinger"
            self._robot_translation = robot_translation

            lidar_prim_path = f"{franka_prim_path}/lidar_sensor"

            lidar_translation = np.array([0.0, 0.0, 0.15])

            self.lidar = self.world.scene.add(
                RotatingLidarPhysX(
                    prim_path=lidar_prim_path,
                    name="franka_lidar",
                    translation=lidar_translation,
                    rotation_frequency=20.0,
                    fov=(360.0, 30.0),
                    resolution=(1.0, 1.0),
                    valid_range=(0.4, 100.0)
                )
            )

            depth_camera_prim_path = "/World/depth_camera"
            position = np.array([0.75, 0.1, 1.9])
            orientation = euler_angles_to_quats(np.array([0.0, 0.0, 0.0]), degrees=True)

            self.depth_camera = SingleViewDepthSensor(
                prim_path=depth_camera_prim_path,
                name="depth_camera",
                translation=position,
                orientation=orientation,
                resolution=(512, 512),
                frequency=10
            )

            self.world.scene.add(self.depth_camera)

            xform_prim = stage.GetPrimAtPath(depth_camera_prim_path)
            if xform_prim:
                xformable = UsdGeom.Xformable(xform_prim)
                scale_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]
                if scale_ops:
                    scale_ops[0].Set(Gf.Vec3d(0.3, 0.3, 0.3))
                else:
                    xform_op_scale = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
                    xform_op_scale.Set(Gf.Vec3d(0.3, 0.3, 0.3))

                orient_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpName() == "xformOp:orient"]
                if orient_ops:
                    quat_orientation = euler_angles_to_quats(np.array([5.0, 0.0, 90.0]), degrees=True)
                    orient_ops[0].Set(Gf.Quatd(float(quat_orientation[0]), float(quat_orientation[1]),
                                               float(quat_orientation[2]), float(quat_orientation[3])))
                else:
                    xform_op_rotate = xformable.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble)
                    xform_op_rotate.Set(Gf.Vec3d(5.0, 0.0, 90.0))

                camera_prim = None
                for child in xform_prim.GetChildren():
                    if child.GetTypeName() == "Camera":
                        camera_prim = child
                        break

                if camera_prim:
                    focal_length_attr = camera_prim.GetAttribute("focalLength")
                    if focal_length_attr:
                        focal_length_attr.Set(1.3)

                    camera_prim.GetAttribute("horizontalAperture").Set(20.955)
                    camera_prim.GetAttribute("verticalAperture").Set(20.955)
                    camera_prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.01, 10000.0))

            await omni.kit.app.get_app().next_update_async()

            container_prim_path = "/World/Container"
            container_usd_path = f"{get_assets_root_path()}/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Storage/Containers/Container_I/Container_I04_160x120x64cm_PR_V_NVD_01.usd"
            add_reference_to_stage(usd_path=container_usd_path, prim_path=container_prim_path)

            container_position = np.array([0.35, 0.5, 0.6])
            scale = np.array([0.3, 0.3, 0.2])
            original_size = np.array([1.60, 1.20, 0.64])
            self.container_dimensions = original_size * scale

            self.container = self.world.scene.add(
                SingleXFormPrim(
                    prim_path=container_prim_path,
                    name="container",
                    translation=container_position,
                    scale=scale
                )
            )

            from omni.isaac.core.utils.stage import get_current_stage
            stage = get_current_stage()
            container_prim = stage.GetPrimAtPath(container_prim_path)
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(container_prim)
            rigid_body_api.CreateKinematicEnabledAttr(True)
            UsdPhysics.CollisionAPI.Apply(container_prim)

            cylinder_radius = 0.0258
            cylinder_height = 0.0515

            total_cubes = self.num_cubes

            if total_cubes <= 4:
                cylinder_spacing = 0.14
            elif total_cubes <= 9:
                cylinder_spacing = 0.12
            else:
                cylinder_spacing = 0.10

            grid_center_x = 0.40
            grid_center_y = -0.08
            grid_extent_x = self.training_grid_size * cylinder_spacing
            grid_extent_y = self.training_grid_size * cylinder_spacing
            start_x = grid_center_x - (grid_extent_x / 2.0)
            start_y = grid_center_y - (grid_extent_y / 2.0)

            create_visual_grid(start_x, start_y, grid_extent_x, grid_extent_y, cylinder_spacing, self.training_grid_size, self.training_grid_size, z_height=0.61)

            random_offset_range = 0.0
            total_cells = self.training_grid_size * self.training_grid_size

            selected_indices = np.random.choice(total_cells, size=self.num_cubes, replace=False)
            selected_cells = set(selected_indices)

            self.cubes = []
            cylinder_index = 0

            for row in range(self.training_grid_size):
                for col in range(self.training_grid_size):
                    cell_index = row * self.training_grid_size + col

                    if cell_index not in selected_cells:
                        continue

                    cell_center_x = start_x + (col * cylinder_spacing) + (cylinder_spacing / 2.0)
                    cell_center_y = start_y + (row * cylinder_spacing) + (cylinder_spacing / 2.0)

                    random_offset_x = np.random.uniform(-random_offset_range, random_offset_range)
                    random_offset_y = np.random.uniform(-random_offset_range, random_offset_range)

                    cylinder_x = cell_center_x + random_offset_x
                    cylinder_y = cell_center_y + random_offset_y
                    cylinder_scale = np.array([1.4, 1.4, 1.4])
                    scaled_cylinder_height = cylinder_height * cylinder_scale[2]
                    cylinder_z = 0.64 + scaled_cylinder_height/2.0

                    cylinder_number = cylinder_index + 1
                    cylinder_name = f"Cylinder_{cylinder_number}"
                    prim_path = f"/World/Cylinder_{cylinder_number}"
                    timestamp = int(time.time() * 1000) + cylinder_index

                    cylinder = self.world.scene.add(
                        DynamicCylinder(
                            name=f"cylinder_{timestamp}",
                            position=np.array([cylinder_x, cylinder_y, cylinder_z]),
                            prim_path=prim_path,
                            radius=cylinder_radius,
                            height=cylinder_height,
                            color=np.array([0.0, 0.0, 0.0]),
                            scale=cylinder_scale
                        )
                    )

                    prim = stage.GetPrimAtPath(prim_path)
                    if prim:
                        display_name = f"Cylinder {cylinder_number} (t{timestamp})"
                        omni.usd.editor.set_display_name(prim, display_name)

                    self.cubes.append((cylinder, cylinder_name))
                    cylinder_index += 1

            await omni.kit.app.get_app().next_update_async()

            shader_count = 0

            looks_prim = stage.GetPrimAtPath("/World/Looks")
            if looks_prim.IsValid():
                for child_prim in looks_prim.GetChildren():
                    material_name = child_prim.GetName()
                    if "visual" in material_name.lower() and "material" in material_name.lower():
                        for shader_name in ["shader", "Shader"]:
                            shader_prim_path = f"/World/Looks/{material_name}/{shader_name}"
                            shader_prim = stage.GetPrimAtPath(shader_prim_path)
                            if shader_prim.IsValid():
                                shader = UsdShade.Shader(shader_prim)
                                emissive_color = Gf.Vec3f(0.0, 0.0, 0.0)
                                emissive_input = shader.GetInput("emissiveColor")
                                if not emissive_input:
                                    emissive_input = shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
                                emissive_input.Set(emissive_color)
                                metallic_input = shader.GetInput("metallic")
                                if not metallic_input:
                                    metallic_input = shader.CreateInput("metallic", Sdf.ValueTypeNames.Float)
                                metallic_input.Set(0.6)
                                shader_count += 1
                                break

            await omni.kit.app.get_app().next_update_async()


            self.gripper = ParallelGripper(
                end_effector_prim_path=self._end_effector_prim_path,
                joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
                joint_opened_positions=np.array([0.05, 0.05]),
                joint_closed_positions=np.array([0.0, 0.0]),
                action_deltas=np.array([0.01, 0.01])
            )

            self.franka = self.world.scene.add(
                SingleManipulator(
                    prim_path=self._franka_prim_path,
                    name=self._franka_name,
                    end_effector_prim_path=self._end_effector_prim_path,
                    gripper=self.gripper,
                    position=np.array([-0.1, 0.0, 0.6]),
                    orientation=np.array([1.0, 0.0, 0.0, 0.0])
                )
            )

            try:
                grasp_file = project_root / "src" / "grippers" / "franka_cylinder_grasp.yaml"
                self.grasp_config = GraspConfig(str(grasp_file))

                if not self.grasp_config.is_loaded():
                    self.grasp_config = None
            except Exception as e:
                self.grasp_config = None

            self._add_initial_obstacles()

            self.world.initialize_physics()
            self.world.reset()

            if hasattr(self, '_initial_obstacle_positions'):
                for obstacle_name, position in self._initial_obstacle_positions.items():
                    if obstacle_name in self.obstacles:
                        self.obstacles[obstacle_name].set_world_pose(position=position)

            self.lidar.add_depth_data_to_frame()
            self.lidar.add_point_cloud_data_to_frame()
            self.lidar.enable_visualization()

            settings = carb.settings.get_settings()

            settings.set("/rtx/post/dlss/execMode", 0)
            settings.set("/rtx/post/aa/op", 0)

            settings.set("/exts/omni.usd.schema.render_settings/rtx/renderSettings/apiSchemas/autoApply", None)
            settings.set("/exts/omni.usd.schema.render_settings/rtx/camera/apiSchemas/autoApply", None)
            settings.set("/exts/omni.usd.schema.render_settings/rtx/renderProduct/apiSchemas/autoApply", None)



            # NOTE: Initialize with attach_rgb_annotator=False for better performance
            self.depth_camera.initialize(attach_rgb_annotator=False)


            self.depth_camera.set_focal_length(1.3)
            self.depth_camera.set_focus_distance(400.0)

            self.depth_camera.set_baseline_mm(55.0)
            self.depth_camera.set_focal_length_pixel(256.0)
            self.depth_camera.set_sensor_size_pixel(1280.0)
            self.depth_camera.set_max_disparity_pixel(110.0)
            self.depth_camera.set_confidence_threshold(0.99)
            self.depth_camera.set_noise_mean(0.1)
            self.depth_camera.set_noise_sigma(0.5)
            self.depth_camera.set_noise_downscale_factor_pixel(1.0)
            self.depth_camera.set_min_distance(0.1)
            self.depth_camera.set_max_distance(2.0)



            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            self.depth_camera.attach_annotator("DepthSensorDistance")

            self.depth_camera.attach_annotator("DepthSensorPointCloudPosition")

            self.depth_camera.attach_annotator("DepthSensorPointCloudColor")





            self.franka.disable_gravity()
            articulation_controller = self.franka.get_articulation_controller()
            kp_gains = np.array([1e15, 1e15, 1e15, 1e15, 1e15, 1e15, 1e15, 2e16, 2e16])
            kd_gains = np.array([1e13, 1e13, 1e13, 1e13, 1e13, 1e13, 1e13, 2e14, 2e14])
            articulation_controller.set_gains(kp_gains, kd_gains)

            default_joint_positions = np.array([0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75, 0.0, 0.0])
            self.franka.set_joints_default_state(positions=default_joint_positions)
            self.gripper.set_default_state(self.gripper.joint_closed_positions)

            articulation_controller = self.franka.get_articulation_controller()
            close_action = ArticulationAction(
                joint_positions=self.gripper.joint_closed_positions,
                joint_indices=np.array([7, 8])
            )
            articulation_controller.apply_action(close_action)

            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            self._setup_rrt()

            self._add_container_as_obstacle()
            self._add_initial_obstacles_to_rrt()

            if self.use_rl:
                self._load_rl_model()

            self.pick_btn.enabled = True
            self.reset_btn.enabled = True
            self.add_obstacle_btn.enabled = True
            self.remove_obstacle_btn.enabled = True

            status_msg = "Scene loaded - Ready to pick and place"
            if self.use_rl and self.rl_model:
                status_msg += " (RL mode)"
            elif self.use_rl:
                status_msg += " (Greedy mode - RL failed)"
            else:
                status_msg += " (Greedy mode)"
            self._update_status(status_msg)

        except Exception as e:
            self._update_status(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def _setup_rrt(self):
        """Setup RRT motion planner and kinematics solvers"""
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")

        robot_description_file = str(project_root / "assets" / "franka_conservative_spheres_robot_description.yaml")

        urdf_path = os.path.join(mg_extension_path, "motion_policy_configs", "franka", "lula_franka_gen.urdf")
        rrt_config_file = os.path.join(mg_extension_path, "path_planner_configs", "franka", "rrt", "franka_planner_config.yaml")

        if not os.path.exists(robot_description_file):
            raise FileNotFoundError(f"Robot description not found: {robot_description_file}")
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        if not os.path.exists(rrt_config_file):
            raise FileNotFoundError(f"RRT config not found: {rrt_config_file}")

        self.rrt = RRT(robot_description_path=robot_description_file, urdf_path=urdf_path,
                       rrt_config_path=rrt_config_file, end_effector_frame_name="right_gripper")
        self.rrt.set_max_iterations(10000)

        self.path_planner_visualizer = PathPlannerVisualizer(robot_articulation=self.franka, path_planner=self.rrt)
        self.kinematics_solver = LulaKinematicsSolver(robot_description_path=robot_description_file, urdf_path=urdf_path)
        self.articulation_kinematics_solver = ArticulationKinematicsSolver(
            self.franka, self.kinematics_solver, "right_gripper")

        self.cspace_trajectory_generator = LulaCSpaceTrajectoryGenerator(robot_description_file, urdf_path)

    def _add_container_as_obstacle(self):
        """Add container as obstacle for RRT collision avoidance"""
        try:
            if self.container is None:
                return

            self.rrt.add_obstacle(self.container, static=True)

        except Exception as e:
            carb.log_error(f"Failed to add container as obstacle: {e}")

    def _add_other_cubes_as_obstacles(self, current_cube_index, target_position):
        """
        Add ONLY nearby cubes as temporary obstacles (within potential collision zone).
        This prevents the robot from colliding with other cubes during motion,
        while avoiding excessive obstacles that cause RRT failures.

        Args:
            current_cube_index: Index of the cube being picked (this cube will NOT be added as obstacle)
            target_position: Position of the target cube (to determine which cubes are nearby)

        Returns:
            int: Number of nearby cubes added as obstacles
        """
        if self.cube_obstacles_enabled:
            return 0

        try:
            added_count = 0
            collision_radius = 0.15

            for i, (cube, cube_name) in enumerate(self.cubes):
                if i == current_cube_index:
                    continue

                if not cube.prim or not cube.prim.IsValid():
                    continue

                cube_pos, _ = cube.get_world_pose()

                distance = np.linalg.norm(cube_pos[:2] - target_position[:2])

                if distance < collision_radius:
                    self.rrt.add_obstacle(cube, static=False)
                    added_count += 1

            self.cube_obstacles_enabled = True
            return added_count

        except Exception as e:
            carb.log_error(f"Failed to add cylinders as obstacles: {e}")
            return 0

    def _remove_all_cube_obstacles(self):
        """
        Remove all cubes from RRT obstacle list.
        Called after pick/place operations are complete.
        """
        if not self.cube_obstacles_enabled:
            return

        try:
            for cube, _ in self.cubes:
                try:
                    self.rrt.remove_obstacle(cube)
                except Exception:
                    pass

            self.cube_obstacles_enabled = False

        except Exception as e:
            carb.log_error(f"Failed to remove cube obstacles: {e}")

    def _process_lidar_data(self):
        """
        Process PhysX Lidar point cloud data to detect obstacles in real-time.
        Returns list of detected obstacle positions in world coordinates.
        """
        if self.lidar is None:
            return []

        try:
            lidar_data = self.lidar.get_current_frame()

            if lidar_data is None or "point_cloud" not in lidar_data:
                return []

            point_cloud_data = lidar_data["point_cloud"]

            if point_cloud_data is None:
                return []

            if hasattr(point_cloud_data, 'cpu'):
                points = point_cloud_data.cpu().numpy()
            elif hasattr(point_cloud_data, 'numpy'):
                points = point_cloud_data.numpy()
            elif isinstance(point_cloud_data, np.ndarray):
                points = point_cloud_data
            else:
                return []

            if len(points.shape) == 3 and points.shape[1] == 1:
                points = points.reshape(-1, 3)

            if len(points) == 0:
                return []

            if points.ndim != 2 or points.shape[1] != 3:
                return []

            lidar_world_pos, lidar_world_rot = self.lidar.get_world_pose()
            from scipy.spatial.transform import Rotation as R
            rot_matrix = R.from_quat([lidar_world_rot[1], lidar_world_rot[2], lidar_world_rot[3], lidar_world_rot[0]]).as_matrix()

            points_world = (rot_matrix @ points.T).T + lidar_world_pos

            valid_points = points_world[(points_world[:, 2] > 0.59) & (points_world[:, 2] < 0.80)]

            robot_pos, _ = self.franka.get_world_pose()
            distances_from_robot = np.linalg.norm(valid_points[:, :2] - robot_pos[:2], axis=1)
            valid_points = valid_points[(distances_from_robot > 0.30) & (distances_from_robot < 0.90)]

            cube_grid_center = np.array([0.40, -0.08])
            cube_grid_margin = 0.28
            cube_region_mask = ~((np.abs(valid_points[:, 0] - cube_grid_center[0]) < cube_grid_margin) &
                                 (np.abs(valid_points[:, 1] - cube_grid_center[1]) < cube_grid_margin))
            valid_points = valid_points[cube_region_mask]

            if self.container_dimensions is not None:
                container_pos = np.array([0.35, 0.5])
                container_margin = 0.08
                container_half_dims = self.container_dimensions / 2.0
                container_region_mask = ~((np.abs(valid_points[:, 0] - container_pos[0]) < (container_half_dims[0] + container_margin)) &
                                          (np.abs(valid_points[:, 1] - container_pos[1]) < (container_half_dims[1] + container_margin)))
                valid_points = valid_points[container_region_mask]

            robot_base_pos = np.array([0.0, 0.0])
            robot_arm_radius = 0.55
            robot_region_mask = np.linalg.norm(valid_points[:, :2] - robot_base_pos, axis=1) > robot_arm_radius
            valid_points = valid_points[robot_region_mask]

            detected_obstacles = []

            if len(valid_points) > 10:
                grid_size = 0.1
                grid_points = np.round(valid_points / grid_size) * grid_size
                unique_cells, counts = np.unique(grid_points, axis=0, return_counts=True)
                obstacle_cells = unique_cells[counts > 5]
                detected_obstacles = obstacle_cells.tolist()

                if len(detected_obstacles) > 1:
                    merged_obstacles = []
                    used = set()
                    for i, obs1 in enumerate(detected_obstacles):
                        if i in used:
                            continue
                        cluster = [obs1]
                        for j, obs2 in enumerate(detected_obstacles[i+1:], start=i+1):
                            if j in used:
                                continue
                            dist_xy = np.linalg.norm(np.array(obs1[:2]) - np.array(obs2[:2]))
                            if dist_xy < 0.25:
                                cluster.append(obs2)
                                used.add(j)
                        cluster_array = np.array(cluster)
                        merged_pos = [
                            np.mean(cluster_array[:, 0]),
                            np.mean(cluster_array[:, 1]),
                            np.min(cluster_array[:, 2])
                        ]
                        merged_obstacles.append(merged_pos)
                    detected_obstacles = merged_obstacles



            return detected_obstacles

        except Exception as e:
            carb.log_warn(f"[LIDAR ERROR] Error processing Lidar data: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _process_depth_camera_data(self):
        """
        Process depth camera data to detect obstacles in real-time.
        Returns list of detected obstacle positions in world coordinates.
        """
        if self.depth_camera is None:
            return []

        try:
            depth_frame = self.depth_camera.get_current_frame()

            if depth_frame is None:
                return []

            depth_data = None
            if "data" in depth_frame:
                depth_data = depth_frame["data"]
            elif "distance" in depth_frame:
                depth_data = depth_frame["distance"]
            elif "depth" in depth_frame:
                depth_data = depth_frame["depth"]

            if depth_data is None or len(depth_data) == 0:
                return []

            height, width = depth_data.shape

            fx = fy = 256.0
            cx, cy = width / 2.0, height / 2.0

            detected_obstacles = []
            point_count = 0

            for v in range(0, height, 20):
                for u in range(0, width, 20):
                    depth = depth_data[v, u]

                    if depth < 0.1 or depth > 1.5:
                        continue

                    point_count += 1

                    x_cam = (u - cx) * depth / fx
                    y_cam = (v - cy) * depth / fy
                    z_cam = depth

                    camera_pos, camera_quat = self.depth_camera.get_world_pose()

                    point_cam = np.array([x_cam, y_cam, z_cam])

                    from isaacsim.core.utils.rotations import quat_to_rot_matrix
                    rot_matrix = quat_to_rot_matrix(camera_quat)
                    point_world = rot_matrix @ point_cam + camera_pos

                    if point_world[2] < 0.05 or point_world[2] > 0.5:
                        continue

                    detected_obstacles.append(point_world.tolist())

            if len(detected_obstacles) > 0:
                merged_obstacles = []
                used = set()

                for i, obs1 in enumerate(detected_obstacles):
                    if i in used:
                        continue
                    used.add(i)
                    cluster = [obs1]
                    for j, obs2 in enumerate(detected_obstacles[i+1:], start=i+1):
                        if j in used:
                            continue
                        dist = np.linalg.norm(np.array(obs1) - np.array(obs2))
                        if dist < 0.10:
                            cluster.append(obs2)
                            used.add(j)

                    cluster_array = np.array(cluster)
                    merged_pos = np.mean(cluster_array, axis=0).tolist()
                    merged_obstacles.append(merged_pos)

                detected_obstacles = merged_obstacles

            return detected_obstacles

        except Exception as e:
            carb.log_warn(f"Error processing depth camera data: {e}")
            return []

    def _clear_lidar_obstacles(self):
        """
        Temporarily clear all Lidar-detected obstacles from RRT.
        Used during retreat planning to avoid detecting target cube as obstacle.
        """
        if self.rrt is None:
            return

        try:
            for obs_name, obs_obj in list(self.lidar_detected_obstacles.items()):
                try:
                    self.rrt.remove_obstacle(obs_obj)
                except Exception:
                    pass
        except Exception as e:
            carb.log_error(f"Failed to clear Lidar obstacles: {e}")

    def _update_dynamic_obstacles(self):
        """
        Update RRT planner with dynamically detected obstacles from Lidar and Depth Camera.
        Also ensures RRT knows current Obstacle_1 position.

        OPTIMIZED: Instead of deleting and recreating obstacles, we:
        1. Reuse existing obstacle prims by moving them
        2. Only create new prims if we need more
        3. Only delete prims if we have too many
        This avoids constant stage updates and maintains 60 FPS
        """
        if self.lidar is None or self.rrt is None:
            return

        try:
            if self.obstacle_1_moving and "obstacle_1" in self.obstacles:
                obs_pos, _ = self.obstacles["obstacle_1"].get_world_pose()
                self.rrt.update_world()

            detected_positions = self._process_lidar_data()

            # NOTE: Disabled for performance - enable when needed for obstacle detection

            detected_positions = detected_positions[:10]

            num_detected = len(detected_positions)
            num_current = len(self.lidar_detected_obstacles)

            existing_obstacles = list(self.lidar_detected_obstacles.items())
            for i in range(min(num_detected, num_current)):
                obs_name, obs_obj = existing_obstacles[i]
                new_pos = np.array(detected_positions[i])

                current_pos, _ = obs_obj.get_world_pose()
                if np.linalg.norm(new_pos - current_pos) > 0.05:
                    obs_obj.set_world_pose(position=new_pos)
                    self.rrt.update_world()

            if num_detected > num_current:
                for i in range(num_current, num_detected):
                    obs_name = f"lidar_obstacle_{i}"
                    obs_prim_path = f"/World/LidarObstacle_{i}"

                    obstacle = self.world.scene.add(
                        FixedCuboid(
                            name=obs_name,
                            prim_path=obs_prim_path,
                            position=np.array(detected_positions[i]),
                            size=1.0,
                            scale=np.array([0.15, 0.15, 0.15]),
                            color=np.array([1.0, 0.0, 0.0]),
                            visible=False
                        )
                    )
                    obstacle.set_visibility(False)
                    self.rrt.add_obstacle(obstacle, static=False)
                    self.lidar_detected_obstacles[obs_name] = obstacle

            elif num_detected < num_current:
                for i in range(num_detected, num_current):
                    obs_name = f"lidar_obstacle_{i}"
                    if obs_name in self.lidar_detected_obstacles:
                        obs_obj = self.lidar_detected_obstacles[obs_name]
                        try:
                            self.rrt.remove_obstacle(obs_obj)
                            self.world.scene.remove_object(obs_name)
                        except:
                            pass
                        del self.lidar_detected_obstacles[obs_name]

        except Exception as e:
            carb.log_warn(f"[RRT ERROR] Error updating dynamic obstacles: {e}")
            import traceback
            traceback.print_exc()

    def _physics_step_callback(self, step_size):
        """
        Physics step callback for continuous sensor updates and obstacle movement.
        Called every physics step (60 Hz).
        """

        if not hasattr(self, '_sensor_log_counter'):
            self._sensor_log_counter = 0

        self._sensor_log_counter += 1
        if self._sensor_log_counter >= 30:
            self._sensor_log_counter = 0

            if self.lidar is not None:
                self._process_lidar_data()

            if self.depth_camera is not None:
                self._process_depth_camera_data()

    def _move_obstacle(self):
        """
        Move Obstacle_1 automatically using PhysX Force API (acceleration mode).
        Applies continuous acceleration to move obstacle back and forth between min_x and max_x.

        Logic:
        - Starts at x=0.63m (right boundary)
        - When x <= 0.2m (left boundary): acceleration = +6 m/s² (move right)
        - When x >= 0.63m (right boundary): acceleration = -6 m/s² (move left)
        """
        if "obstacle_1" not in self.obstacles:
            return

        try:
            obstacle = self.obstacles["obstacle_1"]
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(obstacle.prim_path)

            if not prim or not prim.IsValid():
                return

            if not self.obstacle_1_force_api_applied:
                if not prim.HasAPI(PhysxSchema.PhysxForceAPI):
                    PhysxSchema.PhysxForceAPI.Apply(prim)

                force_api = PhysxSchema.PhysxForceAPI(prim)

                force_api.CreateForceEnabledAttr().Set(True)
                force_api.CreateWorldFrameEnabledAttr().Set(True)
                force_api.CreateModeAttr().Set("acceleration")

                self.obstacle_1_force_api_applied = True

            current_pos, _ = obstacle.get_world_pose()

            if current_pos[0] <= self.obstacle_1_min_x:
                acceleration = self.obstacle_1_acceleration
                if not hasattr(self, '_last_accel') or self._last_accel != acceleration:
                    self._last_accel = acceleration
            elif current_pos[0] >= self.obstacle_1_max_x:
                acceleration = -self.obstacle_1_acceleration
                if not hasattr(self, '_last_accel') or self._last_accel != acceleration:
                    self._last_accel = acceleration
            else:
                if not hasattr(self, '_last_accel'):
                    acceleration = -self.obstacle_1_acceleration
                    self._last_accel = acceleration
                else:
                    acceleration = self._last_accel

            force_api = PhysxSchema.PhysxForceAPI(prim)
            acceleration_vector = Gf.Vec3f(
                acceleration,
                0.0,
                0.0
            )
            force_api.GetForceAttr().Set(acceleration_vector)

            if self.rrt is not None:
                self.rrt.update_world()

        except Exception as e:
            carb.log_warn(f"[OBSTACLE] Error moving Obstacle_1: {e}")
            import traceback
            traceback.print_exc()

    def _load_rl_model(self):
        """Load trained RL model for object selection (supports both PPO and DDQN)"""
        if not self.use_rl or not self.rl_model_path:
            return

        try:
            if self.rl_model_type == 'ppo':
                metadata_path = self.rl_model_path.replace("_final.zip", "_metadata.json")
            else:
                metadata_path = self.rl_model_path.replace(".pt", "_metadata.json")

            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    model_grid_size = metadata.get("training_grid_size", self.training_grid_size)

                    if model_grid_size != self.training_grid_size:
                        self.training_grid_size = model_grid_size

            max_objects = self.training_grid_size * self.training_grid_size
            dummy_env = ObjectSelectionEnv(
                franka_controller=self,
                max_objects=max_objects,
                max_steps=50,
                training_grid_size=self.training_grid_size
            )

            if self.rl_model_type == 'ppo':
                def mask_fn(env):
                    return env.action_masks()
                dummy_env = ActionMasker(dummy_env, mask_fn)
                vec_env = DummyVecEnv([lambda: dummy_env])

                vecnorm_path = self.rl_model_path.replace("_final.zip", "_vecnormalize.pkl")
                if Path(vecnorm_path).exists():
                    vec_env = VecNormalize.load(vecnorm_path, vec_env)
                    vec_env.training = False
                    vec_env.norm_reward = False

                self.rl_model = MaskablePPO.load(self.rl_model_path, env=vec_env)

            else:
                checkpoint = torch.load(self.rl_model_path, map_location='cpu', weights_only=False)

                self.rl_model = DoubleDQNAgent(
                    state_dim=checkpoint['state_dim'],
                    action_dim=checkpoint['action_dim'],
                    gamma=checkpoint['gamma'],
                    epsilon_start=checkpoint['epsilon'],
                    epsilon_end=checkpoint['epsilon_end'],
                    epsilon_decay=checkpoint['epsilon_decay'],
                    batch_size=checkpoint['batch_size'],
                    target_update_freq=checkpoint['target_update_freq']
                )

                self.rl_model.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.rl_model.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                self.rl_model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.rl_model.epsilon = checkpoint['epsilon']
                self.rl_model.steps = checkpoint['steps']
                self.rl_model.episodes = checkpoint['episodes']

                self.rl_model.policy_net.eval()
                self.rl_model.target_net.eval()

        except Exception as e:
            carb.log_error(f"Error loading RL model: {e}")
            self.rl_model = None
            self.use_rl = False

    def _get_observation(self, picked_indices=None):
        """
        Get observation for RL model (same format as training environment).

        UPDATED: 6 values per object (added picked flag)
        1. Distance to robot EE: 1 value
        2. Distance to container: 1 value
        3. Obstacle proximity score: 1 value
        4. Reachability flag: 1 value
        5. Path clearance score: 1 value
        6. Picked flag: 1 value (0.0 = available, 1.0 = already picked)

        Args:
            picked_indices: List of already picked object indices

        Returns:
            Observation array (flattened, shape: (max_objects * 6,))
        """
        if picked_indices is None:
            picked_indices = []

        max_objects = self.training_grid_size * self.training_grid_size
        obs = np.zeros((max_objects, 6), dtype=np.float32)

        ee_pos, _ = self.franka.end_effector.get_world_pose()
        container_pos, _ = self.container.get_world_pose()

        obstacle_positions = []
        if hasattr(self, 'obstacles') and self.obstacles:
            for obs_name, obs_obj in self.obstacles.items():
                try:
                    obs_pos, _ = obs_obj.get_world_pose()
                    obstacle_positions.append(obs_pos)
                except Exception:
                    pass

        if hasattr(self, 'lidar_detected_obstacles') and self.lidar_detected_obstacles:
            for lidar_obs_name, lidar_obs_obj in self.lidar_detected_obstacles.items():
                try:
                    lidar_obs_pos, _ = lidar_obs_obj.get_world_pose()
                    obstacle_positions.append(lidar_obs_pos)
                except Exception:
                    pass

        for i, (cube, _) in enumerate(self.cubes):
            if not cube.prim or not cube.prim.IsValid():
                obs[i, 0] = 0.0
                obs[i, 1] = 0.0
                obs[i, 2] = 0.0
                obs[i, 3] = 0.0
                obs[i, 4] = 0.0
                obs[i, 5] = 1.0
                continue

            cube_pos, _ = cube.get_world_pose()

            dist_to_ee = np.linalg.norm(cube_pos - ee_pos)
            obs[i, 0] = dist_to_ee

            obs[i, 1] = np.linalg.norm(cube_pos - container_pos)

            min_obstacle_dist = float('inf')

            if obstacle_positions:
                for obs_pos in obstacle_positions:
                    dist = np.linalg.norm(cube_pos[:2] - obs_pos[:2])
                    min_obstacle_dist = min(min_obstacle_dist, dist)

            for j, (other_cube, _) in enumerate(self.cubes):
                if j == i or j in picked_indices:
                    continue

                if not other_cube.prim or not other_cube.prim.IsValid():
                    continue

                try:
                    other_pos, _ = other_cube.get_world_pose()
                    dist = np.linalg.norm(cube_pos[:2] - other_pos[:2])
                    min_obstacle_dist = min(min_obstacle_dist, dist)
                except Exception:
                    pass

            if min_obstacle_dist < 0.10:
                obs[i, 2] = 1.0
            elif min_obstacle_dist > 0.30:
                obs[i, 2] = 0.0
            else:
                obs[i, 2] = 1.0 - (min_obstacle_dist - 0.10) / 0.20

            reachable = 1.0 if (0.3 <= dist_to_ee <= 0.9) else 0.0
            obs[i, 3] = reachable

            path_clear = 1.0

            collision_radius = 0.15
            for j, (other_cube, _) in enumerate(self.cubes):
                if j == i or j in picked_indices:
                    continue

                if not other_cube.prim or not other_cube.prim.IsValid():
                    continue

                try:
                    other_pos, _ = other_cube.get_world_pose()
                    dist_to_other = np.linalg.norm(cube_pos - other_pos)

                    if dist_to_other < collision_radius:
                        path_clear = 0.0
                        break
                except Exception:
                    pass

            if obstacle_positions and path_clear > 0.0:
                for obs_pos in obstacle_positions:
                    dist_to_obs = np.linalg.norm(cube_pos - obs_pos)
                    if dist_to_obs < 0.20:
                        path_clear = 0.5
                        break

            obs[i, 4] = path_clear

            picked_flag = 1.0 if i in picked_indices else 0.0
            obs[i, 5] = picked_flag

        return obs.flatten()

    def _get_rl_pick_order(self):
        """
        Get optimal picking order using RL model or greedy baseline.

        Returns:
            List of cube indices in optimal picking order
        """
        if not self.cubes:
            return []

        ee_pos, _ = self.franka.end_effector.get_world_pose()

        distances = []
        for i, (cube, cube_name) in enumerate(self.cubes):
            cube_pos, _ = cube.get_world_pose()
            dist = np.linalg.norm(cube_pos - ee_pos)
            distances.append((i, dist, cube_name))

        if self.use_rl and self.rl_model is not None:
            try:
                pick_order = []
                picked_indices = []
                total_objects = len(self.cubes)

                for step in range(total_objects):
                    obs = self._get_observation(picked_indices=picked_indices)

                    action_mask = np.zeros(self.training_grid_size * self.training_grid_size, dtype=bool)
                    for idx in range(total_objects):
                        if idx not in picked_indices:
                            action_mask[idx] = True

                    if self.rl_model_type == 'ppo':
                        action, _ = self.rl_model.predict(obs, action_masks=action_mask, deterministic=True)
                        action = int(action)
                    else:
                        obs_flat = obs.flatten()
                        obs_tensor = torch.FloatTensor(obs_flat).to(self.rl_model.device)
                        action = self.rl_model.policy_net.get_action(obs_tensor, epsilon=0.0, action_mask=action_mask)

                    if action in picked_indices:
                        unpicked = [idx for idx, _, _ in distances if idx not in picked_indices]
                        if unpicked:
                            action = min(unpicked, key=lambda idx: distances[idx][1])

                    pick_order.append(action)
                    picked_indices.append(action)

                return pick_order

            except Exception as e:
                carb.log_error(f"Error using RL model: {e}")
                pick_order = [idx for idx, _, _ in sorted(distances, key=lambda x: x[1])]
        else:
            pick_order = [idx for idx, _, _ in sorted(distances, key=lambda x: x[1])]

        return pick_order

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

            if self.current_cube_index == 0:
                self.placed_count = 0
                self._update_status("Starting...")

            run_coroutine(self._pick_place_loop())

    async def _pick_place_loop(self):
        """Main pick and place loop"""
        try:
            self.timeline.play()

            if not hasattr(self, '_physics_callback_added'):
                try:
                    self.world.add_physics_callback("sensor_and_obstacle_update", self._physics_step_callback)
                    self._physics_callback_added = True
                except Exception as e:
                    carb.log_warn(f"Could not add physics callback: {e}")
                    self._physics_callback_added = False


            for _ in range(25):
                await omni.kit.app.get_app().next_update_async()

            articulation_controller = self.franka.get_articulation_controller()
            articulation_controller.apply_action(ArticulationAction(
                joint_positions=self.gripper.joint_closed_positions, joint_indices=np.array([7, 8])))
            for _ in range(8):
                await omni.kit.app.get_app().next_update_async()

            default_joint_positions = np.array([0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75, 0.0, 0.0])
            articulation_controller.apply_action(ArticulationAction(joint_positions=default_joint_positions))
            for _ in range(8):
                await omni.kit.app.get_app().next_update_async()

            cubes = self.cubes
            total_cubes = len(cubes)

            if self.current_cube_index == 0:
                pick_order = self._get_rl_pick_order()
                self.pick_order = pick_order
            else:
                pick_order = self.pick_order if hasattr(self, 'pick_order') else list(range(total_cubes))

            for order_idx in range(self.current_cube_index, total_cubes):
                try:
                    cube_idx = pick_order[order_idx]

                    if cube_idx < 0 or cube_idx >= total_cubes:
                        carb.log_error(f"Invalid cube index {cube_idx} (valid range: 0-{total_cubes-1})")
                        self.current_cube_index += 1
                        continue

                    cube, cube_name = cubes[cube_idx]
                    self.current_cube_index = cube_idx

                    success, error_msg = await self._pick_and_place_cube(cube, cube_name.split('_')[1])

                    if success:
                        self.placed_count += 1
                    self._update_status(f"{self.placed_count}/{total_cubes} placed")

                    self.current_cube_index += 1

                except Exception as cube_error:
                    carb.log_error(f"Error with cube: {str(cube_error)}")
                    self._update_status(f"{self.placed_count}/{total_cubes} placed")
                    self.current_cube_index += 1
                    continue

            self._update_status(f"Done: {self.placed_count}/{total_cubes} placed")
            self.is_picking = False
            self.current_cube_index = 0

            self.obstacle_1_moving = False

            selection = omni.usd.get_context().get_selection()
            selection.clear_selected_prim_paths()

            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()

            self.timeline.stop()

        except Exception as e:
            self._update_status(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.is_picking = False

            self.obstacle_1_moving = False

            self.timeline.stop()

    async def _pick_and_place_cube(self, cube, cube_name):
        """Pick and place cylinder using RRT (8 phases: pick with retry, place, return home)"""
        try:
            total_cubes = self.num_cubes
            cylinder_base_height = 0.0515
            cylinder_scale_z = 1.4
            cylinder_height = cylinder_base_height * cylinder_scale_z
            cylinder_half = cylinder_height / 2.0

            safe_height = 0.94

            orientation = euler_angles_to_quats(np.array([np.pi, 0, 0]))
            max_pick_attempts = 3
            pick_success = False

            if cube is None or not hasattr(cube, 'get_world_pose'):
                raise ValueError(f"Invalid cube object for {cube_name}")

            if not cube.prim or not cube.prim.IsValid():
                raise ValueError(f"Cube prim is invalid or already deleted for {cube_name}")

            cube_pos_initial, _ = cube.get_world_pose()

            nearby_obstacle_count = self._add_other_cubes_as_obstacles(self.current_cube_index, cube_pos_initial)

            await self._reset_to_safe_config()

            for pick_attempt in range(1, max_pick_attempts + 1):
                if pick_attempt > 1:
                    await self._reset_to_safe_config()

                cube_pos_current, _ = cube.get_world_pose()

                articulation_controller = self.franka.get_articulation_controller()
                articulation_controller.apply_action(ArticulationAction(
                    joint_positions=self.gripper.joint_opened_positions, joint_indices=np.array([7, 8])))
                for _ in range(5):
                    await omni.kit.app.get_app().next_update_async()

                high_waypoint = np.array([
                    cube_pos_current[0] + 0.003,
                    cube_pos_current[1] + 0.003,
                    safe_height
                ])
                success = await self._move_to_target_rrt(high_waypoint, orientation, skip_factor=4)

                if not success:
                    self._remove_all_cube_obstacles()
                    await self._reset_to_safe_config()
                    return False, f"RRT failed to reach above {cube_name}"

                for _ in range(3):
                    await omni.kit.app.get_app().next_update_async()

                cube_pos_realtime, _ = cube.get_world_pose()
                pick_pos = np.array([
                    cube_pos_realtime[0] + 0.004,
                    cube_pos_realtime[1] + 0.004,
                    cube_pos_realtime[2]
                ])

                success = await self._move_to_target_rrt(pick_pos, orientation, skip_factor=2)
                if not success:
                    self._remove_all_cube_obstacles()
                    await self._reset_to_safe_config()
                    return False, f"RRT failed pick approach for {cube_name}"

                for _ in range(3):
                    await omni.kit.app.get_app().next_update_async()

                retreat_pos = np.array([
                    cube_pos_realtime[0] + 0.004,
                    cube_pos_realtime[1] + 0.004,
                    safe_height
                ])

                self._clear_lidar_obstacles()

                retreat_plan = self._plan_to_target(retreat_pos, orientation, update_obstacles=False)

                if retreat_plan is None:
                    self._remove_all_cube_obstacles()
                    await self._reset_to_safe_config()
                    return False, f"RRT cannot plan retreat for {cube_name}"

                if self.grasp_config and self.grasp_config.is_loaded():
                    grasp_positions = self.grasp_config.get_gripper_joint_positions(self.current_grasp_name)
                    if grasp_positions:
                        joint_positions = np.array([
                            grasp_positions.get("panda_finger_joint1", 0.037),
                            grasp_positions.get("panda_finger_joint2", 0.037)
                        ])
                    else:
                        joint_positions = self.gripper.joint_closed_positions
                else:
                    joint_positions = self.gripper.joint_closed_positions

                articulation_controller.apply_action(ArticulationAction(
                    joint_positions=joint_positions, joint_indices=np.array([7, 8])))
                for _ in range(20):
                    await omni.kit.app.get_app().next_update_async()

                for _ in range(15):
                    await omni.kit.app.get_app().next_update_async()

                success = await self._move_to_target_rrt(retreat_pos, orientation, skip_factor=5)
                if not success:
                    self.franka.gripper.open()
                    for _ in range(5):
                        await omni.kit.app.get_app().next_update_async()
                    self._remove_all_cube_obstacles()
                    await self._reset_to_safe_config()
                    return False, f"RRT failed pick retreat for {cube_name}"

                cube_pos_after_pick, _ = cube.get_world_pose()
                height_lifted = cube_pos_after_pick[2] - cube_pos_realtime[2]

                if height_lifted > 0.05:
                    pick_success = True
                    break
                else:
                    if pick_attempt < max_pick_attempts:
                        articulation_controller.apply_action(ArticulationAction(
                            joint_positions=self.gripper.joint_opened_positions, joint_indices=np.array([7, 8])))
                        for _ in range(3):
                            await omni.kit.app.get_app().next_update_async()
                        await self._reset_to_safe_config()
                        continue
                    else:
                        self._remove_all_cube_obstacles()
                        await self._reset_to_safe_config()
                        return False, f"Failed to pick {cube_name}"

            if not pick_success:
                self._remove_all_cube_obstacles()
                await self._reset_to_safe_config()
                return False, f"Failed to pick {cube_name}"

            self._remove_all_cube_obstacles()

            container_center = np.array([0.30, 0.50, 0.0])
            container_length = self.container_dimensions[0]
            container_width = self.container_dimensions[1]

            place_grid_size = int(np.ceil(np.sqrt(total_cubes)))
            place_row = self.placed_count // place_grid_size
            place_col = self.placed_count % place_grid_size

            if total_cubes <= 4:
                edge_margin_left = 0.11
                edge_margin_right = 0.11
                edge_margin_width = 0.10
            elif total_cubes <= 9:
                edge_margin_left = 0.11
                edge_margin_right = 0.11
                edge_margin_width = 0.10
            else:
                edge_margin_left = 0.09
                edge_margin_right = 0.09
                edge_margin_width = 0.07

            usable_length = container_length - edge_margin_left - edge_margin_right
            usable_width = container_width - (2 * edge_margin_width)
            spacing_length = usable_length / (place_grid_size - 1) if place_grid_size > 1 else 0.0
            spacing_width = usable_width / (place_grid_size - 1) if place_grid_size > 1 else 0.0

            start_x = container_center[0] - (container_length / 2.0) + edge_margin_left
            start_y = container_center[1] - (container_width / 2.0) + edge_margin_width
            cube_x = start_x + (place_row * spacing_length)
            cube_y = start_y + (place_col * spacing_width)

            container_x_min = container_center[0] - container_length / 2.0
            container_x_max = container_center[0] + container_length / 2.0
            container_y_min = container_center[1] - container_width / 2.0
            container_y_max = container_center[1] + container_width / 2.0

            if cube_x < container_x_min or cube_x > container_x_max or cube_y < container_y_min or cube_y > container_y_max:
                carb.log_warn(f"Cube {self.placed_count + 1} placement outside container bounds")

            container_center_z = 0.6
            container_half_height = self.container_dimensions[2] / 2.0
            container_bottom_z = container_center_z - container_half_height

            clearance_offset = 0.0
            place_height = container_bottom_z + cylinder_half + clearance_offset
            place_pos = np.array([cube_x, cube_y, place_height])

            collision_radius = 0.20
            for i in range(self.current_cube_index + 1, len(self.cubes)):
                other_cube, _ = self.cubes[i]

                if not other_cube.prim or not other_cube.prim.IsValid():
                    continue

                other_pos, _ = other_cube.get_world_pose()
                distance = np.linalg.norm(other_pos[:2] - place_pos[:2])
                if distance < collision_radius:
                    try:
                        self.rrt.add_obstacle(other_cube, static=False)
                        self.cube_obstacles_enabled = True
                    except Exception:
                        pass

            pre_place_pos = place_pos + np.array([0.0, 0.0, 0.28])

            above_place = np.array([place_pos[0], place_pos[1], safe_height])
            await self._move_to_target_rrt(above_place, orientation, skip_factor=3)
            await self._move_to_target_rrt(pre_place_pos, orientation, skip_factor=3)

            release_height = place_pos + np.array([0.0, 0.0, 0.16])
            await self._move_to_target_rrt(release_height, orientation, skip_factor=3)
            for _ in range(3):
                await omni.kit.app.get_app().next_update_async()

            articulation_controller = self.franka.get_articulation_controller()
            articulation_controller.apply_action(ArticulationAction(
                joint_positions=self.gripper.joint_opened_positions, joint_indices=np.array([7, 8])))
            for _ in range(12):
                await omni.kit.app.get_app().next_update_async()

            self._remove_all_cube_obstacles()

            cube_pos_final, _ = cube.get_world_pose()
            xy_distance = np.linalg.norm(cube_pos_final[:2] - place_pos[:2])
            placement_successful = (xy_distance < 0.15) and (cube_pos_final[2] > 0.6)

            current_ee_pos, _ = self.franka.end_effector.get_world_pose()
            retreat_pos = current_ee_pos + np.array([0.0, 0.0, 0.15])
            await self._move_to_target_rrt(retreat_pos, orientation, skip_factor=6)

            articulation_controller.apply_action(ArticulationAction(
                joint_positions=self.gripper.joint_closed_positions, joint_indices=np.array([7, 8])))
            for _ in range(2):
                await omni.kit.app.get_app().next_update_async()

            safe_ee_position = np.array([0.40, 0.0, safe_height])
            safe_success = await self._move_to_target_rrt(safe_ee_position, orientation, skip_factor=5)

            if not safe_success:
                await self._reset_to_safe_config()

            return (True, "") if placement_successful else (False, f"{cube_name} not in container")

        except Exception as e:
            self._remove_all_cube_obstacles()

            error_msg = f"Error picking/placing Cube {cube_name}: {str(e)}"
            import traceback
            traceback.print_exc()
            return False, error_msg

    async def _move_to_target_ik(self, target_position, target_orientation, num_steps=8):
        """
        Move to target using IK directly (for simple, straight-line movements)
        This is faster and more predictable than RRT for vertical descents

        Args:
            target_position: Target end effector position
            target_orientation: Target end effector orientation
            num_steps: Number of interpolation steps (default: 8, reduced for FPS)

        Returns:
            bool: True if successful, False if IK failed
        """
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

    async def _move_to_target_rrt(self, target_position, target_orientation, skip_factor=3):
        """
        Move to target using RRT (for long-distance collision-free motion)

        Args:
            target_position: Target end effector position
            target_orientation: Target end effector orientation
            skip_factor: Frame skip factor for execution speed (default=3 for 60 FPS)

        Returns:
            bool: True if successful, False if planning/execution failed
        """
        plan = self._plan_to_target(target_position, target_orientation)
        if plan is None:
            return False
        await self._execute_plan(plan, skip_factor=skip_factor)

        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()

        return True

    async def _reset_to_safe_config(self):
        """Reset robot to a known safe configuration using direct joint control"""
        safe_arm_joints = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])
        safe_joint_positions = np.concatenate([safe_arm_joints, self.gripper.joint_closed_positions])

        articulation_controller = self.franka.get_articulation_controller()
        articulation_controller.apply_action(ArticulationAction(joint_positions=safe_joint_positions))
        for _ in range(8):
            await omni.kit.app.get_app().next_update_async()

    def _plan_to_target(self, target_position, target_orientation, update_obstacles=True):
        """Plan path to target using RRT with smooth trajectory generation

        Args:
            target_position: Target end-effector position
            target_orientation: Target end-effector orientation
            update_obstacles: If True, update dynamic obstacles from Lidar (default: True)
                             Set to False when planning retreat to avoid detecting target cube
        """
        if update_obstacles:
            self._update_dynamic_obstacles()

        robot_base_translation, robot_base_orientation = self.franka.get_world_pose()
        self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        self.rrt.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        self.rrt.update_world()

        _, ik_success = self.articulation_kinematics_solver.compute_inverse_kinematics(target_position, target_orientation)
        if not ik_success:
            carb.log_warn(f"IK failed for {target_position}")
            return None

        self.rrt.set_end_effector_target(target_position, target_orientation)
        self.rrt.update_world()

        active_joints = self.path_planner_visualizer.get_active_joints_subset()
        start_pos = active_joints.get_joint_positions()

        if np.any(np.isnan(start_pos)) or np.any(np.abs(start_pos) > 10.0):
            carb.log_error(f"Invalid robot config: {start_pos}")
            return None

        near_obstacle_1 = False
        if self.obstacle_1_moving and "obstacle_1" in self.obstacles:
            obs_pos, _ = self.obstacles["obstacle_1"].get_world_pose()
            ee_pos, _ = self.franka.end_effector.get_world_pose()
            distance_to_obstacle = np.linalg.norm(ee_pos[:2] - obs_pos[:2])
            near_obstacle_1 = distance_to_obstacle < 0.30

        has_obstacles = self.obstacle_counter > 0

        if near_obstacle_1:
            max_iterations = 10000
        elif has_obstacles:
            max_iterations = 8000
        else:
            max_iterations = 5000

        self.rrt.set_max_iterations(max_iterations)

        rrt_plan = self.rrt.compute_path(start_pos, np.array([]))

        if rrt_plan is None or len(rrt_plan) <= 1:
            ee_pos, _ = self.franka.end_effector.get_world_pose()
            carb.log_warn(f"RRT failed: target={target_position}, current_ee={ee_pos}, obstacles={self.obstacle_counter}, iterations={max_iterations}")
            return None

        return self._convert_rrt_plan_to_trajectory(rrt_plan)

    def _convert_rrt_plan_to_trajectory(self, rrt_plan):
        """Convert RRT waypoints to smooth trajectory"""
        interpolated_path = self.path_planner_visualizer.interpolate_path(rrt_plan, 0.015)
        trajectory = self.cspace_trajectory_generator.compute_c_space_trajectory(interpolated_path)
        art_trajectory = ArticulationTrajectory(self.franka, trajectory, 1.0 / 60.0)
        return art_trajectory.get_action_sequence()

    async def _execute_plan(self, action_sequence, skip_factor=3):
        """Execute trajectory action sequence

        Args:
            action_sequence: Sequence of actions to execute
            skip_factor: Number of frames to skip (higher = faster, default=3 for 60 FPS)
        """
        if action_sequence is None or len(action_sequence) == 0:
            return False

        for i, action in enumerate(action_sequence):
            if i % skip_factor == 0:
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

        robot_base_translation, robot_base_orientation = self.franka.get_world_pose()
        self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        ee_position, ee_rot_mat = self.articulation_kinematics_solver.compute_end_effector_pose()

        return ee_position, ee_rot_mat

    def _on_reset(self):
        """Reset button callback - Delete all prims from stage"""
        try:
            self.is_picking = False
            self.obstacle_1_moving = False

            self.timeline.stop()

            if self.world is not None:
                try:
                    World.clear_instance()
                except Exception as e:
                    carb.log_error(f"Error clearing World instance: {e}")

            import omni.usd
            import omni.kit.commands
            from omni.isaac.core.utils.stage import clear_stage

            stage = omni.usd.get_context().get_stage()

            if stage:
                world_prim = stage.GetPrimAtPath("/World")
                if world_prim.IsValid():
                    children_paths = [str(child.GetPath()) for child in world_prim.GetAllChildren()]
                    if children_paths:
                        omni.kit.commands.execute('DeletePrims', paths=children_paths)

                if world_prim.IsValid():
                    omni.kit.commands.execute('DeletePrims', paths=['/World'])

            clear_stage()

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
            self.lidar = None
            self.lidar_detected_obstacles = {}
            self.depth_camera = None
            self.placed_count = 0
            self.current_cube_index = 0
            self.is_picking = False
            self.obstacle_1_moving = False
            self.obstacle_1_force_api_applied = False
            if hasattr(self, '_last_accel'):
                delattr(self, '_last_accel')
            self._physics_callback_added = False

            self.load_btn.enabled = True
            self.pick_btn.enabled = False
            self.reset_btn.enabled = False
            self.add_obstacle_btn.enabled = False
            self.remove_obstacle_btn.enabled = False

            self._update_status("Reset complete - stage cleared")

        except Exception as e:
            self._update_status(f"Error resetting: {e}")
            import traceback
            traceback.print_exc()

    def _add_initial_obstacles(self):
        """Add Obstacle_1 and Obstacle_2 automatically when scene loads"""
        try:
            self.obstacle_counter += 1
            obstacle_1_name = f"obstacle_{self.obstacle_counter}"
            obstacle_1_prim_path = f"/World/Obstacle_{self.obstacle_counter}"
            obstacle_1_position = np.array([0.55, 0.26, 0.7])
            obstacle_size = np.array([0.20, 0.05, 0.22])

            cube_prim_1 = prim_utils.create_prim(
                prim_path=obstacle_1_prim_path,
                prim_type="Cube",
                position=obstacle_1_position,
                orientation=euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
                scale=obstacle_size,
                attributes={"size": 1.0}
            )

            stage = omni.usd.get_context().get_stage()
            cube_geom_1 = UsdGeom.Cube.Get(stage, obstacle_1_prim_path)
            if cube_geom_1:
                cube_geom_1.CreateDisplayColorAttr([(0.0, 0.0, 1.0)])

            if not cube_prim_1.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI.Apply(cube_prim_1)
            rigid_body_api_1 = UsdPhysics.RigidBodyAPI(cube_prim_1)
            rigid_body_api_1.CreateKinematicEnabledAttr().Set(True)

            if not cube_prim_1.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(cube_prim_1)
            mesh_collision_api_1 = UsdPhysics.MeshCollisionAPI.Apply(cube_prim_1)
            mesh_collision_api_1.GetApproximationAttr().Set("convexHull")

            obstacle_1 = FixedCuboid(
                name=obstacle_1_name,
                prim_path=obstacle_1_prim_path,
                size=1.0,
                scale=obstacle_size
            )
            self.world.scene.add(obstacle_1)
            self.obstacles[obstacle_1_name] = obstacle_1

            if not hasattr(self, '_initial_obstacle_positions'):
                self._initial_obstacle_positions = {}
            self._initial_obstacle_positions[obstacle_1_name] = obstacle_1_position



            self.obstacle_counter += 1
            obstacle_2_name = f"obstacle_{self.obstacle_counter}"
            obstacle_2_prim_path = f"/World/Obstacle_{self.obstacle_counter}"
            obstacle_2_position = np.array([0.55, -0.45, 0.7])
            obstacle_2_size = np.array([0.20, 0.05, 0.22])

            cube_prim_2 = prim_utils.create_prim(
                prim_path=obstacle_2_prim_path,
                prim_type="Cube",
                position=obstacle_2_position,
                orientation=euler_angles_to_quats(np.array([0, 0, 90]), degrees=True),
                scale=obstacle_2_size,
                attributes={"size": 1.0}
            )

            cube_geom_2 = UsdGeom.Cube.Get(stage, obstacle_2_prim_path)
            if cube_geom_2:
                cube_geom_2.CreateDisplayColorAttr([(0.0, 0.0, 1.0)])

            if not cube_prim_2.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI.Apply(cube_prim_2)
            rigid_body_api_2 = UsdPhysics.RigidBodyAPI(cube_prim_2)
            rigid_body_api_2.CreateKinematicEnabledAttr().Set(True)

            if not cube_prim_2.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(cube_prim_2)
            mesh_collision_api_2 = UsdPhysics.MeshCollisionAPI.Apply(cube_prim_2)
            mesh_collision_api_2.GetApproximationAttr().Set("convexHull")

            obstacle_2 = FixedCuboid(
                name=obstacle_2_name,
                prim_path=obstacle_2_prim_path,
                size=1.0,
                scale=obstacle_2_size
            )
            self.world.scene.add(obstacle_2)
            self.obstacles[obstacle_2_name] = obstacle_2

            self._initial_obstacle_positions[obstacle_2_name] = obstacle_2_position

        except Exception as e:
            carb.log_error(f"Failed to add initial obstacles: {e}")

    def _add_initial_obstacles_to_rrt(self):
        """Add initial obstacles (Obstacle_1 and Obstacle_2) to RRT planner after RRT is initialized"""
        try:
            if not self.rrt:
                carb.log_error("RRT planner not initialized")
                return

            for obstacle_name, obstacle in self.obstacles.items():
                is_static = obstacle_name != "obstacle_1"
                self.rrt.add_obstacle(obstacle, static=is_static)

        except Exception as e:
            carb.log_error(f"Failed to add initial obstacles to RRT: {e}")

    def _on_add_obstacle(self):
        """Add obstacle button callback"""
        if not self.world or not self.rrt:
            self._update_status("Load scene first!")
            return

        try:
            self.obstacle_counter += 1
            obstacle_name = f"obstacle_{self.obstacle_counter}"

            obstacle_prim_path = find_unique_string_name(
                initial_name=f"/World/Obstacle_{self.obstacle_counter}",
                is_unique_fn=lambda x: not is_prim_path_valid(x)
            )


            num_existing_obstacles = len(self.obstacles)

            if num_existing_obstacles == 0:
                obstacle_position = np.array([0.55, 0.26, 0.7])
                obstacle_orientation = np.array([0, 0, 0])
            elif num_existing_obstacles == 1:
                obstacle_position = np.array([0.55, -0.45, 0.7])
                obstacle_orientation = np.array([0, 0, 90])
            else:
                offset = (num_existing_obstacles - 1) * 0.08
                obstacle_position = np.array([0.55, -0.45 - offset, 0.7])
                obstacle_orientation = np.array([0, 0, 0])

            obstacle_size = np.array([0.20, 0.05, 0.22])

            cube_prim = prim_utils.create_prim(
                prim_path=obstacle_prim_path,
                prim_type="Cube",
                position=obstacle_position,
                orientation=euler_angles_to_quats(obstacle_orientation, degrees=True),
                scale=obstacle_size,
                attributes={"size": 1.0}
            )

            stage = omni.usd.get_context().get_stage()
            cube_geom = UsdGeom.Cube.Get(stage, obstacle_prim_path)
            if cube_geom:
                cube_geom.CreateDisplayColorAttr([(0.0, 0.0, 1.0)])

            if not cube_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI.Apply(cube_prim)

            rigid_body_api = UsdPhysics.RigidBodyAPI(cube_prim)

            rigid_body_api.CreateKinematicEnabledAttr().Set(True)

            if not cube_prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(cube_prim)

            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(cube_prim)
            mesh_collision_api.GetApproximationAttr().Set("convexHull")

            obstacle = FixedCuboid(
                name=obstacle_name,
                prim_path=obstacle_prim_path,
                size=1.0,
                scale=obstacle_size
            )

            self.world.scene.add(obstacle)

            self.rrt.add_obstacle(obstacle, static=False)

            self.obstacles[obstacle_name] = obstacle

            self._update_status(f"Obstacle added ({len(self.obstacles)})")

        except Exception as e:
            self._update_status(f"Error adding obstacle: {e}")
            import traceback
            traceback.print_exc()

    def _on_remove_obstacle(self):
        """Remove obstacle button callback - removes last added obstacle"""
        if not self.world or not self.rrt:
            self._update_status("Load scene first!")
            return

        if len(self.obstacles) == 0:
            self._update_status("No obstacles to remove!")
            return

        try:
            obstacle_name = list(self.obstacles.keys())[-1]
            obstacle = self.obstacles[obstacle_name]

            obstacle_prim_path = obstacle.prim_path

            try:
                self.rrt.remove_obstacle(obstacle)
            except Exception as e:
                carb.log_warn(f"Could not remove from RRT planner: {e}")

            del self.obstacles[obstacle_name]

            import omni.kit.commands

            omni.kit.commands.execute('DeletePrims', paths=[obstacle_prim_path])

            self._update_status(f"Obstacle removed ({len(self.obstacles)} remaining)")

        except Exception as e:
            self._update_status(f"Error removing obstacle: {e}")


def main():
    """Main function for standalone execution"""
    app = FrankaRRTDynamicGrid(num_cubes=args.num_cubes, training_grid_size=args.training_grid_size)

    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        pass
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

