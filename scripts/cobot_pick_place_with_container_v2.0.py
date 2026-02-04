import carb, asyncio, time
import numpy as np
import sys
import os
from pathlib import Path

project_root = None

try:
    if '__file__' in globals():
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
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

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleXFormPrim, SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.storage.native import get_assets_root_path
import omni.timeline
from omni.kit.async_engine import run_coroutine
from pxr import UsdPhysics

from src.manipulators import SingleManipulator
from src.controllers.franka import PickPlaceController
from src.grippers import ParallelGripper

async def setup_scene():
    """Setup the scene with cobot on stool, object on table, and container"""
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        raise RuntimeError("Could not find Isaac Sim assets folder")

    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    await omni.kit.app.get_app().next_update_async()

    World.clear_instance()
    await omni.kit.app.get_app().next_update_async()

    my_world = World(stage_units_in_meters=1.0, physics_dt=1.0/30.0)
    await omni.kit.app.get_app().next_update_async()

    my_world.scene.add_default_ground_plane()

    timestamp = int(time.time() * 1000)
    cobot_name = f"cobot_{timestamp}"
    object_name = f"object_{timestamp}"
    container_name = "container"
    stool_name = "stool"
    table_name = "table"
    cobot_prim_path = f"/World/Cobot_{timestamp}"
    object_prim_path = f"/World/Object_{timestamp}"
    container_prim_path = "/World/Container"
    stool_prim_path = "/World/Stool"
    table_prim_path = "/World/Table"

    stool_usd_path = str(project_root / "assets" / "Franka_Stool.usd")
    stool_xform = my_world.scene.add(
        SingleXFormPrim(
            prim_path=stool_prim_path,
            name=stool_name,
            translation=np.array([0.0, 0.0, 0.0])
        )
    )
    add_reference_to_stage(usd_path=stool_usd_path, prim_path=stool_prim_path)
    await omni.kit.app.get_app().next_update_async()

    stool_prim = get_prim_at_path(stool_prim_path)
    if not stool_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(stool_prim)
        rigid_body_api.CreateRigidBodyEnabledAttr(True)
        rigid_body_api.CreateKinematicEnabledAttr(True)
    if not stool_prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(stool_prim)
    await omni.kit.app.get_app().next_update_async()

    stool_height = 0.75

    asset_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    robot = add_reference_to_stage(usd_path=asset_path, prim_path=cobot_prim_path)
    robot.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
    robot.GetVariantSet("Mesh").SetVariantSelection("Quality")

    cobot_xform = SingleXFormPrim(
        prim_path=cobot_prim_path,
        name=cobot_name,
        translation=np.array([0.0, 0.0, stool_height])
    )

    gripper = ParallelGripper(
        end_effector_prim_path=f"{cobot_prim_path}/panda_rightfinger",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=np.array([0.05, 0.05]),
        joint_closed_positions=np.array([0.02, 0.02]),
        action_deltas=np.array([0.01, 0.01]),
    )

    my_cobot = my_world.scene.add(
        SingleManipulator(
            prim_path=cobot_prim_path,
            name=cobot_name,
            end_effector_prim_path=f"{cobot_prim_path}/panda_rightfinger",
            gripper=gripper,
        )
    )

    table_usd_path = str(project_root / "assets" / "Table_Rounded.usd")
    table_translation = np.array([0.5, 0.3, 0.0])
    table_xform = my_world.scene.add(
        SingleXFormPrim(
            prim_path=table_prim_path,
            name=table_name,
            translation=table_translation
        )
    )
    add_reference_to_stage(usd_path=table_usd_path, prim_path=table_prim_path)
    await omni.kit.app.get_app().next_update_async()

    table_prim = get_prim_at_path(table_prim_path)
    if not table_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(table_prim)
        rigid_body_api.CreateRigidBodyEnabledAttr(True)
        rigid_body_api.CreateKinematicEnabledAttr(True)

    if not table_prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(table_prim)
    await omni.kit.app.get_app().next_update_async()

    table_height = 0.75

    container_usd_path = assets_root_path + "/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Storage/Containers/Container_I/Container_I04_160x120x64cm_PR_V_NVD_01.usd"
    container_translation = np.array([-0.3, -0.5, stool_height])

    container_xform = my_world.scene.add(
        SingleXFormPrim(
            prim_path=container_prim_path,
            name=container_name,
            scale=np.array([0.3, 0.3, 0.3]),
            translation=container_translation
        )
    )
    add_reference_to_stage(usd_path=container_usd_path, prim_path=container_prim_path)
    await omni.kit.app.get_app().next_update_async()

    container_prim = get_prim_at_path(container_prim_path)
    if not container_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(container_prim)
        rigid_body_api.CreateRigidBodyEnabledAttr(True)
        rigid_body_api.CreateKinematicEnabledAttr(True)
        
    if not container_prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(container_prim)
    await omni.kit.app.get_app().next_update_async()

    object_half_size = 0.0515 / 2.0
    placing_height = stool_height + object_half_size + 0.08
    placing_position = np.array([container_translation[0], container_translation[1], placing_height])

    object_drop_height = table_height + 0.3
    target_object = my_world.scene.add(
        DynamicCuboid(
            name=object_name,
            position=np.array([table_translation[0], table_translation[1], object_drop_height]),
            prim_path=object_prim_path,
            scale=np.array([0.0515, 0.0515, 0.0515]),
            size=1.0,
            color=np.array([0, 0, 1]),
        )
    )

    my_cobot.gripper.set_default_state(my_cobot.gripper.joint_opened_positions)

    my_world.initialize_physics()
    for _ in range(2):
        await omni.kit.app.get_app().next_update_async()

    my_world.reset()
    for _ in range(2):
        await omni.kit.app.get_app().next_update_async()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(10):
        await omni.kit.app.get_app().next_update_async()

    for _ in range(60):
        await omni.kit.app.get_app().next_update_async()

    return my_world, my_cobot, target_object, placing_position

async def run_pick_and_place():
    """Main async function to run the pick and place task"""
    my_world, my_cobot, target_object, placing_position = await setup_scene()

    my_controller = PickPlaceController(
        name="pick_place_controller",
        gripper=my_cobot.gripper,
        robot_articulation=my_cobot
    )
    articulation_controller = my_cobot.get_articulation_controller()

    task_completed = False
    step_count = 0
    max_steps = 3000
    reset_needed = False

    while not task_completed and step_count < max_steps:
        await omni.kit.app.get_app().next_update_async()

        if my_world.is_stopped() and not reset_needed:
            reset_needed = True
            task_completed = False

        if my_world.is_playing():
            if reset_needed:
                my_world.reset()
                my_controller.reset()
                reset_needed = False
                task_completed = False

            step_count += 1

            actions = my_controller.forward(
                picking_position=target_object.get_local_pose()[0],
                placing_position=placing_position,
                current_joint_positions=my_cobot.get_joint_positions(),
                end_effector_offset=np.array([0, 0.005, 0]),
            )

            if my_controller.is_done():
                task_completed = True
            else:
                articulation_controller.apply_action(actions)
        else:
            break

run_coroutine(run_pick_and_place())

