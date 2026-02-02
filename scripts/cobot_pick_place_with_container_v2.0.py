import carb, asyncio, time
import numpy as np
import sys
from pathlib import Path

# Add project root to Python path for local imports
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Isaac Sim Core APIs
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleXFormPrim, SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.storage.native import get_assets_root_path
import omni.timeline
from omni.kit.async_engine import run_coroutine
from pxr import UsdPhysics

# Local project imports
from src.manipulators import SingleManipulator
from src.controllers.franka import PickPlaceController
from src.grippers import ParallelGripper

async def setup_scene():
    """Setup the scene with Franka on stool, cube on table, and container"""

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        raise RuntimeError("Could not find Isaac Sim assets folder")

    # Stop timeline first
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()

    # Wait for timeline to stop
    await omni.kit.app.get_app().next_update_async()

    # Clear any existing World instance to ensure clean state
    print("Creating fresh World instance with optimized physics...")
    World.clear_instance()

    # Wait for cleanup to complete
    await omni.kit.app.get_app().next_update_async()

    # Create new World instance
    my_world = World(stage_units_in_meters=1.0, physics_dt=1.0/30.0)

    # Wait for World to initialize
    await omni.kit.app.get_app().next_update_async()

    print("Adding ground plane...")
    my_world.scene.add_default_ground_plane()

    # Generate unique names using timestamp for Franka and Cube only
    timestamp = int(time.time() * 1000)
    franka_name = f"franka_{timestamp}"
    cube_name = f"cube_{timestamp}"
    container_name = "container"
    stool_name = "stool"
    table_name = "table"
    franka_prim_path = f"/World/Franka_{timestamp}"
    cube_prim_path = f"/World/Cube_{timestamp}"
    container_prim_path = "/World/Container"
    stool_prim_path = "/World/Stool"
    table_prim_path = "/World/Table"

    # Add stool for Franka - add to scene first
    print("Adding stool for Franka...")
    stool_usd_path = str(project_root / "assets" / "Franka_Stool.usd")
    stool_xform = my_world.scene.add(
        SingleXFormPrim(
            prim_path=stool_prim_path,
            name=stool_name,
            translation=np.array([0.0, 0.0, 0.0])
        )
    )
    add_reference_to_stage(usd_path=stool_usd_path, prim_path=stool_prim_path)

    # Wait for stool to be added
    await omni.kit.app.get_app().next_update_async()

    # Add physics to stool (static)
    print("Adding physics to stool...")
    stool_prim = get_prim_at_path(stool_prim_path)
    if not stool_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(stool_prim)
        rigid_body_api.CreateRigidBodyEnabledAttr(True)
        rigid_body_api.CreateKinematicEnabledAttr(True)  # Static
    if not stool_prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(stool_prim)

    await omni.kit.app.get_app().next_update_async()

    # Stool height - adjust based on actual asset (estimate ~0.75m)
    stool_height = 0.75

    # Add Franka robot on top of stool
    print("Adding Franka robot on stool...")
    asset_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    robot = add_reference_to_stage(usd_path=asset_path, prim_path=franka_prim_path)
    robot.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
    robot.GetVariantSet("Mesh").SetVariantSelection("Quality")
    
    # Position Franka on top of stool
    franka_xform = SingleXFormPrim(
        prim_path=franka_prim_path,
        name=franka_name,
        translation=np.array([0.0, 0.0, stool_height])
    )

    # Setup gripper
    gripper = ParallelGripper(
        end_effector_prim_path=f"{franka_prim_path}/panda_rightfinger",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=np.array([0.05, 0.05]),
        joint_closed_positions=np.array([0.02, 0.02]),
        action_deltas=np.array([0.01, 0.01]),
    )

    # Add manipulator to scene
    print("Adding manipulator to scene...")
    my_franka = my_world.scene.add(
        SingleManipulator(
            prim_path=franka_prim_path,
            name=franka_name,
            end_effector_prim_path=f"{franka_prim_path}/panda_rightfinger",
            gripper=gripper,
        )
    )

    # Add table for cube - add to scene first
    print("Adding table for cube...")
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

    # Wait for table to be added
    await omni.kit.app.get_app().next_update_async()

    # Add physics to table (static)
    print("Adding physics to table...")
    table_prim = get_prim_at_path(table_prim_path)
    if not table_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(table_prim)
        rigid_body_api.CreateRigidBodyEnabledAttr(True)
        rigid_body_api.CreateKinematicEnabledAttr(True)  # Static
    if not table_prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(table_prim)

    await omni.kit.app.get_app().next_update_async()

    # Table height - adjust based on actual asset (estimate ~0.75m)
    table_height = 0.75

    # Add container to scene - elevated to be reachable by Franka on stool
    print("Adding container to scene...")
    container_usd_path = assets_root_path + "/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Storage/Containers/Container_I/Container_I04_160x120x64cm_PR_V_NVD_01.usd"

    # Container position - elevated to match Franka's height on stool
    # Place container at stool height so Franka can reach it
    container_translation = np.array([-0.3, -0.5, stool_height])

    # Set container scale and translation using SingleXFormPrim - add to scene first
    container_xform = my_world.scene.add(
        SingleXFormPrim(
            prim_path=container_prim_path,
            name=container_name,
            scale=np.array([0.3, 0.3, 0.3]),
            translation=container_translation
        )
    )
    add_reference_to_stage(usd_path=container_usd_path, prim_path=container_prim_path)

    # Wait for container to be added
    await omni.kit.app.get_app().next_update_async()

    # Add physics to container so cube can rest on it
    print("Adding physics to container...")
    container_prim = get_prim_at_path(container_prim_path)

    # Add rigid body API (static, not dynamic)
    if not container_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(container_prim)
        rigid_body_api.CreateRigidBodyEnabledAttr(True)
        # Make it static (kinematic) so it doesn't fall
        rigid_body_api.CreateKinematicEnabledAttr(True)

    # Add collision API
    if not container_prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(container_prim)

    # Wait for physics to be applied
    await omni.kit.app.get_app().next_update_async()

    # Calculate placing position - cube should be placed inside container
    # Container is at stool_height, scaled to 0.3, original height is 64cm = 0.64m
    # Scaled height = 0.64 * 0.3 = 0.192m
    # Cube size = 0.0515m
    # Place cube higher to prevent sinking - add more offset
    container_scaled_height = 0.64 * 0.3  # 0.192m
    cube_half_size = 0.0515 / 2.0
    placing_height = stool_height + cube_half_size + 0.08  # Container base + cube offset
    placing_position = np.array([container_translation[0], container_translation[1], placing_height])

    # Add cube to scene - above table so it falls and rests on table
    print("Adding cube above table...")
    cube_drop_height = table_height + 0.3  # Drop from above table
    cube = my_world.scene.add(
        DynamicCuboid(
            name=cube_name,
            position=np.array([table_translation[0], table_translation[1], cube_drop_height]),
            prim_path=cube_prim_path,
            scale=np.array([0.0515, 0.0515, 0.0515]),
            size=1.0,
            color=np.array([0, 0, 1]),
        )
    )

    # Set gripper default state
    my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)

    print("Initializing physics with optimizations...")
    my_world.initialize_physics()

    for _ in range(2):
        await omni.kit.app.get_app().next_update_async()

    print("Resetting world...")
    my_world.reset()

    for _ in range(2):
        await omni.kit.app.get_app().next_update_async()

    print("Playing simulation...")
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    for _ in range(10):
        await omni.kit.app.get_app().next_update_async()

    # Wait for cube to settle on table
    print("Waiting for cube to settle on table...")
    for _ in range(60):  # Wait ~2 seconds at 30Hz physics
        await omni.kit.app.get_app().next_update_async()

    print("Cube settled on table, ready for pick and place")

    return my_world, my_franka, cube, placing_position

async def run_pick_and_place():
    """Main async function to run the pick and place task"""

    # Setup scene
    my_world, my_franka, cube, placing_position = await setup_scene()

    # Create pick and place controller
    print("Creating pick and place controller...")
    my_controller = PickPlaceController(
        name="pick_place_controller",
        gripper=my_franka.gripper,
        robot_articulation=my_franka
    )
    articulation_controller = my_franka.get_articulation_controller()

    # Task state variables
    task_completed = False
    step_count = 0
    max_steps = 3000
    reset_needed = False

    print("Starting pick and place task...")
    print(f"World is playing: {my_world.is_playing()}")
    print(f"Placing position: {placing_position}")

    # Main simulation loop
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

            # Run controller every frame
            actions = my_controller.forward(
                picking_position=cube.get_local_pose()[0],
                placing_position=placing_position,
                current_joint_positions=my_franka.get_joint_positions(),
                end_effector_offset=np.array([0, 0.005, 0]),
            )

            if my_controller.is_done():
                print("Done picking and placing!")
                task_completed = True
            else:
                articulation_controller.apply_action(actions)
        else:
            print("Warning: World is not playing")
            break

    if step_count >= max_steps:
        print(f"Task reached maximum steps ({max_steps})")

    print(f"Pick and place task completed! (Steps: {step_count})")

# Run the async function
print("Starting Franka pick and place task with stool, table, and container...")
run_coroutine(run_pick_and_place())

