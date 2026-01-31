import carb, asyncio, time
import numpy as np

# Isaac Sim Core APIs
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
import omni.timeline
from omni.kit.async_engine import run_coroutine

# ORIGINAL Isaac Sim imports (NOT local)
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController

async def setup_scene():
    """Setup the scene with Franka robot and cube"""

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

    # Generate unique names using timestamp
    timestamp = int(time.time() * 1000)
    franka_name = f"franka_{timestamp}"
    cube_name = f"cube_{timestamp}"
    franka_prim_path = f"/World/Franka_{timestamp}"
    cube_prim_path = f"/World/Cube_{timestamp}"

    # Add Franka robot
    print("Adding Franka robot...")
    asset_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    robot = add_reference_to_stage(usd_path=asset_path, prim_path=franka_prim_path)
    robot.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
    robot.GetVariantSet("Mesh").SetVariantSelection("Quality")

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

    # Add cube to scene
    print("Adding cube to scene...")
    cube = my_world.scene.add(
        DynamicCuboid(
            name=cube_name,
            position=np.array([0.3, 0.3, 0.3]),
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

    for _ in range(2):
        await omni.kit.app.get_app().next_update_async()

    return my_world, my_franka, cube

async def run_pick_and_place():
    """Main async function to run the pick and place task"""

    # Setup scene
    my_world, my_franka, cube = await setup_scene()

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
                placing_position=np.array([-0.3, -0.3, 0.0515 / 2.0]),
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
run_coroutine(run_pick_and_place())

