import carb, asyncio, time
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to Python path for local imports
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Isaac Sim Core APIs
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
import omni.timeline
from omni.kit.async_engine import run_coroutine
from pxr import UsdPhysics

# Lula RRT motion planning
from isaacsim.robot_motion.motion_generation import PathPlannerVisualizer, interface_config_loader
from isaacsim.robot_motion.motion_generation.lula import RRT

# Local project imports
from src.manipulators import SingleManipulator
from src.grippers import ParallelGripper

# Pick and place states
class PickPlaceState:
    IDLE = 0
    MOVING_TO_PICK = 1
    PICKING = 2
    MOVING_TO_PLACE = 3
    PLACING = 4
    DONE = 5

async def setup_scene():
    """Setup the scene with Franka robot, cube, and container"""

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

    # Generate unique names using timestamp for Franka and Cube
    timestamp = int(time.time() * 1000)
    franka_name = f"franka_{timestamp}"
    cube_name = f"cube_{timestamp}"
    container_name = "container"
    franka_prim_path = f"/World/Franka_{timestamp}"
    cube_prim_path = f"/World/Cube_{timestamp}"
    container_prim_path = "/World/Container"

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

    # Add container to scene
    print("Adding container to scene...")
    container_usd_path = assets_root_path + "/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Storage/Containers/Container_I/Container_I04_160x120x64cm_PR_V_NVD_01.usd"
    add_reference_to_stage(usd_path=container_usd_path, prim_path=container_prim_path)

    # Container translation and scale
    container_translation = np.array([-0.2, -0.5, 0.0])

    # Set container scale and translation using SingleXFormPrim
    container_xform = SingleXFormPrim(
        prim_path=container_prim_path,
        name=container_name,
        scale=np.array([0.3, 0.3, 0.3]),
        translation=container_translation
    )

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

    # Add obstacle wall between robot and cube
    print("Adding obstacle wall...")
    obstacle = my_world.scene.add(
        VisualCuboid(
            prim_path="/World/Obstacle",
            name="obstacle_wall",
            position=np.array([0.3, 0.0, 0.3]),  # Between robot and cube
            size=1.0,
            scale=np.array([0.1, 0.4, 0.4]),  # Thin wall
            color=np.array([1, 0, 0]),  # Red color
        )
    )

    # Wait for obstacle to be added
    await omni.kit.app.get_app().next_update_async()

    # Calculate placing position - cube should be placed inside container
    cube_half_size = 0.0515 / 2.0
    placing_height = cube_half_size + 0.08
    placing_position = np.array([container_translation[0], container_translation[1], placing_height])

    # Add cube to scene - position to the right of obstacle so RRT must plan around it
    print("Adding cube to scene (right of obstacle)...")
    cube = my_world.scene.add(
        DynamicCuboid(
            name=cube_name,
            position=np.array([0.45, 0.3, 0.3]),  # To the right of obstacle
            prim_path=cube_prim_path,
            scale=np.array([0.0515, 0.0515, 0.0515]),
            size=1.0,
            color=np.array([0, 0, 1]),
        )
    )

    # Set gripper default state
    gripper.set_default_state(gripper.joint_opened_positions)

    # Set Franka to a valid default configuration for RRT
    # Using the default_q from robot_descriptor.yaml: [0.00, -1.3, 0.00, -2.87, 0.00, 2.00, 0.75]
    # Plus gripper joints: [0.04, 0.04]
    print("Setting Franka to default configuration...")
    default_joint_positions = np.array([0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75, 0.04, 0.04])
    my_franka.set_joints_default_state(positions=default_joint_positions)

    print("Initializing physics with optimizations...")
    my_world.initialize_physics()

    for _ in range(2):
        await omni.kit.app.get_app().next_update_async()

    print("Resetting world...")
    my_world.reset()

    for _ in range(5):
        await omni.kit.app.get_app().next_update_async()

    print("Playing simulation...")
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    for _ in range(10):
        await omni.kit.app.get_app().next_update_async()

    print("Franka initialized at default configuration")

    return my_world, my_franka, gripper, cube, placing_position, obstacle

def setup_rrt(franka_articulation, obstacle):
    """Setup Lula RRT motion planner"""

    # Get motion generation extension path
    mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
    rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
    rrt_config_dir = os.path.join(mg_extension_path, "path_planner_configs")

    print(f"Loading RRT config from: {rrt_config_dir}/franka/rrt/")

    # Initialize RRT
    rrt = RRT(
        robot_description_path=rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
        urdf_path=rmp_config_dir + "/franka/lula_franka_gen.urdf",
        rrt_config_path=rrt_config_dir + "/franka/rrt/franka_planner_config.yaml",
        end_effector_frame_name="right_gripper"
    )

    # Add obstacle to RRT for collision avoidance
    print("Adding obstacle wall to RRT...")
    rrt.add_obstacle(obstacle)

    # Set max iterations to prevent blocking (increase for complex scenes)
    rrt.set_max_iterations(10000)

    # Create path planner visualizer wrapper to generate trajectory of ArticulationActions
    path_planner_visualizer = PathPlannerVisualizer(franka_articulation, rrt)

    print("RRT motion planner initialized successfully")
    print(f"RRT max iterations: 10000")

    return rrt, path_planner_visualizer

async def run_pick_and_place():
    """Main async function to run the pick and place task with RRT"""

    # Setup scene
    my_world, my_franka, gripper, cube, placing_position, obstacle = await setup_scene()

    # Setup RRT motion planner
    print("Setting up RRT motion planner...")
    rrt, path_planner_visualizer = setup_rrt(my_franka, obstacle)

    # Task state variables
    state = PickPlaceState.IDLE
    plan = []
    step_count = 0
    max_steps = 5000
    gripper_action_counter = 0

    # End effector offset for grasping
    ee_offset = np.array([0, 0, 0.1])  # Offset above the cube for approach

    print("Starting pick and place task with RRT...")
    print(f"Placing position: {placing_position}")

    # Main simulation loop
    while state != PickPlaceState.DONE and step_count < max_steps:
        await omni.kit.app.get_app().next_update_async()
        step_count += 1

        if not my_world.is_playing():
            print("Warning: World is not playing")
            break

        # State machine for pick and place
        if state == PickPlaceState.IDLE:
            print("State: IDLE -> Planning path to pick position...")

            # Get current robot configuration (all 9 DOF)
            current_joint_positions = my_franka.get_joint_positions()
            print(f"Current joint positions (9 DOF): {current_joint_positions}")

            # Get only the 7 arm joints (exclude gripper joints)
            arm_joint_positions = current_joint_positions[:7]
            print(f"Arm joint positions (7 DOF): {arm_joint_positions}")

            cube_position, _ = cube.get_world_pose()
            pick_target = cube_position + ee_offset
            pick_orientation = euler_angles_to_quats([np.pi, 0, 0])  # Gripper pointing down

            print(f"Pick target: {pick_target}")
            print(f"Planning path with RRT (this may take a few seconds)...")

            # Set target and update world state before planning
            rrt.set_end_effector_target(pick_target, pick_orientation)

            # Update world with current arm joint positions (7 DOF only)
            rrt.update_world(arm_joint_positions)

            # Compute plan using PathPlannerVisualizer
            plan = path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=0.01)

            if plan:
                print(f"✓ Plan computed successfully with {len(plan)} waypoints")
                state = PickPlaceState.MOVING_TO_PICK
            else:
                print("✗ Failed to compute plan to pick position")
                print("  Possible reasons:")
                print("  - Target is unreachable")
                print("  - Obstacle blocks all paths")
                print("  - Initial configuration is invalid")
                state = PickPlaceState.DONE

        elif state == PickPlaceState.MOVING_TO_PICK:
            if plan:
                action = plan.pop(0)
                my_franka.apply_action(action)
            else:
                print("State: MOVING_TO_PICK -> PICKING")
                state = PickPlaceState.PICKING
                gripper_action_counter = 0

        elif state == PickPlaceState.PICKING:
            # Close gripper
            articulation_controller = my_franka.get_articulation_controller()
            gripper.apply_action(articulation_controller, gripper.joint_closed_positions)
            gripper_action_counter += 1

            if gripper_action_counter > 30:  # Wait for gripper to close
                print("State: PICKING -> Planning path to place position...")
                place_target = placing_position + ee_offset
                place_orientation = euler_angles_to_quats([np.pi, 0, 0])

                print(f"Place target: {place_target}")
                print(f"Planning path with RRT (this may take a few seconds)...")

                # Get current arm joint positions (7 DOF only)
                current_joint_positions = my_franka.get_joint_positions()
                arm_joint_positions = current_joint_positions[:7]

                # Set target and update world state before planning
                rrt.set_end_effector_target(place_target, place_orientation)

                # Update world with current arm joint positions (7 DOF only)
                rrt.update_world(arm_joint_positions)

                # Compute plan using PathPlannerVisualizer
                plan = path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=0.01)

                if plan:
                    print(f"✓ Plan computed successfully with {len(plan)} waypoints")
                    state = PickPlaceState.MOVING_TO_PLACE
                else:
                    print("✗ Failed to compute plan to place position")
                    state = PickPlaceState.DONE

        elif state == PickPlaceState.MOVING_TO_PLACE:
            if plan:
                action = plan.pop(0)
                my_franka.apply_action(action)
            else:
                print("State: MOVING_TO_PLACE -> PLACING")
                state = PickPlaceState.PLACING
                gripper_action_counter = 0

        elif state == PickPlaceState.PLACING:
            # Open gripper
            articulation_controller = my_franka.get_articulation_controller()
            gripper.apply_action(articulation_controller, gripper.joint_opened_positions)
            gripper_action_counter += 1

            if gripper_action_counter > 30:  # Wait for gripper to open
                print("State: PLACING -> DONE")
                state = PickPlaceState.DONE

    if step_count >= max_steps:
        print(f"Task reached maximum steps ({max_steps})")
    
    print(f"Pick and place task completed! (Steps: {step_count})")

# Run the async function
print("Starting Franka pick and place task with RRT motion planning...")
run_coroutine(run_pick_and_place())

