# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Franka Pick and Place - Script Editor Version (Async Approach)
# This script is designed to run from Isaac Sim 5.0.0 Script Editor

import carb, asyncio, time
import numpy as np
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Isaac Sim imports (keep these - they're from Isaac Sim installation)
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
import omni.timeline
from omni.kit.async_engine import run_coroutine

# Local project imports (using your local copies)
from src.manipulators import SingleManipulator
from src.controllers.franka import PickPlaceController
from src.grippers import ParallelGripper

async def setup_scene():
    """Setup the scene with Franka robot and cube"""

    # Get assets root path
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        raise RuntimeError("Could not find Isaac Sim assets folder")

    # Stop timeline first
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()

    # Wait a frame for timeline to stop
    await omni.kit.app.get_app().next_update_async()

    # Always create a fresh World instance
    print("Creating fresh World instance...")
    World.clear_instance()
    my_world = World(stage_units_in_meters=1.0)

    # Wait for World to be ready
    for _ in range(3):
        await omni.kit.app.get_app().next_update_async()

    # Add ground plane
    print("Adding ground plane...")
    my_world.scene.add_default_ground_plane()

    # Generate unique names using timestamp
    timestamp = int(time.time() * 1000)
    franka_name = f"franka_{timestamp}"
    cube_name = f"cube_{timestamp}"
    franka_prim_path = f"/World/Franka_{timestamp}"
    cube_prim_path = f"/World/Cube_{timestamp}"

    # Add Franka robot
    print(f"Adding Franka robot at {franka_prim_path}...")
    asset_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    robot = add_reference_to_stage(usd_path=asset_path, prim_path=franka_prim_path)
    robot.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
    robot.GetVariantSet("Mesh").SetVariantSelection("Quality")

    # Setup gripper
    print("Setting up gripper...")
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

    # Add cube
    print(f"Adding cube at {cube_prim_path}...")
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

    # Set gripper default state to open
    print("Setting gripper default state...")
    my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)

    # Initialize physics
    print("Initializing physics...")
    my_world.initialize_physics()

    # Wait for physics to initialize
    for _ in range(5):
        await omni.kit.app.get_app().next_update_async()

    # Reset world to apply default gripper state
    print("Resetting world...")
    my_world.reset()

    # Wait for reset to complete
    for _ in range(5):
        await omni.kit.app.get_app().next_update_async()

    # Manually play the simulation
    print("Starting simulation...")
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # Wait for simulation to start playing
    for _ in range(5):
        await omni.kit.app.get_app().next_update_async()

    return my_world, my_franka, cube, franka_name, cube_name


async def run_pick_and_place():
    """Main async function to run pick and place task"""

    # Setup scene
    my_world, my_franka, cube, franka_name, cube_name = await setup_scene()

    # Create controller
    print("Creating pick and place controller...")
    my_controller = PickPlaceController(
        name="pick_place_controller",
        gripper=my_franka.gripper,
        robot_articulation=my_franka
    )

    # Get articulation controller
    articulation_controller = my_franka.get_articulation_controller()

    # Task state variables
    task_completed = False
    step_count = 0
    max_steps = 5000
    reset_needed = False

    print("Starting pick and place task...")
    print(f"Max steps: {max_steps}")

    # Main simulation loop
    while not task_completed and step_count < max_steps:
        # Wait for next frame
        await omni.kit.app.get_app().next_update_async()

        # Check if simulation stopped
        if my_world.is_stopped() and not reset_needed:
            reset_needed = True
            task_completed = False

        # Run simulation step
        if my_world.is_playing():
            if reset_needed:
                my_world.reset()
                my_controller.reset()
                reset_needed = False
                task_completed = False

            step_count += 1

            # Run controller every frame for correct gripper behavior
            actions = my_controller.forward(
                picking_position=cube.get_local_pose()[0],
                placing_position=np.array([-0.3, -0.3, 0.0515 / 2.0]),
                current_joint_positions=my_franka.get_joint_positions(),
                end_effector_offset=np.array([0, 0.005, 0]),
            )

            # Check if task is done
            if my_controller.is_done():
                print("Done picking and placing!")
                task_completed = True
            else:
                articulation_controller.apply_action(actions)

            # Progress update every 100 steps
            if step_count % 100 == 0:
                print(f"Step: {step_count}, Event: {my_controller.get_current_event()}")

    # Final status
    if task_completed:
        print(f"✅ Task completed successfully in {step_count} steps!")
    else:
        print(f"⚠️ Task did not complete within {max_steps} steps")

    print("Simulation finished.")


# Entry point for Script Editor
print("=" * 60)
print("Franka Pick and Place - Script Editor Version")
print("Using LOCAL project imports from cobotproject/src/")
print("=" * 60)
run_coroutine(run_pick_and_place())

