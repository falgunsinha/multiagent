"""
RRT-based Pick and Place Controller

Handles pick and place operations using RRT path planning with obstacle avoidance.
Integrates with object detection for target selection.
"""

import numpy as np
import asyncio
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import ArticulationTrajectory
import carb
import omni.kit.app


class RRTPickPlaceController:
    """RRT-based pick and place controller with object detection"""
    
    def __init__(self,
                 manipulator,
                 gripper,
                 rrt_planner,
                 kinematics_solver,
                 articulation_kinematics_solver,
                 cspace_trajectory_generator,
                 container_position,
                 container_dimensions,
                 obstacle_update_callback=None):
        """
        Initialize RRT pick and place controller

        Args:
            manipulator: Robot manipulator (SingleManipulator)
            gripper: Gripper controller (ParallelGripper)
            rrt_planner: RRT path planner
            kinematics_solver: Lula kinematics solver
            articulation_kinematics_solver: Articulation kinematics solver
            cspace_trajectory_generator: C-space trajectory generator
            container_position: Container position [x, y, z]
            container_dimensions: Container dimensions [length, width, height]
            obstacle_update_callback: Optional callback to update dynamic obstacles before planning
        """
        self.manipulator = manipulator
        self.gripper = gripper
        self.rrt = rrt_planner
        self.kinematics_solver = kinematics_solver
        self.articulation_kinematics_solver = articulation_kinematics_solver
        self.cspace_trajectory_generator = cspace_trajectory_generator
        self.container_position = np.array(container_position)
        self.container_dimensions = np.array(container_dimensions)
        self.obstacle_update_callback = obstacle_update_callback

        # Pick and place parameters
        self.pick_height_offset = 0.15  # Approach height above target
        self.place_height_offset = 0.20  # Place height above container
        self.max_retries = 3  # Maximum pick attempts per object
        
    async def pick_and_place_object(self, target_position, target_name="Object"):
        """
        Pick object at target position and place in container
        
        Args:
            target_position: Target object position [x, y, z]
            target_name: Name of target object for logging
            
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"\n[PICK] Attempting to pick {target_name} at {target_position}")
        
        for attempt in range(1, self.max_retries + 1):
            if attempt > 1:
                print(f"[PICK] Retry attempt {attempt}/{self.max_retries}")
                # Get fresh position for retry
                await asyncio.sleep(0.1)
            
            # Execute pick sequence
            success = await self._execute_pick_sequence(target_position, target_name)
            
            if success:
                # Execute place sequence
                place_success = await self._execute_place_sequence(target_name)
                if place_success:
                    print(f"[PICK] Successfully placed {target_name}")
                    return True
                else:
                    print(f"[PICK] Failed to place {target_name}")
                    return False
            else:
                if attempt < self.max_retries:
                    print(f"[PICK] Pick failed, retrying...")
                else:
                    print(f"[PICK] Failed to pick {target_name} after {self.max_retries} attempts")
        
        return False
    
    async def _execute_pick_sequence(self, target_position, target_name):
        """Execute pick sequence: approach -> descend -> grasp -> lift"""
        try:
            # 1. Approach position (above target)
            approach_pos = target_position.copy()
            approach_pos[2] += self.pick_height_offset
            
            print(f"[PICK] Moving to approach position: {approach_pos}")
            success = await self._move_to_position(approach_pos, "approach")
            if not success:
                return False
            
            # 2. Open gripper
            await self._open_gripper()
            
            # 3. Descend to target
            print(f"[PICK] Descending to target: {target_position}")
            success = await self._move_to_position(target_position, "pick")
            if not success:
                return False
            
            # 4. Close gripper
            await self._close_gripper()
            
            # 5. Lift object
            lift_pos = target_position.copy()
            lift_pos[2] += self.pick_height_offset
            print(f"[PICK] Lifting object to: {lift_pos}")
            success = await self._move_to_position(lift_pos, "lift")
            
            return success
            
        except Exception as e:
            carb.log_error(f"[PICK] Error in pick sequence: {e}")
            return False
    
    async def _execute_place_sequence(self, target_name):
        """Execute place sequence: move to container -> descend -> release -> retract"""
        try:
            # 1. Move above container
            place_pos = self.container_position.copy()
            place_pos[2] += self.place_height_offset
            
            print(f"[PLACE] Moving to container position: {place_pos}")
            success = await self._move_to_position(place_pos, "place_approach")
            if not success:
                return False
            
            # 2. Descend into container
            drop_pos = self.container_position.copy()
            drop_pos[2] = self.container_dimensions[2] + 0.05  # Just above container bottom
            
            print(f"[PLACE] Descending into container: {drop_pos}")
            success = await self._move_to_position(drop_pos, "place")
            if not success:
                return False
            
            # 3. Open gripper to release
            await self._open_gripper()
            
            # 4. Retract from container
            print(f"[PLACE] Retracting from container")
            success = await self._move_to_position(place_pos, "retract")
            
            return success
            
        except Exception as e:
            carb.log_error(f"[PLACE] Error in place sequence: {e}")
            return False

    async def _move_to_position(self, target_position, phase_name="move"):
        """
        Move end effector to target position using RRT

        Args:
            target_position: Target position [x, y, z]
            phase_name: Name of movement phase for logging

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Update dynamic obstacles before planning (if callback provided)
            if self.obstacle_update_callback is not None:
                self.obstacle_update_callback()

            # Convert Cartesian position to joint positions using IK
            target_joint_positions, success = self.articulation_kinematics_solver.compute_inverse_kinematics(
                target_position=target_position,
                target_orientation=np.array([1.0, 0.0, 0.0, 0.0])  # Default orientation
            )

            if not success:
                carb.log_warn(f"[RRT] IK failed for {phase_name} at {target_position}")
                return False

            # Get current joint positions
            current_joint_positions = self.manipulator.get_joint_positions()

            # Set RRT target and update world
            self.rrt.set_end_effector_target(
                target_position,
                np.array([1.0, 0.0, 0.0, 0.0])  # Default orientation
            )
            self.rrt.update_world()

            # Plan path with RRT using compute_path (correct method)
            path = self.rrt.compute_path(
                current_joint_positions[:7],  # First 7 joints (exclude gripper)
                np.array([])  # Watched joints (empty for Franka)
            )

            if path is None or len(path) == 0:
                carb.log_warn(f"[RRT] Path planning failed for {phase_name}")
                return False

            # Generate smooth trajectory
            trajectory = self.cspace_trajectory_generator.compute_c_space_trajectory(path)

            # Convert to action sequence
            articulation_trajectory = ArticulationTrajectory(
                self.manipulator,
                trajectory,
                1.0 / 60.0  # Physics timestep
            )
            action_sequence = articulation_trajectory.get_action_sequence()

            # Execute action sequence with frame skipping for speed
            skip_factor = 3  # Skip frames for faster motion (3 is optimal for 60 FPS)

            # Debug: Check action sequence length
            if len(action_sequence) == 0:
                carb.log_warn(f"[RRT] Empty action sequence for {phase_name}")
                return False

            print(f"[RRT] Executing {len(action_sequence)} actions (skip_factor={skip_factor}) for {phase_name}")

            executed_count = 0
            for i, action in enumerate(action_sequence):
                if i % skip_factor == 0:  # Skip frames
                    self.manipulator.apply_action(action)
                    await omni.kit.app.get_app().next_update_async()
                    executed_count += 1

            print(f"[RRT] Executed {executed_count}/{len(action_sequence)} actions for {phase_name}")
            return True

        except Exception as e:
            carb.log_error(f"[RRT] Error moving to position: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _open_gripper(self):
        """Open gripper"""
        articulation_controller = self.manipulator.get_articulation_controller()
        open_action = ArticulationAction(
            joint_positions=self.gripper.joint_opened_positions,
            joint_indices=np.array([7, 8])
        )
        articulation_controller.apply_action(open_action)
        await asyncio.sleep(0.5)  # Wait for gripper to open

    async def _close_gripper(self):
        """Close gripper"""
        articulation_controller = self.manipulator.get_articulation_controller()
        close_action = ArticulationAction(
            joint_positions=self.gripper.joint_closed_positions,
            joint_indices=np.array([7, 8])
        )
        articulation_controller.apply_action(close_action)
        await asyncio.sleep(0.5)  # Wait for gripper to close

    async def return_to_home(self):
        """Return robot to home position"""
        print("[RRT] Returning to home position")
        home_joint_positions = np.array([0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75, 0.0, 0.0])

        current_joint_positions = self.manipulator.get_joint_positions()

        # Set C-space target for home position
        self.rrt.set_cspace_target(home_joint_positions[:7])
        self.rrt.update_world()

        # Plan path to home using compute_path
        path = self.rrt.compute_path(
            current_joint_positions[:7],
            np.array([])  # Watched joints (empty for Franka)
        )

        if path is None or len(path) == 0:
            carb.log_warn("[RRT] Failed to plan path to home")
            return False

        # Generate and execute trajectory
        trajectory = self.cspace_trajectory_generator.compute_c_space_trajectory(path)

        # Convert to action sequence
        articulation_trajectory = ArticulationTrajectory(
            self.manipulator,
            trajectory,
            1.0 / 60.0  # Physics timestep
        )
        action_sequence = articulation_trajectory.get_action_sequence()

        # Execute action sequence with frame skipping
        skip_factor = 3
        for i, action in enumerate(action_sequence):
            if i % skip_factor == 0:
                self.manipulator.apply_action(action)
                await omni.kit.app.get_app().next_update_async()

        return True

