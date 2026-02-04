import numpy as np
import os
from pathlib import Path
import sys

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

from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.robot_motion.motion_generation import PathPlannerVisualizer
from isaacsim.robot_motion.motion_generation.lula import RRT

class PickPlaceState:
    """State machine states for pick and place"""
    IDLE = 0
    MOVING_TO_PICK = 1
    PICKING = 2
    MOVING_TO_PLACE = 3
    PLACING = 4
    DONE = 5

class RRTController:
    """Manages RRT motion planning and pick/place execution"""

    def __init__(self, franka, gripper, container_info):
        self.franka = franka
        self.gripper = gripper
        self.container_info = container_info

        self.rrt = None
        self.path_planner_visualizer = None

        self.state = PickPlaceState.IDLE
        self.plan = []
        self.gripper_action_counter = 0

        self.ee_offset = np.array([0.0, 0.0, -0.105])

        self.current_cuboid = None

        self.placing_position = None
        self.placed_count = 0
        
    def setup_rrt(self):
        """Setup Lula RRT motion planner"""
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
        rrt_config_dir = os.path.join(mg_extension_path, "path_planner_configs")

        self.rrt = RRT(
            robot_description_path=rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
            urdf_path=rmp_config_dir + "/franka/lula_franka_gen.urdf",
            rrt_config_path=rrt_config_dir + "/franka/rrt/franka_planner_config.yaml",
            end_effector_frame_name="right_gripper"
        )

        self.rrt.set_max_iterations(10000)
        self.path_planner_visualizer = PathPlannerVisualizer(self.franka, self.rrt)

        return self.rrt, self.path_planner_visualizer
    
    def set_target_cuboid(self, cuboid_info):
        """Set the current target cuboid to pick"""
        self.current_cuboid = cuboid_info
        self.state = PickPlaceState.IDLE
        self.plan = []
        self.gripper_action_counter = 0

        cube_half_size = 0.0515 / 2.0
        placing_height = cube_half_size + self.container_info['placing_height'] + (self.placed_count * 0.0515)

        self.placing_position = np.array([
            self.container_info['translation'][0],
            self.container_info['translation'][1],
            placing_height
        ])
    
    def plan_to_target(self, target_position, target_orientation):
        """Plan path to target using RRT"""
        current_joint_positions = self.franka.get_joint_positions()
        arm_joint_positions = current_joint_positions[:7]

        self.rrt.set_end_effector_target(target_position, target_orientation)
        self.rrt.update_world(arm_joint_positions)

        plan = self.path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=0.01)

        return plan
    
    def step(self):
        """Execute one step of the pick and place state machine"""
        if self.current_cuboid is None:
            return False

        if self.state == PickPlaceState.IDLE:
            cube_position, _ = self.current_cuboid['object'].get_world_pose()
            pick_target = cube_position + self.ee_offset
            pick_orientation = euler_angles_to_quats([np.pi, 0, 0])

            self.plan = self.plan_to_target(pick_target, pick_orientation)

            if self.plan:
                self.state = PickPlaceState.MOVING_TO_PICK
            else:
                self.state = PickPlaceState.DONE
                return False

        elif self.state == PickPlaceState.MOVING_TO_PICK:
            if self.plan:
                action = self.plan.pop(0)
                self.franka.apply_action(action)
            else:
                self.gripper.open()
                self.state = PickPlaceState.PICKING
                self.gripper_action_counter = 0

        elif self.state == PickPlaceState.PICKING:
            self.gripper.close()
            self.gripper_action_counter += 1

            if self.gripper_action_counter > 30:
                place_target = self.placing_position + self.ee_offset
                place_orientation = euler_angles_to_quats([np.pi, 0, 0])

                self.plan = self.plan_to_target(place_target, place_orientation)

                if self.plan:
                    self.state = PickPlaceState.MOVING_TO_PLACE
                else:
                    self.state = PickPlaceState.DONE
                    return False

        elif self.state == PickPlaceState.MOVING_TO_PLACE:
            if self.plan:
                action = self.plan.pop(0)
                self.franka.apply_action(action)
            else:
                self.state = PickPlaceState.PLACING
                self.gripper_action_counter = 0

        elif self.state == PickPlaceState.PLACING:
            self.gripper.open()
            self.gripper_action_counter += 1

            if self.gripper_action_counter > 30:
                self.state = PickPlaceState.DONE
                self.placed_count += 1
                return True

        elif self.state == PickPlaceState.DONE:
            return True

        return False
    
    def is_done(self):
        """Check if current pick and place is complete"""
        return self.state == PickPlaceState.DONE

    def reset_for_next_cuboid(self):
        """Reset controller for next cuboid"""
        self.current_cuboid = None
        self.state = PickPlaceState.IDLE
        self.plan = []
        self.gripper_action_counter = 0