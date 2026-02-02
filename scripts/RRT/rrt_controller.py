import numpy as np
import os
from pathlib import Path
import sys

# Add project root to path (go up 2 levels from RRT folder to cobotproject)
project_root = None

# Method 1: Try __file__ if available
try:
    if '__file__' in globals():
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # RRT -> scripts -> cobotproject
except:
    pass

# Method 2: Use hardcoded path as fallback
if project_root is None:
    project_root = Path(r"C:\isaacsim\cobotproject")

# Method 3: Try from current working directory
if not project_root.exists():
    try:
        project_root = Path(os.getcwd()) / "cobotproject"
        if not project_root.exists():
            project_root = Path(r"C:\isaacsim\cobotproject")
    except:
        project_root = Path(r"C:\isaacsim\cobotproject")

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
        
        # RRT components
        self.rrt = None
        self.path_planner_visualizer = None
        
        # State machine
        self.state = PickPlaceState.IDLE
        self.plan = []
        self.gripper_action_counter = 0
        
        # End effector offset (gripper is below the end effector frame)
        self.ee_offset = np.array([0.0, 0.0, -0.105])
        
        # Current target cuboid
        self.current_cuboid = None
        
        # Placing position
        self.placing_position = None
        self.placed_count = 0
        
    def setup_rrt(self):
        """Setup Lula RRT motion planner"""
        print("\n=== Setting up RRT motion planner ===")
        
        # Get motion generation extension path
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
        rrt_config_dir = os.path.join(mg_extension_path, "path_planner_configs")
        
        print(f"Loading RRT config from: {rrt_config_dir}/franka/rrt/")
        
        # Initialize RRT
        self.rrt = RRT(
            robot_description_path=rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
            urdf_path=rmp_config_dir + "/franka/lula_franka_gen.urdf",
            rrt_config_path=rrt_config_dir + "/franka/rrt/franka_planner_config.yaml",
            end_effector_frame_name="right_gripper"
        )
        
        # Set max iterations
        self.rrt.set_max_iterations(10000)
        
        # Create path planner visualizer
        self.path_planner_visualizer = PathPlannerVisualizer(self.franka, self.rrt)
        
        print("RRT motion planner initialized successfully")
        print(f"RRT max iterations: 10000\n")
        
        return self.rrt, self.path_planner_visualizer
    
    def set_target_cuboid(self, cuboid_info):
        """Set the current target cuboid to pick"""
        self.current_cuboid = cuboid_info
        self.state = PickPlaceState.IDLE
        self.plan = []
        self.gripper_action_counter = 0
        
        # Calculate placing position in container
        # Stack cuboids in container with small offset
        cube_half_size = 0.0515 / 2.0
        placing_height = cube_half_size + self.container_info['placing_height'] + (self.placed_count * 0.0515)
        
        self.placing_position = np.array([
            self.container_info['translation'][0],
            self.container_info['translation'][1],
            placing_height
        ])
        
        print(f"\n=== New Target Set ===")
        print(f"Target: {cuboid_info['name']}")
        print(f"Placing position: {self.placing_position}")
    
    def plan_to_target(self, target_position, target_orientation):
        """Plan path to target using RRT"""
        # Get current arm joint positions (7 DOF only, exclude gripper)
        current_joint_positions = self.franka.get_joint_positions()
        arm_joint_positions = current_joint_positions[:7]
        
        print(f"Current arm joints (7 DOF): {arm_joint_positions}")
        print(f"Target position: {target_position}")
        print(f"Planning path with RRT...")
        
        # Set target and update world state
        self.rrt.set_end_effector_target(target_position, target_orientation)
        self.rrt.update_world(arm_joint_positions)
        
        # Compute plan
        plan = self.path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=0.01)
        
        if plan:
            print(f"✓ Plan computed successfully with {len(plan)} waypoints")
            return plan
        else:
            print("✗ Failed to compute plan")
            return None
    
    def step(self):
        """Execute one step of the pick and place state machine"""
        if self.current_cuboid is None:
            return False  # No target set
        
        if self.state == PickPlaceState.IDLE:
            print("\nState: IDLE -> Planning path to pick position...")
            
            # Get cube position
            cube_position, _ = self.current_cuboid['object'].get_world_pose()
            pick_target = cube_position + self.ee_offset
            pick_orientation = euler_angles_to_quats([np.pi, 0, 0])  # Gripper pointing down
            
            # Plan path to pick
            self.plan = self.plan_to_target(pick_target, pick_orientation)
            
            if self.plan:
                self.state = PickPlaceState.MOVING_TO_PICK
            else:
                print("Failed to plan to pick position!")
                self.state = PickPlaceState.DONE
                return False
        
        elif self.state == PickPlaceState.MOVING_TO_PICK:
            if self.plan:
                action = self.plan.pop(0)
                self.franka.apply_action(action)
            else:
                print("State: MOVING_TO_PICK -> PICKING")
                self.gripper.open()
                self.state = PickPlaceState.PICKING
                self.gripper_action_counter = 0
        
        elif self.state == PickPlaceState.PICKING:
            self.gripper.close()
            self.gripper_action_counter += 1
            
            if self.gripper_action_counter > 30:  # Wait for gripper to close
                print("State: PICKING -> Planning path to place position...")
                
                place_target = self.placing_position + self.ee_offset
                place_orientation = euler_angles_to_quats([np.pi, 0, 0])
                
                # Plan path to place
                self.plan = self.plan_to_target(place_target, place_orientation)
                
                if self.plan:
                    self.state = PickPlaceState.MOVING_TO_PLACE
                else:
                    print("Failed to plan to place position!")
                    self.state = PickPlaceState.DONE
                    return False
        
        elif self.state == PickPlaceState.MOVING_TO_PLACE:
            if self.plan:
                action = self.plan.pop(0)
                self.franka.apply_action(action)
            else:
                print("State: MOVING_TO_PLACE -> PLACING")
                self.state = PickPlaceState.PLACING
                self.gripper_action_counter = 0
        
        elif self.state == PickPlaceState.PLACING:
            self.gripper.open()
            self.gripper_action_counter += 1
            
            if self.gripper_action_counter > 30:  # Wait for gripper to open
                print("State: PLACING -> DONE")
                self.state = PickPlaceState.DONE
                self.placed_count += 1
                return True  # Pick and place complete
        
        elif self.state == PickPlaceState.DONE:
            return True  # Already done
        
        return False  # Still in progress
    
    def is_done(self):
        """Check if current pick and place is complete"""
        return self.state == PickPlaceState.DONE
    
    def reset_for_next_cuboid(self):
        """Reset controller for next cuboid"""
        self.current_cuboid = None
        self.state = PickPlaceState.IDLE
        self.plan = []
        self.gripper_action_counter = 0

