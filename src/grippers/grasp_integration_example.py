"""
Example integration of GraspConfig with Franka pick-and-place scripts.
This shows how to use isaac_grasp files to improve grasping reliability.

Usage in your main script:
1. Import GraspConfig
2. Load the grasp file during initialization
3. Use grasp data to adjust gripper approach and joint positions
"""

import os
import numpy as np
from grasp_config import GraspConfig


class GraspIntegrationExample:
    """
    Example class showing how to integrate grasp configurations
    into a pick-and-place workflow.
    """
    
    def __init__(self):
        # Path to the grasp configuration file
        grasp_file = os.path.join(
            os.path.dirname(__file__), 
            "franka_cylinder_grasp.yaml"
        )
        
        # Load grasp configuration
        self.grasp_config = GraspConfig(grasp_file)
        
        # Select default grasp (can be changed based on conditions)
        self.current_grasp_name = "grasp_0"
        
        if self.grasp_config.is_loaded():
            print(f"Loaded grasps: {self.grasp_config.get_available_grasps()}")
            print(f"Using grasp: {self.current_grasp_name}")
            print(f"Confidence: {self.grasp_config.get_grasp_confidence(self.current_grasp_name)}")
    
    def get_gripper_target_pose(self, cylinder_pos, cylinder_orient):
        """
        Compute the target gripper pose for grasping a cylinder.
        
        Args:
            cylinder_pos: Cylinder position [x, y, z]
            cylinder_orient: Cylinder orientation quaternion [w, x, y, z]
            
        Returns:
            Tuple of (gripper_position, gripper_orientation)
        """
        if not self.grasp_config.is_loaded():
            # Fallback to default behavior if grasp file not loaded
            return cylinder_pos, cylinder_orient
        
        # Compute gripper pose from grasp configuration
        gripper_pos, gripper_orient = self.grasp_config.compute_gripper_pose(
            self.current_grasp_name,
            cylinder_pos,
            cylinder_orient
        )
        
        return gripper_pos, gripper_orient
    
    def get_gripper_close_positions(self):
        """
        Get the joint positions for closing the gripper on the object.
        
        Returns:
            Dictionary of joint positions or None
        """
        if not self.grasp_config.is_loaded():
            return None
        
        return self.grasp_config.get_gripper_joint_positions(self.current_grasp_name)
    
    def get_gripper_open_positions(self):
        """
        Get the joint positions for opening the gripper (pre-grasp).
        
        Returns:
            Dictionary of joint positions or None
        """
        if not self.grasp_config.is_loaded():
            return None
        
        return self.grasp_config.get_pregrasp_joint_positions(self.current_grasp_name)
    
    def select_best_grasp(self, obstacle_positions=None):
        """
        Select the best grasp based on current conditions.
        This can be extended to consider obstacles, object orientation, etc.
        
        Args:
            obstacle_positions: List of obstacle positions (optional)
            
        Returns:
            Name of the selected grasp
        """
        if not self.grasp_config.is_loaded():
            return None
        
        available_grasps = self.grasp_config.get_available_grasps()
        
        # Simple selection: choose highest confidence grasp
        # Can be extended to check collision with obstacles
        best_grasp = None
        best_confidence = 0.0
        
        for grasp_name in available_grasps:
            confidence = self.grasp_config.get_grasp_confidence(grasp_name)
            if confidence > best_confidence:
                best_confidence = confidence
                best_grasp = grasp_name
        
        self.current_grasp_name = best_grasp
        return best_grasp


# Example usage in a pick-and-place script:
"""
# In your main script initialization:
from src.grippers import GraspConfig
import os

# Load grasp configuration
grasp_file = os.path.join("C:/isaacsim/cobotproject/src/grippers", "franka_cylinder_grasp.yaml")
self.grasp_config = GraspConfig(grasp_file)

# In your pick function, before closing gripper:
if self.grasp_config.is_loaded():
    # Get optimal gripper joint positions from grasp file
    grasp_positions = self.grasp_config.get_gripper_joint_positions("grasp_0")
    if grasp_positions:
        # Use these positions instead of default gripper.joint_closed_positions
        joint_positions = [
            grasp_positions.get("panda_finger_joint1", 0.037),
            grasp_positions.get("panda_finger_joint2", 0.037)
        ]
        articulation_controller.apply_action(
            ArticulationAction(joint_positions=joint_positions, joint_indices=np.array([7, 8]))
        )
"""

