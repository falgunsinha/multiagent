"""
Grasp configuration module for loading and applying isaac_grasp files.
This module provides utilities to load grasp configurations and apply them
to improve grasping reliability for different object types.
"""

import os
import numpy as np
from typing import Optional, Tuple, Dict
import carb


class GraspConfig:
    """
    Manages grasp configurations from isaac_grasp YAML files.
    Provides methods to load grasps and compute gripper poses for objects.
    """
    
    def __init__(self, grasp_file_path: str):
        """
        Initialize grasp configuration from an isaac_grasp YAML file.
        
        Args:
            grasp_file_path: Path to the isaac_grasp YAML file
        """
        self.grasp_file_path = grasp_file_path
        self.grasp_spec = None
        self._load_grasp_file()
    
    def _load_grasp_file(self):
        """Load the grasp file using Isaac Sim's grasp editor API."""
        try:
            from isaacsim.robot_setup.grasp_editor import import_grasps_from_file
            
            if not os.path.exists(self.grasp_file_path):
                carb.log_warn(f"Grasp file not found: {self.grasp_file_path}")
                return
            
            self.grasp_spec = import_grasps_from_file(self.grasp_file_path)
            grasp_names = self.grasp_spec.get_grasp_names()
            carb.log_info(f"Loaded {len(grasp_names)} grasps from {self.grasp_file_path}")
            carb.log_info(f"Available grasps: {grasp_names}")
            
        except Exception as e:
            carb.log_error(f"Failed to load grasp file: {e}")
            self.grasp_spec = None
    
    def is_loaded(self) -> bool:
        """Check if grasp configuration is successfully loaded."""
        return self.grasp_spec is not None
    
    def get_available_grasps(self) -> list:
        """Get list of available grasp names."""
        if not self.is_loaded():
            return []
        return self.grasp_spec.get_grasp_names()
    
    def get_grasp_data(self, grasp_name: str) -> Optional[Dict]:
        """
        Get complete grasp data for a specific grasp.
        
        Args:
            grasp_name: Name of the grasp to retrieve
            
        Returns:
            Dictionary containing grasp data or None if not found
        """
        if not self.is_loaded():
            return None
        return self.grasp_spec.get_grasp_dict_by_name(grasp_name)
    
    def compute_gripper_pose(
        self, 
        grasp_name: str, 
        object_translation: np.ndarray, 
        object_orientation: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute the gripper pose needed to execute a specific grasp.
        
        Args:
            grasp_name: Name of the grasp to use
            object_translation: Object position in world frame [x, y, z]
            object_orientation: Object orientation as quaternion [w, x, y, z]
            
        Returns:
            Tuple of (gripper_translation, gripper_orientation) or (None, None) if failed
        """
        if not self.is_loaded():
            carb.log_warn("Grasp configuration not loaded")
            return None, None
        
        try:
            gripper_trans, gripper_quat = self.grasp_spec.compute_gripper_pose_from_rigid_body_pose(
                grasp_name, object_translation, object_orientation
            )
            return gripper_trans, gripper_quat
        except Exception as e:
            carb.log_error(f"Failed to compute gripper pose: {e}")
            return None, None
    
    def get_gripper_joint_positions(self, grasp_name: str) -> Optional[Dict]:
        """
        Get the gripper joint positions for grasping.
        
        Args:
            grasp_name: Name of the grasp
            
        Returns:
            Dictionary mapping joint names to positions, or None if not found
        """
        grasp_data = self.get_grasp_data(grasp_name)
        if grasp_data is None:
            return None
        return grasp_data.get("cspace_position", None)
    
    def get_pregrasp_joint_positions(self, grasp_name: str) -> Optional[Dict]:
        """
        Get the gripper joint positions for pre-grasp (open position).
        
        Args:
            grasp_name: Name of the grasp
            
        Returns:
            Dictionary mapping joint names to positions, or None if not found
        """
        grasp_data = self.get_grasp_data(grasp_name)
        if grasp_data is None:
            return None
        return grasp_data.get("pregrasp_cspace_position", None)
    
    def get_grasp_confidence(self, grasp_name: str) -> float:
        """
        Get the confidence score for a specific grasp.
        
        Args:
            grasp_name: Name of the grasp
            
        Returns:
            Confidence value between 0.0 and 1.0, or 0.0 if not found
        """
        grasp_data = self.get_grasp_data(grasp_name)
        if grasp_data is None:
            return 0.0
        return grasp_data.get("confidence", 0.0)

