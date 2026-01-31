"""
Robot Configuration Loader (cuRobo-style)

Loads robot configuration from YML file (like cuRobo does):
1. Reads YML configuration
2. Loads URDF path from YML
3. Loads collision sphere definitions
4. Provides configuration to batched IK solver

This matches cuRobo's RobotConfig.from_dict() flow.
"""

import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class RobotConfigLoader:
    """
    Loads robot configuration from YML file (cuRobo-style)
    
    Mimics cuRobo's flow:
    YML → RobotConfig → CudaRobotModelConfig → CudaRobotGenerator
    """
    
    def __init__(self, yml_path: str, device: str = "cuda:0"):
        """
        Load robot configuration from YML file
        
        Args:
            yml_path: Path to robot YML file (e.g., franka_curobo.yml)
            device: PyTorch device
        """
        self.yml_path = Path(yml_path)
        self.device = device
        
        # Load YML configuration
        with open(yml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract robot_cfg section
        if "robot_cfg" in config:
            config = config["robot_cfg"]
        
        if "kinematics" not in config:
            raise ValueError(f"YML file must contain 'kinematics' section: {yml_path}")
        
        self.kinematics_config = config["kinematics"]
        
        # Extract key parameters
        self.urdf_path = self._resolve_path(self.kinematics_config.get("urdf_path"))
        self.base_link = self.kinematics_config.get("base_link", "panda_link0")
        self.ee_link = self.kinematics_config.get("ee_link", "panda_hand")
        
        # Collision parameters
        self.collision_spheres_file = self.kinematics_config.get("collision_spheres")
        self.collision_sphere_buffer = self.kinematics_config.get("collision_sphere_buffer", 0.004)
        self.collision_link_names = self.kinematics_config.get("collision_link_names", [])
        self.self_collision_ignore = self.kinematics_config.get("self_collision_ignore", {})
        self.self_collision_buffer = self.kinematics_config.get("self_collision_buffer", {})
        
        # Load collision spheres
        self.collision_spheres = self._load_collision_spheres()
        
        print(f"[RobotConfig] Loaded from: {yml_path}")
        print(f"[RobotConfig] URDF: {self.urdf_path}")
        print(f"[RobotConfig] Base link: {self.base_link}, EE link: {self.ee_link}")
        print(f"[RobotConfig] Collision spheres: {len(self.collision_spheres)} links")
        print(f"[RobotConfig] Collision buffer: {self.collision_sphere_buffer*1000:.1f}mm")
    
    def _resolve_path(self, path: str) -> str:
        """Resolve relative path from YML file location"""
        if path is None:
            return None
        
        path = Path(path)
        if path.is_absolute():
            return str(path)
        
        # Resolve relative to YML file's directory
        yml_dir = self.yml_path.parent
        resolved = yml_dir / path
        
        if not resolved.exists():
            # Try relative to assets directory
            assets_dir = yml_dir
            resolved = assets_dir / path
        
        return str(resolved)
    
    def _load_collision_spheres(self) -> Dict[str, List[Dict]]:
        """
        Load collision sphere definitions from YML file
        
        Returns:
            Dictionary mapping link names to list of spheres
            Each sphere: {"center": [x, y, z], "radius": r}
        """
        if self.collision_spheres_file is None:
            print("[RobotConfig] No collision spheres file specified")
            return {}
        
        spheres_path = self._resolve_path(self.collision_spheres_file)
        
        if not Path(spheres_path).exists():
            print(f"[RobotConfig] WARNING: Collision spheres file not found: {spheres_path}")
            return {}
        
        with open(spheres_path, 'r') as f:
            spheres_config = yaml.safe_load(f)
        
        if "collision_spheres" not in spheres_config:
            print(f"[RobotConfig] WARNING: No 'collision_spheres' section in {spheres_path}")
            return {}
        
        collision_spheres = spheres_config["collision_spheres"]
        
        # Count total spheres
        total_spheres = sum(len(spheres) for spheres in collision_spheres.values())
        print(f"[RobotConfig] Loaded {total_spheres} collision spheres across {len(collision_spheres)} links")
        
        return collision_spheres
    
    def get_link_spheres(self, link_name: str) -> List[Dict]:
        """
        Get collision spheres for a specific link
        
        Args:
            link_name: Name of the link
            
        Returns:
            List of spheres: [{"center": [x, y, z], "radius": r}, ...]
        """
        return self.collision_spheres.get(link_name, [])
    
    def should_check_self_collision(self, link1: str, link2: str) -> bool:
        """
        Check if two links should be checked for self-collision
        
        Args:
            link1, link2: Names of the links
            
        Returns:
            True if collision should be checked, False if ignored
        """
        # Check if link2 is in link1's ignore list
        if link1 in self.self_collision_ignore:
            if link2 in self.self_collision_ignore[link1]:
                return False
        
        # Check if link1 is in link2's ignore list
        if link2 in self.self_collision_ignore:
            if link1 in self.self_collision_ignore[link2]:
                return False
        
        return True

