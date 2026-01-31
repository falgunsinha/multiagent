"""
Collision Checker Module
Unified obstacle detection and management for RRT path planning.

Features:
- Real-time sensor querying (Lidar + OwlV2)
- Automatic obstacle registration with RRT
- Dynamic RRT updates (add/remove/move obstacles)
- Camera-to-world coordinate transformation
- Uses detected sizes instead of hardcoded dimensions
"""

import numpy as np
from typing import List, Dict, Optional
import carb
from scipy.spatial.transform import Rotation


class CollisionChecker:
    """
    Collision checker wrapper for unified obstacle management.
    
    Combines obstacles from multiple sensors (Lidar, OwlV2) and manages
    dynamic obstacle registration with RRT planner.
    """
    
    def __init__(
        self,
        depth_camera_position: np.ndarray = np.array([1.0, 0.1, 0.4]),
        depth_camera_orientation: np.ndarray = np.array([0.0, 67.0, 90.0]),
        verbose: bool = False
    ):
        """
        Initialize collision checker.
        
        Args:
            depth_camera_position: Depth camera world position
            depth_camera_orientation: Depth camera orientation (roll, pitch, yaw) in degrees
            verbose: Print detailed information
        """
        self.depth_camera_position = depth_camera_position
        self.depth_camera_orientation = depth_camera_orientation
        self.verbose = verbose
        
        # Precompute rotation matrix for camera-to-world transform
        self._compute_camera_transform()
    
    def _compute_camera_transform(self):
        """Precompute camera-to-world rotation matrix"""
        try:
            rotation = Rotation.from_euler(
                'xyz',
                self.depth_camera_orientation,
                degrees=True
            )
            self.camera_rotation_matrix = rotation.as_matrix()

            print(f"\n[COLLISION CHECKER] Camera Transform Initialized:")
            print(f"  Position: {self.depth_camera_position}")
            print(f"  Orientation (XYZ Euler): {self.depth_camera_orientation}")
            print(f"  Rotation Matrix:\n{self.camera_rotation_matrix}")

        except Exception as e:
            carb.log_warn(f"[COLLISION CHECKER] Error computing camera transform: {e}")
            self.camera_rotation_matrix = np.eye(3)
    
    def get_owlv2_obstacles(self, detected_obstacles: List[Dict]) -> List[Dict]:
        """
        Get obstacles detected by OwlV2 3D detection.
        
        Args:
            detected_obstacles: List of OwlV2 detections with 3D info
            
        Returns:
            List of obstacle data with position and size in world coordinates
        """
        try:
            obstacles = []
            
            for obs in detected_obstacles:
                # Check if 3D position and size are available
                if 'position_3d' in obs and obs['position_3d'] is not None:
                    position_3d = obs['position_3d']

                    # Convert from camera frame to world frame
                    world_position = self.camera_to_world_transform_v5(position_3d)
                    
                    # Get size (use detected size if available, otherwise default)
                    if 'size_3d' in obs and obs['size_3d'] is not None:
                        size_3d = obs['size_3d']
                    else:
                        # Default size for obstacles
                        size_3d = np.array([0.15, 0.15, 0.15])
                    
                    obstacles.append({
                        'position': world_position,
                        'size': size_3d,
                        'class': obs.get('class', 'unknown')
                    })
            
            return obstacles
            
        except Exception as e:
            carb.log_warn(f"[COLLISION CHECKER] Error getting OwlV2 obstacles: {e}")
            return []
    
    def camera_to_world_transform_v5(self, camera_position: np.ndarray) -> np.ndarray:
        """
        Transform from USD camera frame to world frame.

        Isaac Sim USD Camera Convention (from official docs):
        - +X: right
        - +Y: up
        - -Z: forward (optical axis)

        World Frame (Isaac Sim default):
        - +X: forward
        - +Y: right
        - +Z: up

        The rotation matrix transforms vectors from USD camera frame to world frame.

        Args:
            camera_position: [x, y, z] in USD camera frame (from unprojection)

        Returns:
            [x, y, z] in world frame
        """
        cam_x, cam_y, cam_z = camera_position

        # Transform from USD camera frame to world frame using rotation matrix
        camera_vec = np.array([cam_x, cam_y, cam_z])
        world_vec = self.camera_rotation_matrix @ camera_vec
        world_position = self.depth_camera_position + world_vec

        # Adjust for cube height (cube center is half-height above table)
        cube_half_height = 0.0258
        world_position[2] += cube_half_height

        print(f"[TRANSFORM V5] USD_camera=[{cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f}]")
        print(f"  world_vec={world_vec}")
        print(f"  world_pos={world_position}")

        return world_position
    
    def merge_obstacles(
        self, 
        lidar_obstacles: List[np.ndarray], 
        owlv2_obstacles: List[Dict]
    ) -> List[Dict]:
        """
        Merge obstacles from Lidar and OwlV2 sensors.
        
        Args:
            lidar_obstacles: List of positions from Lidar
            owlv2_obstacles: List of obstacle dicts from OwlV2
            
        Returns:
            Merged list of obstacle data
        """
        merged = []
        
        # Add Lidar obstacles (positions only, default size)
        for pos in lidar_obstacles:
            merged.append({
                'position': np.array(pos),
                'size': np.array([0.15, 0.15, 0.15]),
                'source': 'lidar'
            })
        
        # Add OwlV2 obstacles (positions + sizes)
        for obs in owlv2_obstacles:
            merged.append({
                'position': obs['position'],
                'size': obs['size'],
                'source': 'owlv2',
                'class': obs.get('class', 'unknown')
            })
        
        return merged

