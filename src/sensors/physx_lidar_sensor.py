"""
PhysX Lidar Sensor Module
Handles PhysX Lidar point cloud processing and obstacle detection.

Features:
- Real-time point cloud processing
- Obstacle detection via grid-based clustering
- Workspace filtering (height, distance, exclusion zones)
- Obstacle merging to avoid duplicates
"""

import numpy as np
from typing import List, Optional, Tuple
import carb


class PhysXLidarSensor:
    """
    PhysX Lidar sensor wrapper for obstacle detection.
    
    Processes point cloud data from PhysX Lidar to detect obstacles in the workspace.
    """
    
    def __init__(
        self,
        lidar_sensor,
        robot_articulation,
        container_dimensions: Optional[np.ndarray] = None,
        verbose: bool = False
    ):
        """
        Initialize PhysX Lidar sensor.
        
        Args:
            lidar_sensor: Isaac Sim PhysX Lidar sensor instance
            robot_articulation: Robot articulation for position reference
            container_dimensions: Container dimensions for exclusion zone
            verbose: Print detailed detection information
        """
        self.lidar = lidar_sensor
        self.robot = robot_articulation
        self.container_dimensions = container_dimensions
        self.verbose = verbose
        self._debug_printed = False
        
        # Workspace filtering parameters
        self.height_min = 0.05  # 5cm minimum height
        self.height_max = 0.40  # 40cm maximum height
        self.distance_min = 0.30  # 30cm minimum distance from robot
        self.distance_max = 0.90  # 90cm maximum distance from robot
        
        # Exclusion zones
        self.cube_grid_center = np.array([0.45, -0.10])
        self.cube_grid_margin = 0.30
        self.container_pos = np.array([0.30, 0.50, 0.0])
        self.container_margin = 0.08
        self.robot_base_pos = np.array([0.0, 0.0])
        self.robot_arm_radius = 0.55
        
        # Clustering parameters
        self.grid_size = 0.1  # 10cm grid cells
        self.min_points_per_cell = 5
        self.merge_distance_xy = 0.25  # 25cm merge threshold
        
    def process_point_cloud(self) -> List[np.ndarray]:
        """
        Process PhysX Lidar point cloud data to detect obstacles.
        
        Returns:
            List of detected obstacle positions in world coordinates
        """
        if self.lidar is None:
            return []
        
        try:
            # Get current frame data from PhysX Lidar
            lidar_data = self.lidar.get_current_frame()
            
            if lidar_data is None or "point_cloud" not in lidar_data:
                return []
            
            point_cloud_data = lidar_data["point_cloud"]
            
            if point_cloud_data is None:
                return []
            
            # Convert to numpy array
            points = self._convert_to_numpy(point_cloud_data)
            
            if points is None or len(points) == 0:
                return []
            
            # Validate and fix shape
            points = self._validate_point_cloud_shape(points)
            
            if points is None:
                return []
            
            # Transform to world coordinates
            points_world = self._transform_to_world_coordinates(points)
            
            # Apply workspace filters
            valid_points = self._apply_workspace_filters(points_world)
            
            if len(valid_points) < 10:
                return []
            
            # Detect obstacles via clustering
            detected_obstacles = self._detect_obstacles_clustering(valid_points)
            
            # Log detection results
            if self.verbose and len(detected_obstacles) > 0:
                self._log_detection_results(detected_obstacles, valid_points)
            
            return detected_obstacles
            
        except Exception as e:
            carb.log_warn(f"[PHYSX LIDAR] Error processing point cloud: {e}")
            return []
    
    def _convert_to_numpy(self, point_cloud_data) -> Optional[np.ndarray]:
        """Convert point cloud data to numpy array"""
        # Handle tensor data (convert to numpy)
        if hasattr(point_cloud_data, 'cpu'):
            return point_cloud_data.cpu().numpy()
        elif hasattr(point_cloud_data, 'numpy'):
            return point_cloud_data.numpy()
        elif isinstance(point_cloud_data, np.ndarray):
            return point_cloud_data
        else:
            return None
    
    def _validate_point_cloud_shape(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Validate and fix point cloud shape"""
        if points.ndim == 1:
            if len(points) == 3:
                return None  # Single point - skip
            if len(points) % 3 == 0:
                points = points.reshape(-1, 3)
            else:
                return None
        elif points.ndim != 2 or points.shape[1] != 3:
            return None

        return points

    def _transform_to_world_coordinates(self, points: np.ndarray) -> np.ndarray:
        """Transform points from sensor-local to world coordinates"""
        from scipy.spatial.transform import Rotation as R

        lidar_world_pos, lidar_world_rot = self.lidar.get_world_pose()

        # Convert quaternion (w, x, y, z) to (x, y, z, w) for scipy
        rot_matrix = R.from_quat([
            lidar_world_rot[1],
            lidar_world_rot[2],
            lidar_world_rot[3],
            lidar_world_rot[0]
        ]).as_matrix()

        # Transform all points to world coordinates
        points_world = (rot_matrix @ points.T).T + lidar_world_pos

        return points_world

    def _apply_workspace_filters(self, points_world: np.ndarray) -> np.ndarray:
        """Apply workspace filters to remove invalid points"""
        # Filter by height (world coordinates)
        valid_points = points_world[
            (points_world[:, 2] > self.height_min) &
            (points_world[:, 2] < self.height_max)
        ]

        # Filter by distance from robot base
        robot_pos, _ = self.robot.get_world_pose()
        distances_from_robot = np.linalg.norm(valid_points[:, :2] - robot_pos[:2], axis=1)
        valid_points = valid_points[
            (distances_from_robot > self.distance_min) &
            (distances_from_robot < self.distance_max)
        ]

        # Filter out cube pickup region
        cube_region_mask = ~(
            (np.abs(valid_points[:, 0] - self.cube_grid_center[0]) < self.cube_grid_margin) &
            (np.abs(valid_points[:, 1] - self.cube_grid_center[1]) < self.cube_grid_margin)
        )
        valid_points = valid_points[cube_region_mask]

        # Filter out container/placement region
        if self.container_dimensions is not None:
            container_half_dims = self.container_dimensions / 2.0
            container_region_mask = ~(
                (np.abs(valid_points[:, 0] - self.container_pos[0]) <
                 (container_half_dims[0] + self.container_margin)) &
                (np.abs(valid_points[:, 1] - self.container_pos[1]) <
                 (container_half_dims[1] + self.container_margin))
            )
            valid_points = valid_points[container_region_mask]

        # Filter out robot base and arm region
        robot_region_mask = np.linalg.norm(
            valid_points[:, :2] - self.robot_base_pos, axis=1
        ) > self.robot_arm_radius
        valid_points = valid_points[robot_region_mask]

        return valid_points

    def _detect_obstacles_clustering(self, valid_points: np.ndarray) -> List[np.ndarray]:
        """Detect obstacles using grid-based clustering"""
        # Grid-based clustering
        grid_points = np.round(valid_points / self.grid_size) * self.grid_size
        unique_cells, counts = np.unique(grid_points, axis=0, return_counts=True)
        obstacle_cells = unique_cells[counts > self.min_points_per_cell]
        detected_obstacles = obstacle_cells.tolist()

        # Merge nearby obstacles
        if len(detected_obstacles) > 1:
            detected_obstacles = self._merge_nearby_obstacles(detected_obstacles)

        return detected_obstacles

    def _merge_nearby_obstacles(self, obstacles: List) -> List[np.ndarray]:
        """Merge nearby obstacles to avoid duplicates"""
        merged_obstacles = []
        used = set()

        for i, obs1 in enumerate(obstacles):
            if i in used:
                continue
            cluster = [obs1]
            for j, obs2 in enumerate(obstacles[i+1:], start=i+1):
                if j in used:
                    continue
                # Only check XY distance (ignore Z)
                dist_xy = np.linalg.norm(np.array(obs1[:2]) - np.array(obs2[:2]))
                if dist_xy < self.merge_distance_xy:
                    cluster.append(obs2)
                    used.add(j)

            # Use center position (average XY) and lowest Z
            cluster_array = np.array(cluster)
            merged_pos = np.array([
                np.mean(cluster_array[:, 0]),  # Average X (center)
                np.mean(cluster_array[:, 1]),  # Average Y (center)
                np.min(cluster_array[:, 2])    # Lowest Z (base)
            ])
            merged_obstacles.append(merged_pos)

        return merged_obstacles

    def _log_detection_results(self, detected_obstacles: List, valid_points: np.ndarray):
        """Log detailed detection results"""
        from omni.isaac.core.utils.stage import get_current_stage
        from omni.isaac.core.prims import XFormPrim
        from pxr import UsdPhysics

        print(f"\n[LIDAR] PhysX Lidar - Rotating Detection Report:")
        print(f"[LIDAR] Total point cloud points: {len(valid_points)}")
        print(f"[LIDAR] Detected obstacles: {len(detected_obstacles)}")
        print(f"[LIDAR] ----------------------------------------")

        stage = get_current_stage()

        for i, obs_pos in enumerate(detected_obstacles):
            obs_name = "Unknown"
            obs_type = "Unknown"
            obs_dimensions = "Unknown"
            point_count = 0

            # Check all prims under /World for obstacles
            for prim in stage.Traverse():
                prim_path = str(prim.GetPath())
                if "/World/Obstacle_" in prim_path or "/World/LidarObstacle_" in prim_path:
                    try:
                        xform = XFormPrim(prim_path)
                        prim_pos, _ = xform.get_world_pose()
                        dist_xy = np.linalg.norm(np.array(obs_pos[:2]) - prim_pos[:2])

                        if dist_xy < 0.20:  # Within 20cm in XY
                            obs_name = prim_path.split('/')[-1]

                            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                                obs_type = "DynamicCuboid (Rigid Body)"
                            else:
                                obs_type = "FixedCuboid (Static)"

                            if prim.GetAttribute("xformOp:scale"):
                                scale = prim.GetAttribute("xformOp:scale").Get()
                                obs_dimensions = f"{scale[0]:.2f}m x {scale[1]:.2f}m x {scale[2]:.2f}m"

                            # Count points near this obstacle
                            for pt in valid_points:
                                pt_dist = np.linalg.norm(np.array(pt[:2]) - prim_pos[:2])
                                if pt_dist < 0.20:
                                    point_count += 1
                            break
                    except:
                        pass

            print(f"[LIDAR] Obstacle #{i+1}:")
            print(f"[LIDAR]   Name: {obs_name}")
            print(f"[LIDAR]   Type: {obs_type}")
            print(f"[LIDAR]   Dimensions: {obs_dimensions}")
            print(f"[LIDAR]   Position: ({obs_pos[0]:.3f}, {obs_pos[1]:.3f}, {obs_pos[2]:.3f})m")
            print(f"[LIDAR]   Point cloud hits: {point_count} points")

