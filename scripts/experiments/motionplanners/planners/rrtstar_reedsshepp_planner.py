"""
RRT* with Reeds-Shepp Path Motion Planner Wrapper

Wraps PythonRobotics RRT* with Reeds-Shepp path for car-like robots.
Reeds-Shepp paths are optimal paths for vehicles that can move forward and backward.
"""

import sys
from pathlib import Path
import numpy as np
import time
from typing import Tuple, List, Optional, Dict

# Add project root to path
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.experiments.motionplanners.base_planner import BaseMotionPlanner, PlannerMetrics


class RRTStarReedsSheppPlanner(BaseMotionPlanner):
    """
    Wrapper for PythonRobotics RRT* with Reeds-Shepp path planner.
    
    This planner is designed for car-like robots and uses Reeds-Shepp curves
    for optimal path generation considering vehicle kinematics.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize RRT* Reeds-Shepp planner.

        Args:
            config: Configuration dict with keys:
                - max_iter: Maximum iterations (default: 500)
                - step_size: Step size for steering (default: 0.2)
                - connect_circle_dist: Connection circle distance (default: 50.0)
                - robot_radius: Robot radius for collision checking (default: 0.05)
        """
        super().__init__("RRT_Star_ReedsShepp", config)
        
        # Import PythonRobotics RRT* Reeds-Shepp
        # Add PythonRobotics root to path (needed for utils.angle import)
        pythonrobotics_root = project_root / "PythonRobotics"
        pythonrobotics_root_str = str(pythonrobotics_root)

        # Always insert at the beginning to ensure it's found first
        if pythonrobotics_root_str in sys.path:
            sys.path.remove(pythonrobotics_root_str)
        sys.path.insert(0, pythonrobotics_root_str)

        rrtstar_rs_path = project_root / "PythonRobotics" / "PathPlanning" / "RRTStarReedsShepp"
        rrtstar_rs_path_str = str(rrtstar_rs_path)

        if rrtstar_rs_path_str in sys.path:
            sys.path.remove(rrtstar_rs_path_str)
        sys.path.insert(0, rrtstar_rs_path_str)

        try:
            from rrt_star_reeds_shepp import RRTStarReedsShepp
            self.RRTStarReedsSheppClass = RRTStarReedsShepp
        except ImportError as e:
            print(f"[RRT*-RS ERROR] Failed to import: {e}")
            print(f"[RRT*-RS ERROR] RRTStarReedsShepp Path: {rrtstar_rs_path_str}")
            print(f"[RRT*-RS ERROR] PythonRobotics Root: {pythonrobotics_root_str}")
            print(f"[RRT*-RS ERROR] sys.path[0:3]: {sys.path[0:3]}")
            raise
        
    def plan(self, 
             start_pos: np.ndarray, 
             goal_pos: np.ndarray,
             obstacles: Optional[List] = None) -> Tuple[Optional[np.ndarray], PlannerMetrics]:
        """
        Plan path using RRT* with Reeds-Shepp curves.
        
        Args:
            start_pos: Start position [x, y] or [x, y, yaw]
            goal_pos: Goal position [x, y] or [x, y, yaw]
            obstacles: List of obstacles [[x, y, radius], ...]
            
        Returns:
            path: 2D path (N x 2) or None
            metrics: Planning metrics
        """
        metrics = PlannerMetrics()
        
        # Get config parameters (matching actual PythonRobotics RRTStarReedsShepp API)
        max_iter = self.config.get('max_iter', 500)
        step_size = self.config.get('step_size', 0.2)
        connect_circle_dist = self.config.get('connect_circle_dist', 50.0)
        robot_radius = self.config.get('robot_radius', 0.05)
        
        # Add default yaw if not provided
        if len(start_pos) == 2:
            start = [start_pos[0], start_pos[1], 0.0]
        else:
            start = start_pos[:3].tolist()
        
        if len(goal_pos) == 2:
            goal = [goal_pos[0], goal_pos[1], 0.0]
        else:
            goal = goal_pos[:3].tolist()
        
        # Determine search area
        rand_area = self.config.get('rand_area', [-2, 2])
        
        # Start timing
        start_time = time.time()
        
        try:
            # Create RRT* Reeds-Shepp instance with correct parameters
            rrt_star_rs = self.RRTStarReedsSheppClass(
                start=start,
                goal=goal,
                obstacle_list=obstacles or [],
                rand_area=rand_area,
                max_iter=max_iter,
                step_size=step_size,
                connect_circle_dist=connect_circle_dist,
                robot_radius=robot_radius
            )
            
            # Plan path
            path = rrt_star_rs.planning(animation=False)
            
            # Stop timing
            metrics.search_time = time.time() - start_time

            if path is None:
                metrics.success = False
                return None, metrics
            
            # Convert path to numpy array (extract x, y only)
            path_array = np.array(path)
            if path_array.shape[1] >= 2:
                path_array = path_array[:, :2]  # Take only x, y
            
            # Success
            metrics.success = True
            metrics.num_waypoints = len(path_array)
            metrics.path_length = self.calculate_path_length(path_array)
            metrics.smoothness = self.calculate_path_smoothness(path_array)
            metrics.energy = self.calculate_path_energy(path_array)
            
            return path_array, metrics
            
        except Exception as e:
            print(f"[RRT*-RS ERROR] Planning failed: {e}")
            import traceback
            traceback.print_exc()
            metrics.search_time = time.time() - start_time
            metrics.success = False
            return None, metrics
    
    def reset(self):
        """Reset RRT* Reeds-Shepp planner"""
        # Stateless, no reset needed
        pass

