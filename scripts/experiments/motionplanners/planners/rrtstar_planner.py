"""
RRT* Motion Planner Wrapper for Experimental Comparison

Wraps PythonRobotics RRT* implementation for optimal path planning.
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


class RRTStarPlanner(BaseMotionPlanner):
    """
    Wrapper for PythonRobotics RRT* planner.
    
    This uses the optimal RRT* implementation from PythonRobotics.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize RRT* planner.
        
        Args:
            config: Configuration dict with keys:
                - expand_dis: Expansion distance (default: 0.1)
                - path_resolution: Path resolution (default: 0.05)
                - goal_sample_rate: Goal sampling rate % (default: 20)
                - max_iter: Maximum iterations (default: 500)
                - robot_radius: Robot radius for collision checking (default: 0.05)
                - connect_circle_dist: Connection circle distance (default: 0.5)
        """
        super().__init__("RRT_Star", config)
        
        # Import PythonRobotics RRT*
        rrtstar_path = project_root / "PythonRobotics" / "PathPlanning" / "RRTStar"
        if str(rrtstar_path) not in sys.path:
            sys.path.insert(0, str(rrtstar_path))
        from rrt_star import RRTStar
        self.RRTStarClass = RRTStar
        
    def plan(self, 
             start_pos: np.ndarray, 
             goal_pos: np.ndarray,
             obstacles: Optional[List] = None) -> Tuple[Optional[np.ndarray], PlannerMetrics]:
        """
        Plan path using RRT*.
        
        Args:
            start_pos: Start position [x, y]
            goal_pos: Goal position [x, y]
            obstacles: List of obstacles [[x, y, radius], ...]
            
        Returns:
            path: 2D path (N x 2) or None
            metrics: Planning metrics
        """
        metrics = PlannerMetrics()
        
        # Get config parameters
        expand_dis = self.config.get('expand_dis', 0.1)
        path_resolution = self.config.get('path_resolution', 0.05)
        goal_sample_rate = self.config.get('goal_sample_rate', 20)
        max_iter = self.config.get('max_iter', 500)
        robot_radius = self.config.get('robot_radius', 0.05)
        connect_circle_dist = self.config.get('connect_circle_dist', 0.5)
        
        # Determine search area
        rand_area = self.config.get('rand_area', [-2, 2])
        
        # Start timing
        start_time = time.time()
        
        try:
            # Create RRT* instance
            rrt_star = self.RRTStarClass(
                start=start_pos[:2].tolist(),
                goal=goal_pos[:2].tolist(),
                obstacle_list=obstacles or [],
                rand_area=rand_area,
                expand_dis=expand_dis,
                path_resolution=path_resolution,
                goal_sample_rate=goal_sample_rate,
                max_iter=max_iter,
                robot_radius=robot_radius,
                connect_circle_dist=connect_circle_dist
            )
            
            # Plan path
            path = rrt_star.planning(animation=False)
            
            # Stop timing
            metrics.search_time = time.time() - start_time

            if path is None:
                metrics.success = False
                return None, metrics
            
            # Convert path to numpy array
            path_array = np.array(path).T  # Transpose to get (N x 2)
            
            # Success
            metrics.success = True
            metrics.num_waypoints = len(path_array)
            metrics.path_length = self.calculate_path_length(path_array)
            metrics.smoothness = self.calculate_path_smoothness(path_array)
            metrics.energy = self.calculate_path_energy(path_array)
            
            return path_array, metrics
            
        except Exception as e:
            print(f"[RRT* ERROR] Planning failed: {e}")
            import traceback
            traceback.print_exc()
            metrics.search_time = time.time() - start_time
            metrics.success = False
            return None, metrics
    
    def reset(self):
        """Reset RRT* planner"""
        # RRT* is stateless, no reset needed
        pass

