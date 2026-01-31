"""
LQR-RRT* Motion Planner Wrapper

Wraps PythonRobotics LQR-RRT* implementation.
LQR-RRT* uses Linear Quadratic Regulator for local steering, providing
smoother and more dynamically feasible paths than standard RRT*.
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


class LQRRRTStarPlanner(BaseMotionPlanner):
    """
    Wrapper for PythonRobotics LQR-RRT* planner.
    
    This planner combines RRT* with LQR (Linear Quadratic Regulator) for
    optimal local steering, resulting in smoother and more dynamically
    feasible paths.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize LQR-RRT* planner.
        
        Args:
            config: Configuration dict with keys:
                - max_iter: Maximum iterations (default: 500)
                - goal_sample_rate: Goal sampling rate % (default: 10)
                - robot_radius: Robot radius for collision checking (default: 0.05)
                - connect_circle_dist: Connection circle distance (default: 0.5)
        """
        super().__init__("LQR_RRT_Star", config)
        
        # Import PythonRobotics LQR-RRT*
        lqr_rrtstar_path = project_root / "PythonRobotics" / "PathPlanning" / "LQRRRTStar"
        if str(lqr_rrtstar_path) not in sys.path:
            sys.path.insert(0, str(lqr_rrtstar_path))
        
        try:
            from lqr_rrt_star import LQRRRTStar
            self.LQRRRTStarClass = LQRRRTStar
        except ImportError as e:
            print(f"[LQR-RRT* ERROR] Failed to import: {e}")
            print(f"[LQR-RRT* ERROR] Path: {lqr_rrtstar_path}")
            raise
        
    def plan(self, 
             start_pos: np.ndarray, 
             goal_pos: np.ndarray,
             obstacles: Optional[List] = None) -> Tuple[Optional[np.ndarray], PlannerMetrics]:
        """
        Plan path using LQR-RRT*.
        
        Args:
            start_pos: Start position [x, y] or [x, y, vx, vy]
            goal_pos: Goal position [x, y] or [x, y, vx, vy]
            obstacles: List of obstacles [[x, y, radius], ...]
            
        Returns:
            path: 2D path (N x 2) or None
            metrics: Planning metrics
        """
        metrics = PlannerMetrics()
        
        # Get config parameters (matching actual PythonRobotics LQRRRTStar API)
        max_iter = self.config.get('max_iter', 500)
        goal_sample_rate = self.config.get('goal_sample_rate', 10)
        robot_radius = self.config.get('robot_radius', 0.05)
        connect_circle_dist = self.config.get('connect_circle_dist', 50.0)
        step_size = self.config.get('step_size', 0.2)

        # LQRRRTStar uses [x, y] format (not [x, y, vx, vy])
        if len(start_pos) == 2:
            start = [start_pos[0], start_pos[1]]
        else:
            start = [start_pos[0], start_pos[1]]

        if len(goal_pos) == 2:
            goal = [goal_pos[0], goal_pos[1]]
        else:
            goal = [goal_pos[0], goal_pos[1]]
        
        # Determine search area
        rand_area = self.config.get('rand_area', [-2, 2])
        
        # Start timing
        start_time = time.time()
        
        try:
            # Create LQR-RRT* instance with correct parameters
            lqr_rrt_star = self.LQRRRTStarClass(
                start=start,
                goal=goal,
                obstacle_list=obstacles or [],
                rand_area=rand_area,
                goal_sample_rate=goal_sample_rate,
                max_iter=max_iter,
                connect_circle_dist=connect_circle_dist,
                step_size=step_size,
                robot_radius=robot_radius
            )
            
            # Plan path
            path = lqr_rrt_star.planning(animation=False)
            
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
            print(f"[LQR-RRT* ERROR] Planning failed: {e}")
            import traceback
            traceback.print_exc()
            metrics.search_time = time.time() - start_time
            metrics.success = False
            return None, metrics
    
    def reset(self):
        """Reset LQR-RRT* planner"""
        # Stateless, no reset needed
        pass

