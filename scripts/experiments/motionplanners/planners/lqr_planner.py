"""
LQR Motion Planner Wrapper

Wraps PythonRobotics LQR (Linear Quadratic Regulator) path planner.
LQR provides optimal control for linear systems with quadratic cost.
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


class LQRPlanner(BaseMotionPlanner):
    """
    Wrapper for PythonRobotics LQR planner.
    
    LQR (Linear Quadratic Regulator) provides optimal control for
    linear dynamical systems with quadratic cost functions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize LQR planner.
        
        Args:
            config: Configuration dict with keys:
                - dt: Time step (default: 0.1)
                - max_time: Maximum simulation time (default: 10.0)
                - robot_radius: Robot radius for collision checking (default: 0.05)
        """
        super().__init__("LQR", config)
        
        # Import PythonRobotics LQR
        lqr_path = project_root / "PythonRobotics" / "PathPlanning" / "LQRPlanner"
        if str(lqr_path) not in sys.path:
            sys.path.insert(0, str(lqr_path))

        try:
            from lqr_planner import LQRPlanner as PythonRoboticsLQR
            self.LQRPlannerClass = PythonRoboticsLQR
        except ImportError as e:
            print(f"[LQR ERROR] Failed to import: {e}")
            print(f"[LQR ERROR] Path: {lqr_path}")
            raise
        
    def plan(self, 
             start_pos: np.ndarray, 
             goal_pos: np.ndarray,
             obstacles: Optional[List] = None) -> Tuple[Optional[np.ndarray], PlannerMetrics]:
        """
        Plan path using LQR.
        
        Args:
            start_pos: Start position [x, y] or [x, y, vx, vy]
            goal_pos: Goal position [x, y] or [x, y, vx, vy]
            obstacles: List of obstacles (not used by LQR, but kept for interface consistency)
            
        Returns:
            path: 2D path (N x 2) or None
            metrics: Planning metrics
        """
        metrics = PlannerMetrics()
        
        # Get config parameters
        dt = self.config.get('dt', 0.1)
        max_time = self.config.get('max_time', 100.0)

        # Extract scalar positions (LQR planner API uses sx, sy, gx, gy)
        start_x = float(start_pos[0])
        start_y = float(start_pos[1])
        goal_x = float(goal_pos[0])
        goal_y = float(goal_pos[1])

        # Start timing
        start_time = time.time()

        try:
            # Create LQR planner instance
            lqr_planner = self.LQRPlannerClass()
            lqr_planner.DT = dt
            lqr_planner.MAX_TIME = max_time

            # Plan path - API: lqr_planning(sx, sy, gx, gy, show_animation)
            # Returns: rx, ry (lists of x and y coordinates)
            rx, ry = lqr_planner.lqr_planning(
                sx=start_x,
                sy=start_y,
                gx=goal_x,
                gy=goal_y,
                show_animation=False
            )

            # Stop timing
            metrics.search_time = time.time() - start_time

            if not rx or not ry or len(rx) == 0:
                metrics.success = False
                return None, metrics

            # Convert to numpy array
            path_array = np.column_stack([rx, ry])

            # Check if path reached goal (within tolerance)
            final_pos = path_array[-1]
            goal_tolerance = 0.5
            distance_to_goal = np.linalg.norm(final_pos - goal_pos[:2])

            if distance_to_goal > goal_tolerance:
                metrics.success = False
                return None, metrics

            # Success
            metrics.success = True
            metrics.num_waypoints = len(path_array)
            metrics.path_length = self.calculate_path_length(path_array)
            metrics.smoothness = self.calculate_path_smoothness(path_array)
            metrics.energy = self.calculate_path_energy(path_array)

            return path_array, metrics

        except Exception as e:
            print(f"[LQR ERROR] Planning failed: {e}")
            import traceback
            traceback.print_exc()
            metrics.search_time = time.time() - start_time
            metrics.success = False
            return None, metrics
    
    def reset(self):
        """Reset LQR planner"""
        # Stateless, no reset needed
        pass

