"""
A* Motion Planner Wrapper for Experimental Comparison

Wraps PythonRobotics A* implementation for grid-based path planning.
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


class AStarPlanner(BaseMotionPlanner):
    """
    Wrapper for PythonRobotics A* planner.
    
    This uses the grid-based A* implementation from PythonRobotics.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize A* planner.
        
        Args:
            config: Configuration dict with keys:
                - resolution: Grid resolution in meters (default: 0.05)
                - robot_radius: Robot radius for collision checking (default: 0.05)
        """
        super().__init__("A_Star", config)
        
        # Import PythonRobotics A*
        astar_path = project_root / "PythonRobotics" / "PathPlanning" / "AStar"
        if str(astar_path) not in sys.path:
            sys.path.insert(0, str(astar_path))
        from a_star import AStarPlanner as AStarPlannerImpl
        self.AStarClass = AStarPlannerImpl
        
    def plan(self, 
             start_pos: np.ndarray, 
             goal_pos: np.ndarray,
             obstacles: Optional[List] = None) -> Tuple[Optional[np.ndarray], PlannerMetrics]:
        """
        Plan path using A*.
        
        Args:
            start_pos: Start position [x, y]
            goal_pos: Goal position [x, y]
            obstacles: List of obstacle positions [[x1, y1], [x2, y2], ...]
            
        Returns:
            path: 2D path (N x 2) or None
            metrics: Planning metrics
        """
        metrics = PlannerMetrics()
        
        # Get config parameters
        resolution = self.config.get('resolution', 0.05)
        robot_radius = self.config.get('robot_radius', 0.05)
        
        # Start timing
        start_time = time.time()
        
        try:
            # Prepare obstacle lists
            if obstacles is None or len(obstacles) == 0:
                ox, oy = [], []
            else:
                obstacles_array = np.array(obstacles)
                ox = obstacles_array[:, 0].tolist()
                oy = obstacles_array[:, 1].tolist()

            # Create A* instance
            astar = self.AStarClass(ox, oy, resolution, robot_radius)
            
            # Plan path
            rx, ry = astar.planning(
                start_pos[0], start_pos[1],
                goal_pos[0], goal_pos[1]
            )
            
            # Stop timing
            metrics.search_time = time.time() - start_time

            if rx is None or ry is None:
                metrics.success = False
                return None, metrics
            
            # Convert path to numpy array
            path_array = np.column_stack([rx, ry])
            
            # Success
            metrics.success = True
            metrics.num_waypoints = len(path_array)
            metrics.path_length = self.calculate_path_length(path_array)
            metrics.smoothness = self.calculate_path_smoothness(path_array)
            metrics.energy = self.calculate_path_energy(path_array)
            
            return path_array, metrics
            
        except Exception as e:
            print(f"[A* ERROR] Planning failed: {e}")
            import traceback
            traceback.print_exc()
            metrics.search_time = time.time() - start_time
            metrics.success = False
            return None, metrics
    
    def reset(self):
        """Reset A* planner"""
        # A* is stateless, no reset needed
        pass

