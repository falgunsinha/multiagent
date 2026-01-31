"""
PRM Motion Planner Wrapper for Experimental Comparison

Wraps PythonRobotics PRM (Probabilistic Roadmap) implementation.
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


class PRMPlanner(BaseMotionPlanner):
    """
    Wrapper for PythonRobotics PRM planner.
    
    This uses the Probabilistic Roadmap implementation from PythonRobotics.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize PRM planner.
        
        Args:
            config: Configuration dict with keys:
                - n_sample: Number of sample points (default: 500)
                - n_knn: Number of nearest neighbors (default: 10)
                - max_edge_len: Maximum edge length (default: 30.0)
                - robot_radius: Robot radius for collision checking (default: 0.05)
        """
        super().__init__("PRM", config)
        
        # Import PythonRobotics PRM
        prm_path = project_root / "PythonRobotics" / "PathPlanning" / "ProbabilisticRoadMap"
        if str(prm_path) not in sys.path:
            sys.path.insert(0, str(prm_path))
        from probabilistic_road_map import prm_planning
        self.prm_planning = prm_planning
        
    def plan(self, 
             start_pos: np.ndarray, 
             goal_pos: np.ndarray,
             obstacles: Optional[List] = None) -> Tuple[Optional[np.ndarray], PlannerMetrics]:
        """
        Plan path using PRM.
        
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
        robot_radius = self.config.get('robot_radius', 0.05)
        
        # Update global parameters in the module
        import probabilistic_road_map
        probabilistic_road_map.N_SAMPLE = self.config.get('n_sample', 500)
        probabilistic_road_map.N_KNN = self.config.get('n_knn', 10)
        probabilistic_road_map.MAX_EDGE_LEN = self.config.get('max_edge_len', 30.0)
        
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

            # Add workspace boundary points to ensure proper sampling area
            # This prevents PRM from getting stuck when there are few obstacles
            workspace_bounds = [-1.0, -1.0, 1.0, 1.0]  # [min_x, min_y, max_x, max_y]
            if len(ox) == 0:
                # No obstacles - add boundary points
                ox = [workspace_bounds[0], workspace_bounds[2]]
                oy = [workspace_bounds[1], workspace_bounds[3]]
            else:
                # Add boundary points to existing obstacles
                ox.extend([workspace_bounds[0], workspace_bounds[2]])
                oy.extend([workspace_bounds[1], workspace_bounds[3]])

            # Plan path
            rx, ry = self.prm_planning(
                start_pos[0], start_pos[1],
                goal_pos[0], goal_pos[1],
                ox, oy,
                robot_radius
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
            print(f"[PRM ERROR] Planning failed: {e}")
            import traceback
            traceback.print_exc()
            metrics.search_time = time.time() - start_time
            metrics.success = False
            return None, metrics
    
    def reset(self):
        """Reset PRM planner"""
        # PRM is stateless, no reset needed
        pass

