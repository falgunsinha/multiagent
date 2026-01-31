"""
RRT Motion Planner Wrapper for Experimental Comparison

Wraps both Isaac Sim native RRT and PythonRobotics RRT implementations.
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


class IsaacSimRRTPlanner(BaseMotionPlanner):
    """
    Wrapper for Isaac Sim native RRT planner.
    
    This uses the Lula-based RRT implementation in Isaac Sim for
    robot manipulation tasks.
    """
    
    def __init__(self, 
                 rrt_planner,
                 kinematics_solver,
                 articulation_kinematics_solver,
                 franka_articulation,
                 config: Optional[Dict] = None):
        """
        Initialize Isaac Sim RRT planner.
        
        Args:
            rrt_planner: Isaac Sim RRT instance
            kinematics_solver: Lula kinematics solver
            articulation_kinematics_solver: Articulation kinematics solver
            franka_articulation: Franka robot articulation
            config: Configuration dict with keys:
                - max_iterations: Maximum RRT iterations (default: 10000)
        """
        super().__init__("IsaacSim_RRT", config)
        
        self.rrt = rrt_planner
        self.kinematics_solver = kinematics_solver
        self.articulation_ik_solver = articulation_kinematics_solver
        self.franka = franka_articulation
        
        # Set max iterations
        max_iter = self.config.get('max_iterations', 10000)
        self.rrt.set_max_iterations(max_iter)
        
    def plan(self, 
             start_pos: np.ndarray, 
             goal_pos: np.ndarray,
             obstacles: Optional[List] = None) -> Tuple[Optional[np.ndarray], PlannerMetrics]:
        """
        Plan path using Isaac Sim RRT.
        
        Args:
            start_pos: Start joint configuration (7 DOF)
            goal_pos: Goal end-effector position [x, y, z]
            obstacles: Not used (obstacles are in Isaac Sim scene)
            
        Returns:
            path: Joint space path (N x 7) or None
            metrics: Planning metrics
        """
        metrics = PlannerMetrics()
        
        # Start timing
        start_time = time.time()
        
        try:
            # Set RRT target (goal_pos is end-effector position)
            target_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Default orientation
            self.rrt.set_end_effector_target(goal_pos, target_orientation)
            self.rrt.update_world()
            
            # Plan path
            rrt_path = self.rrt.compute_path(
                start_pos[:7],  # First 7 joints
                np.array([])  # Watched joints (empty for Franka)
            )
            
            # Stop timing
            metrics.search_time = time.time() - start_time

            if rrt_path is None or len(rrt_path) <= 1:
                metrics.success = False
                return None, metrics
            
            # Success
            metrics.success = True
            metrics.num_waypoints = len(rrt_path)
            
            # Calculate path length in Cartesian space using forward kinematics
            cartesian_path = []
            for joint_config in rrt_path:
                ee_pos, _ = self.articulation_ik_solver.compute_end_effector_pose(joint_config)
                cartesian_path.append(ee_pos)
            
            cartesian_path = np.array(cartesian_path)
            metrics.path_length = self.calculate_path_length(cartesian_path)
            metrics.smoothness = self.calculate_path_smoothness(cartesian_path)
            metrics.energy = self.calculate_path_energy(rrt_path)  # Joint space energy
            
            return rrt_path, metrics
            
        except Exception as e:
            print(f"[RRT ERROR] Planning failed: {e}")
            metrics.search_time = time.time() - start_time
            metrics.success = False
            return None, metrics
    
    def reset(self):
        """Reset RRT planner"""
        # RRT is stateless, no reset needed
        pass


class PythonRoboticsRRTPlanner(BaseMotionPlanner):
    """
    Wrapper for PythonRobotics RRT planner.
    
    This uses the 2D RRT implementation from PythonRobotics for
    grid-based path planning experiments.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize PythonRobotics RRT planner.
        
        Args:
            config: Configuration dict with keys:
                - expand_dis: Expansion distance (default: 0.1)
                - path_resolution: Path resolution (default: 0.05)
                - goal_sample_rate: Goal sampling rate % (default: 5)
                - max_iter: Maximum iterations (default: 500)
                - robot_radius: Robot radius for collision checking (default: 0.05)
        """
        super().__init__("PythonRobotics_RRT", config)
        
        # Import PythonRobotics RRT
        sys.path.insert(0, str(project_root / "PythonRobotics" / "PathPlanning" / "RRT"))
        from rrt import RRT
        self.RRTClass = RRT
        
    def plan(self, 
             start_pos: np.ndarray, 
             goal_pos: np.ndarray,
             obstacles: Optional[List] = None) -> Tuple[Optional[np.ndarray], PlannerMetrics]:
        """
        Plan path using PythonRobotics RRT.
        
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
        goal_sample_rate = self.config.get('goal_sample_rate', 5)
        max_iter = self.config.get('max_iter', 500)
        robot_radius = self.config.get('robot_radius', 0.05)

        # Determine search area
        rand_area = self.config.get('rand_area', [-2, 2])

        # Start timing
        start_time = time.time()

        try:
            # Create RRT instance
            rrt = self.RRTClass(
                start=start_pos[:2].tolist(),
                goal=goal_pos[:2].tolist(),
                obstacle_list=obstacles or [],
                rand_area=rand_area,
                expand_dis=expand_dis,
                path_resolution=path_resolution,
                goal_sample_rate=goal_sample_rate,
                max_iter=max_iter,
                robot_radius=robot_radius
            )

            # Plan path
            path = rrt.planning(animation=False)

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
            print(f"[RRT ERROR] Planning failed: {e}")
            metrics.search_time = time.time() - start_time
            metrics.success = False
            return None, metrics

    def reset(self):
        """Reset RRT planner"""
        # RRT is stateless, no reset needed
        pass

