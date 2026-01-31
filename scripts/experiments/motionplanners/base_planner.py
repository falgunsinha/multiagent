"""
Base Motion Planner Interface for Experimental Comparison

This module provides an abstract base class for wrapping different motion planning
algorithms with a common interface for experimental comparison.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict
import numpy as np
import time
import psutil
import os


class PlannerMetrics:
    """Container for motion planning metrics"""

    def __init__(self):
        self.search_time = 0.0  # seconds (renamed from planning_time)
        self.path_length = 0.0  # meters (Cartesian space)
        self.num_waypoints = 0
        self.success = False
        self.iterations = 0
        self.smoothness = 0.0  # Path curvature metric (2D: sum of squared angular changes in X-Y plane)
        self.clearance = 0.0  # Minimum obstacle clearance
        self.energy = 0.0  # Path energy (sum of squared velocities)
        self.memory_mb = 0.0  # Peak memory usage in MB (using tracemalloc for peak measurement)

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary"""
        return {
            'search_time': self.search_time,
            'path_length': self.path_length,
            'num_waypoints': self.num_waypoints,
            'success': self.success,
            'iterations': self.iterations,
            'smoothness': self.smoothness,
            'clearance': self.clearance,
            'energy': self.energy,
            'memory_mb': self.memory_mb
        }


class BaseMotionPlanner(ABC):
    """
    Abstract base class for motion planning algorithms.
    
    All motion planners should inherit from this class and implement
    the required methods for experimental comparison.
    """
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        """
        Initialize base planner.
        
        Args:
            name: Name of the planner (e.g., "RRT", "A*", "PRM")
            config: Configuration dictionary for planner parameters
        """
        self.name = name
        self.config = config or {}
        self.metrics = PlannerMetrics()
        
    @abstractmethod
    def plan(self, 
             start_pos: np.ndarray, 
             goal_pos: np.ndarray,
             obstacles: Optional[List] = None) -> Tuple[Optional[np.ndarray], PlannerMetrics]:
        """
        Plan a path from start to goal.
        
        Args:
            start_pos: Start position [x, y, z] or joint configuration
            goal_pos: Goal position [x, y, z] or joint configuration
            obstacles: List of obstacles (format depends on planner)
            
        Returns:
            path: Planned path as numpy array (N x dim) or None if planning failed
            metrics: PlannerMetrics object with planning statistics
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset planner state for new planning query"""
        pass
    
    def get_name(self) -> str:
        """Get planner name"""
        return self.name
    
    def get_config(self) -> Dict:
        """Get planner configuration"""
        return self.config

    def get_memory_usage_mb(self) -> float:
        """
        Get current memory usage in MB.

        Returns:
            Memory usage in megabytes
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

    def plan_pick_and_place(self,
                           robot_pos: np.ndarray,
                           pick_pos: np.ndarray,
                           place_pos: np.ndarray,
                           obstacles: Optional[List] = None) -> Tuple[Optional[np.ndarray], PlannerMetrics]:
        """
        Plan complete pick-and-place task (Robot→Pick→Place).

        Args:
            robot_pos: Robot starting position [x, y]
            pick_pos: Pick location [x, y]
            place_pos: Place location [x, y]
            obstacles: List of obstacles

        Returns:
            complete_path: Combined path or None if failed
            total_metrics: Aggregated metrics from both segments
        """
        # Phase 1: Robot → Pick
        memory_before = self.get_memory_usage_mb()
        path1, metrics1 = self.plan(robot_pos, pick_pos, obstacles)

        if not metrics1.success:
            # Failed to reach pick location
            failed_metrics = PlannerMetrics()
            failed_metrics.success = False
            failed_metrics.search_time = metrics1.search_time
            failed_metrics.memory_mb = max(0.0, self.get_memory_usage_mb() - memory_before)
            return None, failed_metrics

        # Phase 2: Pick → Place
        path2, metrics2 = self.plan(pick_pos, place_pos, obstacles)
        memory_after = self.get_memory_usage_mb()

        if not metrics2.success:
            # Failed to reach place location
            failed_metrics = PlannerMetrics()
            failed_metrics.success = False
            failed_metrics.search_time = metrics1.search_time + metrics2.search_time
            failed_metrics.memory_mb = max(0.0, memory_after - memory_before)
            return None, failed_metrics

        # Combine paths
        complete_path = np.vstack([path1, path2])

        # Aggregate metrics
        total_metrics = PlannerMetrics()
        total_metrics.success = True
        total_metrics.search_time = metrics1.search_time + metrics2.search_time
        total_metrics.path_length = metrics1.path_length + metrics2.path_length
        total_metrics.num_waypoints = metrics1.num_waypoints + metrics2.num_waypoints
        total_metrics.memory_mb = max(0.0, memory_after - memory_before)

        # Recalculate smoothness for complete path
        total_metrics.smoothness = self.calculate_path_smoothness(complete_path)

        # Recalculate clearance for complete path
        if obstacles:
            min_clearance = float('inf')
            for point in complete_path:
                for obs in obstacles:
                    dist = np.linalg.norm(point - obs[:2]) - obs[2]  # distance - radius
                    min_clearance = min(min_clearance, dist)
            total_metrics.clearance = max(0.0, min_clearance)

        return complete_path, total_metrics

    def calculate_path_smoothness(self, path: np.ndarray) -> float:
        """
        Calculate path smoothness metric (lower is smoother).
        
        Measures the sum of squared angular changes along the path.
        """
        if path is None or len(path) < 3:
            return 0.0
            
        smoothness = 0.0
        for i in range(1, len(path) - 1):
            # Calculate angle change at each waypoint
            v1 = path[i] - path[i-1]
            v2 = path[i+1] - path[i]
            
            # Normalize vectors
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 1e-6 and v2_norm > 1e-6:
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm
                
                # Calculate angle change
                cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                smoothness += angle ** 2
                
        return smoothness
    
    def calculate_path_length(self, path: np.ndarray) -> float:
        """Calculate total path length in Cartesian space"""
        if path is None or len(path) < 2:
            return 0.0
            
        length = 0.0
        for i in range(1, len(path)):
            length += np.linalg.norm(path[i] - path[i-1])
            
        return length
    
    def calculate_path_energy(self, path: np.ndarray, dt: float = 0.1) -> float:
        """
        Calculate path energy (sum of squared velocities).
        
        Args:
            path: Path waypoints
            dt: Time step between waypoints
        """
        if path is None or len(path) < 2:
            return 0.0
            
        energy = 0.0
        for i in range(1, len(path)):
            velocity = (path[i] - path[i-1]) / dt
            energy += np.sum(velocity ** 2)
            
        return energy

