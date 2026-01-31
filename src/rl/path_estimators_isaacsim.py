"""
Path estimation using Isaac Sim native RRT for RL training.
Uses actual Isaac Sim RRT planner for accurate path length estimation during training.

This is separate from path_estimators.py (PythonRobotics) which is used for visualization.
"""

import numpy as np
from typing import List, Tuple, Optional
import math


class IsaacSimRRTPathEstimator:
    """
    RRT path estimator using Isaac Sim's native RRT planner.
    Provides accurate path length estimation for RL training.
    
    This class interfaces with Isaac Sim's motion generation RRT planner
    to get actual C-space trajectories and calculate path lengths.
    """

    def __init__(self, grid_size: int = 6, cell_size: float = 0.20,
                 rrt_planner=None, kinematics_solver=None,
                 articulation_kinematics_solver=None, franka_articulation=None):
        """
        Initialize Isaac Sim RRT path estimator.

        Args:
            grid_size: Size of the grid (e.g., 6 for 6x6)
            cell_size: Size of each grid cell in meters (UPDATED: 0.20m to fit 15.2cm gripper)
            rrt_planner: Isaac Sim RRT planner instance (from isaacsim.robot_motion.motion_generation.lula.RRT)
            kinematics_solver: Lula kinematics solver instance
            articulation_kinematics_solver: Articulation kinematics solver instance
            franka_articulation: Franka robot articulation instance
        """
        self.grid_size = grid_size
        self.cell_size = cell_size  # default 0.20m for gripper width
        self.grid_center = np.array([0.45, -0.10])  # Grid center in world coordinates
        
        # Isaac Sim components
        self.rrt_planner = rrt_planner
        self.kinematics_solver = kinematics_solver
        self.articulation_kinematics_solver = articulation_kinematics_solver
        self.franka = franka_articulation
        
        # Occupancy grid for obstacle tracking
        self.obstacle_map = np.zeros((grid_size, grid_size), dtype=bool)
        self.obstacle_positions = []  # List of obstacle world positions

        # Performance tracking for debugging
        self.total_calls = 0
        self.ik_failures = 0
        self.rrt_failures = 0
        self.rrt_successes = 0

        # Validate that all required components are provided
        if (self.rrt_planner is None or self.kinematics_solver is None or
            self.articulation_kinematics_solver is None or self.franka is None):
            print("[IsaacSimRRT WARNING] Some components are None - training will fail if path estimation is called!")
            print(f"  - rrt_planner: {'OK' if rrt_planner else 'MISSING'}")
            print(f"  - kinematics_solver: {'OK' if kinematics_solver else 'MISSING'}")
            print(f"  - articulation_kinematics_solver: {'OK' if articulation_kinematics_solver else 'MISSING'}")
            print(f"  - franka_articulation: {'OK' if franka_articulation else 'MISSING'}")
        else:
            print("[IsaacSimRRT] All components initialized - using actual Isaac Sim RRT")

    def set_rrt_planner(self, rrt_planner):
        """Set or update the RRT planner instance"""
        self.rrt_planner = rrt_planner

    def set_kinematics_solver(self, kinematics_solver):
        """Set or update the kinematics solver instance"""
        self.kinematics_solver = kinematics_solver

    def set_articulation_kinematics_solver(self, articulation_kinematics_solver):
        """Set or update the articulation kinematics solver instance"""
        self.articulation_kinematics_solver = articulation_kinematics_solver

    def set_franka_articulation(self, franka_articulation):
        """Set or update the Franka articulation instance"""
        self.franka = franka_articulation

    def _world_to_grid(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        grid_extent = (self.grid_size - 1) * self.cell_size
        start_x = self.grid_center[0] - (grid_extent / 2.0)
        start_y = self.grid_center[1] - (grid_extent / 2.0)

        grid_x = round((world_pos[0] - start_x) / self.cell_size)
        grid_y = round((world_pos[1] - start_y) / self.cell_size)

        # Clamp to valid grid bounds
        grid_x = max(0, min(self.grid_size - 1, grid_x))
        grid_y = max(0, min(self.grid_size - 1, grid_y))

        return grid_x, grid_y

    def _grid_to_world(self, grid_x: int, grid_y: int) -> np.ndarray:
        """Convert grid coordinates to world coordinates"""
        grid_extent = (self.grid_size - 1) * self.cell_size
        start_x = self.grid_center[0] - (grid_extent / 2.0)
        start_y = self.grid_center[1] - (grid_extent / 2.0)

        world_x = start_x + (grid_x * self.cell_size)
        world_y = start_y + (grid_y * self.cell_size)

        return np.array([world_x, world_y])

    def update_occupancy_grid(self, object_positions: List[np.ndarray], obstacle_positions: List[np.ndarray]):
        """
        Update occupancy grid with obstacles.
        
        Args:
            object_positions: List of object positions (unpicked cubes)
            obstacle_positions: List of static obstacle positions
        """
        self.obstacle_map.fill(False)
        self.obstacle_positions = []

        # Add unpicked cubes as obstacles
        for obj_pos in object_positions:
            grid_x, grid_y = self._world_to_grid(obj_pos[:2])
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                self.obstacle_map[grid_y, grid_x] = True
                self.obstacle_positions.append(obj_pos)

        # Add static obstacles
        for obs_pos in obstacle_positions:
            grid_x, grid_y = self._world_to_grid(obs_pos[:2])
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                self.obstacle_map[grid_y, grid_x] = True
                self.obstacle_positions.append(obs_pos)

    def update_rrt_world(self):
        """Update RRT planner's world state with current obstacles"""
        if self.rrt_planner is None:
            return
        
        # Update robot base pose
        if self.franka is not None:
            robot_base_translation, robot_base_orientation = self.franka.get_world_pose()
            self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
            self.rrt_planner.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        
        # Update world (obstacles are handled by Isaac Sim's collision detection)
        self.rrt_planner.update_world()

    def estimate_path_length(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> float:
        """
        Estimate path length using Isaac Sim native RRT planning.

        Args:
            start_pos: Start position (x, y, z) in world coordinates
            goal_pos: Goal position (x, y, z) in world coordinates

        Returns:
            Estimated path length in meters
            Returns 2.0 × Euclidean distance if planning fails (same as A*)

        Raises:
            RuntimeError: If required Isaac Sim components are not available
        """
        self.total_calls += 1

        # Calculate Euclidean distance for failure penalty
        euclidean_distance = np.linalg.norm(goal_pos[:2] - start_pos[:2])

        # Check if required components are available - FAIL EXPLICITLY if not
        if (self.rrt_planner is None or self.kinematics_solver is None or
            self.articulation_kinematics_solver is None or self.franka is None):
            raise RuntimeError(
                "[IsaacSimRRT] Required components not available! "
                f"rrt_planner={'OK' if self.rrt_planner else 'MISSING'}, "
                f"kinematics_solver={'OK' if self.kinematics_solver else 'MISSING'}, "
                f"articulation_kinematics_solver={'OK' if self.articulation_kinematics_solver else 'MISSING'}, "
                f"franka={'OK' if self.franka else 'MISSING'}"
            )

        # Update RRT world state
        self.update_rrt_world()

        # Set target position (use default orientation for pick position)
        target_position = goal_pos if len(goal_pos) == 3 else np.array([goal_pos[0], goal_pos[1], 0.1])
        target_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Default orientation (quaternion)

        # Check if IK solution exists
        _, ik_success = self.articulation_kinematics_solver.compute_inverse_kinematics(
            target_position, target_orientation
        )

        if not ik_success:
            # No IK solution - return 2.0 × Euclidean (same as A* failure penalty)
            self.ik_failures += 1
            return 2.0 * euclidean_distance

        # Set RRT target
        self.rrt_planner.set_end_effector_target(target_position, target_orientation)
        self.rrt_planner.update_world()

        # Get current joint positions
        current_joint_positions = self.franka.get_joint_positions()

        # Plan path using RRT
        rrt_path = self.rrt_planner.compute_path(
            current_joint_positions[:7],  # First 7 joints (exclude gripper)
            np.array([])  # Watched joints (empty for Franka)
        )

        if rrt_path is None or len(rrt_path) <= 1:
            # RRT planning failed - return 2.0 × Euclidean (same as A* failure penalty)
            self.rrt_failures += 1
            return 2.0 * euclidean_distance

        # RRT planning succeeded
        self.rrt_successes += 1

        # Calculate path length in Cartesian space using forward kinematics
        path_length = 0.0

        # Use forward kinematics to get Cartesian positions for each waypoint
        for i in range(len(rrt_path) - 1):
            # Get joint positions for consecutive waypoints
            joints_current = rrt_path[i]
            joints_next = rrt_path[i + 1]

            # Use forward kinematics to get end-effector positions
            # Compute FK for current waypoint
            pos_current, _ = self.kinematics_solver.compute_forward_kinematics(
                "right_gripper", joints_current
            )

            # Compute FK for next waypoint
            pos_next, _ = self.kinematics_solver.compute_forward_kinematics(
                "right_gripper", joints_next
            )

            # Calculate Cartesian distance between consecutive waypoints
            segment_length = np.linalg.norm(pos_next - pos_current)
            path_length += segment_length

        return path_length

    def check_reachability(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> bool:
        """
        Quick reachability check using Isaac Sim RRT.
        Used for calculating reachability flag in observations.

        Args:
            start_pos: Start position (x, y, z) in world coordinates
            goal_pos: Goal position (x, y, z) in world coordinates

        Returns:
            True if RRT can find a path, False otherwise
        """
        # Use path length estimation to check reachability
        # If path length < 2.0 × Euclidean, RRT succeeded
        path_length = self.estimate_path_length(start_pos, goal_pos)
        euclidean_distance = np.linalg.norm(goal_pos[:2] - start_pos[:2])

        # RRT succeeded if path length is less than failure penalty (2.0 × Euclidean)
        return path_length < 2.0 * euclidean_distance

    def get_statistics(self) -> dict:
        """Get RRT planning statistics for debugging"""
        success_rate = (self.rrt_successes / self.total_calls * 100) if self.total_calls > 0 else 0.0
        failure_rate = ((self.ik_failures + self.rrt_failures) / self.total_calls * 100) if self.total_calls > 0 else 0.0

        return {
            "total_calls": self.total_calls,
            "rrt_successes": self.rrt_successes,
            "ik_failures": self.ik_failures,
            "rrt_failures": self.rrt_failures,
            "success_rate": success_rate,
            "failure_rate": failure_rate
        }

    def print_statistics(self):
        """Print RRT planning statistics"""
        stats = self.get_statistics()
        print("\n" + "="*60)
        print("ISAAC SIM RRT PATH ESTIMATOR STATISTICS")
        print("="*60)
        print(f"Total calls:        {stats['total_calls']}")
        print(f"RRT successes:      {stats['rrt_successes']} ({stats['success_rate']:.1f}%)")
        print(f"IK failures:        {stats['ik_failures']}")
        print(f"RRT failures:       {stats['rrt_failures']}")
        print(f"Overall failure rate: {stats['failure_rate']:.1f}%")
        print("="*60 + "\n")

