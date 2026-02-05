import numpy as np
from typing import List, Tuple


class IsaacSimRRTPathEstimator:
    """RRT path estimator using Isaac Sim's native RRT planner"""

    def __init__(self, grid_size: int = 6, cell_size: float = 0.20,
                 rrt_planner=None, kinematics_solver=None,
                 articulation_kinematics_solver=None, franka_articulation=None):
        """Initialize Isaac Sim RRT path estimator"""
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.grid_center = np.array([0.45, -0.10])

        self.rrt_planner = rrt_planner
        self.kinematics_solver = kinematics_solver
        self.articulation_kinematics_solver = articulation_kinematics_solver
        self.franka = franka_articulation

        self.obstacle_map = np.zeros((grid_size, grid_size), dtype=bool)
        self.obstacle_positions = []

        self.total_calls = 0
        self.ik_failures = 0
        self.rrt_failures = 0
        self.rrt_successes = 0

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
        """Update occupancy grid with obstacles"""
        self.obstacle_map.fill(False)
        self.obstacle_positions = []

        for obj_pos in object_positions:
            grid_x, grid_y = self._world_to_grid(obj_pos[:2])

            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                self.obstacle_map[grid_y, grid_x] = True
                self.obstacle_positions.append(obj_pos)

        for obs_pos in obstacle_positions:
            grid_x, grid_y = self._world_to_grid(obs_pos[:2])

            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                self.obstacle_map[grid_y, grid_x] = True
                self.obstacle_positions.append(obs_pos)

    def update_rrt_world(self):
        """Update RRT planner's world state with current obstacles"""

        if self.rrt_planner is None:
            return

        if self.franka is not None:
            robot_base_translation, robot_base_orientation = self.franka.get_world_pose()
            self.kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
            self.rrt_planner.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        self.rrt_planner.update_world()

    def estimate_path_length(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> float:
        """Estimate path length using Isaac Sim native RRT planning"""
        self.total_calls += 1

        euclidean_distance = np.linalg.norm(goal_pos[:2] - start_pos[:2])

        if (self.rrt_planner is None or self.kinematics_solver is None or
            self.articulation_kinematics_solver is None or self.franka is None):
            raise RuntimeError(
                "[IsaacSimRRT] Required components not available! "
                f"rrt_planner={'OK' if self.rrt_planner else 'MISSING'}, "
                f"kinematics_solver={'OK' if self.kinematics_solver else 'MISSING'}, "
                f"articulation_kinematics_solver={'OK' if self.articulation_kinematics_solver else 'MISSING'}, "
                f"franka={'OK' if self.franka else 'MISSING'}"
            )

        self.update_rrt_world()

        target_position = goal_pos if len(goal_pos) == 3 else np.array([goal_pos[0], goal_pos[1], 0.1])
        target_orientation = np.array([1.0, 0.0, 0.0, 0.0])

        _, ik_success = self.articulation_kinematics_solver.compute_inverse_kinematics(
            target_position, target_orientation
        )

        if not ik_success:
            self.ik_failures += 1
            return 2.0 * euclidean_distance

        self.rrt_planner.set_end_effector_target(target_position, target_orientation)
        self.rrt_planner.update_world()

        current_joint_positions = self.franka.get_joint_positions()

        rrt_path = self.rrt_planner.compute_path(
            current_joint_positions[:7],
            np.array([])
        )

        if rrt_path is None or len(rrt_path) <= 1:
            self.rrt_failures += 1
            return 2.0 * euclidean_distance

        self.rrt_successes += 1

        path_length = 0.0

        for i in range(len(rrt_path) - 1):
            joints_current = rrt_path[i]
            joints_next = rrt_path[i + 1]

            pos_current, _ = self.kinematics_solver.compute_forward_kinematics(
                "right_gripper", joints_current
            )

            pos_next, _ = self.kinematics_solver.compute_forward_kinematics(
                "right_gripper", joints_next
            )

            segment_length = np.linalg.norm(pos_next - pos_current)
            path_length += segment_length

        return path_length

    def check_reachability(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> bool:
        """Quick reachability check using Isaac Sim RRT"""
        path_length = self.estimate_path_length(start_pos, goal_pos)
        euclidean_distance = np.linalg.norm(goal_pos[:2] - start_pos[:2])

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

