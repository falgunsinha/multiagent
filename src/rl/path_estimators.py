"""
Path estimation algorithms for RL training.
Provides A* path length estimation for more accurate reward calculation.

This implementation is adapted from PythonRobotics A* implementation:
C:/isaacsim/cobotproject/PythonRobotics/PathPlanning/AStar/a_star.py

Key adaptations for RL environment:
1. Adapted for grid-based pathfinding (discrete grid cells)
2. Goal cell is always expandable (pick-and-place exception)
3. Returns path length for reward calculation
"""

import numpy as np
from typing import List, Tuple, Optional
import math


class Node:
    """
    Node class for A* search (from PythonRobotics).

    Attributes:
        x: x index of grid
        y: y index of grid
        cost: cost from start to current node (g-cost)
        parent_index: index of parent node in closed set
    """

    def __init__(self, x: int, y: int, cost: float, parent_index: int):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index

    def __str__(self):
        return f"{self.x},{self.y},{self.cost},{self.parent_index}"


class AStarPathEstimator:
    """
    A* path estimator for grid-based environments (adapted from PythonRobotics).
    Estimates path length considering obstacles.

    Based on: C:/isaacsim/cobotproject/PythonRobotics/PathPlanning/AStar/a_star.py
    """

    def __init__(self, grid_size: int = 6, cell_size: float = 0.26):
        """
        Initialize A* path estimator.

        Args:
            grid_size: Size of the grid (e.g., 6 for 6x6)
            cell_size: Size of each grid cell in meters (default: 0.26m = 26cm, ensures 15cm gripper fits between objects)
        """
        self.grid_size = grid_size
        # UPDATED: Increased from 0.20 to 0.26 to ensure gripper (15cm) can fit between objects
        # With 26cm spacing: cube-to-cube gap = 26cm - 5.15cm = 20.85cm (gripper 15cm fits with 5.85cm clearance)
        self.cell_size = cell_size  # resolution (default 26cm to fit 15cm gripper + safety margin)
        self.grid_center = np.array([0.45, -0.10])  # Grid center in world coordinates

        # Grid bounds (matching PythonRobotics structure)
        self.min_x = 0
        self.min_y = 0
        self.max_x = grid_size
        self.max_y = grid_size
        self.x_width = grid_size
        self.y_width = grid_size

        # Motion model (8-connected grid: 4 orthogonal + 4 diagonal)
        self.motion = self.get_motion_model()

        # Create occupancy grid (False = free, True = occupied)
        # This matches PythonRobotics obstacle_map structure
        self.obstacle_map = [[False for _ in range(self.y_width)]
                            for _ in range(self.x_width)]

    @staticmethod
    def get_motion_model():
        """
        Get motion model for 8-connected grid (from PythonRobotics).
        Returns: List of [dx, dy, cost] for each possible move
        """
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]
        return motion

    def update_occupancy_grid(self, object_positions: List[np.ndarray], obstacle_positions: List[np.ndarray] = None):
        """
        Update occupancy grid based on object and obstacle positions.

        Args:
            object_positions: List of object positions (x, y, z)
            obstacle_positions: List of obstacle positions (x, y, z)
        """
        # Reset grid (False = free, True = occupied)
        for ix in range(self.x_width):
            for iy in range(self.y_width):
                self.obstacle_map[ix][iy] = False

        # Mark object cells as occupied
        # Objects (cubes) are obstacles that block paths (except when they are the goal)
        for pos in object_positions:
            grid_x, grid_y = self._world_to_grid(pos[:2])
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                self.obstacle_map[grid_x][grid_y] = True  # Occupied

        # Mark obstacle cells as occupied
        if obstacle_positions:
            for pos in obstacle_positions:
                grid_x, grid_y = self._world_to_grid(pos[:2])
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    self.obstacle_map[grid_x][grid_y] = True  # Occupied
    
    def _world_to_grid(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        # Calculate grid extent
        grid_extent = (self.grid_size - 1) * self.cell_size
        start_x = self.grid_center[0] - (grid_extent / 2.0)
        start_y = self.grid_center[1] - (grid_extent / 2.0)

        # Convert to grid coordinates using ROUNDING (not truncation)
        # This ensures cubes map to the nearest grid cell
        grid_x = round((world_pos[0] - start_x) / self.cell_size)
        grid_y = round((world_pos[1] - start_y) / self.cell_size)

        # CLAMP to valid grid bounds [0, grid_size-1]
        # Cubes outside grid boundaries get clamped to edge cells
        grid_x = max(0, min(self.grid_size - 1, grid_x))
        grid_y = max(0, min(self.grid_size - 1, grid_y))

        return grid_x, grid_y
    
    def _grid_to_world(self, grid_x: int, grid_y: int) -> np.ndarray:
        """Convert grid coordinates to world coordinates."""
        grid_extent = (self.grid_size - 1) * self.cell_size
        start_x = self.grid_center[0] - (grid_extent / 2.0)
        start_y = self.grid_center[1] - (grid_extent / 2.0)
        
        world_x = start_x + (grid_x * self.cell_size)
        world_y = start_y + (grid_y * self.cell_size)
        
        return np.array([world_x, world_y])
    
    def estimate_path_length(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> float:
        """
        Estimate path length from start to goal using A*.

        Args:
            start_pos: Start position (x, y, z) or (x, y)
            goal_pos: Goal position (x, y, z) or (x, y)

        Returns:
            Estimated path length in meters
        """
        # Convert to grid coordinates
        start_grid = self._world_to_grid(start_pos[:2])
        goal_grid = self._world_to_grid(goal_pos[:2])

        # Check if start or goal is out of bounds
        if not (0 <= start_grid[0] < self.grid_size and 0 <= start_grid[1] < self.grid_size):
            return np.linalg.norm(goal_pos[:2] - start_pos[:2])  # Fallback to Euclidean
        if not (0 <= goal_grid[0] < self.grid_size and 0 <= goal_grid[1] < self.grid_size):
            return np.linalg.norm(goal_pos[:2] - start_pos[:2])  # Fallback to Euclidean

        # Run A* planning (PythonRobotics implementation)
        rx, ry = self.planning(start_grid[0], start_grid[1], goal_grid[0], goal_grid[1])

        if rx is None or len(rx) == 0:
            # No path found, return large penalty
            return np.linalg.norm(goal_pos[:2] - start_pos[:2]) * 2.0

        # Calculate path length in grid units
        path_length = 0.0
        for i in range(len(rx) - 1):
            dx = rx[i+1] - rx[i]
            dy = ry[i+1] - ry[i]
            path_length += math.sqrt(dx**2 + dy**2)

        # Convert to meters (multiply by cell_size)
        path_length *= self.cell_size

        # Add Z-axis distance (vertical movement)
        if len(start_pos) > 2 and len(goal_pos) > 2:
            path_length += abs(goal_pos[2] - start_pos[2])

        return path_length

    def check_reachability(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> bool:
        """
        Quick reachability check using A*.
        Used for calculating reachability flag in observations.

        Args:
            start_pos: Start position (x, y, z) or (x, y)
            goal_pos: Goal position (x, y, z) or (x, y)

        Returns:
            True if A* can find a path, False otherwise
        """
        # Convert to grid coordinates
        start_grid = self._world_to_grid(start_pos[:2])
        goal_grid = self._world_to_grid(goal_pos[:2])

        # Check if start or goal is out of bounds
        if not (0 <= start_grid[0] < self.grid_size and 0 <= start_grid[1] < self.grid_size):
            return False
        if not (0 <= goal_grid[0] < self.grid_size and 0 <= goal_grid[1] < self.grid_size):
            return False

        # Run A* planning
        rx, ry = self.planning(start_grid[0], start_grid[1], goal_grid[0], goal_grid[1])

        # Return True if path found
        return rx is not None and len(rx) > 0

    @staticmethod
    def calc_heuristic(n1: Node, n2: Node) -> float:
        """
        Calculate heuristic cost (Euclidean distance) - from PythonRobotics.

        Args:
            n1: First node
            n2: Second node

        Returns:
            Heuristic cost
        """
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_index(self, node: Node) -> int:
        """
        Calculate grid index from node position - from PythonRobotics.

        Args:
            node: Node to calculate index for

        Returns:
            Grid index
        """
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node: Node, goal_node: Node) -> bool:
        """
        Verify if node is valid (not obstacle, within bounds) - from PythonRobotics.

        PICK-AND-PLACE EXCEPTION: Goal cell is always valid (we can pick the target cube).

        Args:
            node: Node to verify
            goal_node: Goal node (for pick-and-place exception)

        Returns:
            True if node is valid, False otherwise
        """
        # PICK-AND-PLACE EXCEPTION: Goal is always reachable
        if node.x == goal_node.x and node.y == goal_node.y:
            return True

        # Check bounds
        if node.x < self.min_x:
            return False
        elif node.y < self.min_y:
            return False
        elif node.x >= self.max_x:
            return False
        elif node.y >= self.max_y:
            return False

        # Collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_final_path(self, goal_node: Node, closed_set: dict) -> Tuple[List[int], List[int]]:
        """
        Calculate final path by backtracking from goal - from PythonRobotics.

        Args:
            goal_node: Goal node
            closed_set: Closed set dictionary

        Returns:
            Tuple of (rx, ry) - lists of x and y coordinates
        """
        # Generate final course
        rx, ry = [goal_node.x], [goal_node.y]
        parent_index = goal_node.parent_index

        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(n.x)
            ry.append(n.y)
            parent_index = n.parent_index

        return rx, ry

    def planning(self, sx: int, sy: int, gx: int, gy: int) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        """
        A* path search - from PythonRobotics.

        Args:
            sx: start x position (grid index)
            sy: start y position (grid index)
            gx: goal x position (grid index)
            gy: goal y position (grid index)

        Returns:
            rx: x position list of the final path
            ry: y position list of the final path
        """
        start_node = Node(sx, sy, 0.0, -1)
        goal_node = Node(gx, gy, 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                # No path found
                return None, None

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o])
            )
            current = open_set[c_id]

            # Goal found
            if current.x == goal_node.x and current.y == goal_node.y:
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove from open set
            del open_set[c_id]

            # Add to closed set
            closed_set[c_id] = current

            # Expand search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = Node(current.x + self.motion[i][0],
                           current.y + self.motion[i][1],
                           current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node, goal_node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry


class RRTPathEstimator:
    """
    RRT path estimator for grid-based environments (adapted from PythonRobotics).
    Estimates path length considering obstacles using RRT algorithm.

    Based on: C:/isaacsim/cobotproject/PythonRobotics/PathPlanning/RRT/rrt.py
    """

    class Node:
        """
        RRT Node (from PythonRobotics).
        """
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    def __init__(self, grid_size: int = 6, cell_size: float = 0.26, franka_controller=None):
        """
        Initialize RRT path estimator.

        Args:
            grid_size: Size of the grid (e.g., 6 for 6x6)
            cell_size: Size of each grid cell in meters (UPDATED: 0.26m to ensure 15cm gripper fits between objects)
            franka_controller: Reference to Franka controller with RRT planner (optional)
        """
        self.grid_size = grid_size
        # UPDATED: Increased from 0.20 to 0.26 to ensure gripper (15cm) can fit between objects
        # With 26cm spacing: cube-to-cube gap = 26cm - 5.15cm = 20.85cm (gripper 15cm fits with 5.85cm clearance)
        self.cell_size = cell_size  # default 0.26m for gripper width + safety margin
        self.grid_center = np.array([0.45, -0.10])  # Grid center in world coordinates
        self.franka_controller = franka_controller
        self.rrt_planner = None

        # RRT parameters (OPTIMIZED for speed with larger cells)
        # With 26cm cells (vs 20cm), grid is more sparse, so we can use more aggressive parameters
        self.expand_dis = 1.5  # Expansion distance in grid cells (INCREASED for faster exploration)
        self.path_resolution = 1.0  # Path resolution for steering (INCREASED - fewer collision checks)
        self.max_iter = 10000  # Maximum iterations (kept same as requested)
        self.goal_sample_rate = 40  # Goal sampling rate (%) - INCREASED for faster convergence
        self.robot_radius = 0.0  # Robot radius for collision checking (handled by cell size)

        # Occupancy grid (same as A*)
        self.obstacle_map = np.zeros((grid_size, grid_size), dtype=bool)
        self.obstacle_list = []  # List of obstacles in (x, y, size) format for RRT

    def _world_to_grid(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates (MUST match A* exactly)"""
        # Calculate grid extent (same as A*)
        grid_extent = (self.grid_size - 1) * self.cell_size
        start_x = self.grid_center[0] - (grid_extent / 2.0)
        start_y = self.grid_center[1] - (grid_extent / 2.0)

        # Convert to grid coordinates using ROUNDING (same as A*)
        grid_x = round((world_pos[0] - start_x) / self.cell_size)
        grid_y = round((world_pos[1] - start_y) / self.cell_size)

        # CLAMP to valid grid bounds [0, grid_size-1]
        grid_x = max(0, min(self.grid_size - 1, grid_x))
        grid_y = max(0, min(self.grid_size - 1, grid_y))

        return grid_x, grid_y

    def _grid_to_world(self, grid_x: int, grid_y: int) -> np.ndarray:
        """Convert grid coordinates to world coordinates (MUST match A* exactly)"""
        # Calculate grid extent (same as A*)
        grid_extent = (self.grid_size - 1) * self.cell_size
        start_x = self.grid_center[0] - (grid_extent / 2.0)
        start_y = self.grid_center[1] - (grid_extent / 2.0)

        # Convert grid to world
        world_x = start_x + (grid_x * self.cell_size)
        world_y = start_y + (grid_y * self.cell_size)

        return np.array([world_x, world_y])

    def update_occupancy_grid(self, object_positions: List[np.ndarray], obstacle_positions: List[np.ndarray]):
        """Update occupancy grid with obstacles (same as A*)"""
        self.obstacle_map.fill(False)
        self.obstacle_list = []

        # Update grid map - use different obstacle sizes for cubes vs static obstacles
        # CUBES (object_positions): Size 0.7 to prevent paths from passing too close to other cubes
        # STATIC OBSTACLES (obstacle_positions): Size 0.51 to keep original behavior
        # With robot_radius=0.0, collision threshold = size
        # With path_resolution=0.5, intermediate points at distance 0.5 will collide if size > 0.5
        for obj_pos in object_positions:
            grid_x, grid_y = self._world_to_grid(obj_pos[:2])
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                self.obstacle_map[grid_y, grid_x] = True
                # Add to obstacle list for RRT (x, y, size) - size 0.7 for cubes (larger safety margin)
                self.obstacle_list.append((float(grid_x), float(grid_y), 0.7))

        for obs_pos in obstacle_positions:
            grid_x, grid_y = self._world_to_grid(obs_pos[:2])
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                self.obstacle_map[grid_y, grid_x] = True
                # Add to obstacle list for RRT (x, y, size) - size 0.51 for static obstacles (original)
                self.obstacle_list.append((float(grid_x), float(grid_y), 0.51))

    def planning(self, sx: int, sy: int, gx: int, gy: int, return_tree: bool = False) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[List]]:
        """
        RRT path planning (adapted from PythonRobotics).

        Args:
            sx, sy: Start grid coordinates
            gx, gy: Goal grid coordinates
            return_tree: If True, also return the tree structure for visualization

        Returns:
            rx, ry: Lists of x and y coordinates of path (from start to goal)
            tree_edges: List of (parent, child) tuples representing tree edges (only if return_tree=True)
        """
        import random
        import math

        # Validate start and goal
        if not (0 <= sx < self.grid_size and 0 <= sy < self.grid_size):
            return (None, None, None) if return_tree else (None, None)
        if not (0 <= gx < self.grid_size and 0 <= gy < self.grid_size):
            return (None, None, None) if return_tree else (None, None)

        start = self.Node(float(sx), float(sy))
        end = self.Node(float(gx), float(gy))
        node_list = [start]

        for iteration in range(self.max_iter):
            # Random sampling (with goal bias) - from PythonRobotics
            rnd_node = self.get_random_node(end)

            # Find nearest node - from PythonRobotics
            nearest_ind = self.get_nearest_node_index(node_list, rnd_node)
            nearest_node = node_list[nearest_ind]

            # Steer towards random node - from PythonRobotics
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            # Check collision - from PythonRobotics
            if self.check_collision(new_node, self.obstacle_list, self.robot_radius):
                node_list.append(new_node)

                # Check if reached goal - from PythonRobotics
                if self.calc_dist_to_goal(new_node, end) <= self.expand_dis:
                    final_node = self.steer(new_node, end, self.expand_dis)
                    if self.check_collision(final_node, self.obstacle_list, self.robot_radius):
                        # Add final_node to tree and generate path
                        final_node.parent = new_node
                        node_list.append(final_node)

                        # Generate path
                        rx, ry = self.generate_final_course(len(node_list) - 1, node_list)

                        # If tree visualization requested, extract tree edges
                        if return_tree:
                            tree_edges = []
                            for node in node_list:
                                if node.parent is not None:
                                    tree_edges.append(((node.parent.x, node.parent.y), (node.x, node.y)))
                            return rx, ry, tree_edges
                        else:
                            return rx, ry

        # No path found after max iterations
        # Debug: Log why RRT failed
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"RRT failed: max_iter={self.max_iter} reached. Start=({sx},{sy}), Goal=({gx},{gy}), "
                    f"Obstacles={len(self.obstacle_list)}, Tree nodes={len(node_list)}")

        if return_tree:
            return None, None, None
        else:
            return None, None

    def steer(self, from_node, to_node, extend_length=float("inf")):
        """Steer from one node towards another (from PythonRobotics)"""
        import math

        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind, node_list):
        """
        Generate final path from tree nodes (from PythonRobotics).

        Uses only the node positions (x, y), not the path_x/path_y arrays.
        The path_x/path_y arrays are for tree visualization, not the final path.
        """
        # Build path from goal to start by following parent pointers
        path = []
        node = node_list[goal_ind]

        # Follow parent chain back to start (including goal node)
        while node is not None:
            path.append([node.x, node.y])
            node = node.parent

        # Reverse to get START -> GOAL order
        path = list(reversed(path))

        # Extract x and y coordinates as integers, removing consecutive duplicates
        rx, ry = [], []
        for i, (px, py) in enumerate(path):
            ix, iy = int(round(px)), int(round(py))
            # Only add if different from previous point
            if i == 0 or ix != rx[-1] or iy != ry[-1]:
                rx.append(ix)
                ry.append(iy)

        # Validate that integer path doesn't go through obstacles
        # This is critical because rounding can create paths through obstacles!
        path_waypoints = set(zip(rx, ry))
        for obs_x, obs_y, _ in self.obstacle_list:
            obs_grid = (int(round(obs_x)), int(round(obs_y)))
            if obs_grid in path_waypoints:
                # Path goes through obstacle after rounding - reject it!
                return None, None

        return rx, ry

    def calc_dist_to_goal(self, node, goal):
        """Calculate distance to goal (from PythonRobotics)"""
        import math
        dx = node.x - goal.x
        dy = node.y - goal.y
        return math.hypot(dx, dy)

    def get_random_node(self, goal):
        """Get random node with goal bias (from PythonRobotics)"""
        import random
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(0, self.grid_size - 1),
                random.uniform(0, self.grid_size - 1))
        else:  # goal point sampling
            rnd = self.Node(goal.x, goal.y)
        return rnd

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        """Find nearest node index (from PythonRobotics)"""
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        """Calculate distance and angle between nodes (from PythonRobotics)"""
        import math
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    @staticmethod
    def check_collision(node, obstacle_list, robot_radius):
        """Check collision with obstacles (from PythonRobotics)"""
        if node is None:
            return False

        for (ox, oy, size) in obstacle_list:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            min_dist_sq = min(d_list) if d_list else float('inf')
            threshold_sq = (size + robot_radius)**2

            if min_dist_sq <= threshold_sq:
                # Collision detected
                return False  # collision

        return True  # safe

    def estimate_path_length(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> float:
        """
        Estimate path length using RRT planning.

        Args:
            start_pos: Start position (x, y) or (x, y, z)
            goal_pos: Goal position (x, y) or (x, y, z)

        Returns:
            RRT path length in meters, or large penalty if planning fails
        """
        start_grid = self._world_to_grid(start_pos[:2])
        goal_grid = self._world_to_grid(goal_pos[:2])

        rx, ry = self.planning(start_grid[0], start_grid[1], goal_grid[0], goal_grid[1])

        if rx is None or ry is None:
            return 999.0  # Large penalty for unreachable

        # Calculate path length
        length = 0.0
        for i in range(len(rx) - 1):
            pos1 = self._grid_to_world(rx[i], ry[i])
            pos2 = self._grid_to_world(rx[i+1], ry[i+1])
            length += np.linalg.norm(pos2 - pos1)

        return length

    def check_reachability(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> bool:
        """
        Quick reachability check using RRT.
        Used for calculating reachability flag in observations.

        Args:
            start_pos: Start position (x, y) or (x, y, z)
            goal_pos: Goal position (x, y) or (x, y, z)

        Returns:
            True if RRT can find a path, False otherwise
        """
        start_grid = self._world_to_grid(start_pos[:2])
        goal_grid = self._world_to_grid(goal_pos[:2])

        # Check if start or goal is out of bounds
        if not (0 <= start_grid[0] < self.grid_size and 0 <= start_grid[1] < self.grid_size):
            return False
        if not (0 <= goal_grid[0] < self.grid_size and 0 <= goal_grid[1] < self.grid_size):
            return False

        # Run RRT planning
        rx, ry = self.planning(start_grid[0], start_grid[1], goal_grid[0], goal_grid[1])

        # Return True if path found
        return rx is not None and ry is not None and len(rx) > 0

