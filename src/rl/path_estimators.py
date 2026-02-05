import numpy as np
from typing import List, Tuple, Optional
import math


class Node:
    """Node class for A* search"""

    def __init__(self, x: int, y: int, cost: float, parent_index: int):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index

    def __str__(self):
        return f"{self.x},{self.y},{self.cost},{self.parent_index}"


class AStarPathEstimator:
    """A* path estimator for grid-based environments"""

    def __init__(self, grid_size: int = 6, cell_size: float = 0.26):
        """Initialize A* path estimator"""
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.grid_center = np.array([0.45, -0.10])

        self.min_x = 0
        self.min_y = 0
        self.max_x = grid_size
        self.max_y = grid_size
        self.x_width = grid_size
        self.y_width = grid_size

        self.motion = self.get_motion_model()

        self.obstacle_map = [[False for _ in range(self.y_width)]
                            for _ in range(self.x_width)]

    @staticmethod
    def get_motion_model():
        """Get motion model for 8-connected grid"""
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
        """Update occupancy grid based on object and obstacle positions"""

        for ix in range(self.x_width):

            for iy in range(self.y_width):
                self.obstacle_map[ix][iy] = False

        for pos in object_positions:
            grid_x, grid_y = self._world_to_grid(pos[:2])

            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                self.obstacle_map[grid_x][grid_y] = True

        if obstacle_positions:

            for pos in obstacle_positions:
                grid_x, grid_y = self._world_to_grid(pos[:2])

                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    self.obstacle_map[grid_x][grid_y] = True

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

    def estimate_path_length(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> float:
        """Estimate path length from start to goal using A*"""
        start_grid = self._world_to_grid(start_pos[:2])
        goal_grid = self._world_to_grid(goal_pos[:2])

        if not (0 <= start_grid[0] < self.grid_size and 0 <= start_grid[1] < self.grid_size):
            return np.linalg.norm(goal_pos[:2] - start_pos[:2])

        if not (0 <= goal_grid[0] < self.grid_size and 0 <= goal_grid[1] < self.grid_size):
            return np.linalg.norm(goal_pos[:2] - start_pos[:2])

        rx, ry = self.planning(start_grid[0], start_grid[1], goal_grid[0], goal_grid[1])

        if rx is None or len(rx) == 0:
            return np.linalg.norm(goal_pos[:2] - start_pos[:2]) * 2.0

        path_length = 0.0

        for i in range(len(rx) - 1):
            dx = rx[i+1] - rx[i]
            dy = ry[i+1] - ry[i]
            path_length += math.sqrt(dx**2 + dy**2)

        path_length *= self.cell_size

        if len(start_pos) > 2 and len(goal_pos) > 2:
            path_length += abs(goal_pos[2] - start_pos[2])

        return path_length

    def check_reachability(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> bool:
        """Quick reachability check using A*"""
        start_grid = self._world_to_grid(start_pos[:2])
        goal_grid = self._world_to_grid(goal_pos[:2])

        if not (0 <= start_grid[0] < self.grid_size and 0 <= start_grid[1] < self.grid_size):
            return False

        if not (0 <= goal_grid[0] < self.grid_size and 0 <= goal_grid[1] < self.grid_size):
            return False

        rx, _ = self.planning(start_grid[0], start_grid[1], goal_grid[0], goal_grid[1])

        return rx is not None and len(rx) > 0

    @staticmethod
    def calc_heuristic(n1: Node, n2: Node) -> float:
        """Calculate heuristic cost (Euclidean distance)"""
        w = 1.0
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_index(self, node: Node) -> int:
        """Calculate grid index from node position"""
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node: Node, goal_node: Node) -> bool:
        """Verify if node is valid (not obstacle, within bounds)"""

        if node.x == goal_node.x and node.y == goal_node.y:
            return True

        if node.x < self.min_x:
            return False

        elif node.y < self.min_y:
            return False

        elif node.x >= self.max_x:
            return False

        elif node.y >= self.max_y:
            return False

        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_final_path(self, goal_node: Node, closed_set: dict) -> Tuple[List[int], List[int]]:
        """Calculate final path by backtracking from goal"""
        rx, ry = [goal_node.x], [goal_node.y]
        parent_index = goal_node.parent_index

        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(n.x)
            ry.append(n.y)
            parent_index = n.parent_index

        return rx, ry

    def planning(self, sx: int, sy: int, gx: int, gy: int) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        """A* path search"""
        start_node = Node(sx, sy, 0.0, -1)
        goal_node = Node(gx, gy, 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:

            if len(open_set) == 0:
                return None, None

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o])
            )
            current = open_set[c_id]

            if current.x == goal_node.x and current.y == goal_node.y:
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            del open_set[c_id]

            closed_set[c_id] = current

            for i, _ in enumerate(self.motion):
                node = Node(current.x + self.motion[i][0],
                           current.y + self.motion[i][1],
                           current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                if not self.verify_node(node, goal_node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node

                else:

                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry


class RRTPathEstimator:
    """RRT path estimator for grid-based environments"""

    class Node:
        """RRT Node"""

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    def __init__(self, grid_size: int = 6, cell_size: float = 0.26, franka_controller=None):
        """Initialize RRT path estimator"""
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.grid_center = np.array([0.45, -0.10])
        self.franka_controller = franka_controller
        self.rrt_planner = None

        self.expand_dis = 1.5
        self.path_resolution = 1.0
        self.max_iter = 10000
        self.goal_sample_rate = 40
        self.robot_radius = 0.0

        self.obstacle_map = np.zeros((grid_size, grid_size), dtype=bool)
        self.obstacle_list = []

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
        self.obstacle_list = []

        for obj_pos in object_positions:
            grid_x, grid_y = self._world_to_grid(obj_pos[:2])

            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                self.obstacle_map[grid_y, grid_x] = True
                self.obstacle_list.append((float(grid_x), float(grid_y), 0.7))

        for obs_pos in obstacle_positions:
            grid_x, grid_y = self._world_to_grid(obs_pos[:2])

            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                self.obstacle_map[grid_y, grid_x] = True
                self.obstacle_list.append((float(grid_x), float(grid_y), 0.51))

    def planning(self, sx: int, sy: int, gx: int, gy: int, return_tree: bool = False) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[List]]:
        """RRT path planning"""

        if not (0 <= sx < self.grid_size and 0 <= sy < self.grid_size):
            return (None, None, None) if return_tree else (None, None)

        if not (0 <= gx < self.grid_size and 0 <= gy < self.grid_size):
            return (None, None, None) if return_tree else (None, None)

        start = self.Node(float(sx), float(sy))
        end = self.Node(float(gx), float(gy))
        node_list = [start]

        for _ in range(self.max_iter):
            rnd_node = self.get_random_node(end)

            nearest_ind = self.get_nearest_node_index(node_list, rnd_node)
            nearest_node = node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_list, self.robot_radius):
                node_list.append(new_node)

                if self.calc_dist_to_goal(new_node, end) <= self.expand_dis:
                    final_node = self.steer(new_node, end, self.expand_dis)

                    if self.check_collision(final_node, self.obstacle_list, self.robot_radius):
                        final_node.parent = new_node
                        node_list.append(final_node)

                        rx, ry = self.generate_final_course(len(node_list) - 1, node_list)

                        if return_tree:
                            tree_edges = []

                            for node in node_list:

                                if node.parent is not None:
                                    tree_edges.append(((node.parent.x, node.parent.y), (node.x, node.y)))

                            return rx, ry, tree_edges

                        else:
                            return rx, ry

        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"RRT failed: max_iter={self.max_iter} reached. Start=({sx},{sy}), Goal=({gx},{gy}), "
                    f"Obstacles={len(self.obstacle_list)}, Tree nodes={len(node_list)}")

        if return_tree:
            return None, None, None

        else:
            return None, None

    def steer(self, from_node, to_node, extend_length=float("inf")):
        """Steer from one node towards another"""
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
        """Generate final path from tree nodes"""
        path = []
        node = node_list[goal_ind]

        while node is not None:
            path.append([node.x, node.y])
            node = node.parent

        path = list(reversed(path))

        rx, ry = [], []

        for i, (px, py) in enumerate(path):
            ix, iy = int(round(px)), int(round(py))

            if i == 0 or ix != rx[-1] or iy != ry[-1]:
                rx.append(ix)
                ry.append(iy)

        path_waypoints = set(zip(rx, ry))

        for obs_x, obs_y, _ in self.obstacle_list:
            obs_grid = (int(round(obs_x)), int(round(obs_y)))

            if obs_grid in path_waypoints:
                return None, None

        return rx, ry

    def calc_dist_to_goal(self, node, goal):
        """Calculate distance to goal"""
        import math
        dx = node.x - goal.x
        dy = node.y - goal.y
        return math.hypot(dx, dy)

    def get_random_node(self, goal):
        """Get random node with goal bias"""
        import random

        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(0, self.grid_size - 1),
                random.uniform(0, self.grid_size - 1))

        else:
            rnd = self.Node(goal.x, goal.y)

        return rnd

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        """Find nearest node index"""
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        """Calculate distance and angle between nodes"""
        import math
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    @staticmethod
    def check_collision(node, obstacle_list, robot_radius):
        """Check collision with obstacles"""

        if node is None:
            return False

        for (ox, oy, size) in obstacle_list:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            min_dist_sq = min(d_list) if d_list else float('inf')
            threshold_sq = (size + robot_radius)**2

            if min_dist_sq <= threshold_sq:
                return False

        return True

    def estimate_path_length(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> float:
        """Estimate path length using RRT planning"""
        start_grid = self._world_to_grid(start_pos[:2])
        goal_grid = self._world_to_grid(goal_pos[:2])

        rx, ry = self.planning(start_grid[0], start_grid[1], goal_grid[0], goal_grid[1])

        if rx is None or ry is None:
            return 999.0

        length = 0.0

        for i in range(len(rx) - 1):
            pos1 = self._grid_to_world(rx[i], ry[i])
            pos2 = self._grid_to_world(rx[i+1], ry[i+1])
            length += np.linalg.norm(pos2 - pos1)

        return length

    def check_reachability(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> bool:
        """Quick reachability check using RRT"""
        start_grid = self._world_to_grid(start_pos[:2])
        goal_grid = self._world_to_grid(goal_pos[:2])

        if not (0 <= start_grid[0] < self.grid_size and 0 <= start_grid[1] < self.grid_size):
            return False

        if not (0 <= goal_grid[0] < self.grid_size and 0 <= goal_grid[1] < self.grid_size):
            return False

        rx, ry = self.planning(start_grid[0], start_grid[1], goal_grid[0], goal_grid[1])

        return rx is not None and ry is not None and len(rx) > 0

