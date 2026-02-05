
import numpy as np
from typing import Optional


def calculate_cube_to_cube_edge_distance(pos1: np.ndarray, pos2: np.ndarray,
                                         half_edge: float = 0.025) -> float:
    """Calculate edge-to-edge distance between two cubes using conservative bounding circle"""
    center_distance = np.linalg.norm(pos1[:2] - pos2[:2])
    bounding_radius = half_edge * np.sqrt(2)
    edge_distance = center_distance - (2 * bounding_radius)
    return max(0.0, edge_distance)


def calculate_cylinder_to_cylinder_edge_distance(pos1: np.ndarray, pos2: np.ndarray,
                                                 radius: float = 0.025) -> float:
    """Calculate edge-to-edge distance between two cylinders"""
    center_distance = np.linalg.norm(pos1[:2] - pos2[:2])
    edge_distance = center_distance - (2 * radius)
    return max(0.0, edge_distance)


def calculate_object_to_obstacle_edge_distance_conservative(
    obj_pos: np.ndarray,
    obs_pos: np.ndarray,
    obj_radius: float = 0.0354,
    obs_radius: float = 0.05
) -> float:
    """Calculate conservative edge-to-edge distance between object and obstacle"""
    center_distance = np.linalg.norm(obj_pos[:2] - obs_pos[:2])
    edge_distance = center_distance - obj_radius - obs_radius
    return max(0.0, edge_distance)


def calculate_placement_position(
    cube_index: int,
    total_cubes: int,
    container_center: np.ndarray,
    container_dimensions: np.ndarray
) -> np.ndarray:
    """Calculate the actual placement position for a cube in the container"""
    container_length = container_dimensions[0]
    container_width = container_dimensions[1]

    place_grid_size = int(np.ceil(np.sqrt(total_cubes)))
    place_row = cube_index // place_grid_size
    place_col = cube_index % place_grid_size

    if total_cubes <= 4:
        edge_margin_left = 0.11
        edge_margin_right = 0.11
        edge_margin_width = 0.10
    elif total_cubes <= 9:
        edge_margin_left = 0.11
        edge_margin_right = 0.11
        edge_margin_width = 0.10
    else:
        edge_margin_left = 0.09
        edge_margin_right = 0.09
        edge_margin_width = 0.07

    usable_length = container_length - edge_margin_left - edge_margin_right
    usable_width = container_width - (2 * edge_margin_width)
    spacing_length = usable_length / (place_grid_size - 1) if place_grid_size > 1 else 0.0
    spacing_width = usable_width / (place_grid_size - 1) if place_grid_size > 1 else 0.0

    start_x = container_center[0] - (container_length / 2.0) + edge_margin_left
    start_y = container_center[1] - (container_width / 2.0) + edge_margin_width
    cube_x = start_x + (place_row * spacing_length)
    cube_y = start_y + (place_col * spacing_width)

    return np.array([cube_x, cube_y, container_center[2]])

