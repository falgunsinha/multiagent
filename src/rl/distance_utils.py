"""
Utility functions for calculating distances between objects and obstacles.
Supports edge-to-edge distance calculations for accurate collision avoidance.
"""

import numpy as np
from typing import Optional


def calculate_cube_to_cube_edge_distance(pos1: np.ndarray, pos2: np.ndarray, 
                                         half_edge: float = 0.025) -> float:
    """
    Calculate edge-to-edge distance between two cubes using conservative bounding circle.
    
    For axis-aligned cubes, the minimum distance from center to surface varies with angle:
    - Along axes (0°, 90°): half_edge
    - Along diagonals (45°): half_edge * sqrt(2)
    
    We use a conservative bounding circle approach (half_diagonal) that works for all angles.
    
    Args:
        pos1, pos2: Center positions (x, y, z) or (x, y)
        half_edge: Half of cube edge length (default: 0.025m for 5cm cube)
    
    Returns:
        Edge-to-edge distance in meters (clamped to 0.0 if overlapping)
    """
    # Use 2D distance (X, Y only) for obstacle proximity
    center_distance = np.linalg.norm(pos1[:2] - pos2[:2])
    
    # Conservative bounding circle: radius = half_diagonal
    bounding_radius = half_edge * np.sqrt(2)  # 0.025 * 1.414 = 0.0354m
    
    # Subtract both bounding radii to get edge-to-edge distance
    edge_distance = center_distance - (2 * bounding_radius)
    
    return max(0.0, edge_distance)


def calculate_cylinder_to_cylinder_edge_distance(pos1: np.ndarray, pos2: np.ndarray,
                                                 radius: float = 0.025) -> float:
    """
    Calculate edge-to-edge distance between two cylinders.
    
    Cylinders are rotationally symmetric around Z-axis, so this works for all angles.
    
    Args:
        pos1, pos2: Center positions (x, y, z) or (x, y)
        radius: Cylinder radius (default: 0.025m for 5cm diameter)
    
    Returns:
        Edge-to-edge distance in meters (clamped to 0.0 if overlapping)
    """
    # Use 2D distance (X, Y only) for obstacle proximity
    center_distance = np.linalg.norm(pos1[:2] - pos2[:2])
    
    # Subtract both radii to get edge-to-edge distance
    edge_distance = center_distance - (2 * radius)
    
    return max(0.0, edge_distance)


def calculate_object_to_obstacle_edge_distance_conservative(
    obj_pos: np.ndarray,
    obs_pos: np.ndarray,
    obj_radius: float = 0.0354,  # Conservative: cube bounding radius
    obs_radius: float = 0.05     # Conservative: assume 5cm radius obstacle
) -> float:
    """
    Calculate conservative edge-to-edge distance between object and obstacle.
    
    This is a fallback when PyBullet is not available. It treats both objects
    as circles/spheres with conservative radii.
    
    Args:
        obj_pos: Object center position (x, y, z) or (x, y)
        obs_pos: Obstacle center position (x, y, z) or (x, y)
        obj_radius: Conservative radius for object (default: 0.0354m for cube bounding circle)
        obs_radius: Conservative radius for obstacle (default: 0.05m)
    
    Returns:
        Conservative edge-to-edge distance in meters
    """
    # Use 2D distance (X, Y only)
    center_distance = np.linalg.norm(obj_pos[:2] - obs_pos[:2])
    
    # Subtract both radii
    edge_distance = center_distance - obj_radius - obs_radius
    
    return max(0.0, edge_distance)


def calculate_placement_position(
    cube_index: int,
    total_cubes: int,
    container_center: np.ndarray,
    container_dimensions: np.ndarray
) -> np.ndarray:
    """
    Calculate the actual placement position for a cube in the container.
    
    This matches the deployment script's placement logic with grid-based
    placement and asymmetric margins.
    
    Args:
        cube_index: Index of cube in pick order (0 to total_cubes-1)
        total_cubes: Total number of cubes to place
        container_center: Container center position (x, y, z)
        container_dimensions: Container dimensions [length, width, height]
    
    Returns:
        Placement position (x, y, z) where the cube will be placed
    """
    container_length = container_dimensions[0]  # X-axis
    container_width = container_dimensions[1]   # Y-axis
    
    # Calculate grid size from actual number of cubes
    # e.g., 9 cubes -> 3x3, 4 cubes -> 2x2, 6 cubes -> 3x2
    place_grid_size = int(np.ceil(np.sqrt(total_cubes)))
    place_row = cube_index // place_grid_size
    place_col = cube_index % place_grid_size
    
    # Same margins as deployment script (lines 2310-2321)
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
    
    # Calculate usable space with asymmetric margins (lines 2324-2327)
    usable_length = container_length - edge_margin_left - edge_margin_right
    usable_width = container_width - (2 * edge_margin_width)
    spacing_length = usable_length / (place_grid_size - 1) if place_grid_size > 1 else 0.0
    spacing_width = usable_width / (place_grid_size - 1) if place_grid_size > 1 else 0.0
    
    # Start from left edge with larger margin (lines 2330-2333)
    start_x = container_center[0] - (container_length / 2.0) + edge_margin_left
    start_y = container_center[1] - (container_width / 2.0) + edge_margin_width
    cube_x = start_x + (place_row * spacing_length)
    cube_y = start_y + (place_col * spacing_width)
    
    # Return placement position (use container center Z)
    return np.array([cube_x, cube_y, container_center[2]])

