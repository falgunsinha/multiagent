"""
Waypoint Selection Methods for LLM-A* Paper Replication

Implements 4 waypoint selection strategies from the LLM-A* paper:
- Start-Prioritized: Select waypoints closest to start
- Uniform: Uniformly select waypoints
- Random: Randomly select waypoints  
- Goal-Prioritized: Select waypoints closest to goal

Based on: "A 1000× Faster LLM-enhanced Algorithm For Path Planning in Large-scale Grid Maps"
"""

import numpy as np
from typing import List, Tuple
import random


def calculate_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(np.array(point1) - np.array(point2))


def select_waypoints_start(waypoints: List[np.ndarray], start: np.ndarray, goal: np.ndarray, 
                           num_select: int) -> List[np.ndarray]:
    """
    Start-Prioritized Selection: Select waypoints closest to start position
    
    Args:
        waypoints: List of all generated waypoints
        start: Start position
        goal: Goal position
        num_select: Number of waypoints to select
        
    Returns:
        Selected waypoints (sorted by distance from start)
    """
    if len(waypoints) <= num_select:
        return waypoints
    
    # Calculate distance from start for each waypoint
    distances = [(wp, calculate_distance(wp, start)) for wp in waypoints]
    
    # Sort by distance from start (ascending)
    distances.sort(key=lambda x: x[1])
    
    # Select first num_select waypoints
    selected = [wp for wp, _ in distances[:num_select]]
    
    return selected


def select_waypoints_uniform(waypoints: List[np.ndarray], start: np.ndarray, goal: np.ndarray,
                             num_select: int) -> List[np.ndarray]:
    """
    Uniform Selection: Uniformly select waypoints along the path
    
    Args:
        waypoints: List of all generated waypoints
        start: Start position
        goal: Goal position
        num_select: Number of waypoints to select
        
    Returns:
        Uniformly selected waypoints
    """
    if len(waypoints) <= num_select:
        return waypoints
    
    # Calculate uniform indices
    indices = np.linspace(0, len(waypoints) - 1, num_select, dtype=int)
    
    # Select waypoints at uniform intervals
    selected = [waypoints[i] for i in indices]
    
    return selected


def select_waypoints_random(waypoints: List[np.ndarray], start: np.ndarray, goal: np.ndarray,
                            num_select: int, seed: int = None) -> List[np.ndarray]:
    """
    Random Selection: Randomly select waypoints
    
    Args:
        waypoints: List of all generated waypoints
        start: Start position
        goal: Goal position
        num_select: Number of waypoints to select
        seed: Random seed for reproducibility
        
    Returns:
        Randomly selected waypoints
    """
    if len(waypoints) <= num_select:
        return waypoints
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Randomly select waypoints
    selected = random.sample(waypoints, num_select)
    
    return selected


def select_waypoints_goal(waypoints: List[np.ndarray], start: np.ndarray, goal: np.ndarray,
                          num_select: int) -> List[np.ndarray]:
    """
    Goal-Prioritized Selection: Select waypoints closest to goal position
    
    Args:
        waypoints: List of all generated waypoints
        start: Start position
        goal: Goal position
        num_select: Number of waypoints to select
        
    Returns:
        Selected waypoints (sorted by distance from goal)
    """
    if len(waypoints) <= num_select:
        return waypoints
    
    # Calculate distance from goal for each waypoint
    distances = [(wp, calculate_distance(wp, goal)) for wp in waypoints]
    
    # Sort by distance from goal (ascending)
    distances.sort(key=lambda x: x[1])
    
    # Select first num_select waypoints
    selected = [wp for wp, _ in distances[:num_select]]
    
    return selected


def select_waypoints(waypoints: List[np.ndarray], start: np.ndarray, goal: np.ndarray,
                    method: str, num_select: int, seed: int = None) -> List[np.ndarray]:
    """
    Select waypoints using specified method
    
    Args:
        waypoints: List of all generated waypoints
        start: Start position
        goal: Goal position
        method: Selection method ('start', 'uniform', 'random', 'goal')
        num_select: Number of waypoints to select
        seed: Random seed (only used for 'random' method)
        
    Returns:
        Selected waypoints
    """
    method = method.lower()
    
    if method == 'start':
        return select_waypoints_start(waypoints, start, goal, num_select)
    elif method == 'uniform':
        return select_waypoints_uniform(waypoints, start, goal, num_select)
    elif method == 'random':
        return select_waypoints_random(waypoints, start, goal, num_select, seed)
    elif method == 'goal':
        return select_waypoints_goal(waypoints, start, goal, num_select)
    else:
        raise ValueError(f"Unknown waypoint selection method: {method}")

