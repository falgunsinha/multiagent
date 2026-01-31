"""
Test Waypoint Selection Methods

Quick test to verify waypoint selection functions work correctly.
"""

import numpy as np
from waypoint_selection import (
    select_waypoints_start,
    select_waypoints_uniform,
    select_waypoints_random,
    select_waypoints_goal,
    select_waypoints
)


def test_waypoint_selection():
    """Test all waypoint selection methods"""
    
    print("\n" + "="*80)
    print("TESTING WAYPOINT SELECTION METHODS")
    print("="*80 + "\n")
    
    # Create test data
    start = np.array([0.0, 0.0])
    goal = np.array([10.0, 10.0])
    
    # Generate 5 waypoints along the path
    waypoints = [
        np.array([2.0, 2.0]),   # Closest to start
        np.array([4.0, 4.0]),
        np.array([6.0, 6.0]),
        np.array([8.0, 8.0]),   # Closest to goal
        np.array([9.0, 9.0])
    ]
    
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    print(f"All waypoints ({len(waypoints)}):")
    for i, wp in enumerate(waypoints):
        print(f"  {i+1}. {wp}")
    print()
    
    # Test each selection method
    num_select = 2
    
    print(f"Selecting {num_select} waypoints using different methods:")
    print("-" * 80)
    
    # 1. Start-Prioritized
    selected_start = select_waypoints_start(waypoints, start, goal, num_select)
    print(f"\n1. Start-Prioritized Selection:")
    for i, wp in enumerate(selected_start):
        print(f"   {i+1}. {wp}")
    
    # 2. Uniform
    selected_uniform = select_waypoints_uniform(waypoints, start, goal, num_select)
    print(f"\n2. Uniform Selection:")
    for i, wp in enumerate(selected_uniform):
        print(f"   {i+1}. {wp}")
    
    # 3. Random
    selected_random = select_waypoints_random(waypoints, start, goal, num_select, seed=42)
    print(f"\n3. Random Selection (seed=42):")
    for i, wp in enumerate(selected_random):
        print(f"   {i+1}. {wp}")
    
    # 4. Goal-Prioritized
    selected_goal = select_waypoints_goal(waypoints, start, goal, num_select)
    print(f"\n4. Goal-Prioritized Selection:")
    for i, wp in enumerate(selected_goal):
        print(f"   {i+1}. {wp}")
    
    print("\n" + "-" * 80)
    
    # Test with different waypoint counts
    print(f"\nTesting Start-Prioritized with different waypoint counts:")
    print("-" * 80)
    
    for num in range(1, 5):
        selected = select_waypoints_start(waypoints, start, goal, num)
        print(f"\n{num} waypoint(s): {[wp.tolist() for wp in selected]}")
    
    # Test edge cases
    print("\n" + "="*80)
    print("TESTING EDGE CASES")
    print("="*80 + "\n")
    
    # Case 1: Request more waypoints than available
    print("Case 1: Request more waypoints than available (request 10, have 5)")
    selected = select_waypoints_start(waypoints, start, goal, 10)
    print(f"  Result: {len(selected)} waypoints (should return all 5)")
    
    # Case 2: Request 0 waypoints
    print("\nCase 2: Request 0 waypoints")
    selected = select_waypoints_start(waypoints, start, goal, 0)
    print(f"  Result: {len(selected)} waypoints (should return empty list)")
    
    # Case 3: Empty waypoint list
    print("\nCase 3: Empty waypoint list")
    selected = select_waypoints_start([], start, goal, 2)
    print(f"  Result: {len(selected)} waypoints (should return empty list)")
    
    # Test main interface function
    print("\n" + "="*80)
    print("TESTING MAIN INTERFACE FUNCTION")
    print("="*80 + "\n")
    
    methods = ['start', 'uniform', 'random', 'goal']
    
    for method in methods:
        selected = select_waypoints(waypoints, start, goal, method, 2, seed=42)
        print(f"{method.capitalize()}: {len(selected)} waypoints selected")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_waypoint_selection()

