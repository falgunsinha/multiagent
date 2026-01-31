"""
Quick Test Script for Motion Planners

Tests each planner with a simple scenario to verify they work correctly.

Usage:
    python test_planners.py
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.experiments.motionplanners.planners.rrt_planner import PythonRoboticsRRTPlanner
from scripts.experiments.motionplanners.planners.astar_planner import AStarPlanner
from scripts.experiments.motionplanners.planners.prm_planner import PRMPlanner
from scripts.experiments.motionplanners.planners.rrtstar_planner import RRTStarPlanner


def test_planner(planner, planner_name):
    """Test a single planner with a simple scenario"""
    print(f"\n{'='*60}")
    print(f"Testing {planner_name}")
    print(f"{'='*60}")
    
    # Simple test scenario
    start_pos = np.array([0.3, -0.2])
    goal_pos = np.array([0.6, 0.1])
    
    # Simple obstacles
    obstacles = [
        [0.45, -0.05, 0.05],  # [x, y, radius]
        [0.5, 0.0, 0.05],
    ]
    
    print(f"Start: {start_pos}")
    print(f"Goal: {goal_pos}")
    print(f"Obstacles: {len(obstacles)}")
    
    # Plan path
    try:
        path, metrics = planner.plan(start_pos, goal_pos, obstacles)
        
        if metrics.success:
            print(f"\n✓ SUCCESS!")
            print(f"  Planning Time: {metrics.planning_time:.4f}s")
            print(f"  Path Length: {metrics.path_length:.4f}m")
            print(f"  Waypoints: {metrics.num_waypoints}")
            print(f"  Smoothness: {metrics.smoothness:.4f}")
            print(f"  Energy: {metrics.energy:.4f}")
            return True
        else:
            print(f"\n✗ FAILED - No path found")
            print(f"  Planning Time: {metrics.planning_time:.4f}s")
            return False
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all planners"""
    print("\n" + "="*60)
    print("MOTION PLANNER TEST SUITE")
    print("="*60)
    
    results = {}
    
    # Test RRT
    print("\n[1/4] Testing RRT...")
    rrt_config = {
        'expand_dis': 0.1,
        'path_resolution': 0.05,
        'goal_sample_rate': 5,
        'max_iter': 500,
        'robot_radius': 0.05,
        'rand_area': [-2, 2]
    }
    rrt = PythonRoboticsRRTPlanner(rrt_config)
    results['RRT'] = test_planner(rrt, "RRT")
    
    # Test A*
    print("\n[2/4] Testing A*...")
    astar_config = {
        'resolution': 0.05,
        'robot_radius': 0.05
    }
    astar = AStarPlanner(astar_config)
    results['A*'] = test_planner(astar, "A*")
    
    # Test PRM
    print("\n[3/4] Testing PRM...")
    prm_config = {
        'n_sample': 500,
        'n_knn': 10,
        'max_edge_len': 30.0,
        'robot_radius': 0.05
    }
    prm = PRMPlanner(prm_config)
    results['PRM'] = test_planner(prm, "PRM")
    
    # Test RRT*
    print("\n[4/4] Testing RRT*...")
    rrtstar_config = {
        'expand_dis': 0.1,
        'path_resolution': 0.05,
        'goal_sample_rate': 20,
        'max_iter': 500,
        'robot_radius': 0.05,
        'connect_circle_dist': 0.5,
        'rand_area': [-2, 2]
    }
    rrtstar = RRTStarPlanner(rrtstar_config)
    results['RRT*'] = test_planner(rrtstar, "RRT*")
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for planner_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{planner_name:<15} {status}")
    
    print("="*60)
    
    # Overall result
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All tests passed! Ready to run experiments.")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

