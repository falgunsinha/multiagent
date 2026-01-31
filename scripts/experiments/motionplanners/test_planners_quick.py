"""
Quick test script to check if planners can find paths with different obstacle densities.
Runs inside Isaac Sim environment to test all 8 planners including Isaac Sim RRT.
No experiment logging - just success/fail checking.

Usage:
    C:\isaacsim\python.bat test_planners_quick.py
"""

# Import Isaac Sim first
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})  # Run headless for speed

import sys
import os
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import time

# Disable matplotlib display to prevent warnings and hanging
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Isaac Sim imports
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.utils.rotations import euler_angles_to_quats
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.robot_motion.motion_generation.lula import RRT
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from pxr import UsdPhysics

# Add project root to path
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import planners
from scripts.experiments.motionplanners.planners.rrt_planner import PythonRoboticsRRTPlanner, IsaacSimRRTPlanner
from scripts.experiments.motionplanners.planners.astar_planner import AStarPlanner
from scripts.experiments.motionplanners.planners.prm_planner import PRMPlanner
from scripts.experiments.motionplanners.planners.rrtstar_planner import RRTStarPlanner
from scripts.experiments.motionplanners.planners.rrtstar_reedsshepp_planner import RRTStarReedsSheppPlanner
from scripts.experiments.motionplanners.planners.lqr_rrtstar_planner import LQRRRTStarPlanner


def create_planner(planner_name):
    """Create planner instance with configuration"""
    if planner_name == 'rrt':
        config = {
            'expand_dis': 0.5,
            'path_resolution': 0.1,
            'goal_sample_rate': 5,
            'max_iter': 1000,
            'robot_radius': 0.08,  # 8cm - Franka end-effector radius
        }
        return PythonRoboticsRRTPlanner(config)

    elif planner_name == 'astar':
        config = {
            'resolution': 0.1,
            'robot_radius': 0.08  # 8cm - Franka end-effector radius
        }
        return AStarPlanner(config)

    elif planner_name == 'prm':
        config = {
            'n_sample': 500,
            'n_knn': 10,
            'max_edge_len': 2.0,
            'robot_radius': 0.08  # 8cm - Franka end-effector radius
        }
        return PRMPlanner(config)

    elif planner_name == 'rrtstar':
        config = {
            'expand_dis': 0.5,
            'path_resolution': 0.1,
            'goal_sample_rate': 20,
            'max_iter': 1000,
            'robot_radius': 0.08,  # 8cm - Franka end-effector radius
            'connect_circle_dist': 2.0,
        }
        return RRTStarPlanner(config)

    elif planner_name == 'rrtstar_rs':
        config = {
            'max_iter': 1000,
            'step_size': 0.2,
            'connect_circle_dist': 2.0,
            'robot_radius': 0.08,  # 8cm - Franka end-effector radius
        }
        return RRTStarReedsSheppPlanner(config)

    elif planner_name == 'lqr_rrtstar':
        config = {
            'max_iter': 1000,
            'goal_sample_rate': 10,
            'robot_radius': 0.08,  # 8cm - Franka end-effector radius
            'connect_circle_dist': 2.0,
            'step_size': 0.2,
        }
        return LQRRRTStarPlanner(config)

    elif planner_name == 'lqr':
        config = {
            'dt': 0.1,
            'max_time': 100.0,
            'robot_radius': 0.08  # 8cm - Franka end-effector radius
        }
        return LQRPlanner(config)

    return None


def generate_obstacles(obstacle_density, obstacle_type, grid_size=4):
    """Generate obstacles for testing"""
    cell_size = 0.20 if grid_size > 3 else 0.22
    grid_center = np.array([0.30, 0.50])
    grid_extent = (grid_size - 1) * cell_size
    
    grid_area = grid_extent * grid_extent
    
    # Spawning bounds
    start_x = grid_center[0] - (grid_extent / 2.0) - 0.05
    end_x = grid_center[0] + (grid_extent / 2.0) + 0.05
    start_y = grid_center[1] - (grid_extent / 2.0) - 0.05
    end_y = grid_center[1] + (grid_extent / 2.0) + 0.05
    
    # Calculate number of obstacles
    workspace_area = grid_area
    
    if obstacle_type == 'cube':
        obstacle_size = 0.05
        obstacle_area = obstacle_size * obstacle_size
        num_obstacles = int((workspace_area * obstacle_density) / obstacle_area)
    elif obstacle_type == 'bar':
        bar_length = 0.20
        bar_width = 0.05
        obstacle_area = bar_length * bar_width
        num_obstacles = int((workspace_area * obstacle_density) / obstacle_area)
    elif obstacle_type == 'giant_bar':
        bar_length = 0.40
        bar_width = 0.08
        obstacle_area = bar_length * bar_width
        num_obstacles = int((workspace_area * obstacle_density) / obstacle_area)
    
    obstacles = []
    np.random.seed(42)  # Fixed seed for reproducibility

    for _ in range(num_obstacles):
        x = np.random.uniform(start_x, end_x)
        y = np.random.uniform(start_y, end_y)
        
        if obstacle_type == 'cube':
            scale = np.array([0.05, 0.05, 0.05])
        elif obstacle_type == 'bar':
            bar_length = 0.20
            bar_width = 0.05
            angle = np.random.choice([0, 1])
            scale = np.array([bar_length, bar_width, 0.05]) if angle == 0 else np.array([bar_width, bar_length, 0.05])
        else:  # giant_bar
            bar_length = 0.40
            bar_width = 0.08
            angle = np.random.choice([0, 1])
            scale = np.array([bar_length, bar_width, 0.08]) if angle == 0 else np.array([bar_width, bar_length, 0.08])
        
        # Convert to 2D obstacle (circle with radius = diagonal/2)
        length_x = scale[0]
        width_y = scale[1]
        diagonal_2d = np.sqrt(length_x**2 + width_y**2)
        radius = diagonal_2d / 2.0
        
        obstacles.append([x, y, radius])
    
    return obstacles


if __name__ == "__main__":
    print("="*80)
    print("QUICK PLANNER TEST - SUCCESS/FAIL CHECK")
    print("="*80)

    # 6 2D planners (Isaac Sim RRT requires full scene setup, LQR hangs with obstacles)
    planners = ['rrt', 'astar', 'prm', 'rrtstar', 'rrtstar_rs', 'lqr_rrtstar']
    densities = [0.10, 0.25, 0.40]  # 10%, 25%, 40%
    obstacle_types = ['cube', 'bar', 'giant_bar']

    # Test positions (1 cube, 1 trial per config)
    robot_pos = np.array([0.0, 0.0])
    pick_pos = np.array([-0.5, -0.3])
    place_pos = np.array([0.3, 0.5])

    print(f"\nTest Configuration:")
    print(f"  Planners: {len(planners)} ({', '.join(planners)})")
    print(f"  Densities: {len(densities)} ({', '.join([f'{d*100:.0f}%' for d in densities])})")
    print(f"  Obstacle types: {len(obstacle_types)} ({', '.join(obstacle_types)})")
    print(f"  Total tests: {len(planners) * len(densities) * len(obstacle_types)} (1 trial each)")
    print(f"  Robot→Pick: {np.linalg.norm(pick_pos - robot_pos):.3f}m, Pick→Place: {np.linalg.norm(place_pos - pick_pos):.3f}m")
    print(f"\nRunning tests (only failures will be shown)...")
    print()

    # Results table
    results = {}
    test_count = 0
    total_tests = len(planners) * len(densities) * len(obstacle_types)

    for obstacle_type in obstacle_types:
        for density in densities:
            # Generate obstacles
            obstacles = generate_obstacles(density, obstacle_type, grid_size=4)

            for planner_name in planners:
                test_count += 1
                key = (obstacle_type, density, planner_name)

                try:
                    # Create planner
                    planner = create_planner(planner_name)
                    if planner is None:
                        print(f"[{test_count}/{total_tests}] {obstacle_type:10s} {density*100:3.0f}% [{planner_name:15s}] ✗ FAILED (creation)")
                        results[key] = False
                        continue

                    # Test robot→pick
                    planner.reset()
                    path_to_pick, metrics_pick = planner.plan(robot_pos, pick_pos, obstacles)

                    if not metrics_pick.success:
                        print(f"[{test_count}/{total_tests}] {obstacle_type:10s} {density*100:3.0f}% [{planner_name:15s}] ✗ FAILED (robot→pick)")
                        results[key] = False
                        continue

                    # Test pick→place
                    planner.reset()
                    path_to_place, metrics_place = planner.plan(pick_pos, place_pos, obstacles)

                    if not metrics_place.success:
                        print(f"[{test_count}/{total_tests}] {obstacle_type:10s} {density*100:3.0f}% [{planner_name:15s}] ✗ FAILED (pick→place)")
                        results[key] = False
                        continue

                    # Success! (don't print, only show failures)
                    results[key] = True

                except KeyboardInterrupt:
                    print(f"\n[{test_count}/{total_tests}] Test interrupted by user!")
                    results[key] = False
                    raise  # Re-raise to stop the test

                except Exception as e:
                    error_msg = str(e)[:60]
                    print(f"[{test_count}/{total_tests}] {obstacle_type:10s} {density*100:3.0f}% [{planner_name:15s}] ✗ EXCEPTION: {error_msg}")
                    results[key] = False

    # Summary - Comprehensive Table
    print("\n" + "="*80)
    print("SUMMARY - PLANNER SUCCESS/FAIL MATRIX")
    print("="*80)
    print("✓ = Success, ✗ = Failed")

    # Header
    print(f"\n{'Obstacle Type':<15} {'Density':<10}", end="")
    for planner in planners:
        print(f"{planner:<12}", end="")
    print()
    print("-" * (25 + 12 * len(planners)))

    # Data rows
    for obstacle_type in obstacle_types:
        for density in densities:
            print(f"{obstacle_type:<15} {density*100:>3.0f}%      ", end="")
            for planner_name in planners:
                key = (obstacle_type, density, planner_name)
                status = "✓" if results.get(key, False) else "✗"
                print(f"{status:<12}", end="")
            print()

    # List of failures
    print("\n" + "="*80)
    print("FAILED CASES (Planner - Obstacle Type - Density)")
    print("="*80)

    failures = [(k[2], k[0], k[1]) for k, v in results.items() if not v]
    if failures:
        failures.sort()  # Sort by planner name
        for planner_name, obstacle_type, density in failures:
            print(f"  ✗ {planner_name:<15} - {obstacle_type:<12} - {density*100:>3.0f}%")
    else:
        print("  No failures! All tests passed! 🎉")

    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)

    total_success = sum(1 for v in results.values() if v)
    total_tests = len(results)

    print(f"\nTotal tests: {total_tests}")
    print(f"Successful: {total_success} ({total_success/total_tests*100:.1f}%)")
    print(f"Failed: {total_tests - total_success} ({(total_tests - total_success)/total_tests*100:.1f}%)")

    # Per-planner statistics
    print(f"\nPer-Planner Success Rate:")
    for planner_name in planners:
        planner_success = sum(1 for k, v in results.items() if k[2] == planner_name and v)
        planner_total = sum(1 for k in results.keys() if k[2] == planner_name)
        print(f"  {planner_name:<15}: {planner_success}/{planner_total} ({planner_success/planner_total*100:.1f}%)")

    # Per-density statistics
    print(f"\nPer-Density Success Rate:")
    for density in densities:
        density_success = sum(1 for k, v in results.items() if k[1] == density and v)
        density_total = sum(1 for k in results.keys() if k[1] == density)
        print(f"  {density*100:>3.0f}%: {density_success}/{density_total} ({density_success/density_total*100:.1f}%)")

    # Per-obstacle-type statistics
    print(f"\nPer-Obstacle-Type Success Rate:")
    for obstacle_type in obstacle_types:
        obs_success = sum(1 for k, v in results.items() if k[0] == obstacle_type and v)
        obs_total = sum(1 for k in results.keys() if k[0] == obstacle_type)
        print(f"  {obstacle_type:<12}: {obs_success}/{obs_total} ({obs_success/obs_total*100:.1f}%)")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

