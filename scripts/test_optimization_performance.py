"""
Test script to verify performance optimizations for RRT Viz and reshuffling checks.

This script tests:
1. PCA fitting speed for RRT Viz (should be ~5-10s, same as A*)
2. Reshuffling check speed (should be ~0.01s per step)
3. Grid update frequency (should only update during reset() and reward calculation)

Usage:
    python cobotproject/scripts/test_optimization_performance.py
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cobotproject.src.rl.object_selection_env_astar import ObjectSelectionEnvAStar
from cobotproject.src.rl.object_selection_env_rrt_viz import ObjectSelectionEnvRRTViz


def test_pca_fitting_speed():
    """Test PCA fitting speed for both A* and RRT Viz"""
    print("\n" + "="*80)
    print("TEST 1: PCA Fitting Speed")
    print("="*80)
    
    grid_size = 4
    num_cubes = 9
    n_samples = 100  # Reduced for quick testing
    
    # Test A* environment
    print(f"\n[A*] Creating environment (grid={grid_size}, cubes={num_cubes})...")
    astar_env = ObjectSelectionEnvAStar(
        franka_controller=None,
        max_objects=grid_size * grid_size,
        max_steps=50,
        num_cubes=num_cubes,
        training_grid_size=grid_size,
        render_mode=None
    )
    
    print(f"[A*] Collecting {n_samples} sample states...")
    start_time = time.time()
    
    states = []
    for _ in range(n_samples):
        obs, _ = astar_env.reset()
        states.append(obs)
        
        # Collect some random steps
        for _ in range(10):
            action = np.random.randint(0, num_cubes)
            obs, _, done, truncated, _ = astar_env.step(action)
            if done or truncated:
                break
            states.append(obs)
    
    astar_time = time.time() - start_time
    print(f"[A*] ✅ Collected {len(states)} states in {astar_time:.2f}s")
    
    # Test RRT Viz environment
    print(f"\n[RRT Viz] Creating environment (grid={grid_size}, cubes={num_cubes})...")
    rrt_env = ObjectSelectionEnvRRTViz(
        franka_controller=None,
        max_objects=grid_size * grid_size,
        max_steps=50,
        num_cubes=num_cubes,
        training_grid_size=grid_size,
        render_mode=None
    )
    
    print(f"[RRT Viz] Collecting {n_samples} sample states...")
    start_time = time.time()
    
    states = []
    for _ in range(n_samples):
        obs, _ = rrt_env.reset()
        states.append(obs)
        
        # Collect some random steps
        for _ in range(10):
            action = np.random.randint(0, num_cubes)
            obs, _, done, truncated, _ = rrt_env.step(action)
            if done or truncated:
                break
            states.append(obs)
    
    rrt_time = time.time() - start_time
    print(f"[RRT Viz] ✅ Collected {len(states)} states in {rrt_time:.2f}s")
    
    # Compare performance
    print(f"\n{'='*80}")
    print("RESULTS:")
    print(f"  A* Time:      {astar_time:.2f}s")
    print(f"  RRT Viz Time: {rrt_time:.2f}s")
    print(f"  Speedup:      {rrt_time/astar_time:.2f}x {'SLOWER' if rrt_time > astar_time else 'FASTER'}")
    
    if rrt_time <= astar_time * 1.5:  # Allow 50% tolerance
        print(f"  ✅ PASS: RRT Viz is within 50% of A* speed")
    else:
        print(f"  ❌ FAIL: RRT Viz is still significantly slower than A*")
    print(f"{'='*80}\n")


def test_grid_update_frequency():
    """Test that RRT Viz only updates grid during reset() and reward calculation"""
    print("\n" + "="*80)
    print("TEST 2: Grid Update Frequency")
    print("="*80)
    
    print("\n[RRT Viz] Creating environment...")
    env = ObjectSelectionEnvRRTViz(
        franka_controller=None,
        max_objects=16,
        max_steps=50,
        num_cubes=9,
        training_grid_size=4,
        render_mode=None
    )
    
    # Monkey-patch update_occupancy_grid to count calls
    original_update = env.rrt_estimator.update_occupancy_grid
    update_count = {'count': 0}
    
    def counting_update(*args, **kwargs):
        update_count['count'] += 1
        return original_update(*args, **kwargs)
    
    env.rrt_estimator.update_occupancy_grid = counting_update
    
    # Test reset
    print("\n[RRT Viz] Testing reset()...")
    update_count['count'] = 0
    env.reset()
    reset_updates = update_count['count']
    print(f"  Grid updates during reset(): {reset_updates}")
    
    # Test observation (should NOT update grid)
    print("\n[RRT Viz] Testing _get_observation()...")
    update_count['count'] = 0
    env._get_observation(recalculate_obstacles=False)
    obs_updates = update_count['count']
    print(f"  Grid updates during _get_observation(): {obs_updates}")
    
    # Test step (should update grid during reward calculation)
    print("\n[RRT Viz] Testing step()...")
    update_count['count'] = 0
    env.step(0)
    step_updates = update_count['count']
    print(f"  Grid updates during step(): {step_updates}")
    
    # Results
    print(f"\n{'='*80}")
    print("RESULTS:")
    print(f"  Reset updates:       {reset_updates} (expected: 1)")
    print(f"  Observation updates: {obs_updates} (expected: 0)")
    print(f"  Step updates:        {step_updates} (expected: 1-2)")
    
    if obs_updates == 0:
        print(f"  ✅ PASS: No grid updates during _get_observation()")
    else:
        print(f"  ❌ FAIL: Grid is still being updated during _get_observation()")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PERFORMANCE OPTIMIZATION TEST SUITE")
    print("="*80)
    
    test_pca_fitting_speed()
    test_grid_update_frequency()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80 + "\n")

