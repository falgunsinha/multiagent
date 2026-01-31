"""
Quick test script to verify pick-and-place experiments work.

This runs a minimal test with 3 trials on a 3x3 grid with GUI (headless=False).

Usage:
    C:\isaacsim\python.bat test_quick.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Inject minimal test parameters (NO --headless flag = GUI will be shown)
sys.argv.extend([
    '--planners', 'isaac_rrt', 'astar',
    '--num_cubes', '1',
    '--grid_sizes', '3',
    '--num_trials', '3',
    '--obstacle_densities', '0.25',
    '--obstacle_types', 'cube',
    '--output_dir', r'C:\isaacsim\cobotproject\scripts\experiments\motionplanners\results\test'
])

# Now import and run
from scripts.experiments.motionplanners.run_pick_place_experiments import main

if __name__ == "__main__":
    print("\n" + "="*60)
    print("QUICK TEST: Pick-and-Place Experiments (GUI MODE)")
    print("="*60)
    print("Config: 2 planners, 1 cube, 3x3 grid, 3 trials, 25% density")
    print("="*60 + "\n")

    main()

