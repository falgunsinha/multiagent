"""
Run MASAC tests with multiple seeds for MAPPO-style multi-run aggregation

This script runs the MASAC tests multiple times with different random seeds
to generate data for creating MAPPO-style comparison tables and learning curves
with confidence intervals.

IMPORTANT: This script automatically uses the correct Python interpreter:
- Isaac Sim tests: Uses C:\isaacsim\python.bat
- Native tests (A* & RRT Viz): Uses py -3.11 or python

Usage:
    # Run all 3 planners (Isaac Sim RRT, A*, RRT Viz) with 2 seeds
    python run_multi_seed_tests.py --env both --seeds 42 123 --episodes 500

    # Run Isaac Sim tests only with 3 seeds
    python run_multi_seed_tests.py --env isaacsim --seeds 42 123 456 --episodes 10

    # Run native Python tests only (A* & RRT Viz) with 5 seeds
    python run_multi_seed_tests.py --env native --seeds 42 123 456 789 1011 --episodes 10
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_isaacsim_tests(seeds: list, episodes: int, log_dir: str):
    """Run Isaac Sim tests with multiple seeds"""
    print(f"\n{'='*80}")
    print(f"Running Isaac Sim RRT tests with {len(seeds)} seeds")
    print(f"Seeds: {seeds}")
    print(f"Episodes per seed: {episodes}")
    print(f"{'='*80}\n")

    script_path = Path(__file__).parent / "test_masac_grid4_cubes9_isaacsim.py"

    # Use Isaac Sim Python for Isaac Sim tests
    isaac_python = r"C:\isaacsim\python.bat"

    for run_id, seed in enumerate(seeds, start=1):
        print(f"\n{'='*80}")
        print(f"RUN {run_id}/{len(seeds)} - Seed: {seed}")
        print(f"{'='*80}\n")

        cmd = [
            isaac_python,
            str(script_path),
            "--episodes", str(episodes),
            "--log_dir", log_dir,
            "--seed", str(seed),
            "--run_id", str(run_id)
        ]

        print(f"[CMD] {' '.join(cmd)}\n")

        try:
            result = subprocess.run(cmd, check=True)
            print(f"\n✅ Run {run_id} completed successfully (seed={seed})")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Run {run_id} failed (seed={seed}): {e}")
            continue


def run_native_tests(seeds: list, episodes: int, log_dir: str):
    """Run native Python tests with multiple seeds"""
    print(f"\n{'='*80}")
    print(f"Running Native Python tests (A* & RRT Viz) with {len(seeds)} seeds")
    print(f"Seeds: {seeds}")
    print(f"Episodes per seed: {episodes}")
    print(f"{'='*80}\n")

    script_path = Path(__file__).parent / "test_masac_grid4_cubes9_native.py"

    # Use Isaac Sim Python for native tests too (has all dependencies including sklearn)
    # This avoids dependency issues with regular Python
    native_python = r"C:\isaacsim\python.bat"
    python_args = []

    for run_id, seed in enumerate(seeds, start=1):
        print(f"\n{'='*80}")
        print(f"RUN {run_id}/{len(seeds)} - Seed: {seed}")
        print(f"{'='*80}\n")

        cmd = [native_python] + python_args + [
            str(script_path),
            "--episodes", str(episodes),
            "--log_dir", log_dir,
            "--seed", str(seed),
            "--run_id", str(run_id)
        ]

        print(f"[CMD] {' '.join(cmd)}\n")

        try:
            subprocess.run(cmd, check=True)
            print(f"\n✅ Run {run_id} completed successfully (seed={seed})")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Run {run_id} failed (seed={seed}): {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description='Run MASAC tests with multiple seeds')
    parser.add_argument('--env', type=str, choices=['isaacsim', 'native', 'both'], default='both',
                        help='Which environment to test (isaacsim, native, or both)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                        help='List of random seeds to use')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes per seed')
    parser.add_argument('--log_dir', type=str, default='cobotproject/scripts/Reinforcement Learning/MASAC/logs',
                        help='Base directory to save logs (or specific directory to resume)')
    parser.add_argument('--resume_dir', type=str, default=None,
                        help='Specific log directory to resume/continue (e.g., logs/multi_seed_20260120_033815)')
    args = parser.parse_args()

    # Use resume_dir if provided, otherwise create new timestamped directory
    if args.resume_dir:
        log_dir = args.resume_dir
        print(f"[RESUME MODE] Using existing log directory: {log_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"{args.log_dir}/multi_seed_{timestamp}"
        print(f"[NEW RUN] Creating new log directory: {log_dir}")
    
    print(f"\n{'='*80}")
    print(f"MASAC Multi-Seed Testing")
    print(f"{'='*80}")
    print(f"Environment: {args.env}")
    print(f"Seeds: {args.seeds} ({len(args.seeds)} runs)")
    print(f"Episodes per seed: {args.episodes}")
    print(f"Log directory: {log_dir}")
    print(f"{'='*80}\n")
    
    if args.env in ['isaacsim', 'both']:
        run_isaacsim_tests(args.seeds, args.episodes, log_dir)
    
    if args.env in ['native', 'both']:
        run_native_tests(args.seeds, args.episodes, log_dir)
    
    print(f"\n{'='*80}")
    print(f"Multi-Seed Testing Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {log_dir}")
    print(f"Total runs: {len(args.seeds)}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

