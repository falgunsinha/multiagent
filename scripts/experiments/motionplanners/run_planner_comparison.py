"""
Motion Planner Comparison Experiment Runner

Runs experiments comparing different motion planning algorithms for pick-and-place
tasks in Isaac Sim, similar to ablation studies in research papers.

Usage:
    C:\isaacsim\python.bat run_planner_comparison.py --planners rrt astar prm rrtstar --num_trials 50 --grid_size 4
"""

import argparse
import sys
from pathlib import Path
import json
import csv
from datetime import datetime
import numpy as np

# Parse command-line arguments BEFORE importing Isaac Sim
parser = argparse.ArgumentParser(description="Compare motion planning algorithms in Isaac Sim")
parser.add_argument("--planners", nargs='+', default=['rrt', 'astar', 'prm', 'rrtstar'],
                   choices=['rrt', 'astar', 'prm', 'rrtstar', 'dijkstra'],
                   help="Planners to compare (default: rrt astar prm rrtstar)")
parser.add_argument("--num_trials", type=int, default=50,
                   help="Number of trials per planner (default: 50)")
parser.add_argument("--grid_size", type=int, default=4,
                   help="Grid size for cube placement (default: 4)")
parser.add_argument("--num_cubes", type=int, default=9,
                   help="Number of cubes to place (default: 9)")
parser.add_argument("--output_dir", type=str, 
                   default=r"C:\isaacsim\cobotproject\scripts\experiments\motionplanners\results",
                   help="Output directory for results")
parser.add_argument("--headless", action="store_true",
                   help="Run in headless mode (no GUI)")
parser.add_argument("--use_isaac_rrt", action="store_true",
                   help="Use Isaac Sim native RRT (requires Isaac Sim scene)")
args = parser.parse_args()

# Create SimulationApp if using Isaac Sim RRT
simulation_app = None
if args.use_isaac_rrt:
    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp
    
    simulation_app = SimulationApp({"headless": args.headless})

import os
import time

# Add project root to path
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import planner wrappers
from scripts.experiments.motionplanners.planners.rrt_planner import PythonRoboticsRRTPlanner, IsaacSimRRTPlanner
from scripts.experiments.motionplanners.planners.astar_planner import AStarPlanner
from scripts.experiments.motionplanners.planners.prm_planner import PRMPlanner
from scripts.experiments.motionplanners.planners.rrtstar_planner import RRTStarPlanner


class PlannerComparison:
    """
    Manages comparison experiments between different motion planners.
    """
    
    def __init__(self, planners_to_test, num_trials, grid_size, num_cubes, output_dir):
        """
        Initialize comparison experiment.
        
        Args:
            planners_to_test: List of planner names to test
            num_trials: Number of trials per planner
            grid_size: Grid size for cube placement
            num_cubes: Number of cubes
            output_dir: Output directory for results
        """
        self.planners_to_test = planners_to_test
        self.num_trials = num_trials
        self.grid_size = grid_size
        self.num_cubes = num_cubes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this experiment run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize planners
        self.planners = {}
        self._initialize_planners()
        
        # Results storage
        self.results = {planner_name: [] for planner_name in self.planners.keys()}
        
    def _initialize_planners(self):
        """Initialize all requested planners with default configurations"""
        print("\n" + "="*60)
        print("INITIALIZING PLANNERS")
        print("="*60)
        
        for planner_name in self.planners_to_test:
            if planner_name == 'rrt':
                config = {
                    'expand_dis': 0.1,
                    'path_resolution': 0.05,
                    'goal_sample_rate': 5,
                    'max_iter': 500,
                    'robot_radius': 0.05,
                    'rand_area': [-2, 2]
                }
                self.planners['rrt'] = PythonRoboticsRRTPlanner(config)
                print(f"✓ Initialized RRT planner")
                
            elif planner_name == 'astar':
                config = {
                    'resolution': 0.05,
                    'robot_radius': 0.05
                }
                self.planners['astar'] = AStarPlanner(config)
                print(f"✓ Initialized A* planner")
                
            elif planner_name == 'prm':
                config = {
                    'n_sample': 500,
                    'n_knn': 10,
                    'max_edge_len': 30.0,
                    'robot_radius': 0.05
                }
                self.planners['prm'] = PRMPlanner(config)
                print(f"✓ Initialized PRM planner")
                
            elif planner_name == 'rrtstar':
                config = {
                    'expand_dis': 0.1,
                    'path_resolution': 0.05,
                    'goal_sample_rate': 20,
                    'max_iter': 500,
                    'robot_radius': 0.05,
                    'connect_circle_dist': 0.5,
                    'rand_area': [-2, 2]
                }
                self.planners['rrtstar'] = RRTStarPlanner(config)
                print(f"✓ Initialized RRT* planner")
        
        print("="*60 + "\n")

    def generate_test_scenarios(self):
        """
        Generate test scenarios (start/goal pairs with obstacles).

        Returns:
            List of scenarios, each containing:
                - start_pos: Start position
                - goal_pos: Goal position
                - obstacles: List of obstacles
        """
        scenarios = []

        # Grid parameters
        cube_spacing = 0.13 if self.grid_size > 3 else 0.15
        grid_center = np.array([0.45, -0.10])
        grid_extent = (self.grid_size - 1) * cube_spacing
        start_x = grid_center[0] - (grid_extent / 2.0)
        start_y = grid_center[1] - (grid_extent / 2.0)

        # Generate random scenarios
        for trial in range(self.num_trials):
            # Random start and goal positions within grid
            start_row = np.random.randint(0, self.grid_size)
            start_col = np.random.randint(0, self.grid_size)

            goal_row = np.random.randint(0, self.grid_size)
            goal_col = np.random.randint(0, self.grid_size)

            # Ensure start != goal
            while (start_row == goal_row) and (start_col == goal_col):
                goal_row = np.random.randint(0, self.grid_size)
                goal_col = np.random.randint(0, self.grid_size)

            # Convert to world coordinates
            start_pos = np.array([
                start_x + (start_row * cube_spacing),
                start_y + (start_col * cube_spacing)
            ])

            goal_pos = np.array([
                start_x + (goal_row * cube_spacing),
                start_y + (goal_col * cube_spacing)
            ])

            # Generate random obstacles (avoid start and goal)
            num_obstacles = np.random.randint(2, min(6, self.grid_size))
            obstacles = []

            occupied_cells = {(start_row, start_col), (goal_row, goal_col)}

            for _ in range(num_obstacles):
                obs_row = np.random.randint(0, self.grid_size)
                obs_col = np.random.randint(0, self.grid_size)

                # Avoid occupied cells
                attempts = 0
                while (obs_row, obs_col) in occupied_cells and attempts < 20:
                    obs_row = np.random.randint(0, self.grid_size)
                    obs_col = np.random.randint(0, self.grid_size)
                    attempts += 1

                if (obs_row, obs_col) not in occupied_cells:
                    obs_pos = np.array([
                        start_x + (obs_row * cube_spacing),
                        start_y + (obs_col * cube_spacing)
                    ])
                    obstacles.append([obs_pos[0], obs_pos[1], 0.055])  # [x, y, radius]
                    occupied_cells.add((obs_row, obs_col))

            scenarios.append({
                'start_pos': start_pos,
                'goal_pos': goal_pos,
                'obstacles': obstacles,
                'trial_id': trial
            })

        return scenarios

    def run_experiments(self):
        """Run comparison experiments for all planners"""
        print("\n" + "="*60)
        print("RUNNING PLANNER COMPARISON EXPERIMENTS")
        print("="*60)
        print(f"Planners: {list(self.planners.keys())}")
        print(f"Trials per planner: {self.num_trials}")
        print(f"Grid size: {self.grid_size}x{self.grid_size}")
        print(f"Number of cubes: {self.num_cubes}")
        print("="*60 + "\n")

        # Generate test scenarios
        print("Generating test scenarios...")
        scenarios = self.generate_test_scenarios()
        print(f"Generated {len(scenarios)} test scenarios\n")

        # Run each planner on all scenarios
        for planner_name, planner in self.planners.items():
            print(f"\n{'='*60}")
            print(f"TESTING: {planner_name.upper()}")
            print(f"{'='*60}")

            for i, scenario in enumerate(scenarios):
                # Run planner
                path, metrics = planner.plan(
                    scenario['start_pos'],
                    scenario['goal_pos'],
                    scenario['obstacles']
                )

                # Store results
                result = {
                    'trial_id': scenario['trial_id'],
                    'planner': planner_name,
                    'start_pos': scenario['start_pos'].tolist(),
                    'goal_pos': scenario['goal_pos'].tolist(),
                    'num_obstacles': len(scenario['obstacles']),
                    **metrics.to_dict()
                }
                self.results[planner_name].append(result)

                # Print progress
                if (i + 1) % 10 == 0:
                    success_rate = np.mean([r['success'] for r in self.results[planner_name]])
                    avg_time = np.mean([r['planning_time'] for r in self.results[planner_name] if r['success']])
                    print(f"  Progress: {i+1}/{len(scenarios)} | "
                          f"Success: {success_rate:.1%} | "
                          f"Avg Time: {avg_time:.4f}s")

            # Print summary for this planner
            self._print_planner_summary(planner_name)

        # Save results
        self._save_results()

        # Print final comparison
        self._print_final_comparison()

    def _print_planner_summary(self, planner_name):
        """Print summary statistics for a planner"""
        results = self.results[planner_name]

        if not results:
            print(f"\n  No results for {planner_name}")
            return

        success_results = [r for r in results if r['success']]
        success_rate = len(success_results) / len(results)

        print(f"\n  {planner_name.upper()} SUMMARY:")
        print(f"  {'─'*50}")
        print(f"  Success Rate: {success_rate:.1%} ({len(success_results)}/{len(results)})")

        if success_results:
            avg_time = np.mean([r['planning_time'] for r in success_results])
            avg_length = np.mean([r['path_length'] for r in success_results])
            avg_waypoints = np.mean([r['num_waypoints'] for r in success_results])
            avg_smoothness = np.mean([r['smoothness'] for r in success_results])

            print(f"  Avg Planning Time: {avg_time:.4f}s")
            print(f"  Avg Path Length: {avg_length:.4f}m")
            print(f"  Avg Waypoints: {avg_waypoints:.1f}")
            print(f"  Avg Smoothness: {avg_smoothness:.4f}")
        print(f"  {'─'*50}\n")

    def _save_results(self):
        """Save results to CSV and JSON files"""
        # Create results directory
        results_dir = self.output_dir / f"comparison_{self.timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results to CSV
        csv_file = results_dir / "detailed_results.csv"

        # Collect all results
        all_results = []
        for planner_name, results in self.results.items():
            all_results.extend(results)

        if all_results:
            # Write CSV
            with open(csv_file, 'w', newline='') as f:
                fieldnames = all_results[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)

            print(f"\n✓ Saved detailed results to: {csv_file}")

        # Save summary statistics to JSON
        summary = {}
        for planner_name, results in self.results.items():
            success_results = [r for r in results if r['success']]
            success_rate = len(success_results) / len(results) if results else 0.0

            summary[planner_name] = {
                'total_trials': len(results),
                'successes': len(success_results),
                'success_rate': success_rate,
                'avg_planning_time': np.mean([r['planning_time'] for r in success_results]) if success_results else 0.0,
                'std_planning_time': np.std([r['planning_time'] for r in success_results]) if success_results else 0.0,
                'avg_path_length': np.mean([r['path_length'] for r in success_results]) if success_results else 0.0,
                'std_path_length': np.std([r['path_length'] for r in success_results]) if success_results else 0.0,
                'avg_waypoints': np.mean([r['num_waypoints'] for r in success_results]) if success_results else 0.0,
                'avg_smoothness': np.mean([r['smoothness'] for r in success_results]) if success_results else 0.0,
                'avg_energy': np.mean([r['energy'] for r in success_results]) if success_results else 0.0,
            }

        json_file = results_dir / "summary_statistics.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Saved summary statistics to: {json_file}")

        # Save experiment configuration
        config = {
            'timestamp': self.timestamp,
            'planners': self.planners_to_test,
            'num_trials': self.num_trials,
            'grid_size': self.grid_size,
            'num_cubes': self.num_cubes,
            'planner_configs': {name: planner.get_config() for name, planner in self.planners.items()}
        }

        config_file = results_dir / "experiment_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✓ Saved experiment config to: {config_file}\n")

    def _print_final_comparison(self):
        """Print final comparison table"""
        print("\n" + "="*80)
        print("FINAL COMPARISON RESULTS")
        print("="*80)

        # Create comparison table
        print(f"\n{'Planner':<15} {'Success':<12} {'Avg Time (s)':<15} {'Avg Length (m)':<18} {'Avg Smoothness':<15}")
        print("─" * 80)

        for planner_name in self.planners.keys():
            results = self.results[planner_name]
            success_results = [r for r in results if r['success']]
            success_rate = len(success_results) / len(results) if results else 0.0

            if success_results:
                avg_time = np.mean([r['planning_time'] for r in success_results])
                avg_length = np.mean([r['path_length'] for r in success_results])
                avg_smoothness = np.mean([r['smoothness'] for r in success_results])

                print(f"{planner_name:<15} {success_rate:>6.1%}      {avg_time:>10.4f}      {avg_length:>12.4f}        {avg_smoothness:>10.4f}")
            else:
                print(f"{planner_name:<15} {success_rate:>6.1%}      {'N/A':>10}      {'N/A':>12}        {'N/A':>10}")

        print("="*80)

        # Highlight best performers
        print("\nBEST PERFORMERS:")
        print("─" * 40)

        # Best success rate
        best_success = max(self.planners.keys(),
                          key=lambda p: len([r for r in self.results[p] if r['success']]) / len(self.results[p]))
        success_rate = len([r for r in self.results[best_success] if r['success']]) / len(self.results[best_success])
        print(f"✓ Highest Success Rate: {best_success.upper()} ({success_rate:.1%})")

        # Best planning time
        avg_times = {}
        for planner_name in self.planners.keys():
            success_results = [r for r in self.results[planner_name] if r['success']]
            if success_results:
                avg_times[planner_name] = np.mean([r['planning_time'] for r in success_results])

        if avg_times:
            best_time = min(avg_times.keys(), key=lambda p: avg_times[p])
            print(f"✓ Fastest Planning: {best_time.upper()} ({avg_times[best_time]:.4f}s)")

        # Best path length
        avg_lengths = {}
        for planner_name in self.planners.keys():
            success_results = [r for r in self.results[planner_name] if r['success']]
            if success_results:
                avg_lengths[planner_name] = np.mean([r['path_length'] for r in success_results])

        if avg_lengths:
            best_length = min(avg_lengths.keys(), key=lambda p: avg_lengths[p])
            print(f"✓ Shortest Paths: {best_length.upper()} ({avg_lengths[best_length]:.4f}m)")

        # Best smoothness
        avg_smoothness = {}
        for planner_name in self.planners.keys():
            success_results = [r for r in self.results[planner_name] if r['success']]
            if success_results:
                avg_smoothness[planner_name] = np.mean([r['smoothness'] for r in success_results])

        if avg_smoothness:
            best_smooth = min(avg_smoothness.keys(), key=lambda p: avg_smoothness[p])
            print(f"✓ Smoothest Paths: {best_smooth.upper()} ({avg_smoothness[best_smooth]:.4f})")

        print("="*80 + "\n")


def main():
    """Main function"""
    print("\n" + "="*80)
    print("MOTION PLANNER COMPARISON EXPERIMENT")
    print("="*80)
    print(f"Planners to test: {', '.join(args.planners)}")
    print(f"Number of trials: {args.num_trials}")
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"Number of cubes: {args.num_cubes}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)

    # Create comparison experiment
    comparison = PlannerComparison(
        planners_to_test=args.planners,
        num_trials=args.num_trials,
        grid_size=args.grid_size,
        num_cubes=args.num_cubes,
        output_dir=args.output_dir
    )

    # Run experiments
    comparison.run_experiments()

    print("\n✓ Experiment complete!")

    # Close simulation app if used
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()

