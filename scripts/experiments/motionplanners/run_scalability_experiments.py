"""
Scalability Experiments for Motion Planning Algorithms

Runs experiments varying map size, obstacle density, and complexity
similar to research paper ablation studies.

This script tests how planners scale with:
- Map size (grid size: 3x3, 4x4, 5x5, 6x6, 7x7)
- Obstacle density (low, medium, high)
- Path complexity (simple, moderate, complex)

Usage:
    C:\isaacsim\python.bat run_scalability_experiments.py --planners rrt astar rrtstar --num_trials 30
"""

import argparse
import sys
from pathlib import Path
import json
import csv
from datetime import datetime
import numpy as np

# Parse command-line arguments BEFORE importing Isaac Sim
parser = argparse.ArgumentParser(description="Run scalability experiments for motion planners")
parser.add_argument("--planners", nargs='+', 
                   default=['rrt', 'astar', 'prm', 'rrtstar', 'rrtstar_rs', 'lqr_rrtstar'],
                   choices=['rrt', 'astar', 'prm', 'rrtstar', 'rrtstar_rs', 'lqr_rrtstar', 'lqr', 'isaac_rrt'],
                   help="Planners to compare")
parser.add_argument("--num_trials", type=int, default=30,
                   help="Number of trials per configuration (default: 30)")
parser.add_argument("--map_sizes", nargs='+', type=int, default=[3, 4, 5, 6, 7],
                   help="Map sizes to test (default: 3 4 5 6 7)")
parser.add_argument("--obstacle_densities", nargs='+', default=['low', 'medium', 'high'],
                   choices=['low', 'medium', 'high'],
                   help="Obstacle densities to test")
parser.add_argument("--output_dir", type=str, 
                   default=r"C:\isaacsim\cobotproject\scripts\experiments\motionplanners\results",
                   help="Output directory for results")
parser.add_argument("--headless", action="store_true",
                   help="Run in headless mode (no GUI)")
parser.add_argument("--use_isaac_sim", action="store_true",
                   help="Use Isaac Sim environment (for Isaac RRT)")
args = parser.parse_args()

# Create SimulationApp if using Isaac Sim
simulation_app = None
if args.use_isaac_sim or 'isaac_rrt' in args.planners:
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
from scripts.experiments.motionplanners.planners.rrtstar_reedsshepp_planner import RRTStarReedsSheppPlanner
from scripts.experiments.motionplanners.planners.lqr_rrtstar_planner import LQRRRTStarPlanner
from scripts.experiments.motionplanners.planners.lqr_planner import LQRPlanner


class ScalabilityExperiments:
    """
    Manages scalability experiments for motion planners.
    
    Tests how planners perform as:
    - Map size increases
    - Obstacle density increases
    - Path complexity increases
    """
    
    def __init__(self, planners_to_test, num_trials, map_sizes, obstacle_densities, output_dir):
        """
        Initialize scalability experiments.
        
        Args:
            planners_to_test: List of planner names to test
            num_trials: Number of trials per configuration
            map_sizes: List of map sizes (grid dimensions)
            obstacle_densities: List of obstacle densities
            output_dir: Output directory for results
        """
        self.planners_to_test = planners_to_test
        self.num_trials = num_trials
        self.map_sizes = map_sizes
        self.obstacle_densities = obstacle_densities
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this experiment run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize planners
        self.planners = {}
        self._initialize_planners()
        
        # Results storage: {planner_name: {map_size: {density: [results]}}}
        self.results = {
            planner_name: {
                map_size: {
                    density: []
                    for density in obstacle_densities
                }
                for map_size in map_sizes
            }
            for planner_name in self.planners.keys()
        }
        
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
                    'max_iter': 1000,
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
                    'max_iter': 1000,
                    'robot_radius': 0.05,
                    'connect_circle_dist': 0.5,
                    'rand_area': [-2, 2]
                }
                self.planners['rrtstar'] = RRTStarPlanner(config)
                print(f"✓ Initialized RRT* planner")

            elif planner_name == 'rrtstar_rs':
                config = {
                    'expand_dis': 1.0,
                    'path_resolution': 0.5,
                    'goal_sample_rate': 10,
                    'max_iter': 1000,
                    'robot_radius': 0.05,
                    'connect_circle_dist': 5.0,
                    'curvature': 1.0,
                    'rand_area': [-2, 2]
                }
                self.planners['rrtstar_rs'] = RRTStarReedsSheppPlanner(config)
                print(f"✓ Initialized RRT*-ReedsShepp planner")

            elif planner_name == 'lqr_rrtstar':
                config = {
                    'max_iter': 1000,
                    'goal_sample_rate': 10,
                    'robot_radius': 0.05,
                    'connect_circle_dist': 0.5,
                    'rand_area': [-2, 2]
                }
                self.planners['lqr_rrtstar'] = LQRRRTStarPlanner(config)
                print(f"✓ Initialized LQR-RRT* planner")

            elif planner_name == 'lqr':
                config = {
                    'dt': 0.1,
                    'max_time': 10.0,
                    'robot_radius': 0.05
                }
                self.planners['lqr'] = LQRPlanner(config)
                print(f"✓ Initialized LQR planner")

        print("="*60 + "\n")

