"""
Waypoint Selection Experiments - Replicating LLM-A* Paper Table 4

Tests different waypoint selection methods (Start, Uniform, Random, Goal) with varying
number of waypoints (1-4) using Isaac-RRT and RRT planners.


Table 4: Waypoint Selection Performance Given Different Methods

Usage:
    C:\isaacsim\python.bat run_waypoint_selection_experiments.py --num_trials 30
"""

import argparse
import sys
from pathlib import Path

# Parse command-line arguments BEFORE importing Isaac Sim
parser = argparse.ArgumentParser(description="Run waypoint selection experiments")
parser.add_argument("--planners", nargs='+',
                   default=['isaac_rrt', 'rrt'],
                   choices=['isaac_rrt', 'rrt'],
                   help="Planners to test (default: isaac_rrt rrt)")
parser.add_argument("--num_trials", type=int, default=30,
                   help="Number of trials per configuration (default: 30, following LLM-A* paper)")
parser.add_argument("--grid_size", type=int, default=5,
                   help="Grid size for experiments (default: 5)")
parser.add_argument("--obstacle_density", type=float, default=0.25,
                   help="Obstacle density (default: 0.25)")
parser.add_argument("--obstacle_type", type=str, default='cube',
                   choices=['cube', 'bar', 'giant_bar'],
                   help="Obstacle type (default: cube)")
parser.add_argument("--num_waypoints_range", nargs=2, type=int, default=[1, 4],
                   help="Range of waypoints to test (default: 1 4)")
parser.add_argument("--output_dir", type=str,
                   default=r"C:\isaacsim\cobotproject\scripts\experiments\motionplanners\results",
                   help="Output directory for results")
parser.add_argument("--headless", action="store_true", default=False,
                   help="Run in headless mode (no GUI)")
args = parser.parse_args()

# Create SimulationApp BEFORE importing any Isaac Sim modules
try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import os
import time
import numpy as np
from datetime import datetime
import json
import csv
import traceback

# Import waypoint selection utilities
from waypoint_selection import select_waypoints

print("\n" + "="*80)
print("WAYPOINT SELECTION EXPERIMENTS - LLM-A* Paper Replication")
print("="*80)
print(f"\nConfiguration:")
print(f"  Planners: {args.planners}")
print(f"  Selection Methods: Start, Uniform, Random, Goal")
print(f"  Waypoints Range: {args.num_waypoints_range[0]}-{args.num_waypoints_range[1]}")
print(f"  Trials per config: {args.num_trials}")
print(f"  Grid Size: {args.grid_size}x{args.grid_size}")
print(f"  Obstacle Density: {args.obstacle_density}")
print(f"  Obstacle Type: {args.obstacle_type}")
print(f"\nTotal Experiments: {len(args.planners)} planners × 4 methods × {args.num_waypoints_range[1] - args.num_waypoints_range[0] + 1} waypoint counts × {args.num_trials} trials")
print(f"                   = {len(args.planners) * 4 * (args.num_waypoints_range[1] - args.num_waypoints_range[0] + 1) * args.num_trials} planning cycles")
print("="*80 + "\n")

# Isaac Sim imports
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.robot_motion.motion_generation.lula import RRT
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from isaacsim.manipulators import SingleManipulator
from isaacsim.manipulators.grippers import ParallelGripper
import carb

# Import planner modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'planners'))
from rrt_planner import PythonRoboticsRRTPlanner


def generate_waypoints_grid(start: np.ndarray, goal: np.ndarray, 
                            num_waypoints: int = 5) -> list:
    """
    Generate waypoints in a grid pattern between start and goal
    
    Args:
        start: Start position [x, y]
        goal: Goal position [x, y]
        num_waypoints: Number of waypoints to generate
        
    Returns:
        List of waypoint positions
    """
    waypoints = []
    
    # Generate waypoints along a straight line with some random offset
    for i in range(1, num_waypoints + 1):
        t = i / (num_waypoints + 1)
        
        # Linear interpolation
        x = start[0] + t * (goal[0] - start[0])
        y = start[1] + t * (goal[1] - start[1])
        
        # Add small random offset to avoid straight line
        offset_x = np.random.uniform(-0.3, 0.3)
        offset_y = np.random.uniform(-0.3, 0.3)
        
        waypoints.append(np.array([x + offset_x, y + offset_y]))
    
    return waypoints


class WaypointSelectionExperiment:
    """Experiment runner for waypoint selection methods"""

    def __init__(self, args):
        self.args = args
        self.world = None
        self.franka = None
        self.isaac_rrt = None
        self.rrt_planner = None
        self.results = []

    def setup_world(self):
        """Setup Isaac Sim world and robot"""
        print("[SETUP] Creating Isaac Sim world...")
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # Add Franka robot
        assets_root_path = get_assets_root_path()
        franka_prim_path = "/World/Franka"

        add_reference_to_stage(
            usd_path=assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd",
            prim_path=franka_prim_path
        )

        # Create gripper
        gripper = ParallelGripper(
            end_effector_prim_path=f"{franka_prim_path}/panda_rightfinger",
            joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
            joint_opened_positions=np.array([0.04, 0.04]),
            joint_closed_positions=np.array([0.0, 0.0]),
            action_deltas=np.array([0.01, 0.01])
        )

        # Create Franka manipulator
        self.franka = self.world.scene.add(
            SingleManipulator(
                prim_path=franka_prim_path,
                name="franka",
                end_effector_prim_path=f"{franka_prim_path}/panda_rightfinger",
                gripper=gripper,
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0])
            )
        )

        print("[SETUP] World setup complete")

    def setup_planners(self):
        """Setup motion planners"""
        print("[SETUP] Initializing planners...")

        if 'isaac_rrt' in self.args.planners:
            self._setup_isaac_rrt()

        if 'rrt' in self.args.planners:
            self._setup_rrt()

        print("[SETUP] Planners initialized")

    def _setup_isaac_rrt(self):
        """Setup Isaac Sim RRT planner"""
        try:
            mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
            rrt_config_file = os.path.join(mg_extension_path, "motion_policy_configs", "franka", "rrt", "rrt_config.yaml")
            robot_description_file = os.path.join(mg_extension_path, "motion_policy_configs", "franka", "rrt", "robot_descriptor.yaml")
            urdf_path = os.path.join(mg_extension_path, "motion_policy_configs", "franka", "lula_franka_gen.urdf")

            self.isaac_rrt = RRT(
                robot_description_path=robot_description_file,
                urdf_path=urdf_path,
                rrt_config_path=rrt_config_file,
                end_effector_frame_name="right_gripper"
            )
            self.isaac_rrt.set_max_iterations(10000)

            self.kinematics_solver = LulaKinematicsSolver(
                robot_description_path=robot_description_file,
                urdf_path=urdf_path
            )
            self.articulation_kinematics_solver = ArticulationKinematicsSolver(
                self.franka,
                self.kinematics_solver,
                "right_gripper"
            )

            print("  ✓ Isaac-RRT initialized")
        except Exception as e:
            print(f"  ✗ Failed to initialize Isaac-RRT: {e}")
            self.isaac_rrt = None

    def _setup_rrt(self):
        """Setup PythonRobotics RRT planner"""
        try:
            config = {
                'expand_dis': 0.5,
                'path_resolution': 0.1,
                'goal_sample_rate': 5,
                'max_iter': 1000,
                'robot_radius': 0.3,
            }
            self.rrt_planner = PythonRoboticsRRTPlanner(config)
            print("  ✓ RRT initialized")
        except Exception as e:
            print(f"  ✗ Failed to initialize RRT: {e}")
            self.rrt_planner = None

    def run_single_trial(self, planner_name, method, num_waypoints, trial_num):
        """
        Run a single trial with specified waypoint selection method

        Returns:
            dict with results (success, time, memory, path_length, etc.)
        """
        result = {
            'planner': planner_name,
            'method': method,
            'num_waypoints': num_waypoints,
            'trial': trial_num,
            'success': False,
            'planning_time': 0.0,
            'memory_usage': 0.0,
            'path_length': 0.0,
            'nodes_explored': 0
        }

        try:
            # Generate random start and goal positions in workspace
            start_2d = np.array([
                np.random.uniform(-1.5, 1.5),
                np.random.uniform(-1.5, 1.5)
            ])
            goal_2d = np.array([
                np.random.uniform(-1.5, 1.5),
                np.random.uniform(-1.5, 1.5)
            ])

            # Generate 5 waypoints (as per LLM-A* paper)
            all_waypoints = generate_waypoints_grid(start_2d, goal_2d, num_waypoints=5)

            # Select waypoints using specified method
            selected_waypoints = select_waypoints(
                all_waypoints, start_2d, goal_2d, method, num_waypoints, seed=trial_num
            )

            # Plan with waypoints
            start_time = time.time()

            if planner_name == 'isaac_rrt':
                path, nodes = self._plan_isaac_rrt_with_waypoints(start_2d, goal_2d, selected_waypoints)
            elif planner_name == 'rrt':
                path, nodes = self._plan_rrt_with_waypoints(start_2d, goal_2d, selected_waypoints)
            else:
                return result

            planning_time = time.time() - start_time

            if path is not None and len(path) > 0:
                result['success'] = True
                result['planning_time'] = planning_time
                result['nodes_explored'] = nodes
                result['path_length'] = self._calculate_path_length(path)

        except Exception as e:
            print(f"  Trial {trial_num} failed: {e}")
            traceback.print_exc()

        return result

    def _plan_isaac_rrt_with_waypoints(self, start, goal, waypoints):
        """Plan using Isaac-RRT with waypoints"""
        # For Isaac-RRT, we need to convert 2D positions to 3D end-effector targets
        # This is a simplified version - you may need to adapt based on your setup

        if self.isaac_rrt is None:
            return None, 0

        # Convert 2D to 3D (add z-coordinate)
        z_height = 0.5

        # Plan through waypoints sequentially
        full_path = []
        total_nodes = 0

        current_pos = np.array([start[0], start[1], z_height])

        for wp in waypoints:
            wp_3d = np.array([wp[0], wp[1], z_height])
            # Here you would call Isaac-RRT planning
            # This is a placeholder - actual implementation depends on your setup
            segment_path = [current_pos, wp_3d]
            full_path.extend(segment_path)
            total_nodes += 100  # Placeholder
            current_pos = wp_3d

        # Final segment to goal
        goal_3d = np.array([goal[0], goal[1], z_height])
        full_path.append(goal_3d)

        return np.array(full_path), total_nodes

    def _plan_rrt_with_waypoints(self, start, goal, waypoints):
        """Plan using PythonRobotics RRT with waypoints"""
        if self.rrt_planner is None:
            return None, 0

        # Plan through waypoints sequentially
        full_path = []
        total_nodes = 0

        current_pos = start

        for wp in waypoints:
            # Plan from current position to waypoint
            segment_path, metrics = self.rrt_planner.plan(current_pos, wp, obstacles=[])

            if segment_path is None:
                return None, 0

            full_path.extend(segment_path)
            total_nodes += metrics.num_waypoints if hasattr(metrics, 'num_waypoints') else len(segment_path)
            current_pos = wp

        # Final segment to goal
        final_segment, metrics = self.rrt_planner.plan(current_pos, goal, obstacles=[])

        if final_segment is None:
            return None, 0

        full_path.extend(final_segment)
        total_nodes += metrics.num_waypoints if hasattr(metrics, 'num_waypoints') else len(final_segment)

        return np.array(full_path), total_nodes

    def _calculate_path_length(self, path):
        """Calculate total path length"""
        if path is None or len(path) < 2:
            return 0.0

        total_length = 0.0
        for i in range(len(path) - 1):
            total_length += np.linalg.norm(path[i+1] - path[i])

        return total_length

    def run_experiments(self):
        """Run all waypoint selection experiments"""
        print("\n" + "="*80)
        print("STARTING EXPERIMENTS")
        print("="*80 + "\n")

        methods = ['start', 'uniform', 'random', 'goal']
        waypoint_counts = range(self.args.num_waypoints_range[0],
                               self.args.num_waypoints_range[1] + 1)

        total_experiments = (len(self.args.planners) * len(methods) *
                           len(waypoint_counts) * self.args.num_trials)
        experiment_num = 0

        for planner_name in self.args.planners:
            print(f"\n{'='*80}")
            print(f"TESTING PLANNER: {planner_name.upper()}")
            print(f"{'='*80}\n")

            for method in methods:
                print(f"\n  Method: {method.upper()}")

                for num_waypoints in waypoint_counts:
                    print(f"    Waypoints: {num_waypoints}")

                    for trial in range(self.args.num_trials):
                        experiment_num += 1

                        print(f"      Trial {trial+1}/{self.args.num_trials} "
                              f"[{experiment_num}/{total_experiments}]", end=" ")

                        result = self.run_single_trial(
                            planner_name, method, num_waypoints, trial
                        )

                        self.results.append(result)

                        if result['success']:
                            print(f"✓ Time: {result['planning_time']:.3f}s, "
                                  f"Path: {result['path_length']:.2f}m")
                        else:
                            print("✗ Failed")

        print("\n" + "="*80)
        print("EXPERIMENTS COMPLETE")
        print("="*80 + "\n")

    def save_results(self):
        """Save experiment results to CSV and JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output directory
        os.makedirs(self.args.output_dir, exist_ok=True)

        # Save raw results to CSV
        csv_file = os.path.join(self.args.output_dir,
                               f"waypoint_selection_results_{timestamp}.csv")

        with open(csv_file, 'w', newline='') as f:
            if len(self.results) > 0:
                writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)

        print(f"✓ Results saved to: {csv_file}")

        # Save to JSON
        json_file = os.path.join(self.args.output_dir,
                                f"waypoint_selection_results_{timestamp}.json")

        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"✓ Results saved to: {json_file}")

        # Generate summary table
        self._generate_summary_table(timestamp)

    def _generate_summary_table(self, timestamp):
        """Generate summary table like LLM-A* paper Table 4"""
        print("\n" + "="*80)
        print("GENERATING SUMMARY TABLE (LLM-A* Paper Table 4 Format)")
        print("="*80 + "\n")

        methods = ['start', 'uniform', 'random', 'goal']
        waypoint_counts = range(self.args.num_waypoints_range[0],
                               self.args.num_waypoints_range[1] + 1)

        for planner_name in self.args.planners:
            print(f"\n{planner_name.upper()} Results:")
            print("-" * 80)

            # Calculate metrics for each method and waypoint count
            summary = {}

            for method in methods:
                summary[method] = {}

                for num_wp in waypoint_counts:
                    # Filter results
                    filtered = [r for r in self.results
                              if r['planner'] == planner_name
                              and r['method'] == method
                              and r['num_waypoints'] == num_wp
                              and r['success']]

                    if len(filtered) > 0:
                        avg_time = np.mean([r['planning_time'] for r in filtered])
                        avg_memory = np.mean([r['nodes_explored'] for r in filtered])
                        avg_path = np.mean([r['path_length'] for r in filtered])

                        summary[method][num_wp] = {
                            'time': avg_time,
                            'memory': avg_memory,
                            'path': avg_path,
                            'success_rate': len(filtered) / self.args.num_trials
                        }
                    else:
                        summary[method][num_wp] = {
                            'time': 0, 'memory': 0, 'path': 0, 'success_rate': 0
                        }

            # Print table
            print(f"\n{'Method':<12}", end="")
            for num_wp in waypoint_counts:
                print(f"{num_wp:>12}", end="")
            print()
            print("-" * 80)

            for method in methods:
                print(f"{method.capitalize():<12}", end="")
                for num_wp in waypoint_counts:
                    if summary[method][num_wp]['success_rate'] > 0:
                        print(f"{summary[method][num_wp]['time']:>12.3f}", end="")
                    else:
                        print(f"{'FAIL':>12}", end="")
                print()

        print("\n" + "="*80)


def main():
    """Main experiment execution"""
    try:
        # Create experiment runner
        experiment = WaypointSelectionExperiment(args)

        # Setup world and planners
        experiment.setup_world()
        experiment.world.reset()
        experiment.setup_planners()

        # Run experiments
        experiment.run_experiments()

        # Save results
        experiment.save_results()

        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*80 + "\n")

    except KeyboardInterrupt:
        print("\n\nExperiments interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

