"""
Pure Heuristic Baseline (No RL)
Selects objects using only heuristic scoring without any RL training.
This serves as a baseline to compare against RL-trained models.

Usage:
    C:\\isaacsim\\python.bat franka_pure_heuristic_baseline.py --num_cubes 4 --grid_size 3
"""

import argparse
import numpy as np
import time
from datetime import datetime

# Parse arguments BEFORE importing Isaac Sim
parser = argparse.ArgumentParser(description="Pure Heuristic Baseline (No RL)")
parser.add_argument("--num_cubes", type=int, default=4, help="Number of cubes (default: 4)")
parser.add_argument("--grid_size", type=int, default=3, help="Grid size (default: 3)")
parser.add_argument("--headless", action="store_true", help="Run headless")
args = parser.parse_args()

# Import Isaac Sim
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import XFormPrim
import omni.isaac.core.utils.prims as prim_utils
from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema

# Import Franka controller components
from src.controllers.franka_rrt_controller import FrankaRRTController
from src.grippers.suction_gripper import SuctionGripper
from isaacsim.robot_motion.motion_generation.lula import RRT, PathPlannerVisualizer


class PureHeuristicSelector:
    """Pure heuristic object selection (no RL)"""
    
    def __init__(self, franka_controller):
        self.franka = franka_controller
        
    def calculate_heuristic_score(self, obj_position, container_position, ee_position, obstacle_positions):
        """
        Calculate heuristic score for an object.
        Higher score = better to pick.
        
        Parameters considered:
        1. Distance to robot EE (closer is better)
        2. Distance to container (closer is better for place)
        3. Obstacle proximity (fewer/farther obstacles is better)
        """
        # 1. Distance to EE (inverse - closer is better)
        dist_to_ee = np.linalg.norm(obj_position - ee_position)
        ee_score = 10.0 / (1.0 + dist_to_ee)  # Range: ~10 to ~1
        
        # 2. Distance to container (inverse - closer is better)
        dist_to_container = np.linalg.norm(obj_position - container_position)
        container_score = 5.0 / (1.0 + dist_to_container)  # Range: ~5 to ~0.5
        
        # 3. Obstacle proximity penalty
        obstacle_penalty = 0.0
        if len(obstacle_positions) > 0:
            obstacle_count = 0
            min_obstacle_dist = float('inf')
            
            for obs_pos in obstacle_positions:
                dist = np.linalg.norm(obj_position - obs_pos)
                if dist < 0.3:  # Within 30cm
                    obstacle_count += 1
                    min_obstacle_dist = min(min_obstacle_dist, dist)
            
            if obstacle_count > 0:
                count_penalty = min(obstacle_count / 3.0, 1.0) * 5.0
                dist_penalty = (1.0 - min(min_obstacle_dist / 0.3, 1.0)) * 3.0
                obstacle_penalty = count_penalty + dist_penalty
        
        # Total score
        total_score = ee_score + container_score - obstacle_penalty
        
        return total_score, {
            "ee_score": ee_score,
            "container_score": container_score,
            "obstacle_penalty": obstacle_penalty,
            "total": total_score
        }
    
    def select_next_object(self, cubes, picked_indices, container_position, obstacle_positions):
        """
        Select next object to pick using pure heuristics.
        
        Returns:
            (best_index, best_score, score_breakdown)
        """
        ee_position = self.franka.franka.end_effector.get_world_pose()[0]
        
        best_index = -1
        best_score = -float('inf')
        best_breakdown = None
        
        for i, (cube, cube_name) in enumerate(cubes):
            if i in picked_indices:
                continue  # Already picked
            
            obj_position = cube.get_world_pose()[0]
            score, breakdown = self.calculate_heuristic_score(
                obj_position, container_position, ee_position, obstacle_positions
            )
            
            if score > best_score:
                best_score = score
                best_index = i
                best_breakdown = breakdown
        
        return best_index, best_score, best_breakdown


def main():
    print("=" * 80)
    print("PURE HEURISTIC BASELINE (NO RL)")
    print("=" * 80)
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"Number of cubes: {args.num_cubes}")
    print(f"Headless: {args.headless}")
    print("=" * 80)
    print()
    
    # Create world
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    print("[SETUP] Creating Franka robot with gripper...")

    # Load Franka with variant selection
    from omni.isaac.franka import Franka
    franka_prim_path = "/World/Franka"

    # Add Franka USD
    prim_utils.create_prim(
        franka_prim_path,
        usd_path="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Franka/franka.usd"
    )

    # Set variant to suction gripper
    franka_prim = prim_utils.get_prim_at_path(franka_prim_path)
    if franka_prim:
        variant_api = franka_prim.GetVariantSets()
        if variant_api.HasVariantSet("gripper"):
            variant_set = variant_api.GetVariantSet("gripper")
            variant_set.SetVariantSelection("suction")

    # Create Franka instance
    franka = Franka(prim_path=franka_prim_path, name="franka")
    world.scene.add(franka)

    # Create suction gripper
    gripper = SuctionGripper(end_effector_prim_path=f"{franka_prim_path}/panda_hand/geometry")

    # Create container
    print("[SETUP] Creating container...")
    container = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Container",
            name="container",
            position=np.array([0.6, 0.0, 0.05]),
            scale=np.array([0.15, 0.15, 0.10]),
            color=np.array([0.2, 0.6, 0.2])
        )
    )

    # Spawn cubes in grid
    print(f"[SETUP] Spawning {args.num_cubes} cubes in {args.grid_size}x{args.grid_size} grid...")
    cubes = []
    # CRITICAL: cube_spacing MUST match cell_size used by path estimators!
    # Updated from 0.20/0.22 to 0.26/0.28 to ensure gripper (15cm) can fit between objects
    # With 26cm spacing: cube-to-cube gap = 26cm - 5.15cm = 20.85cm (gripper 15cm fits with 5.85cm clearance)
    cube_spacing = 0.26 if args.grid_size > 3 else 0.28
    grid_center_x = 0.45
    grid_center_y = -0.10
    grid_extent_x = (args.grid_size - 1) * cube_spacing
    grid_extent_y = (args.grid_size - 1) * cube_spacing
    start_x = grid_center_x - (grid_extent_x / 2.0)
    start_y = grid_center_y - (grid_extent_y / 2.0)

    # Random cell selection
    total_cells = args.grid_size * args.grid_size
    selected_indices = np.random.choice(total_cells, size=args.num_cubes, replace=False)

    for idx in selected_indices:
        row = idx // args.grid_size
        col = idx % args.grid_size

        x = start_x + (row * cube_spacing)
        y = start_y + (col * cube_spacing)
        z = 0.05

        cube_name = f"cube_{row}_{col}"
        cube = world.scene.add(
            DynamicCuboid(
                prim_path=f"/World/{cube_name}",
                name=cube_name,
                position=np.array([x, y, z]),
                scale=np.array([0.04, 0.04, 0.04]),
                color=np.array([np.random.rand(), np.random.rand(), np.random.rand()])
            )
        )
        cubes.append((cube, cube_name))

    # Reset world
    print("[SETUP] Resetting world...")
    world.reset()

    # Create RRT planner
    print("[SETUP] Creating RRT planner...")
    from isaacsim.robot_motion.motion_generation.lula import RmpFlow
    rrt = RRT(
        robot_description_path="path_planner_configs/franka/rmpflow/robot_descriptor.yaml",
        urdf_path="path_planner_configs/franka/franka.urdf",
        rrt_config_path="path_planner_configs/franka/rrt/franka_rrt.yaml",
        end_effector_frame_name="panda_hand"
    )

    path_planner_visualizer = PathPlannerVisualizer(
        robot_prim_path=franka_prim_path,
        robot_articulation=franka
    )

    # Create controller
    franka_controller = type('FrankaController', (), {
        'franka': franka,
        'gripper': gripper,
        'container': container,
        'rrt': rrt,
        'path_planner_visualizer': path_planner_visualizer,
        'cubes': cubes
    })()

    # Create heuristic selector
    selector = PureHeuristicSelector(franka_controller)

    # Run pick-and-place with pure heuristics
    print("\n" + "=" * 80)
    print("STARTING PURE HEURISTIC PICK-AND-PLACE")
    print("=" * 80)

    picked_indices = []
    obstacle_positions = []  # No obstacles for now
    container_position = container.get_world_pose()[0]

    start_time = time.time()

    for pick_num in range(args.num_cubes):
        print(f"\n[PICK {pick_num + 1}/{args.num_cubes}]")

        # Select next object using heuristics
        best_index, best_score, breakdown = selector.select_next_object(
            cubes, picked_indices, container_position, obstacle_positions
        )

        if best_index == -1:
            print("No more objects to pick!")
            break

        cube, cube_name = cubes[best_index]
        print(f"Selected: {cube_name}")
        print(f"  Heuristic Score: {best_score:.2f}")
        print(f"    - EE Distance Score: {breakdown['ee_score']:.2f}")
        print(f"    - Container Score: {breakdown['container_score']:.2f}")
        print(f"    - Obstacle Penalty: {breakdown['obstacle_penalty']:.2f}")

        # Mark as picked
        picked_indices.append(best_index)

        # Simulate pick (just for timing - actual RRT execution would go here)
        time.sleep(0.5)

    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("PURE HEURISTIC BASELINE COMPLETE")
    print("=" * 80)
    print(f"Total picks: {len(picked_indices)}/{args.num_cubes}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per pick: {total_time/len(picked_indices):.2f}s")
    print("=" * 80)

    simulation_app.close()


if __name__ == "__main__":
    main()


