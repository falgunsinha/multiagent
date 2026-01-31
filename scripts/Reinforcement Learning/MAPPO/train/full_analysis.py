import pandas as pd
import json
import os

print('=' * 100)
print('COMPREHENSIVE ANALYSIS: Isaac Sim vs A* vs RRT Viz')
print('=' * 100)
print()

# ============================================================================
# PART 1: CSV COLUMN COMPARISON
# ============================================================================
print('PART 1: CSV COLUMN COMPARISON')
print('=' * 100)

# Isaac Sim
isaac_csv = pd.read_csv(r'..\logs\mappo_rrt_isaacsim_grid4_cubes9_20260114_171716_20260114_171747_episode_log.csv')
print('ISAAC SIM CSV Columns:')
print(list(isaac_csv.columns))
print()

# A*
astar_csv = pd.read_csv(r'..\logs\mappo_astar_grid4_cubes9_20260114_173732_20260114_173746_episode_log.csv')
print('A* CSV Columns:')
print(list(astar_csv.columns))
print()

# RRT Viz
rrt_csv = pd.read_csv(r'..\logs\mappo_rrt_viz_grid4_cubes9_20260114_173017_20260114_173040_episode_log.csv')
print('RRT VIZ CSV Columns:')
print(list(rrt_csv.columns))
print()

has_dist_isaac = 'total_distance_reduced' in isaac_csv.columns
has_time_isaac = 'total_time_saved' in isaac_csv.columns
has_dist_astar = 'total_distance_reduced' in astar_csv.columns
has_time_astar = 'total_time_saved' in astar_csv.columns
has_dist_rrt = 'total_distance_reduced' in rrt_csv.columns
has_time_rrt = 'total_time_saved' in rrt_csv.columns

print('Distance/Time Columns Present:')
print(f'  Isaac Sim: distance={has_dist_isaac}, time={has_time_isaac}')
print(f'  A*: distance={has_dist_astar}, time={has_time_astar}')
print(f'  RRT Viz: distance={has_dist_rrt}, time={has_time_rrt}')
print()

# ============================================================================
# PART 2: CSV DATA COMPARISON
# ============================================================================
print('PART 2: CSV DATA COMPARISON (Final Episodes)')
print('=' * 100)

final_isaac = isaac_csv.iloc[-1]
final_astar = astar_csv.iloc[-1]
final_rrt = rrt_csv.iloc[-1]

print('ISAAC SIM:')
print(f'  Episodes: {len(isaac_csv)}')
print(f'  Final reward: {final_isaac["total_reward"]:.2f}')
print(f'  Final steps: {int(final_isaac["episode_length"])}')
print(f'  Final reshuffles: {int(final_isaac["reshuffles_performed"])}')
if has_dist_isaac:
    print(f'  Distance reduced: {final_isaac["total_distance_reduced"]:.6f} m')
if has_time_isaac:
    print(f'  Time saved: {final_isaac["total_time_saved"]:.6f} s')
print()

print('A*:')
print(f'  Episodes: {len(astar_csv)}')
print(f'  Final reward: {final_astar["total_reward"]:.2f}')
print(f'  Final steps: {int(final_astar["episode_length"])}')
print(f'  Final reshuffles: {int(final_astar["reshuffles_performed"])}')
if has_dist_astar:
    print(f'  Distance reduced: {final_astar["total_distance_reduced"]:.6f} m')
if has_time_astar:
    print(f'  Time saved: {final_astar["total_time_saved"]:.6f} s')
print()

print('RRT VIZ:')
print(f'  Episodes: {len(rrt_csv)}')
print(f'  Final reward: {final_rrt["total_reward"]:.2f}')
print(f'  Final steps: {int(final_rrt["episode_length"])}')
print(f'  Final reshuffles: {int(final_rrt["reshuffles_performed"])}')
if has_dist_rrt:
    print(f'  Distance reduced: {final_rrt["total_distance_reduced"]:.6f} m')
if has_time_rrt:
    print(f'  Time saved: {final_rrt["total_time_saved"]:.6f} s')
print()

# ============================================================================
# PART 3: WANDB DATA
# ============================================================================
print('PART 3: WANDB DATA')
print('=' * 100)

# Isaac Sim WandB
isaac_wandb_file = r'wandb\run-20260114_171728-xac7e6kh\files\wandb-summary.json'
if os.path.exists(isaac_wandb_file):
    with open(isaac_wandb_file, 'r') as f:
        isaac_wandb = json.load(f)
    
    print('ISAAC SIM WandB:')
    print(f'  cubes_picked: {isaac_wandb.get("episode/cubes_picked", "N/A")}')
    print(f'  distance_reduced: {isaac_wandb.get("episode/distance_reduced", "N/A")}')
    print(f'  time_saved: {isaac_wandb.get("episode/time_saved", "N/A")}')
    print(f'  total_reward: {isaac_wandb.get("episode/total_reward", "N/A")}')
    print(f'  reshuffles: {isaac_wandb.get("episode/reshuffles_performed", "N/A")}')
    print()

# Check for A* and RRT Viz WandB runs
wandb_dir = r'wandb'
if os.path.exists(wandb_dir):
    wandb_runs = [d for d in os.listdir(wandb_dir) if d.startswith('run-2026011')]
    print(f'Found {len(wandb_runs)} WandB runs')
    for run in sorted(wandb_runs):
        summary_file = os.path.join(wandb_dir, run, 'files', 'wandb-summary.json')
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                data = json.load(f)
            
            # Try to identify the run type
            config_file = os.path.join(wandb_dir, run, 'files', 'config.yaml')
            run_type = 'Unknown'
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = f.read()
                    if 'rrt_isaacsim' in config:
                        run_type = 'Isaac Sim'
                    elif 'astar' in config:
                        run_type = 'A*'
                    elif 'rrt_viz' in config:
                        run_type = 'RRT Viz'
            
            print(f'{run} ({run_type}):')
            print(f'  cubes_picked: {data.get("episode/cubes_picked", "N/A")}')
            print(f'  distance_reduced: {data.get("episode/distance_reduced", "N/A")}')
            print(f'  time_saved: {data.get("episode/time_saved", "N/A")}')
            print()

print()
print('=' * 100)
print('KEY FINDINGS')
print('=' * 100)
print('1. CSV columns may differ between runs (old vs new logger version)')
print('2. WandB always has cubes_picked, distance_reduced, time_saved')
print('3. Need to check WandB for all three runs to compare properly')

