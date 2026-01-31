"""
Compare action masking behavior across A*, RRT Viz, and Isaac Sim environments
"""

import pandas as pd
import numpy as np
from pathlib import Path

print('=' * 100)
print('MAPPO ENVIRONMENT COMPARISON - Action Masking & Reward Analysis')
print('=' * 100)
print()

# Read all three log files
logs = {
    'A*': Path(r'..\logs\mappo_astar_grid4_cubes9_20260114_173732_20260114_173746_episode_log.csv'),
    'RRT Viz': Path(r'..\logs\mappo_rrt_viz_grid4_cubes9_20260114_173017_20260114_173040_episode_log.csv'),
    'Isaac Sim': Path(r'..\logs\mappo_rrt_isaacsim_grid4_cubes9_20260114_171716_20260114_171747_episode_log.csv'),
}

data = {}
for env_name, log_file in logs.items():
    if log_file.exists():
        data[env_name] = pd.read_csv(log_file)
    else:
        print(f'WARNING: {env_name} log not found: {log_file}')

print('EPISODE COMPLETION COMPARISON (4x4 grid, 9 cubes, 50 timesteps)')
print('-' * 100)
print(f'{"Environment":<15} {"Episodes":<10} {"Avg Length":<12} {"Avg Reshuffles":<15} {"Avg Reward":<12} {"Avg Cubes Picked":<18}')
print('-' * 100)

cube_dist_cols = None
for env_name, df in data.items():
    if cube_dist_cols is None:
        cube_dist_cols = [col for col in df.columns if 'cube_' in col and 'final_dist' in col]
    
    # Calculate average cubes picked per episode
    cubes_picked_per_episode = []
    for idx, row in df.iterrows():
        cubes_picked = sum([1 for col in cube_dist_cols if row[col] < 0.01])
        cubes_picked_per_episode.append(cubes_picked)
    
    avg_cubes_picked = np.mean(cubes_picked_per_episode)
    
    print(f'{env_name:<15} {len(df):<10} {df["episode_length"].mean():<12.1f} '
          f'{df["reshuffles_performed"].mean():<15.1f} {df["total_reward"].mean():<12.1f} '
          f'{avg_cubes_picked:<18.1f}')

print()
print('=' * 100)
print('DETAILED EPISODE ANALYSIS (First 6 episodes)')
print('=' * 100)

for env_name, df in data.items():
    print()
    print(f'{env_name} Environment:')
    print('-' * 100)
    print(f'{"Episode":<10} {"Length":<10} {"Reshuffles":<12} {"Reward":<10} {"Cubes Picked":<15} {"Status"}')
    print('-' * 100)
    
    for idx, row in df.head(6).iterrows():
        ep = row['episode']
        length = row['episode_length']
        reshuffles = row['reshuffles_performed']
        reward = row['total_reward']
        
        # Count cubes picked
        cubes_picked = sum([1 for col in cube_dist_cols if row[col] < 0.01])
        total_cubes = len(cube_dist_cols)
        
        # Determine status
        if cubes_picked == total_cubes:
            status = '✓ COMPLETE'
        elif cubes_picked == 0:
            status = '✗ NO PICKS'
        else:
            status = f'⚠ PARTIAL ({cubes_picked}/{total_cubes})'
        
        print(f'{ep:<10} {length:<10} {reshuffles:<12} {reward:<10.1f} {cubes_picked}/{total_cubes:<12} {status}')

print()
print('=' * 100)
print('CUBE DISTANCE ANALYSIS (Episode 1)')
print('=' * 100)

for env_name, df in data.items():
    print()
    print(f'{env_name} Environment - Cube Final Distances:')
    print('-' * 100)
    
    row = df.iloc[0]
    for i, col in enumerate(cube_dist_cols):
        dist = row[col]
        status = 'PICKED ✓' if dist < 0.01 else f'REMAINING ({dist:.3f}m)'
        print(f'  Cube {i}: {status}')
    
    cubes_picked = sum([1 for col in cube_dist_cols if row[col] < 0.01])
    print(f'  Total: {cubes_picked}/{len(cube_dist_cols)} cubes picked')

print()
print('=' * 100)
print('DIAGNOSIS: Why is Isaac Sim different?')
print('=' * 100)
print()
print('KEY FINDINGS:')
print()
print('1. ACTION MASKING BEHAVIOR:')
print('   - A*: Uses A* path planning for reachability check')
print('   - RRT Viz: Uses PythonRobotics RRT for reachability check')
print('   - Isaac Sim: Uses Isaac Sim RRT (franka_controller.rrt.compute_path) with max_retries=1')
print()
print('2. REACHABILITY CHECK DIFFERENCES:')
print('   - A* & RRT Viz: Fast, deterministic path planning')
print('   - Isaac Sim: Slower, probabilistic RRT with potential failures')
print()
print('3. POTENTIAL ISSUES IN ISAAC SIM:')
print('   - RRT planning may fail due to:')
print('     a) Obstacles blocking paths (Lidar-detected obstacles)')
print('     b) max_retries=1 (only 1 attempt per cube)')
print('     c) max_iterations=8000 (may be insufficient)')
print('     d) Collision checking with actual robot geometry')
print()
print('4. EPISODE TERMINATION:')
print('   - If ALL cubes are masked as unreachable → Episode ends immediately')
print('   - This explains 0 cubes picked in Isaac Sim')
print()
print('5. REWARD CALCULATION (MAPPO):')
print('   - Agent 2 reward = reshuffle_reward + (0.5 × pick_reward)')
print('   - If no picks happen, only reshuffle rewards are earned')
print('   - Reshuffle rewards are based on distance improvement')
print()

