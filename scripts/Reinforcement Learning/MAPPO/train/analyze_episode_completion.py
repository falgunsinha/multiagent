"""
Analyze why Isaac Sim episodes are not completing
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Read the Isaac Sim log
log_file = Path(r'..\logs\mappo_rrt_isaacsim_grid4_cubes9_20260114_171716_20260114_171747_episode_log.csv')
df = pd.read_csv(log_file)

print('=' * 80)
print('Isaac Sim RRT - 4x4, 9 cubes Analysis')
print('=' * 80)
print(f'Total episodes: {len(df)}')
print(f'Avg episode length: {df["episode_length"].mean():.1f} steps')
print(f'Avg reshuffles: {df["reshuffles_performed"].mean():.1f}')
print(f'Avg reward: {df["total_reward"].mean():.1f}')
print()

# Check final cube distances
cube_dist_cols = [col for col in df.columns if 'cube_' in col and 'final_dist' in col]
print(f'Number of cubes tracked: {len(cube_dist_cols)}')
print()

# Analyze episode completion
print('Episode Completion Analysis:')
print('-' * 80)
for idx, row in df.head(10).iterrows():
    ep = row['episode']
    length = row['episode_length']
    reshuffles = row['reshuffles_performed']
    reward = row['total_reward']
    
    # Count how many cubes were picked (distance = 0 means picked)
    cubes_remaining = sum([1 for col in cube_dist_cols if row[col] > 0.01])
    cubes_picked = len(cube_dist_cols) - cubes_remaining
    
    print(f'Episode {ep:2d}: Length={length:2d}, Reshuffles={reshuffles}, '
          f'Cubes Picked={cubes_picked}/{len(cube_dist_cols)}, Reward={reward:.1f}')

print()
print('=' * 80)
print('Cube Distance Statistics (Episode 1):')
print('=' * 80)
for col in cube_dist_cols:
    dist = df[col].iloc[0]
    status = 'PICKED' if dist < 0.01 else 'REMAINING'
    print(f'{col:20s}: {dist:6.3f}m  [{status}]')

print()
print('=' * 80)
print('Diagnosis:')
print('=' * 80)
print('Expected: 9 cubes should be picked (all distances ~0)')
print(f'Actual: Only {9 - cubes_remaining} cubes picked on average')
print()
print('Possible causes:')
print('1. Action masking too aggressive (cubes marked as unreachable)')
print('2. RRT path planning failures (cubes blocked by obstacles)')
print('3. DDQN agent selecting invalid actions')
print('4. Episode timeout (max_episode_steps=50 in TwoAgentEnv)')
print('5. Reshuffling placing cubes in unreachable positions')

