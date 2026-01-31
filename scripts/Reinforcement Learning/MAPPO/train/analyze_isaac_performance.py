import pandas as pd
import numpy as np

# Isaac Sim MAPPO training run
csv_file = r'..\logs\mappo_rrt_isaacsim_grid4_cubes9_20260114_171716_20260114_171747_episode_log.csv'
df = pd.read_csv(csv_file)

print('=' * 100)
print('ISAAC SIM MAPPO TRAINING - Episode Summary')
print('=' * 100)
print(f'Total episodes: {len(df)}')
print()

for idx, row in df.iterrows():
    ep = int(row['episode'])
    reward = row['total_reward']
    steps = int(row['episode_length'])
    reshuffles = int(row['reshuffles_performed'])
    dist_reduced = row['total_distance_reduced']
    time_saved = row['total_time_saved']
    
    print(f'Episode {ep}:')
    print(f'  Reward: {reward:.2f}')
    print(f'  Steps: {steps}')
    print(f'  Reshuffles: {reshuffles}')
    print(f'  Distance reduced: {dist_reduced:.6f} m')
    print(f'  Time saved: {time_saved:.6f} s')
    print()

print('=' * 100)
print('FINAL EPISODE ANALYSIS')
print('=' * 100)
final = df.iloc[-1]
ep_num = int(final['episode'])
reward = final['total_reward']
steps = int(final['episode_length'])
reshuffles = int(final['reshuffles_performed'])
dist_reduced = final['total_distance_reduced']
time_saved = final['total_time_saved']

print(f'Episode: {ep_num}')
print(f'Total reward: {reward:.2f}')
print(f'Episode length: {steps} steps')
print(f'Reshuffles: {reshuffles}')
print(f'Distance reduced: {dist_reduced:.6f} m')
print(f'Time saved: {time_saved:.6f} s')
print()

# Check cube distances
cube_cols = [col for col in df.columns if 'cube_' in col and 'final_dist' in col]
print(f'Cube final distances (Episode {ep_num}):')
for col in sorted(cube_cols):
    cube_idx = col.split('_')[1]
    dist = final[col]
    print(f'  Cube {cube_idx}: {dist:.3f} m')
print()

print('=' * 100)
print('COMPARISON WITH A* AND RRT VIZ')
print('=' * 100)
print()
print('Expected (from A*/RRT Viz):')
print('  - Distance reduced: ~0.5-1.0 m (POSITIVE)')
print('  - Time saved: ~5-10 s (POSITIVE)')
print('  - All 9 cubes picked')
print()
print('Actual (Isaac Sim MAPPO):')
print(f'  - Distance reduced: {dist_reduced:.6f} m (NEGATIVE!)')
print(f'  - Time saved: {time_saved:.6f} s (NEGATIVE!)')
print(f'  - Only 7 cubes picked')
print()
print('PROBLEM: Reshuffling is making things WORSE, not better!')

