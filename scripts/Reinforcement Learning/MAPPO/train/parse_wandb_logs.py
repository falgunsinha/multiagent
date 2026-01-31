"""
Parse WandB logs to extract cubes_picked data
"""

import json
from pathlib import Path

# Isaac Sim run
wandb_run = Path(r'wandb\run-20260114_171728-xac7e6kh')

print('=' * 100)
print('PARSING WANDB LOGS - Isaac Sim Run (171728)')
print('=' * 100)
print()

# Read summary
summary_file = wandb_run / 'files' / 'wandb-summary.json'
if summary_file.exists():
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    print('WandB Summary (Final Episode):')
    print('-' * 100)
    print(f'  global_step: {summary.get("global_step")}')
    print(f'  episode/cubes_picked: {summary.get("episode/cubes_picked")}')
    print(f'  episode/total_reward: {summary.get("episode/total_reward")}')
    print(f'  episode/reshuffles_performed: {summary.get("episode/reshuffles_performed")}')
    print(f'  episode/distance_reduced: {summary.get("episode/distance_reduced")}')
    print(f'  episode/time_saved: {summary.get("episode/time_saved")}')
    print()

# Read config
config_file = wandb_run / 'files' / 'config.yaml'
if config_file.exists():
    with open(config_file, 'r') as f:
        config = f.read()
    
    print('WandB Config:')
    print('-' * 100)
    # Extract key config values
    for line in config.split('\n'):
        if 'grid_size' in line or 'num_cubes' in line or 'timesteps' in line:
            print(f'  {line.strip()}')
    print()

# Try to read the wandb event file
wandb_file = wandb_run / f'run-{wandb_run.name.split("-")[-1]}.wandb'
if wandb_file.exists():
    print(f'WandB Event File: {wandb_file}')
    print(f'File size: {wandb_file.stat().st_size} bytes')
    print()
    print('Note: WandB event files are binary and require wandb library to parse.')
    print('Use: wandb sync <run_path> to sync offline runs')
    print()

# Compare with CSV log
csv_log = Path(r'..\logs\mappo_rrt_isaacsim_grid4_cubes9_20260114_171716_20260114_171747_episode_log.csv')
if csv_log.exists():
    import pandas as pd
    df = pd.read_csv(csv_log)
    
    print('CSV Log Data:')
    print('-' * 100)
    print(f'Total episodes: {len(df)}')
    
    # Check cube distances in final episode
    cube_dist_cols = [col for col in df.columns if 'cube_' in col and 'final_dist' in col]
    final_episode = df.iloc[-1]
    
    cubes_picked_csv = sum([1 for col in cube_dist_cols if final_episode[col] < 0.01])
    
    print(f'Final episode (CSV):')
    print(f'  Episode: {final_episode["episode"]}')
    print(f'  Total reward: {final_episode["total_reward"]}')
    print(f'  Reshuffles: {final_episode["reshuffles_performed"]}')
    print(f'  Episode length: {final_episode["episode_length"]}')
    print(f'  Cubes picked (from distances): {cubes_picked_csv}/{len(cube_dist_cols)}')
    print()
    
    print('DISCREPANCY FOUND:')
    print('-' * 100)
    print(f'  WandB shows: {summary.get("episode/cubes_picked")} cubes picked')
    print(f'  CSV shows: {cubes_picked_csv} cubes picked (based on final distances)')
    print()
    
    if summary.get("episode/cubes_picked") != cubes_picked_csv:
        print('ANALYSIS:')
        print('  WandB logs "cubes_picked" from info dict: len(self.base_env.objects_picked)')
        print('  CSV logs final cube distances from environment state')
        print()
        print('  Possible explanations:')
        print('  1. objects_picked list is being populated but cubes are not actually moved')
        print('  2. Cubes are picked but then reset/moved back')
        print('  3. Distance calculation is incorrect')
        print('  4. WandB is logging cumulative picks across episodes')
        print('  5. There is a bug in how objects_picked is tracked')

