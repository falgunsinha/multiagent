import json
import glob
from pathlib import Path

# Find all wandb summary files
wandb_dirs = glob.glob(r'wandb\run-*')

results = []
for wandb_dir in wandb_dirs:
    summary_file = Path(wandb_dir) / 'files' / 'wandb-summary.json'
    metadata_file = Path(wandb_dir) / 'files' / 'wandb-metadata.json'

    if summary_file.exists():
        with open(summary_file, 'r') as f:
            data = json.load(f)

        # Extract metrics
        distance_reduced = data.get('episode/distance_reduced', 0)
        time_saved = data.get('episode/time_saved', 0)
        reward = data.get('episode/total_reward', 0)
        reshuffles = data.get('episode/reshuffles_performed', 0)
        cubes_picked = data.get('episode/cubes_picked', 0)

        # Get config info from metadata
        env_type = 'Unknown'
        grid_size = '?'
        num_cubes = '?'

        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                args = metadata.get('args', [])
                program = metadata.get('program', '')

                # Parse args to find grid_size and num_cubes
                for i, arg in enumerate(args):
                    if arg == '--grid_size' and i+1 < len(args):
                        grid_size = args[i+1]
                    if arg == '--num_cubes' and i+1 < len(args):
                        num_cubes = args[i+1]

                # Determine environment type from program path
                if 'train_isaacsim_mappo.py' in program:
                    env_type = 'Isaac Sim'
                elif 'train_rrt_mappo.py' in program:
                    env_type = 'RRT Viz'
                elif 'train_astar_mappo.py' in program:
                    env_type = 'A*'

        config = f'{grid_size}x{grid_size}, {num_cubes} cubes'

        results.append({
            'config': config,
            'env_type': env_type,
            'distance_reduced': distance_reduced,
            'time_saved': time_saved,
            'reward': reward,
            'reshuffles': reshuffles,
            'cubes_picked': cubes_picked
        })

# Sort by config and env_type
results.sort(key=lambda x: (x['config'], x['env_type']))

# Print results
print('=' * 130)
header = f"{'Config':<15} | {'Environment':<10} | {'Dist Reduced':>12} | {'Time Saved':>10} | {'Reward':>8} | {'Reshuffles':>10} | {'Cubes':>6}"
print(header)
print('=' * 130)
for r in results:
    row = f"{r['config']:<15} | {r['env_type']:<10} | {r['distance_reduced']:12.3f} | {r['time_saved']:10.3f} | {r['reward']:8.1f} | {r['reshuffles']:10} | {r['cubes_picked']:6}"
    print(row)
print('=' * 130)

