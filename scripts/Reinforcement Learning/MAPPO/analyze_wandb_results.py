import json
import yaml
from pathlib import Path
import pandas as pd

# Find all wandb run directories
wandb_dir = Path("cobotproject/scripts/Reinforcement Learning/MAPPO/train/wandb")
run_dirs = sorted([d for d in wandb_dir.iterdir() if d.is_dir() and d.name.startswith("run-")])

results = []

for run_dir in run_dirs:
    try:
        # Load config to identify the run
        config_file = run_dir / "files" / "config.yaml"
        summary_file = run_dir / "files" / "wandb-summary.json"
        
        if not config_file.exists() or not summary_file.exists():
            continue
            
        # Parse config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract configuration details
        args_dict = {}
        if '_wandb' in config and 'value' in config['_wandb'] and 'e' in config['_wandb']['value']:
            # Get the first entry (there's only one)
            entry = list(config['_wandb']['value']['e'].values())[0]
            args = entry.get('args', [])
            
            # Parse args into dict
            for i in range(0, len(args), 2):
                if i + 1 < len(args):
                    key = args[i].replace('--', '')
                    value = args[i + 1]
                    args_dict[key] = value
        
        grid_size = args_dict.get('grid_size', 'unknown')
        num_cubes = args_dict.get('num_cubes', 'unknown')
        ddqn_model = args_dict.get('ddqn_model_path', '')
        
        # Determine environment type from model path
        if 'rrt_isaacsim' in ddqn_model.lower():
            env_type = 'RRT IsaacSim'
        elif 'rrt_viz' in ddqn_model.lower():
            env_type = 'RRT Viz'
        elif 'astar' in ddqn_model.lower():
            env_type = 'A*'
        else:
            env_type = 'Unknown'
        
        # Load summary metrics
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        # Extract metrics
        total_reward = summary.get('episode/total_reward', 0)
        reshuffles = summary.get('episode/reshuffles_performed', 0)
        distance_reduced = summary.get('episode/distance_reduced', 0)
        time_saved = summary.get('episode/time_saved', 0)
        cubes_picked = summary.get('episode/cubes_picked', 0)
        
        results.append({
            'Environment': env_type,
            'Grid': f'{grid_size}x{grid_size}',
            'Cubes': int(num_cubes),
            'Total Reward': round(total_reward, 2),
            'Reshuffles': int(reshuffles),
            'Distance Reduced (m)': round(distance_reduced, 3),
            'Time Saved (s)': round(time_saved, 2),
            'Cubes Picked': int(cubes_picked),
            'Run ID': run_dir.name
        })
        
    except Exception as e:
        print(f"Error processing {run_dir.name}: {e}")

# Sort by environment, grid, cubes
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(['Environment', 'Grid', 'Cubes'])

print()
print('='*130)
print('MAPPO WANDB RESULTS COMPARISON (All 9 Configurations - Final Episode Metrics)')
print('='*130)
print(results_df[['Environment', 'Grid', 'Cubes', 'Total Reward', 'Reshuffles', 
                   'Distance Reduced (m)', 'Time Saved (s)', 'Cubes Picked']].to_string(index=False))
print('='*130)
print()

# Summary by environment
print('SUMMARY BY ENVIRONMENT:')
print('-'*130)
for env in ['RRT IsaacSim', 'RRT Viz', 'A*']:
    env_results = results_df[results_df['Environment'] == env]
    if not env_results.empty:
        print(f'\n{env}:')
        for _, row in env_results.iterrows():
            print(f"  {row['Grid']} {row['Cubes']} cubes: "
                  f"Reward={row['Total Reward']:.2f}, "
                  f"Reshuffles={row['Reshuffles']}, "
                  f"Distance={row['Distance Reduced (m)']:.3f}m, "
                  f"Time={row['Time Saved (s)']:.2f}s, "
                  f"Picked={row['Cubes Picked']}")
print('-'*130)
print()

# Overall statistics
print('OVERALL STATISTICS (Final Episode):')
print('-'*130)
print(f"Average Reward: {results_df['Total Reward'].mean():.2f}")
print(f"Average Reshuffles: {results_df['Reshuffles'].mean():.2f}")
print(f"Average Distance Reduced: {results_df['Distance Reduced (m)'].mean():.3f}m")
print(f"Average Time Saved: {results_df['Time Saved (s)'].mean():.2f}s")
print(f"Average Cubes Picked: {results_df['Cubes Picked'].mean():.2f}")
print('-'*130)
print()

# Key insights
print('KEY INSIGHTS:')
print('-'*130)
best_reward = results_df.loc[results_df['Total Reward'].idxmax()]
print(f"✅ Highest Reward: {best_reward['Environment']} {best_reward['Grid']} {best_reward['Cubes']} cubes ({best_reward['Total Reward']:.2f})")
best_distance = results_df.loc[results_df['Distance Reduced (m)'].idxmax()]
print(f"✅ Most Distance Reduced: {best_distance['Environment']} {best_distance['Grid']} {best_distance['Cubes']} cubes ({best_distance['Distance Reduced (m)']:.3f}m)")
worst_distance = results_df.loc[results_df['Distance Reduced (m)'].idxmin()]
print(f"⚠️  Worst Distance (NEGATIVE): {worst_distance['Environment']} {worst_distance['Grid']} {worst_distance['Cubes']} cubes ({worst_distance['Distance Reduced (m)']:.3f}m)")
print('-'*130)

