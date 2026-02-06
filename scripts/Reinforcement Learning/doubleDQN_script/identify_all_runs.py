from pathlib import Path
import yaml

wandb_dir = Path(r'C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\wandb')

runs_by_config = {}

print("Analyzing all wandb runs...")
print("="*80)

for run_dir in sorted(wandb_dir.glob('run-*')):
    config_file = run_dir / 'files' / 'config.yaml'
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    
        method = config.get('method', {}).get('value', 'unknown')
        grid_size = config.get('grid_size', {}).get('value', 'unknown')
        num_cubes = config.get('num_cubes', {}).get('value', 'unknown')
        timesteps = config.get('timesteps', {}).get('value', 'unknown')
        config_key = f"{method}_grid{grid_size}_cubes{num_cubes}"
        if config_key not in runs_by_config:
            runs_by_config[config_key] = []
        
        runs_by_config[config_key].append({
            'run_id': run_dir.name,
            'method': method,
            'grid_size': grid_size,
            'num_cubes': num_cubes,
            'timesteps': timesteps,
            'run_dir': run_dir
        })

print("\nRuns grouped by configuration:")
print("="*80)

for config_key, runs in sorted(runs_by_config.items()):
    print(f"\n{config_key}:")
    print(f"  Number of runs: {len(runs)}")
    for i, run in enumerate(runs, 1):
        print(f"  Run {i}: {run['run_id']} (timesteps: {run['timesteps']})")


print("\n" + "="*80)
print("Grid 4 Objects 9 configurations:")
print("="*80)

grid4_cubes9_configs = {k: v for k, v in runs_by_config.items() if 'grid4_cubes9' in k}

for config_key, runs in sorted(grid4_cubes9_configs.items()):
    print(f"\n{config_key}:")
    print(f"  Number of runs: {len(runs)}")
    for i, run in enumerate(runs, 1):
        print(f"  Run {i}: {run['run_id']}")
        
        # Check if wandb file exists
        wandb_file = list(run['run_dir'].glob('*.wandb'))
        if wandb_file:
            size_mb = wandb_file[0].stat().st_size / (1024 * 1024)
            print(f"         File: {wandb_file[0].name} ({size_mb:.1f} MB)")

print("\n" + "="*80)
print("Summary:")
print("="*80)
print(f"Total configurations: {len(runs_by_config)}")
print(f"Total runs: {sum(len(runs) for runs in runs_by_config.values())}")
print(f"Grid 4 Objects 9 configurations: {len(grid4_cubes9_configs)}")
print(f"Grid 4 Objects 9 runs: {sum(len(runs) for runs in grid4_cubes9_configs.values())}")

