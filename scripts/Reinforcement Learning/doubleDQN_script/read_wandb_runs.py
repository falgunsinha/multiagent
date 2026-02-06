import pandas as pd
from pathlib import Path
import wandb
import os

runs_to_analyze = {
    'A* Grid 4 Objects 9': 'run-20251220_021934-saemm4ho',
    'RRT Viz Grid 4 Objects 9': 'run-20251220_134743-dbi680zb',
    'RRT IsaacSim Grid 4 Objects 9': 'run-20251224_185719-q0xud95c'
}

wandb_dir = Path(r'C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\wandb')

print("Attempting to read wandb runs using Python API...")
print("="*80)

try:
    api = wandb.Api()
    
    print("Checking for available runs...")
    from wandb.sdk.lib import runid
    from wandb.sdk.wandb_run import Run
    
    for name, run_id in runs_to_analyze.items():
        run_dir = wandb_dir / run_id
        actual_run_id = run_dir.name.split('-')[-1]
        
        print(f"\n{name} (ID: {actual_run_id}):")
        
        import yaml
        config_file = run_dir / 'files' / 'config.yaml'
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                print(f"  Config loaded")
        
        metadata_file = run_dir / 'files' / 'wandb-metadata.json'
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                print(f"  Metadata: {metadata.keys()}")
                
                # Try to get entity and project
                if 'entity' in metadata:
                    print(f"    Entity: {metadata['entity']}")
                if 'project' in metadata:
                    print(f"    Project: {metadata['project']}")
                if 'name' in metadata:
                    print(f"    Name: {metadata['name']}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Trying to use wandb.restore() to get run data...")
print("="*80)

for name, run_id in runs_to_analyze.items():
    run_dir = wandb_dir / run_id
    actual_run_id = run_dir.name.split('-')[-1]
    
    print(f"\n{name}:")
    
    os.chdir(run_dir)
    
    try:
        wandb_file = list(run_dir.glob('*.wandb'))[0]
        print(f"  Wandb file: {wandb_file.name}")
        print(f"  File size: {wandb_file.stat().st_size} bytes")
        
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "="*80)
print("Checking if we can use wandb's FileStream to read the .wandb file...")
print("="*80)

try:
    from wandb.sdk.internal.file_stream import FileStreamApi
    from wandb.proto import wandb_internal_pb2
    
    for name, run_id in runs_to_analyze.items():
        run_dir = wandb_dir / run_id
        wandb_file = list(run_dir.glob('*.wandb'))[0]
        
        print(f"\n{name}:")
        print(f"  Reading: {wandb_file}")
        
        # Try to read the file
        # This is getting into wandb's internal implementation
        
except ImportError as e:
    print(f"Cannot import wandb internals: {e}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Summary: Need to find alternative method to read wandb offline data")
print("="*80)

