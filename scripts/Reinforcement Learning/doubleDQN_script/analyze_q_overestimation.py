import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import wandb
from wandb.sdk.wandb_run import Run

# Define the runs we want to analyze
runs_to_analyze = {
    'A* Grid 4 Objects 9': 'run-20251220_021934-saemm4ho',
    'RRT Viz Grid 4 Objects 9': 'run-20251220_134743-dbi680zb',
    'RRT IsaacSim Grid 4 Objects 9': 'run-20251224_185719-q0xud95c'
}

wandb_dir = Path(r'C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\wandb')

def read_wandb_history(run_dir):
    """Read wandb history from local run directory using wandb's internal tools"""
    try:
        # Read the wandb binary file
        wandb_file = list(run_dir.glob('*.wandb'))[0]

        # Use wandb's history reader
        from wandb.sdk.internal.internal_api import Api
        from wandb.sdk.lib import runid
        from wandb.sdk.wandb_run import Run
        from wandb.old.core import wandb_dir as get_wandb_dir

        # Try to read the history using wandb's internal format
        # The .wandb file is a protobuf-based format

        # Alternative: parse the file manually
        import struct

        history_data = []

        with open(wandb_file, 'rb') as f:
            # The wandb file format is complex, let's try a different approach
            pass

        return None

    except Exception as e:
        print(f"Error reading wandb history: {e}")
        import traceback
        traceback.print_exc()
        return None

# Let's try using wandb's offline sync feature
print("Reading wandb data from local files...")
print("="*80)

all_data = {}

# Try to use wandb API to read offline runs
try:
    api = wandb.Api()

    for name, run_id in runs_to_analyze.items():
        run_dir = wandb_dir / run_id
        print(f"\nProcessing: {name}")
        print(f"Run directory: {run_dir}")

        # Extract the actual run ID from the directory name
        actual_run_id = run_dir.name.split('-')[-1]
        print(f"Run ID: {actual_run_id}")

        # Try to read the run - this might not work for offline runs
        # We need to sync the run first or read it differently

        # Let's try reading the wandb file directly
        df = read_wandb_history(run_dir)
        if df is not None:
            all_data[name] = df

except Exception as e:
    print(f"Error using wandb API: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Alternative approach: Using wandb offline sync")
print("="*80)

# Since reading the binary format is complex, let's use wandb's sync command
# to sync the offline runs, which will make them readable via the API
import subprocess

for name, run_id in runs_to_analyze.items():
    run_dir = wandb_dir / run_id
    print(f"\nSyncing {name}...")

    # Note: This will try to sync to wandb servers
    # For offline analysis, we need a different approach

print("\n" + "="*80)
print("Let me try reading the event logs instead...")
print("="*80)

