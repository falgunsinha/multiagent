import json
import pandas as pd
from pathlib import Path
import sys
runs_to_analyze = {
    'A* Grid 4 Objects 9': 'run-20251220_021934-saemm4ho',
    'RRT Viz Grid 4 Objects 9': 'run-20251220_134743-dbi680zb',
    'RRT IsaacSim Grid 4 Objects 9': 'run-20251224_185719-q0xud95c'
}

wandb_dir = Path(r'C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\wandb')

def parse_wandb_file(wandb_file_path):
    """
    Parse the wandb binary file to extract history data.
    """
    try:
      
        from wandb.proto import wandb_internal_pb2
        
        history_data = []
        
        with open(wandb_file_path, 'rb') as f:
            while True:
                # Read the length prefix (varint)
                try:
                    # Read records from the file
                    # Each record is length-prefixed
                    length_bytes = f.read(4)
                    if not length_bytes or len(length_bytes) < 4:
                        break
                    
                    # Parse the record
                    record = wandb_internal_pb2.Record()
                    record_bytes = f.read()
                    
                    # This is getting complex, let's try a different approach
                    break
                    
                except Exception as e:
                    break
        
        return history_data
        
    except ImportError:
        print("wandb.proto module not available")
        return None
    except Exception as e:
        print(f"Error parsing wandb file: {e}")
        return None

import subprocess
import os

print("Exporting wandb data using CLI...")
print("="*80)

all_data = {}

for name, run_id in runs_to_analyze.items():
    run_dir = wandb_dir / run_id
    actual_run_id = run_dir.name.split('-')[-1]
    
    print(f"\nProcessing: {name}")
    print(f"Run ID: {actual_run_id}")
    output_csv = run_dir / 'files' / 'exported_history.csv'
    os.chdir(run_dir.parent)
    cmd = f'wandb export --id {actual_run_id} --csv'
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"stdout: {result.stdout[:500]}")
        if result.stderr:
            print(f"stderr: {result.stderr[:500]}")
            
    except subprocess.TimeoutExpired:
        print("Command timed out")
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "="*80)
print("Trying alternative: Read from wandb service database")
print("="*80)
wandb_service_dir = wandb_dir.parent / '.wandb'
if wandb_service_dir.exists():
    print(f"Found wandb service directory: {wandb_service_dir}")
    for item in wandb_service_dir.iterdir():
        print(f"  - {item.name}")
else:
    print("No wandb service directory found")

print("\n" + "="*80)
print("Final attempt: Parse the output.log file for metrics")
print("="*80)
for name, run_id in runs_to_analyze.items():
    run_dir = wandb_dir / run_id
    output_log = run_dir / 'files' / 'output.log'
    
    if output_log.exists():
        print(f"\n{name}:")
        print(f"  Log file size: {output_log.stat().st_size} bytes")
        
        # Read first few lines to see the format
        with open(output_log, 'r') as f:
            lines = f.readlines()
            print(f"  Total lines: {len(lines)}")
            print(f"  First 5 lines:")
            for line in lines[:5]:
                print(f"    {line.rstrip()}")

