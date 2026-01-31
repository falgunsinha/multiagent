"""
Read wandb files using wandb's FileStreamApi reader.
"""

import pandas as pd
from pathlib import Path
import os
import json

# Disable wandb online mode
os.environ['WANDB_MODE'] = 'offline'

# Define the runs we want to analyze
runs_to_analyze = {
    'A* Grid 4 Objects 9': 'run-20251220_021934-saemm4ho',
    'RRT Viz Grid 4 Objects 9': 'run-20251220_134743-dbi680zb',
    'RRT IsaacSim Grid 4 Objects 9': 'run-20251224_185719-q0xud95c'
}

wandb_dir = Path(r'C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\wandb')

print("Reading wandb files using FileStreamApi...")
print("="*80)

try:
    from wandb.sdk.internal import file_stream
    from wandb.proto import wandb_internal_pb2 as pb
    
    all_data = {}
    
    for name, run_id in runs_to_analyze.items():
        run_dir = wandb_dir / run_id
        wandb_file = list(run_dir.glob('*.wandb'))[0]
        
        print(f"\nProcessing: {name}")
        print(f"File: {wandb_file}")
        
        # Use FileStreamApi to read the file
        fs = file_stream.FileStreamApi(str(wandb_file), 'r')
        
        history_data = []
        record_count = 0
        
        try:
            while True:
                # Read next record
                data = fs.read()
                if data is None:
                    break
                
                record_count += 1
                
                # Parse the record
                record = pb.Record()
                record.ParseFromString(data)
                
                # Check for history data
                if record.HasField('history'):
                    history = record.history
                    
                    metrics = {}
                    for item in history.item:
                        if item.key in ['global_step', 'ddqn/q_overestimation', '_step']:
                            # Parse the value
                            if item.HasField('value_json'):
                                try:
                                    value = json.loads(item.value_json)
                                    metrics[item.key] = value
                                except:
                                    metrics[item.key] = item.value_json
                    
                    if 'ddqn/q_overestimation' in metrics:
                        history_data.append(metrics)
                
                if record_count % 10000 == 0:
                    print(f"  Processed {record_count} records, found {len(history_data)} q_overestimation points")
        
        except EOFError:
            pass
        except Exception as e:
            print(f"  Error reading records: {e}")
        
        finally:
            fs.close()
        
        print(f"  Total records: {record_count}")
        print(f"  Q-overestimation data points: {len(history_data)}")
        
        if history_data:
            df = pd.DataFrame(history_data)
            all_data[name] = df
            print(f"  DataFrame shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()}")
            if len(df) > 0:
                print(f"  Sample data:")
                print(df.head())
    
    print("\n" + "="*80)
    print("Data extraction complete!")
    print("="*80)
    
    if all_data:
        # Save the data
        for name, df in all_data.items():
            safe_name = name.replace(' ', '_').replace('*', 'star')
            output_file = wandb_dir / f"{safe_name}_data.csv"
            df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")
        
        print("\nData successfully extracted!")
    else:
        print("\nNo data extracted. The wandb files might use a different format.")
        print("Let me check the file structure...")
        
        # Try to peek at the file structure
        wandb_file = wandb_dir / runs_to_analyze['A* Grid 4 Objects 9'] / 'run-saemm4ho.wandb'
        with open(wandb_file, 'rb') as f:
            # Read first 100 bytes
            header = f.read(100)
            print(f"\nFirst 100 bytes (hex): {header[:100].hex()}")
            print(f"First 100 bytes (repr): {repr(header[:100])}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

