"""
Parse wandb offline runs by reading the protobuf binary files directly.
"""

import pandas as pd
from pathlib import Path
import os

# Disable wandb online mode
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_DISABLE_CODE'] = 'true'

# Define the runs we want to analyze
runs_to_analyze = {
    'A* Grid 4 Objects 9': 'run-20251220_021934-saemm4ho',
    'RRT Viz Grid 4 Objects 9': 'run-20251220_134743-dbi680zb',
    'RRT IsaacSim Grid 4 Objects 9': 'run-20251224_185719-q0xud95c'
}

wandb_dir = Path(r'C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\wandb')

print("Parsing wandb offline runs...")
print("="*80)

# Try to read the wandb files using wandb's internal protobuf reader
try:
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import file_stream
    
    all_data = {}
    
    for name, run_id in runs_to_analyze.items():
        run_dir = wandb_dir / run_id
        wandb_file = list(run_dir.glob('*.wandb'))[0]
        
        print(f"\nProcessing: {name}")
        print(f"File: {wandb_file}")
        
        # Read the wandb file
        history_data = []
        
        with open(wandb_file, 'rb') as f:
            # The file contains a series of protobuf records
            # Each record is prefixed with its length
            
            record_count = 0
            history_count = 0
            
            while True:
                try:
                    # Read the length prefix (4 bytes, little-endian)
                    length_bytes = f.read(4)
                    if not length_bytes or len(length_bytes) < 4:
                        break
                    
                    length = int.from_bytes(length_bytes, byteorder='little')
                    
                    # Read the record data
                    record_bytes = f.read(length)
                    if len(record_bytes) < length:
                        break
                    
                    # Parse the record
                    record = pb.Record()
                    record.ParseFromString(record_bytes)
                    
                    record_count += 1
                    
                    # Check if this is a history record
                    if record.HasField('history'):
                        history_count += 1
                        history = record.history
                        
                        # Extract the metrics
                        metrics = {}
                        
                        # Get global_step
                        for item in history.item:
                            if item.key == 'global_step':
                                metrics['global_step'] = item.value_json
                            elif item.key == 'ddqn/q_overestimation':
                                metrics['q_overestimation'] = item.value_json
                            elif item.key == '_step':
                                metrics['_step'] = item.value_json
                        
                        if 'q_overestimation' in metrics:
                            history_data.append(metrics)
                    
                    # Limit to avoid memory issues
                    if record_count % 10000 == 0:
                        print(f"  Processed {record_count} records, found {history_count} history records, {len(history_data)} with q_overestimation")
                    
                except Exception as e:
                    # End of file or parse error
                    break
            
            print(f"  Total records: {record_count}")
            print(f"  History records: {history_count}")
            print(f"  Q-overestimation data points: {len(history_data)}")
            
            if history_data:
                # Convert to DataFrame
                df = pd.DataFrame(history_data)
                
                # Parse JSON strings to floats
                for col in df.columns:
                    try:
                        df[col] = df[col].apply(lambda x: float(x) if isinstance(x, str) else x)
                    except:
                        pass
                
                all_data[name] = df
                print(f"  DataFrame shape: {df.shape}")
                print(f"  Columns: {df.columns.tolist()}")
                if len(df) > 0:
                    print(f"  First few rows:")
                    print(df.head())
    
    print("\n" + "="*80)
    print("Data extraction complete!")
    print("="*80)
    
    # Save the data
    for name, df in all_data.items():
        output_file = wandb_dir / f"{name.replace(' ', '_').replace('*', 'star')}_data.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")
    
except ImportError as e:
    print(f"Error importing wandb protobuf: {e}")
    print("Trying alternative method...")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

