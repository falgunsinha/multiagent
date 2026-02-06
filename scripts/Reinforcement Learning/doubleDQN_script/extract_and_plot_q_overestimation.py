import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore
runs_to_analyze = {
    'A* Grid 4 Objects 9': 'run-20251220_021934-saemm4ho',
    'RRT Viz Grid 4 Objects 9': 'run-20251220_134743-dbi680zb',
    'RRT IsaacSim Grid 4 Objects 9': 'run-20251224_185719-q0xud95c'
}

wandb_dir = Path(r'C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\wandb')

def extract_history_from_wandb(wandb_file_path, debug=False):
    """Extract history data from wandb binary file"""

    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_file_path))

    history_data = []
    record_count = 0
    history_count = 0
    all_keys_seen = set()

    print(f"  Reading {wandb_file_path.name}...")

    while True:
        try:
            data = ds.scan_data()
            if data is None:
                break
            pb = wandb_internal_pb2.Record()
            pb.ParseFromString(data)

            record_count += 1
            record_type = pb.WhichOneof("record_type")
            if record_type == "history":
                history_count += 1
                metrics = {}

                for item in pb.history.item:
                    if item.nested_key:
                        key = item.nested_key[0]  # Get first element of nested_key
                    else:
                        key = item.key

                    all_keys_seen.add(key)
                    try:
                        value = json.loads(item.value_json)
                        metrics[key] = value
                    except:
                        try:
                            metrics[key] = float(item.value_json)
                        except:
                            metrics[key] = item.value_json

                if debug and history_count <= 3:
                    print(f"    History record {history_count} keys: {list(metrics.keys())[:15]}")

                history_data.append(metrics)

            if record_count % 50000 == 0:
                print(f"    Processed {record_count} records, {history_count} history records")

        except Exception as e:
            if debug:
                print(f"    Error at record {record_count}: {e}")
            break

    print(f"  Total records: {record_count}")
    print(f"  History records: {history_count}")
    print(f"  Unique keys found: {len(all_keys_seen)}")
    if debug:
        print(f"  Keys: {sorted(all_keys_seen)}")

    if 'ddqn/q_overestimation' in all_keys_seen:
        print(f"  ✓ Found 'ddqn/q_overestimation' key")
        # Filter to only records with q_overestimation
        filtered_data = [d for d in history_data if 'ddqn/q_overestimation' in d]
        print(f"  Records with q_overestimation: {len(filtered_data)}")
        return pd.DataFrame(filtered_data) if filtered_data else None
    else:
        print(f"  ✗ 'ddqn/q_overestimation' key NOT found")
        print(f"  Available DDQN keys: {[k for k in sorted(all_keys_seen) if 'ddqn' in k.lower() or 'q_' in k.lower()]}")
        return None

print("Extracting data from wandb runs...")
print("="*80)

all_data = {}

for name, run_id in runs_to_analyze.items():
    run_dir = wandb_dir / run_id
    wandb_file = list(run_dir.glob('*.wandb'))[0]

    print(f"\n{name}:")
    df = extract_history_from_wandb(wandb_file, debug=True)

    if df is not None and len(df) > 0:
        # Convert string values to numeric if needed
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

        all_data[name] = df
        print(f"  DataFrame shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()[:20]}")  # Show first 20 columns
        print(f"  Sample data:")
        print(df[['global_step', 'ddqn/q_overestimation']].head() if 'global_step' in df.columns else df.head())

print("\n" + "="*80)
print("Creating visualization...")
print("="*80)

if all_data:
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = {
        'A* Grid 4 Objects 9': '#FF6B6B',  # Red
        'RRT Viz Grid 4 Objects 9': '#4ECDC4',  # Teal
        'RRT IsaacSim Grid 4 Objects 9': '#45B7D1'  # Blue
    }

    for name, df in all_data.items():
        # Use global_step as x-axis (timesteps)
        x_col = 'global_step' if 'global_step' in df.columns else '_step'

        # Sort by timestep
        df_sorted = df.sort_values(x_col).reset_index(drop=True)

        x = df_sorted[x_col].values
        y = df_sorted['ddqn/q_overestimation'].values

        # Apply exponential moving average for smoothing
        def smooth(data, weight=0.85):
            """Exponential moving average smoothing"""
            smoothed = []
            last = data[0]
            for point in data:
                smoothed_val = last * weight + (1 - weight) * point
                smoothed.append(smoothed_val)
                last = smoothed_val
            return smoothed

        y_smooth = smooth(y, weight=0.85)
        color = colors.get(name, '#888888')
        plt.plot(x, y_smooth, label=name, color=color, linewidth=2.5, alpha=0.9)
        plt.plot(x, y, color=color, alpha=0.15, linewidth=0.8)

    plt.xlabel('Timesteps', fontsize=13, fontweight='bold')
    plt.ylabel('Q-Overestimation', fontsize=13, fontweight='bold')
    plt.title('Q-Overestimation vs Timesteps (Grid 4 Objects 9)',
              fontsize=15, fontweight='bold', pad=15)
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax = plt.gca()
    ax.set_facecolor('#F8F9FA')

    plt.tight_layout()
    output_file = wandb_dir.parent / 'q_overestimation_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved to: {output_file}")
    plt.show()

    print("\nVisualization complete!")
else:
    print("\nNo data extracted. Please check the wandb files.")

