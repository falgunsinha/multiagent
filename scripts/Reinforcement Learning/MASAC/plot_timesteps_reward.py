"""
Timesteps vs Reward Curve Plot using Seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set seaborn theme
sns.set_theme(style="darkgrid")

def load_timestep_data(log_dir: Path):
    """Load MASAC timestep-level results"""
    all_data = []
    
    for summary_file in log_dir.glob("masac_*_summary.json"):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
            env_type = summary['env_type']
            timestamp = summary['timestamp']
            
            # Load timestep log
            timestep_log = log_dir / f"masac_{env_type}_grid{summary['grid_size']}_cubes{summary['num_cubes']}_{timestamp}_timestep_log.csv"
            
            if timestep_log.exists():
                df = pd.read_csv(timestep_log)
                df['planner'] = env_type.upper().replace('_', ' ')
                all_data.append(df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None

if __name__ == "__main__":
    log_dir = Path(__file__).parent / "logs"
    df = load_timestep_data(log_dir)
    
    if df is None or df.empty:
        print("❌ No timestep data found. Please run the test script first.")
        exit(1)
    
    print(f"Loaded {len(df)} timesteps for planners: {df['planner'].unique()}")
    
    # Create curve plot: Timesteps vs Cumulative Reward
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        data=df,
        x='timestep',
        y='cumulative_reward',
        hue='planner',
        palette=['#2E86AB', '#A23B72'],
        linewidth=2.5
    )
    
    plt.xlabel('Timesteps', fontsize=14, fontweight='bold')
    plt.ylabel('Reward', fontsize=14, fontweight='bold')
    plt.title('MASAC Performance: Timesteps vs Reward', fontsize=16, fontweight='bold')
    plt.legend(title='Planner', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    
    save_path = log_dir / "masac_timesteps_reward_curve.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved to: {save_path}")
    plt.show()

