"""
Episodes vs Reward Curve Plot using Seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set seaborn theme
sns.set_theme(style="darkgrid")

def load_episode_data(log_dir: Path):
    """Load MASAC episode-level results"""
    all_data = []
    
    for summary_file in log_dir.glob("masac_*_summary.json"):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
            env_type = summary['env_type']
            timestamp = summary['timestamp']
            
            # Load episode log
            episode_log = log_dir / f"masac_{env_type}_grid{summary['grid_size']}_cubes{summary['num_cubes']}_{timestamp}_episode_log.csv"
            
            if episode_log.exists():
                df = pd.read_csv(episode_log)
                # Map env_type to readable planner names
                planner_name_map = {
                    'astar': 'A*',
                    'rrt_viz': 'RRT Viz',
                    'rrt_isaacsim': 'Isaac Sim RRT'
                }
                df['planner'] = planner_name_map.get(env_type, env_type.upper().replace('_', ' '))
                all_data.append(df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None

if __name__ == "__main__":
    log_dir = Path(__file__).parent / "logs"
    df = load_episode_data(log_dir)
    
    if df is None or df.empty:
        print("❌ No episode data found. Please run the test script first.")
        exit(1)
    
    print(f"Loaded {len(df)} episodes for planners: {df['planner'].unique()}")
    
    # Create curve plot: Episodes vs Total Reward
    plt.figure(figsize=(10, 6))
    
    # Define color palette
    palette = {
        'A*': '#2E86AB',
        'RRT Viz': '#A23B72',
        'Isaac Sim RRT': '#F18F01'
    }
    
    sns.lineplot(
        data=df,
        x='episode',
        y='total_reward',
        hue='planner',
        palette=palette,
       
    )
    
    plt.xlabel('Episodes', fontsize=14, fontweight='bold')
    plt.ylabel('Reward', fontsize=14, fontweight='bold')
    plt.title('MASAC Performance: Episodes vs Reward', fontsize=16, fontweight='bold')
    plt.legend(title='Planner', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    
    save_path = log_dir / "masac_episodes_reward_curve.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved to: {save_path}")
    plt.show()

