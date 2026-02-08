import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime


def load_timestep_data(log_dir: Path):
    """Load all timestep CSV files from a log directory"""
    timestep_files = list(log_dir.glob("*_timestep_log.csv"))
    
    if not timestep_files:
        print(f"  No timestep files found in {log_dir}")
        return None
    
    # Load and concatenate all timestep files
    dfs = []
    for file_path in timestep_files:
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f" Loaded {len(timestep_files)} timestep files ({len(combined_df)} total timesteps)")
    
    return combined_df


def create_learning_curve(df: pd.DataFrame, metric: str = 'cumulative_reward', 
                         window_size: int = 100, save_path: Path = None):
    """
    Create MAPPO-style learning curve with confidence intervals
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    scenarios = df['scenario'].unique()
    
    for scenario in scenarios:
        scenario_df = df[df['scenario'] == scenario]
        seeds = scenario_df['seed'].unique()
        smoothed_data = []
        for seed in seeds:
            seed_df = scenario_df[seed_df['seed'] == seed].sort_values('global_timestep')
            
            # Apply rolling mean for smoothing
            smoothed = seed_df[metric].rolling(window=window_size, min_periods=1).mean()
            
            smoothed_data.append({
                'global_timestep': seed_df['global_timestep'].values,
                'smoothed_metric': smoothed.values
            })
        
        if smoothed_data:
            min_timesteps = min(len(d['global_timestep']) for d in smoothed_data)
            
            timesteps = smoothed_data[0]['global_timestep'][:min_timesteps]
            values = np.array([d['smoothed_metric'][:min_timesteps] for d in smoothed_data])
            
            mean_values = np.mean(values, axis=0)
            std_values = np.std(values, axis=0)
            
            planner = scenario_df['planner'].iloc[0]
            
            plt.plot(timesteps, mean_values, label=planner, linewidth=2)
    
            plt.fill_between(timesteps, 
                           mean_values - std_values, 
                           mean_values + std_values, 
                           alpha=0.2)
    
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Episode Rewards', fontsize=12)
    plt.title('MASAC Learning Curves (Mean ± Std across seeds)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved learning curve: {save_path}")
    
    plt.show()


def create_comparison_plots(df: pd.DataFrame, save_dir: Path):
    """Create multiple comparison plots"""
    metrics = [
        ('cumulative_reward', 'Episode Rewards'),
        ('distance_reduced', 'Distance Reduced (m)'),
        ('time_saved', 'Time Saved (s)'),
        ('reshuffled', 'Reshuffles Performed')
    ]
    
    for metric, ylabel in metrics:
        if metric not in df.columns:
            continue
        
        save_path = save_dir / f"learning_curve_{metric}.png"
        
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # Plot by scenario
        for scenario in df['scenario'].unique():
            scenario_df = df[df['scenario'] == scenario]
            planner = scenario_df['planner'].iloc[0]
            
            # Group by timestep and calculate mean/std
            grouped = scenario_df.groupby('global_timestep')[metric].agg(['mean', 'std']).reset_index()
        
            window = 100
            grouped['mean_smooth'] = grouped['mean'].rolling(window=window, min_periods=1).mean()
            grouped['std_smooth'] = grouped['std'].rolling(window=window, min_periods=1).mean()
            plt.plot(grouped['global_timestep'], grouped['mean_smooth'], label=planner, linewidth=2)
            plt.fill_between(grouped['global_timestep'],
                           grouped['mean_smooth'] - grouped['std_smooth'],
                           grouped['mean_smooth'] + grouped['std_smooth'],
                           alpha=0.2)
        
        plt.xlabel('Timesteps', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f'{ylabel} over Time', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Saved {metric} plot: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize MASAC learning curves')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Directory containing multi-seed test results')
    parser.add_argument('--metric', type=str, default='cumulative_reward',
                        help='Metric to plot (default: cumulative_reward)')
    parser.add_argument('--window', type=int, default=100,
                        help='Smoothing window size (default: 100)')
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    
    if not log_dir.exists():
        print(f" Log directory not found: {log_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Creating Learning Curves")
    print(f"{'='*80}")
    print(f"Log directory: {log_dir}")
    print(f"Metric: {args.metric}")
    print(f"Smoothing window: {args.window}")
    print(f"{'='*80}\n")
    
    # Load timestep data
    df = load_timestep_data(log_dir)
    
    if df is None or df.empty:
        print(" No data to visualize")
        return
    plots_dir = log_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    save_path = plots_dir / f"learning_curve_{args.metric}.png"
    create_learning_curve(df, metric=args.metric, window_size=args.window, save_path=save_path)
    create_comparison_plots(df, plots_dir)
    
    print(f"\n{'='*80}")
    print(f"Visualization Complete!")
    print(f"{'='*80}")
    print(f"Plots saved to: {plots_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

