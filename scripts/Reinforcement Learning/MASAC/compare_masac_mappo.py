"""
Compare MASAC and MAPPO results across all configurations
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime


def load_latest_results(log_dir: Path, algorithm: str, env_type: str, grid_size: int, num_cubes: int):
    """Load the latest test results for a configuration"""
    pattern = f"{algorithm}_{env_type}_grid{grid_size}_cubes{num_cubes}_*_summary.json"
    files = list(log_dir.glob(pattern))
    
    if not files:
        return None
    
    # Get the latest file
    latest_file = max(files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def compare_algorithms():
    """Compare MASAC and MAPPO across all configurations"""
    
    masac_log_dir = Path("cobotproject/scripts/Reinforcement Learning/MASAC/logs")
    mappo_log_dir = Path("cobotproject/scripts/Reinforcement Learning/MAPPO/logs")
    
    configurations = [
        ('rrt_isaacsim', 3, 4),
        ('rrt_isaacsim', 4, 6),
        ('rrt_isaacsim', 4, 9),
        ('rrt_viz', 3, 4),
        ('rrt_viz', 4, 6),
        ('rrt_viz', 4, 9),
        ('astar', 3, 4),
        ('astar', 4, 6),
        ('astar', 4, 9),
    ]
    
    comparison_data = []
    
    for env_type, grid_size, num_cubes in configurations:
        masac_results = load_latest_results(masac_log_dir, 'masac', env_type, grid_size, num_cubes)
        mappo_results = load_latest_results(mappo_log_dir, 'mappo', env_type, grid_size, num_cubes)
        
        row = {
            'Environment': env_type.upper(),
            'Grid': f"{grid_size}x{grid_size}",
            'Cubes': num_cubes,
        }
        
        if masac_results:
            row['MASAC_Reward'] = masac_results['avg_reward']
            row['MASAC_Reshuffles'] = masac_results['avg_reshuffles']
            row['MASAC_Distance'] = masac_results['avg_distance_reduced']
            row['MASAC_Time'] = masac_results['avg_time_saved']
            row['MASAC_Cubes'] = masac_results['avg_cubes_picked']
        else:
            row['MASAC_Reward'] = None
            row['MASAC_Reshuffles'] = None
            row['MASAC_Distance'] = None
            row['MASAC_Time'] = None
            row['MASAC_Cubes'] = None
        
        if mappo_results:
            row['MAPPO_Reward'] = mappo_results['avg_reward']
            row['MAPPO_Reshuffles'] = mappo_results['avg_reshuffles']
            row['MAPPO_Distance'] = mappo_results['avg_distance_reduced']
            row['MAPPO_Time'] = mappo_results['avg_time_saved']
            row['MAPPO_Cubes'] = mappo_results['avg_cubes_picked']
        else:
            row['MAPPO_Reward'] = None
            row['MAPPO_Reshuffles'] = None
            row['MAPPO_Distance'] = None
            row['MAPPO_Time'] = None
            row['MAPPO_Cubes'] = None
        
        # Calculate differences (MASAC - MAPPO)
        if masac_results and mappo_results:
            row['Reward_Diff'] = masac_results['avg_reward'] - mappo_results['avg_reward']
            row['Distance_Diff'] = masac_results['avg_distance_reduced'] - mappo_results['avg_distance_reduced']
            row['Time_Diff'] = masac_results['avg_time_saved'] - mappo_results['avg_time_saved']
        else:
            row['Reward_Diff'] = None
            row['Distance_Diff'] = None
            row['Time_Diff'] = None
        
        comparison_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    output_dir = Path("cobotproject/scripts/Reinforcement Learning/MASAC/logs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"masac_vs_mappo_comparison_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*100}")
    print("MASAC vs MAPPO Comparison")
    print(f"{'='*100}\n")
    print(df.to_string(index=False))
    print(f"\n{'='*100}")
    print(f"Saved to: {csv_path}")
    print(f"{'='*100}\n")
    
    # Print summary statistics
    if df['Reward_Diff'].notna().any():
        print("\nSummary Statistics (MASAC - MAPPO):")
        print(f"  Average Reward Difference: {df['Reward_Diff'].mean():.3f}")
        print(f"  Average Distance Difference: {df['Distance_Diff'].mean():.4f}m")
        print(f"  Average Time Difference: {df['Time_Diff'].mean():.3f}s")
        print(f"\n  MASAC wins: {(df['Reward_Diff'] > 0).sum()}/{len(df)} configurations")
        print(f"  MAPPO wins: {(df['Reward_Diff'] < 0).sum()}/{len(df)} configurations")
    
    return df


if __name__ == "__main__":
    compare_algorithms()

