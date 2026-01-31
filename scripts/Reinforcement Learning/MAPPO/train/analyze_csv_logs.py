"""
Analyze MAPPO training logs from CSV files (no WandB required)
Extracts distance_reduced and time_saved metrics from episode logs.

Usage:
    py -3.11 analyze_csv_logs.py
"""

import pandas as pd
import glob
from pathlib import Path
import numpy as np

def analyze_csv_logs():
    """Analyze all CSV log files in the logs directory"""
    
    # Find all episode log CSV files
    log_dir = Path(r'..\logs')
    csv_files = list(log_dir.glob('*episode_log.csv'))
    
    if not csv_files:
        print("No CSV log files found in ../logs/")
        return
    
    print(f"Found {len(csv_files)} CSV log files\n")
    
    results = []
    
    for csv_file in csv_files:
        # Parse filename to extract config info
        filename = csv_file.stem
        
        # Determine environment type
        if 'astar' in filename.lower():
            env_type = 'A*'
        elif 'isaacsim' in filename.lower():
            env_type = 'Isaac Sim'
        elif 'rrt' in filename.lower():
            env_type = 'RRT Viz'
        else:
            env_type = 'Unknown'
        
        # Extract grid size and num cubes
        parts = filename.split('_')
        grid_size = '?'
        num_cubes = '?'
        
        for i, part in enumerate(parts):
            if 'grid' in part and i+1 < len(parts):
                grid_size = part.replace('grid', '')
            if 'cubes' in part and i+1 < len(parts):
                num_cubes = part.replace('cubes', '')
        
        config = f'{grid_size}x{grid_size}, {num_cubes} cubes'
        
        # Read CSV file
        try:
            df = pd.read_csv(csv_file)
            
            # Check if new columns exist
            has_distance = 'total_distance_reduced' in df.columns
            has_time = 'total_time_saved' in df.columns
            
            if has_distance and has_time:
                avg_distance = df['total_distance_reduced'].mean()
                avg_time = df['total_time_saved'].mean()
            else:
                avg_distance = 0.0
                avg_time = 0.0
                print(f"⚠️  {csv_file.name}: Missing distance/time columns (old format)")
            
            results.append({
                'config': config,
                'env_type': env_type,
                'episodes': len(df),
                'avg_reward': df['total_reward'].mean(),
                'avg_reshuffles': df['reshuffles_performed'].mean(),
                'avg_episode_length': df['episode_length'].mean(),
                'avg_distance_reduced': avg_distance,
                'avg_time_saved': avg_time,
                'filename': csv_file.name
            })
            
        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")
    
    # Sort by config and env_type
    results.sort(key=lambda x: (x['config'], x['env_type']))
    
    # Print results
    print('=' * 150)
    header = f"{'Config':<15} | {'Environment':<10} | {'Episodes':>8} | {'Avg Reward':>10} | {'Reshuffles':>10} | {'Dist Reduced':>12} | {'Time Saved':>10}"
    print(header)
    print('=' * 150)
    
    for r in results:
        row = (f"{r['config']:<15} | {r['env_type']:<10} | {r['episodes']:8} | "
               f"{r['avg_reward']:10.2f} | {r['avg_reshuffles']:10.2f} | "
               f"{r['avg_distance_reduced']:12.3f} | {r['avg_time_saved']:10.3f}")
        print(row)
    
    print('=' * 150)
    print(f"\nTotal runs analyzed: {len(results)}")
    
    # Summary statistics
    if results:
        print("\n📊 Summary Statistics:")
        print(f"  Best distance reduction: {max(r['avg_distance_reduced'] for r in results):.3f}m")
        print(f"  Best time saved: {max(r['avg_time_saved'] for r in results):.3f}s")
        print(f"  Worst distance reduction: {min(r['avg_distance_reduced'] for r in results):.3f}m")
        print(f"  Worst time saved: {min(r['avg_time_saved'] for r in results):.3f}s")
        
        # Group by environment
        print("\n📈 By Environment:")
        for env in ['A*', 'RRT Viz', 'Isaac Sim']:
            env_results = [r for r in results if r['env_type'] == env]
            if env_results:
                avg_dist = np.mean([r['avg_distance_reduced'] for r in env_results])
                avg_time = np.mean([r['avg_time_saved'] for r in env_results])
                print(f"  {env:10}: Avg distance = {avg_dist:+.3f}m, Avg time = {avg_time:+.3f}s")

if __name__ == '__main__':
    analyze_csv_logs()

