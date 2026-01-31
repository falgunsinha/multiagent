"""
Aggregate multi-seed MASAC test results into MAPPO-style summary tables

This script reads multiple test runs (with different seeds) and aggregates them
into summary statistics suitable for MAPPO-style comparison tables.

Usage:
    python aggregate_multi_seed_results.py --log_dir logs/multi_seed_20250116_120000
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def load_summary_files(log_dir: Path):
    """Load all summary JSON files from a log directory"""
    summary_files = list(log_dir.glob("*_summary.json"))
    
    if not summary_files:
        print(f"⚠️  No summary files found in {log_dir}")
        return []
    
    summaries = []
    for file_path in summary_files:
        with open(file_path, 'r') as f:
            summary = json.load(f)
            summaries.append(summary)
    
    print(f"✅ Loaded {len(summaries)} summary files")
    return summaries


def aggregate_by_scenario(summaries: list):
    """Aggregate summaries by scenario (algorithm + planner + grid + cubes)"""
    # Group by scenario
    scenario_groups = {}
    for summary in summaries:
        scenario = summary.get('scenario', 'unknown')
        if scenario not in scenario_groups:
            scenario_groups[scenario] = []
        scenario_groups[scenario].append(summary)
    
    # Aggregate each scenario
    aggregated = []
    for scenario, group in scenario_groups.items():
        if not group:
            continue
        
        # Extract metrics across all seeds
        rewards = [s['reward']['mean'] for s in group]
        success_rates = [s['success_rate']['percentage'] for s in group]
        reshuffles = [s['reshuffles']['mean'] for s in group]
        distances = [s['distance_reduced_m']['mean'] for s in group]
        times = [s['time_saved_s']['mean'] for s in group]
        
        # Calculate aggregated statistics (mean and std across seeds)
        agg = {
            'scenario': scenario,
            'algorithm': group[0].get('algorithm', 'MASAC'),
            'planner': group[0].get('planner', 'unknown'),
            'grid_size': group[0].get('grid_size', 0),
            'num_cubes': group[0].get('num_cubes', 0),
            'num_seeds': len(group),
            
            # Reward (mean ± std across seeds)
            'reward_mean': float(np.mean(rewards)),
            'reward_std': float(np.std(rewards)),
            'reward_median': float(np.median(rewards)),
            
            # Success rate (mean ± std across seeds)
            'success_rate_mean': float(np.mean(success_rates)),
            'success_rate_std': float(np.std(success_rates)),
            
            # Reshuffles (mean ± std across seeds)
            'reshuffles_mean': float(np.mean(reshuffles)),
            'reshuffles_std': float(np.std(reshuffles)),
            
            # Distance reduced (mean ± std across seeds)
            'distance_reduced_mean': float(np.mean(distances)),
            'distance_reduced_std': float(np.std(distances)),
            
            # Time saved (mean ± std across seeds)
            'time_saved_mean': float(np.mean(times)),
            'time_saved_std': float(np.std(times)),
        }
        
        aggregated.append(agg)
    
    return aggregated


def create_mappo_style_table(aggregated: list):
    """Create MAPPO-style table with Mean(Std) format"""
    print(f"\n{'='*80}")
    print("MAPPO-Style Comparison Table")
    print(f"{'='*80}\n")
    
    # Create DataFrame
    df = pd.DataFrame(aggregated)
    
    # Format as Mean(Std)
    df['Reward'] = df.apply(lambda row: f"{row['reward_mean']:.1f}({row['reward_std']:.1f})", axis=1)
    df['Success Rate'] = df.apply(lambda row: f"{row['success_rate_mean']:.1f}({row['success_rate_std']:.1f})", axis=1)
    df['Reshuffles'] = df.apply(lambda row: f"{row['reshuffles_mean']:.2f}({row['reshuffles_std']:.2f})", axis=1)
    df['Distance (m)'] = df.apply(lambda row: f"{row['distance_reduced_mean']:.2f}({row['distance_reduced_std']:.2f})", axis=1)
    df['Time (s)'] = df.apply(lambda row: f"{row['time_saved_mean']:.2f}({row['time_saved_std']:.2f})", axis=1)
    
    # Select columns for display
    display_df = df[['scenario', 'planner', 'num_seeds', 'Reward', 'Success Rate', 'Reshuffles', 'Distance (m)', 'Time (s)']]
    
    print(display_df.to_string(index=False))
    print(f"\n{'='*80}\n")
    
    return display_df


def save_aggregated_results(aggregated: list, log_dir: Path):
    """Save aggregated results to CSV and JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV
    csv_path = log_dir / f"aggregated_multi_seed_{timestamp}.csv"
    df = pd.DataFrame(aggregated)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved aggregated CSV: {csv_path}")
    
    # Save as JSON
    json_path = log_dir / f"aggregated_multi_seed_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"✅ Saved aggregated JSON: {json_path}")
    
    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description='Aggregate multi-seed MASAC test results')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Directory containing multi-seed test results')
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    
    if not log_dir.exists():
        print(f"❌ Log directory not found: {log_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Aggregating Multi-Seed Results")
    print(f"{'='*80}")
    print(f"Log directory: {log_dir}")
    print(f"{'='*80}\n")
    
    # Load summary files
    summaries = load_summary_files(log_dir)
    
    if not summaries:
        print("❌ No summaries to aggregate")
        return
    
    # Aggregate by scenario
    aggregated = aggregate_by_scenario(summaries)
    
    # Create MAPPO-style table
    table_df = create_mappo_style_table(aggregated)
    
    # Save results
    csv_path, json_path = save_aggregated_results(aggregated, log_dir)
    
    print(f"\n{'='*80}")
    print(f"Aggregation Complete!")
    print(f"{'='*80}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

