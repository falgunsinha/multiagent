"""
Analyze Waypoint Selection Experiment Results

Generates Table 4 from LLM-A* paper:
- Memory Score (↑): Normalized memory efficiency
- Time Score (↑): Normalized time efficiency  
- Path Length (%, ↑): Path length as percentage of optimal

Usage:
    python analyze_waypoint_results.py --input results/waypoint_selection_results_TIMESTAMP.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json


def calculate_memory_score(memory_values, baseline_memory):
    """
    Calculate memory score (higher is better)
    
    Score = baseline_memory / current_memory
    Normalized to [0, 1] range
    """
    if baseline_memory == 0 or memory_values == 0:
        return 0.0
    
    score = baseline_memory / memory_values
    return min(score, 1.0)  # Cap at 1.0


def calculate_time_score(time_values, baseline_time):
    """
    Calculate time score (higher is better)
    
    Score = baseline_time / current_time
    Normalized to [0, 1] range
    """
    if baseline_time == 0 or time_values == 0:
        return 0.0
    
    score = baseline_time / time_values
    return min(score, 1.0)  # Cap at 1.0


def calculate_path_length_percentage(path_length, optimal_path_length):
    """
    Calculate path length as percentage of optimal
    
    Returns: percentage (e.g., 105 means 5% longer than optimal)
    """
    if optimal_path_length == 0:
        return 0
    
    return int((path_length / optimal_path_length) * 100)


def analyze_results(csv_file):
    """Analyze waypoint selection experiment results"""
    
    print("\n" + "="*80)
    print("WAYPOINT SELECTION RESULTS ANALYSIS")
    print("="*80 + "\n")
    
    # Load results
    df = pd.read_csv(csv_file)
    
    # Filter successful trials only
    df_success = df[df['success'] == True].copy()
    
    if len(df_success) == 0:
        print("ERROR: No successful trials found!")
        return
    
    print(f"Loaded {len(df)} total trials ({len(df_success)} successful)")
    
    # Get unique values
    planners = df_success['planner'].unique()
    methods = ['start', 'uniform', 'random', 'goal']
    waypoint_counts = sorted(df_success['num_waypoints'].unique())
    
    print(f"Planners: {planners}")
    print(f"Methods: {methods}")
    print(f"Waypoint counts: {waypoint_counts}")
    
    # Analyze each planner
    for planner in planners:
        print(f"\n{'='*80}")
        print(f"PLANNER: {planner.upper()}")
        print(f"{'='*80}\n")
        
        df_planner = df_success[df_success['planner'] == planner]
        
        # Find baseline (best performance) for normalization
        # Use the best method with 2 waypoints as baseline (as per paper)
        baseline_data = df_planner[df_planner['num_waypoints'] == 2]
        
        if len(baseline_data) == 0:
            print("WARNING: No baseline data (2 waypoints) found!")
            continue
        
        # Calculate baseline metrics (minimum values = best performance)
        baseline_time = baseline_data.groupby('method')['planning_time'].mean().min()
        baseline_memory = baseline_data.groupby('method')['nodes_explored'].mean().min()
        baseline_path = baseline_data.groupby('method')['path_length'].mean().min()
        
        print(f"Baseline metrics (best with 2 waypoints):")
        print(f"  Time: {baseline_time:.4f}s")
        print(f"  Memory: {baseline_memory:.1f} nodes")
        print(f"  Path: {baseline_path:.2f}m")
        print()
        
        # Create results table
        results_table = {
            'memory_score': {},
            'time_score': {},
            'path_length_pct': {}
        }
        
        for method in methods:
            results_table['memory_score'][method] = {}
            results_table['time_score'][method] = {}
            results_table['path_length_pct'][method] = {}
            
            for num_wp in waypoint_counts:
                # Filter data
                data = df_planner[
                    (df_planner['method'] == method) & 
                    (df_planner['num_waypoints'] == num_wp)
                ]
                
                if len(data) > 0:
                    avg_time = data['planning_time'].mean()
                    avg_memory = data['nodes_explored'].mean()
                    avg_path = data['path_length'].mean()
                    
                    # Calculate scores
                    memory_score = calculate_memory_score(avg_memory, baseline_memory)
                    time_score = calculate_time_score(avg_time, baseline_time)
                    path_pct = calculate_path_length_percentage(avg_path, baseline_path)
                    
                    results_table['memory_score'][method][num_wp] = memory_score
                    results_table['time_score'][method][num_wp] = time_score
                    results_table['path_length_pct'][method][num_wp] = path_pct
                else:
                    results_table['memory_score'][method][num_wp] = 0
                    results_table['time_score'][method][num_wp] = 0
                    results_table['path_length_pct'][method][num_wp] = 0
        
        # Print tables (like LLM-A* paper Table 4)
        print_table_4_format(results_table, methods, waypoint_counts)


def print_table_4_format(results, methods, waypoint_counts):
    """Print results in Table 4 format from LLM-A* paper"""
    
    metrics_display = [
        ('Memory Score (↑)', 'memory_score', '.3f'),
        ('Time Score (↑)', 'time_score', '.3f'),
        ('Path Length (%, ↑)', 'path_length_pct', 'd')
    ]
    
    for metric_name, metric_key, fmt in metrics_display:
        print(f"\n{metric_name}")
        print("-" * 80)
        
        # Header
        print(f"{'Method':<12}", end="")
        for num_wp in waypoint_counts:
            print(f"{num_wp:>12}", end="")
        print()
        print("-" * 80)
        
        # Data rows
        for method in methods:
            print(f"{method.capitalize():<12}", end="")
            
            for num_wp in waypoint_counts:
                value = results[metric_key][method].get(num_wp, 0)
                
                if value > 0:
                    # Highlight best values in each column
                    print(f"{value:>12{fmt}}", end="")
                else:
                    print(f"{'N/A':>12}", end="")
            
            print()
        
        print()


def main():
    parser = argparse.ArgumentParser(description="Analyze waypoint selection results")
    parser.add_argument("--input", type=str, required=True,
                       help="Input CSV file with experiment results")
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"ERROR: File not found: {args.input}")
        return
    
    analyze_results(args.input)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

