"""
Analyze Two-Agent System Results
Reads CSV and JSON files and generates summary statistics and comparisons.

Usage:
    python analyze_results.py --action_space discrete
    python analyze_results.py --action_space continuous
    python analyze_results.py --action_space both
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_results(action_space: str, base_dir: str = "two_agent_results") -> Dict[str, pd.DataFrame]:
    """
    Load all results for a given action space
    
    Returns:
        Dictionary mapping seed to episode DataFrame
    """
    results = {}
    base_path = Path(base_dir) / action_space
    
    if not base_path.exists():
        print(f"❌ No results found for action space: {action_space}")
        return results
    
    # Find all seed directories
    for seed_dir in base_path.glob("seed_*"):
        seed = int(seed_dir.name.split("_")[1])
        
        # Load episode results
        episode_file = seed_dir / "episode_results.csv"
        if episode_file.exists():
            df = pd.read_csv(episode_file)
            results[seed] = df
            print(f"✅ Loaded {len(df)} episodes for seed {seed}")
        else:
            print(f"⚠️  No episode results found for seed {seed}")
    
    return results


def analyze_by_model(results: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate results by model across all seeds
    
    Returns:
        DataFrame with model-level statistics
    """
    # Combine all seeds
    all_data = pd.concat(results.values(), ignore_index=True)
    
    # Group by model
    model_stats = all_data.groupby('model').agg({
        # Agent 1 metrics
        'agent1_reward': ['mean', 'std'],
        'success': ['mean', 'std'],
        'cubes_picked': ['mean', 'std'],
        'pick_failures': ['mean', 'std'],
        'path_efficiency': ['mean', 'std'],
        # Agent 2 metrics
        'agent2_reward': ['mean', 'std'],
        'reshuffles_performed': ['mean', 'std'],
        'total_distance_reduced': ['mean', 'std'],
        'total_time_saved': ['mean', 'std'],
        # Combined metrics
        'total_reward': ['mean', 'std'],
        'episode_length': ['mean', 'std'],
        'duration': ['mean', 'std']
    }).round(3)
    
    # Flatten column names
    model_stats.columns = ['_'.join(col).strip() for col in model_stats.columns.values]
    
    return model_stats


def print_summary(action_space: str, results: Dict[int, pd.DataFrame]):
    """Print summary statistics"""
    print(f"\n{'='*80}")
    print(f"SUMMARY: {action_space.upper()} Action Space")
    print(f"{'='*80}\n")
    
    # Overall statistics
    total_episodes = sum(len(df) for df in results.values())
    total_seeds = len(results)
    models = set()
    for df in results.values():
        models.update(df['model'].unique())
    
    print(f"Total Seeds: {total_seeds}")
    print(f"Total Episodes: {total_episodes}")
    print(f"Models Tested: {len(models)}")
    print(f"Models: {', '.join(sorted(models))}\n")
    
    # Model-level statistics
    model_stats = analyze_by_model(results)
    
    print("="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print("\nAgent 1 (Pick Sequence) Performance:")
    print("-" * 80)
    agent1_cols = [col for col in model_stats.columns if 'agent1' in col or 'success' in col or 'cubes_picked' in col]
    print(model_stats[agent1_cols].to_string())
    
    print("\n\nAgent 2 (Reshuffling) Performance:")
    print("-" * 80)
    agent2_cols = [col for col in model_stats.columns if 'agent2' in col or 'reshuffles' in col or 'distance' in col or 'time_saved' in col]
    print(model_stats[agent2_cols].to_string())
    
    print("\n\nCombined Performance:")
    print("-" * 80)
    combined_cols = [col for col in model_stats.columns if 'total_reward' in col or 'episode_length' in col or 'duration' in col]
    print(model_stats[combined_cols].to_string())
    
    # Ranking
    print("\n\n" + "="*80)
    print("MODEL RANKINGS")
    print("="*80)
    
    # Rank by total reward
    print("\nBy Total Reward (Mean):")
    print("-" * 80)
    ranking = model_stats.sort_values('total_reward_mean', ascending=False)
    for i, (model, row) in enumerate(ranking.iterrows(), 1):
        print(f"{i}. {model:20s} - {row['total_reward_mean']:8.3f} ± {row['total_reward_std']:6.3f}")
    
    # Rank by success rate
    print("\nBy Success Rate (Mean):")
    print("-" * 80)
    ranking = model_stats.sort_values('success_mean', ascending=False)
    for i, (model, row) in enumerate(ranking.iterrows(), 1):
        print(f"{i}. {model:20s} - {row['success_mean']*100:6.2f}% ± {row['success_std']*100:5.2f}%")
    
    # Rank by reshuffles
    print("\nBy Reshuffles Performed (Mean):")
    print("-" * 80)
    ranking = model_stats.sort_values('reshuffles_performed_mean', ascending=False)
    for i, (model, row) in enumerate(ranking.iterrows(), 1):
        print(f"{i}. {model:20s} - {row['reshuffles_performed_mean']:6.3f} ± {row['reshuffles_performed_std']:5.3f}")
    
    print("\n" + "="*80 + "\n")


def create_summary_table(base_dir: str = "two_agent_results", seed: int = 42):
    """
    Create a clean summary table with requested metrics
    """
    # Load both continuous and discrete results
    continuous_results = load_results("continuous", base_dir)
    discrete_results = load_results("discrete", base_dir)

    # Combine all results
    all_dfs = []
    if seed in continuous_results:
        all_dfs.append(continuous_results[seed])
    if seed in discrete_results:
        all_dfs.append(discrete_results[seed])

    if not all_dfs:
        print(f"❌ No results found for seed {seed}")
        return None

    df = pd.concat(all_dfs, ignore_index=True)

    # Calculate summary statistics
    summary = df.groupby('model').agg({
        'total_reward': 'mean',
        'success': 'mean',
        'reshuffles_performed': 'mean',
        'total_distance_reduced': 'mean',
        'total_time_saved': 'mean'
    }).round(3)

    # Rename columns
    summary.columns = [
        'Avg Total Reward',
        'Avg Success Rate',
        'Avg Reshuffles',
        'Avg Distance Reduced',
        'Avg Time Saved'
    ]

    # Convert success rate to percentage
    summary['Avg Success Rate'] = (summary['Avg Success Rate'] * 100).round(1)

    # Sort by total reward
    summary = summary.sort_values('Avg Total Reward', ascending=False)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze Two-Agent System Results")
    parser.add_argument("--action_space", type=str, default="both",
                       choices=["discrete", "continuous", "both"],
                       help="Action space to analyze (default: both)")
    parser.add_argument("--base_dir", type=str, default="two_agent_results",
                       help="Base directory for results (default: two_agent_results)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Seed to analyze (default: 42)")
    parser.add_argument("--summary_only", action="store_true",
                       help="Show only summary table with requested metrics")
    args = parser.parse_args()

    if args.summary_only:
        # Show clean summary table
        print("\n" + "="*100)
        print("TWO-AGENT SYSTEM RESULTS SUMMARY")
        print("="*100)
        print(f"Seed: {args.seed}\n")

        summary = create_summary_table(args.base_dir, args.seed)
        if summary is not None:
            print(summary.to_string())
            print("\n" + "="*100)

            # Save to CSV
            output_file = Path(args.base_dir) / f"seed_{args.seed}_summary.csv"
            summary.to_csv(output_file)
            print(f"\n✅ Summary saved to: {output_file}\n")
    else:
        # Show detailed analysis
        if args.action_space == "both":
            action_spaces = ["discrete", "continuous"]
        else:
            action_spaces = [args.action_space]

        for action_space in action_spaces:
            results = load_results(action_space, args.base_dir)
            if results:
                print_summary(action_space, results)
            else:
                print(f"⚠️  No results to analyze for {action_space}")


if __name__ == "__main__":
    main()

