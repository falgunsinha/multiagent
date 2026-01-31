"""
Unified Visualization Script - All Metrics (No W&B Required)
Generates all visualizations from CSV results using existing visualization modules.

This script:
1. Reads CSV files (comparison_results.csv, episode_results.csv)
2. Uses existing visualization modules from visualization/ folder
3. Generates new metric visualizations (collisions, path efficiency, etc.)
4. Saves all plots to results/{experiment}/all_visualizations/

Usage:
    python visualize_all_metrics.py --experiment exp1
    python visualize_all_metrics.py --experiment exp2
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_data(experiment_name):
    """Load CSV data from experiment results"""
    results_dir = Path(r"C:\isaacsim\cobotproject\scripts\experiments\rlmodels\results") / experiment_name
    
    summary_file = results_dir / "comparison_results.csv"
    episode_file = results_dir / "episode_results.csv"
    
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary results not found: {summary_file}")
    
    summary_df = pd.read_csv(summary_file)
    
    episode_df = None
    if episode_file.exists():
        episode_df = pd.read_csv(episode_file)
        print(f"✓ Loaded {len(episode_df)} episode records")
    else:
        print("⚠ Episode-level data not found (will skip time-series plots)")
    
    return summary_df, episode_df, results_dir


def prepare_data_for_existing_visualizations(summary_df, episode_df):
    """
    Convert CSV data to format expected by existing visualization modules.
    
    Existing modules expect data in specific formats (from W&B).
    This function transforms CSV data to match those expectations.
    """
    # For cross-model comparisons (summary level)
    cross_df = summary_df.copy()
    cross_df = cross_df.rename(columns={
        'avg_reward': 'avg_reward',
        'avg_length': 'avg_length',
        'success_rate': 'success_rate',
        'model': 'model'
    })
    
    # For episode-level data (if available)
    if episode_df is not None:
        # Reshape episode data for time-series plots
        episode_long = episode_df.copy()
        episode_long = episode_long.rename(columns={
            'reward': 'reward',
            'length': 'length',
            'success': 'success',
            'episode': 'episode',
            'model': 'model'
        })
    else:
        episode_long = None
    
    return cross_df, episode_long


def generate_existing_visualizations(cross_df, episode_df, output_dir):
    """
    Generate visualizations using existing modules from visualization/ folder.
    
    These are the visualizations that were previously logged to W&B.
    Now we generate them from CSV and save to disk.
    """
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS FROM EXISTING MODULES")
    print("="*60)
    
    # Add visualization module to path
    viz_path = Path(__file__).parent / "visualization"
    if str(viz_path) not in sys.path:
        sys.path.insert(0, str(viz_path))
    
    existing_viz_dir = output_dir / "existing_visualizations"
    existing_viz_dir.mkdir(exist_ok=True)
    
    plot_count = 0
    
    # Try to import and use existing visualization modules
    try:
        # Individual model visualizations (Seaborn)
        try:
            from individual_models.seaborn.success_rate import plot_success_rate_seaborn
            from individual_models.seaborn.reward_distribution import plot_reward_distribution_seaborn
            from individual_models.seaborn.steps_distribution import plot_steps_distribution_seaborn
            
            for model in cross_df['model'].unique():
                model_data = cross_df[cross_df['model'] == model]
                
                # Create DataFrame in expected format
                viz_df = pd.DataFrame({
                    'episode': range(1, len(model_data) + 1),
                    'reward': [model_data['avg_reward'].values[0]] * len(model_data),
                    'success': [model_data['success_rate'].values[0]] * len(model_data),
                    'length': [model_data['avg_length'].values[0]] * len(model_data),
                })
                
                try:
                    fig = plot_success_rate_seaborn(viz_df, model)
                    fig.savefig(existing_viz_dir / f"{model.replace(' ', '_')}_success_rate.png", dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    plot_count += 1
                except Exception as e:
                    print(f"  ⚠ Skipped success_rate for {model}: {e}")
                
                try:
                    fig = plot_reward_distribution_seaborn(viz_df, model)
                    fig.savefig(existing_viz_dir / f"{model.replace(' ', '_')}_reward_dist.png", dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    plot_count += 1
                except Exception as e:
                    print(f"  ⚠ Skipped reward_dist for {model}: {e}")
            
            print(f"✓ Generated {plot_count} individual model plots")
        
        except ImportError as e:
            print(f"⚠ Individual model visualizations not available: {e}")
    
    except Exception as e:
        print(f"⚠ Could not load existing visualization modules: {e}")
        print("  Continuing with new metric visualizations only...")
    
    # Cross-model comparisons (simple matplotlib/seaborn)
    try:
        # Success rate comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        models = cross_df['model']
        success_rates = cross_df['success_rate']
        bars = ax.bar(range(len(models)), success_rates, color='skyblue', alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate Comparison Across Models')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1%}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(existing_viz_dir / 'cross_model_success_rate.png', dpi=300, bbox_inches='tight')
        plt.close()
        plot_count += 1
        print("✓ Generated cross-model success rate plot")
    except Exception as e:
        print(f"⚠ Skipped cross-model success rate: {e}")
    
    return plot_count


def generate_new_metric_visualizations(summary_df, episode_df, output_dir):
    """Generate visualizations for NEW metrics (collisions, path efficiency, etc.)"""
    print("\n" + "="*60)
    print("GENERATING NEW METRIC VISUALIZATIONS")
    print("="*60)

    new_viz_dir = output_dir / "new_metrics"
    new_viz_dir.mkdir(exist_ok=True)

    plot_count = 0

    # 1. Collision Comparison
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        models = summary_df['model']
        collisions = summary_df['avg_collisions']
        bars = ax.bar(range(len(models)), collisions, color='coral', alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Average Collisions per Episode')
        ax.set_title('Collision Count Comparison Across Models')
        ax.grid(axis='y', alpha=0.3)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(new_viz_dir / 'collision_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        plot_count += 1
        print("✓ collision_comparison.png")
    except Exception as e:
        print(f"⚠ Skipped collision comparison: {e}")

    # 2. Path Efficiency Comparison
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        models = summary_df['model']
        efficiency = summary_df['avg_path_efficiency']
        colors = ['green' if e < 1.5 else 'orange' if e < 2.0 else 'red' for e in efficiency]
        bars = ax.bar(range(len(models)), efficiency, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=1.0, color='blue', linestyle='--', linewidth=2, label='Optimal (1.0)')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Path Efficiency (Steps / Optimal Steps)')
        ax.set_title('Path Efficiency Comparison (Lower is Better)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(new_viz_dir / 'path_efficiency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        plot_count += 1
        print("✓ path_efficiency_comparison.png")
    except Exception as e:
        print(f"⚠ Skipped path efficiency: {e}")

    # 3. Action Diversity (Entropy + Unique Actions)
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        models = summary_df['model']
        entropy = summary_df['avg_action_entropy']
        unique_actions = summary_df['avg_unique_actions']

        bars1 = ax1.bar(range(len(models)), entropy, color='skyblue', alpha=0.7, edgecolor='black')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.set_ylabel('Action Entropy (bits)')
        ax1.set_title('Action Entropy (Higher = More Diverse)')
        ax1.grid(axis='y', alpha=0.3)
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom', fontsize=9)

        bars2 = ax2.bar(range(len(models)), unique_actions, color='lightgreen', alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.set_ylabel('Average Unique Actions per Episode')
        ax2.set_title('Unique Actions Used')
        ax2.grid(axis='y', alpha=0.3)
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(new_viz_dir / 'action_diversity_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        plot_count += 1
        print("✓ action_diversity_comparison.png")
    except Exception as e:
        print(f"⚠ Skipped action diversity: {e}")

    # 4. Time to Completion
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        models = summary_df['model']
        duration = summary_df['avg_duration']
        bars = ax.bar(range(len(models)), duration, color='mediumpurple', alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Average Duration (seconds)')
        ax.set_title('Time to Completion per Episode')
        ax.grid(axis='y', alpha=0.3)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}s', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(new_viz_dir / 'time_to_completion.png', dpi=300, bbox_inches='tight')
        plt.close()
        plot_count += 1
        print("✓ time_to_completion.png")
    except Exception as e:
        print(f"⚠ Skipped time to completion: {e}")

    # 5. Multi-Metric Radar Chart
    try:
        from math import pi
        metrics = ['success_rate', 'avg_path_efficiency', 'avg_action_entropy', 'avg_collisions']
        metric_labels = ['Success Rate', 'Path Efficiency\n(inverted)', 'Action Entropy', 'Collisions\n(inverted)']

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        angles = [n / len(metrics) * 2 * pi for n in range(len(metrics))]
        angles += angles[:1]

        for idx, row in summary_df.iterrows():
            values = [
                row['success_rate'],
                1.0 / (row['avg_path_efficiency'] + 0.1),
                row['avg_action_entropy'] / 5.0,
                1.0 / (row['avg_collisions'] + 1.0)
            ]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=row['model'])
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, size=10)
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Metric Performance Radar\n(Higher is Better for All)', size=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(new_viz_dir / 'multi_metric_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
        plot_count += 1
        print("✓ multi_metric_radar.png")
    except Exception as e:
        print(f"⚠ Skipped radar chart: {e}")

    return plot_count


def main():
    parser = argparse.ArgumentParser(description="Generate all visualizations from CSV results")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name (exp1, exp2)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"UNIFIED VISUALIZATION - {args.experiment.upper()}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    summary_df, episode_df, results_dir = load_data(args.experiment)
    print(f"✓ Loaded {len(summary_df)} models from CSV")

    # Create output directory
    output_dir = results_dir / "all_visualizations"
    output_dir.mkdir(exist_ok=True)

    # Prepare data for existing visualizations
    cross_df, episode_long = prepare_data_for_existing_visualizations(summary_df, episode_df)

    # Generate visualizations
    total_plots = 0

    # Try to use existing visualization modules
    existing_count = generate_existing_visualizations(cross_df, episode_long, output_dir)
    total_plots += existing_count

    # Generate new metric visualizations
    new_count = generate_new_metric_visualizations(summary_df, episode_df, output_dir)
    total_plots += new_count

    print(f"\n{'='*60}")
    print(f"✅ VISUALIZATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Total plots generated: {total_plots}")
    print(f"  - Existing visualizations: {existing_count}")
    print(f"  - New metric visualizations: {new_count}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()


