"""
Enhanced Metrics Visualization Script
Generates visualizations from CSV results after testing completes.

Usage:
    python visualize_enhanced_metrics.py --experiment exp1
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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
    
    return summary_df, episode_df, results_dir


def plot_collision_comparison(summary_df, output_dir):
    """Plot collision count comparison across models"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = summary_df['model']
    collisions = summary_df['avg_collisions']
    
    bars = ax.bar(range(len(models)), collisions, color='coral', alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Average Collisions per Episode')
    ax.set_title('Collision Count Comparison Across Models')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'collision_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: collision_comparison.png")


def plot_path_efficiency_comparison(summary_df, output_dir):
    """Plot path efficiency comparison (lower is better)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = summary_df['model']
    efficiency = summary_df['avg_path_efficiency']
    
    # Color code: green for efficient (close to 1.0), red for inefficient
    colors = ['green' if e < 1.5 else 'orange' if e < 2.0 else 'red' for e in efficiency]
    
    bars = ax.bar(range(len(models)), efficiency, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=1.0, color='blue', linestyle='--', linewidth=2, label='Optimal (1.0)')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Path Efficiency (Steps / Optimal Steps)')
    ax.set_title('Path Efficiency Comparison (Lower is Better)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'path_efficiency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: path_efficiency_comparison.png")


def plot_action_diversity_comparison(summary_df, output_dir):
    """Plot action diversity metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    models = summary_df['model']
    entropy = summary_df['avg_action_entropy']
    unique_actions = summary_df['avg_unique_actions']
    
    # Entropy plot
    bars1 = ax1.bar(range(len(models)), entropy, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylabel('Action Entropy (bits)')
    ax1.set_title('Action Entropy (Higher = More Diverse)')
    ax1.grid(axis='y', alpha=0.3)
    
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=9)
    
    # Unique actions plot
    bars2 = ax2.bar(range(len(models)), unique_actions, color='lightgreen', alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_ylabel('Average Unique Actions per Episode')
    ax2.set_title('Unique Actions Used')
    ax2.grid(axis='y', alpha=0.3)
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'action_diversity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: action_diversity_comparison.png")


def plot_time_to_completion(summary_df, output_dir):
    """Plot average episode duration"""
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
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'time_to_completion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: time_to_completion.png")


def plot_multi_metric_radar(summary_df, output_dir):
    """Create radar chart comparing all metrics"""
    from math import pi

    # Normalize metrics to 0-1 scale for comparison
    metrics = ['success_rate', 'avg_path_efficiency', 'avg_action_entropy', 'avg_collisions']
    metric_labels = ['Success Rate', 'Path Efficiency\n(inverted)', 'Action Entropy', 'Collisions\n(inverted)']

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    angles = [n / len(metrics) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]

    for idx, row in summary_df.iterrows():
        values = [
            row['success_rate'],
            1.0 / (row['avg_path_efficiency'] + 0.1),  # Invert (lower is better)
            row['avg_action_entropy'] / 5.0,  # Normalize to ~0-1
            1.0 / (row['avg_collisions'] + 1.0)  # Invert (lower is better)
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
    plt.savefig(output_dir / 'multi_metric_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: multi_metric_radar.png")


def plot_episode_metrics_over_time(episode_df, output_dir):
    """Plot how metrics evolve over episodes"""
    if episode_df is None:
        print("⚠ Episode data not available, skipping time series plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Collisions over time
    for model in episode_df['model'].unique():
        model_data = episode_df[episode_df['model'] == model]
        axes[0, 0].plot(model_data['episode'], model_data['collisions'],
                       marker='o', alpha=0.6, label=model, markersize=3)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Collisions')
    axes[0, 0].set_title('Collisions Over Episodes')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: Path efficiency over time
    for model in episode_df['model'].unique():
        model_data = episode_df[episode_df['model'] == model]
        # Filter out infinite values
        valid_data = model_data[model_data['path_efficiency'] < 10]
        axes[0, 1].plot(valid_data['episode'], valid_data['path_efficiency'],
                       marker='o', alpha=0.6, label=model, markersize=3)
    axes[0, 1].axhline(y=1.0, color='red', linestyle='--', label='Optimal')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Path Efficiency')
    axes[0, 1].set_title('Path Efficiency Over Episodes')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    # Plot 3: Action entropy over time
    for model in episode_df['model'].unique():
        model_data = episode_df[episode_df['model'] == model]
        axes[1, 0].plot(model_data['episode'], model_data['action_entropy'],
                       marker='o', alpha=0.6, label=model, markersize=3)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Action Entropy')
    axes[1, 0].set_title('Action Entropy Over Episodes')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    # Plot 4: Duration over time
    for model in episode_df['model'].unique():
        model_data = episode_df[episode_df['model'] == model]
        axes[1, 1].plot(model_data['episode'], model_data['duration'],
                       marker='o', alpha=0.6, label=model, markersize=3)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Duration (s)')
    axes[1, 1].set_title('Episode Duration Over Time')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: metrics_over_time.png")


def plot_correlation_heatmap(summary_df, output_dir):
    """Plot correlation heatmap of all metrics"""
    metrics_cols = ['avg_reward', 'success_rate', 'avg_length', 'avg_collisions',
                    'avg_path_efficiency', 'avg_action_entropy', 'avg_unique_actions', 'avg_duration']

    corr_data = summary_df[metrics_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Heatmap of Performance Metrics', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: correlation_heatmap.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize enhanced metrics from experiment results")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name (exp1, exp2)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"ENHANCED METRICS VISUALIZATION - {args.experiment.upper()}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    summary_df, episode_df, results_dir = load_data(args.experiment)
    print(f"✓ Loaded {len(summary_df)} models")
    if episode_df is not None:
        print(f"✓ Loaded {len(episode_df)} episode records")

    # Create visualizations directory
    viz_dir = results_dir / "enhanced_visualizations"
    viz_dir.mkdir(exist_ok=True)

    print(f"\nGenerating visualizations...")

    # Generate all plots
    plot_collision_comparison(summary_df, viz_dir)
    plot_path_efficiency_comparison(summary_df, viz_dir)
    plot_action_diversity_comparison(summary_df, viz_dir)
    plot_time_to_completion(summary_df, viz_dir)
    plot_multi_metric_radar(summary_df, viz_dir)
    plot_correlation_heatmap(summary_df, viz_dir)

    if episode_df is not None:
        plot_episode_metrics_over_time(episode_df, viz_dir)

    print(f"\n{'='*60}")
    print(f"✅ ALL VISUALIZATIONS COMPLETE!")
    print(f"{'='*60}")
    print(f"Output directory: {viz_dir}")
    print(f"Total plots generated: 7")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()


