"""
Cross-Model Comparison - Seaborn
Compare all models across key metrics using seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set Linux Libertine font globally for all plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Linux Libertine', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Linux Libertine'


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

def load_all_results():
    """Load results from both experiments"""
    base_path = Path(__file__).parent.parent.parent / "results"
    
    results = {}
    for exp in ['exp1', 'exp2']:
        csv_path = base_path / exp / "comparison_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['experiment'] = exp
            results[exp] = df
            print(f"[INFO] Loaded {len(df)} results from {exp}")
    
    if not results:
        print("[ERROR] No results found!")
        return None
    
    # Combine all results
    combined = pd.concat(results.values(), ignore_index=True)
    return combined

def plot_success_rate_comparison(df, output_dir):
    """Compare success rates across all models"""
    plt.figure(figsize=(16, 8))
    
    # Calculate success rate per model
    success_rates = df.groupby(['model', 'experiment'])['success_rate'].mean().reset_index()
    
    # Create grouped bar plot
    sns.barplot(data=success_rates, x='model', y='success_rate', hue='experiment', palette='Set2')
    
    plt.title('Success Rate Comparison - All Models', fontsize=18, fontweight='bold')
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Success Rate (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.legend(title='Experiment', labels=['Discrete Models', 'Continuous Models'])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'cross_model_success_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] cross_model_success_rates.png")

def plot_reward_comparison(df, output_dir):
    """Compare reward distributions across all models"""
    plt.figure(figsize=(16, 8))
    
    # Create violin plot
    sns.violinplot(data=df, x='model', y='reward', hue='experiment', split=True, palette='muted')
    
    plt.title('Reward Distribution - All Models', fontsize=18, fontweight='bold')
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Experiment', labels=['Discrete Models', 'Continuous Models'])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'cross_model_reward_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] cross_model_reward_distribution.png")

def plot_performance_heatmap(df, output_dir):
    """Create heatmap of model performance across metrics"""
    plt.figure(figsize=(12, 10))
    
    # Calculate mean metrics per model
    metrics = df.groupby('model').agg({
        'success_rate': 'mean',
        'reward': 'mean',
        'steps': 'mean',
        'time': 'mean'
    }).round(2)
    
    # Normalize for heatmap (0-1 scale)
    metrics_norm = (metrics - metrics.min()) / (metrics.max() - metrics.min())
    
    # Create heatmap
    sns.heatmap(metrics_norm.T, annot=metrics.T, fmt='.2f', cmap='RdYlGn', 
                cbar_kws={'label': 'Normalized Score'}, linewidths=0.5)
    
    plt.title('Model Performance Heatmap', fontsize=18, fontweight='bold')
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Metric', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'cross_model_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] cross_model_performance_heatmap.png")

def plot_steps_vs_reward(df, output_dir):
    """Scatter plot of steps vs reward"""
    plt.figure(figsize=(14, 8))
    
    # Create scatter plot
    sns.scatterplot(data=df, x='steps', y='reward', hue='model', style='experiment', 
                    s=100, alpha=0.6, palette='tab10')
    
    plt.title('Steps vs Reward - All Models', fontsize=18, fontweight='bold')
    plt.xlabel('Steps per Episode', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'cross_model_steps_vs_reward.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] cross_model_steps_vs_reward.png")

def generate_comparison_table(df, output_dir):
    """Generate comprehensive comparison table"""
    summary = df.groupby(['model', 'experiment']).agg({
        'reward': ['mean', 'std'],
        'steps': ['mean', 'std'],
        'success_rate': 'mean',
        'time': 'mean'
    }).round(2)
    
    summary.to_csv(output_dir / 'cross_model_comparison_table.csv')
    print(f"[SAVED] cross_model_comparison_table.csv")
    
    print("\n" + "="*100)
    print("CROSS-MODEL COMPARISON TABLE")
    print("="*100)
    print(summary)
    print("="*100 + "\n")

def plot_parallel_coordinates_seaborn(df):
    """Create parallel coordinates plot - returns figure for W&B"""
    from pandas.plotting import parallel_coordinates

    # Calculate average metrics per model
    metrics = df.groupby(['model']).agg({
        'success_rate': 'mean',
        'avg_reward': 'mean',
        'avg_length': 'mean'
    }).reset_index()

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot parallel coordinates
    parallel_coordinates(
        metrics,
        'model',
        cols=['success_rate', 'avg_reward', 'avg_length'],
        colormap='tab10',
        alpha=0.7,
        linewidth=2,
        ax=ax
    )

    ax.set_title('Cross-Model Performance - Parallel Coordinates', fontsize=16, fontweight='bold')
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig

def plot_parallel_coordinates(df, output_dir):
    """Create parallel coordinates plot using pandas/matplotlib"""
    from pandas.plotting import parallel_coordinates

    # Calculate average metrics per model
    metrics = df.groupby(['model', 'experiment']).agg({
        'success_rate': 'mean',
        'reward': 'mean',
        'steps': 'mean',
        'time': 'mean'
    }).reset_index()

    # Create figure
    plt.figure(figsize=(16, 8))

    # Plot parallel coordinates
    parallel_coordinates(
        metrics,
        'model',
        cols=['success_rate', 'reward', 'steps', 'time'],
        colormap='tab10',
        alpha=0.7,
        linewidth=2
    )

    plt.title('Cross-Model Performance - Parallel Coordinates', fontsize=18, fontweight='bold')
    plt.xlabel('Metrics', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'cross_model_parallel_coordinates.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] cross_model_parallel_coordinates.png")

def main():
    """Main function"""
    print("\n" + "="*80)
    print("CROSS-MODEL COMPARISON VISUALIZATION (Seaborn)")
    print("="*80 + "\n")
    
    # Load all results
    df = load_all_results()
    if df is None:
        return
    
    # Create output directory
    output_dir = Path(__file__).parent.parent.parent / "results" / "cross_model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    plot_success_rate_comparison(df, output_dir)
    plot_reward_comparison(df, output_dir)
    plot_performance_heatmap(df, output_dir)
    plot_steps_vs_reward(df, output_dir)
    plot_parallel_coordinates(df, output_dir)
    generate_comparison_table(df, output_dir)

    print(f"\n[COMPLETE] All cross-model visualizations saved to: {output_dir}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

