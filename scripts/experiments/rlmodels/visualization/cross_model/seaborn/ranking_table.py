"""
Ranking Table - Seaborn
Generate ranking table for all models
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set Linux Libertine font globally for all plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Linux Libertine', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Linux Libertine'


def load_all_results():
    base_path = Path(__file__).parent.parent.parent / "results"
    results = {}
    for exp in ['exp1', 'exp2']:
        csv_path = base_path / exp / "comparison_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['experiment'] = 'Discrete' if exp == 'exp1' else 'Continuous'
            results[exp] = df
    if not results:
        return None
    return pd.concat(results.values(), ignore_index=True)

def plot_distribution_histograms_seaborn(df):
    """Create stacked histogram distribution plot - returns figure for W&B"""
    # Create figure with 2 rows, 2 columns for 4 key metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    # Map column names from test script to expected names
    column_mapping = {
        'avg_reward': 'reward',
        'success_rate': 'success_rate',
        'avg_length': 'steps',
        'avg_duration': 'time'
    }

    metrics = ['avg_reward', 'success_rate', 'avg_length']
    metric_labels = ['Reward Distribution', 'Success Rate Distribution', 'Steps Distribution']

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        if idx >= len(axes):
            break
        ax = axes[idx]

        # Stacked Histogram
        sns.histplot(
            data=df,
            x=metric,
            hue='model',
            multiple='stack',
            ax=ax,
            palette='tab10',
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5,
            bins=15
        )

        ax.set_title(label, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        # Only add legend if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(title='Model', fontsize=8, title_fontsize=9, loc='best')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Hide the 4th subplot if we only have 3 metrics
    if len(metrics) < 4:
        axes[3].set_visible(False)

    plt.suptitle('Cross-Model Distribution Comparison - Stacked Histograms',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig

def plot_distribution_kde_seaborn(df):
    """Create KDE distribution plot - returns figure for W&B"""
    # Create figure with 2 rows, 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    metrics = ['avg_reward', 'success_rate', 'avg_length']
    metric_labels = ['Reward Distribution', 'Success Rate Distribution', 'Steps Distribution']

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        if idx >= len(axes):
            break
        ax = axes[idx]

        # KDE plot with filled areas
        try:
            sns.kdeplot(
                data=df,
                x=metric,
                hue='model',
                fill=True,
                common_norm=False,
                alpha=0.5,
                linewidth=2,
                ax=ax,
                palette='tab10',
                multiple='layer',
                warn_singular=False  # Suppress low variance warning
            )
        except:
            # Fallback if KDE fails
            sns.histplot(data=df, x=metric, hue='model', ax=ax, palette='tab10', alpha=0.5)

        ax.set_title(label, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        # Only add legend if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(title='Model', fontsize=8, title_fontsize=9, loc='best')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Hide the 4th subplot
    if len(metrics) < 4:
        axes[3].set_visible(False)

    plt.suptitle('Cross-Model Distribution Comparison - Kernel Density Estimation',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig

def create_ranking_table(df, output_dir):
    """Create distribution comparison visualizations (stacked histogram + KDE)"""

    # Create figure with 2 rows, 2 columns for 4 key metrics
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    metrics = ['reward', 'success_rate', 'steps', 'time']
    metric_labels = ['Reward Distribution', 'Success Rate Distribution',
                     'Steps Distribution', 'Inference Time Distribution']

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        # 1. Stacked Histogram
        sns.histplot(
            data=df,
            x=metric,
            hue='model',
            multiple='stack',
            ax=ax,
            palette='tab10',
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5,
            bins=20
        )

        ax.set_title(label, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.legend(title='Model', fontsize=9, title_fontsize=10, loc='best')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle('Cross-Model Distribution Comparison - Stacked Histograms',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'distribution_stacked_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] distribution_stacked_histograms.png")

    # 2. KDE Plots (Kernel Density Estimation)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        # KDE plot with filled areas
        sns.kdeplot(
            data=df,
            x=metric,
            hue='model',
            fill=True,
            common_norm=False,
            alpha=0.5,
            linewidth=2,
            ax=ax,
            palette='tab10',
            multiple='stack'
        )

        ax.set_title(label, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(title='Model', fontsize=9, title_fontsize=10, loc='best')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle('Cross-Model Distribution Comparison - Kernel Density Estimation',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'distribution_kde_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] distribution_kde_plots.png")

def main():
    print("\n" + "="*80)
    print("DISTRIBUTION COMPARISON (Seaborn)")
    print("="*80 + "\n")
    
    df = load_all_results()
    if df is None:
        print("[ERROR] No results found!")
        return
    
    output_dir = Path(__file__).parent.parent.parent / "results" / "cross_model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    create_ranking_table(df, output_dir)
    
    print("\n[COMPLETE] Ranking table generated")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

