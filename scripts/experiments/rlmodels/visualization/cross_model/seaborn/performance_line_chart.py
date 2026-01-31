"""
Performance Line Chart - Seaborn
Track performance metrics over episodes
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set theme to "ticks" style (Example 1 style)
sns.set_theme(style="ticks")

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

def plot_performance_line_seaborn(df):
    """Plot performance over episodes using relplot - returns figure for W&B"""

    # Check if we have 'episode' column, if not create it
    if 'episode' not in df.columns:
        df = df.copy()
        df['episode'] = df.groupby('model').cumcount() + 1

    # Define the palette with enough colors for all models
    n_models = df['model'].nunique()
    palette = sns.color_palette("rocket_r", n_colors=max(n_models, 10))

    # Melt the dataframe to have a 'metric' column for faceting
    df_melted = df.melt(
        id_vars=['episode', 'model'],
        value_vars=['avg_reward', 'success_rate'],
        var_name='metric',
        value_name='value'
    )

    # Map metric names to readable labels
    df_melted['metric'] = df_melted['metric'].map({
        'avg_reward': 'Reward',
        'success_rate': 'Success Rate (%)'
    })

    # Create relplot with facets
    g = sns.relplot(
        data=df_melted,
        x="episode",
        y="value",
        hue="model",
        col="metric",
        kind="line",
        palette=palette,
        height=5,
        aspect=1.2,
        facet_kws=dict(sharex=True, sharey=False),
        linewidth=2.5,
        alpha=0.8
    )

    # Customize titles and labels
    g.set_titles("{col_name}", fontsize=14, fontweight='bold')
    g.set_axis_labels("Episode", "", fontsize=11, fontweight='bold')

    # Set y-axis limits for success rate
    for ax, title in zip(g.axes.flat, ['Reward', 'Success Rate (%)']):
        if 'Success Rate' in title:
            ax.set_ylim(0, 100)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')

    # Adjust legend
    g._legend.set_title('Model')
    g._legend.set_bbox_to_anchor((1.05, 0.5))

    plt.tight_layout()
    return g.fig

def plot_performance_line_chart(df, output_dir):
    """Plot performance over episodes with Example 2 style (whitegrid + dashed lines)"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Use tab10 palette for multiple models (like Example 2)
    palette = sns.color_palette("tab10", n_colors=df['model'].nunique())

    # Reward over episodes with dashed/dotted line styles
    sns.lineplot(data=df, x='episode', y='reward', hue='model', style='experiment',
                 ax=axes[0], linewidth=2.5, alpha=0.8, palette=palette,
                 dashes=[(2, 2), (5, 2), (1, 1), (3, 1, 1, 1)])  # Various dash patterns
    axes[0].set_title('Reward Over Episodes - All Models', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Episode', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Reward', fontsize=12, fontweight='bold')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, frameon=True)
    axes[0].grid(alpha=0.3, linestyle='--')

    # Success rate over episodes with dashed/dotted line styles
    sns.lineplot(data=df, x='episode', y='success_rate', hue='model', style='experiment',
                 ax=axes[1], linewidth=2.5, alpha=0.8, palette=palette,
                 dashes=[(2, 2), (5, 2), (1, 1), (3, 1, 1, 1)])  # Various dash patterns
    axes[1].set_title('Success Rate Over Episodes - All Models', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Episode', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, frameon=True)
    axes[1].grid(alpha=0.3, linestyle='--')
    axes[1].set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_line_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] performance_line_chart.png")

def main():
    print("\n" + "="*80)
    print("PERFORMANCE LINE CHART (Seaborn)")
    print("="*80 + "\n")
    
    df = load_all_results()
    if df is None:
        print("[ERROR] No results found!")
        return
    
    output_dir = Path(__file__).parent.parent.parent / "results" / "cross_model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_performance_line_chart(df, output_dir)
    
    print("\n[COMPLETE] Performance line chart generated")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

