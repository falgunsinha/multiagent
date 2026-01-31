"""
Success Rate Comparison - Seaborn
Compare success rates across all models
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


sns.set_style("whitegrid")

def load_all_results():
    """Load results from both experiments"""
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

def plot_success_rate_line_seaborn(df):
    """Plot success rate over episodes with confidence bands - returns figure for W&B"""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Check if we have 'episode' column, if not create it
    if 'episode' not in df.columns:
        df = df.copy()
        df['episode'] = df.groupby('model').cumcount() + 1

    # Create line plot with confidence interval bands (95% CI)
    sns.lineplot(
        data=df,
        x='episode',
        y='success_rate',
        hue='model',
        ax=ax,
        linewidth=2.5,
        alpha=0.9,
        palette='tab10',
        errorbar=('ci', 95)  # 95% confidence interval
    )

    ax.set_title('Success Rate Over Episodes - All Models', fontsize=16, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(title='Model', fontsize=10, title_fontsize=11, loc='best', frameon=True, shadow=True)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig

def plot_success_rate_comparison(df, output_dir):
    """Plot success rate comparison"""
    plt.figure(figsize=(16, 8))
    
    success_rates = df.groupby(['model', 'experiment'])['success_rate'].mean().reset_index()
    
    sns.barplot(data=success_rates, x='model', y='success_rate', hue='experiment', palette='Set2')
    
    plt.title('Success Rate Comparison - All Models', fontsize=18, fontweight='bold')
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Success Rate (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.legend(title='Experiment Type', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.1f%%', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'success_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] success_rate_comparison.png")

def main():
    """Main function"""
    print("\n" + "="*80)
    print("SUCCESS RATE COMPARISON (Seaborn)")
    print("="*80 + "\n")
    
    df = load_all_results()
    if df is None:
        print("[ERROR] No results found!")
        return
    
    output_dir = Path(__file__).parent.parent.parent / "results" / "cross_model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_success_rate_comparison(df, output_dir)
    
    print("\n[COMPLETE] Success rate comparison generated")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

