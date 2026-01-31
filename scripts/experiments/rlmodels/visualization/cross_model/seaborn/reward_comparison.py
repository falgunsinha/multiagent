"""
Reward Comparison - Seaborn
Compare reward distributions across all models
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

def plot_reward_box_seaborn(df):
    """Plot reward comparison using box plot - returns figure for W&B"""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Use box plot with hue assigned to fix deprecation warning
    sns.boxplot(data=df, x='model', y='avg_reward', hue='model', palette='Set2', ax=ax, legend=False)

    ax.set_title('Reward Distribution Comparison - All Models', fontsize=16, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig

def plot_reward_comparison(df, output_dir):
    """Plot reward comparison using box plot"""
    plt.figure(figsize=(16, 8))

    # Use box plot instead of violin plot
    sns.boxplot(data=df, x='model', y='reward', hue='experiment', palette='Set2')

    plt.title('Reward Distribution Comparison - All Models', fontsize=18, fontweight='bold')
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Experiment Type', fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'reward_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] reward_comparison.png")

def main():
    print("\n" + "="*80)
    print("REWARD COMPARISON (Seaborn)")
    print("="*80 + "\n")
    
    df = load_all_results()
    if df is None:
        print("[ERROR] No results found!")
        return
    
    output_dir = Path(__file__).parent.parent.parent / "results" / "cross_model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_reward_comparison(df, output_dir)
    
    print("\n[COMPLETE] Reward comparison generated")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

