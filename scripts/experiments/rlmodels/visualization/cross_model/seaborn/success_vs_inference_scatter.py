"""
Success vs Inference Time Scatter - Seaborn
Analyze trade-off between success rate and inference time
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

def plot_success_vs_inference_seaborn(df):
    """Plot success rate vs inference time - returns figure for W&B"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Calculate average metrics per model
    metrics = df.groupby('model').agg({
        'success_rate': 'mean',
        'avg_duration': 'mean'
    }).reset_index()

    # Create scatter plot
    sns.scatterplot(data=metrics, x='avg_duration', y='success_rate', hue='model',
                    s=200, alpha=0.7, palette='tab10', ax=ax)

    # Add model labels
    for idx, row in metrics.iterrows():
        ax.annotate(row['model'], (row['avg_duration'], row['success_rate']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.7)

    ax.set_title('Success Rate vs Inference Time - All Models', fontsize=16, fontweight='bold')
    ax.set_xlabel('Average Inference Time (s)', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig

def plot_success_vs_inference(df, output_dir):
    """Plot success rate vs inference time"""
    plt.figure(figsize=(14, 8))
    
    # Calculate average metrics per model
    metrics = df.groupby(['model', 'experiment']).agg({
        'success_rate': 'mean',
        'time': 'mean'
    }).reset_index()
    
    # Create scatter plot
    sns.scatterplot(data=metrics, x='time', y='success_rate', hue='model', 
                    style='experiment', s=200, alpha=0.7, palette='tab10')
    
    # Add model labels
    for idx, row in metrics.iterrows():
        plt.annotate(row['model'], (row['time'], row['success_rate']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.7)
    
    plt.title('Success Rate vs Inference Time - All Models', fontsize=18, fontweight='bold')
    plt.xlabel('Average Inference Time (s)', fontsize=14)
    plt.ylabel('Success Rate (%)', fontsize=14)
    plt.ylim(0, 100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'success_vs_inference_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] success_vs_inference_scatter.png")

def main():
    print("\n" + "="*80)
    print("SUCCESS VS INFERENCE SCATTER (Seaborn)")
    print("="*80 + "\n")
    
    df = load_all_results()
    if df is None:
        print("[ERROR] No results found!")
        return
    
    output_dir = Path(__file__).parent.parent.parent / "results" / "cross_model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_success_vs_inference(df, output_dir)
    
    print("\n[COMPLETE] Success vs inference scatter generated")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

