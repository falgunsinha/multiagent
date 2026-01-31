"""
Radar Overlay - Seaborn
Overlay radar charts for all models
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

def plot_radar_overlay(df, output_dir):
    """Create radar chart overlay"""
    # Calculate metrics per model
    metrics = df.groupby('model').agg({
        'success_rate': 'mean',
        'reward': 'mean',
        'steps': 'mean',
        'time': 'mean'
    }).reset_index()
    
    # Normalize (0-100 scale)
    for col in ['success_rate', 'reward', 'steps', 'time']:
        if col in ['steps', 'time']:  # Lower is better
            metrics[f'{col}_norm'] = 100 - ((metrics[col] - metrics[col].min()) / 
                                            (metrics[col].max() - metrics[col].min() + 1e-6) * 100)
        else:  # Higher is better
            metrics[f'{col}_norm'] = (metrics[col] - metrics[col].min()) / \
                                     (metrics[col].max() - metrics[col].min() + 1e-6) * 100
    
    # Setup radar chart
    categories = ['Success\nRate', 'Reward', 'Efficiency\n(Steps)', 'Speed\n(Time)']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
    for idx, (_, row) in enumerate(metrics.iterrows()):
        values = [row['success_rate_norm'], row['reward_norm'], 
                 row['steps_norm'], row['time_norm']]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], 
                color=colors[idx], alpha=0.7)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.title('Performance Radar Overlay - All Models', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'radar_overlay.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] radar_overlay.png")

def main():
    print("\n" + "="*80)
    print("RADAR OVERLAY (Seaborn)")
    print("="*80 + "\n")
    
    df = load_all_results()
    if df is None:
        print("[ERROR] No results found!")
        return
    
    output_dir = Path(__file__).parent.parent.parent / "results" / "cross_model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_radar_overlay(df, output_dir)
    
    print("\n[COMPLETE] Radar overlay generated")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

