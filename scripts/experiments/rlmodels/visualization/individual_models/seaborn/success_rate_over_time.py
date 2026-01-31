"""
Success Rate Over Time - Seaborn
Track success rate progression across episodes
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

def plot_success_rate_over_time(experiment_name, output_dir):
    """Plot success rate progression over episodes"""
    results_path = Path(__file__).parent.parent.parent / "results" / experiment_name / "comparison_results.csv"
    if not results_path.exists():
        print(f"[ERROR] Results not found: {results_path}")
        return
    
    df = pd.read_csv(results_path)
    
    plt.figure(figsize=(14, 8))
    
    # Calculate rolling success rate for each model
    window_size = 10
    for model in df['model'].unique():
        model_data = df[df['model'] == model].sort_values('episode')
        rolling_success = model_data['success_rate'].rolling(window=window_size, min_periods=1).mean()
        plt.plot(model_data['episode'], rolling_success, label=model, linewidth=2, alpha=0.8)
    
    plt.title(f'Success Rate Over Time - {experiment_name.upper()}', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Success Rate (%) - Rolling Average', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{experiment_name}_success_rate_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {experiment_name}_success_rate_over_time.png")

def main():
    """Generate for both experiments"""
    print("\n" + "="*80)
    print("SUCCESS RATE OVER TIME (Seaborn)")
    print("="*80 + "\n")
    
    for exp in ['exp1', 'exp2']:
        output_dir = Path(__file__).parent.parent.parent / "results" / exp / "visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_success_rate_over_time(exp, output_dir)
    
    print("\n[COMPLETE] Success rate over time visualizations generated")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

