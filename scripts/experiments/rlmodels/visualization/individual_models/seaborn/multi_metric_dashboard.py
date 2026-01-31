"""
Multi-Metric Dashboard - Seaborn Version
2x2 GridSpec showing success rate, reward, steps, and collision rate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Optional

# Set Linux Libertine font globally for all plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Linux Libertine', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Linux Libertine'



def plot_multi_metric_dashboard_seaborn(df: pd.DataFrame, 
                                        model_name: str,
                                        output_path: Optional[str] = None):
    """
    Plot multi-metric dashboard with 4 subplots
    
    Args:
        df: DataFrame with metrics columns
        model_name: Name of the model to plot
        output_path: Path to save figure (optional)
    """
    # Filter data for this model
    model_data = df[df['model_name'] == model_name]
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    sns.set_style("whitegrid")
    
    # 1. Success Rate Over Time
    ax1 = fig.add_subplot(gs[0, 0])
    model_data['success_rate'] = model_data['success'].rolling(10, min_periods=1).mean() * 100
    ax1.plot(model_data['episode'], model_data['success_rate'], linewidth=2, color='#2E86AB')
    ax1.set_xlabel('Episode', fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontweight='bold')
    ax1.set_title('Success Rate Over Time', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # 2. Reward Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(data=model_data, x='reward', bins=20, kde=True, color='#A23B72', alpha=0.7, ax=ax2)
    mean_reward = model_data['reward'].mean()
    ax2.axvline(mean_reward, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_reward:.1f}')
    ax2.set_xlabel('Reward', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Reward Distribution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Steps Box Plot
    ax3 = fig.add_subplot(gs[1, 0])
    sns.boxplot(data=model_data, y='steps', color='#F18F01', ax=ax3)
    ax3.set_ylabel('Steps per Episode', fontweight='bold')
    ax3.set_title('Steps Distribution', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Collision Rate
    ax4 = fig.add_subplot(gs[1, 1])
    if 'collision' in model_data.columns:
        model_data['collision_rate'] = model_data['collision'].rolling(10, min_periods=1).mean() * 100
        ax4.plot(model_data['episode'], model_data['collision_rate'], linewidth=2, color='#C73E1D')
        ax4.set_xlabel('Episode', fontweight='bold')
        ax4.set_ylabel('Collision Rate (%)', fontweight='bold')
        ax4.set_title('Collision Rate Over Time', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 105)
    
    # Main title
    fig.suptitle(f'Multi-Metric Dashboard - {model_name}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return fig


if __name__ == "__main__":
    np.random.seed(42)
    data = {
        'episode': list(range(100)),
        'success': np.random.binomial(1, 0.6, 100),
        'reward': np.random.normal(100, 30, 100),
        'steps': np.random.poisson(25, 100),
        'collision': np.random.binomial(1, 0.2, 100),
        'model_name': ['SAC-Discrete'] * 100
    }
    df = pd.DataFrame(data)
    
    plot_multi_metric_dashboard_seaborn(df, 'SAC-Discrete', 'test_dashboard_seaborn.png')
    plt.show()

