"""
Reward Distribution - Seaborn Version
Histogram showing distribution of rewards
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

# Set Linux Libertine font globally for all plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Linux Libertine', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Linux Libertine'


def plot_reward_distribution_seaborn(df: pd.DataFrame,
                                     model_name: str,
                                     output_path: Optional[str] = None):
    """
    Plot reward distribution histogram
    
    Args:
        df: DataFrame with columns ['reward', 'model_name']
        model_name: Name of the model to plot
        output_path: Path to save figure (optional)
    """
    # Filter data for this model
    model_data = df[df['model_name'] == model_name]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Plot histogram with KDE
    sns.histplot(data=model_data, x='reward', bins=30, kde=True,
                color='#A23B72', alpha=0.7, ax=ax)
    
    # Add mean line
    mean_reward = model_data['reward'].mean()
    ax.axvline(mean_reward, color='red', linestyle='--', linewidth=2,
              label=f'Mean: {mean_reward:.2f}')
    
    # Add median line
    median_reward = model_data['reward'].median()
    ax.axvline(median_reward, color='green', linestyle='--', linewidth=2,
              label=f'Median: {median_reward:.2f}')
    
    # Styling
    ax.set_xlabel('Reward', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Reward Distribution - {model_name}', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return fig


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    data = {
        'reward': np.random.normal(100, 30, 100),
        'model_name': ['SAC-Discrete'] * 100
    }
    df = pd.DataFrame(data)
    
    plot_reward_distribution_seaborn(df, 'SAC-Discrete', 'test_reward_dist_seaborn.png')
    plt.show()

