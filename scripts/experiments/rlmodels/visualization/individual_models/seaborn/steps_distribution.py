"""
Steps Distribution - Seaborn Version
Histogram and box plot showing distribution of steps per episode
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



def plot_steps_distribution_seaborn(df: pd.DataFrame, 
                                    model_name: str,
                                    output_path: Optional[str] = None):
    """
    Plot steps distribution with histogram and box plot
    
    Args:
        df: DataFrame with columns ['steps', 'model_name']
        model_name: Name of the model to plot
        output_path: Path to save figure (optional)
    """
    # Filter data for this model
    model_data = df[df['model_name'] == model_name]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.set_style("whitegrid")
    
    # Histogram
    sns.histplot(data=model_data, x='steps', bins=20, kde=True,
                color='#F18F01', alpha=0.7, ax=ax1)
    
    mean_steps = model_data['steps'].mean()
    ax1.axvline(mean_steps, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_steps:.1f}')
    
    ax1.set_xlabel('Steps per Episode', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(f'Steps Distribution - {model_name}', 
                 fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    sns.boxplot(data=model_data, y='steps', color='#F18F01', ax=ax2)
    
    ax2.set_ylabel('Steps per Episode', fontsize=12, fontweight='bold')
    ax2.set_title(f'Steps Box Plot - {model_name}', 
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return fig


if __name__ == "__main__":
    np.random.seed(42)
    data = {
        'steps': np.random.poisson(25, 100),
        'model_name': ['SAC-Discrete'] * 100
    }
    df = pd.DataFrame(data)
    
    plot_steps_distribution_seaborn(df, 'SAC-Discrete', 'test_steps_dist_seaborn.png')
    plt.show()

