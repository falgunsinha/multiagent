"""
Success Rate Over Time - Seaborn Version
Line plot with confidence intervals showing success rate progression
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


def plot_success_rate_seaborn(df: pd.DataFrame,
                              model_name: str,
                              output_path: Optional[str] = None,
                              window_size: int = 10):
    """
    Plot success rate over episodes with confidence intervals
    
    Args:
        df: DataFrame with columns ['episode', 'success', 'model_name']
        model_name: Name of the model to plot
        output_path: Path to save figure (optional)
        window_size: Rolling window size for smoothing
    """
    # Filter data for this model
    model_data = df[df['model_name'] == model_name].copy()
    
    # Calculate rolling success rate
    model_data['success_rate'] = model_data['success'].rolling(
        window=window_size, min_periods=1
    ).mean() * 100
    
    # Calculate confidence intervals
    model_data['success_std'] = model_data['success'].rolling(
        window=window_size, min_periods=1
    ).std() * 100
    
    model_data['ci_lower'] = model_data['success_rate'] - 1.96 * model_data['success_std'] / np.sqrt(window_size)
    model_data['ci_upper'] = model_data['success_rate'] + 1.96 * model_data['success_std'] / np.sqrt(window_size)
    
    # Create figure with whitegrid style (Example 2 style)
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    # Plot line with dashed style and markers (like Example 2)
    plt.plot(model_data['episode'], model_data['success_rate'],
            linewidth=2.5, linestyle='-', marker='o', markersize=4, markevery=max(1, len(model_data)//20),
            label=f'{model_name} Success Rate', color='#2E86AB')

    plt.fill_between(model_data['episode'],
                     model_data['ci_lower'],
                     model_data['ci_upper'],
                     alpha=0.2, color='#2E86AB', label='95% CI')

    # Styling
    plt.xlabel('Episode', fontsize=12, fontweight='bold')
    plt.ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    plt.title(f'Success Rate Over Time - {model_name}',
             fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='best', frameon=True, shadow=True, fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(0, 105)

    # Add horizontal line at 50%
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, linewidth=2, label='50% Baseline')
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return plt.gcf()


def main():
    """Example usage"""
    # Create sample data
    np.random.seed(42)
    episodes = 100
    
    data = {
        'episode': range(episodes),
        'success': np.random.binomial(1, 0.6, episodes),
        'model_name': ['SAC-Discrete'] * episodes
    }
    
    df = pd.DataFrame(data)
    
    # Plot
    fig = plot_success_rate_seaborn(
        df, 
        'SAC-Discrete',
        'test_success_rate_seaborn.png'
    )
    
    plt.show()


if __name__ == "__main__":
    main()

