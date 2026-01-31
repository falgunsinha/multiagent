"""
Performance Metrics Bar Chart - Seaborn Version
Clean bar chart showing multiple metrics comparison
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



def plot_performance_radar_seaborn(df: pd.DataFrame,
                                   model_name: str,
                                   output_path: Optional[str] = None):
    """
    Plot performance metrics as a grouped bar chart

    Args:
        df: DataFrame with metrics columns
        model_name: Name of the model to plot
        output_path: Path to save figure (optional)
    """
    # Filter data for this model
    model_data = df[df['model_name'] == model_name]

    # Calculate metrics
    success_rate = model_data['success'].mean() * 100
    avg_reward = model_data['reward'].mean()
    avg_steps = model_data['steps'].mean()

    # Normalize reward (assuming range -100 to 200)
    norm_reward = min(100, max(0, (avg_reward + 100) / 3))

    # Efficiency (inverse of steps - fewer steps is better)
    efficiency = max(0, 100 - avg_steps * 2) if avg_steps < 50 else 0

    # Create dataframe for plotting
    metrics_df = pd.DataFrame({
        'Metric': ['Success\nRate (%)', 'Reward\nScore', 'Efficiency\nScore', 'Avg Steps'],
        'Value': [success_rate, norm_reward, efficiency, avg_steps],
        'Category': ['Performance', 'Performance', 'Performance', 'Steps']
    })

    # Create figure with custom style
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set seaborn style
    sns.set_style("whitegrid")

    # Create bar plot
    bars = ax.bar(
        metrics_df['Metric'],
        metrics_df['Value'],
        color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
        alpha=0.8,
        edgecolor='black',
        linewidth=1.5
    )

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.1f}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )

    # Customize appearance
    ax.set_ylabel('Metric Value', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
    ax.set_title(f'Performance Metrics - {model_name}',
                 fontsize=16, fontweight='bold', pad=20)

    # Add horizontal grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Set y-axis limits
    ax.set_ylim(0, max(metrics_df['Value']) * 1.15)

    # Customize ticks
    plt.xticks(fontsize=11, fontweight='bold')
    plt.yticks(fontsize=10)

    # Add reference line at 50
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, linewidth=1)

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
        'success': np.random.binomial(1, 0.6, 100),
        'reward': np.random.normal(100, 30, 100),
        'steps': np.random.poisson(25, 100),
        'collision': np.random.binomial(1, 0.2, 100),
        'inference_time': np.random.uniform(5, 15, 100),
        'model_name': ['SAC-Discrete'] * 100
    }
    df = pd.DataFrame(data)
    
    plot_performance_radar_seaborn(df, 'SAC-Discrete', 'test_radar_seaborn.png')
    plt.show()

