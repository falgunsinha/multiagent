"""
Cross-Model Pairplot Matrix - Seaborn
Correlation matrix showing relationships between metrics across all models
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List

# Set Linux Libertine font globally for all plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Linux Libertine', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Linux Libertine'


def plot_cross_model_pairplot_seaborn(df: pd.DataFrame,
                                       metrics: Optional[List[str]] = None,
                                       hue: str = 'model',
                                       output_path: Optional[str] = None):
    """
    Create a pairplot matrix showing correlations between metrics across all models
    
    Args:
        df: DataFrame with metrics columns and model identifier
        metrics: List of metric column names to include (default: ['success_rate', 'avg_reward', 'avg_length'])
        hue: Column name to use for color coding (default: 'model')
        output_path: Path to save figure (optional)
    
    Returns:
        matplotlib Figure object
    """
    # Set seaborn theme
    sns.set_theme(style="ticks")
    
    # Default metrics if not provided
    if metrics is None:
        metrics = ['success_rate', 'avg_reward', 'avg_length']
    
    # Ensure hue column exists
    if hue not in df.columns:
        raise ValueError(f"Hue column '{hue}' not found in DataFrame")
    
    # Select only the metrics and hue column
    plot_df = df[metrics + [hue]].copy()
    
    # Check if we have enough variance in the data
    for metric in metrics:
        if plot_df[metric].std() < 1e-6:
            print(f"[WARNING] Low variance in {metric}, pairplot may not be informative")
    
    # Create pairplot
    pairplot = sns.pairplot(
        plot_df,
        hue=hue,
        diag_kind='kde',  # KDE on diagonal
        plot_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'k', 'linewidth': 0.5},
        diag_kws={'alpha': 0.7, 'linewidth': 2},
        corner=False  # Show full matrix
    )
    
    # Customize the plot
    pairplot.fig.suptitle('Cross-Model Performance Correlation Matrix', 
                          fontsize=16, fontweight='bold', y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        pairplot.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] {output_path}")
    
    return pairplot.fig


def plot_cross_model_corner_pairplot_seaborn(df: pd.DataFrame,
                                              metrics: Optional[List[str]] = None,
                                              hue: str = 'model',
                                              output_path: Optional[str] = None):
    """
    Create a corner pairplot (lower triangle only) for cross-model comparison
    
    Args:
        df: DataFrame with metrics columns and model identifier
        metrics: List of metric column names to include
        hue: Column name to use for color coding (default: 'model')
        output_path: Path to save figure (optional)
    
    Returns:
        matplotlib Figure object
    """
    # Set seaborn theme
    sns.set_theme(style="ticks")
    
    # Default metrics if not provided
    if metrics is None:
        metrics = ['success_rate', 'avg_reward', 'avg_length']
    
    # Select only the metrics and hue column
    plot_df = df[metrics + [hue]].copy()
    
    # Create corner pairplot
    pairplot = sns.pairplot(
        plot_df,
        hue=hue,
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'k', 'linewidth': 0.5},
        diag_kws={'alpha': 0.7, 'linewidth': 2},
        corner=True  # Show only lower triangle
    )
    
    # Customize the plot
    pairplot.fig.suptitle('Cross-Model Performance Correlation (Corner Plot)', 
                          fontsize=16, fontweight='bold', y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        pairplot.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] {output_path}")
    
    return pairplot.fig


if __name__ == "__main__":
    # Example usage
    print("\n" + "="*80)
    print("CROSS-MODEL PAIRPLOT MATRIX (Seaborn)")
    print("="*80 + "\n")
    
    # This would normally load actual data
    # For testing, create sample data
    import numpy as np
    
    models = ['DDQN', 'DQN', 'PPO', 'A2C', 'SAC']
    n_samples = 20
    
    data = []
    for model in models:
        for _ in range(n_samples):
            data.append({
                'model': model,
                'success_rate': np.random.uniform(60, 95),
                'avg_reward': np.random.uniform(100, 200),
                'avg_length': np.random.uniform(20, 50)
            })
    
    df = pd.DataFrame(data)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent.parent / "results" / "cross_model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate pairplot
    fig = plot_cross_model_pairplot_seaborn(df, output_path=output_dir / 'cross_model_pairplot.png')
    
    print("\n[COMPLETE] Cross-model pairplot generated")
    print("="*80 + "\n")

