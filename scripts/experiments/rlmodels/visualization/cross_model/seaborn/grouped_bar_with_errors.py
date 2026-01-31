"""
Grouped Bar Charts with Error Bars - Seaborn
Like the Darts NLL/D example with confidence intervals
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
import numpy as np

sns.set_style("whitegrid")


def plot_grouped_bars_with_ci_seaborn(df: pd.DataFrame,
                                       x: str = 'model_name',
                                       y: str = 'reward',
                                       hue: Optional[str] = None,
                                       output_path: Optional[str] = None,
                                       ci: int = 68):
    """
    Plot grouped bar chart with confidence interval error bars
    
    Args:
        df: DataFrame with data
        x: Column for x-axis (e.g., 'model_name')
        y: Column for y-axis metric (e.g., 'reward')
        hue: Optional grouping column
        output_path: Path to save figure
        ci: Confidence interval (68 for 1 std, 95 for 1.96 std)
    """
    plt.figure(figsize=(12, 6))

    # Create bar plot with error bars
    # Fix deprecation warnings: use hue=x when hue is None, and err_kws instead of errwidth
    if hue is None:
        ax = sns.barplot(
            data=df,
            x=x,
            y=y,
            hue=x,
            palette='Dark2',
            errorbar=('ci', ci),
            err_kws={'linewidth': 1.5},
            capsize=0.1,
            alpha=0.8,
            legend=False
        )
    else:
        ax = sns.barplot(
            data=df,
            x=x,
            y=y,
            hue=hue,
            palette='Dark2',
            errorbar=('ci', ci),
            err_kws={'linewidth': 1.5},
            capsize=0.1,
            alpha=0.8
        )
    
    # Styling
    ax.set_xlabel(x.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_ylabel(y.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f'{y.title()} Comparison with {ci}% CI', fontsize=14, fontweight='bold', pad=20)
    
    # Rotate x labels if needed
    plt.xticks(rotation=45, ha='right')
    
    # Grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Legend
    if hue:
        ax.legend(title=hue.replace('_', ' ').title(), frameon=True, shadow=True)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return ax.figure


def plot_multi_metric_grouped_bars(df: pd.DataFrame,
                                    metrics: List[str] = ['reward', 'steps', 'success_rate'],
                                    x: str = None,
                                    output_path: Optional[str] = None):
    """
    Plot multiple metrics as grouped bar charts in subplots

    Args:
        df: DataFrame with metrics
        metrics: List of metric columns
        x: Column for x-axis (auto-detects 'model' or 'model_name' if None)
        output_path: Save path
    """
    # Auto-detect model column if not specified
    if x is None:
        if 'model' in df.columns:
            x = 'model'
        elif 'model_name' in df.columns:
            x = 'model_name'
        else:
            raise ValueError("Could not find 'model' or 'model_name' column in DataFrame")

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    palette = sns.color_palette('Dark2', len(df[x].unique()))

    for idx, metric in enumerate(metrics):
        sns.barplot(
            data=df,
            x=x,
            y=metric,
            hue=x,
            ax=axes[idx],
            palette=palette,
            errorbar=('ci', 68),
            err_kws={'linewidth': 1.5},
            capsize=0.1,
            alpha=0.8,
            legend=False
        )
        
        axes[idx].set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Multi-Metric Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return fig


def plot_dataset_comparison_bars(df: pd.DataFrame,
                                  metric: str = 'reward',
                                  x: str = 'dataset',
                                  hue: str = 'model_name',
                                  output_path: Optional[str] = None):
    """
    Plot comparison across datasets (like Darts example)
    
    Args:
        df: DataFrame with dataset, model, and metric columns
        metric: Metric to plot
        x: Dataset column
        hue: Model grouping column
        output_path: Save path
    """
    plt.figure(figsize=(14, 6))
    
    # Custom palette (like Darts example)
    palette = sns.color_palette('Dark2', len(df[hue].unique()))
    
    ax = sns.barplot(
        data=df,
        x=x,
        y=metric,
        hue=hue,
        palette=palette,
        errorbar=('ci', 68),
        errwidth=1.5,
        capsize=0.05,
        alpha=0.85
    )
    
    # Styling
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f'{metric.title()} by Dataset', fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels
    plt.xticks(rotation=0)
    
    # Legend outside
    ax.legend(
        title='Model',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=True,
        shadow=True
    )
    
    # Grid
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return ax.figure

