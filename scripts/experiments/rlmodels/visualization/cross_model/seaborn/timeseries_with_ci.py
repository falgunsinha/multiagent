"""
Time Series with Shaded Confidence Intervals - Seaborn
Like AirPassengers/GasRateCO2 example with prediction intervals
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
import numpy as np

sns.set_style("whitegrid")


def plot_timeseries_with_ci_seaborn(df: pd.DataFrame,
                                     x: str = 'episode',
                                     y: str = 'reward',
                                     hue: Optional[str] = 'model_name',
                                     output_path: Optional[str] = None,
                                     ci: int = 95):
    """
    Plot time series with shaded confidence interval
    
    Args:
        df: DataFrame with time series data
        x: Column for x-axis (e.g., 'episode', 'timestep')
        y: Column for y-axis metric
        hue: Optional grouping column
        output_path: Path to save
        ci: Confidence interval (68, 95, etc.)
    """
    plt.figure(figsize=(14, 6))
    
    # Line plot with confidence interval
    ax = sns.lineplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        errorbar=('ci', ci),  # Shaded CI
        linewidth=2.5,
        palette='Dark2',
        alpha=0.8
    )
    
    # Styling
    ax.set_xlabel(x.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_ylabel(y.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f'{y.title()} Over Time with {ci}% CI', fontsize=14, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    if hue:
        ax.legend(title=hue.replace('_', ' ').title(), frameon=True, shadow=True, loc='best')
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return ax.figure


def plot_learning_curves_with_ci(df: pd.DataFrame,
                                  x: str = 'episode',
                                  y: str = 'reward',
                                  hue: str = 'model_name',
                                  output_path: Optional[str] = None,
                                  window: int = 10):
    """
    Plot smoothed learning curves with confidence intervals
    
    Args:
        df: DataFrame with episode data
        x: Episode/timestep column
        y: Metric column
        hue: Model grouping
        output_path: Save path
        window: Rolling window for smoothing
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Top: Raw data with CI
    sns.lineplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        errorbar=('ci', 95),
        linewidth=1.5,
        alpha=0.7,
        palette='tab10',
        ax=ax1
    )
    ax1.set_title('Raw Learning Curves with 95% CI', fontsize=12, fontweight='bold')
    ax1.set_ylabel(y.title(), fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(title='Model', frameon=True)
    
    # Bottom: Smoothed with CI
    # Apply rolling mean
    smoothed_df = df.copy()
    for model in df[hue].unique():
        mask = smoothed_df[hue] == model
        smoothed_df.loc[mask, f'{y}_smooth'] = (
            smoothed_df.loc[mask, y].rolling(window=window, min_periods=1).mean()
        )
    
    sns.lineplot(
        data=smoothed_df,
        x=x,
        y=f'{y}_smooth',
        hue=hue,
        errorbar=('ci', 95),
        linewidth=2.5,
        alpha=0.8,
        palette='tab10',
        ax=ax2
    )
    ax2.set_title(f'Smoothed Learning Curves (window={window}) with 95% CI', fontsize=12, fontweight='bold')
    ax2.set_xlabel(x.title(), fontsize=11)
    ax2.set_ylabel(f'{y.title()} (Smoothed)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(title='Model', frameon=True)
    
    plt.suptitle('Learning Curve Analysis', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return fig


def plot_prediction_intervals(df: pd.DataFrame,
                               x: str = 'episode',
                               y_true: str = 'reward',
                               y_pred: str = 'predicted_reward',
                               output_path: Optional[str] = None):
    """
    Plot predictions with confidence intervals (like AirPassengers example)
    
    Args:
        df: DataFrame with actual and predicted values
        x: Time column
        y_true: Actual values column
        y_pred: Predicted values column
        output_path: Save path
    """
    plt.figure(figsize=(14, 6))
    
    # Plot actual values
    plt.plot(df[x], df[y_true], 'k-', linewidth=2, label='Actual', alpha=0.8)
    
    # Plot predictions
    plt.plot(df[x], df[y_pred], color='#2E86AB', linewidth=2.5, label='Predicted')
    
    # Calculate prediction intervals (assuming you have std or quantiles)
    if 'pred_lower' in df.columns and 'pred_upper' in df.columns:
        plt.fill_between(
            df[x],
            df['pred_lower'],
            df['pred_upper'],
            alpha=0.3,
            color='#2E86AB',
            label='90% Prediction Interval'
        )
    else:
        # Calculate from residuals if available
        residuals = df[y_true] - df[y_pred]
        std = residuals.std()
        plt.fill_between(
            df[x],
            df[y_pred] - 1.645 * std,  # 90% interval
            df[y_pred] + 1.645 * std,
            alpha=0.3,
            color='#2E86AB',
            label='90% Prediction Interval'
        )
    
    plt.xlabel(x.title(), fontsize=12, fontweight='bold')
    plt.ylabel(y_true.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    plt.title('Predictions with Confidence Intervals', fontsize=14, fontweight='bold', pad=20)
    plt.legend(frameon=True, shadow=True, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return plt.gcf()

