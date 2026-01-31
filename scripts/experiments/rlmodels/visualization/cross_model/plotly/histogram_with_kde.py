"""
Histogram with KDE Distribution Curve - Plotly
Shows distribution with overlaid kernel density estimate
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional, List
import numpy as np
from scipy import stats

PLOTLY_FONT = dict(
    family="Linux Libertine, serif",
    size=12,
    color="black"
)


def plot_histogram_with_kde_plotly(df: pd.DataFrame,
                                    metric: str = 'reward',
                                    group_by: str = 'model_name',
                                    output_path: Optional[str] = None):
    """
    Plot histogram with KDE overlay for multiple groups
    
    Args:
        df: DataFrame with metric column
        metric: Column name for the metric
        group_by: Column to group by (e.g., 'model_name')
        output_path: Path to save figure
    """
    fig = go.Figure()
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    groups = df[group_by].unique()
    
    for idx, group in enumerate(groups):
        group_data = df[df[group_by] == group][metric].dropna()
        color = colors[idx % len(colors)]

        # Add histogram
        fig.add_trace(go.Histogram(
            x=group_data,
            name=group,
            opacity=0.6,
            marker_color=color,
            nbinsx=30,
            histnorm='probability density'  # Normalize for KDE overlay
        ))

        # Calculate KDE - only if we have enough variance
        if len(group_data) > 2 and np.std(group_data) > 1e-6:
            try:
                kde = stats.gaussian_kde(group_data)
                x_range = np.linspace(group_data.min(), group_data.max(), 200)
                kde_values = kde(x_range)

                # Add KDE curve
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=kde_values,
                    mode='lines',
                    name=f'{group} (KDE)',
                    line=dict(color=color, width=3),
                    showlegend=True
                ))
            except np.linalg.LinAlgError:
                # Skip KDE if data is singular (no variance)
                pass
    
    fig.update_layout(
        title=dict(
            text=f'{metric.capitalize()} Distribution with KDE',
            font=dict(size=18, family=PLOTLY_FONT['family'])
        ),
        xaxis=dict(title=dict(text=metric.capitalize(), font=PLOTLY_FONT)),
        yaxis=dict(title=dict(text='Density', font=PLOTLY_FONT)),
        barmode='overlay',
        template='plotly_white',
        font=PLOTLY_FONT,
        width=1000,
        height=600
    )
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"✅ Saved: {output_path}")
    
    return fig


def plot_multi_histogram_kde_grid(df: pd.DataFrame,
                                   metrics: List[str] = ['reward', 'steps'],
                                   group_by: str = 'model_name',
                                   output_path: Optional[str] = None):
    """
    Plot multiple histograms with KDE in a grid layout
    
    Args:
        df: DataFrame with metric columns
        metrics: List of metrics to plot
        group_by: Column to group by
        output_path: Path to save figure
    """
    n_metrics = len(metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[m.capitalize() for m in metrics]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    groups = df[group_by].unique()
    
    for metric_idx, metric in enumerate(metrics):
        row = metric_idx // n_cols + 1
        col = metric_idx % n_cols + 1
        
        for group_idx, group in enumerate(groups):
            group_data = df[df[group_by] == group][metric].dropna()
            color = colors[group_idx % len(colors)]
            
            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=group_data,
                    name=group,
                    opacity=0.5,
                    marker_color=color,
                    nbinsx=25,
                    histnorm='probability density',
                    showlegend=(metric_idx == 0)
                ),
                row=row, col=col
            )
            
            # KDE
            if len(group_data) > 1:
                kde = stats.gaussian_kde(group_data)
                x_range = np.linspace(group_data.min(), group_data.max(), 100)
                kde_values = kde(x_range)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=kde_values,
                        mode='lines',
                        line=dict(color=color, width=2),
                        showlegend=False
                    ),
                    row=row, col=col
                )
    
    fig.update_layout(
        title=dict(
            text='Multi-Metric Distributions with KDE',
            font=dict(size=18, family=PLOTLY_FONT['family'])
        ),
        barmode='overlay',
        template='plotly_white',
        font=PLOTLY_FONT,
        height=400 * n_rows,
        width=500 * n_cols
    )
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"✅ Saved: {output_path}")
    
    return fig

