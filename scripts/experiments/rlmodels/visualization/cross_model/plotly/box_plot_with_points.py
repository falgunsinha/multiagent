"""
Box Plot with All Data Points - Plotly
Shows distribution with individual data points like NBA scoring example
"""

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from typing import Optional
import numpy as np

PLOTLY_FONT = dict(
    family="Linux Libertine, serif",
    size=12,
    color="black"
)


def plot_box_with_points_plotly(df: pd.DataFrame,
                                 metric: str = 'reward',
                                 group_by: str = 'model_name',
                                 output_path: Optional[str] = None):
    """
    Plot box plots with all individual data points shown
    
    Args:
        df: DataFrame with metric column and grouping column
        metric: Column name for the metric to plot (e.g., 'reward', 'steps')
        group_by: Column name to group by (e.g., 'model_name')
        output_path: Path to save figure (optional)
    """
    # Color palette (similar to NBA example)
    colors = [
        'rgba(93, 164, 214, 0.5)',   # Blue
        'rgba(255, 144, 14, 0.5)',   # Orange
        'rgba(44, 160, 101, 0.5)',   # Green
        'rgba(255, 65, 54, 0.5)',    # Red
        'rgba(207, 114, 255, 0.5)',  # Purple
        'rgba(127, 96, 0, 0.5)',     # Brown
        'rgba(255, 140, 184, 0.5)',  # Pink
        'rgba(0, 191, 196, 0.5)',    # Cyan
    ]
    
    fig = go.Figure()
    
    # Get unique groups
    groups = df[group_by].unique()
    
    # Add box plot for each group
    for idx, group in enumerate(groups):
        group_data = df[df[group_by] == group][metric]
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Box(
            y=group_data,
            name=group,
            boxpoints='all',      # Show all points
            jitter=0.5,           # Spread points horizontally
            whiskerwidth=0.2,
            fillcolor=color,
            marker=dict(
                size=4,
                opacity=0.6
            ),
            line=dict(width=2)
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{metric.capitalize()} Distribution Across Models',
            font=dict(size=18, family=PLOTLY_FONT['family'], color='black')
        ),
        yaxis=dict(
            title=dict(text=metric.capitalize(), font=PLOTLY_FONT),
            autorange=True,
            showgrid=True,
            zeroline=True,
            gridcolor='rgb(255, 255, 255)',
            gridwidth=1,
            zerolinecolor='rgb(255, 255, 255)',
            zerolinewidth=2
        ),
        xaxis=dict(
            title=dict(text=group_by.replace('_', ' ').title(), font=PLOTLY_FONT)
        ),
        margin=dict(l=60, r=40, b=80, t=100),
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',
        showlegend=False,
        font=PLOTLY_FONT,
        width=1000,
        height=600
    )
    
    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"✅ Saved: {output_path}")
    
    return fig


def plot_multi_metric_boxes_plotly(df: pd.DataFrame,
                                    metrics: list = ['reward', 'steps'],
                                    group_by: str = 'model_name',
                                    output_path: Optional[str] = None):
    """
    Plot multiple metrics as box plots with points in subplots
    
    Args:
        df: DataFrame with metric columns
        metrics: List of metric column names
        group_by: Column to group by
        output_path: Path to save figure
    """
    from plotly.subplots import make_subplots
    
    n_metrics = len(metrics)
    fig = make_subplots(
        rows=1, cols=n_metrics,
        subplot_titles=[m.capitalize() for m in metrics]
    )
    
    colors = [
        'rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)',
        'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)',
        'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)',
    ]
    
    groups = df[group_by].unique()
    
    for col_idx, metric in enumerate(metrics, 1):
        for idx, group in enumerate(groups):
            group_data = df[df[group_by] == group][metric]
            
            fig.add_trace(
                go.Box(
                    y=group_data,
                    name=group,
                    boxpoints='all',
                    jitter=0.5,
                    whiskerwidth=0.2,
                    fillcolor=colors[idx % len(colors)],
                    marker=dict(size=3),
                    line=dict(width=1.5),
                    showlegend=(col_idx == 1)
                ),
                row=1, col=col_idx
            )
    
    fig.update_layout(
        title=dict(
            text='Multi-Metric Distribution Comparison',
            font=dict(size=18, family=PLOTLY_FONT['family'])
        ),
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',
        font=PLOTLY_FONT,
        height=500,
        width=400 * n_metrics
    )
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"✅ Saved: {output_path}")
    
    return fig

