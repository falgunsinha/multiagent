"""
Grouped Bar Charts with Error Bars - Plotly
Interactive version with confidence intervals
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional, List
import numpy as np

PLOTLY_FONT = dict(
    family="Linux Libertine, serif",
    size=12,
    color="black"
)


def plot_grouped_bars_with_ci_plotly(df: pd.DataFrame,
                                      x: str = 'model_name',
                                      y: str = 'reward',
                                      hue: Optional[str] = None,
                                      output_path: Optional[str] = None):
    """
    Plot grouped bar chart with error bars (interactive)
    
    Args:
        df: DataFrame with data
        x: Column for x-axis
        y: Column for y-axis metric
        hue: Optional grouping column
        output_path: Path to save
    """
    fig = go.Figure()
    
    if hue:
        # Grouped bars
        groups = df[hue].unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for idx, group in enumerate(groups):
            group_data = df[df[hue] == group]
            
            # Calculate mean and std for each x category
            stats = group_data.groupby(x)[y].agg(['mean', 'std', 'count']).reset_index()
            stats['se'] = stats['std'] / np.sqrt(stats['count'])  # Standard error
            stats['ci'] = 1.96 * stats['se']  # 95% CI
            
            fig.add_trace(go.Bar(
                x=stats[x],
                y=stats['mean'],
                name=group,
                marker_color=colors[idx % len(colors)],
                error_y=dict(
                    type='data',
                    array=stats['ci'],
                    visible=True,
                    thickness=1.5,
                    width=5
                )
            ))
    else:
        # Single group
        stats = df.groupby(x)[y].agg(['mean', 'std', 'count']).reset_index()
        stats['se'] = stats['std'] / np.sqrt(stats['count'])
        stats['ci'] = 1.96 * stats['se']
        
        fig.add_trace(go.Bar(
            x=stats[x],
            y=stats['mean'],
            marker_color='#2E86AB',
            error_y=dict(
                type='data',
                array=stats['ci'],
                visible=True
            )
        ))
    
    fig.update_layout(
        title=dict(
            text=f'{y.title()} Comparison with 95% CI',
            font=dict(size=18, family=PLOTLY_FONT['family'])
        ),
        xaxis=dict(title=dict(text=x.replace('_', ' ').title(), font=PLOTLY_FONT)),
        yaxis=dict(title=dict(text=y.replace('_', ' ').title(), font=PLOTLY_FONT)),
        barmode='group',
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


def plot_multi_metric_grouped_bars_plotly(df: pd.DataFrame,
                                           metrics: List[str] = ['reward', 'steps'],
                                           x: str = None,
                                           output_path: Optional[str] = None):
    """
    Multiple metrics as grouped bars in subplots

    Args:
        df: DataFrame
        metrics: List of metrics
        x: X-axis column (auto-detects 'model' or 'model_name' if None)
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
    fig = make_subplots(
        rows=1, cols=n_metrics,
        subplot_titles=[m.replace('_', ' ').title() for m in metrics]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    x_categories = df[x].unique()
    
    for col_idx, metric in enumerate(metrics, 1):
        stats = df.groupby(x)[metric].agg(['mean', 'std', 'count']).reset_index()
        stats['ci'] = 1.96 * stats['std'] / np.sqrt(stats['count'])
        
        for idx, category in enumerate(x_categories):
            cat_stats = stats[stats[x] == category].iloc[0]
            
            fig.add_trace(
                go.Bar(
                    x=[category],
                    y=[cat_stats['mean']],
                    name=category,
                    marker_color=colors[idx % len(colors)],
                    error_y=dict(
                        type='data',
                        array=[cat_stats['ci']],
                        visible=True
                    ),
                    showlegend=(col_idx == 1)
                ),
                row=1, col=col_idx
            )
    
    fig.update_layout(
        title=dict(
            text='Multi-Metric Comparison',
            font=dict(size=18, family=PLOTLY_FONT['family'])
        ),
        template='plotly_white',
        font=PLOTLY_FONT,
        height=500,
        width=500 * n_metrics
    )
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"✅ Saved: {output_path}")
    
    return fig

