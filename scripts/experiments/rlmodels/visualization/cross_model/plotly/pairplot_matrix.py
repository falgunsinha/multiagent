"""
Cross-Model Pairplot Matrix - Plotly
Interactive correlation matrix showing relationships between metrics across all models
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional, List
import numpy as np

# Plotly font configuration for Linux Libertine
PLOTLY_FONT = dict(
    family="Linux Libertine, serif",
    size=12,
    color="black"
)


def plot_cross_model_pairplot_plotly(df: pd.DataFrame,
                                      metrics: Optional[List[str]] = None,
                                      color: str = 'model',
                                      output_path: Optional[str] = None):
    """
    Create an interactive pairplot matrix using Plotly's scatter_matrix
    
    Args:
        df: DataFrame with metrics columns and model identifier
        metrics: List of metric column names to include (default: ['success_rate', 'avg_reward', 'avg_length'])
        color: Column name to use for color coding (default: 'model')
        output_path: Path to save figure (optional)
    
    Returns:
        plotly Figure object
    """
    # Default metrics if not provided
    if metrics is None:
        metrics = ['success_rate', 'avg_reward', 'avg_length']
    
    # Ensure color column exists
    if color not in df.columns:
        raise ValueError(f"Color column '{color}' not found in DataFrame")
    
    # Select only the metrics and color column
    plot_df = df[metrics + [color]].copy()
    
    # Create scatter matrix
    fig = px.scatter_matrix(
        plot_df,
        dimensions=metrics,
        color=color,
        title='Cross-Model Performance Correlation Matrix',
        labels={col: col.replace('_', ' ').title() for col in metrics},
        opacity=0.7,
        height=800,
        width=1000
    )
    
    # Update traces for better visibility
    fig.update_traces(
        diagonal_visible=True,
        showupperhalf=True,
        marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey'))
    )
    
    # Update layout
    fig.update_layout(
        font=PLOTLY_FONT,
        title=dict(
            text='Cross-Model Performance Correlation Matrix',
            font=dict(size=18, family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        hovermode='closest',
        plot_bgcolor='rgba(240,240,240,0.5)'
    )
    
    # Save if output path provided
    if output_path:
        if str(output_path).endswith('.html'):
            fig.write_html(output_path)
        else:
            fig.write_html(str(output_path) + '.html')
        print(f"[SAVED] {output_path}")
    
    return fig


def plot_cross_model_3d_scatter_plotly(df: pd.DataFrame,
                                        metrics: Optional[List[str]] = None,
                                        color: str = 'model',
                                        output_path: Optional[str] = None):
    """
    Create an interactive 3D scatter plot for cross-model metrics comparison

    Args:
        df: DataFrame with metrics columns and model identifier
        metrics: List of 3 metric column names to include (default: ['success_rate', 'avg_reward', 'avg_length'])
        color: Column name to use for color coding (default: 'model')
        output_path: Path to save figure (optional)

    Returns:
        plotly Figure object
    """
    # Default metrics if not provided
    if metrics is None:
        metrics = ['success_rate', 'avg_reward', 'avg_length']

    # Ensure we have exactly 3 metrics for 3D plot
    if len(metrics) < 3:
        metrics = metrics + ['avg_length'] * (3 - len(metrics))
    metrics = metrics[:3]

    # Ensure color column exists
    if color not in df.columns:
        raise ValueError(f"Color column '{color}' not found in DataFrame")

    # Create 3D scatter plot
    fig = px.scatter_3d(
        df,
        x=metrics[0],
        y=metrics[1],
        z=metrics[2],
        color=color,
        title='Cross-Model 3D Performance Comparison',
        labels={
            metrics[0]: metrics[0].replace('_', ' ').title(),
            metrics[1]: metrics[1].replace('_', ' ').title(),
            metrics[2]: metrics[2].replace('_', ' ').title()
        },
        opacity=0.7,
        height=700,
        width=900
    )

    # Update traces for better visibility
    fig.update_traces(
        marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey'))
    )

    # Update layout
    fig.update_layout(
        font=PLOTLY_FONT,
        title=dict(
            text='Cross-Model 3D Performance Comparison',
            font=dict(size=18),
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title=metrics[0].replace('_', ' ').title(),
            yaxis_title=metrics[1].replace('_', ' ').title(),
            zaxis_title=metrics[2].replace('_', ' ').title(),
            bgcolor='rgba(240,240,240,0.5)'
        ),
        legend=dict(title=color.replace('_', ' ').title())
    )
    
    # Save if output path provided
    if output_path:
        if str(output_path).endswith('.html'):
            fig.write_html(output_path)
        else:
            fig.write_html(str(output_path) + '.html')
        print(f"[SAVED] {output_path}")
    
    return fig


if __name__ == "__main__":
    # Example usage
    print("\n" + "="*80)
    print("CROSS-MODEL PAIRPLOT MATRIX (Plotly)")
    print("="*80 + "\n")
    
    # Create sample data for testing
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
    fig = plot_cross_model_pairplot_plotly(df, output_path=output_dir / 'cross_model_pairplot_interactive.html')
    
    print("\n[COMPLETE] Cross-model pairplot generated")
    print("="*80 + "\n")

