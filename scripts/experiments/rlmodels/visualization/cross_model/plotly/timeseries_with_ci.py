"""
Time Series with Shaded Confidence Intervals - Plotly
Interactive version with prediction intervals
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional
import numpy as np

PLOTLY_FONT = dict(
    family="Linux Libertine, serif",
    size=12,
    color="black"
)


def plot_timeseries_with_ci_plotly(df: pd.DataFrame,
                                    x: str = 'episode',
                                    y: str = 'reward',
                                    hue: Optional[str] = 'model_name',
                                    output_path: Optional[str] = None):
    """
    Plot time series with shaded confidence interval (interactive)
    
    Args:
        df: DataFrame with time series data
        x: X-axis column (time/episode)
        y: Y-axis metric
        hue: Grouping column
        output_path: Save path
    """
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    if hue:
        groups = df[hue].unique()
        
        for idx, group in enumerate(groups):
            group_data = df[df[hue] == group].sort_values(x)
            
            # Calculate rolling statistics
            window = max(10, len(group_data) // 20)
            group_data['mean'] = group_data[y].rolling(window=window, min_periods=1).mean()
            group_data['std'] = group_data[y].rolling(window=window, min_periods=1).std()
            group_data['upper'] = group_data['mean'] + 1.96 * group_data['std']
            group_data['lower'] = group_data['mean'] - 1.96 * group_data['std']
            
            color = colors[idx % len(colors)]
            
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=pd.concat([group_data[x], group_data[x][::-1]]),
                y=pd.concat([group_data['upper'], group_data['lower'][::-1]]),
                fill='toself',
                fillcolor=color.replace(')', ', 0.2)').replace('rgb', 'rgba'),
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name=f'{group} CI',
                hoverinfo='skip'
            ))
            
            # Add mean line
            fig.add_trace(go.Scatter(
                x=group_data[x],
                y=group_data['mean'],
                mode='lines',
                name=group,
                line=dict(color=color, width=2.5)
            ))
    else:
        # Single series
        df_sorted = df.sort_values(x)
        window = max(10, len(df_sorted) // 20)
        df_sorted['mean'] = df_sorted[y].rolling(window=window, min_periods=1).mean()
        df_sorted['std'] = df_sorted[y].rolling(window=window, min_periods=1).std()
        df_sorted['upper'] = df_sorted['mean'] + 1.96 * df_sorted['std']
        df_sorted['lower'] = df_sorted['mean'] - 1.96 * df_sorted['std']
        
        # CI
        fig.add_trace(go.Scatter(
            x=pd.concat([df_sorted[x], df_sorted[x][::-1]]),
            y=pd.concat([df_sorted['upper'], df_sorted['lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(46, 134, 171, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name='95% CI'
        ))
        
        # Mean
        fig.add_trace(go.Scatter(
            x=df_sorted[x],
            y=df_sorted['mean'],
            mode='lines',
            name=y.title(),
            line=dict(color='#2E86AB', width=2.5)
        ))
    
    fig.update_layout(
        title=dict(
            text=f'{y.title()} Over Time with 95% CI',
            font=dict(size=18, family=PLOTLY_FONT['family'])
        ),
        xaxis=dict(title=dict(text=x.title(), font=PLOTLY_FONT)),
        yaxis=dict(title=dict(text=y.title(), font=PLOTLY_FONT)),
        template='plotly_white',
        font=PLOTLY_FONT,
        width=1200,
        height=600,
        hovermode='x unified'
    )
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"✅ Saved: {output_path}")
    
    return fig


def plot_prediction_intervals_plotly(df: pd.DataFrame,
                                      x: str = 'episode',
                                      y_true: str = 'reward',
                                      y_pred: str = 'predicted_reward',
                                      output_path: Optional[str] = None):
    """
    Plot predictions with intervals (like AirPassengers)
    
    Args:
        df: DataFrame with actual and predicted
        x: Time column
        y_true: Actual values
        y_pred: Predicted values
        output_path: Save path
    """
    fig = go.Figure()
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=df[x],
        y=df[y_true],
        mode='lines',
        name='Actual',
        line=dict(color='black', width=2)
    ))
    
    # Predicted values
    fig.add_trace(go.Scatter(
        x=df[x],
        y=df[y_pred],
        mode='lines',
        name='Predicted',
        line=dict(color='#2E86AB', width=2.5)
    ))
    
    # Prediction interval
    if 'pred_lower' in df.columns and 'pred_upper' in df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df[x], df[x][::-1]]),
            y=pd.concat([df['pred_upper'], df['pred_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(46, 134, 171, 0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            name='90% Prediction Interval'
        ))
    
    fig.update_layout(
        title=dict(
            text='Predictions with Confidence Intervals',
            font=dict(size=18, family=PLOTLY_FONT['family'])
        ),
        xaxis=dict(title=dict(text=x.title(), font=PLOTLY_FONT)),
        yaxis=dict(title=dict(text=y_true.title(), font=PLOTLY_FONT)),
        template='plotly_white',
        font=PLOTLY_FONT,
        width=1200,
        height=600
    )
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"✅ Saved: {output_path}")
    
    return fig

