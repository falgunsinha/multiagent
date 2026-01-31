"""
Success Rate Over Time - Plotly Version
Interactive line plot with confidence intervals
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import Optional

# Plotly font configuration for Linux Libertine
PLOTLY_FONT = dict(
    family="Linux Libertine, serif",
    size=12,
    color="black"
)



def plot_success_rate_plotly(df: pd.DataFrame, 
                             model_name: str,
                             output_path: Optional[str] = None,
                             window_size: int = 10):
    """
    Plot interactive success rate over episodes
    
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
    
    # Create figure
    fig = go.Figure()
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=model_data['episode'].tolist() + model_data['episode'].tolist()[::-1],
        y=model_data['ci_upper'].tolist() + model_data['ci_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(46, 134, 171, 0.3)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% CI',
        showlegend=True
    ))
    
    # Add success rate line
    fig.add_trace(go.Scatter(
        x=model_data['episode'],
        y=model_data['success_rate'],
        mode='lines',
        name=f'{model_name} Success Rate',
        line=dict(color='#2E86AB', width=3)
    ))
    
    # Add 50% baseline
    fig.add_hline(y=50, line_dash="dash", line_color="red", 
                 annotation_text="50% Baseline", annotation_position="right")
    
    # Update layout with Linux Libertine font
    fig.update_layout(
        title=dict(
            text=f'Success Rate Over Time - {model_name}',
            font=dict(size=18, family='Linux Libertine, serif')
        ),
        xaxis_title='Episode',
        yaxis_title='Success Rate (%)',
        yaxis=dict(range=[0, 105]),
        hovermode='x unified',
        template='plotly_white',
        width=1200,
        height=600,
        font=PLOTLY_FONT
    )
    
    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"✅ Saved: {output_path}")
    
    return fig


if __name__ == "__main__":
    np.random.seed(42)
    episodes = 100
    
    data = {
        'episode': range(episodes),
        'success': np.random.binomial(1, 0.6, episodes),
        'model_name': ['SAC-Discrete'] * episodes
    }
    
    df = pd.DataFrame(data)
    
    fig = plot_success_rate_plotly(df, 'SAC-Discrete', 'test_success_rate_plotly.html')
    fig.show()

