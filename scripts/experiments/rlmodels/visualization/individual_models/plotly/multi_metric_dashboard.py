"""
Multi-Metric Dashboard - Plotly
Interactive dashboard with multiple metrics
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional
import numpy as np

# Plotly font configuration for Linux Libertine
PLOTLY_FONT = dict(
    family="Linux Libertine, serif",
    size=12,
    color="black"
)


def plot_multi_metric_dashboard_plotly(df: pd.DataFrame,
                                       model_name: str,
                                       output_path: Optional[str] = None):
    """
    Plot interactive multi-metric dashboard

    Args:
        df: DataFrame with columns ['reward', 'success', 'steps', 'episode', 'model_name']
        model_name: Name of the model to plot
        output_path: Path to save figure (optional)
    """
    # Filter data for this model
    model_data = df[df['model_name'] == model_name]

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Episode Rewards',
            'Success Rate Over Time',
            'Steps per Episode',
            'Reward Distribution'
        ),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'histogram'}]]
    )

    # 1. Episode Rewards
    fig.add_trace(
        go.Scatter(
            x=model_data['episode'],
            y=model_data['reward'],
            mode='lines+markers',
            name='Reward',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )

    # 2. Success Rate Over Time (rolling average)
    window = min(5, len(model_data))
    if window > 1:
        rolling_success = model_data['success'].rolling(window=window, min_periods=1).mean() * 100
    else:
        rolling_success = model_data['success'] * 100

    fig.add_trace(
        go.Scatter(
            x=model_data['episode'],
            y=rolling_success,
            mode='lines+markers',
            name='Success Rate',
            line=dict(color='#A23B72', width=2),
            marker=dict(size=6)
        ),
        row=1, col=2
    )

    # 3. Steps per Episode
    fig.add_trace(
        go.Scatter(
            x=model_data['episode'],
            y=model_data['steps'],
            mode='lines+markers',
            name='Steps',
            line=dict(color='#F18F01', width=2),
            marker=dict(size=6)
        ),
        row=2, col=1
    )

    # 4. Reward Distribution
    fig.add_trace(
        go.Histogram(
            x=model_data['reward'],
            nbinsx=15,
            name='Reward Dist',
            marker_color='#06A77D'
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Multi-Metric Dashboard - {model_name}',
            font=dict(size=20, family='Arial Black')
        ),
        height=800,
        width=1400,
        showlegend=False,
        template='plotly_white'
    )

    # Update axes
    fig.update_xaxes(title_text="Episode", row=1, col=1)
    fig.update_yaxes(title_text="Reward", row=1, col=1)

    fig.update_xaxes(title_text="Episode", row=1, col=2)
    fig.update_yaxes(title_text="Success Rate (%)", row=1, col=2)

    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_yaxes(title_text="Steps", row=2, col=1)

    fig.update_xaxes(title_text="Reward", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)

    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"✅ Saved: {output_path}")

    return fig


def create_multi_metric_dashboard(experiment_name, output_dir):
    """Legacy function - Create comprehensive multi-metric dashboard"""
    results_path = Path(__file__).parent.parent.parent / "results" / experiment_name / "comparison_results.csv"
    if not results_path.exists():
        print(f"[ERROR] Results not found: {results_path}")
        return

    df = pd.read_csv(results_path)

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Success Rate', 'Reward Distribution',
                       'Steps per Episode', 'Inference Time'),
        specs=[[{'type': 'bar'}, {'type': 'box'}],
               [{'type': 'bar'}, {'type': 'scatter'}]]
    )

    # 1. Success Rate
    success_rates = df.groupby('model')['success_rate'].mean().sort_values(ascending=False)
    fig.add_trace(
        go.Bar(x=success_rates.index, y=success_rates.values, 
               name='Success Rate', marker_color='steelblue',
               hovertemplate='%{x}<br>Success Rate: %{y:.1f}%<extra></extra>'),
        row=1, col=1
    )
    
    # 2. Reward Distribution
    for model in df['model'].unique():
        model_data = df[df['model'] == model]['reward']
        fig.add_trace(
            go.Box(y=model_data, name=model, showlegend=False),
            row=1, col=2
        )
    
    # 3. Average Steps
    avg_steps = df.groupby('model')['steps'].mean().sort_values()
    fig.add_trace(
        go.Bar(x=avg_steps.index, y=avg_steps.values,
               name='Avg Steps', marker_color='coral',
               hovertemplate='%{x}<br>Avg Steps: %{y:.1f}<extra></extra>'),
        row=2, col=1
    )
    
    # 4. Steps vs Reward Scatter
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        fig.add_trace(
            go.Scatter(x=model_data['steps'], y=model_data['reward'],
                      mode='markers', name=model, showlegend=True,
                      hovertemplate='%{fullData.name}<br>Steps: %{x}<br>Reward: %{y:.2f}<extra></extra>'),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=900,
        title_text=f"Multi-Metric Dashboard - {experiment_name.upper()}",
        showlegend=True,
        font=dict(size=12)
    )
    
    fig.update_xaxes(tickangle=45, row=1, col=1)
    fig.update_xaxes(tickangle=45, row=2, col=1)
    fig.update_yaxes(title_text="Success Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Reward", row=1, col=2)
    fig.update_yaxes(title_text="Steps", row=2, col=1)
    fig.update_xaxes(title_text="Steps", row=2, col=2)
    fig.update_yaxes(title_text="Reward", row=2, col=2)
    
    output_path = output_dir / f'{experiment_name}_multi_metric_dashboard_interactive.html'
    fig.write_html(output_path)
    print(f"[SAVED] {output_path.name}")

def main():
    """Generate for both experiments"""
    print("\n" + "="*80)
    print("MULTI-METRIC DASHBOARD (Plotly)")
    print("="*80 + "\n")
    
    for exp in ['exp1', 'exp2']:
        output_dir = Path(__file__).parent.parent.parent / "results" / exp / "visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        create_multi_metric_dashboard(exp, output_dir)
    
    print("\n[COMPLETE] Multi-metric dashboard visualizations generated")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

