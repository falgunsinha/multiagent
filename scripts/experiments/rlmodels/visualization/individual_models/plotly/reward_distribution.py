"""
Individual Model Reward Distribution - Plotly
Interactive reward distribution visualization
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Optional
import numpy as np

# Plotly font configuration for Linux Libertine
PLOTLY_FONT = dict(
    family="Linux Libertine, serif",
    size=12,
    color="black"
)


def plot_reward_distribution_plotly(df: pd.DataFrame,
                                    model_name: str,
                                    output_path: Optional[str] = None):
    """
    Plot interactive reward distribution

    Args:
        df: DataFrame with columns ['reward', 'model_name']
        model_name: Name of the model to plot
        output_path: Path to save figure (optional)
    """
    # Filter data for this model
    model_data = df[df['model_name'] == model_name]

    # Create figure with histogram
    fig = go.Figure()

    # Add histogram
    fig.add_trace(go.Histogram(
        x=model_data['reward'],
        nbinsx=20,
        name='Rewards',
        marker_color='#A23B72',
        opacity=0.7
    ))

    # Add mean line
    mean_reward = model_data['reward'].mean()
    fig.add_vline(x=mean_reward, line_dash="dash", line_color="red",
                 annotation_text=f"Mean: {mean_reward:.2f}",
                 annotation_position="top right")

    # Add median line
    median_reward = model_data['reward'].median()
    fig.add_vline(x=median_reward, line_dash="dash", line_color="green",
                 annotation_text=f"Median: {median_reward:.2f}",
                 annotation_position="bottom right")

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Reward Distribution - {model_name}',
            font=dict(size=18, family='Arial Black')
        ),
        xaxis_title='Reward',
        yaxis_title='Frequency',
        hovermode='x unified',
        template='plotly_white',
        width=1000,
        height=600,
        font=dict(size=12)
    )

    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"✅ Saved: {output_path}")

    return fig


def create_reward_distribution(experiment_name, output_dir):
    """Legacy function - Create interactive reward distribution chart"""
    # Load results
    results_path = Path(__file__).parent.parent.parent / "results" / experiment_name / "comparison_results.csv"
    if not results_path.exists():
        print(f"[ERROR] Results not found: {results_path}")
        return

    df = pd.read_csv(results_path)

    # Create box plot
    fig = px.box(df, x='model', y='reward', color='model',
                 title=f'Reward Distribution - {experiment_name.upper()}',
                 labels={'reward': 'Reward', 'model': 'Model'},
                 points='all')

    fig.update_layout(
        height=600,
        font=dict(size=14),
        showlegend=False,
        hovermode='closest'
    )

    fig.update_xaxes(tickangle=45)

    # Save
    output_path = output_dir / f'{experiment_name}_reward_distribution_interactive.html'
    fig.write_html(output_path)
    print(f"[SAVED] {output_path.name}")

def main():
    """Generate for both experiments"""
    print("\n" + "="*80)
    print("INDIVIDUAL MODEL REWARD DISTRIBUTION (Plotly)")
    print("="*80 + "\n")
    
    for exp in ['exp1', 'exp2']:
        output_dir = Path(__file__).parent.parent.parent / "results" / exp / "visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        create_reward_distribution(exp, output_dir)
    
    print("\n[COMPLETE] Reward distribution visualizations generated")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

