"""
Individual Model Performance Parallel Coordinates - Plotly Express
Interactive parallel coordinates plot for model performance comparison
"""

import pandas as pd
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


def plot_performance_radar_plotly(df: pd.DataFrame,
                                  model_name: str,
                                  output_path: Optional[str] = None):
    """
    Plot interactive performance parallel coordinates chart using Plotly Express

    Args:
        df: DataFrame with columns ['reward', 'success', 'steps', 'model_name']
        model_name: Name of the model to plot
        output_path: Path to save figure (optional)
    """
    # Filter data for this model
    model_data = df[df['model_name'] == model_name]

    # Calculate metrics
    avg_reward = model_data['reward'].mean()
    success_rate = model_data['success'].mean() * 100
    avg_steps = model_data['steps'].mean()

    # Normalize reward (assuming range -100 to 200)
    norm_reward = min(100, max(0, (avg_reward + 100) / 3))

    # Efficiency (inverse of steps - fewer steps is better)
    efficiency = max(0, 100 - avg_steps * 2) if avg_steps < 50 else 0

    # Create DataFrame for parallel coordinates
    metrics_df = pd.DataFrame({
        'Model': [model_name],
        'Success Rate (%)': [success_rate],
        'Reward Score': [norm_reward],
        'Efficiency Score': [efficiency],
        'Avg Steps': [avg_steps],
        'color_value': [75]  # For color mapping
    })

    # Create parallel coordinates plot using Plotly Express
    fig = px.parallel_coordinates(
        metrics_df,
        dimensions=['Success Rate (%)', 'Reward Score', 'Efficiency Score', 'Avg Steps'],
        color='color_value',
        labels={
            'Success Rate (%)': 'Success<br>Rate (%)',
            'Reward Score': 'Reward<br>Score',
            'Efficiency Score': 'Efficiency<br>Score',
            'Avg Steps': 'Avg Steps<br>(lower=better)'
        },
        color_continuous_scale=px.colors.sequential.Blues,
        color_continuous_midpoint=50,
        range_color=[0, 100]
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Performance Metrics - {model_name}',
            font=dict(size=18, family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        width=1000,
        height=600,
        template='plotly_white',
        margin=dict(l=80, r=80, t=100, b=80),
        font=dict(size=12, family='Arial'),
        coloraxis_showscale=False  # Hide color scale since we only have one model
    )

    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"✅ Saved: {output_path}")

    return fig


def create_performance_radar(experiment_name, output_dir):
    """Legacy function - Create interactive performance radar chart"""
    # Load results
    results_path = Path(__file__).parent.parent.parent / "results" / experiment_name / "comparison_results.csv"
    if not results_path.exists():
        print(f"[ERROR] Results not found: {results_path}")
        return

    df = pd.read_csv(results_path)

    # Calculate metrics per model
    metrics = df.groupby('model').agg({
        'success_rate': 'mean',
        'reward': 'mean',
        'steps': 'mean',
        'time': 'mean'
    }).reset_index()

    # Normalize metrics (0-100 scale)
    for col in ['success_rate', 'reward', 'steps', 'time']:
        if col in ['steps', 'time']:  # Lower is better
            metrics[f'{col}_norm'] = 100 - ((metrics[col] - metrics[col].min()) / 
                                            (metrics[col].max() - metrics[col].min() + 1e-6) * 100)
        else:  # Higher is better
            metrics[f'{col}_norm'] = (metrics[col] - metrics[col].min()) / \
                                     (metrics[col].max() - metrics[col].min() + 1e-6) * 100
    
    # Create radar chart
    fig = go.Figure()
    
    categories = ['Success Rate', 'Reward', 'Efficiency<br>(Steps)', 'Speed<br>(Time)']
    
    for _, row in metrics.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['success_rate_norm'], row['reward_norm'], 
               row['steps_norm'], row['time_norm']],
            theta=categories,
            fill='toself',
            name=row['model'],
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         '%{theta}: %{r:.1f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix='%'
            )
        ),
        showlegend=True,
        title=f'Model Performance Radar - {experiment_name.upper()}',
        height=700,
        font=dict(size=14),
        legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.1)
    )
    
    # Save
    output_path = output_dir / f'{experiment_name}_performance_radar_interactive.html'
    fig.write_html(output_path)
    print(f"[SAVED] {output_path.name}")

def main():
    """Generate for both experiments"""
    print("\n" + "="*80)
    print("INDIVIDUAL MODEL PERFORMANCE RADAR (Plotly)")
    print("="*80 + "\n")
    
    for exp in ['exp1', 'exp2']:
        output_dir = Path(__file__).parent.parent.parent / "results" / exp / "visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        create_performance_radar(exp, output_dir)
    
    print("\n[COMPLETE] Performance radar visualizations generated")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

