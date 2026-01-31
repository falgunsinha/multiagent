"""
Success Rate Over Time - Plotly
Interactive success rate progression
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Plotly font configuration for Linux Libertine
PLOTLY_FONT = dict(
    family="Linux Libertine, serif",
    size=12,
    color="black"
)


def create_success_rate_over_time(experiment_name, output_dir):
    """Create interactive success rate over time chart"""
    results_path = Path(__file__).parent.parent.parent / "results" / experiment_name / "comparison_results.csv"
    if not results_path.exists():
        print(f"[ERROR] Results not found: {results_path}")
        return
    
    df = pd.read_csv(results_path)
    
    fig = go.Figure()
    
    # Calculate rolling average for each model
    window_size = 10
    for model in df['model'].unique():
        model_data = df[df['model'] == model].sort_values('episode')
        rolling_success = model_data['success_rate'].rolling(window=window_size, min_periods=1).mean()
        
        fig.add_trace(go.Scatter(
            x=model_data['episode'],
            y=rolling_success,
            mode='lines',
            name=model,
            line=dict(width=2),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Episode: %{x}<br>' +
                         'Success Rate: %{y:.1f}%<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'Success Rate Over Time - {experiment_name.upper()}',
        xaxis_title='Episode',
        yaxis_title='Success Rate (%) - Rolling Average',
        height=600,
        font=dict(size=14),
        hovermode='x unified',
        yaxis=dict(range=[0, 100])
    )
    
    output_path = output_dir / f'{experiment_name}_success_rate_over_time_interactive.html'
    fig.write_html(output_path)
    print(f"[SAVED] {output_path.name}")

def main():
    """Generate for both experiments"""
    print("\n" + "="*80)
    print("SUCCESS RATE OVER TIME (Plotly)")
    print("="*80 + "\n")
    
    for exp in ['exp1', 'exp2']:
        output_dir = Path(__file__).parent.parent.parent / "results" / exp / "visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        create_success_rate_over_time(exp, output_dir)
    
    print("\n[COMPLETE] Success rate over time visualizations generated")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

