"""
Individual Model Learning Curves - Plotly
Interactive learning curves visualization
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


def create_learning_curves(experiment_name, output_dir):
    """Create interactive learning curves"""
    # Load results
    results_path = Path(__file__).parent.parent.parent / "results" / experiment_name / "comparison_results.csv"
    if not results_path.exists():
        print(f"[ERROR] Results not found: {results_path}")
        return
    
    df = pd.read_csv(results_path)
    
    # Create line plot
    fig = px.line(df, x='episode', y='reward', color='model',
                  title=f'Learning Curves - {experiment_name.upper()}',
                  labels={'reward': 'Reward', 'episode': 'Episode', 'model': 'Model'},
                  line_shape='spline')
    
    fig.update_layout(
        height=600,
        font=dict(size=14),
        hovermode='x unified',
        legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.02)
    )
    
    # Save
    output_path = output_dir / f'{experiment_name}_learning_curves_interactive.html'
    fig.write_html(output_path)
    print(f"[SAVED] {output_path.name}")

def main():
    """Generate for both experiments"""
    print("\n" + "="*80)
    print("INDIVIDUAL MODEL LEARNING CURVES (Plotly)")
    print("="*80 + "\n")
    
    for exp in ['exp1', 'exp2']:
        output_dir = Path(__file__).parent.parent.parent / "results" / exp / "visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        create_learning_curves(exp, output_dir)
    
    print("\n[COMPLETE] Learning curves visualizations generated")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

