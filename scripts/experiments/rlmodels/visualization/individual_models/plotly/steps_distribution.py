"""
Steps Distribution - Plotly
Interactive steps distribution visualization
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Optional

# Plotly font configuration for Linux Libertine
PLOTLY_FONT = dict(
    family="Linux Libertine, serif",
    size=12,
    color="black"
)


def plot_steps_distribution_plotly(df: pd.DataFrame,
                                   model_name: str,
                                   output_path: Optional[str] = None):
    """
    Plot interactive steps distribution

    Args:
        df: DataFrame with columns ['steps', 'model_name']
        model_name: Name of the model to plot
        output_path: Path to save figure (optional)
    """
    # Filter data for this model
    model_data = df[df['model_name'] == model_name]

    # Create figure with box plot and histogram
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Steps Distribution', 'Steps Box Plot'),
        specs=[[{'type': 'histogram'}, {'type': 'box'}]]
    )

    # Add histogram
    fig.add_trace(
        go.Histogram(x=model_data['steps'], nbinsx=15,
                    marker_color='#2E86AB', name='Steps'),
        row=1, col=1
    )

    # Add box plot
    fig.add_trace(
        go.Box(y=model_data['steps'], marker_color='#F18F01',
               name='Steps', showlegend=False),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Steps Distribution - {model_name}',
            font=dict(size=18, family='Arial Black')
        ),
        height=500,
        width=1200,
        showlegend=False,
        template='plotly_white'
    )

    fig.update_xaxes(title_text="Steps", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Steps", row=1, col=2)

    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"✅ Saved: {output_path}")

    return fig


def create_steps_distribution(experiment_name, output_dir):
    """Legacy function - Create interactive steps distribution chart"""
    results_path = Path(__file__).parent.parent.parent / "results" / experiment_name / "comparison_results.csv"
    if not results_path.exists():
        print(f"[ERROR] Results not found: {results_path}")
        return

    df = pd.read_csv(results_path)

    # Create box plot
    fig = px.box(df, x='model', y='steps', color='model',
                 title=f'Steps Distribution - {experiment_name.upper()}',
                 labels={'steps': 'Steps per Episode', 'model': 'Model'},
                 points='all')

    fig.update_layout(
        height=600,
        font=dict(size=14),
        showlegend=False,
        hovermode='closest'
    )

    fig.update_xaxes(tickangle=45)

    output_path = output_dir / f'{experiment_name}_steps_distribution_interactive.html'
    fig.write_html(output_path)
    print(f"[SAVED] {output_path.name}")

def main():
    """Generate for both experiments"""
    print("\n" + "="*80)
    print("STEPS DISTRIBUTION (Plotly)")
    print("="*80 + "\n")
    
    for exp in ['exp1', 'exp2']:
        output_dir = Path(__file__).parent.parent.parent / "results" / exp / "visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        create_steps_distribution(exp, output_dir)
    
    print("\n[COMPLETE] Steps distribution visualizations generated")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

