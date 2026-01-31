"""
Success vs Inference Scatter - Plotly
Interactive scatter plot analyzing success vs inference time trade-off
"""

import pandas as pd
import plotly.express as px
from pathlib import Path

# Plotly font configuration for Linux Libertine
PLOTLY_FONT = dict(
    family="Linux Libertine, serif",
    size=12,
    color="black"
)


def load_all_results():
    base_path = Path(__file__).parent.parent.parent / "results"
    results = {}
    for exp in ['exp1', 'exp2']:
        csv_path = base_path / exp / "comparison_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['experiment'] = 'Discrete' if exp == 'exp1' else 'Continuous'
            results[exp] = df
    if not results:
        return None
    return pd.concat(results.values(), ignore_index=True)

def plot_success_vs_inference_plotly(df):
    """Create interactive success vs inference scatter - returns figure for W&B"""
    # Calculate average metrics per model
    metrics = df.groupby('model').agg({
        'success_rate': 'mean',
        'avg_duration': 'mean'
    }).reset_index()

    fig = px.scatter(metrics, x='avg_duration', y='success_rate', color='model',
                     size_max=15,
                     title='Success Rate vs Inference Time - All Models',
                     labels={'avg_duration': 'Average Inference Time (s)',
                            'success_rate': 'Success Rate (%)'},
                     hover_data=['model'])

    fig.update_traces(marker=dict(size=15, line=dict(width=2, color='DarkSlateGrey')))

    fig.update_layout(
        height=600,
        font=PLOTLY_FONT,
        yaxis=dict(range=[0, 100]),
        showlegend=True
    )

    return fig

def create_success_vs_inference_scatter(df, output_dir):
    """Create interactive success vs inference scatter"""
    # Calculate average metrics per model
    metrics = df.groupby(['model', 'experiment']).agg({
        'success_rate': 'mean',
        'time': 'mean'
    }).reset_index()
    
    fig = px.scatter(metrics, x='time', y='success_rate', color='model',
                     symbol='experiment', size_max=15,
                     title='Success Rate vs Inference Time - All Models',
                     labels={'time': 'Average Inference Time (s)', 
                            'success_rate': 'Success Rate (%)'},
                     hover_data=['model', 'experiment'])
    
    fig.update_traces(marker=dict(size=15, line=dict(width=2, color='DarkSlateGrey')))
    
    fig.update_layout(
        height=600,
        font=dict(size=14),
        yaxis=dict(range=[0, 100]),
        showlegend=True
    )
    
    fig.write_html(output_dir / 'success_vs_inference_scatter_interactive.html')
    print(f"[SAVED] success_vs_inference_scatter_interactive.html")

def main():
    print("\n" + "="*80)
    print("SUCCESS VS INFERENCE SCATTER (Plotly)")
    print("="*80 + "\n")
    
    df = load_all_results()
    if df is None:
        print("[ERROR] No results found!")
        return
    
    output_dir = Path(__file__).parent.parent.parent / "results" / "cross_model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    create_success_vs_inference_scatter(df, output_dir)
    
    print("\n[COMPLETE] Success vs inference scatter generated")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

