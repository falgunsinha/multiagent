"""
Success Rate Comparison - Plotly
Interactive success rate comparison across all models
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

def plot_success_rate_line_plotly(df):
    """Create interactive success rate line chart with confidence bands - returns figure for W&B"""
    import numpy as np

    # Check if we have 'episode' column, if not create it
    if 'episode' not in df.columns:
        df = df.copy()
        df['episode'] = df.groupby('model').cumcount() + 1

    fig = go.Figure()

    # Get unique models
    models = df['model'].unique()
    colors = px.colors.qualitative.Plotly

    for idx, model in enumerate(models):
        model_data = df[df['model'] == model].sort_values('episode')

        # Calculate mean and std for confidence bands
        grouped = model_data.groupby('episode')['success_rate'].agg(['mean', 'std', 'count']).reset_index()

        # Calculate 95% confidence interval
        confidence = 0.95
        z_score = 1.96  # for 95% CI
        grouped['ci'] = z_score * (grouped['std'] / np.sqrt(grouped['count']))
        grouped['upper'] = grouped['mean'] + grouped['ci']
        grouped['lower'] = grouped['mean'] - grouped['ci']

        color = colors[idx % len(colors)]

        # Add confidence band
        fig.add_trace(go.Scatter(
            x=grouped['episode'].tolist() + grouped['episode'].tolist()[::-1],
            y=grouped['upper'].tolist() + grouped['lower'].tolist()[::-1],
            fill='toself',
            fillcolor=color,
            opacity=0.2,
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name=model,
            hoverinfo='skip'
        ))

        # Add mean line
        fig.add_trace(go.Scatter(
            x=grouped['episode'],
            y=grouped['mean'],
            mode='lines',
            name=model,
            line=dict(color=color, width=3),
            hovertemplate='%{fullData.name}<br>Episode: %{x}<br>Success Rate: %{y:.1f}%<extra></extra>'
        ))

    fig.update_layout(
        title='Success Rate Over Episodes - All Models',
        xaxis_title='Episode',
        yaxis_title='Success Rate (%)',
        height=600,
        font=PLOTLY_FONT,
        hovermode='x unified',
        yaxis=dict(range=[0, 100]),
        legend=dict(title='Model', orientation='v')
    )

    return fig

def create_success_rate_comparison(df, output_dir):
    """Create interactive success rate comparison"""
    success_rates = df.groupby(['model', 'experiment'])['success_rate'].mean().reset_index()
    
    fig = px.bar(success_rates, x='model', y='success_rate', color='experiment',
                 barmode='group', title='Success Rate Comparison - All Models',
                 labels={'success_rate': 'Success Rate (%)', 'model': 'Model'},
                 color_discrete_map={'Discrete': '#2E86AB', 'Continuous': '#A23B72'})
    
    fig.update_layout(
        height=600,
        font=dict(size=14),
        hovermode='x unified',
        yaxis=dict(range=[0, 100]),
        xaxis_tickangle=-45
    )
    
    fig.write_html(output_dir / 'success_rate_comparison_interactive.html')
    print(f"[SAVED] success_rate_comparison_interactive.html")

def main():
    print("\n" + "="*80)
    print("SUCCESS RATE COMPARISON (Plotly)")
    print("="*80 + "\n")
    
    df = load_all_results()
    if df is None:
        print("[ERROR] No results found!")
        return
    
    output_dir = Path(__file__).parent.parent.parent / "results" / "cross_model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    create_success_rate_comparison(df, output_dir)
    
    print("\n[COMPLETE] Success rate comparison generated")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

