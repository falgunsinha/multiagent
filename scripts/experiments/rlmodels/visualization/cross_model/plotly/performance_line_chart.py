"""
Performance Line Chart - Plotly
Interactive performance tracking over episodes
"""

import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
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

def plot_performance_line_plotly(df):
    """Create interactive performance line chart - returns figure for W&B"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Reward Over Episodes', 'Success Rate Over Episodes'),
        vertical_spacing=0.12
    )

    # Check if we have 'episode' column, if not create it
    if 'episode' not in df.columns:
        df = df.copy()
        df['episode'] = df.groupby('model').cumcount() + 1

    # Reward over episodes
    for model in df['model'].unique():
        model_data = df[df['model'] == model].sort_values('episode')
        fig.add_trace(
            go.Scatter(x=model_data['episode'], y=model_data['avg_reward'],
                      mode='lines', name=model, showlegend=True,
                      hovertemplate='%{fullData.name}<br>Episode: %{x}<br>Reward: %{y:.2f}<extra></extra>'),
            row=1, col=1
        )

    # Success rate over episodes
    for model in df['model'].unique():
        model_data = df[df['model'] == model].sort_values('episode')
        fig.add_trace(
            go.Scatter(x=model_data['episode'], y=model_data['success_rate'],
                      mode='lines', name=model, showlegend=False,
                      hovertemplate='%{fullData.name}<br>Episode: %{x}<br>Success: %{y:.1f}%<extra></extra>'),
            row=2, col=1
        )

    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Success Rate (%)", range=[0, 100], row=2, col=1)

    fig.update_layout(
        height=800,
        title_text="Performance Over Episodes - All Models",
        hovermode='x unified',
        font=PLOTLY_FONT
    )

    return fig

def create_performance_line_chart(df, output_dir):
    """Create interactive performance line chart"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Reward Over Episodes', 'Success Rate Over Episodes'),
        vertical_spacing=0.12
    )
    
    # Reward over episodes
    for model in df['model'].unique():
        model_data = df[df['model'] == model].sort_values('episode')
        fig.add_trace(
            go.Scatter(x=model_data['episode'], y=model_data['reward'],
                      mode='lines', name=model, showlegend=True,
                      hovertemplate='%{fullData.name}<br>Episode: %{x}<br>Reward: %{y:.2f}<extra></extra>'),
            row=1, col=1
        )
    
    # Success rate over episodes
    for model in df['model'].unique():
        model_data = df[df['model'] == model].sort_values('episode')
        fig.add_trace(
            go.Scatter(x=model_data['episode'], y=model_data['success_rate'],
                      mode='lines', name=model, showlegend=False,
                      hovertemplate='%{fullData.name}<br>Episode: %{x}<br>Success: %{y:.1f}%<extra></extra>'),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Success Rate (%)", range=[0, 100], row=2, col=1)
    
    fig.update_layout(
        height=900,
        title_text="Performance Over Episodes - All Models",
        hovermode='x unified',
        font=dict(size=12)
    )
    
    fig.write_html(output_dir / 'performance_line_chart_interactive.html')
    print(f"[SAVED] performance_line_chart_interactive.html")

def main():
    print("\n" + "="*80)
    print("PERFORMANCE LINE CHART (Plotly)")
    print("="*80 + "\n")
    
    df = load_all_results()
    if df is None:
        print("[ERROR] No results found!")
        return
    
    output_dir = Path(__file__).parent.parent.parent / "results" / "cross_model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    create_performance_line_chart(df, output_dir)
    
    print("\n[COMPLETE] Performance line chart generated")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

