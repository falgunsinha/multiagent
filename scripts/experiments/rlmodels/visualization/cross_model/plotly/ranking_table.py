"""
Ranking Table - Plotly
Interactive ranking table for all models
"""

import pandas as pd
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

def plot_distribution_distplot_plotly(df):
    """Create interactive distribution plot with histogram + KDE - returns figure for W&B"""
    import plotly.figure_factory as ff

    metrics = ['avg_reward', 'success_rate', 'avg_length']
    models = df['model'].unique()

    # Create distplot for reward (most important metric)
    hist_data = []
    group_labels = []

    for model in models:
        model_data = df[df['model'] == model]['avg_reward'].dropna().values
        if len(model_data) > 1:  # Need at least 2 points for KDE
            hist_data.append(model_data)
            group_labels.append(model)

    if len(hist_data) == 0:
        # Fallback: create simple histogram
        import plotly.express as px
        fig = px.histogram(df, x='avg_reward', color='model', barmode='overlay',
                          title='Reward Distribution Comparison')
        return fig

    # Create distplot with histogram + KDE
    try:
        fig = ff.create_distplot(
            hist_data,
            group_labels,
            bin_size='auto',
            show_hist=True,
            show_curve=True,
            show_rug=False,  # Disable rug for cleaner look
            histnorm='probability density',
            curve_type='kde'
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text='Reward Distribution - Histogram + KDE',
                font=dict(size=16, family='Linux Libertine, serif')
            ),
            xaxis_title='Reward',
            yaxis_title='Density',
            height=500,
            font=PLOTLY_FONT,
            hovermode='x unified',
            legend=dict(title='Model', orientation='v')
        )
    except:
        # Fallback if distplot fails
        import plotly.express as px
        fig = px.histogram(df, x='avg_reward', color='model', barmode='overlay',
                          title='Reward Distribution Comparison')

    return fig

def create_ranking_table(df, output_dir):
    """Create interactive distribution comparison using ff.create_distplot"""
    import plotly.figure_factory as ff
    from plotly.subplots import make_subplots

    metrics = ['reward', 'success_rate', 'steps', 'time']
    metric_labels = ['Reward Distribution', 'Success Rate Distribution',
                     'Steps Distribution', 'Inference Time Distribution']

    models = df['model'].unique()

    # Create a 2x2 subplot for all metrics
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # We'll create individual distplots and combine them
    for metric_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        # Prepare data for this metric
        hist_data = []
        group_labels = []

        for model in models:
            model_data = df[df['model'] == model][metric].dropna().values
            if len(model_data) > 0:
                hist_data.append(model_data)
                group_labels.append(model)

        # Create distplot with histogram + KDE + rug plot
        fig = ff.create_distplot(
            hist_data,
            group_labels,
            bin_size='auto',
            show_hist=True,
            show_curve=True,
            show_rug=True,
            histnorm='probability density',
            curve_type='kde',
            colors=None,  # Use default colors
            rug_text=None
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{label} - Histogram + KDE + Rug Plot',
                font=dict(size=18, family='Arial Black')
            ),
            xaxis_title=metric.replace('_', ' ').title(),
            yaxis_title='Density',
            height=600,
            width=1200,
            template='plotly_white',
            font=dict(size=12),
            hovermode='x unified',
            legend=dict(
                title='Model',
                orientation='v',
                yanchor='top',
                y=1,
                xanchor='left',
                x=1.02
            )
        )

        # Save individual metric plot
        filename = f'distribution_{metric}_distplot.html'
        fig.write_html(output_dir / filename)
        print(f"[SAVED] {filename}")

    # Also create a combined view with all 4 metrics in subplots
    # Note: ff.create_distplot doesn't work well with subplots, so we'll create a simpler combined view
    fig_combined = make_subplots(
        rows=2, cols=2,
        subplot_titles=metric_labels,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    import plotly.express as px
    colors = px.colors.qualitative.Plotly

    for idx, metric in enumerate(metrics):
        row = idx // 2 + 1
        col = idx % 2 + 1

        for model_idx, model in enumerate(models):
            model_data = df[df['model'] == model][metric].dropna()

            # Add histogram
            fig_combined.add_trace(
                go.Histogram(
                    x=model_data,
                    name=model,
                    marker_color=colors[model_idx % len(colors)],
                    opacity=0.6,
                    showlegend=(idx == 0),
                    legendgroup=model,
                    histnorm='probability density',
                    nbinsx=25,
                    hovertemplate=f'<b>{model}</b><br>Value: %{{x}}<br>Density: %{{y}}<extra></extra>'
                ),
                row=row, col=col
            )

        fig_combined.update_xaxes(title_text=metric.replace('_', ' ').title(), row=row, col=col)
        fig_combined.update_yaxes(title_text='Density', row=row, col=col)

    fig_combined.update_layout(
        title=dict(
            text='Cross-Model Distribution Comparison - All Metrics',
            font=dict(size=20, family='Arial Black')
        ),
        barmode='overlay',
        height=900,
        width=1400,
        template='plotly_white',
        font=dict(size=12),
        hovermode='closest'
    )

    fig_combined.write_html(output_dir / 'distribution_all_metrics_combined.html')
    print(f"[SAVED] distribution_all_metrics_combined.html")

def main():
    print("\n" + "="*80)
    print("DISTRIBUTION COMPARISON (Plotly)")
    print("="*80 + "\n")
    
    df = load_all_results()
    if df is None:
        print("[ERROR] No results found!")
        return
    
    output_dir = Path(__file__).parent.parent.parent / "results" / "cross_model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    create_ranking_table(df, output_dir)
    
    print("\n[COMPLETE] Ranking table generated")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

