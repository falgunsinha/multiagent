"""
Cross-Model Interactive Comparison - Plotly
Interactive visualizations for comparing all models
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

# Plotly font configuration for Linux Libertine
PLOTLY_FONT = dict(
    family="Linux Libertine, serif",
    size=12,
    color="black"
)


def load_all_results():
    """Load results from both experiments"""
    base_path = Path(__file__).parent.parent.parent / "results"
    
    results = {}
    for exp in ['exp1', 'exp2']:
        csv_path = base_path / exp / "comparison_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['experiment'] = 'Discrete' if exp == 'exp1' else 'Continuous'
            results[exp] = df
            print(f"[INFO] Loaded {len(df)} results from {exp}")
    
    if not results:
        print("[ERROR] No results found!")
        return None
    
    combined = pd.concat(results.values(), ignore_index=True)
    return combined

def create_success_rate_chart(df, output_dir):
    """Interactive success rate comparison"""
    success_rates = df.groupby(['model', 'experiment'])['success_rate'].mean().reset_index()
    
    fig = px.bar(success_rates, x='model', y='success_rate', color='experiment',
                 barmode='group', title='Success Rate Comparison - All Models',
                 labels={'success_rate': 'Success Rate (%)', 'model': 'Model'},
                 color_discrete_map={'Discrete': '#2E86AB', 'Continuous': '#A23B72'})
    
    fig.update_layout(
        height=600,
        font=dict(size=14),
        hovermode='x unified',
        yaxis=dict(range=[0, 100])
    )
    
    fig.write_html(output_dir / 'interactive_success_rates.html')
    print(f"[SAVED] interactive_success_rates.html")

def create_reward_distribution(df, output_dir):
    """Interactive reward distribution"""
    fig = px.box(df, x='model', y='reward', color='experiment',
                 title='Reward Distribution - All Models',
                 labels={'reward': 'Reward', 'model': 'Model'},
                 color_discrete_map={'Discrete': '#2E86AB', 'Continuous': '#A23B72'})
    
    fig.update_layout(
        height=600,
        font=dict(size=14),
        hovermode='x unified'
    )
    
    fig.write_html(output_dir / 'interactive_reward_distribution.html')
    print(f"[SAVED] interactive_reward_distribution.html")

def create_performance_radar(df, output_dir):
    """Interactive radar chart for model performance"""
    # Calculate normalized metrics
    metrics = df.groupby('model').agg({
        'success_rate': 'mean',
        'reward': 'mean',
        'steps': 'mean',
        'time': 'mean'
    }).reset_index()
    
    # Normalize (0-100 scale)
    for col in ['success_rate', 'reward', 'steps', 'time']:
        if col == 'steps' or col == 'time':  # Lower is better
            metrics[f'{col}_norm'] = 100 - ((metrics[col] - metrics[col].min()) / 
                                            (metrics[col].max() - metrics[col].min()) * 100)
        else:  # Higher is better
            metrics[f'{col}_norm'] = (metrics[col] - metrics[col].min()) / \
                                     (metrics[col].max() - metrics[col].min()) * 100
    
    fig = go.Figure()
    
    categories = ['Success Rate', 'Reward', 'Efficiency (Steps)', 'Speed (Time)']
    
    for _, row in metrics.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['success_rate_norm'], row['reward_norm'], 
               row['steps_norm'], row['time_norm']],
            theta=categories,
            fill='toself',
            name=row['model']
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title='Model Performance Radar Chart',
        height=700
    )
    
    fig.write_html(output_dir / 'interactive_performance_radar.html')
    print(f"[SAVED] interactive_performance_radar.html")

def plot_parallel_coordinates_plotly(df):
    """Create parallel coordinates plot - returns figure for W&B"""
    # Calculate average metrics per model
    metrics = df.groupby(['model']).agg({
        'success_rate': 'mean',
        'avg_reward': 'mean',
        'avg_length': 'mean'
    }).reset_index()

    # Create a numeric ID for coloring
    metrics['model_id'] = pd.factorize(metrics['model'])[0]

    # Create parallel coordinates plot
    fig = px.parallel_coordinates(
        metrics,
        dimensions=['success_rate', 'avg_reward', 'avg_length'],
        color='model_id',
        labels={
            'success_rate': 'Success Rate (%)',
            'avg_reward': 'Avg Reward',
            'avg_length': 'Avg Steps',
            'model_id': 'Model'
        },
        color_continuous_scale=px.colors.diverging.Tealrose,
        color_continuous_midpoint=metrics['model_id'].median(),
        title='Cross-Model Performance - Parallel Coordinates'
    )

    # Update layout
    fig.update_layout(
        height=500,
        font=PLOTLY_FONT,
        title=dict(
            text='Cross-Model Performance - Parallel Coordinates',
            font=dict(size=16, family='Linux Libertine, serif')
        )
    )

    return fig

def create_parallel_coordinates(df, output_dir):
    """Create parallel coordinates plot for multi-dimensional comparison"""
    # Calculate average metrics per model
    metrics = df.groupby(['model', 'experiment']).agg({
        'success_rate': 'mean',
        'reward': 'mean',
        'steps': 'mean',
        'time': 'mean'
    }).reset_index()

    # Create a numeric ID for coloring
    metrics['model_id'] = pd.factorize(metrics['model'])[0]

    # Create parallel coordinates plot
    fig = px.parallel_coordinates(
        metrics,
        dimensions=['success_rate', 'reward', 'steps', 'time'],
        color='model_id',
        labels={
            'success_rate': 'Success Rate (%)',
            'reward': 'Avg Reward',
            'steps': 'Avg Steps',
            'time': 'Avg Time (s)',
            'model_id': 'Model'
        },
        color_continuous_scale=px.colors.diverging.Tealrose,
        color_continuous_midpoint=metrics['model_id'].median(),
        title='Cross-Model Performance - Parallel Coordinates'
    )

    # Update layout
    fig.update_layout(
        height=600,
        font=dict(size=12),
        title=dict(
            text='Cross-Model Performance - Parallel Coordinates',
            font=dict(size=18, family='Arial Black')
        )
    )

    fig.write_html(output_dir / 'interactive_parallel_coordinates.html')
    print(f"[SAVED] interactive_parallel_coordinates.html")

def create_learning_curves(df, output_dir):
    """Interactive learning curves"""
    fig = px.line(df, x='episode', y='reward', color='model',
                  facet_col='experiment',
                  title='Learning Curves - All Models',
                  labels={'reward': 'Reward', 'episode': 'Episode'})
    
    fig.update_layout(
        height=600,
        font=dict(size=14),
        hovermode='x unified'
    )
    
    fig.write_html(output_dir / 'interactive_learning_curves.html')
    print(f"[SAVED] interactive_learning_curves.html")

def create_dashboard(df, output_dir):
    """Create comprehensive dashboard"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Success Rate by Model', 'Reward Distribution',
                       'Steps per Episode', 'Inference Time'),
        specs=[[{'type': 'bar'}, {'type': 'box'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Success rates
    success_rates = df.groupby('model')['success_rate'].mean().sort_values(ascending=False)
    fig.add_trace(go.Bar(x=success_rates.index, y=success_rates.values, 
                         name='Success Rate', marker_color='steelblue'),
                  row=1, col=1)
    
    # Reward distribution
    for model in df['model'].unique():
        model_data = df[df['model'] == model]['reward']
        fig.add_trace(go.Box(y=model_data, name=model), row=1, col=2)
    
    # Steps
    avg_steps = df.groupby('model')['steps'].mean().sort_values()
    fig.add_trace(go.Bar(x=avg_steps.index, y=avg_steps.values,
                         name='Avg Steps', marker_color='coral'),
                  row=2, col=1)
    
    # Time
    avg_time = df.groupby('model')['time'].mean().sort_values()
    fig.add_trace(go.Bar(x=avg_time.index, y=avg_time.values,
                         name='Avg Time', marker_color='lightgreen'),
                  row=2, col=2)
    
    fig.update_layout(height=900, showlegend=False, title_text="Model Performance Dashboard")
    fig.update_xaxes(tickangle=45)
    
    fig.write_html(output_dir / 'interactive_dashboard.html')
    print(f"[SAVED] interactive_dashboard.html")

def main():
    """Main function"""
    print("\n" + "="*80)
    print("CROSS-MODEL INTERACTIVE VISUALIZATION (Plotly)")
    print("="*80 + "\n")
    
    df = load_all_results()
    if df is None:
        return
    
    output_dir = Path(__file__).parent.parent.parent / "results" / "cross_model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    create_success_rate_chart(df, output_dir)
    create_reward_distribution(df, output_dir)
    create_performance_radar(df, output_dir)
    create_parallel_coordinates(df, output_dir)
    create_learning_curves(df, output_dir)
    create_dashboard(df, output_dir)

    print(f"\n[COMPLETE] All interactive visualizations saved to: {output_dir}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

