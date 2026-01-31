"""
Radar Overlay - Plotly
Interactive radar chart overlay for all models
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

def create_radar_overlay(df, output_dir):
    """Create interactive radar overlay"""
    # Calculate metrics per model
    metrics = df.groupby('model').agg({
        'success_rate': 'mean',
        'reward': 'mean',
        'steps': 'mean',
        'time': 'mean'
    }).reset_index()
    
    # Normalize (0-100 scale)
    for col in ['success_rate', 'reward', 'steps', 'time']:
        if col in ['steps', 'time']:  # Lower is better
            metrics[f'{col}_norm'] = 100 - ((metrics[col] - metrics[col].min()) / 
                                            (metrics[col].max() - metrics[col].min() + 1e-6) * 100)
        else:  # Higher is better
            metrics[f'{col}_norm'] = (metrics[col] - metrics[col].min()) / \
                                     (metrics[col].max() - metrics[col].min() + 1e-6) * 100
    
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
        title='Performance Radar Overlay - All Models',
        height=700,
        font=dict(size=14)
    )
    
    fig.write_html(output_dir / 'radar_overlay_interactive.html')
    print(f"[SAVED] radar_overlay_interactive.html")

def main():
    print("\n" + "="*80)
    print("RADAR OVERLAY (Plotly)")
    print("="*80 + "\n")
    
    df = load_all_results()
    if df is None:
        print("[ERROR] No results found!")
        return
    
    output_dir = Path(__file__).parent.parent.parent / "results" / "cross_model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    create_radar_overlay(df, output_dir)
    
    print("\n[COMPLETE] Radar overlay generated")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

