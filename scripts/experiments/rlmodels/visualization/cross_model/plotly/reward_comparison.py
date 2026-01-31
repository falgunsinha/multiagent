"""
Reward Comparison - Plotly
Interactive reward distribution comparison
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

def plot_reward_box_plotly(df):
    """Create interactive reward comparison - returns figure for W&B"""
    fig = px.box(df, x='model', y='avg_reward',
                 title='Reward Distribution Comparison - All Models',
                 labels={'avg_reward': 'Reward', 'model': 'Model'})

    fig.update_layout(
        height=600,
        font=PLOTLY_FONT,
        hovermode='x unified',
        xaxis_tickangle=-45
    )

    return fig

def create_reward_comparison(df, output_dir):
    """Create interactive reward comparison"""
    fig = px.box(df, x='model', y='reward', color='experiment',
                 title='Reward Distribution Comparison - All Models',
                 labels={'reward': 'Reward', 'model': 'Model'},
                 color_discrete_map={'Discrete': '#2E86AB', 'Continuous': '#A23B72'})
    
    fig.update_layout(
        height=600,
        font=dict(size=14),
        hovermode='x unified',
        xaxis_tickangle=-45
    )
    
    fig.write_html(output_dir / 'reward_comparison_interactive.html')
    print(f"[SAVED] reward_comparison_interactive.html")

def main():
    print("\n" + "="*80)
    print("REWARD COMPARISON (Plotly)")
    print("="*80 + "\n")
    
    df = load_all_results()
    if df is None:
        print("[ERROR] No results found!")
        return
    
    output_dir = Path(__file__).parent.parent.parent / "results" / "cross_model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    create_reward_comparison(df, output_dir)
    
    print("\n[COMPLETE] Reward comparison generated")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

