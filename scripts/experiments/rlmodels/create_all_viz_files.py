"""
Create all remaining visualization files
This script generates all 22 visualization scripts (11 charts × 2 libraries)
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Template for Plotly visualizations
PLOTLY_TEMPLATES = {
    'reward_distribution': '''"""
Reward Distribution - Plotly Version
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

def plot_reward_distribution_plotly(df, model_name, output_path=None):
    model_data = df[df['model_name'] == model_name]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=model_data['reward'], nbinsx=30, name='Reward',
                               marker_color='#A23B72', opacity=0.7))
    
    mean_reward = model_data['reward'].mean()
    fig.add_vline(x=mean_reward, line_dash="dash", line_color="red",
                 annotation_text=f"Mean: {mean_reward:.2f}")
    
    fig.update_layout(title=f'Reward Distribution - {model_name}',
                     xaxis_title='Reward', yaxis_title='Frequency',
                     template='plotly_white', width=1000, height=600)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"✅ Saved: {output_path}")
    
    return fig

if __name__ == "__main__":
    import numpy as np
    np.random.seed(42)
    data = {'reward': np.random.normal(100, 30, 100), 'model_name': ['Test'] * 100}
    df = pd.DataFrame(data)
    plot_reward_distribution_plotly(df, 'Test', 'test.html')
''',

    'steps_distribution': '''"""
Steps Distribution - Plotly Version
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

def plot_steps_distribution_plotly(df, model_name, output_path=None):
    model_data = df[df['model_name'] == model_name]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Histogram', 'Box Plot'))
    
    fig.add_trace(go.Histogram(x=model_data['steps'], nbinsx=20, name='Steps',
                               marker_color='#F18F01'), row=1, col=1)
    
    fig.add_trace(go.Box(y=model_data['steps'], name='Steps',
                        marker_color='#F18F01'), row=1, col=2)
    
    fig.update_layout(title=f'Steps Distribution - {model_name}',
                     template='plotly_white', width=1400, height=600, showlegend=False)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"✅ Saved: {output_path}")
    
    return fig

if __name__ == "__main__":
    import numpy as np
    np.random.seed(42)
    data = {'steps': np.random.poisson(25, 100), 'model_name': ['Test'] * 100}
    df = pd.DataFrame(data)
    plot_steps_distribution_plotly(df, 'Test', 'test.html')
''',

    'multi_metric_dashboard': '''"""
Multi-Metric Dashboard - Plotly Version
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

def plot_multi_metric_dashboard_plotly(df, model_name, output_path=None):
    model_data = df[df['model_name'] == model_name].copy()
    
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=('Success Rate', 'Reward Dist', 'Steps', 'Collisions'))
    
    # Success rate
    model_data['success_rate'] = model_data['success'].rolling(10, min_periods=1).mean() * 100
    fig.add_trace(go.Scatter(x=model_data['episode'], y=model_data['success_rate'],
                            mode='lines', name='Success', line=dict(color='#2E86AB')),
                 row=1, col=1)
    
    # Reward distribution
    fig.add_trace(go.Histogram(x=model_data['reward'], nbinsx=20, name='Reward',
                              marker_color='#A23B72'), row=1, col=2)
    
    # Steps box plot
    fig.add_trace(go.Box(y=model_data['steps'], name='Steps',
                        marker_color='#F18F01'), row=2, col=1)
    
    # Collision rate
    if 'collision' in model_data.columns:
        model_data['collision_rate'] = model_data['collision'].rolling(10, min_periods=1).mean() * 100
        fig.add_trace(go.Scatter(x=model_data['episode'], y=model_data['collision_rate'],
                                mode='lines', name='Collision', line=dict(color='#C73E1D')),
                     row=2, col=2)
    
    fig.update_layout(title=f'Multi-Metric Dashboard - {model_name}',
                     template='plotly_white', width=1400, height=1000, showlegend=False)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"✅ Saved: {output_path}")
    
    return fig
''',

    'performance_radar': '''"""
Performance Radar - Plotly Version
"""

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

def plot_performance_radar_plotly(df, model_name, output_path=None):
    model_data = df[df['model_name'] == model_name]
    
    metrics = {
        'Success Rate': model_data['success'].mean() * 100,
        'Avg Reward': min(100, max(0, (model_data['reward'].mean() + 100) / 3)),
        'Efficiency': max(0, 100 - model_data['steps'].mean() * 2),
        'Safety': max(0, 100 - model_data.get('collision', pd.Series([0])).mean() * 100),
        'Speed': min(100, 1000 / (model_data.get('inference_time', pd.Series([10])).mean() + 1))
    }
    
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself',
                                  name=model_name, line_color='#2E86AB'))
    
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                     title=f'Performance Radar - {model_name}',
                     template='plotly_white', width=800, height=800)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"✅ Saved: {output_path}")
    
    return fig
'''
}

def create_viz_file(path, content):
    """Create a visualization file"""
    file_path = BASE_DIR / path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"✅ Created: {path}")

def main():
    print("=" * 60)
    print("CREATING ALL VISUALIZATION FILES")
    print("=" * 60)
    
    # Create remaining Plotly individual model visualizations
    for name, template in PLOTLY_TEMPLATES.items():
        create_viz_file(f"visualization/individual_models/plotly/{name}.py", template)
    
    print("\n✅ Created 4 Plotly individual model visualizations")
    print("⏳ Cross-model visualizations need to be created separately")
    print("\nTo complete all visualizations, run the individual scripts in:")
    print("  - visualization/individual_models/seaborn/")
    print("  - visualization/individual_models/plotly/")
    print("  - visualization/cross_model/seaborn/")
    print("  - visualization/cross_model/plotly/")

if __name__ == "__main__":
    main()

