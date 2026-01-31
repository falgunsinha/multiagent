"""
Learning Curves Visualization (Seaborn)
Shows episode reward, success rate, and steps over time
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

# Set Linux Libertine font globally for all plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Linux Libertine', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Linux Libertine'


# Add parent directory to path for font_config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from font_config import configure_matplotlib_fonts

# Configure fonts
configure_matplotlib_fonts()

def create_learning_curves(results_csv, output_dir, experiment_name):
    """
    Create learning curves showing metrics over episodes
    
    Args:
        results_csv: Path to results CSV file
        output_dir: Directory to save output
        experiment_name: Name of experiment (e.g., 'exp1', 'exp2')
    """
    # Load data
    df = pd.read_csv(results_csv)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get unique models
    models = df['model'].unique()
    
    # Create figure with 3 subplots (reward, success rate, steps)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Color palette
    colors = sns.color_palette("husl", len(models))
    
    # Plot 1: Episode Reward over Time
    ax1 = axes[0]
    for idx, model in enumerate(models):
        model_data = df[df['model'] == model].sort_values('episode')
        
        # Plot raw data with transparency
        ax1.plot(model_data['episode'], model_data['reward'], 
                alpha=0.3, color=colors[idx], linewidth=1)
        
        # Plot smoothed curve (rolling average)
        window = min(10, len(model_data))
        if window > 1:
            smoothed = model_data['reward'].rolling(window=window, center=True).mean()
            ax1.plot(model_data['episode'], smoothed, 
                    label=model, color=colors[idx], linewidth=2.5)
    
    ax1.set_xlabel('Episode', fontsize=14)
    ax1.set_ylabel('Episode Reward', fontsize=14)
    ax1.set_title(f'Learning Curve: Episode Reward - {experiment_name.upper()}', 
                  fontsize=18, fontweight='bold', pad=20)
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=12)
    
    # Plot 2: Success Rate over Time
    ax2 = axes[1]
    for idx, model in enumerate(models):
        model_data = df[df['model'] == model].sort_values('episode')
        
        # Plot raw data with transparency
        ax2.plot(model_data['episode'], model_data['success_rate'], 
                alpha=0.3, color=colors[idx], linewidth=1)
        
        # Plot smoothed curve
        window = min(10, len(model_data))
        if window > 1:
            smoothed = model_data['success_rate'].rolling(window=window, center=True).mean()
            ax2.plot(model_data['episode'], smoothed, 
                    label=model, color=colors[idx], linewidth=2.5)
    
    ax2.set_xlabel('Episode', fontsize=14)
    ax2.set_ylabel('Success Rate (%)', fontsize=14)
    ax2.set_title(f'Learning Curve: Success Rate - {experiment_name.upper()}', 
                  fontsize=18, fontweight='bold', pad=20)
    ax2.legend(loc='best', fontsize=11, framealpha=0.9)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.tick_params(labelsize=12)
    ax2.set_ylim(0, 105)
    
    # Plot 3: Steps per Episode over Time
    ax3 = axes[2]
    for idx, model in enumerate(models):
        model_data = df[df['model'] == model].sort_values('episode')
        
        # Plot raw data with transparency
        ax3.plot(model_data['episode'], model_data['steps'], 
                alpha=0.3, color=colors[idx], linewidth=1)
        
        # Plot smoothed curve
        window = min(10, len(model_data))
        if window > 1:
            smoothed = model_data['steps'].rolling(window=window, center=True).mean()
            ax3.plot(model_data['episode'], smoothed, 
                    label=model, color=colors[idx], linewidth=2.5)
    
    ax3.set_xlabel('Episode', fontsize=14)
    ax3.set_ylabel('Steps per Episode', fontsize=14)
    ax3.set_title(f'Learning Curve: Steps per Episode - {experiment_name.upper()}', 
                  fontsize=18, fontweight='bold', pad=20)
    ax3.legend(loc='best', fontsize=11, framealpha=0.9)
    ax3.grid(alpha=0.3, linestyle='--')
    ax3.tick_params(labelsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / f"{experiment_name}_learning_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[Seaborn] Saved: {output_file}")
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate learning curves visualization')
    parser.add_argument('--results', type=str, required=True, help='Path to results CSV')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    
    args = parser.parse_args()
    
    create_learning_curves(args.results, args.output, args.experiment)

