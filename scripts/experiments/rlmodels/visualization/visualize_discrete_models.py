"""
Visualization for Discrete Models (Experiment 1)
Generates charts and plots for discrete action space models
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import sys

# Set Linux Libertine font globally for all plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Linux Libertine', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Linux Libertine'


# Add individual_models to path
sys.path.insert(0, str(Path(__file__).parent / "individual_models" / "seaborn"))
sys.path.insert(0, str(Path(__file__).parent / "individual_models" / "plotly"))

# Import visualization modules
from individual_models.seaborn import learning_curves as seaborn_learning
from individual_models.plotly import learning_curves as plotly_learning

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_results(experiment_name="exp1"):
    """Load results from CSV"""
    results_path = Path(__file__).parent.parent / "results" / experiment_name / "comparison_results.csv"
    if not results_path.exists():
        print(f"[ERROR] Results file not found: {results_path}")
        return None

    df = pd.read_csv(results_path)
    print(f"[INFO] Loaded {len(df)} results from {results_path}")
    return df

def plot_success_rates(df, output_dir):
    """Plot success rates for all discrete models"""
    plt.figure(figsize=(12, 6))
    
    # Calculate success rate per model
    success_rates = df.groupby('model')['success_rate'].mean().sort_values(ascending=False)
    
    # Create bar plot
    ax = success_rates.plot(kind='bar', color='steelblue')
    plt.title('Success Rates - Discrete Models', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(success_rates):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'discrete_success_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] discrete_success_rates.png")

def plot_reward_comparison(df, output_dir):
    """Plot reward comparison across discrete models"""
    plt.figure(figsize=(14, 6))

    # Check if we have episode-level data or summary data
    reward_col = 'reward' if 'reward' in df.columns else 'avg_reward'

    if reward_col == 'avg_reward':
        # Summary data: use bar plot with error bars
        models = df['model']
        rewards = df['avg_reward']
        errors = df.get('std_reward', [0] * len(df))

        bars = plt.bar(range(len(models)), rewards, yerr=errors, capsize=5,
                       color='skyblue', alpha=0.7, edgecolor='black')
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        plt.title('Average Reward - Discrete Models', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Average Reward', fontsize=12)
        plt.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, reward) in enumerate(zip(bars, rewards)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{reward:.1f}', ha='center', va='bottom', fontsize=9)
    else:
        # Episode-level data: use box plot
        sns.boxplot(data=df, x='model', y=reward_col, palette='Set2')
        plt.title('Reward Distribution - Discrete Models', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'discrete_reward_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] discrete_reward_comparison.png")

def plot_learning_curves(df, output_dir):
    """Plot learning curves (reward over episodes)"""
    # Skip if we don't have episode-level data
    if 'episode' not in df.columns or 'reward' not in df.columns:
        print(f"[SKIPPED] discrete_learning_curves.png (requires episode-level data)")
        return

    plt.figure(figsize=(14, 8))

    for model in df['model'].unique():
        model_data = df[df['model'] == model].sort_values('episode')
        plt.plot(model_data['episode'], model_data['reward'], label=model, alpha=0.7, linewidth=2)

    plt.title('Learning Curves - Discrete Models', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'discrete_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] discrete_learning_curves.png")

def plot_steps_comparison(df, output_dir):
    """Plot steps comparison"""
    # Check for correct column names
    steps_col = 'steps' if 'steps' in df.columns else 'avg_length'

    if steps_col not in df.columns:
        print(f"[SKIPPED] discrete_steps_comparison.png (no steps/avg_length column)")
        return

    plt.figure(figsize=(12, 6))

    if steps_col == 'avg_length':
        # Summary data: use bar plot
        models = df['model']
        steps = df['avg_length']

        bars = plt.barh(range(len(models)), steps, color='coral', alpha=0.7, edgecolor='black')
        plt.yticks(range(len(models)), models)
        plt.title('Average Steps per Episode - Discrete Models', fontsize=16, fontweight='bold')
        plt.xlabel('Average Steps', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, step) in enumerate(zip(bars, steps)):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{step:.1f}', ha='left', va='center', fontsize=9)
    else:
        # Episode-level data: calculate average
        avg_steps = df.groupby('model')[steps_col].mean().sort_values()
        ax = avg_steps.plot(kind='barh', color='coral')
        plt.title('Average Steps per Episode - Discrete Models', fontsize=16, fontweight='bold')
        plt.xlabel('Average Steps', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(avg_steps):
        ax.text(v + 0.5, i, f'{v:.1f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'discrete_steps_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] discrete_steps_comparison.png")

def generate_summary_table(df, output_dir):
    """Generate summary statistics table"""
    summary = df.groupby('model').agg({
        'reward': ['mean', 'std', 'min', 'max'],
        'steps': ['mean', 'std'],
        'success_rate': 'mean',
        'time': 'mean'
    }).round(2)
    
    # Save to CSV
    summary.to_csv(output_dir / 'discrete_summary_stats.csv')
    print(f"[SAVED] discrete_summary_stats.csv")
    
    # Print to console
    print("\n" + "="*80)
    print("DISCRETE MODELS - SUMMARY STATISTICS")
    print("="*80)
    print(summary)
    print("="*80 + "\n")

def main():
    """Main visualization function"""
    print("\n" + "="*80)
    print("DISCRETE MODELS VISUALIZATION")
    print("="*80 + "\n")

    # Load results
    df = load_results("exp1")
    if df is None:
        return

    # Create output directory
    output_dir = Path(__file__).parent.parent / "results" / "exp1" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results CSV path
    results_csv = Path(__file__).parent.parent / "results" / "exp1" / "comparison_results.csv"

    # Generate plots
    plot_success_rates(df, output_dir)
    plot_reward_comparison(df, output_dir)
    plot_learning_curves(df, output_dir)
    plot_steps_comparison(df, output_dir)
    generate_summary_table(df, output_dir)

    # Generate learning curves (Seaborn)
    print("\n[INFO] Generating Seaborn learning curves...")
    seaborn_learning.create_learning_curves(str(results_csv), str(output_dir), "exp1")

    # Generate learning curves (Plotly)
    print("[INFO] Generating Plotly learning curves...")
    plotly_learning.create_learning_curves(str(results_csv), str(output_dir), "exp1")

    print(f"\n[COMPLETE] All visualizations saved to: {output_dir}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

