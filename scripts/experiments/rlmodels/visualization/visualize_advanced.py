"""
Advanced Visualizations for RL Experiments
Generates all 5 new advanced visualization types for experiment results
"""

import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from individual_models.plotly import (
    plot_box_with_points_plotly,
    plot_multi_metric_boxes_plotly,
    plot_histogram_with_kde_plotly,
    plot_multi_histogram_kde_grid,
    plot_timeseries_with_ci_plotly
)

from individual_models.seaborn import (
    plot_pairplot_matrix_seaborn,
    plot_corner_pairplot_seaborn,
    plot_timeseries_with_ci_seaborn,
    plot_learning_curves_with_ci
)

from cross_model.seaborn import (
    plot_grouped_bars_with_ci_seaborn,
    plot_multi_metric_grouped_bars
)

from cross_model.plotly import (
    plot_grouped_bars_with_ci_plotly
)


def load_results(experiment_name):
    """Load experiment results"""
    results_path = Path(__file__).parent.parent / "results" / experiment_name / "comparison_results.csv"
    
    if not results_path.exists():
        print(f"⚠️  Results not found: {results_path}")
        return None
    
    df = pd.read_csv(results_path)
    print(f"✅ Loaded {len(df)} results from {experiment_name}")
    return df


def generate_advanced_visualizations(experiment_name):
    """Generate all advanced visualizations for an experiment"""
    
    print(f"\n{'='*80}")
    print(f"ADVANCED VISUALIZATIONS - {experiment_name.upper()}")
    print(f"{'='*80}\n")
    
    # Load results
    df = load_results(experiment_name)
    if df is None:
        return
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "results" / experiment_name / "advanced_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating visualizations...\n")
    
    # 1. Box Plots with All Data Points
    print("1️⃣  Box Plots with All Data Points...")
    try:
        plot_box_with_points_plotly(
            df, metric='reward', group_by='model',
            output_path=output_dir / '1_reward_box_with_points.html'
        )
        plot_multi_metric_boxes_plotly(
            df, metrics=['reward', 'steps'],
            group_by='model',
            output_path=output_dir / '1_multi_metric_boxes.html'
        )
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    # 2. Histograms with KDE
    print("2️⃣  Histograms with KDE Distribution Curves...")
    try:
        plot_histogram_with_kde_plotly(
            df, metric='reward', group_by='model',
            output_path=output_dir / '2_reward_histogram_kde.html'
        )
        plot_multi_histogram_kde_grid(
            df, metrics=['reward', 'steps'],
            group_by='model',
            output_path=output_dir / '2_multi_histogram_kde.html'
        )
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    # 3. Pairplot Scatter Matrices
    print("3️⃣  Pairplot Scatter Matrices...")
    try:
        # Check if we have the required columns
        available_metrics = []
        for metric in ['reward', 'steps', 'success_rate']:
            if metric in df.columns:
                available_metrics.append(metric)
        
        if len(available_metrics) >= 2:
            plot_pairplot_matrix_seaborn(
                df, metrics=available_metrics,
                hue='model',
                output_path=output_dir / '3_pairplot_full.png'
            )
            plot_corner_pairplot_seaborn(
                df, metrics=available_metrics,
                hue='model',
                output_path=output_dir / '3_pairplot_corner.png'
            )
        else:
            print(f"   ⚠️  Not enough metrics for pairplot (need at least 2)")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    # 4. Grouped Bar Charts with Error Bars
    print("4️⃣  Grouped Bar Charts with Confidence Intervals...")
    try:
        plot_grouped_bars_with_ci_seaborn(
            df, x='model', y='reward', ci=95,
            output_path=output_dir / '4_reward_comparison_ci95.png'
        )
        
        available_metrics = [m for m in ['reward', 'steps', 'success_rate'] if m in df.columns]
        if len(available_metrics) >= 2:
            plot_multi_metric_grouped_bars(
                df, metrics=available_metrics, x='model',
                output_path=output_dir / '4_multi_metric_comparison.png'
            )
        
        plot_grouped_bars_with_ci_plotly(
            df, x='model', y='reward',
            output_path=output_dir / '4_reward_comparison_interactive.html'
        )
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    # 5. Time Series with Shaded Confidence Intervals
    print("5️⃣  Time Series with Shaded Confidence Intervals...")
    try:
        if 'episode' in df.columns:
            plot_timeseries_with_ci_seaborn(
                df, x='episode', y='reward', hue='model', ci=95,
                output_path=output_dir / '5_learning_curves_ci.png'
            )
            plot_learning_curves_with_ci(
                df, x='episode', y='reward', hue='model', window=10,
                output_path=output_dir / '5_learning_curves_smoothed.png'
            )
            plot_timeseries_with_ci_plotly(
                df, x='episode', y='reward', hue='model',
                output_path=output_dir / '5_learning_curves_interactive.html'
            )
        else:
            print(f"   ⚠️  'episode' column not found, skipping time series plots")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    print(f"\n{'='*80}")
    print(f"✅ All advanced visualizations saved to:")
    print(f"   {output_dir}")
    print(f"{'='*80}\n")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate advanced visualizations')
    parser.add_argument('--experiment', type=str, default='all',
                       help='Experiment name (exp1, exp2, or all)')
    
    args = parser.parse_args()
    
    if args.experiment == 'all':
        # Generate for all experiments
        for exp in ['exp1', 'exp2']:
            generate_advanced_visualizations(exp)
    else:
        generate_advanced_visualizations(args.experiment)


if __name__ == "__main__":
    main()

