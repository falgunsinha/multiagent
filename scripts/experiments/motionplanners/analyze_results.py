"""
Results Analysis and Visualization for Motion Planner Comparison

Analyzes experimental results and generates comparison plots and tables
similar to those found in research papers.

Usage:
    python analyze_results.py --results_dir results/comparison_20250106_123456
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


class ResultsAnalyzer:
    """
    Analyzes and visualizes motion planner comparison results.
    """
    
    def __init__(self, results_dir):
        """
        Initialize analyzer.
        
        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        
        # Load data
        self.df = pd.read_csv(self.results_dir / "detailed_results.csv")
        
        with open(self.results_dir / "summary_statistics.json", 'r') as f:
            self.summary = json.load(f)
        
        with open(self.results_dir / "experiment_config.json", 'r') as f:
            self.config = json.load(f)
        
        # Create output directory for plots
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        print(f"Loaded results from: {self.results_dir}")
        print(f"Total trials: {len(self.df)}")
        print(f"Planners: {self.df['planner'].unique().tolist()}")
    
    def generate_all_plots(self):
        """Generate all analysis plots"""
        print("\nGenerating analysis plots...")
        
        self.plot_success_rates()
        self.plot_planning_times()
        self.plot_path_lengths()
        self.plot_smoothness_comparison()
        self.plot_performance_radar()
        self.plot_statistical_comparison()
        self.generate_latex_table()
        
        print(f"\n✓ All plots saved to: {self.plots_dir}")
    
    def plot_success_rates(self):
        """Plot success rates comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        planners = []
        success_rates = []
        
        for planner in self.df['planner'].unique():
            planner_data = self.df[self.df['planner'] == planner]
            success_rate = planner_data['success'].mean() * 100
            planners.append(planner.upper())
            success_rates.append(success_rate)
        
        bars = ax.bar(planners, success_rates, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
        ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Motion Planner', fontsize=12, fontweight='bold')
        ax.set_title('Success Rate Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 105])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "success_rates.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Generated success_rates.png")
    
    def plot_planning_times(self):
        """Plot planning time comparison with box plots"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter successful trials only
        success_df = self.df[self.df['success'] == True]
        
        # Create box plot
        planners = success_df['planner'].unique()
        data = [success_df[success_df['planner'] == p]['planning_time'].values for p in planners]
        
        bp = ax.boxplot(data, labels=[p.upper() for p in planners], patch_artist=True)
        
        # Color boxes
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        for patch, color in zip(bp['boxes'], colors[:len(planners)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Planning Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Motion Planner', fontsize=12, fontweight='bold')
        ax.set_title('Planning Time Comparison (Successful Trials)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "planning_times.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Generated planning_times.png")
    
    def plot_path_lengths(self):
        """Plot path length comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter successful trials only
        success_df = self.df[self.df['success'] == True]
        
        # Create box plot
        planners = success_df['planner'].unique()
        data = [success_df[success_df['planner'] == p]['path_length'].values for p in planners]
        
        bp = ax.boxplot(data, labels=[p.upper() for p in planners], patch_artist=True)
        
        # Color boxes
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        for patch, color in zip(bp['boxes'], colors[:len(planners)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Path Length (meters)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Motion Planner', fontsize=12, fontweight='bold')
        ax.set_title('Path Length Comparison (Successful Trials)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "path_lengths.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Generated path_lengths.png")

    def plot_smoothness_comparison(self):
        """Plot path smoothness comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Filter successful trials only
        success_df = self.df[self.df['success'] == True]

        planners = []
        smoothness_vals = []

        for planner in success_df['planner'].unique():
            planner_data = success_df[success_df['planner'] == planner]
            avg_smoothness = planner_data['smoothness'].mean()
            planners.append(planner.upper())
            smoothness_vals.append(avg_smoothness)

        bars = ax.bar(planners, smoothness_vals, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
        ax.set_ylabel('Average Smoothness (lower is better)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Motion Planner', fontsize=12, fontweight='bold')
        ax.set_title('Path Smoothness Comparison', fontsize=14, fontweight='bold')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.plots_dir / "smoothness.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Generated smoothness.png")

    def plot_performance_radar(self):
        """Create radar chart comparing multiple metrics"""
        from math import pi

        # Metrics to compare (normalized to 0-1 scale)
        metrics = ['Success Rate', 'Speed', 'Path Quality', 'Smoothness']

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
        angles += angles[:1]

        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')

        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

        for idx, planner in enumerate(self.df['planner'].unique()):
            planner_data = self.df[self.df['planner'] == planner]
            success_data = planner_data[planner_data['success'] == True]

            # Calculate normalized metrics
            success_rate = planner_data['success'].mean()

            # Speed: inverse of planning time (normalized)
            if len(success_data) > 0:
                avg_time = success_data['planning_time'].mean()
                max_time = self.df[self.df['success'] == True]['planning_time'].max()
                speed = 1.0 - (avg_time / max_time) if max_time > 0 else 0.0

                # Path quality: inverse of path length (normalized)
                avg_length = success_data['path_length'].mean()
                max_length = self.df[self.df['success'] == True]['path_length'].max()
                path_quality = 1.0 - (avg_length / max_length) if max_length > 0 else 0.0

                # Smoothness: inverse of smoothness metric (normalized)
                avg_smoothness = success_data['smoothness'].mean()
                max_smoothness = self.df[self.df['success'] == True]['smoothness'].max()
                smoothness = 1.0 - (avg_smoothness / max_smoothness) if max_smoothness > 0 else 0.0
            else:
                speed = 0.0
                path_quality = 0.0
                smoothness = 0.0

            values = [success_rate, speed, path_quality, smoothness]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=planner.upper(), color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])

        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        ax.set_title('Multi-Metric Performance Comparison', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "performance_radar.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Generated performance_radar.png")

    def plot_statistical_comparison(self):
        """Plot statistical significance comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Filter successful trials
        success_df = self.df[self.df['success'] == True]
        planners = success_df['planner'].unique()

        # Planning time comparison with error bars
        means_time = []
        stds_time = []
        for planner in planners:
            data = success_df[success_df['planner'] == planner]['planning_time']
            means_time.append(data.mean())
            stds_time.append(data.std())

        x_pos = np.arange(len(planners))
        ax1.bar(x_pos, means_time, yerr=stds_time, capsize=5,
               color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'][:len(planners)], alpha=0.7)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([p.upper() for p in planners])
        ax1.set_ylabel('Planning Time (s)', fontsize=12, fontweight='bold')
        ax1.set_title('Planning Time (Mean ± Std)', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Path length comparison with error bars
        means_length = []
        stds_length = []
        for planner in planners:
            data = success_df[success_df['planner'] == planner]['path_length']
            means_length.append(data.mean())
            stds_length.append(data.std())

        ax2.bar(x_pos, means_length, yerr=stds_length, capsize=5,
               color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'][:len(planners)], alpha=0.7)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([p.upper() for p in planners])
        ax2.set_ylabel('Path Length (m)', fontsize=12, fontweight='bold')
        ax2.set_title('Path Length (Mean ± Std)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "statistical_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Generated statistical_comparison.png")

    def generate_latex_table(self):
        """Generate LaTeX table for research paper"""
        success_df = self.df[self.df['success'] == True]

        latex_lines = []
        latex_lines.append("\\begin{table}[h]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{Motion Planning Algorithm Comparison Results}")
        latex_lines.append("\\label{tab:planner_comparison}")
        latex_lines.append("\\begin{tabular}{lcccccc}")
        latex_lines.append("\\hline")
        latex_lines.append("\\textbf{Planner} & \\textbf{Success} & \\textbf{Time (s)} & \\textbf{Length (m)} & \\textbf{Waypoints} & \\textbf{Smoothness} & \\textbf{Energy} \\\\")
        latex_lines.append("\\hline")

        for planner in self.df['planner'].unique():
            planner_data = self.df[self.df['planner'] == planner]
            success_data = planner_data[planner_data['success'] == True]

            success_rate = planner_data['success'].mean() * 100

            if len(success_data) > 0:
                avg_time = success_data['planning_time'].mean()
                std_time = success_data['planning_time'].std()
                avg_length = success_data['path_length'].mean()
                std_length = success_data['path_length'].std()
                avg_waypoints = success_data['num_waypoints'].mean()
                avg_smoothness = success_data['smoothness'].mean()
                avg_energy = success_data['energy'].mean()

                latex_lines.append(
                    f"{planner.upper()} & "
                    f"{success_rate:.1f}\\% & "
                    f"{avg_time:.4f}$\\pm${std_time:.4f} & "
                    f"{avg_length:.4f}$\\pm${std_length:.4f} & "
                    f"{avg_waypoints:.1f} & "
                    f"{avg_smoothness:.4f} & "
                    f"{avg_energy:.4f} \\\\"
                )
            else:
                latex_lines.append(
                    f"{planner.upper()} & "
                    f"{success_rate:.1f}\\% & "
                    f"N/A & N/A & N/A & N/A & N/A \\\\"
                )

        latex_lines.append("\\hline")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")

        # Save to file
        latex_file = self.plots_dir / "comparison_table.tex"
        with open(latex_file, 'w') as f:
            f.write('\n'.join(latex_lines))

        print(f"  ✓ Generated comparison_table.tex")

        # Also generate markdown table
        md_lines = []
        md_lines.append("# Motion Planning Algorithm Comparison Results\n")
        md_lines.append("| Planner | Success Rate | Avg Time (s) | Avg Length (m) | Avg Waypoints | Avg Smoothness | Avg Energy |")
        md_lines.append("|---------|--------------|--------------|----------------|---------------|----------------|------------|")

        for planner in self.df['planner'].unique():
            planner_data = self.df[self.df['planner'] == planner]
            success_data = planner_data[planner_data['success'] == True]

            success_rate = planner_data['success'].mean() * 100

            if len(success_data) > 0:
                avg_time = success_data['planning_time'].mean()
                avg_length = success_data['path_length'].mean()
                avg_waypoints = success_data['num_waypoints'].mean()
                avg_smoothness = success_data['smoothness'].mean()
                avg_energy = success_data['energy'].mean()

                md_lines.append(
                    f"| {planner.upper()} | "
                    f"{success_rate:.1f}% | "
                    f"{avg_time:.4f} | "
                    f"{avg_length:.4f} | "
                    f"{avg_waypoints:.1f} | "
                    f"{avg_smoothness:.4f} | "
                    f"{avg_energy:.4f} |"
                )
            else:
                md_lines.append(
                    f"| {planner.upper()} | "
                    f"{success_rate:.1f}% | "
                    f"N/A | N/A | N/A | N/A | N/A |"
                )

        # Add analysis section
        md_lines.append("\n## Key Findings\n")

        # Best performers
        best_success = max(self.df['planner'].unique(),
                          key=lambda p: self.df[self.df['planner'] == p]['success'].mean())
        md_lines.append(f"- **Highest Success Rate**: {best_success.upper()}")

        if len(success_df) > 0:
            planners_with_success = success_df['planner'].unique()

            best_time = min(planners_with_success,
                          key=lambda p: success_df[success_df['planner'] == p]['planning_time'].mean())
            md_lines.append(f"- **Fastest Planning**: {best_time.upper()}")

            best_length = min(planners_with_success,
                            key=lambda p: success_df[success_df['planner'] == p]['path_length'].mean())
            md_lines.append(f"- **Shortest Paths**: {best_length.upper()}")

            best_smooth = min(planners_with_success,
                            key=lambda p: success_df[success_df['planner'] == p]['smoothness'].mean())
            md_lines.append(f"- **Smoothest Paths**: {best_smooth.upper()}")

        # Save markdown
        md_file = self.plots_dir / "comparison_table.md"
        with open(md_file, 'w') as f:
            f.write('\n'.join(md_lines))

        print(f"  ✓ Generated comparison_table.md")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Analyze motion planner comparison results")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing experiment results")
    args = parser.parse_args()

    # Create analyzer
    analyzer = ResultsAnalyzer(args.results_dir)

    # Generate all plots and tables
    analyzer.generate_all_plots()

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()

