"""
Analysis Script for GAT+CVD vs DDQN Comparison

Usage:
    python analyze_results.py --ddqn_log path/to/ddqn_episodes.csv --gat_cvd_log path/to/gat_cvd_episodes.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def load_data(ddqn_path, gat_cvd_path):
    """Load DDQN and GAT+CVD episode logs"""
    print(f"Loading DDQN data from: {ddqn_path}")
    ddqn_df = pd.read_csv(ddqn_path)
    
    print(f"Loading GAT+CVD data from: {gat_cvd_path}")
    gat_cvd_df = pd.read_csv(gat_cvd_path)
    
    print(f"DDQN: {len(ddqn_df)} episodes")
    print(f"GAT+CVD: {len(gat_cvd_df)} episodes")
    
    return ddqn_df, gat_cvd_df


def plot_learning_curves(ddqn_df, gat_cvd_df, output_dir):
    """Plot learning curves comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Average Reward
    ax = axes[0, 0]
    ax.plot(ddqn_df['episode'], ddqn_df['avg_reward_100'], label='DDQN', linewidth=2, alpha=0.8)
    ax.plot(gat_cvd_df['episode'], gat_cvd_df['avg_reward_100'], label='GAT+CVD', linewidth=2, alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward (100 ep)')
    ax.set_title('Learning Curve: Average Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Success Rate
    ax = axes[0, 1]
    ax.plot(ddqn_df['episode'], ddqn_df['success_rate_100'] * 100, label='DDQN', linewidth=2, alpha=0.8)
    ax.plot(gat_cvd_df['episode'], gat_cvd_df['success_rate_100'] * 100, label='GAT+CVD', linewidth=2, alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Learning Curve: Success Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Episode Length
    ax = axes[1, 0]
    # Smooth episode length with rolling average
    ddqn_length_smooth = ddqn_df['length'].rolling(window=100, min_periods=1).mean()
    gat_cvd_length_smooth = gat_cvd_df['length'].rolling(window=100, min_periods=1).mean()
    ax.plot(ddqn_df['episode'], ddqn_length_smooth, label='DDQN', linewidth=2, alpha=0.8)
    ax.plot(gat_cvd_df['episode'], gat_cvd_length_smooth, label='GAT+CVD', linewidth=2, alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length (100 ep avg)')
    ax.set_title('Episode Length Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Total Reward Distribution (last 100 episodes)
    ax = axes[1, 1]
    ddqn_final = ddqn_df['total_reward'].tail(100)
    gat_cvd_final = gat_cvd_df['total_reward'].tail(100)
    ax.hist(ddqn_final, bins=20, alpha=0.5, label='DDQN', density=True)
    ax.hist(gat_cvd_final, bins=20, alpha=0.5, label='GAT+CVD', density=True)
    ax.set_xlabel('Total Reward')
    ax.set_ylabel('Density')
    ax.set_title('Reward Distribution (Last 100 Episodes)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'learning_curves_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_sample_efficiency(ddqn_df, gat_cvd_df, output_dir, threshold=0.9):
    """Plot sample efficiency (steps to reach threshold success rate)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Find first episode where success rate >= threshold
    ddqn_threshold_ep = ddqn_df[ddqn_df['success_rate_100'] >= threshold]['episode'].min()
    gat_cvd_threshold_ep = gat_cvd_df[gat_cvd_df['success_rate_100'] >= threshold]['episode'].min()
    
    # Plot success rate
    ax.plot(ddqn_df['episode'], ddqn_df['success_rate_100'] * 100, label='DDQN', linewidth=2, alpha=0.8)
    ax.plot(gat_cvd_df['episode'], gat_cvd_df['success_rate_100'] * 100, label='GAT+CVD', linewidth=2, alpha=0.8)
    
    # Add threshold line
    ax.axhline(y=threshold * 100, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'{threshold*100}% Threshold')
    
    # Add vertical lines at threshold episodes
    if not pd.isna(ddqn_threshold_ep):
        ax.axvline(x=ddqn_threshold_ep, color='blue', linestyle=':', alpha=0.5)
        ax.text(ddqn_threshold_ep, 50, f'DDQN: {ddqn_threshold_ep} ep', rotation=90, va='bottom')
    
    if not pd.isna(gat_cvd_threshold_ep):
        ax.axvline(x=gat_cvd_threshold_ep, color='orange', linestyle=':', alpha=0.5)
        ax.text(gat_cvd_threshold_ep, 50, f'GAT+CVD: {gat_cvd_threshold_ep} ep', rotation=90, va='bottom')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(f'Sample Efficiency: Episodes to {threshold*100}% Success')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'sample_efficiency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    return ddqn_threshold_ep, gat_cvd_threshold_ep


def statistical_comparison(ddqn_df, gat_cvd_df, output_dir):
    """Perform statistical comparison"""
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON")
    print("="*60)
    
    # Compare last 100 episodes
    ddqn_final_reward = ddqn_df['total_reward'].tail(100)
    gat_cvd_final_reward = gat_cvd_df['total_reward'].tail(100)
    
    ddqn_final_success = ddqn_df['success'].tail(100)
    gat_cvd_final_success = gat_cvd_df['success'].tail(100)
    
    # T-test for reward
    t_stat_reward, p_value_reward = stats.ttest_ind(ddqn_final_reward, gat_cvd_final_reward)
    
    # T-test for success
    t_stat_success, p_value_success = stats.ttest_ind(ddqn_final_success, gat_cvd_final_success)
    
    # Summary statistics
    results = {
        'Metric': ['Avg Reward', 'Success Rate', 'Episode Length'],
        'DDQN Mean': [
            ddqn_final_reward.mean(),
            ddqn_final_success.mean(),
            ddqn_df['length'].tail(100).mean()
        ],
        'DDQN Std': [
            ddqn_final_reward.std(),
            ddqn_final_success.std(),
            ddqn_df['length'].tail(100).std()
        ],
        'GAT+CVD Mean': [
            gat_cvd_final_reward.mean(),
            gat_cvd_final_success.mean(),
            gat_cvd_df['length'].tail(100).mean()
        ],
        'GAT+CVD Std': [
            gat_cvd_final_reward.std(),
            gat_cvd_final_success.std(),
            gat_cvd_df['length'].tail(100).std()
        ],
        'Improvement (%)': [
            ((gat_cvd_final_reward.mean() - ddqn_final_reward.mean()) / ddqn_final_reward.mean()) * 100,
            ((gat_cvd_final_success.mean() - ddqn_final_success.mean()) / ddqn_final_success.mean()) * 100,
            ((ddqn_df['length'].tail(100).mean() - gat_cvd_df['length'].tail(100).mean()) / ddqn_df['length'].tail(100).mean()) * 100
        ]
    }
    
    results_df = pd.DataFrame(results)
    print("\nFinal Performance (Last 100 Episodes):")
    print(results_df.to_string(index=False))
    
    print(f"\nT-Test Results:")
    print(f"  Reward: t={t_stat_reward:.4f}, p={p_value_reward:.4f} {'✅ Significant' if p_value_reward < 0.05 else '❌ Not significant'}")
    print(f"  Success: t={t_stat_success:.4f}, p={p_value_success:.4f} {'✅ Significant' if p_value_success < 0.05 else '❌ Not significant'}")
    
    # Save to file
    output_path = output_dir / 'statistical_comparison.txt'
    with open(output_path, 'w') as f:
        f.write("STATISTICAL COMPARISON\n")
        f.write("="*60 + "\n\n")
        f.write("Final Performance (Last 100 Episodes):\n")
        f.write(results_df.to_string(index=False) + "\n\n")
        f.write(f"T-Test Results:\n")
        f.write(f"  Reward: t={t_stat_reward:.4f}, p={p_value_reward:.4f} {'✅ Significant' if p_value_reward < 0.05 else '❌ Not significant'}\n")
        f.write(f"  Success: t={t_stat_success:.4f}, p={p_value_success:.4f} {'✅ Significant' if p_value_success < 0.05 else '❌ Not significant'}\n")
    
    print(f"\nSaved: {output_path}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Analyze GAT+CVD vs DDQN results")
    parser.add_argument("--ddqn_log", type=str, required=True,
                       help="Path to DDQN episode log CSV")
    parser.add_argument("--gat_cvd_log", type=str, required=True,
                       help="Path to GAT+CVD episode log CSV")
    parser.add_argument("--output_dir", type=str, default="analysis_results",
                       help="Output directory for plots and results")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_dir}")
    
    # Load data
    ddqn_df, gat_cvd_df = load_data(args.ddqn_log, args.gat_cvd_log)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_learning_curves(ddqn_df, gat_cvd_df, output_dir)
    ddqn_threshold, gat_cvd_threshold = plot_sample_efficiency(ddqn_df, gat_cvd_df, output_dir)
    
    # Statistical comparison
    results_df = statistical_comparison(ddqn_df, gat_cvd_df, output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"DDQN episodes to 90% success: {ddqn_threshold if not pd.isna(ddqn_threshold) else 'Not reached'}")
    print(f"GAT+CVD episodes to 90% success: {gat_cvd_threshold if not pd.isna(gat_cvd_threshold) else 'Not reached'}")
    print(f"\nAll results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

