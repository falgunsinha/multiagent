"""
MASAC Planner Comparison - Multi-Seed Analysis

Compares Isaac Sim RRT, RRT Viz, and A* planners with seed-wise breakdown
Analyzes logs from log3 directory with separate metrics for seed 42 and 123
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import json
import argparse


def plot_reward_vs_timestep():
    """Plot episode reward vs episode number with smooth curves and shaded variance"""
    print("\n" + "="*80)
    print("MASAC Planner Comparison - Episode Reward vs Episode Number")
    print("="*80 + "\n")

    log_dir = Path("cobotproject/scripts/Reinforcement Learning/MASAC/logs")

    # Find the latest episode files for each planner (ONLY A* and Isaac Sim RRT)
    planners = {
        'Isaac Sim RRT': 'masac_rrt_isaacsim_grid4_cubes9_*_episode_log.csv',
        'A*': 'masac_astar_grid4_cubes9_*_episode_log.csv'
    }

    # Load all data
    planner_data = {}
    num_episodes = 50  # All planners have 50 episodes

    for planner_name, pattern in planners.items():
        files = list(log_dir.glob(pattern))

        if not files:
            print(f"⚠️  No data found for {planner_name}")
            continue

        # Get the latest file
        latest_file = max(files, key=lambda p: p.stat().st_mtime)
        print(f"✅ Loading {planner_name}: {latest_file.name}")

        # Load episode data
        df = pd.read_csv(latest_file)
        print(f"   Episodes: {len(df)}")

        # Store data
        planner_data[planner_name] = df

    # Create the plot with slightly wider aspect ratio
    fig, ax = plt.subplots(figsize=(12, 10))

    # Set light grey background
    ax.set_facecolor('#f0f0f0')

    colors = {'Isaac Sim RRT': '#1f77b4', 'A*': '#2ca02c'}

    # Find the minimum starting reward across all planners for baseline alignment
    min_start_reward = min([df['total_reward'].iloc[0] for df in planner_data.values()])
    print(f"\n📍 Baseline reward (minimum starting reward): {min_start_reward:.2f}\n")

    # Process and plot each planner
    all_planner_data = {}

    for planner_name, df in planner_data.items():
        # Get episode numbers and rewards
        episodes = np.arange(1, len(df) + 1)
        episode_rewards = df['total_reward'].values

        # Align to baseline: shift all rewards so they start from the same minimum
        reward_offset = episode_rewards[0] - min_start_reward
        episode_rewards_aligned = episode_rewards - reward_offset

        print(f"📊 {planner_name}: {len(episodes)} episodes")

        # Apply very light Gaussian smoothing to preserve detail
        sigma = 1  # Minimal smoothing for episode-level data
        reward_smooth = gaussian_filter1d(episode_rewards_aligned, sigma=sigma)

        # Calculate rolling standard deviation for variance bands
        window = 5  # Window for std calculation (episodes)
        reward_std = pd.Series(episode_rewards_aligned).rolling(window=window, min_periods=1, center=True).std()
        reward_std = reward_std.fillna(0).values

        # Reduce variance bands to 1/4 of their actual values
        reward_std = reward_std * 0.25

        # Store data for plotting
        all_planner_data[planner_name] = {
            'episodes': episodes,
            'reward_smooth': reward_smooth,
            'reward_std': reward_std,
            'reward_raw': episode_rewards_aligned  # Store raw aligned rewards
        }

    # Calculate and print average rewards over all episodes
    print(f"\n📊 Average Rewards over {num_episodes} episodes:")
    print("=" * 80)
    for planner_name, data in all_planner_data.items():
        avg_reward = np.mean(data['reward_raw'])
        std_reward = np.std(data['reward_raw'])
        print(f"{planner_name:20s}: {avg_reward:8.2f} ± {std_reward:.2f}")
    print("=" * 80)
    print()

    # Plot all planners
    for planner_name, data in all_planner_data.items():
        episodes = data['episodes']
        reward_smooth = data['reward_smooth']
        reward_std = data['reward_std']

        # Plot the smooth line (DOUBLED THICKNESS TO 1.0)
        plt.plot(episodes, reward_smooth,
                label=planner_name, color=colors[planner_name], linewidth=1.0, alpha=0.9)

        # # Add shaded variance area (using standard deviation)
        # plt.fill_between(episodes,
        #                 reward_smooth - reward_std,
        #                 reward_smooth + reward_std,
        #                 color=colors[planner_name], alpha=0.2, interpolate=True)

    # Set x-axis to start from 1
    ax.set_xlim(left=1, right=num_episodes)

    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Episode Reward', fontsize=12, fontweight='bold')
    ax.set_title('MASAC: Episode Reward vs Episode Number (A* vs Isaac Sim RRT)\nGrid 4x4, 9 Cubes',
              fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Planner', fontsize=10, title_fontsize=11, loc='best')

    # Darker grey dashed grid lines on light grey background
    ax.grid(True, linestyle='--', linewidth=0.8, color='#b0b0b0', alpha=0.7)
    plt.tight_layout()

    # Save the plot
    output_path = log_dir / "reward_vs_episode_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved plot to: {output_path}")

    print(f"\n{'='*80}\n")


def compare_planner_statistics():
    """Compare statistics for all three planners"""
    print("\n" + "="*100)
    print("MASAC Planner Statistics Comparison - Grid 4x4, 9 Cubes")
    print("="*100 + "\n")

    log_dir = Path("cobotproject/scripts/Reinforcement Learning/MASAC/logs")

    # Find the latest episode files for each planner
    planners = {
        'Isaac Sim RRT': 'masac_rrt_isaacsim_grid4_cubes9_*_episode_log.csv',
        'RRT Viz': 'masac_rrt_viz_grid4_cubes9_*_episode_log.csv',
        'A*': 'masac_astar_grid4_cubes9_*_episode_log.csv'
    }

    results = {}

    for planner_name, pattern in planners.items():
        files = list(log_dir.glob(pattern))

        if not files:
            print(f"⚠️  No data found for {planner_name}")
            continue

        # Get the latest file
        latest_file = max(files, key=lambda p: p.stat().st_mtime)
        print(f"✅ Loading {planner_name}: {latest_file.name}")

        # Load and calculate statistics
        df = pd.read_csv(latest_file)

        # Calculate timestep statistics
        total_timesteps = df['episode_length'].sum()
        truncated_episodes = (df['episode_length'] >= 50).sum()
        avg_episode_length = df['episode_length'].mean()
        std_episode_length = df['episode_length'].std()

        stats = {
            'num_episodes': len(df),
            'total_timesteps': total_timesteps,
            'truncated_episodes': truncated_episodes,
            'avg_episode_length': avg_episode_length,
            'std_episode_length': std_episode_length,
            'avg_reward': df['total_reward'].mean(),
            'std_reward': df['total_reward'].std(),
            'avg_reshuffles': df['reshuffles_performed'].mean(),
            'std_reshuffles': df['reshuffles_performed'].std(),
            'avg_distance_reduced': df['total_distance_reduced'].mean(),
            'std_distance_reduced': df['total_distance_reduced'].std(),
            'avg_time_saved': df['total_time_saved'].mean(),
            'std_time_saved': df['total_time_saved'].std(),
            'avg_cubes_picked': df['cubes_picked'].mean(),
            'success_rate': (df['success'].sum() / len(df) * 100) if 'success' in df.columns else None
        }

        results[planner_name] = stats

    if not results:
        print("❌ No data found!")
        return

    # Create comparison table
    print(f"\n{'='*100}")
    print("Performance Comparison")
    print(f"{'='*100}\n")

    # Header
    print(f"{'Metric':<30} {'Isaac Sim RRT':<25} {'RRT Viz':<25} {'A*':<25}")
    print("-" * 100)

    # Metrics to compare
    metrics = [
        ('Episodes', 'num_episodes', ''),
        ('Total Timesteps', 'total_timesteps', ''),
        ('Truncated Episodes', 'truncated_episodes', ''),
        ('Avg Episode Length', 'avg_episode_length', '±'),
        ('Avg Reward', 'avg_reward', '±'),
        ('Avg Reshuffles', 'avg_reshuffles', '±'),
        ('Avg Distance Reduced (m)', 'avg_distance_reduced', '±'),
        ('Avg Time Saved (s)', 'avg_time_saved', '±'),
        ('Avg Cubes Picked', 'avg_cubes_picked', ''),
        ('Success Rate (%)', 'success_rate', '')
    ]

    for metric_name, key, symbol in metrics:
        row = f"{metric_name:<30}"

        for planner in ['Isaac Sim RRT', 'RRT Viz', 'A*']:
            if planner in results:
                value = results[planner].get(key)

                if value is None:
                    row += f"{'N/A':<25}"
                elif symbol == '±' and key.startswith('avg_'):
                    # Show mean ± std
                    std_key = key.replace('avg_', 'std_')
                    std_value = results[planner].get(std_key, 0)
                    row += f"{value:.2f} ± {std_value:.2f}".ljust(25)
                elif key in ['num_episodes', 'total_timesteps', 'truncated_episodes']:
                    row += f"{int(value):<25}"
                else:
                    row += f"{value:.2f}".ljust(25)
            else:
                row += f"{'N/A':<25}"

        print(row)

    print(f"\n{'='*100}")

    # Ranking
    print("\n📊 Rankings (Higher is Better):")
    print("-" * 100)

    ranking_metrics = [
        ('Reward', 'avg_reward'),
        ('Distance Reduced', 'avg_distance_reduced'),
        ('Time Saved', 'avg_time_saved'),
        ('Cubes Picked', 'avg_cubes_picked')
    ]

    for metric_name, key in ranking_metrics:
        values = [(planner, results[planner][key]) for planner in results if key in results[planner]]
        values.sort(key=lambda x: x[1], reverse=True)

        print(f"\n{metric_name}:")
        for i, (planner, value) in enumerate(values, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            print(f"  {emoji} {i}. {planner:<20} {value:.4f}")

    # Reshuffles (Lower is Better)
    print(f"\nReshuffles (Lower is Better):")
    values = [(planner, results[planner]['avg_reshuffles']) for planner in results]
    values.sort(key=lambda x: x[1])

    for i, (planner, value) in enumerate(values, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        print(f"  {emoji} {i}. {planner:<20} {value:.2f}")

    print(f"\n{'='*100}\n")

    # Save to CSV
    df_data = []
    for planner, stats in results.items():
        row = {'Planner': planner}
        row.update(stats)
        df_data.append(row)

    df = pd.DataFrame(df_data)
    output_path = log_dir / "planner_comparison.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Saved comparison to: {output_path}\n")

    return results


def analyze_log3_by_seed():
    """Analyze log3 directory with seed-wise breakdown"""
    print("\n" + "="*120)
    print("MASAC Multi-Seed Analysis - Log3 Directory")
    print("="*120 + "\n")

    log_dir = Path("cobotproject/scripts/Reinforcement Learning/MASAC/logs/log3")

    if not log_dir.exists():
        print(f"❌ Directory not found: {log_dir}")
        return

    # Find all summary JSON files
    summary_files = list(log_dir.glob("*_summary.json"))

    if not summary_files:
        print(f"❌ No summary files found in {log_dir}")
        return

    print(f"✅ Found {len(summary_files)} summary files\n")

    # Load all data
    all_data = []
    for file in summary_files:
        with open(file, 'r') as f:
            data = json.load(f)
            all_data.append(data)
            print(f"   Loaded: {file.name}")
            print(f"      Planner: {data['planner']}, Seed: {data['seed']}, Episodes: {data['num_episodes']}")

    print(f"\n{'='*120}\n")

    # Organize data by planner and seed
    planner_seed_data = {}
    for data in all_data:
        planner = data['planner']
        seed = data['seed']

        if planner not in planner_seed_data:
            planner_seed_data[planner] = {}

        planner_seed_data[planner][seed] = data

    # Print detailed metrics for each planner and seed
    print("📊 DETAILED METRICS BY PLANNER AND SEED")
    print("="*120)

    for planner in sorted(planner_seed_data.keys()):
        print(f"\n{'='*120}")
        print(f"🎯 {planner}")
        print(f"{'='*120}\n")

        seeds = sorted(planner_seed_data[planner].keys())

        # Create header
        header = f"{'Metric':<35}"
        for seed in seeds:
            header += f"{'Seed ' + str(seed):>20}"
        print(header)
        print("-" * 120)

        # Metrics to display
        metrics = [
            ('Avg Reward', 'reward', 'mean'),
            ('Avg Cubes Picked', 'cubes_picked', 'mean'),
            ('Success Rate (%)', 'success_rate', 'percentage'),
            ('Avg Reshuffles', 'reshuffles', 'mean'),
            ('Avg Distance Reduced (m)', 'distance_reduced_m', 'mean'),
            ('Avg Time Saved (s)', 'time_saved_s', 'mean'),
            ('Total Episodes', 'num_episodes', None),
            ('Total Timesteps', 'total_timesteps', None),
        ]

        for metric_name, key, subkey in metrics:
            row = f"{metric_name:<35}"

            for seed in seeds:
                data = planner_seed_data[planner][seed]

                if subkey is None:
                    # Direct value
                    value = data.get(key, 'N/A')
                    if isinstance(value, (int, float)):
                        row += f"{value:>20.2f}"
                    else:
                        row += f"{str(value):>20}"
                else:
                    # Nested value
                    nested_data = data.get(key, {})
                    if isinstance(nested_data, dict):
                        value = nested_data.get(subkey, 'N/A')
                        if isinstance(value, (int, float)):
                            row += f"{value:>20.2f}"
                        else:
                            row += f"{str(value):>20}"
                    else:
                        row += f"{'N/A':>20}"

            print(row)

        print()

    # Summary comparison table - BY SEED
    print(f"\n{'='*120}")
    print("📊 RESULTS BY SEED - PLANNERS AS ROWS, METRICS AS COLUMNS")
    print(f"{'='*120}\n")

    # Get all unique seeds
    all_seeds = set()
    for planner_data in planner_seed_data.values():
        all_seeds.update(planner_data.keys())
    all_seeds = sorted(all_seeds)

    # Print table for each seed
    for seed in all_seeds:
        print(f"\n{'='*120}")
        print(f"SEED {seed}")
        print(f"{'='*120}")

        # Create header
        header = f"{'Planner':<25}"
        header += f"{'Avg Reward':>15} {'Cubes Picked':>15} {'Success %':>12} {'Reshuffles':>12} {'Dist (m)':>12} {'Time (s)':>12}"
        print(header)
        print("-" * 120)

        # Print each planner's data for this seed
        for planner in sorted(planner_seed_data.keys()):
            if seed in planner_seed_data[planner]:
                data = planner_seed_data[planner][seed]

                row = f"{planner:<25}"
                row += f"{data['reward']['mean']:>15.2f}"
                row += f"{data['cubes_picked']['mean']:>15.2f}"
                row += f"{data['success_rate']['percentage']:>12.2f}"
                row += f"{data['reshuffles']['mean']:>12.2f}"
                row += f"{data['distance_reduced_m']['mean']:>12.2f}"
                row += f"{data['time_saved_s']['mean']:>12.2f}"

                print(row)
            else:
                print(f"{planner:<25} {'N/A':>15} {'N/A':>15} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12}")

        print()

    print(f"{'='*120}\n")

    # Calculate mean ± std across seeds for each planner
    print("📊 MEAN ± STD ACROSS SEEDS (MAPPO-style)")
    print("="*120)

    header = f"{'Planner':<25}"
    header += f"{'Avg Reward':>20} {'Cubes Picked':>20} {'Success %':>20} {'Reshuffles':>20} {'Dist (m)':>20} {'Time (s)':>20}"
    print(header)
    print("-" * 120)

    for planner in sorted(planner_seed_data.keys()):
        seeds_data = planner_seed_data[planner]

        # Collect values across seeds
        rewards = [seeds_data[seed]['reward']['mean'] for seed in seeds_data]
        cubes = [seeds_data[seed]['cubes_picked']['mean'] for seed in seeds_data]
        success = [seeds_data[seed]['success_rate']['percentage'] for seed in seeds_data]
        reshuffles = [seeds_data[seed]['reshuffles']['mean'] for seed in seeds_data]
        distance = [seeds_data[seed]['distance_reduced_m']['mean'] for seed in seeds_data]
        time_saved = [seeds_data[seed]['time_saved_s']['mean'] for seed in seeds_data]

        # Calculate mean and std
        row = f"{planner:<25}"
        row += f"{np.mean(rewards):>8.2f}({np.std(rewards):>5.2f})     "
        row += f"{np.mean(cubes):>8.2f}({np.std(cubes):>5.2f})     "
        row += f"{np.mean(success):>8.2f}({np.std(success):>5.2f})     "
        row += f"{np.mean(reshuffles):>8.2f}({np.std(reshuffles):>5.2f})     "
        row += f"{np.mean(distance):>8.2f}({np.std(distance):>5.2f})     "
        row += f"{np.mean(time_saved):>8.2f}({np.std(time_saved):>5.2f})     "

        print(row)

    print(f"\n{'='*120}\n")

    # Save to CSV
    csv_data = []
    for planner in sorted(planner_seed_data.keys()):
        for seed in sorted(planner_seed_data[planner].keys()):
            data = planner_seed_data[planner][seed]
            csv_data.append({
                'Planner': planner,
                'Seed': seed,
                'Avg_Reward': data['reward']['mean'],
                'Avg_Cubes_Picked': data['cubes_picked']['mean'],
                'Success_Rate_%': data['success_rate']['percentage'],
                'Avg_Reshuffles': data['reshuffles']['mean'],
                'Avg_Distance_Reduced_m': data['distance_reduced_m']['mean'],
                'Avg_Time_Saved_s': data['time_saved_s']['mean'],
                'Total_Episodes': data['num_episodes'],
                'Total_Timesteps': data['total_timesteps']
            })

    df = pd.DataFrame(csv_data)
    output_path = log_dir / "multi_seed_analysis.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Saved detailed analysis to: {output_path}\n")

    return planner_seed_data


def main():
    """Run comparison and plot"""
    parser = argparse.ArgumentParser(description='MASAC Data Analysis')
    parser.add_argument('--log3', action='store_true', help='Analyze log3 directory with seed breakdown')
    args = parser.parse_args()

    if args.log3:
        # Analyze log3 directory with seed breakdown
        analyze_log3_by_seed()
    else:
        # Original analysis
        # First show statistics comparison
        compare_planner_statistics()

        # Then plot reward vs timestep
        plot_reward_vs_timestep()


if __name__ == "__main__":
    main()

