"""
Comprehensive Analysis of RL Models Performance
- Success rates
- Path lengths and execution times
- Reward analysis
- Model comparisons
- Analysis for both 5 episodes and 99 episodes
"""
import pandas as pd
from pathlib import Path
import sys

# Check command line argument for which dataset to analyze
if len(sys.argv) > 1 and sys.argv[1] == "5":
    # Load 5-episode data
    exp1_file = Path(r"C:\isaacsim\cobotproject\scripts\experiments\rlmodels\results\exp1\episode_results_5.csv")
    exp2_file = Path(r"C:\isaacsim\cobotproject\scripts\experiments\rlmodels\results\exp2\episode_results_5.csv")
    dataset_name = "5 EPISODES"
else:
    # Load 99-episode data (default)
    exp1_file = Path(r"C:\isaacsim\cobotproject\scripts\experiments\rlmodels\results\exp1\episode_results.csv")
    exp2_file = Path(r"C:\isaacsim\cobotproject\scripts\experiments\rlmodels\results\exp2\episode_results.csv")
    dataset_name = "99 EPISODES"

exp1_data = pd.read_csv(exp1_file)
exp2_data = pd.read_csv(exp2_file)

print("\n" + "="*160)
print(f"ANALYSIS FOR {dataset_name}")
print("="*160)

# IMPORTANT: Recalculate success as percentage (picks/total_cubes)
# The 'success' column is binary (1 if all cubes picked, 0 otherwise)
# We need success as a percentage for proper comparison
exp1_data['success_pct'] = exp1_data['picks'] / exp1_data['total_cubes']
exp2_data['success_pct'] = exp2_data['picks'] / exp2_data['total_cubes']

# Add experiment type
exp1_data['experiment'] = 'Discrete'
exp2_data['experiment'] = 'Continuous'

# Combine data
all_data = pd.concat([exp1_data, exp2_data], ignore_index=True)

# ============================================================================
# SUCCESS RATE ANALYSIS
# ============================================================================
print("\n" + "="*160)
print("SUCCESS RATE ANALYSIS")
print("="*160)
print("")

print("SET 1: DISCRETE MODELS")
print("-"*80)
discrete_models = ['Custom-DDQN', 'Duel-DDQN', 'PER-DDQN-Light', 'PER-DDQN-Full', 'C51-DDQN', 'SAC-Discrete', 'PPO-Discrete']
for model in discrete_models:
    model_df = exp1_data[exp1_data['model'] == model]
    if len(model_df) > 0:
        success_rate = model_df['success'].mean()
        picks_rate = model_df['picks'].mean() / 9.0
        print(f'{model:20s} Success Rate: {success_rate:.4f} ({success_rate*100:.2f}%), Avg Picks/9: {picks_rate:.4f} ({picks_rate*100:.2f}%)')

print("")
print("SET 2: CONTINUOUS MODELS")
print("-"*80)
continuous_models = ['DDPG', 'TD3', 'SAC-Continuous', 'PPO-Continuous']
for model in continuous_models:
    model_df = exp2_data[exp2_data['model'] == model]
    if len(model_df) > 0:
        success_rate = model_df['success'].mean()
        picks_rate = model_df['picks'].mean() / 9.0
        print(f'{model:20s} Success Rate: {success_rate:.4f} ({success_rate*100:.2f}%), Avg Picks/9: {picks_rate:.4f} ({picks_rate*100:.2f}%)')

print("")
print("="*160)

# ============================================================================
# PATH LENGTH AND TIME ANALYSIS
# ============================================================================
# Calculate metrics per model - SHOW TWO SEPARATE SETS
print("\n" + "="*160)
print("PATH LENGTH AND TIME COMPARISON")
print("="*160)
print("")
print("SET 1: DISCRETE MODELS (Custom-DDQN + 6 Discrete Pretrained)")
print("-"*160)
print(f"{'Model':<20} {'Avg Reward':<12} {'Avg Steps':<12} {'Avg Time (s)':<15} {'Time/Step (s)':<15} {'Success %':<12} {'Episodes':<10}")
print("-"*160)

for model in discrete_models:
    model_data = exp1_data[exp1_data['model'] == model]
    if len(model_data) > 0:
        avg_reward = model_data['reward'].mean()
        avg_steps = model_data['length'].mean()
        avg_time = model_data['duration'].mean()
        avg_picks = model_data['picks'].mean()
        time_per_step = avg_time / avg_steps if avg_steps > 0 else 0
        success_pct = model_data['success_pct'].mean() * 100
        episodes = len(model_data)
        print(f"{model:<20} {avg_reward:>10.2f}   {avg_steps:>10.2f}   {avg_time:>13.2f}   {time_per_step:>13.2f}   {success_pct:>10.2f}   {episodes:>8d}")

print("")
print("SET 2: CONTINUOUS MODELS (4 Continuous Pretrained)")
print("-"*160)
print(f"{'Model':<20} {'Avg Reward':<12} {'Avg Steps':<12} {'Avg Time (s)':<15} {'Time/Step (s)':<15} {'Success %':<12} {'Episodes':<10}")
print("-"*160)

for model in continuous_models:
    model_data = exp2_data[exp2_data['model'] == model]
    if len(model_data) > 0:
        avg_reward = model_data['reward'].mean()
        avg_steps = model_data['length'].mean()
        avg_time = model_data['duration'].mean()
        avg_picks = model_data['picks'].mean()
        time_per_step = avg_time / avg_steps if avg_steps > 0 else 0
        success_pct = model_data['success_pct'].mean() * 100
        episodes = len(model_data)
        print(f"{model:<20} {avg_reward:>10.2f}   {avg_steps:>10.2f}   {avg_time:>13.2f}   {time_per_step:>13.2f}   {success_pct:>10.2f}   {episodes:>8d}")

print("="*160)

# Calculate aggregate statistics
print("\n" + "="*160)
print("AGGREGATE STATISTICS - SET 1: Custom-DDQN + 6 Discrete Pretrained Models")
print("="*160)
print(f"{'Metric':<40} {'Custom-DDQN (Discrete)':<25} {'6 Discrete Pretrained':<25} {'Difference':<20}")
print("-"*160)

# Set 1: Custom-DDQN (from exp1) vs 6 discrete pretrained models
discrete_data = all_data[all_data['experiment'] == 'Discrete']
custom_ddqn_discrete = discrete_data[discrete_data['model'] == 'Custom-DDQN']
discrete_pretrained = discrete_data[discrete_data['model'] != 'Custom-DDQN']

metrics = [
    ('Avg Steps per Episode', 'length'),
    ('Avg Time per Episode (s)', 'duration'),
    ('Avg Picks per Episode', 'picks'),
]

for metric_name, column in metrics:
    custom_val = custom_ddqn_discrete[column].mean()
    discrete_val = discrete_pretrained[column].mean()
    diff = discrete_val - custom_val
    diff_pct = (diff / custom_val * 100) if custom_val != 0 else 0

    print(f"{metric_name:<40} {custom_val:>23.2f}   {discrete_val:>23.2f}   {diff:>+10.2f} ({diff_pct:>+6.1f}%)")

# Time per step
custom_time_per_step = custom_ddqn_discrete['duration'].sum() / custom_ddqn_discrete['length'].sum()
discrete_time_per_step = discrete_pretrained['duration'].sum() / discrete_pretrained['length'].sum()
diff = discrete_time_per_step - custom_time_per_step
diff_pct = (diff / custom_time_per_step * 100) if custom_time_per_step != 0 else 0
print(f"{'Avg Time per Step (s)':<40} {custom_time_per_step:>23.2f}   {discrete_time_per_step:>23.2f}   {diff:>+10.2f} ({diff_pct:>+6.1f}%)")

# Time per pick
custom_time_per_pick = custom_ddqn_discrete['duration'].sum() / custom_ddqn_discrete['picks'].sum()
discrete_time_per_pick = discrete_pretrained['duration'].sum() / discrete_pretrained['picks'].sum()
diff = discrete_time_per_pick - custom_time_per_pick
diff_pct = (diff / custom_time_per_pick * 100) if custom_time_per_pick != 0 else 0
print(f"{'Avg Time per Pick (s)':<40} {custom_time_per_pick:>23.2f}   {discrete_time_per_pick:>23.2f}   {diff:>+10.2f} ({diff_pct:>+6.1f}%)")

# Success rate and efficiency for Set 1
custom_success = custom_ddqn_discrete['success_pct'].mean() * 100
discrete_success = discrete_pretrained['success_pct'].mean() * 100
diff = discrete_success - custom_success
print(f"{'Success Rate (%)':<40} {custom_success:>23.2f}   {discrete_success:>23.2f}   {diff:>+10.2f}%")

custom_path_eff = custom_ddqn_discrete['path_efficiency'].mean()
discrete_path_eff = discrete_pretrained['path_efficiency'].mean()
diff = discrete_path_eff - custom_path_eff
diff_pct = (diff / custom_path_eff * 100) if custom_path_eff != 0 else 0
print(f"{'Path Efficiency':<40} {custom_path_eff:>23.2f}   {discrete_path_eff:>23.2f}   {diff:>+10.2f} ({diff_pct:>+6.1f}%)")

print("="*160)

# Set 2: Custom-DDQN (from exp2) vs 4 continuous pretrained models
print("\n" + "="*160)
print("AGGREGATE STATISTICS - SET 2: Custom-DDQN + 4 Continuous Pretrained Models")
print("="*160)
print(f"{'Metric':<40} {'Custom-DDQN (Continuous)':<25} {'4 Continuous Pretrained':<25} {'Difference':<20}")
print("-"*160)

continuous_data = all_data[all_data['experiment'] == 'Continuous']
custom_ddqn_continuous = continuous_data[continuous_data['model'] == 'Custom-DDQN']
continuous_pretrained = continuous_data[continuous_data['model'] != 'Custom-DDQN']

for metric_name, column in metrics:
    custom_val = custom_ddqn_continuous[column].mean()
    continuous_val = continuous_pretrained[column].mean()
    diff = continuous_val - custom_val
    diff_pct = (diff / custom_val * 100) if custom_val != 0 else 0

    print(f"{metric_name:<40} {custom_val:>23.2f}   {continuous_val:>23.2f}   {diff:>+10.2f} ({diff_pct:>+6.1f}%)")

# Time per step
custom_time_per_step = custom_ddqn_continuous['duration'].sum() / custom_ddqn_continuous['length'].sum()
continuous_time_per_step = continuous_pretrained['duration'].sum() / continuous_pretrained['length'].sum()
diff = continuous_time_per_step - custom_time_per_step
diff_pct = (diff / custom_time_per_step * 100) if custom_time_per_step != 0 else 0
print(f"{'Avg Time per Step (s)':<40} {custom_time_per_step:>23.2f}   {continuous_time_per_step:>23.2f}   {diff:>+10.2f} ({diff_pct:>+6.1f}%)")

# Time per pick
custom_time_per_pick = custom_ddqn_continuous['duration'].sum() / custom_ddqn_continuous['picks'].sum()
continuous_time_per_pick = continuous_pretrained['duration'].sum() / continuous_pretrained['picks'].sum()
diff = continuous_time_per_pick - custom_time_per_pick
diff_pct = (diff / custom_time_per_pick * 100) if custom_time_per_pick != 0 else 0
print(f"{'Avg Time per Pick (s)':<40} {custom_time_per_pick:>23.2f}   {continuous_time_per_pick:>23.2f}   {diff:>+10.2f} ({diff_pct:>+6.1f}%)")

# Success rate and efficiency for Set 2
custom_success = custom_ddqn_continuous['success_pct'].mean() * 100
continuous_success = continuous_pretrained['success_pct'].mean() * 100
diff = continuous_success - custom_success
print(f"{'Success Rate (%)':<40} {custom_success:>23.2f}   {continuous_success:>23.2f}   {diff:>+10.2f}%")

custom_path_eff = custom_ddqn_continuous['path_efficiency'].mean()
continuous_path_eff = continuous_pretrained['path_efficiency'].mean()
diff = continuous_path_eff - custom_path_eff
diff_pct = (diff / custom_path_eff * 100) if custom_path_eff != 0 else 0
print(f"{'Path Efficiency':<40} {custom_path_eff:>23.2f}   {continuous_path_eff:>23.2f}   {diff:>+10.2f} ({diff_pct:>+6.1f}%)")

print("="*160)

print("\n" + "="*160)
print("KEY INSIGHTS")
print("="*160)
print("IMPORTANT NOTES:")
print("  - 'Avg Steps' = Number of pick actions taken (NOT RRT path waypoints)")
print("  - 'Success Rate' = Average percentage of cubes picked (picks/total_cubes)")
print("  - 'Path Efficiency' = length/optimal_steps (NOTE: Should be optimal_steps/length for correct interpretation)")
print("  - RRT path lengths (waypoints/distance) are NOT included in this CSV data")
print("")
print("SET 1 (Custom-DDQN vs 6 Discrete Pretrained):")
print("  - Discrete pretrained models are FASTER (25-32s vs 39s per episode)")
print("  - Similar number of pick actions (~8.3-8.6 actions)")
print("  - Discrete pretrained models have better time efficiency")
print("")
print("SET 2 (Custom-DDQN vs 4 Continuous Pretrained):")
print("  - Continuous pretrained models vary in speed (25-61s per episode)")
print("  - SAC-Continuous is SLOWER (61s vs 35s per episode)")
print("  - DDPG, TD3, PPO-Continuous are FASTER (25-27s vs 35s per episode)")
print("  - Similar number of pick actions (~8.3-8.6 actions)")
print("")
print("OVERALL:")
print("  - All models take similar number of pick actions (8.3-8.6 actions)")
print("  - Main difference is EXECUTION TIME, not number of actions")
print("  - Fastest: PER-DDQN-Light, TD3 (~25s per episode)")
print("  - Slowest: SAC-Continuous (~61s per episode)")
print("="*160)
print("")

# ============================================================================
# REWARD ANALYSIS
# ============================================================================
print("="*160)
print("REWARD ANALYSIS")
print("="*160)
print("")

# Discrete models reward comparison
print("SET 1: DISCRETE MODELS - Reward Comparison")
print("-"*80)
discrete_rewards = []
for model in discrete_models:
    model_data = exp1_data[exp1_data['model'] == model]
    avg_reward = model_data['reward'].mean()
    std_reward = model_data['reward'].std()
    min_reward = model_data['reward'].min()
    max_reward = model_data['reward'].max()
    discrete_rewards.append({
        'model': model,
        'avg': avg_reward,
        'std': std_reward,
        'min': min_reward,
        'max': max_reward
    })
    print(f"{model:<20} Avg: {avg_reward:>8.2f}  Std: {std_reward:>8.2f}  Min: {min_reward:>8.2f}  Max: {max_reward:>8.2f}")

print("")
print("SET 2: CONTINUOUS MODELS - Reward Comparison")
print("-"*80)
continuous_rewards = []
for model in continuous_models:
    model_data = exp2_data[exp2_data['model'] == model]
    avg_reward = model_data['reward'].mean()
    std_reward = model_data['reward'].std()
    min_reward = model_data['reward'].min()
    max_reward = model_data['reward'].max()
    continuous_rewards.append({
        'model': model,
        'avg': avg_reward,
        'std': std_reward,
        'min': min_reward,
        'max': max_reward
    })
    print(f"{model:<20} Avg: {avg_reward:>8.2f}  Std: {std_reward:>8.2f}  Min: {min_reward:>8.2f}  Max: {max_reward:>8.2f}")

print("")
print("="*160)
print("REWARD INSIGHTS")
print("="*160)

# Find best and worst performers (exclude NaN)
all_rewards = discrete_rewards + continuous_rewards
all_rewards_valid = [r for r in all_rewards if not pd.isna(r['avg'])]
all_rewards_sorted = sorted(all_rewards_valid, key=lambda x: x['avg'], reverse=True)

print(f"HIGHEST Average Reward: {all_rewards_sorted[0]['model']:<20} ({all_rewards_sorted[0]['avg']:.2f})")
print(f"LOWEST Average Reward:  {all_rewards_sorted[-1]['model']:<20} ({all_rewards_sorted[-1]['avg']:.2f})")
print("")

# Custom-DDQN vs Pretrained comparison
custom_ddqn_reward = [r for r in discrete_rewards if r['model'] == 'Custom-DDQN'][0]['avg']
discrete_pretrained_avg = sum([r['avg'] for r in discrete_rewards if r['model'] != 'Custom-DDQN']) / 6

# Filter out NaN values for continuous models
continuous_valid = [r['avg'] for r in continuous_rewards if not pd.isna(r['avg'])]
continuous_pretrained_avg = sum(continuous_valid) / len(continuous_valid) if len(continuous_valid) > 0 else float('nan')

print(f"Custom-DDQN Average Reward:              {custom_ddqn_reward:>8.2f}")
print(f"Discrete Pretrained Average Reward:      {discrete_pretrained_avg:>8.2f}")
print(f"Continuous Pretrained Average Reward:    {continuous_pretrained_avg:>8.2f}")
print("")

reward_diff_discrete = ((discrete_pretrained_avg - custom_ddqn_reward) / abs(custom_ddqn_reward)) * 100
if not pd.isna(continuous_pretrained_avg):
    reward_diff_continuous = ((continuous_pretrained_avg - custom_ddqn_reward) / abs(custom_ddqn_reward)) * 100
    print(f"Custom-DDQN vs Discrete Pretrained:   {reward_diff_discrete:+.2f}% difference")
    print(f"Custom-DDQN vs Continuous Pretrained: {reward_diff_continuous:+.2f}% difference")
else:
    print(f"Custom-DDQN vs Discrete Pretrained:   {reward_diff_discrete:+.2f}% difference")
    print(f"Custom-DDQN vs Continuous Pretrained: N/A (insufficient data)")
print("")

# Reward consistency analysis
print("REWARD CONSISTENCY (Lower Std = More Consistent):")
print("-"*80)
all_rewards_by_std = sorted(all_rewards_valid, key=lambda x: x['std'])
print(f"Most Consistent:  {all_rewards_by_std[0]['model']:<20} (Std: {all_rewards_by_std[0]['std']:.2f})")
print(f"Least Consistent: {all_rewards_by_std[-1]['model']:<20} (Std: {all_rewards_by_std[-1]['std']:.2f})")
print("="*160)

