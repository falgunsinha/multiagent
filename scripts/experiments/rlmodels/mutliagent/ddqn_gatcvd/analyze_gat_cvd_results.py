"""
Analyze GAT+CVD test results for all 8 discrete models
Report mean ± standard deviation across seeds (42 and 123)
Generate metrics comparison table and graphs
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline

# Load both seeds
seed_42_csv = Path("cobotproject/scripts/experiments/rlmodels/mutliagent/ddqn_gatcvd/gat_cvd_test_results/discrete/seed_42/episode_results.csv")
seed_123_csv = Path("cobotproject/scripts/experiments/rlmodels/mutliagent/ddqn_gatcvd/gat_cvd_test_results/discrete/seed_123/episode_results.csv")

if not seed_42_csv.exists() or not seed_123_csv.exists():
    print(f"⚠️  Files not found")
    exit()

df_42 = pd.read_csv(seed_42_csv)
df_123 = pd.read_csv(seed_123_csv)

print("="*160)
print("GAT+CVD TEST RESULTS - ALL DISCRETE MODELS (Mean ± Std)")
print("="*160)
print(f"Seed 42: {len(df_42)} total episodes, {len(df_42) // len(df_42['model'].unique())} episodes per model")
print(f"Seed 123: {len(df_123)} total episodes, {len(df_123) // len(df_123['model'].unique())} episodes per model")
print(f"Models: {df_42['model'].unique()}")
print("="*160)

# Define custom order for models (matching the thesis)
model_order = [
    'DDQN+GAT',  # This is DDQN+GAT+CVD (DDQN + mGAT)
    'Heuristic',
    'Duel-DDQN+SAC',
    'PER-DDQN-Full+SAC',
    'PER-DDQN-Light+SAC',
    'C51-DDQN+SAC',
    'PPO-Discrete+SAC',
    'SAC-Discrete+SAC'
]

# Collect results for both seeds
results = []

for model in model_order:
    # Seed 42
    model_df_42 = df_42[df_42['model'] == model]
    num_episodes_42 = len(model_df_42)
    num_cubes_per_episode = model_df_42['num_cubes'].iloc[0]
    
    total_cubes_42 = num_episodes_42 * num_cubes_per_episode
    total_cubes_picked_42 = model_df_42['cubes_picked'].sum()
    success_rate_42 = (total_cubes_picked_42 / total_cubes_42) * 100
    
    if model == 'Heuristic':
        reward_42 = None
    else:
        reward_42 = model_df_42['total_reward'].mean()
    
    distance_42 = model_df_42['total_distance_reduced'].mean()
    time_42 = model_df_42['total_time_saved'].mean()
    distance_eff_42 = model_df_42['distance_efficiency'].mean()
    time_eff_42 = model_df_42['time_efficiency'].mean()
    
    # Seed 123
    model_df_123 = df_123[df_123['model'] == model]
    num_episodes_123 = len(model_df_123)
    
    total_cubes_123 = num_episodes_123 * num_cubes_per_episode
    total_cubes_picked_123 = model_df_123['cubes_picked'].sum()
    success_rate_123 = (total_cubes_picked_123 / total_cubes_123) * 100
    
    if model == 'Heuristic':
        reward_123 = None
    else:
        reward_123 = model_df_123['total_reward'].mean()
    
    distance_123 = model_df_123['total_distance_reduced'].mean()
    time_123 = model_df_123['total_time_saved'].mean()
    distance_eff_123 = model_df_123['distance_efficiency'].mean()
    time_eff_123 = model_df_123['time_efficiency'].mean()
    
    # Calculate mean and std across seeds
    success_mean = np.mean([success_rate_42, success_rate_123])
    success_std = np.std([success_rate_42, success_rate_123], ddof=1)  # Sample std
    
    if model == 'Heuristic':
        reward_mean = None
        reward_std = None
    else:
        reward_mean = np.mean([reward_42, reward_123])
        reward_std = np.std([reward_42, reward_123], ddof=1)
    
    distance_mean = np.mean([distance_42, distance_123])
    distance_std = np.std([distance_42, distance_123], ddof=1)
    
    time_mean = np.mean([time_42, time_123])
    time_std = np.std([time_42, time_123], ddof=1)
    
    distance_eff_mean = np.mean([distance_eff_42, distance_eff_123])
    distance_eff_std = np.std([distance_eff_42, distance_eff_123], ddof=1)
    
    time_eff_mean = np.mean([time_eff_42, time_eff_123])
    time_eff_std = np.std([time_eff_42, time_eff_123], ddof=1)
    
    # Format model name for display
    if model == 'Heuristic':
        model_name = 'Heuristic'
    elif model == 'DDQN+GAT':
        model_name = 'DDQN + mGAT'  # DDQN+GAT+CVD
    else:
        model_name = model.replace('+SAC', ' + SAC')
    
    results.append({
        'model_name': model_name,
        'reward_mean': reward_mean,
        'reward_std': reward_std,
        'success_mean': success_mean,
        'success_std': success_std,
        'distance_mean': distance_mean,
        'distance_std': distance_std,
        'time_mean': time_mean,
        'time_std': time_std,
        'distance_eff_mean': distance_eff_mean,
        'distance_eff_std': distance_eff_std,
        'time_eff_mean': time_eff_mean,
        'time_eff_std': time_eff_std
    })

# Print table
print(f"\n{'Model':<25} {'Reward':>15} {'Pick Success (%)':>25} {'Distance Reduced (m)':>30} {'Time Saved (s)':>25} {'Distance Efficiency (%)':>30} {'Time Efficiency (%)':>30}")
print("-"*160)

for result in results:
    model_name = result['model_name']

    if result['reward_mean'] is None:
        reward_str = "N/A"
    else:
        reward_str = f"{result['reward_mean']:.2f} ± {result['reward_std']:.2f}"

    success_str = f"{result['success_mean']:.2f} ± {result['success_std']:.2f}"
    distance_str = f"{result['distance_mean']:.4f} ± {result['distance_std']:.4f}"
    time_str = f"{result['time_mean']:.4f} ± {result['time_std']:.4f}"
    distance_eff_str = f"{result['distance_eff_mean']:.2f} ± {result['distance_eff_std']:.2f}"
    time_eff_str = f"{result['time_eff_mean']:.2f} ± {result['time_eff_std']:.2f}"

    print(f"{model_name:<25} {reward_str:>15} {success_str:>25} {distance_str:>30} {time_str:>25} {distance_eff_str:>30} {time_eff_str:>30}")

print("="*160)

# Generate LaTeX table
print("\n" + "="*160)
print("LaTeX TABLE CODE:")
print("="*160)

# Find maximum values for each metric (excluding Heuristic for reward)
max_reward = max([r['reward_mean'] for r in results if r['reward_mean'] is not None])
max_success = max([r['success_mean'] for r in results])
max_distance = max([r['distance_mean'] for r in results])
max_time = max([r['time_mean'] for r in results])
max_distance_eff = max([r['distance_eff_mean'] for r in results])
max_time_eff = max([r['time_eff_mean'] for r in results])

latex_code = """\\begin{table}[h]
\\centering
\\caption{Discrete Algorithm Performance - Mean ± Std across Seeds 42 and 123}
\\label{tab:gat_cvd_discrete_results}
\\begin{tabular}{lcccccc}
\\toprule
Model & Reward & Pick Success (\\%) & Distance Reduced (m) & Time Saved (s) & Distance Efficiency (\\%) & Time Efficiency (\\%) \\\\
\\midrule
"""

for result in results:
    model_name = result['model_name']
    is_heuristic = (model_name == 'Heuristic')

    # Reward column
    if result['reward_mean'] is None:
        reward_str = "N/A"
    else:
        reward_val = f"{result['reward_mean']:.2f} $\\pm$ {result['reward_std']:.2f}"
        if result['reward_mean'] == max_reward:
            reward_str = f"\\textbf{{{reward_val}}}"
        else:
            reward_str = reward_val

    # Pick Success column
    success_val = f"{result['success_mean']:.2f} $\\pm$ {result['success_std']:.2f}"
    if result['success_mean'] == max_success:
        success_str = f"\\textbf{{{success_val}}}"
    else:
        success_str = success_val

    # Distance Reduced column
    distance_val = f"{result['distance_mean']:.4f} $\\pm$ {result['distance_std']:.4f}"
    if result['distance_mean'] == max_distance:
        distance_str = f"\\textbf{{{distance_val}}}"
    else:
        distance_str = distance_val

    # Time Saved column
    time_val = f"{result['time_mean']:.4f} $\\pm$ {result['time_std']:.4f}"
    if result['time_mean'] == max_time:
        time_str = f"\\textbf{{{time_val}}}"
    else:
        time_str = time_val

    # Distance Efficiency column
    distance_eff_val = f"{result['distance_eff_mean']:.2f} $\\pm$ {result['distance_eff_std']:.2f}"
    if result['distance_eff_mean'] == max_distance_eff:
        distance_eff_str = f"\\textbf{{{distance_eff_val}}}"
    else:
        distance_eff_str = distance_eff_val

    # Time Efficiency column
    time_eff_val = f"{result['time_eff_mean']:.2f} $\\pm$ {result['time_eff_std']:.2f}"
    if result['time_eff_mean'] == max_time_eff:
        time_eff_str = f"\\textbf{{{time_eff_val}}}"
    else:
        time_eff_str = time_eff_val

    latex_code += f"{model_name} & {reward_str} & {success_str} & {distance_str} & {time_str} & {distance_eff_str} & {time_eff_str} \\\\\n"

latex_code += """\\bottomrule
\\end{tabular}
\\end{table}"""

print(latex_code)
print("="*160)

