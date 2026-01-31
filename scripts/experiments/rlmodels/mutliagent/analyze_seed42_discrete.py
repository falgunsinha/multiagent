"""
Analyze only seed 42 results for discrete models with CORRECT success rate calculation
Success rate = (total cubes picked) / (total episodes * num_cubes per episode)
"""
import pandas as pd
from pathlib import Path

csv_path = Path("two_agent_results/discrete/seed_42/episode_results.csv")

if not csv_path.exists():
    print(f"⚠️  File not found: {csv_path}")
    exit()

df = pd.read_csv(csv_path)

print("="*140)
print("DISCRETE ALGORITHMS - SEED 42 ONLY (CORRECT SUCCESS RATE CALCULATION)")
print("="*140)
print(f"Total episodes: {len(df)}")
print(f"Number of models: {len(df['model'].unique())}")
print(f"Models: {sorted(df['model'].unique())}")
print(f"Episodes per model: {len(df) // len(df['model'].unique())}")
print(f"Cubes per episode: {df['num_cubes'].iloc[0]}")
print("="*140)

print(f"\n{'Model':<30} {'Episodes':>8} {'Total Cubes':>12} {'Cubes Picked':>12} {'Success Rate':>15} {'Avg Reward':>15} {'Reshuffles':>12} {'Distance':>12} {'Time Saved':>12}")
print("-"*140)

for model in sorted(df['model'].unique()):
    model_df = df[df['model'] == model]
    num_episodes = len(model_df)
    num_cubes_per_episode = model_df['num_cubes'].iloc[0]

    # Calculate metrics
    total_cubes = num_episodes * num_cubes_per_episode
    total_cubes_picked = model_df['cubes_picked'].sum()
    success_rate = (total_cubes_picked / total_cubes) * 100

    if model == 'Heuristic':
        avg_reward = "N/A"
    else:
        avg_reward = f"{model_df['total_reward'].mean():.2f}"

    avg_reshuffles = model_df['reshuffles_performed'].mean()
    avg_distance = model_df['total_distance_reduced'].mean()
    avg_time = model_df['total_time_saved'].mean()

    print(f"{model:<30} {num_episodes:>8} {total_cubes:>12} {total_cubes_picked:>12} {success_rate:>14.1f}% {avg_reward:>15} {avg_reshuffles:>12.2f} {avg_distance:>12.4f} {avg_time:>12.4f}")

print("="*140)

# Generate LaTeX table
print("\n" + "="*140)
print("LaTeX TABLE CODE:")
print("="*140)

# Define custom order for models
model_order = [
    'DDQN+MASAC',
    'Heuristic',
    'Duel-DDQN+MASAC',
    'PER-DDQN-Full+MASAC',
    'PER-DDQN-Light+MASAC',
    'C51-DDQN+MASAC',
    'PPO-Discrete+MASAC',
    'SAC-Discrete+MASAC'
]

# Collect results in order
latex_results = []
for model in model_order:
    if model not in df['model'].unique():
        continue

    model_df = df[df['model'] == model]
    num_episodes = len(model_df)
    num_cubes_per_episode = model_df['num_cubes'].iloc[0]

    total_cubes = num_episodes * num_cubes_per_episode
    total_cubes_picked = model_df['cubes_picked'].sum()
    success_rate = (total_cubes_picked / total_cubes) * 100

    if model == 'Heuristic':
        avg_reward = None
        model_name = 'Heuristic'
    elif model == 'DDQN+MASAC':
        avg_reward = model_df['total_reward'].mean()
        model_name = 'DDQN+MASAC'
    else:
        avg_reward = model_df['total_reward'].mean()
        model_name = model.replace('+MASAC', ' + SAC')

    avg_distance = model_df['total_distance_reduced'].mean()
    avg_time = model_df['total_time_saved'].mean()

    latex_results.append({
        'model_name': model_name,
        'reward': avg_reward,
        'success': success_rate,
        'distance': avg_distance,
        'time': avg_time
    })

# Print LaTeX table
latex_code = """\\begin{table}[h]
\\centering
\\caption{Discrete Algorithm Performance - Seed 42}
\\label{tab:discrete_seed42}
\\begin{tabular}{lcccc}
\\toprule
Model & Reward & Success (\\%) & Distance Reduced (m) & Time Saved (s) \\\\
\\midrule
"""

for result in latex_results:
    model_name = result['model_name']
    if result['reward'] is None:
        reward = "N/A"
    else:
        reward = f"{result['reward']:.2f}"
    success = f"{result['success']:.1f}"
    distance = f"{result['distance']:.4f}"
    time_saved = f"{result['time']:.4f}"

    latex_code += f"{model_name} & {reward} & {success} & {distance} & {time_saved} \\\\\n"

latex_code += """\\bottomrule
\\end{tabular}
\\end{table}"""

print(latex_code)
print("="*140)

# Show episode-by-episode breakdown for each model
print("\n" + "="*140)
print("EPISODE-BY-EPISODE BREAKDOWN")
print("="*140)

for model in sorted(df['model'].unique()):
    model_df = df[df['model'] == model].sort_values('episode')

    print(f"\n{model}:")
    print(f"{'Episode':>8} {'Cubes Picked':>12} {'Success (bool)':>15} {'Reward':>12} {'Reshuffles':>12} {'Distance':>12} {'Time Saved':>12}")
    print("-"*100)

    for _, row in model_df.iterrows():
        reward_str = "N/A" if model == 'Heuristic' else f"{row['total_reward']:.2f}"
        success_bool = "✓" if row['success'] else "✗"

        print(f"{row['episode']:>8} {row['cubes_picked']:>12} {success_bool:>15} {reward_str:>12} {row['reshuffles_performed']:>12.0f} {row['total_distance_reduced']:>12.4f} {row['total_time_saved']:>12.4f}")

    # Summary for this model
    total_cubes = len(model_df) * model_df['num_cubes'].iloc[0]
    total_picked = model_df['cubes_picked'].sum()
    success_rate = (total_picked / total_cubes) * 100
    print(f"{'TOTAL':<8} {total_picked:>12} / {total_cubes} cubes = {success_rate:.1f}% success rate")

print("\n" + "="*140)
print("✅ ANALYSIS COMPLETE")
print("="*140)

