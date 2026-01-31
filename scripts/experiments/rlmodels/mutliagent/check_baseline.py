import pandas as pd
import numpy as np

# Load data
df_42 = pd.read_csv('cobotproject/scripts/experiments/rlmodels/mutliagent/two_agent_results/discrete/seed_42/episode_results.csv')
df_123 = pd.read_csv('cobotproject/scripts/experiments/rlmodels/mutliagent/two_agent_results/discrete/seed_123/episode_results.csv')
df = pd.concat([df_42, df_123])

print("="*80)
print("CHECKING BASELINE CALCULATION APPROACH")
print("="*80)

# Check Heuristic statistics
heuristic = df[df['model'] == 'Heuristic']
print('\nHeuristic distance_reduced statistics:')
print(f'  Mean: {heuristic["total_distance_reduced"].mean():.4f}m')
print(f'  Std: {heuristic["total_distance_reduced"].std():.4f}m')
print(f'  Min: {heuristic["total_distance_reduced"].min():.4f}m')
print(f'  Max: {heuristic["total_distance_reduced"].max():.4f}m')

# Check DDQN+MASAC statistics
ddqn = df[df['model'] == 'DDQN+MASAC']
print('\nDDQN+MASAC distance_reduced statistics:')
print(f'  Mean: {ddqn["total_distance_reduced"].mean():.4f}m')
print(f'  Std: {ddqn["total_distance_reduced"].std():.4f}m')
print(f'  Min: {ddqn["total_distance_reduced"].min():.4f}m')
print(f'  Max: {ddqn["total_distance_reduced"].max():.4f}m')

print("\n" + "="*80)
print("APPROACH: Use maximum distance_reduced as baseline proxy")
print("="*80)
print("\nRationale:")
print("  - The maximum distance_reduced across all models in an episode")
print("  - Represents the best possible improvement achievable")
print("  - Can be used as a proxy for the baseline scenario")

# For each episode, find the maximum distance_reduced
print("\nSample calculation for Episode 1, Seed 42:")
ep1_seed42 = df[(df['episode'] == 1) & (df['seed'] == 42)]
max_dist = ep1_seed42['total_distance_reduced'].max()
print(f"  Maximum distance_reduced: {max_dist:.4f}m")
print(f"\n  Model breakdown:")
for _, row in ep1_seed42.iterrows():
    efficiency = (row['total_distance_reduced'] / max_dist) * 100
    print(f"    {row['model']:25s}: {row['total_distance_reduced']:.4f}m -> Efficiency: {efficiency:.1f}%")

print("\n" + "="*80)
print("ALTERNATIVE: Use a fixed baseline estimate")
print("="*80)
print("\nSince we don't have actual traveled distance, we can estimate:")
print("  - Grid size: 4x4")
print("  - Number of cubes: 9")
print("  - Typical robot travel distance per cube: ~0.5-1.0m")
print("  - Estimated baseline total distance: 4.5-9.0m")
print("\nLet's use the average of (Heuristic + DDQN) as baseline estimate:")
avg_baseline_estimate = (heuristic["total_distance_reduced"].mean() + ddqn["total_distance_reduced"].mean())
print(f"  Baseline estimate: {avg_baseline_estimate:.4f}m")
print(f"\n  Heuristic efficiency: {(heuristic['total_distance_reduced'].mean() / avg_baseline_estimate) * 100:.1f}%")
print(f"  DDQN+MASAC efficiency: {(ddqn['total_distance_reduced'].mean() / avg_baseline_estimate) * 100:.1f}%")

