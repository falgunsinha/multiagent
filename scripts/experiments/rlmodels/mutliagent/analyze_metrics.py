import pandas as pd
import numpy as np

# Load both seeds
df_42 = pd.read_csv('two_agent_results/discrete/seed_42/episode_results.csv')
df_123 = pd.read_csv('two_agent_results/discrete/seed_123/episode_results.csv')
df_combined = pd.concat([df_42, df_123])

# Filter models
models = ['DDQN+MASAC', 'Heuristic', 'PER-DDQN-Light+MASAC']
df_filtered = df_combined[df_combined['model'].isin(models)]

# Episodes to analyze
episodes = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

print("="*100)
print("METRIC ANALYSIS FOR GRAPH RECOMMENDATIONS")
print("="*100)

# Analyze each metric
metrics = {
    'cubes_picked': 'Pick Success (count)',
    'total_distance_reduced': 'Distance Reduced (m)',
    'total_time_saved': 'Time Saved (s)'
}

for metric_col, metric_name in metrics.items():
    print(f"\n{metric_name}:")
    print("-"*100)
    
    for model in models:
        model_data = df_filtered[df_filtered['model'] == model]
        episode_data = model_data[model_data['episode'].isin(episodes)]
        
        print(f"\n  {model}:")
        
        # Overall statistics
        overall_mean = episode_data[metric_col].mean()
        overall_std = episode_data[metric_col].std()
        overall_min = episode_data[metric_col].min()
        overall_max = episode_data[metric_col].max()
        
        print(f"    Overall: mean={overall_mean:.3f}, std={overall_std:.3f}, min={overall_min:.3f}, max={overall_max:.3f}")
        
        # Variance across episodes
        episode_means = episode_data.groupby('episode')[metric_col].mean()
        episode_variance = episode_means.std()
        
        print(f"    Variance across episodes: {episode_variance:.3f}")
        
        # Check for trends
        correlation = episode_data['episode'].corr(episode_data[metric_col])
        print(f"    Correlation with episode number: {correlation:.3f}")
        
        # Range
        value_range = overall_max - overall_min
        print(f"    Range: {value_range:.3f}")

print("\n" + "="*100)
print("RECOMMENDATIONS:")
print("="*100)

# Calculate success rate
print("\nSuccess Rate Analysis:")
for model in models:
    model_data = df_filtered[df_filtered['model'] == model]
    episode_data = model_data[model_data['episode'].isin(episodes)]
    
    num_cubes_per_episode = episode_data['num_cubes'].iloc[0]
    total_cubes = len(episode_data) * num_cubes_per_episode
    total_picked = episode_data['cubes_picked'].sum()
    success_rate = (total_picked / total_cubes) * 100
    
    print(f"  {model}: {success_rate:.2f}%")
    
    # Per episode success rate
    episode_success = []
    for ep in episodes:
        ep_data = episode_data[episode_data['episode'] == ep]
        if len(ep_data) > 0:
            ep_total = len(ep_data) * num_cubes_per_episode
            ep_picked = ep_data['cubes_picked'].sum()
            ep_rate = (ep_picked / ep_total) * 100
            episode_success.append(ep_rate)
    
    print(f"    Episode success rates: min={min(episode_success):.2f}%, max={max(episode_success):.2f}%, std={np.std(episode_success):.2f}%")

