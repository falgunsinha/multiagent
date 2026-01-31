import pandas as pd
import numpy as np

df_42 = pd.read_csv('two_agent_results/discrete/seed_42/episode_results.csv')
df_123 = pd.read_csv('two_agent_results/discrete/seed_123/episode_results.csv')
df_combined = pd.concat([df_42, df_123])

models = ['DDQN+MASAC', 'Heuristic']
df_filtered = df_combined[df_combined['model'].isin(models)]

episodes = list(range(1, 18))
df_ep = df_filtered[df_filtered['episode'].isin(episodes)]

df_ep['success_rate'] = (df_ep['cubes_picked'] / df_ep['num_cubes']) * 100

print('Success rate range:')
print(f"Min: {df_ep['success_rate'].min():.2f}%")
print(f"Max: {df_ep['success_rate'].max():.2f}%")
print(f"Mean: {df_ep['success_rate'].mean():.2f}%")

