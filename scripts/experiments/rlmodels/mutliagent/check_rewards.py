import pandas as pd

# Load both seeds
df_42 = pd.read_csv('two_agent_results/discrete/seed_42/episode_results.csv')
df_123 = pd.read_csv('two_agent_results/discrete/seed_123/episode_results.csv')
df_combined = pd.concat([df_42, df_123])

# Filter models
models = ['DDQN+MASAC', 'Duel-DDQN+MASAC', 'PER-DDQN-Full+MASAC']
df_filtered = df_combined[df_combined['model'].isin(models)]

# Check episodes
episodes = [2, 4, 6, 8, 10, 12, 14, 16, 18]

print('Checking rewards for episodes:', episodes)
print('\nReward statistics by episode:')

for ep in episodes:
    ep_data = df_filtered[df_filtered['episode'] == ep]
    print(f'\nEpisode {ep}:')
    for model in models:
        model_data = ep_data[ep_data['model'] == model]['total_reward']
        if len(model_data) > 0:
            print(f'  {model}: min={model_data.min():.2f}, max={model_data.max():.2f}, mean={model_data.mean():.2f}')

print('\n' + '='*80)
print('Overall statistics:')
print(f'Overall min reward: {df_filtered[df_filtered["episode"].isin(episodes)]["total_reward"].min():.2f}')
print(f'Overall max reward: {df_filtered[df_filtered["episode"].isin(episodes)]["total_reward"].max():.2f}')

# Find episodes with only positive rewards
print('\n' + '='*80)
print('Episodes with only positive rewards:')
positive_episodes = []
for ep in episodes:
    ep_data = df_filtered[df_filtered['episode'] == ep]
    min_reward = ep_data['total_reward'].min()
    if min_reward >= 0:
        positive_episodes.append(ep)
        print(f'  Episode {ep}: min reward = {min_reward:.2f}')

print(f'\nEpisodes with all positive rewards: {positive_episodes}')

