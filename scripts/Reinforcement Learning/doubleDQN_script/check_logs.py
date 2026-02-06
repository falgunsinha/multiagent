import pandas as pd
print("=" * 60)
print("TRAINING CSV ANALYSIS")
print("=" * 60)
df_train = pd.read_csv('logs/ddqn_rrt_viz_grid4_cubes6_20251219_191646_training.csv')
print(f"Total rows: {len(df_train)}")
print(f"Last step: {df_train['step'].max()}")

df_at_30k = df_train[df_train['step'] <= 30000]
print(f"Rows at step <= 30000: {len(df_at_30k)}")
print(f"Episode at step 30000: {df_at_30k['episode'].max()}")
print(f"Rows to REMOVE: {len(df_train) - len(df_at_30k)}")
print("\n" + "=" * 60)
print("EPISODES CSV ANALYSIS")
print("=" * 60)
df_episodes = pd.read_csv('logs/ddqn_rrt_viz_grid4_cubes6_20251219_191646_episodes.csv')
print(f"Total rows: {len(df_episodes)}")
print(f"Last episode: {df_episodes['episode'].max()}")

max_episode_at_30k = df_at_30k['episode'].max()
df_episodes_at_30k = df_episodes[df_episodes['episode'] <= max_episode_at_30k]
print(f"Rows at episode <= {max_episode_at_30k}: {len(df_episodes_at_30k)}")
print(f"Rows to REMOVE: {len(df_episodes) - len(df_episodes_at_30k)}")
print("=" * 60)

