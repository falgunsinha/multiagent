"""
Check filtered data to understand distance traveled and time taken metrics
"""
import pandas as pd

# Load filtered data
df = pd.read_csv('cobotproject/scripts/experiments/rlmodels/mutliagent/ddqn_gatcvd/gat_cvd_test_results/discrete/filtered_combined_results.csv')

print("="*120)
print("FILTERED DATA ANALYSIS")
print("="*120)
print(f"Total episodes: {len(df)}")
print(f"\nEpisodes per model:")
print(df.groupby('model').size())

print("\n" + "="*120)
print("DISTANCE TRAVELED AND TIME TAKEN ANALYSIS")
print("="*120)

models = ['DDQN+GAT', 'Heuristic', 'C51-DDQN+SAC', 'Duel-DDQN+SAC']

for model in models:
    model_df = df[df['model'] == model]
    print(f"\n--- {model} ---")
    print(f"Episodes: {len(model_df)}")
    print(f"Avg distance_traveled: {model_df['total_distance_traveled'].mean():.4f} m")
    print(f"Avg time_taken: {model_df['total_time_taken'].mean():.4f} s")
    print(f"Avg cubes_picked: {model_df['cubes_picked'].mean():.2f}")
    print(f"Avg episode_length: {model_df['episode_length'].mean():.2f}")
    
    # Show range
    print(f"Distance range: {model_df['total_distance_traveled'].min():.4f} - {model_df['total_distance_traveled'].max():.4f}")
    print(f"Time range: {model_df['total_time_taken'].min():.4f} - {model_df['total_time_taken'].max():.4f}")

print("\n" + "="*120)
print("SAMPLE EPISODES - C51-DDQN+SAC (Low distance, moderate time)")
print("="*120)
c51 = df[df['model'] == 'C51-DDQN+SAC'].sort_values('total_distance_traveled')
print(c51[['episode', 'seed', 'total_distance_traveled', 'total_time_taken', 'cubes_picked', 'episode_length']].head(10))

print("\n" + "="*120)
print("SAMPLE EPISODES - DDQN+GAT (Moderate distance, low time)")
print("="*120)
ddqn_gat = df[df['model'] == 'DDQN+GAT'].sort_values('total_time_taken')
print(ddqn_gat[['episode', 'seed', 'total_distance_traveled', 'total_time_taken', 'cubes_picked', 'episode_length']].head(10))

print("\n" + "="*120)
print("KEY INSIGHT: Why can distance be low but time be high?")
print("="*120)
print("""
Possible reasons:
1. Episode Length: Models that fail to pick cubes may have longer episodes (more timesteps)
   but travel less distance if they're stuck or making small movements.

2. Movement Speed: Some models might move slowly (small steps) resulting in:
   - Low total distance traveled
   - High total time taken (many timesteps)

3. Reshuffling: Models with more reshuffling operations may take more time
   but not necessarily travel more distance.

4. Failed Attempts: Models that repeatedly fail to grasp may spend time
   attempting picks without moving much.
""")

print("\n" + "="*120)
print("CORRELATION ANALYSIS")
print("="*120)
for model in models:
    model_df = df[df['model'] == model]
    corr = model_df[['total_distance_traveled', 'total_time_taken', 'episode_length', 'cubes_picked']].corr()
    print(f"\n{model}:")
    print(f"  Distance vs Time correlation: {corr.loc['total_distance_traveled', 'total_time_taken']:.3f}")
    print(f"  Distance vs Episode Length: {corr.loc['total_distance_traveled', 'episode_length']:.3f}")
    print(f"  Time vs Episode Length: {corr.loc['total_time_taken', 'episode_length']:.3f}")

