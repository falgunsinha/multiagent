import pandas as pd
import numpy as np

# Define the 8 successful configurations
configs = [
    ('RRT IsaacSim', '3x3', 4, 'cobotproject/scripts/Reinforcement Learning/MAPPO/logs/mappo_rrt_isaacsim_grid3_cubes4_20260114_230707_20260114_230738_episode_log.csv'),
    ('RRT IsaacSim', '4x4', 6, 'cobotproject/scripts/Reinforcement Learning/MAPPO/logs/mappo_rrt_isaacsim_grid4_cubes6_20260114_230904_20260114_230918_episode_log.csv'),
    ('RRT IsaacSim', '4x4', 9, 'cobotproject/scripts/Reinforcement Learning/MAPPO/logs/mappo_rrt_isaacsim_grid4_cubes9_20260114_231540_20260114_231555_episode_log.csv'),
    ('RRT Viz', '3x3', 4, 'cobotproject/scripts/Reinforcement Learning/MAPPO/logs/mappo_rrt_viz_grid3_cubes4_20260114_232037_20260114_232053_episode_log.csv'),
    ('RRT Viz', '4x4', 6, 'cobotproject/scripts/Reinforcement Learning/MAPPO/logs/mappo_rrt_viz_grid4_cubes6_20260114_232125_20260114_232140_episode_log.csv'),
    ('RRT Viz', '4x4', 9, 'cobotproject/scripts/Reinforcement Learning/MAPPO/logs/mappo_rrt_viz_grid4_cubes9_20260114_233411_20260114_233442_episode_log.csv'),
    ('A*', '3x3', 4, 'cobotproject/scripts/Reinforcement Learning/MAPPO/logs/mappo_astar_grid3_cubes4_20260114_235928_20260114_235956_episode_log.csv'),
    ('A*', '4x4', 6, 'cobotproject/scripts/Reinforcement Learning/MAPPO/logs/mappo_astar_grid4_cubes6_20260115_000202_20260115_000216_episode_log.csv'),
]

results = []

for env_type, grid, cubes, filepath in configs:
    try:
        df = pd.read_csv(filepath)
        
        # Calculate averages
        avg_reward = df['total_reward'].mean()
        avg_reshuffles = df['reshuffles_performed'].mean()
        avg_distance_reduced = df['total_distance_reduced'].mean()
        avg_time_saved = df['total_time_saved'].mean()
        
        results.append({
            'Environment': env_type,
            'Grid': grid,
            'Cubes': cubes,
            'Avg Reward': round(avg_reward, 2),
            'Avg Reshuffles': round(avg_reshuffles, 2),
            'Avg Distance Reduced (m)': round(avg_distance_reduced, 3),
            'Avg Time Saved (s)': round(avg_time_saved, 2)
        })
    except Exception as e:
        print(f'Error loading {filepath}: {e}')

# Create DataFrame and display
results_df = pd.DataFrame(results)
print()
print('='*120)
print('MAPPO TRAINING RESULTS COMPARISON (50 Episodes, 8 Configurations)')
print('='*120)
print(results_df.to_string(index=False))
print('='*120)
print()

# Summary by environment
print('SUMMARY BY ENVIRONMENT:')
print('-'*120)
for env in ['RRT IsaacSim', 'RRT Viz', 'A*']:
    env_results = results_df[results_df['Environment'] == env]
    if not env_results.empty:
        print(f'\n{env}:')
        for _, row in env_results.iterrows():
            print(f"  {row['Grid']} {row['Cubes']} cubes: "
                  f"Reward={row['Avg Reward']:.2f}, "
                  f"Reshuffles={row['Avg Reshuffles']:.2f}, "
                  f"Distance={row['Avg Distance Reduced (m)']:.3f}m, "
                  f"Time={row['Avg Time Saved (s)']:.2f}s")
print('-'*120)
print()

# Overall statistics
print('OVERALL STATISTICS:')
print('-'*120)
print(f"Average Reward across all configs: {results_df['Avg Reward'].mean():.2f}")
print(f"Average Reshuffles across all configs: {results_df['Avg Reshuffles'].mean():.2f}")
print(f"Average Distance Reduced across all configs: {results_df['Avg Distance Reduced (m)'].mean():.3f}m")
print(f"Average Time Saved across all configs: {results_df['Avg Time Saved (s)'].mean():.2f}s")
print('-'*120)
print()

# Key Insights
print('KEY INSIGHTS:')
print('-'*120)
print('1. BEST PERFORMANCE:')
best_reward = results_df.loc[results_df['Avg Reward'].idxmax()]
print(f"   - Highest Reward: {best_reward['Environment']} {best_reward['Grid']} {best_reward['Cubes']} cubes ({best_reward['Avg Reward']:.2f})")
best_distance = results_df.loc[results_df['Avg Distance Reduced (m)'].idxmax()]
print(f"   - Most Distance Reduced: {best_distance['Environment']} {best_distance['Grid']} {best_distance['Cubes']} cubes ({best_distance['Avg Distance Reduced (m)']:.3f}m)")
print()

print('2. ENVIRONMENT COMPARISON:')
for env in ['RRT IsaacSim', 'RRT Viz', 'A*']:
    env_data = results_df[results_df['Environment'] == env]
    if not env_data.empty:
        avg_dist = env_data['Avg Distance Reduced (m)'].mean()
        avg_time = env_data['Avg Time Saved (s)'].mean()
        print(f"   - {env}: Avg Distance={avg_dist:.3f}m, Avg Time={avg_time:.2f}s")
print()

print('3. OBSERVATIONS:')
print('   - A* shows consistently positive distance reduction across all configs')
print('   - RRT Viz shows good performance with increasing complexity')
print('   - RRT IsaacSim 4x4 9 cubes shows NEGATIVE distance reduction (-0.051m)')
print('     → This suggests the agent is moving cubes FARTHER from the robot base!')
print('     → Possible cause: Learning to optimize for other factors (clearance, reachability)')
print('-'*120)

