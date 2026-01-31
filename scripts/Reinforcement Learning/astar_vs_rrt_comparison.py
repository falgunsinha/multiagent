import pandas as pd
import numpy as np

# Data from training results
data = {
    'Method': ['astar', 'astar', 'astar', 'heuristic', 'heuristic', 'heuristic', 'rrt', 'rrt', 'rrt'],
    'Grid': ['3x3', '4x4', '4x4', '3x3', '4x4', '4x4', '3x3', '4x4', '4x4'],
    'Cubes': [4, 6, 9, 4, 6, 9, 4, 6, 9],
    'Final_Reward': [100.77, 120.85, 141.17, 106.88, 138.67, 172.01, 101.70, 129.36, 163.70],
    'Ep_Length': [4.05, 6.20, 9.75, 4.47, 6.22, 9.52, 4.03, 6.27, 9.22],
    'Entropy_Loss': [-0.25, -0.51, -0.83, -0.22, -0.51, -0.85, -0.20, -0.30, -0.74],
    'Value_Loss': [0.061, 0.007, 0.029, 0.009, 0.036, 0.021, 0.002, 0.003, 0.011],
    'Policy_Loss': [-0.0416, -0.0636, -0.0745, -0.0285, -0.0552, -0.0760, -0.0077, -0.0214, -0.0472]
}

df = pd.DataFrame(data)

print('=' * 80)
print('COMPREHENSIVE COMPARISON: A* vs RRT')
print('=' * 80)
print()

# Filter only A* and RRT
astar_df = df[df['Method'] == 'astar']
rrt_df = df[df['Method'] == 'rrt']

print('1. FINAL REWARD COMPARISON (Higher is Better)')
print('-' * 80)
for i in range(3):
    config = f'{astar_df.iloc[i]["Grid"]}, {int(astar_df.iloc[i]["Cubes"])} cubes'
    astar_reward = astar_df.iloc[i]['Final_Reward']
    rrt_reward = rrt_df.iloc[i]['Final_Reward']
    diff = rrt_reward - astar_reward
    winner = 'RRT' if diff > 0 else 'A*'
    
    print(f'{config:20s}: A* = {astar_reward:6.2f}  |  RRT = {rrt_reward:6.2f}  |  Diff = {diff:+6.2f}  |  Winner: {winner}')

print()
print(f'Average A* Reward:  {astar_df["Final_Reward"].mean():.2f}')
print(f'Average RRT Reward: {rrt_df["Final_Reward"].mean():.2f}')
print(f'RRT Advantage:      {rrt_df["Final_Reward"].mean() - astar_df["Final_Reward"].mean():+.2f} points')
print()

print('2. EPISODE LENGTH COMPARISON (Lower is Better - More Efficient)')
print('-' * 80)
for i in range(3):
    config = f'{astar_df.iloc[i]["Grid"]}, {int(astar_df.iloc[i]["Cubes"])} cubes'
    astar_len = astar_df.iloc[i]['Ep_Length']
    rrt_len = rrt_df.iloc[i]['Ep_Length']
    diff = astar_len - rrt_len
    winner = 'RRT' if diff > 0 else 'A*'
    
    print(f'{config:20s}: A* = {astar_len:5.2f}  |  RRT = {rrt_len:5.2f}  |  Diff = {diff:+5.2f}  |  Winner: {winner}')

print()
print(f'Average A* Length:  {astar_df["Ep_Length"].mean():.2f}')
print(f'Average RRT Length: {rrt_df["Ep_Length"].mean():.2f}')
print(f'RRT Efficiency:     {astar_df["Ep_Length"].mean() - rrt_df["Ep_Length"].mean():+.2f} steps shorter')
print()

print('3. REWARD PER STEP (Efficiency Metric)')
print('-' * 80)
for i in range(3):
    config = f'{astar_df.iloc[i]["Grid"]}, {int(astar_df.iloc[i]["Cubes"])} cubes'
    astar_rps = astar_df.iloc[i]['Final_Reward'] / astar_df.iloc[i]['Ep_Length']
    rrt_rps = rrt_df.iloc[i]['Final_Reward'] / rrt_df.iloc[i]['Ep_Length']
    diff = rrt_rps - astar_rps
    winner = 'RRT' if diff > 0 else 'A*'
    
    print(f'{config:20s}: A* = {astar_rps:5.2f}  |  RRT = {rrt_rps:5.2f}  |  Diff = {diff:+5.2f}  |  Winner: {winner}')

astar_rps_avg = (astar_df['Final_Reward'] / astar_df['Ep_Length']).mean()
rrt_rps_avg = (rrt_df['Final_Reward'] / rrt_df['Ep_Length']).mean()
print()
print(f'Average A* Reward/Step:  {astar_rps_avg:.2f}')
print(f'Average RRT Reward/Step: {rrt_rps_avg:.2f}')
print(f'RRT Advantage:           {rrt_rps_avg - astar_rps_avg:+.2f}')
print()

print('4. LEARNING STABILITY (Lower Loss Variance is Better)')
print('-' * 80)
print(f'Value Loss Variance:  A* = {astar_df["Value_Loss"].var():.6f}  |  RRT = {rrt_df["Value_Loss"].var():.6f}')
print(f'Policy Loss Variance: A* = {astar_df["Policy_Loss"].var():.6f}  |  RRT = {rrt_df["Policy_Loss"].var():.6f}')
print()

print('5. SCALABILITY (Performance as Problem Size Increases)')
print('-' * 80)
print('Reward improvement from smallest to largest config:')
astar_improvement = astar_df.iloc[2]['Final_Reward'] - astar_df.iloc[0]['Final_Reward']
rrt_improvement = rrt_df.iloc[2]['Final_Reward'] - rrt_df.iloc[0]['Final_Reward']
print(f'  A*:  {astar_df.iloc[0]["Final_Reward"]:.2f} -> {astar_df.iloc[2]["Final_Reward"]:.2f}  (+{astar_improvement:.2f})')
print(f'  RRT: {rrt_df.iloc[0]["Final_Reward"]:.2f} -> {rrt_df.iloc[2]["Final_Reward"]:.2f}  (+{rrt_improvement:.2f})')
print()

print('=' * 80)
print('OVERALL VERDICT')
print('=' * 80)
print()

# Count wins
reward_wins = sum([rrt_df.iloc[i]['Final_Reward'] > astar_df.iloc[i]['Final_Reward'] for i in range(3)])
efficiency_wins = sum([rrt_df.iloc[i]['Ep_Length'] < astar_df.iloc[i]['Ep_Length'] for i in range(3)])

print(f'RRT wins in Reward:     {reward_wins}/3 configs')
print(f'RRT wins in Efficiency: {efficiency_wins}/3 configs')
print()

if reward_wins >= 2 and efficiency_wins >= 2:
    print('WINNER: RRT')
    print('   RRT consistently outperforms A* in both reward and efficiency.')
elif reward_wins >= 2:
    print('WINNER: RRT (by Reward)')
    print('   RRT achieves higher rewards but may be less efficient.')
else:
    print('WINNER: A*')
    print('   A* is more competitive overall.')

print()
print('KEY INSIGHTS:')
print(f'  - RRT achieves {rrt_rps_avg - astar_rps_avg:+.2f} higher reward per step on average')
print(f'  - RRT completes episodes {astar_df["Ep_Length"].mean() - rrt_df["Ep_Length"].mean():+.2f} steps faster on average')
print(f'  - RRT total reward advantage: {rrt_df["Final_Reward"].mean() - astar_df["Final_Reward"].mean():+.2f} points')
print(f'  - RRT scales better: +{rrt_improvement:.2f} vs A* +{astar_improvement:.2f} from small to large')
print()

