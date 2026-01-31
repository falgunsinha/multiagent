import json
import pandas as pd

# Read WandB summary for Isaac Sim
with open(r'wandb\run-20260114_171728-xac7e6kh\files\wandb-summary.json', 'r') as f:
    isaac_wandb = json.load(f)

# Read CSV for Isaac Sim
isaac_csv = pd.read_csv(r'..\logs\mappo_rrt_isaacsim_grid4_cubes9_20260114_171716_20260114_171747_episode_log.csv')

# Read CSV for A*
astar_csv = pd.read_csv(r'..\logs\mappo_astar_grid4_cubes9_20260114_173732_20260114_173746_episode_log.csv')

# Read CSV for RRT Viz
rrt_viz_csv = pd.read_csv(r'..\logs\mappo_rrt_viz_grid4_cubes9_20260114_173017_20260114_173040_episode_log.csv')

print('=' * 100)
print('PERFORMANCE COMPARISON: Isaac Sim vs A* vs RRT Viz')
print('=' * 100)
print()

print('ISAAC SIM (RRT + Isaac Sim):')
print('-' * 100)
print(f'Episodes: {len(isaac_csv)}')
final_isaac = isaac_csv.iloc[-1]
print(f'Final episode reward: {final_isaac["total_reward"]:.2f}')
print(f'Final episode steps: {int(final_isaac["episode_length"])}')
print(f'Final episode reshuffles: {int(final_isaac["reshuffles_performed"])}')
print(f'Distance reduced: {isaac_wandb.get("episode/distance_reduced", 0):.6f} m')
print(f'Time saved: {isaac_wandb.get("episode/time_saved", 0):.6f} s')
print(f'Cubes picked: {isaac_wandb.get("episode/cubes_picked", 0)} / 9')
print()

print('A* (A* pathfinding):')
print('-' * 100)
print(f'Episodes: {len(astar_csv)}')
final_astar = astar_csv.iloc[-1]
print(f'Final episode reward: {final_astar["total_reward"]:.2f}')
print(f'Final episode steps: {int(final_astar["episode_length"])}')
print(f'Final episode reshuffles: {int(final_astar["reshuffles_performed"])}')
if 'total_distance_reduced' in astar_csv.columns:
    print(f'Distance reduced: {final_astar["total_distance_reduced"]:.6f} m')
if 'total_time_saved' in astar_csv.columns:
    print(f'Time saved: {final_astar["total_time_saved"]:.6f} s')
print()

print('RRT VIZ (RRT + Visualization):')
print('-' * 100)
print(f'Episodes: {len(rrt_viz_csv)}')
final_rrt = rrt_viz_csv.iloc[-1]
print(f'Final episode reward: {final_rrt["total_reward"]:.2f}')
print(f'Final episode steps: {int(final_rrt["episode_length"])}')
print(f'Final episode reshuffles: {int(final_rrt["reshuffles_performed"])}')
if 'total_distance_reduced' in rrt_viz_csv.columns:
    print(f'Distance reduced: {final_rrt["total_distance_reduced"]:.6f} m')
if 'total_time_saved' in rrt_viz_csv.columns:
    print(f'Time saved: {final_rrt["total_time_saved"]:.6f} s')
print()

print('=' * 100)
print('KEY DIFFERENCES')
print('=' * 100)
print()
print('Isaac Sim Issues:')
print(f'  - NEGATIVE distance reduced: {isaac_wandb.get("episode/distance_reduced", 0):.6f} m')
print(f'  - NEGATIVE time saved: {isaac_wandb.get("episode/time_saved", 0):.6f} s')
print(f'  - Only {isaac_wandb.get("episode/cubes_picked", 0)}/9 cubes picked')
print(f'  - Episodes terminate early')
print()
print('Possible Causes:')
print('  1. Reshuffling moves cubes to unreachable positions')
print('  2. Action masking blocks cubes after reshuffling')
print('  3. Distance/time calculation is incorrect')
print('  4. RRT planning fails after reshuffling')
print('  5. execute_picks=False causes different behavior')

