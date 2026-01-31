import pandas as pd
import numpy as np

# Analyze all three methods
files = [
    ('A*', 'logs/ddqn_astar_grid4_cubes9_20251214_042536_training.csv'),
    ('RRT Viz', 'logs/ddqn_rrt_viz_grid4_cubes9_20251214_044921_training.csv'),
    ('RRT Isaac', 'logs/ddqn_rrt_isaacsim_grid4_cubes9_20251216_010838_training.csv')
]

print('='*80)
print('COMPREHENSIVE DDQN CONVERGENCE ANALYSIS (4x4 grid, 9 cubes)')
print('='*80)
print('\nPART 1: BASIC ANALYSIS')
print('='*80)

for method_name, file_path in files:
    print(f'\n{"="*80}')
    print(f'Method: {method_name}')
    print(f'{"="*80}')
    
    df = pd.read_csv(file_path)
    
    # Count timesteps per episode
    episode_lengths = df.groupby('episode').size()
    
    # For RRT Isaac, check if bug is at resume point
    if method_name == 'RRT Isaac':
        print('\n[CHECKING RESUME BUG]')
        anomalies = episode_lengths[episode_lengths != 9]
        if len(anomalies) > 0:
            first_anomaly_ep = anomalies.index[0]
            first_anomaly_step = df[df['episode'] == first_anomaly_ep]['step'].iloc[0]
            print(f'  First anomaly at episode {first_anomaly_ep}, timestep {first_anomaly_step}')
            print(f'  Expected resume point: ~45,000 timesteps')
            if 44000 <= first_anomaly_step <= 46000:
                print(f'  ✓ BUG CONFIRMED: Anomaly starts at resume point!')
            else:
                print(f'  ⚠️  Anomaly NOT at resume point (unexpected)')
        
        # Exclude buggy episodes for analysis
        print(f'\n[EXCLUDING BUGGY EPISODES]')
        valid_episodes = episode_lengths[episode_lengths == 9].index
        df_clean = df[df['episode'].isin(valid_episodes)]
        print(f'  Original episodes: {len(episode_lengths)}')
        print(f'  Valid episodes: {len(valid_episodes)}')
        print(f'  Excluded episodes: {len(episode_lengths) - len(valid_episodes)}')
        df = df_clean
    
    # Calculate episode total rewards
    episode_rewards = df.groupby('episode')['reward'].sum()
    
    # Calculate rolling average (window=100)
    window = 100
    rolling_avg = episode_rewards.rolling(window=window, min_periods=1).mean()
    
    print(f'\n[BASIC STATS]')
    print(f'  Total episodes analyzed: {len(episode_rewards)}')
    print(f'  Total timesteps: {len(df)}')
    print(f'  Timesteps per episode: {len(df) / len(episode_rewards):.1f}')
    
    # Show reward progression
    print(f'\n[REWARD PROGRESSION] (Rolling avg, window={window})')
    print(f'{"Episode":<12} {"Timestep":<12} {"Reward":<12} {"Change"}')
    print('-' * 55)
    
    checkpoints = [100, 500, 1000, 2000, 3000, 4000, 5000]
    prev_reward = None
    for ep in checkpoints:
        if ep < len(rolling_avg):
            timestep = ep * 9
            reward = rolling_avg.iloc[ep]
            change = f'{reward - prev_reward:+.2f}' if prev_reward is not None else '-'
            print(f'{ep:<12} {timestep:<12} {reward:<12.2f} {change}')
            prev_reward = reward
    
    # Final performance
    if len(episode_rewards) >= 100:
        final_ep = len(episode_rewards) - 1
        final_timestep = final_ep * 9
        final_reward = rolling_avg.iloc[final_ep]
        change = f'{final_reward - prev_reward:+.2f}' if prev_reward is not None else '-'
        print(f'{final_ep:<12} {final_timestep:<12} {final_reward:<12.2f} {change}')
    
    # Convergence detection: when does rolling avg stabilize?
    # Method: Find when rolling avg stays within ±5% of final value for 500+ episodes
    print(f'\n[CONVERGENCE ANALYSIS]')
    
    if len(episode_rewards) >= 600:
        final_reward = rolling_avg.iloc[-100:].mean()
        threshold_low = final_reward * 0.95
        threshold_high = final_reward * 1.05
        
        # Find first episode where rolling avg enters and stays in ±5% range
        in_range = (rolling_avg >= threshold_low) & (rolling_avg <= threshold_high)
        
        convergence_ep = None
        for i in range(len(in_range) - 500):
            if in_range.iloc[i:i+500].all():
                convergence_ep = i
                break
        
        if convergence_ep is not None:
            convergence_step = convergence_ep * 9
            convergence_reward = rolling_avg.iloc[convergence_ep]
            print(f'  Final reward (last 100 ep avg): {final_reward:.2f}')
            print(f'  Convergence threshold: ±5% ({threshold_low:.2f} to {threshold_high:.2f})')
            print(f'  Converged at episode: {convergence_ep}')
            print(f'  Converged at timestep: {convergence_step}')
            print(f'  Percentage of training: {convergence_step/45000*100:.1f}%')
            print(f'  Reward at convergence: {convergence_reward:.2f}')
        else:
            print(f'  Final reward: {final_reward:.2f}')
            print(f'  ⚠️  Did not reach stable convergence (±5% for 500 episodes)')
    
    # Final performance stats
    print(f'\n[FINAL PERFORMANCE] (Last 100 episodes)')
    final_100 = episode_rewards.iloc[-100:]
    print(f'  Mean reward: {final_100.mean():.2f}')
    print(f'  Std reward: {final_100.std():.2f}')
    print(f'  Min reward: {final_100.min():.2f}')
    print(f'  Max reward: {final_100.max():.2f}')

print(f'\n{"="*80}')
print('PART 2: PRECISE CONVERGENCE DETECTION (A* and RRT Viz only)')
print('='*80)

# Precise convergence analysis for A* and RRT Viz
for method_name, file_path in files[:2]:  # Only A* and RRT Viz
    print(f'\n{"="*80}')
    print(f'Method: {method_name} - PRECISE CONVERGENCE')
    print(f'{"="*80}')

    df = pd.read_csv(file_path)
    episode_rewards = df.groupby('episode')['reward'].sum()

    # Check when epsilon reaches 0.01 (stops exploring)
    print(f'\n[EPSILON ANALYSIS]')
    epsilon_001 = df[df['epsilon'] <= 0.01]
    if len(epsilon_001) > 0:
        first_exploit_step = epsilon_001.iloc[0]['step']
        first_exploit_ep = epsilon_001.iloc[0]['episode']
        print(f'  Epsilon reaches 0.01 (pure exploitation) at:')
        print(f'    Timestep: {int(first_exploit_step)}')
        print(f'    Episode: {int(first_exploit_ep)}')
        print(f'    Percentage of training: {first_exploit_step/50000*100:.1f}%')

    # Final performance baseline (last 500 episodes)
    final_baseline = episode_rewards.iloc[-500:].mean()
    print(f'\n[BASELINE]')
    print(f'  Final performance (last 500 episodes): {final_baseline:.2f}')

    # Method 1: Find when rolling avg reaches 95% of final and stays there
    print(f'\n[METHOD 1: 95% of Final Performance]')
    threshold = 0.95 * final_baseline
    print(f'  Convergence threshold (95%): {threshold:.2f}')

    windows = [10, 50, 100, 200]
    for window in windows:
        rolling_avg = episode_rewards.rolling(window=window, min_periods=1).mean()
        above_threshold = rolling_avg >= threshold

        # Find first sustained convergence
        convergence_ep = None
        for i in range(len(above_threshold) - window):
            if above_threshold.iloc[i:i+window].all():
                convergence_ep = i
                break

        if convergence_ep is not None:
            convergence_step = convergence_ep * 9
            convergence_reward = rolling_avg.iloc[convergence_ep]
            print(f'  Window={window:3d}: Episode {convergence_ep:4d}, Timestep {convergence_step:5d}, Reward {convergence_reward:.2f}')

    # Method 2: Check reward stability AFTER epsilon reaches 0.01
    print(f'\n[METHOD 2: Reward Stability After Epsilon=0.01]')
    rolling_avg_100 = episode_rewards.rolling(window=100, min_periods=1).mean()

    if len(epsilon_001) > 0:
        # Start checking from when epsilon reaches 0.01
        start_ep = int(first_exploit_ep)

        # Check if rewards are stable for next 500 episodes
        if start_ep + 500 < len(rolling_avg_100):
            reward_at_start = rolling_avg_100.iloc[start_ep]
            reward_after_500 = rolling_avg_100.iloc[start_ep + 500]
            improvement = (reward_after_500 - reward_at_start) / reward_at_start * 100

            print(f'  Reward at epsilon=0.01 (ep {start_ep}): {reward_at_start:.2f}')
            print(f'  Reward 500 episodes later (ep {start_ep+500}): {reward_after_500:.2f}')
            print(f'  Improvement: {improvement:+.2f}%')

            if abs(improvement) < 2.0:
                print(f'  ✓ CONVERGED: Rewards stable (<2% change) after epsilon=0.01')
                print(f'  → Convergence timestep: {int(first_exploit_step)}')
            else:
                print(f'  ⚠️  Still improving after epsilon=0.01')

    # Method 3: Find when no significant improvement (< 1% over next 500 episodes)
    print(f'\n[METHOD 3: No Significant Improvement (<1% over next 500 episodes)]')

    convergence_ep = None
    for i in range(100, len(rolling_avg_100) - 500):
        current_reward = rolling_avg_100.iloc[i]
        future_reward = rolling_avg_100.iloc[i+500]
        improvement = (future_reward - current_reward) / current_reward * 100

        if abs(improvement) < 1.0:
            convergence_ep = i
            break

    if convergence_ep is not None:
        convergence_step = convergence_ep * 9
        convergence_reward = rolling_avg_100.iloc[convergence_ep]
        future_reward = rolling_avg_100.iloc[convergence_ep+500]
        print(f'  Converged at Episode {convergence_ep:4d}, Timestep {convergence_step:5d}')
        print(f'  Reward at convergence: {convergence_reward:.2f}')
        print(f'  Reward 500 episodes later: {future_reward:.2f}')
        print(f'  Improvement: {(future_reward - convergence_reward) / convergence_reward * 100:.2f}%')

    # Method 4: Early progression check
    print(f'\n[METHOD 4: Early Reward Progression (Rolling Avg, Window=100)]')
    print(f'{"Episode":<10} {"Timestep":<10} {"Reward":<10} {"Change from Ep 100"}')
    print('-' * 55)

    checkpoints = [10, 20, 50, 100, 109, 200, 300, 500, 700, 1000]  # Added 109 (epsilon=0.01)
    baseline_ep100 = rolling_avg_100.iloc[100] if len(rolling_avg_100) > 100 else 0

    for ep in checkpoints:
        if ep < len(rolling_avg_100):
            timestep = ep * 9
            reward = rolling_avg_100.iloc[ep]
            change = reward - baseline_ep100
            change_pct = (change / baseline_ep100 * 100) if baseline_ep100 > 0 else 0
            marker = ' <-- Epsilon=0.01' if ep == int(first_exploit_ep) else ''
            print(f'{ep:<10} {timestep:<10} {reward:<10.2f} {change:+.2f} ({change_pct:+.1f}%){marker}')

print(f'\n{"="*80}')
print('ANALYSIS COMPLETE')
print('='*80)

