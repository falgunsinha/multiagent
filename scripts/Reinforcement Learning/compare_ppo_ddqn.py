import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
from scipy.ndimage import gaussian_filter1d
import os

# Set font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # STIX fonts are similar to Times New Roman for math

# Set seaborn style
sns.set_theme(style="whitegrid", palette="muted")
sns.set_context("notebook", font_scale=2.4)  # Double the font size

# Paths
ppo_log_path = r'cobotproject\scripts\Reinforcement Learning\logs\object_selection\rrt_viz_grid4_cubes9_20251210_220052'
ddqn_training_path = r'cobotproject\scripts\Reinforcement Learning\doubleDQN_script\logs\log1\ddqn_rrt_viz_grid4_cubes9_20251220_134808_training.csv'
ddqn_episodes_path = r'cobotproject\scripts\Reinforcement Learning\doubleDQN_script\logs\log1\ddqn_rrt_viz_grid4_cubes9_20251220_134808_episodes.csv'

# Timestep limit for comparison
TIMESTEP_LIMIT = 30000

print("="*80)
print(f"COMPARISON: PPO vs DDQN for RRT VIZ Grid 4x4 with 9 Cubes ({TIMESTEP_LIMIT:,} timesteps)")
print("="*80)

# Load DDQN data
print("\n[1] Loading DDQN Data...")
ddqn_training = pd.read_csv(ddqn_training_path)
ddqn_episodes = pd.read_csv(ddqn_episodes_path)

# Filter to timestep limit
ddqn_training = ddqn_training[ddqn_training['step'] <= TIMESTEP_LIMIT].copy()
max_episode = ddqn_training['episode'].max()
ddqn_episodes = ddqn_episodes[ddqn_episodes['episode'] <= max_episode].copy()

print(f"DDQN Training steps: {ddqn_training['step'].max():,}")
print(f"DDQN Episodes: {len(ddqn_episodes)}")
print(f"DDQN Final avg reward (last 100 eps): {ddqn_training['avg_reward_100'].iloc[-1]:.2f}")
print(f"DDQN Final success rate: {ddqn_training['success_rate'].iloc[-1]:.2%}")
print(f"DDQN Final Q-value: {ddqn_training['q_value'].iloc[-1]:.2f}")

# Load PPO data from TensorBoard
print("\n[2] Loading PPO Data from TensorBoard...")
ea = event_accumulator.EventAccumulator(ppo_log_path)
ea.Reload()

# Extract PPO scalars
ppo_ep_rew = ea.Scalars('rollout/ep_rew_mean')
ppo_ep_len = ea.Scalars('rollout/ep_len_mean')
ppo_explained_var = ea.Scalars('train/explained_variance')  # Explained variance (how well value function predicts returns)

# Convert to DataFrame
ppo_data = pd.DataFrame({
    'step': [x.step for x in ppo_ep_rew],
    'reward': [x.value for x in ppo_ep_rew],
    'episode_length': [x.value for x in ppo_ep_len]
})

ppo_variance_data = pd.DataFrame({
    'step': [x.step for x in ppo_explained_var],
    'explained_variance': [x.value for x in ppo_explained_var]
})

# Filter to timestep limit first
ppo_data = ppo_data[ppo_data['step'] <= TIMESTEP_LIMIT].copy()
ppo_variance_data = ppo_variance_data[ppo_variance_data['step'] <= TIMESTEP_LIMIT].copy()

# Convert timesteps to thousands
ppo_data['step_k'] = ppo_data['step'] / 1000
ppo_variance_data['step_k'] = ppo_variance_data['step'] / 1000
ddqn_training['step_k'] = ddqn_training['step'] / 1000

print(f"PPO Training steps: {ppo_data['step'].max():,}")
print(f"PPO Data points: {len(ppo_data)}")
print(f"PPO Final avg reward: {ppo_data['reward'].iloc[-1]:.2f}")
print(f"PPO Final avg episode length: {ppo_data['episode_length'].iloc[-1]:.2f}")
print(f"PPO Final explained variance: {ppo_variance_data['explained_variance'].iloc[-1]:.4f}")

# Statistical comparison
print("\n" + "="*80)
print("STATISTICAL COMPARISON")
print("="*80)

print("\n[Reward Comparison]")
print(f"PPO  - Mean Reward: {ppo_data['reward'].mean():.2f} ± {ppo_data['reward'].std():.2f}")
print(f"PPO  - Final Reward: {ppo_data['reward'].iloc[-1]:.2f}")
print(f"PPO  - Max Reward: {ppo_data['reward'].max():.2f}")

print(f"\nDDQN - Mean Reward: {ddqn_training['avg_reward_100'].mean():.2f} ± {ddqn_training['avg_reward_100'].std():.2f}")
print(f"DDQN - Final Reward: {ddqn_training['avg_reward_100'].iloc[-1]:.2f}")
print(f"DDQN - Max Reward: {ddqn_training['avg_reward_100'].max():.2f}")

print("\n[Explained Variance Comparison]")
print(f"PPO  - Mean Explained Variance: {ppo_variance_data['explained_variance'].mean():.4f} ± {ppo_variance_data['explained_variance'].std():.4f}")
print(f"PPO  - Final Explained Variance: {ppo_variance_data['explained_variance'].iloc[-1]:.4f}")
print("DDQN - Explained Variance: Not available (Q-learning doesn't use this metric)")

print("\n[Q-Value Analysis - DDQN Only]")
print(f"DDQN - Mean Q-value: {ddqn_training['q_value'].mean():.2f} ± {ddqn_training['q_value'].std():.2f}")
print(f"DDQN - Final Q-value: {ddqn_training['q_value'].iloc[-1]:.2f}")
print(f"DDQN - Max Q-value: {ddqn_training['q_value'].max():.2f}")

print("\n[Learning Efficiency - Convergence Analysis]")
# Convergence thresholds
reward_threshold_high = 180
reward_threshold_mid = 150

# PPO convergence (no change needed)
ppo_converge_high = ppo_data[ppo_data['reward'] > reward_threshold_high]
ppo_converge_mid = ppo_data[ppo_data['reward'] > reward_threshold_mid]

# DDQN convergence - CORRECTED to only consider after 100 episodes
# This ensures we're measuring true 100-episode moving average
ddqn_after_100 = ddqn_training[ddqn_training['episode'] >= 100]
ddqn_converge_high = ddqn_after_100[ddqn_after_100['avg_reward_100'] > reward_threshold_high]
ddqn_converge_mid = ddqn_after_100[ddqn_after_100['avg_reward_100'] > reward_threshold_mid]

print(f"\nConvergence to reward > {reward_threshold_high}:")
if len(ppo_converge_high) > 0:
    ppo_high_step = ppo_converge_high.iloc[0]['step']
    print(f"  PPO  - Steps: {ppo_high_step:,.0f} | Reward: {ppo_converge_high.iloc[0]['reward']:.2f}")
else:
    print(f"  PPO  - Did NOT reach (Max: {ppo_data['reward'].max():.2f})")

if len(ddqn_converge_high) > 0:
    ddqn_high_step = ddqn_converge_high.iloc[0]['step']
    ddqn_high_episode = ddqn_converge_high.iloc[0]['episode']
    print(f"  DDQN - Steps: {ddqn_high_step:,.0f} (Episode {ddqn_high_episode:.0f}) | Reward: {ddqn_converge_high.iloc[0]['avg_reward_100']:.2f}")
    if len(ppo_converge_high) > 0:
        speedup = ppo_high_step / ddqn_high_step
        print(f"  → DDQN converged {speedup:.1f}× faster than PPO")
else:
    print(f"  DDQN - Did NOT reach after ep 100 (Max: {ddqn_after_100['avg_reward_100'].max():.2f})")

print(f"\nConvergence to reward > {reward_threshold_mid}:")
if len(ppo_converge_mid) > 0:
    ppo_mid_step = ppo_converge_mid.iloc[0]['step']
    print(f"  PPO  - Steps: {ppo_mid_step:,.0f} | Reward: {ppo_converge_mid.iloc[0]['reward']:.2f}")
else:
    print(f"  PPO  - Did NOT reach")

if len(ddqn_converge_mid) > 0:
    ddqn_mid_step = ddqn_converge_mid.iloc[0]['step']
    ddqn_mid_episode = ddqn_converge_mid.iloc[0]['episode']
    print(f"  DDQN - Steps: {ddqn_mid_step:,.0f} (Episode {ddqn_mid_episode:.0f}) | Reward: {ddqn_converge_mid.iloc[0]['avg_reward_100']:.2f}")
    if len(ppo_converge_mid) > 0:
        speedup = ppo_mid_step / ddqn_mid_step
        print(f"  → DDQN converged {speedup:.1f}× faster than PPO")
else:
    print(f"  DDQN - Did NOT reach after ep 100")

print("\n[Episode Length Comparison - Policy Efficiency]")
ppo_mean_ep_len = ppo_data['episode_length'].mean()
ppo_final_ep_len = ppo_data['episode_length'].iloc[-1]
ddqn_mean_ep_len = ddqn_training['episode_length'].mean()
ddqn_final_ep_len = ddqn_training['episode_length'].iloc[-1]

print(f"  PPO  - Mean: {ppo_mean_ep_len:.2f} | Final: {ppo_final_ep_len:.2f}")
print(f"  DDQN - Mean: {ddqn_mean_ep_len:.2f} | Final: {ddqn_final_ep_len:.2f}")
if ddqn_final_ep_len < ppo_final_ep_len:
    efficiency_gain = (ppo_final_ep_len - ddqn_final_ep_len) / ppo_final_ep_len * 100
    print(f"  → DDQN is {efficiency_gain:.1f}% more efficient (shorter episodes)")

print("\n[Training Stability - Reward Variance]")
ppo_reward_std = ppo_data['reward'].std()
ddqn_reward_std = ddqn_training['avg_reward_100'].std()
print(f"  PPO  - Std Dev: {ppo_reward_std:.2f}")
print(f"  DDQN - Std Dev: {ddqn_reward_std:.2f}")
if ddqn_reward_std < ppo_reward_std:
    stability_gain = (ppo_reward_std - ddqn_reward_std) / ppo_reward_std * 100
    print(f"  → DDQN is {stability_gain:.1f}% more stable (lower variance)")
else:
    stability_loss = (ddqn_reward_std - ppo_reward_std) / ppo_reward_std * 100
    print(f"  → PPO is {stability_loss:.1f}% more stable (lower variance)")

print("\n[Sample Efficiency - Episodes vs Timesteps]")
ppo_total_episodes = len(ppo_data)
ddqn_total_episodes = int(ddqn_training['episode'].max() + 1)
ppo_steps_per_episode = ppo_data['step'].max() / ppo_total_episodes
ddqn_steps_per_episode = ddqn_training['step'].max() / ddqn_total_episodes

print(f"  PPO  - Total Episodes: {ppo_total_episodes} | Avg Steps/Episode: {ppo_steps_per_episode:.1f}")
print(f"  DDQN - Total Episodes: {ddqn_total_episodes} | Avg Steps/Episode: {ddqn_steps_per_episode:.1f}")
if ddqn_total_episodes > ppo_total_episodes:
    episode_ratio = ddqn_total_episodes / ppo_total_episodes
    print(f"  → DDQN collected {episode_ratio:.1f}× more episodes (better exploration)")

print("\n[Learning Progress - Reward Improvement]")
# Calculate reward improvement from first 10% to last 10% of training
ppo_early = ppo_data.head(max(1, len(ppo_data) // 10))
ppo_late = ppo_data.tail(max(1, len(ppo_data) // 10))
ddqn_early = ddqn_training.head(max(1, len(ddqn_training) // 10))
ddqn_late = ddqn_training.tail(max(1, len(ddqn_training) // 10))

ppo_improvement = ppo_late['reward'].mean() - ppo_early['reward'].mean()
ddqn_improvement = ddqn_late['avg_reward_100'].mean() - ddqn_early['avg_reward_100'].mean()

print(f"  PPO  - Early Avg: {ppo_early['reward'].mean():.2f} | Late Avg: {ppo_late['reward'].mean():.2f} | Improvement: {ppo_improvement:+.2f}")
print(f"  DDQN - Early Avg: {ddqn_early['avg_reward_100'].mean():.2f} | Late Avg: {ddqn_late['avg_reward_100'].mean():.2f} | Improvement: {ddqn_improvement:+.2f}")

print("\n[Success Rate - DDQN Only]")
print(f"  DDQN - Mean Success Rate: {ddqn_training['success_rate'].mean():.2%}")
print(f"  DDQN - Final Success Rate: {ddqn_training['success_rate'].iloc[-1]:.2%}")

print("\n[Exploration Metrics - DDQN Only]")
ddqn_mean_epsilon = ddqn_training['epsilon'].mean()
ddqn_final_epsilon = ddqn_training['epsilon'].iloc[-1]
ddqn_mean_loss = ddqn_training['loss'].mean()
ddqn_final_loss = ddqn_training['loss'].iloc[-1]
ddqn_mean_q = ddqn_training['q_value'].mean()
ddqn_final_q = ddqn_training['q_value'].iloc[-1]

print(f"  Epsilon - Mean: {ddqn_mean_epsilon:.4f} | Final: {ddqn_final_epsilon:.4f}")
print(f"  Loss    - Mean: {ddqn_mean_loss:.4f} | Final: {ddqn_final_loss:.4f}")
print(f"  Q-Value - Mean: {ddqn_mean_q:.2f} | Final: {ddqn_final_q:.2f}")

# Create comparison plots
print("\n[3] Creating Comparison Plots...")

# ============================================================================
# PLOT 1: Reward Comparison
# ============================================================================
print("  [3.1] Creating Reward Comparison Plot...")
fig1, ax1 = plt.subplots(1, 1, figsize=(16, 10))

# Plot DDQN with solid line
sns.lineplot(x='step_k', y='avg_reward_100', data=ddqn_training, label='DDQN', ax=ax1, linewidth=3, color='#A23B72')
# Plot PPO with dashed line
ax1.plot(ppo_data['step_k'], ppo_data['reward'], label='PPO', linewidth=3, color='#2E86AB', linestyle='--')

ax1.set_xlabel(r'Timesteps ($\times 10^3$)', fontsize=26, fontweight='bold')
ax1.set_ylabel('Reward', fontsize=26, fontweight='bold')
ax1.set_title('Reward Comparison', fontsize=28, fontweight='bold', pad=20)
ax1.legend(fontsize=24, loc='lower right', frameon=True, shadow=True)
ax1.grid(True, alpha=0.5, linestyle='-', linewidth=2.0)  # Increased alpha to 0.5 for better visibility
ax1.set_xticks([0, 5, 10, 15, 20, 25, 30])
ax1.set_xlim([0, 30])
ax1.set_xticklabels(['', '5', '10', '15', '20', '25', '30'])  # Hide '0' label
y_max = max(ppo_data['reward'].max(), ddqn_training['avg_reward_100'].max())
ax1.set_ylim([0, y_max * 1.05])
ax1.tick_params(labelsize=22)

plt.tight_layout()
output_path_1 = r'cobotproject\scripts\Reinforcement Learning\ppo_vs_ddqn_rewards.png'
plt.savefig(output_path_1, dpi=300, bbox_inches='tight')
print(f"  Reward comparison saved to: {output_path_1}")
plt.close()

# ============================================================================
# PLOT 2: Convergence Comparison
# ============================================================================
print("  [3.2] Creating Convergence Comparison Plot...")
fig2, ax2 = plt.subplots(1, 1, figsize=(16, 10))

# Define convergence thresholds - every 20 from 0 to 200
thresholds = list(range(0, 201, 20))  # [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
ppo_convergence_steps = []
ddqn_convergence_steps = []

# Get PPO's maximum reward
ppo_max_reward = ppo_data['reward'].max()  # Should be 196.06
print(f"  PPO Maximum Reward: {ppo_max_reward:.2f}")

for threshold in thresholds:
    # PPO convergence - check all thresholds, append None if not reached
    ppo_conv = ppo_data[ppo_data['reward'] > threshold]
    if len(ppo_conv) > 0:
        ppo_convergence_steps.append(ppo_conv.iloc[0]['step'] / 1000)  # Convert to thousands
    else:
        ppo_convergence_steps.append(None)  # PPO didn't reach this threshold

    # DDQN convergence (after episode 100)
    ddqn_conv = ddqn_after_100[ddqn_after_100['avg_reward_100'] > threshold]
    if len(ddqn_conv) > 0:
        ddqn_convergence_steps.append(ddqn_conv.iloc[0]['step'] / 1000)  # Convert to thousands
    else:
        ddqn_convergence_steps.append(None)  # DDQN didn't reach this threshold

# Plot convergence curves
# PPO: x marker = 10 (doubled), DDQN: o marker = 8 (original size)
# Plot both algorithms for all thresholds (None values will create gaps)
ax2.plot(thresholds, ppo_convergence_steps, marker='x', markersize=10, linewidth=3,
         color='#2E86AB', label='PPO', linestyle='--', markeredgewidth=2)
ax2.plot(thresholds, ddqn_convergence_steps, marker='o', markersize=8, linewidth=3,
         color='#A23B72', label='DDQN')

ax2.set_xlabel('Reward Threshold', fontsize=26, fontweight='bold')
ax2.set_ylabel(r'Timesteps ($\times 10^3$)', fontsize=26, fontweight='bold')
ax2.set_title('Convergence Speed', fontsize=28, fontweight='bold', pad=20)
ax2.legend(fontsize=24, loc='upper right', frameon=True, shadow=True)
ax2.grid(True, alpha=0.5, linestyle='-', linewidth=2.0)  # Increased alpha to 0.5 for better visibility
ax2.tick_params(labelsize=22)
# Set y-axis ticks to 5, 10, 15, 20, 25, 30
ax2.set_yticks([5, 10, 15, 20, 25, 30])
ax2.set_ylim([0, 32])
# Set x-axis to show every 20 points
ax2.set_xticks(thresholds)
ax2.set_xlim([0, 210])

plt.tight_layout()
output_path_2 = r'cobotproject\scripts\Reinforcement Learning\ppo_vs_ddqn_convergence.png'
plt.savefig(output_path_2, dpi=300, bbox_inches='tight')
print(f"  Convergence comparison saved to: {output_path_2}")
plt.close()

# ============================================================================
# PLOT 3: Training Stability (Box Plot Comparison)
# ============================================================================
print("  [3.3] Creating Training Stability Plot...")
fig3, ax3 = plt.subplots(1, 1, figsize=(16, 10))

# Calculate rolling standard deviation for CSV export
ddqn_training['rolling_std'] = ddqn_training['avg_reward_100'].rolling(window=100, min_periods=1).std()
ppo_data['rolling_std'] = ppo_data['reward'].rolling(window=5, min_periods=1).std()

# Prepare data for box plot - divide training into 6 phases
num_phases = 6
ddqn_phase_size = len(ddqn_training) // num_phases
ppo_phase_size = len(ppo_data) // num_phases

box_data = []
box_labels = []
box_positions = []
box_colors = []

for i in range(num_phases):
    # DDQN phase
    ddqn_phase = ddqn_training.iloc[i*ddqn_phase_size:(i+1)*ddqn_phase_size]['avg_reward_100'].values
    box_data.append(ddqn_phase)
    box_labels.append(f'DDQN')
    box_positions.append(i * 3 + 0.5)
    box_colors.append('#A23B72')

    # PPO phase
    if i < len(ppo_data) // ppo_phase_size:
        ppo_phase = ppo_data.iloc[i*ppo_phase_size:(i+1)*ppo_phase_size]['reward'].values
        box_data.append(ppo_phase)
        box_labels.append(f'PPO')
        box_positions.append(i * 3 + 1.5)
        box_colors.append('#2E86AB')

# Filter out low-value outliers (below 150) from box_data
filtered_box_data = []
for data in box_data:
    # Keep only values >= 150 to avoid showing near-zero outliers
    filtered_data = data[data >= 150]
    filtered_box_data.append(filtered_data)

# Create box plot with outliers shown (white circles with thin grey border)
bp = ax3.boxplot(filtered_box_data, positions=box_positions, widths=0.6, patch_artist=True,
                 showfliers=True, medianprops=dict(color='black', linewidth=2),
                 flierprops=dict(marker='o', markerfacecolor='white', markersize=6,
                                linestyle='none', markeredgecolor='grey', markeredgewidth=0.8))

# Color the boxes
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Set labels - use discrete timesteps instead of ranges
phase_centers = [i * 3 + 1 for i in range(num_phases)]
phase_labels = [f'{int((i+1)*30/num_phases)}' for i in range(num_phases)]  # Show end of each phase: 5, 10, 15, 20, 25, 30
ax3.set_xticks(phase_centers)
ax3.set_xticklabels(phase_labels)

ax3.set_xlabel(r'Timesteps ($\times 10^3$)', fontsize=26, fontweight='bold')
ax3.set_ylabel('Reward Deviation', fontsize=26, fontweight='bold')
ax3.set_title('Training Stability', fontsize=28, fontweight='bold', pad=20)

# Set y-axis limits to original range (170 to 210)
ax3.set_ylim([170, 210])

# Create custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#A23B72', alpha=0.7, label='DDQN'),
                   Patch(facecolor='#2E86AB', alpha=0.7, label='PPO')]
ax3.legend(handles=legend_elements, fontsize=24, loc='lower right', frameon=True, shadow=True)

ax3.grid(True, alpha=0.5, linestyle='-', linewidth=2.0, axis='y')  # Increased alpha to 0.5 for better visibility
ax3.tick_params(labelsize=22)

plt.tight_layout()
output_path_3 = r'cobotproject\scripts\Reinforcement Learning\ppo_vs_ddqn_stability.png'
plt.savefig(output_path_3, dpi=300, bbox_inches='tight')
print(f"  Training stability saved to: {output_path_3}")
plt.close()

print("\n  All plots created successfully!")

# Save summary to CSV
print("\n[4] Saving Summary...")
summary_data = {
    'Metric': [
        'Total Steps Analyzed',
        'Total Episodes',
        'Avg Steps per Episode',
        'Final Avg Reward',
        'Max Reward',
        'Mean Reward',
        'Std Reward (Stability)',
        'Reward Improvement (Early→Late)',
        'Steps to Converge (>180)',
        'Steps to Converge (>150)',
        'Final Episode Length',
        'Mean Episode Length',
        'Final Explained Variance',
        'Final Success Rate',
        'Final Epsilon',
        'Final Loss',
        'Final Q-Value'
    ],
    'PPO': [
        f"{ppo_data['step'].max():,}",
        f"{ppo_total_episodes}",
        f"{ppo_steps_per_episode:.1f}",
        f"{ppo_data['reward'].iloc[-1]:.2f}",
        f"{ppo_data['reward'].max():.2f}",
        f"{ppo_data['reward'].mean():.2f}",
        f"{ppo_reward_std:.2f}",
        f"{ppo_improvement:+.2f}",
        f"{ppo_converge_high.iloc[0]['step']:,.0f}" if len(ppo_converge_high) > 0 else 'Not Reached',
        f"{ppo_converge_mid.iloc[0]['step']:,.0f}" if len(ppo_converge_mid) > 0 else 'Not Reached',
        f"{ppo_final_ep_len:.2f}",
        f"{ppo_mean_ep_len:.2f}",
        f"{ppo_variance_data['explained_variance'].iloc[-1]:.4f}",
        'N/A',
        'N/A',
        'N/A',
        'N/A'
    ],
    'DDQN': [
        f"{ddqn_training['step'].max():,}",
        f"{ddqn_total_episodes}",
        f"{ddqn_steps_per_episode:.1f}",
        f"{ddqn_training['avg_reward_100'].iloc[-1]:.2f}",
        f"{ddqn_training['avg_reward_100'].max():.2f}",
        f"{ddqn_training['avg_reward_100'].mean():.2f}",
        f"{ddqn_reward_std:.2f}",
        f"{ddqn_improvement:+.2f}",
        f"{ddqn_converge_high.iloc[0]['step']:,.0f}" if len(ddqn_converge_high) > 0 else 'Not Reached',
        f"{ddqn_converge_mid.iloc[0]['step']:,.0f}" if len(ddqn_converge_mid) > 0 else 'Not Reached',
        f"{ddqn_final_ep_len:.2f}",
        f"{ddqn_mean_ep_len:.2f}",
        'N/A',
        f"{ddqn_training['success_rate'].iloc[-1]:.2%}",
        f"{ddqn_final_epsilon:.4f}",
        f"{ddqn_final_loss:.4f}",
        f"{ddqn_final_q:.2f}"
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_path = r'cobotproject\scripts\Reinforcement Learning\ppo_vs_ddqn_summary.csv'
summary_df.to_csv(summary_path, index=False)
print(f"Summary saved to: {summary_path}")

# Save convergence data to separate CSV
print("\n[5] Saving Convergence Data...")
convergence_data = {
    'Reward_Threshold': thresholds,
    'PPO_Steps': ppo_convergence_steps,
    'DDQN_Steps': ddqn_convergence_steps
}
convergence_df = pd.DataFrame(convergence_data)
convergence_path = r'cobotproject\scripts\Reinforcement Learning\ppo_vs_ddqn_convergence_data.csv'
convergence_df.to_csv(convergence_path, index=False)
print(f"Convergence data saved to: {convergence_path}")

# Save stability data (rolling std) to separate CSV
print("\n[6] Saving Stability Data...")
# Sample every 100 steps for DDQN to reduce file size
ddqn_stability_sample = ddqn_training[::100][['step', 'rolling_std']].copy()
ddqn_stability_sample.columns = ['Timestep', 'DDQN_Rolling_Std']

# Create PPO stability data
ppo_stability = ppo_data[['step', 'rolling_std']].copy()
ppo_stability.columns = ['Timestep', 'PPO_Rolling_Std']

# Merge on nearest timestep
stability_df = pd.merge_asof(
    ddqn_stability_sample.sort_values('Timestep'),
    ppo_stability.sort_values('Timestep'),
    on='Timestep',
    direction='nearest'
)

stability_path = r'cobotproject\scripts\Reinforcement Learning\ppo_vs_ddqn_stability_data.csv'
stability_df.to_csv(stability_path, index=False)
print(f"Stability data saved to: {stability_path}")

# Print summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(summary_df.to_string(index=False))

# Determine winners
print("\n" + "="*80)
print("WINNER ANALYSIS")
print("="*80)
winner_final_reward = "PPO" if ppo_data['reward'].iloc[-1] > ddqn_training['avg_reward_100'].iloc[-1] else "DDQN"
winner_max_reward = "PPO" if ppo_data['reward'].max() > ddqn_training['avg_reward_100'].max() else "DDQN"
winner_mean_reward = "PPO" if ppo_data['reward'].mean() > ddqn_training['avg_reward_100'].mean() else "DDQN"

print(f"Winner (Final Reward):    {winner_final_reward}")
print(f"Winner (Max Reward):      {winner_max_reward}")
print(f"Winner (Mean Reward):     {winner_mean_reward}")

if len(ppo_converge_mid) > 0 and len(ddqn_converge_mid) > 0:
    winner_learning = "PPO" if ppo_converge_mid.iloc[0]['step'] < ddqn_converge_mid.iloc[0]['step'] else "DDQN"
    print(f"Winner (Faster Learning): {winner_learning}")
elif len(ppo_converge_mid) > 0:
    print(f"Winner (Faster Learning): PPO (DDQN did not converge)")
elif len(ddqn_converge_mid) > 0:
    print(f"Winner (Faster Learning): DDQN (PPO did not converge)")
else:
    print(f"Winner (Faster Learning): Neither converged to reward > {reward_threshold_mid}")

print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)

