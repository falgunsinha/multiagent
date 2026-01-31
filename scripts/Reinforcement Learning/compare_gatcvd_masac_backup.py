"""
GAT-CVD vs MASAC Training Comparison Script
Compares training metrics at 12,933 timesteps for thesis Results section

Author: Master's Thesis Analysis
Date: 2026-01-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# COLOR SCHEME - Using seaborn default qualitative palette
# ============================================================================
sns.set_theme(style="ticks")
default_palette = sns.color_palette()  # Default qualitative palette with 10 distinct hues

COLORS = {
    'gatcvd_main': default_palette[4],    # Purple for mGAT (5th color)
    'masac_main': default_palette[2],     # Green for mSAC (3rd color)
    'qvalue': default_palette[1],         # Orange for Q-value (2nd color)
    'qover': default_palette[5],          # Brown for Q-overestimation (6th color)
}

# ============================================================================
# CONFIGURATION
# ============================================================================
MAX_TIMESTEPS = 12933  # MASAC Seed 42 total timesteps
ROLLING_WINDOW = 100   # Rolling average window
SMOOTHING_SIGMA = 2    # Gaussian smoothing for wave curves
DPI = 300              # High resolution for thesis
GRID_LINEWIDTH = 0.8   # Grid line width (decreased from 1.5)
GRID_ALPHA = 0.7       # Grid visibility (increased for better visibility)

# File paths
GAT_CVD_TRAINING = Path("cobotproject/scripts/Reinforcement Learning/MARL/src/gat_cvd/logs/gat_cvd_isaacsim_grid4_cubes9_20260123_132522_training.csv")
GAT_CVD_EPISODES = Path("cobotproject/scripts/Reinforcement Learning/MARL/src/gat_cvd/logs/gat_cvd_isaacsim_grid4_cubes9_20260123_132522_episodes.csv")
MASAC_TIMESTEP = Path("cobotproject/scripts/Reinforcement Learning/MASAC/logs/multi_seed_20260120_033815/masac_rrt_isaacsim_grid4_cubes9_20260120_092042_timestep_log.csv")
MASAC_EPISODE = Path("cobotproject/scripts/Reinforcement Learning/MASAC/logs/multi_seed_20260120_033815/masac_rrt_isaacsim_grid4_cubes9_20260120_092042_episode_log.csv")
MASAC_SUMMARY = Path("cobotproject/scripts/Reinforcement Learning/MASAC/logs/multi_seed_20260120_033815/masac_rrt_isaacsim_grid4_cubes9_20260120_092042_summary.json")

OUTPUT_DIR = Path("cobotproject/scripts/Reinforcement Learning/comparison_results")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_plot_style():
    """Setup consistent plot style for all graphs"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 14,           # Increased from 11
        'axes.labelsize': 16,      # Increased from 12
        'axes.titlesize': 18,      # Increased from 13
        'xtick.labelsize': 14,     # Increased from 10
        'ytick.labelsize': 14,     # Increased from 10
        'legend.fontsize': 14,     # Increased from 10
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'grid.linewidth': GRID_LINEWIDTH,
        'grid.alpha': GRID_ALPHA,
        'grid.linestyle': '-',
    })


def format_x_axis(ax, max_val):
    """Format x-axis as 1, 2, 3, ..., 13 (×10³)"""
    # Set x-axis limits starting from 0
    ax.set_xlim(0, max_val)

    # Create ticks at intervals of 1000
    ticks = np.arange(0, max_val + 1000, 1000)
    labels = ['' if x == 0 else str(int(x/1000)) for x in ticks]

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Timesteps (×10³)', fontsize=16, fontweight='bold')  # Increased from 12


def smooth_curve(data, sigma=SMOOTHING_SIGMA):
    """Apply Gaussian smoothing for wave-like curves"""
    return gaussian_filter1d(data, sigma=sigma)


def calculate_rolling_mean(series, window=ROLLING_WINDOW):
    """Calculate rolling mean"""
    return series.rolling(window=window, min_periods=1).mean()


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_gatcvd_data():
    """Load and process GAT-CVD data up to MAX_TIMESTEPS"""
    print("Loading GAT-CVD data...")
    
    # Load training data
    training_df = pd.read_csv(GAT_CVD_TRAINING)
    episodes_df = pd.read_csv(GAT_CVD_EPISODES)
    
    # Filter up to MAX_TIMESTEPS
    training_df = training_df[training_df['step'] <= MAX_TIMESTEPS].copy()
    
    # Get corresponding episode number
    max_episode = training_df['episode'].max()
    episodes_df = episodes_df[episodes_df['episode'] <= max_episode].copy()
    
    print(f"  GAT-CVD: {len(episodes_df)} episodes, {len(training_df)} timesteps")
    
    return training_df, episodes_df


def load_masac_data():
    """Load and process MASAC data (Seed 42)"""
    print("Loading MASAC data...")
    
    # Load data
    timestep_df = pd.read_csv(MASAC_TIMESTEP)
    episode_df = pd.read_csv(MASAC_EPISODE)
    
    import json
    with open(MASAC_SUMMARY, 'r') as f:
        summary = json.load(f)
    
    print(f"  MASAC: {len(episode_df)} episodes, {len(timestep_df)} timesteps")
    
    return timestep_df, episode_df, summary


# ============================================================================
# MAIN FUNCTION
# ============================================================================

# ============================================================================
# METRIC CALCULATION FUNCTIONS
# ============================================================================

def prepare_comparison_data(gat_training, gat_episodes, masac_timestep, masac_episode):
    """Prepare aligned comparison data"""
    print("\nPreparing comparison data...")

    # GAT-CVD: Use training CSV directly (has all needed columns)
    # Use episode_reward column as requested
    gat_df = gat_training[['step', 'episode', 'episode_reward', 'avg_reward_100', 'success_rate',
                            'episode_length', 'loss_ddqn', 'loss_masac', 'loss_cvd',
                            'q_value_ddqn', 'q_overestimation_ddqn']].copy()
    gat_df.rename(columns={'step': 'timestep', 'episode_reward': 'reward'}, inplace=True)

    # MASAC: Create timestep-aligned data with rolling metrics
    masac_episode['success_rate_100'] = masac_episode['success'].rolling(window=100, min_periods=1).mean()
    masac_episode['avg_reward_100'] = masac_episode['total_reward'].rolling(window=100, min_periods=1).mean()

    masac_data = []
    for _, row in masac_timestep.iterrows():
        timestep = int(row['global_timestep'])
        episode = int(row['episode'])

        if episode > 0 and episode <= len(masac_episode):
            ep_data = masac_episode.iloc[episode - 1]
            masac_data.append({
                'timestep': timestep,
                'episode': episode,
                'reward': row['cumulative_reward'],  # Use cumulative_reward from timestep log directly
                'success_rate': ep_data['success_rate_100'],
                'avg_reward_100': ep_data['avg_reward_100'],
                'episode_length': ep_data['episode_length'],
            })

    masac_df = pd.DataFrame(masac_data)

    print(f"  GAT-CVD: {len(gat_df)} timesteps")
    print(f"  MASAC: {len(masac_df)} timesteps")

    return gat_df, masac_df


def calculate_convergence_metrics(gat_df, masac_df):
    """Calculate convergence metrics (episodes to reach thresholds)"""
    print("\nCalculating convergence metrics...")

    metrics = {}

    # GAT-CVD convergence
    gat_80 = gat_df[gat_df['success_rate'] >= 0.80]
    gat_90 = gat_df[gat_df['success_rate'] >= 0.90]

    metrics['gat_episodes_to_80'] = gat_80.iloc[0]['episode'] if len(gat_80) > 0 else None
    metrics['gat_episodes_to_90'] = gat_90.iloc[0]['episode'] if len(gat_90) > 0 else None
    metrics['gat_timesteps_to_80'] = gat_80.iloc[0]['timestep'] if len(gat_80) > 0 else None
    metrics['gat_timesteps_to_90'] = gat_90.iloc[0]['timestep'] if len(gat_90) > 0 else None

    # MASAC convergence
    masac_80 = masac_df[masac_df['success_rate'] >= 0.80]
    masac_90 = masac_df[masac_df['success_rate'] >= 0.90]

    metrics['masac_episodes_to_80'] = masac_80.iloc[0]['episode'] if len(masac_80) > 0 else None
    metrics['masac_episodes_to_90'] = masac_90.iloc[0]['episode'] if len(masac_90) > 0 else None
    metrics['masac_timesteps_to_80'] = masac_80.iloc[0]['timestep'] if len(masac_80) > 0 else None
    metrics['masac_timesteps_to_90'] = masac_90.iloc[0]['timestep'] if len(masac_90) > 0 else None

    print(f"  GAT-CVD: 80% at episode {metrics['gat_episodes_to_80']}, 90% at episode {metrics['gat_episodes_to_90']}")
    print(f"  MASAC: 80% at episode {metrics['masac_episodes_to_80']}, 90% at episode {metrics['masac_episodes_to_90']}")

    return metrics


# ============================================================================
# GRAPH GENERATION FUNCTIONS
# ============================================================================

def plot_graph1_learning_curves(gat_df, masac_df, output_dir):
    """Graph 1: Reward Comparison - Plot episode_reward and cumulative_reward with running average smoothing"""
    print("\nGenerating Graph 1: Learning Curves...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter out negative rewards for both algorithms
    gat_filtered = gat_df[gat_df['reward'] >= 0].copy()
    masac_filtered = masac_df[masac_df['reward'] >= 0].copy()

    print(f"  GAT-CVD after filtering: {len(gat_filtered)} timesteps")
    print(f"  MASAC after filtering: {len(masac_filtered)} timesteps")

    # Apply running average (rolling mean) smoothing with increased window size
    # Increase window size for smoother curves (from 100 to 300)
    window_size = 300  # Increased for more smoothing

    gat_smooth = gat_filtered['reward'].rolling(window=window_size, min_periods=1).mean().values
    masac_smooth = masac_filtered['reward'].rolling(window=window_size, min_periods=1).mean().values

    # Interpolate mSAC curve to match mGAT timesteps
    from scipy.interpolate import interp1d

    # Create interpolation function for MASAC
    masac_interp_func = interp1d(masac_filtered['timestep'].values, masac_smooth,
                                  kind='linear', fill_value='extrapolate')

    # Interpolate MASAC to GAT timesteps
    masac_smooth_interp = masac_interp_func(gat_filtered['timestep'].values)

    print(f"  Interpolated mSAC to match mGAT timesteps: {len(gat_filtered)} timesteps")

    # Plot with reduced line thickness (2.5 -> 2.0)
    ax.plot(gat_filtered['timestep'].values, gat_smooth,
            color=COLORS['gatcvd_main'], linewidth=2.0, label='mGAT', alpha=0.9)
    ax.plot(gat_filtered['timestep'].values, masac_smooth_interp,
            color=COLORS['masac_main'], linewidth=2.0, label='mSAC', alpha=0.9)

    # Formatting
    ax.set_ylabel('Reward', fontsize=16, fontweight='bold')  # Increased from 12
    ax.set_title('Reward Comparison', fontsize=18, fontweight='bold')  # Increased from 13
    ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=14)  # Added fontsize=14
    ax.grid(True, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA, linestyle='-')

    # Set proper y-axis limits to avoid compression at top
    # Add some padding (5%) to the y-axis range
    y_min = min(gat_smooth.min(), masac_smooth_interp.min())
    y_max = max(gat_smooth.max(), masac_smooth_interp.max())
    y_padding = (y_max - y_min) * 0.05
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Update x-axis to reflect actual max timestep
    max_timestep = gat_filtered['timestep'].max()
    format_x_axis(ax, max_timestep)

    # Save (PNG only)
    plt.savefig(output_dir / 'graph1_learning_curves.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: graph1_learning_curves.png")


def plot_graph2_success_rate(gat_df, masac_df, output_dir):
    """Graph 2: Success Rate Over Time - Wave format"""
    print("\nGenerating Graph 2: Success Rate...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Smooth the curves
    gat_smooth = smooth_curve(gat_df['success_rate'].values * 100)
    masac_smooth = smooth_curve(masac_df['success_rate'].values * 100)

    # Plot with reduced line thickness (2.5 -> 2.0)
    ax.plot(gat_df['timestep'], gat_smooth,
            color=COLORS['gatcvd_main'], linewidth=2.0, label='mGAT')
    ax.plot(masac_df['timestep'], masac_smooth,
            color=COLORS['masac_main'], linewidth=2.0, label='mSAC')

    # Reference lines (without labels in legend)
    ax.axhline(y=80, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(y=90, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

    # Formatting
    ax.set_ylabel('Success Rate (%)', fontsize=16, fontweight='bold')  # Increased from 12
    ax.set_title('Success Rate Over Time', fontsize=18, fontweight='bold')  # Increased from 13
    ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=14)  # Added fontsize=14
    ax.grid(True, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA, linestyle='-')

    # Set y-axis to start at 0 and align with x-axis intersection
    ax.set_ylim(bottom=0)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))

    format_x_axis(ax, MAX_TIMESTEPS)

    # Save (PNG only)
    plt.savefig(output_dir / 'graph2_success_rate.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: graph2_success_rate.png")


def plot_graph3_convergence(gat_df, masac_df, output_dir):
    """Graph 3: Convergence Speed - Reward threshold based (like PPO vs DDQN)"""
    print("\nGenerating Graph 3: Convergence Analysis...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define reward thresholds (similar to PPO vs DDQN)
    # Adjust range based on actual reward values
    thresholds = list(range(-200, 1, 20))  # [-200, -180, -160, ..., -20, 0]
    gat_convergence_steps = []
    masac_convergence_steps = []

    # Calculate convergence for each threshold
    for threshold in thresholds:
        # GAT-CVD convergence
        gat_conv = gat_df[gat_df['avg_reward_100'] > threshold]
        if len(gat_conv) > 0:
            gat_convergence_steps.append(gat_conv.iloc[0]['timestep'] / 1000)  # Convert to thousands
        else:
            gat_convergence_steps.append(None)  # Not reached

        # MASAC convergence
        masac_conv = masac_df[masac_df['avg_reward_100'] > threshold]
        if len(masac_conv) > 0:
            masac_convergence_steps.append(masac_conv.iloc[0]['timestep'] / 1000)  # Convert to thousands
        else:
            masac_convergence_steps.append(None)  # Not reached

    # Plot convergence curves with circle (o) and cross (x) markers
    # Increased cross marker size from 4 to 4.5
    ax.plot(thresholds, gat_convergence_steps, marker='o', markersize=4, linewidth=2.5,
            color=COLORS['gatcvd_main'], label='mGAT')
    ax.plot(thresholds, masac_convergence_steps, marker='x', markersize=4.5, linewidth=2.5,
            color=COLORS['masac_main'], label='mSAC', markeredgewidth=2)

    # Formatting
    ax.set_xlabel('Reward Threshold', fontsize=16, fontweight='bold')  # Increased from 12
    ax.set_ylabel(r'Timesteps ($\times 10^3$)', fontsize=16, fontweight='bold')  # Increased from 12
    ax.set_title('Convergence Speed', fontsize=18, fontweight='bold')  # Increased from 13
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=14)  # Added fontsize=14
    ax.grid(True, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA, linestyle='-')

    # Save (PNG only)
    plt.savefig(output_dir / 'graph3_convergence.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: graph3_convergence.png")


def plot_graph4_qvalue_curves(gat_training, output_dir):
    """Graph 4: Q-Value and Q-Overestimation (GAT-CVD only) - Line curves vs timesteps"""
    print("\nGenerating Graph 4: Q-Value Curves (GAT-CVD only)...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Remove NaN values for Q-values
    gat_qvalue = gat_training.dropna(subset=['q_value_ddqn'])
    gat_qover = gat_training.dropna(subset=['q_overestimation_ddqn'])

    # Plot Q-value and Q-overestimation curves (NO SMOOTHING to show actual values)
    if len(gat_qvalue) > 0:
        ax.plot(gat_qvalue['step'], gat_qvalue['q_value_ddqn'],
                color=COLORS['qvalue'], linewidth=2.0, label='Q-Value (DDQN)', alpha=0.8)

    if len(gat_qover) > 0:
        ax.plot(gat_qover['step'], gat_qover['q_overestimation_ddqn'],
                color=COLORS['qover'], linewidth=2.0, label='Q-Overestimation (DDQN)', alpha=0.8)

    # Formatting
    ax.set_ylabel('Q-Value', fontsize=16, fontweight='bold')  # Increased from 12
    ax.set_title('GAT-CVD Q-Value Analysis', fontsize=18, fontweight='bold')  # Increased from 13
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=14)  # Added fontsize=14
    ax.grid(True, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA, linestyle='-')
    format_x_axis(ax, MAX_TIMESTEPS)

    # Save (PNG only)
    plt.savefig(output_dir / 'graph4_qvalue_curves.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: graph4_qvalue_curves.png")


def plot_graph5_epsilon_decay(gat_training, output_dir):
    """Graph 5: Epsilon Decay Curves - Exploration rate during training (by timesteps)"""
    print("\nGenerating Graph 5: Epsilon Decay Curves (Timesteps)...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter to 1000 timesteps only
    max_timesteps_epsilon = 1000
    gat_filtered = gat_training[gat_training['step'] <= max_timesteps_epsilon].copy()

    # Use seaborn color palette - gray and light green
    import seaborn as sns
    colors_epsilon = ['#808080', '#B8CD50']  # Gray and light green

    # Plot epsilon values (exact values from CSV - NO SMOOTHING)
    ax.plot(gat_filtered['step'], gat_filtered['epsilon_ddqn'],
            color=colors_epsilon[0], linewidth=2.0, label='ε-DDQN', alpha=0.9)
    ax.plot(gat_filtered['step'], gat_filtered['epsilon_masac'],
            color=colors_epsilon[1], linewidth=2.0, label='ε-mSAC', alpha=0.9)

    # Find where epsilon reaches 0.01 for annotation
    epsilon_ddqn_min = gat_filtered[gat_filtered['epsilon_ddqn'] <= 0.01]
    epsilon_masac_min = gat_filtered[gat_filtered['epsilon_masac'] <= 0.01]

    if not epsilon_ddqn_min.empty:
        first_ddqn = epsilon_ddqn_min.iloc[0]
        timestep_ddqn = int(first_ddqn['step'])
        epsilon_val_ddqn = first_ddqn['epsilon_ddqn']
        print(f"  ε-DDQN reaches 0.01 at timestep {timestep_ddqn}, episode {first_ddqn['episode']}")

        # Draw point on the curve where ε-DDQN reaches 0.01
        ax.scatter([timestep_ddqn], [epsilon_val_ddqn], color=colors_epsilon[0], s=100, zorder=5, edgecolors='black', linewidths=1.5)
        # Add text annotation
        ax.annotate(f'ε = ({epsilon_val_ddqn:.2f}, {timestep_ddqn})',
                    xy=(timestep_ddqn, epsilon_val_ddqn),
                    xytext=(timestep_ddqn - 150, epsilon_val_ddqn + 0.08),
                    fontsize=9, color=colors_epsilon[0], fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=colors_epsilon[0], alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color=colors_epsilon[0], lw=1.5))

    if not epsilon_masac_min.empty:
        first_masac = epsilon_masac_min.iloc[0]
        timestep_masac = int(first_masac['step'])
        epsilon_val_masac = first_masac['epsilon_masac']
        print(f"  ε-MASAC reaches 0.01 at timestep {timestep_masac}, episode {first_masac['episode']}")

        # Draw point on the curve where ε-MASAC reaches 0.01
        ax.scatter([timestep_masac], [epsilon_val_masac], color=colors_epsilon[1], s=100, zorder=5, edgecolors='black', linewidths=1.5)
        # Add text annotation
        ax.annotate(f'ε = ({epsilon_val_masac:.2f}, {timestep_masac})',
                    xy=(timestep_masac, epsilon_val_masac),
                    xytext=(timestep_masac - 150, epsilon_val_masac + 0.05),
                    fontsize=9, color=colors_epsilon[1], fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=colors_epsilon[1], alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color=colors_epsilon[1], lw=1.5))

    # Formatting
    ax.set_xlabel('Timesteps', fontsize=16, fontweight='bold')  # Increased from 12
    ax.set_ylabel('Epsilon (ε)', fontsize=16, fontweight='bold')  # Increased from 12
    ax.set_title('Exploration: Epsilon Decay Over Time', fontsize=18, fontweight='bold')  # Increased from 13
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=14)  # Increased from 9
    ax.grid(True, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA, linestyle='-')

    # Set specific axis ranges
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 0.48)

    # Set specific x-axis ticks: 100, 200, ..., 1000 (hide 0)
    ax.set_xticks(np.arange(100, 1001, 100))

    # Set specific y-axis ticks: 0, 0.04, 0.08, ..., 0.48
    ax.set_yticks(np.arange(0, 0.49, 0.04))

    # Save (PNG only)
    plt.savefig(output_dir / 'graph5_epsilon_decay_timesteps.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: graph5_epsilon_decay_timesteps.png")


def plot_graph5b_epsilon_decay_episodes(gat_training, output_dir):
    """Graph 5b: Epsilon Decay Curves - Exploration rate during training (by episodes)"""
    print("\nGenerating Graph 5b: Epsilon Decay Curves (Episodes)...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter to episodes where timestep <= 1000
    max_timesteps_epsilon = 1000
    gat_filtered = gat_training[gat_training['step'] <= max_timesteps_epsilon].copy()

    # Get the max episode number in this range
    max_episode = int(gat_filtered['episode'].max())

    # Use seaborn color palette - gray and light green
    import seaborn as sns
    colors_epsilon = ['#808080', '#B8CD50']  # Gray and light green

    # Plot epsilon values vs episodes (exact values from CSV - NO SMOOTHING)
    ax.plot(gat_filtered['episode'], gat_filtered['epsilon_ddqn'],
            color=colors_epsilon[0], linewidth=2.0, label='ε-DDQN', alpha=0.9)
    ax.plot(gat_filtered['episode'], gat_filtered['epsilon_masac'],
            color=colors_epsilon[1], linewidth=2.0, label='ε-mSAC', alpha=0.9)

    # Find where epsilon reaches 0.01 for annotation
    epsilon_ddqn_min = gat_filtered[gat_filtered['epsilon_ddqn'] <= 0.01]
    epsilon_masac_min = gat_filtered[gat_filtered['epsilon_masac'] <= 0.01]

    if not epsilon_ddqn_min.empty:
        first_ddqn = epsilon_ddqn_min.iloc[0]
        episode_ddqn = int(first_ddqn['episode'])
        epsilon_val_ddqn = first_ddqn['epsilon_ddqn']
        print(f"  ε-DDQN reaches 0.01 at episode {episode_ddqn}, timestep {first_ddqn['step']}")

        # Draw point on the curve where ε-DDQN reaches 0.01
        ax.scatter([episode_ddqn], [epsilon_val_ddqn], color=colors_epsilon[0], s=100, zorder=5, edgecolors='black', linewidths=1.5)
        # Add text annotation
        ax.annotate(f'ε = ({epsilon_val_ddqn:.2f}, {episode_ddqn})',
                    xy=(episode_ddqn, epsilon_val_ddqn),
                    xytext=(episode_ddqn - 15, epsilon_val_ddqn + 0.08),
                    fontsize=9, color=colors_epsilon[0], fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=colors_epsilon[0], alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color=colors_epsilon[0], lw=1.5))

    if not epsilon_masac_min.empty:
        first_masac = epsilon_masac_min.iloc[0]
        episode_masac = int(first_masac['episode'])
        epsilon_val_masac = first_masac['epsilon_masac']
        print(f"  ε-MASAC reaches 0.01 at episode {episode_masac}, timestep {first_masac['step']}")

        # Draw point on the curve where ε-MASAC reaches 0.01
        ax.scatter([episode_masac], [epsilon_val_masac], color=colors_epsilon[1], s=100, zorder=5, edgecolors='black', linewidths=1.5)
        # Add text annotation
        ax.annotate(f'ε = ({epsilon_val_masac:.2f}, {episode_masac})',
                    xy=(episode_masac, epsilon_val_masac),
                    xytext=(episode_masac - 15, epsilon_val_masac + 0.05),
                    fontsize=9, color=colors_epsilon[1], fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=colors_epsilon[1], alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color=colors_epsilon[1], lw=1.5))

    # Formatting
    ax.set_xlabel('Episodes', fontsize=16, fontweight='bold')  # Increased from 12
    ax.set_ylabel('Epsilon (ε)', fontsize=16, fontweight='bold')  # Increased from 12
    ax.set_title('Exploration: Epsilon Decay Over Time', fontsize=18, fontweight='bold')  # Increased from 13
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=14)  # Increased from 9
    ax.grid(True, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA, linestyle='-')

    # Set specific axis ranges
    ax.set_xlim(0, max_episode)
    ax.set_ylim(0, 0.48)

    # Set x-axis ticks dynamically based on max episode (hide 0)
    if max_episode <= 100:
        x_tick_interval = 10
    else:
        x_tick_interval = max(10, max_episode // 10)
    ax.set_xticks(np.arange(x_tick_interval, max_episode + 1, x_tick_interval))

    # Set specific y-axis ticks: 0, 0.04, 0.08, ..., 0.48
    ax.set_yticks(np.arange(0, 0.49, 0.04))

    # Save (PNG only)
    plt.savefig(output_dir / 'graph5_epsilon_decay_episodes.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: graph5_epsilon_decay_episodes.png")


def plot_graph6_reward_distribution(gat_df, masac_df, output_dir):
    """Graph 6: Reward Distribution - Using same rewards as Graph 1 (filtered and smoothed)"""
    print("\nGenerating Graph 6: Reward Distribution (KDE)...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use the same reward processing as Graph 1
    # Filter out negative rewards
    gat_filtered = gat_df[gat_df['reward'] >= 0].copy()
    masac_filtered = masac_df[masac_df['reward'] >= 0].copy()

    # Apply running average smoothing (same as Graph 1)
    window_size = 300
    gat_rewards = gat_filtered['reward'].rolling(window=window_size, min_periods=1).mean().values
    masac_rewards = masac_filtered['reward'].rolling(window=window_size, min_periods=1).mean().values

    print(f"  GAT-CVD: {len(gat_rewards)} reward values (after filtering and smoothing)")
    print(f"  MASAC: {len(masac_rewards)} reward values (after filtering and smoothing)")

    # Calculate KDE
    from scipy.stats import gaussian_kde

    if len(gat_rewards) > 1:
        gat_kde = gaussian_kde(gat_rewards)
        x_gat = np.linspace(gat_rewards.min(), gat_rewards.max(), 300)
        y_gat = gat_kde(x_gat)

        # Plot filled KDE curve (no border line)
        ax.fill_between(x_gat, 0, y_gat, color=COLORS['gatcvd_main'], alpha=0.5, label='mGAT')

    if len(masac_rewards) > 1:
        masac_kde = gaussian_kde(masac_rewards)
        x_masac = np.linspace(masac_rewards.min(), masac_rewards.max(), 300)
        y_masac = masac_kde(x_masac)

        # Plot filled KDE curve (no border line)
        ax.fill_between(x_masac, 0, y_masac, color=COLORS['masac_main'], alpha=0.5, label='mSAC')

    # Formatting
    ax.set_xlabel('Reward', fontsize=16, fontweight='bold')  # Increased from 12
    ax.set_ylabel('Density', fontsize=16, fontweight='bold')  # Increased from 12
    ax.set_title('Reward Distribution', fontsize=18, fontweight='bold')  # Increased from 13
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=14)  # Added fontsize=14
    ax.grid(True, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA, linestyle='-')

    # Save (PNG only)
    plt.savefig(output_dir / 'graph6_reward_distribution.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: graph6_reward_distribution.png")


# ============================================================================
# CSV EXPORT FUNCTIONS
# ============================================================================

def export_csv_results(gat_df, masac_df, gat_episodes, masac_episode, metrics, output_dir):
    """Export all comparison results to CSV files"""
    print("\nExporting CSV results...")

    # 1. Timestep-aligned comparison
    timestep_comparison = pd.DataFrame({
        'timestep': gat_df['timestep'],
        'gat_episode': gat_df['episode'],
        'gat_reward': gat_df['reward'],
        'gat_avg_reward_100': gat_df['avg_reward_100'],
        'gat_success_rate': gat_df['success_rate'],
        'gat_episode_length': gat_df['episode_length'],
    })

    # Add MASAC data (interpolated to match GAT-CVD timesteps)
    masac_interp = masac_df.set_index('timestep').reindex(gat_df['timestep'], method='ffill')
    timestep_comparison['masac_episode'] = masac_interp['episode'].values
    timestep_comparison['masac_reward'] = masac_interp['reward'].values
    timestep_comparison['masac_avg_reward_100'] = masac_interp['avg_reward_100'].values
    timestep_comparison['masac_success_rate'] = masac_interp['success_rate'].values
    timestep_comparison['masac_episode_length'] = masac_interp['episode_length'].values

    timestep_comparison.to_csv(output_dir / 'comparison_timestep_aligned.csv', index=False)
    print("  ✓ Saved: comparison_timestep_aligned.csv")

    # 2. Summary statistics
    summary_stats = pd.DataFrame({
        'Metric': [
            'Final Mean Reward',
            'Final Success Rate (%)',
            'Final Episode Length',
            'Episodes to 80% Success',
            'Episodes to 90% Success',
            'Timesteps to 80% Success',
            'Timesteps to 90% Success',
        ],
        'GAT-CVD': [
            gat_df['avg_reward_100'].iloc[-1],
            gat_df['success_rate'].iloc[-1] * 100,
            gat_df['episode_length'].iloc[-1],
            metrics['gat_episodes_to_80'],
            metrics['gat_episodes_to_90'],
            metrics['gat_timesteps_to_80'],
            metrics['gat_timesteps_to_90'],
        ],
        'MASAC': [
            masac_df['avg_reward_100'].iloc[-1],
            masac_df['success_rate'].iloc[-1] * 100,
            masac_df['episode_length'].iloc[-1],
            metrics['masac_episodes_to_80'],
            metrics['masac_episodes_to_90'],
            metrics['masac_timesteps_to_80'],
            metrics['masac_timesteps_to_90'],
        ]
    })

    summary_stats.to_csv(output_dir / 'comparison_summary_statistics.csv', index=False)
    print("  ✓ Saved: comparison_summary_statistics.csv")

    # 3. Convergence metrics
    convergence_df = pd.DataFrame({
        'Threshold (%)': [80, 90],
        'GAT-CVD Episodes': [metrics['gat_episodes_to_80'], metrics['gat_episodes_to_90']],
        'GAT-CVD Timesteps': [metrics['gat_timesteps_to_80'], metrics['gat_timesteps_to_90']],
        'MASAC Episodes': [metrics['masac_episodes_to_80'], metrics['masac_episodes_to_90']],
        'MASAC Timesteps': [metrics['masac_timesteps_to_80'], metrics['masac_timesteps_to_90']],
    })

    convergence_df.to_csv(output_dir / 'convergence_metrics.csv', index=False)
    print("  ✓ Saved: convergence_metrics.csv")

    # 4. Statistical comparison (last 100 episodes)
    gat_last_100 = gat_episodes.tail(100)
    masac_last_100 = masac_episode.tail(100)

    # Perform t-tests
    t_stat_reward, p_value_reward = stats.ttest_ind(
        gat_last_100['total_reward'], masac_last_100['total_reward']
    )
    t_stat_success, p_value_success = stats.ttest_ind(
        gat_last_100['success'], masac_last_100['success']
    )

    statistical_comparison = pd.DataFrame({
        'Metric': ['Reward', 'Success Rate'],
        'GAT-CVD Mean': [
            gat_last_100['total_reward'].mean(),
            gat_last_100['success'].mean() * 100
        ],
        'GAT-CVD Std': [
            gat_last_100['total_reward'].std(),
            gat_last_100['success'].std() * 100
        ],
        'MASAC Mean': [
            masac_last_100['total_reward'].mean(),
            masac_last_100['success'].mean() * 100
        ],
        'MASAC Std': [
            masac_last_100['total_reward'].std(),
            masac_last_100['success'].std() * 100
        ],
        't-statistic': [t_stat_reward, t_stat_success],
        'p-value': [p_value_reward, p_value_success],
        'Significant (p<0.05)': [p_value_reward < 0.05, p_value_success < 0.05]
    })

    statistical_comparison.to_csv(output_dir / 'statistical_comparison.csv', index=False)
    print("  ✓ Saved: statistical_comparison.csv")

    # 5. LaTeX table
    latex_table = generate_latex_table(summary_stats, statistical_comparison)
    with open(output_dir / 'comparison_table.tex', 'w') as f:
        f.write(latex_table)
    print("  ✓ Saved: comparison_table.tex")


def generate_latex_table(summary_stats, statistical_comparison):
    """Generate LaTeX table for thesis"""
    latex = r"""\begin{table}[htbp]
\centering
\caption{Training Comparison: GAT-CVD vs MASAC (12,933 Timesteps)}
\label{tab:gatcvd_vs_masac}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{GAT-CVD} & \textbf{MASAC} \\
\midrule
"""

    for _, row in summary_stats.iterrows():
        metric = row['Metric']
        gat_val = row['GAT-CVD']
        masac_val = row['MASAC']

        if 'Episodes' in metric or 'Timesteps' in metric:
            latex += f"{metric} & {gat_val:.0f} & {masac_val:.0f} \\\\\n"
        else:
            latex += f"{metric} & {gat_val:.2f} & {masac_val:.2f} \\\\\n"

    latex += r"""\midrule
\multicolumn{3}{l}{\textit{Statistical Comparison (Last 100 Episodes)}} \\
"""

    for _, row in statistical_comparison.iterrows():
        metric = row['Metric']
        gat_mean = row['GAT-CVD Mean']
        gat_std = row['GAT-CVD Std']
        masac_mean = row['MASAC Mean']
        masac_std = row['MASAC Std']
        p_val = row['p-value']
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''

        latex += f"{metric} & ${gat_mean:.2f} \\pm {gat_std:.2f}$ & ${masac_mean:.2f} \\pm {masac_std:.2f}${sig} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    return latex


def main():
    print("="*80)
    print("GAT-CVD vs MASAC Training Comparison")
    print("="*80)

    # Setup
    setup_plot_style()
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Load data
    gat_training, gat_episodes = load_gatcvd_data()
    masac_timestep, masac_episode, masac_summary = load_masac_data()

    # Prepare comparison data
    gat_df, masac_df = prepare_comparison_data(gat_training, gat_episodes, masac_timestep, masac_episode)

    # Calculate metrics
    convergence_metrics = calculate_convergence_metrics(gat_df, masac_df)

    # Generate graphs
    plot_graph1_learning_curves(gat_df, masac_df, OUTPUT_DIR)
    plot_graph2_success_rate(gat_df, masac_df, OUTPUT_DIR)
    plot_graph3_convergence(gat_df, masac_df, OUTPUT_DIR)  # Updated to use reward thresholds
    plot_graph4_qvalue_curves(gat_training, OUTPUT_DIR)  # Plot Q-values instead of loss
    plot_graph5_epsilon_decay(gat_training, OUTPUT_DIR)  # Epsilon decay curves (timesteps)
    plot_graph5b_epsilon_decay_episodes(gat_training, OUTPUT_DIR)  # Epsilon decay curves (episodes)
    plot_graph6_reward_distribution(gat_df, masac_df, OUTPUT_DIR)  # Use same rewards as Graph 1

    # Export CSV files
    export_csv_results(gat_df, masac_df, gat_episodes, masac_episode, convergence_metrics, OUTPUT_DIR)

    print("\n" + "="*80)
    print("Comparison complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()

