"""
Generate graphs for GAT+CVD test results
- Reward by episode (bar graph)
- Pick success, distance reduced, distance efficiency, time saved, time efficiency (line graphs)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline

# Load both seeds
seed_42_csv = Path("multiagent/scripts/experiments/rlmodels/mutliagent/ddqn_gatcvd/gat_cvd_test_results/discrete/seed_42/episode_results.csv")
seed_123_csv = Path("multiagent/scripts/experiments/rlmodels/mutliagent/ddqn_gatcvd/gat_cvd_test_results/discrete/seed_123/episode_results.csv")

df_42 = pd.read_csv(seed_42_csv)
df_123 = pd.read_csv(seed_123_csv)

# Combine data from both seeds
df_combined = pd.concat([df_42, df_123], ignore_index=True)

# Output directory
output_dir = Path("cobotproject/scripts/experiments/rlmodels/mutliagent/ddqn_gatcvd/gat_cvd_test_results/discrete")
output_dir.mkdir(parents=True, exist_ok=True)

# Set up seaborn theme and font
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'Linux Libertine'
plt.rcParams['font.size'] = 10

# Define custom color palette
custom_colors = {
    'DDQN + mGAT': '#17BECF',  # teal
    'Heuristic': '#CD5C5C',  # light maroon
    'Duel-DDQN + SAC': '#8C564B',  # brown
    'PER-DDQN-Full + SAC': '#FF7F0E'  # orange
}

# ============================================================================
# BAR GRAPH: Reward by Episode (DDQN+mGAT and 2 other models - user will decide)
# ============================================================================

# For now, using DDQN+mGAT, Duel-DDQN+SAC, PER-DDQN-Full+SAC
models_to_plot_reward = ['DDQN+GAT', 'Duel-DDQN+SAC', 'PER-DDQN-Full+SAC']
model_display_names_reward = {
    'DDQN+GAT': 'DDQN + mGAT',
    'Duel-DDQN+SAC': 'Duel-DDQN + SAC',
    'PER-DDQN-Full+SAC': 'PER-DDQN-Full + SAC'
}

# Filter for selected models
df_plot_reward = df_combined[df_combined['model'].isin(models_to_plot_reward)].copy()
df_plot_reward['model_display'] = df_plot_reward['model'].map(model_display_names_reward)

# Filter for episodes 2, 4, 6, 8, 10, 12, 14, 16
episodes_to_plot = [2, 4, 6, 8, 10, 12, 14, 16]
df_filtered_reward = df_plot_reward[df_plot_reward['episode'].isin(episodes_to_plot)].copy()

# Calculate average reward for each model at each episode (across both seeds)
df_grouped_reward = df_filtered_reward.groupby(['model_display', 'episode'])['total_reward'].mean().reset_index()

# Create figure
fig, ax = plt.subplots(figsize=(7, 4.5))

# Create bar plot
bar_plot = sns.barplot(
    data=df_grouped_reward,
    x='episode',
    y='total_reward',
    hue='model_display',
    palette=[custom_colors['DDQN + mGAT'], custom_colors['Duel-DDQN + SAC'], custom_colors['PER-DDQN-Full + SAC']],
    ax=ax,
    width=0.4
)

# Customize grid
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)

# Set labels and title
ax.set_xlabel('Episode', fontsize=10)
ax.set_ylabel('Reward', fontsize=10)
ax.set_title('Reward by episode', fontsize=10)

# Customize legend
ax.legend(title='', fontsize=8, frameon=True, loc='lower right')

# Set x-axis tick labels
ax.set_xticklabels(episodes_to_plot, fontsize=10)
ax.tick_params(axis='y', labelsize=10)

# Tight layout
plt.tight_layout()

# Save figure
output_path_reward = output_dir / "reward_by_episode.png"
plt.savefig(output_path_reward, dpi=300, bbox_inches='tight')
print(f"✓ Bar graph saved to: {output_path_reward}")
plt.close()

# ============================================================================
# Helper function for line charts
# ============================================================================

def create_line_chart(df_combined, models_to_plot, model_display_names, metric_column, 
                      y_label, title, output_filename, y_lim=None, y_ticks=None):
    """Create a line chart comparing models across episodes"""
    
    # Filter for selected models
    df_plot = df_combined[df_combined['model'].isin(models_to_plot)].copy()
    df_plot['model_display'] = df_plot['model'].map(model_display_names)
    
    # Filter for all episodes from 1 to 20
    all_episodes = list(range(1, 21))
    df_filtered = df_plot[df_plot['episode'].isin(all_episodes)].copy()
    
    # Calculate metric if needed (for success rate)
    if metric_column == 'success_rate':
        df_filtered['num_cubes'] = df_filtered['num_cubes'].astype(int)
        df_filtered['success_rate'] = (df_filtered['cubes_picked'] / df_filtered['num_cubes']) * 100
    
    # Calculate average metric across both seeds
    df_grouped = df_filtered.groupby(['model_display', 'episode'])[metric_column].mean().reset_index()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Create line plot with smoothing
    for model in model_display_names.values():
        model_data = df_grouped[df_grouped['model_display'] == model].sort_values('episode')
        
        # Get x and y data
        x = model_data['episode'].values
        y = model_data[metric_column].values
        
        # Create smooth curve using spline interpolation
        if len(x) > 3:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spl = make_interp_spline(x, y, k=3)
            y_smooth = spl(x_smooth)
            
            # Plot smooth curve
            ax.plot(x_smooth, y_smooth, linewidth=0.75, color=custom_colors[model])
            
            # Plot markers at actual data points
            ax.plot(x, y, marker='o', linewidth=0, markersize=2, label=model, color=custom_colors[model])
        else:
            ax.plot(x, y, marker='o', linewidth=0.75, markersize=2, label=model, color=custom_colors[model])
    
    # Customize grid
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    
    # Set labels and title
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=10)
    
    # Customize legend
    ax.legend(fontsize=8, frameon=True, loc='lower right')
    
    # Set x-axis ticks
    x_ticks_display = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    ax.set_xticks(x_ticks_display)
    ax.set_xticklabels(x_ticks_display, fontsize=10)
    ax.set_xlim(1, 20)
    
    # Set y-axis
    if y_lim:
        ax.set_ylim(y_lim)
    if y_ticks:
        ax.set_yticks(y_ticks)
    ax.tick_params(axis='y', labelsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Line chart saved to: {output_path}")
    plt.close()

# ============================================================================
# LINE CHARTS: DDQN+mGAT vs Heuristic
# ============================================================================

models_to_plot_line = ['DDQN+GAT', 'Heuristic']
model_display_names_line = {
    'DDQN+GAT': 'DDQN + mGAT',
    'Heuristic': 'Heuristic'
}

# 1. Pick Success by Episode
create_line_chart(
    df_combined=df_combined,
    models_to_plot=models_to_plot_line,
    model_display_names=model_display_names_line,
    metric_column='success_rate',
    y_label='Pick success (%)',
    title='Pick success by models',
    output_filename='pick_success_by_episode.png',
    y_lim=(0, 105),
    y_ticks=[0, 20, 40, 60, 80, 100]
)

# 2. Distance Reduced by Episode
create_line_chart(
    df_combined=df_combined,
    models_to_plot=models_to_plot_line,
    model_display_names=model_display_names_line,
    metric_column='total_distance_reduced',
    y_label='Distance reduced (m)',
    title='Distance reduced by models',
    output_filename='distance_reduced_by_episode.png'
)

# 3. Distance Efficiency by Episode
create_line_chart(
    df_combined=df_combined,
    models_to_plot=models_to_plot_line,
    model_display_names=model_display_names_line,
    metric_column='distance_efficiency',
    y_label='Distance efficiency (%)',
    title='Distance efficiency by models',
    output_filename='distance_efficiency_by_episode.png'
)

# 4. Time Saved by Episode
create_line_chart(
    df_combined=df_combined,
    models_to_plot=models_to_plot_line,
    model_display_names=model_display_names_line,
    metric_column='total_time_saved',
    y_label='Time saved (s)',
    title='Time saved by models',
    output_filename='time_saved_by_episode.png'
)

# 5. Time Efficiency by Episode
create_line_chart(
    df_combined=df_combined,
    models_to_plot=models_to_plot_line,
    model_display_names=model_display_names_line,
    metric_column='time_efficiency',
    y_label='Time efficiency (%)',
    title='Time efficiency by models',
    output_filename='time_efficiency_by_episode.png'
)

print("\n✅ All graphs generated successfully!")

