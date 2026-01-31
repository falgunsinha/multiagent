"""
Compare Training Results Across All Configurations
Analyzes TensorBoard logs and metadata to find the best model.
Generates comparison charts and plots training curves.
Uses Plotly for interactive visualizations and Seaborn for static plots.

Usage:
    py -3.11 compare_training_results.py
    py -3.11 compare_training_results.py --save_plots  # Save plots to files
    py -3.11 compare_training_results.py --save_plots --interactive  # Use Plotly for interactive plots
"""

import os
import json
import glob
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import argparse

def load_metadata(metadata_path):
    """Load metadata JSON file"""
    with open(metadata_path, 'r') as f:
        return json.load(f)

def parse_tensorboard_logs(log_dir):
    """Parse TensorBoard event files to get metrics and full training curves"""
    try:
        from tensorboard.backend.event_processing import event_accumulator

        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()

        # Get episode reward curve
        rewards_curve = []
        if 'rollout/ep_rew_mean' in ea.Tags()['scalars']:
            rewards = ea.Scalars('rollout/ep_rew_mean')
            rewards_curve = [(r.step, r.value) for r in rewards]
            final_reward = rewards[-1].value if rewards else None
        else:
            final_reward = None

        # Get episode length curve
        lengths_curve = []
        if 'rollout/ep_len_mean' in ea.Tags()['scalars']:
            lengths = ea.Scalars('rollout/ep_len_mean')
            lengths_curve = [(l.step, l.value) for l in lengths]
            final_length = lengths[-1].value if lengths else None
        else:
            final_length = None

        # Get entropy loss curve
        entropy_curve = []
        if 'train/entropy_loss' in ea.Tags()['scalars']:
            entropy_losses = ea.Scalars('train/entropy_loss')
            entropy_curve = [(e.step, e.value) for e in entropy_losses]
            entropy_loss = entropy_losses[-1].value if entropy_losses else None
        else:
            entropy_loss = None

        # Get value loss curve
        value_curve = []
        if 'train/value_loss' in ea.Tags()['scalars']:
            value_losses = ea.Scalars('train/value_loss')
            value_curve = [(v.step, v.value) for v in value_losses]
            value_loss = value_losses[-1].value if value_losses else None
        else:
            value_loss = None

        # Get policy loss curve
        policy_curve = []
        if 'train/policy_gradient_loss' in ea.Tags()['scalars']:
            policy_losses = ea.Scalars('train/policy_gradient_loss')
            policy_curve = [(p.step, p.value) for p in policy_losses]
            policy_loss = policy_losses[-1].value if policy_losses else None
        else:
            policy_loss = None

        # Get learning rate curve
        lr_curve = []
        if 'train/learning_rate' in ea.Tags()['scalars']:
            lrs = ea.Scalars('train/learning_rate')
            lr_curve = [(lr.step, lr.value) for lr in lrs]

        return {
            'final_reward': final_reward,
            'final_length': final_length,
            'entropy_loss': entropy_loss,
            'value_loss': value_loss,
            'policy_loss': policy_loss,
            'rewards_curve': rewards_curve,
            'lengths_curve': lengths_curve,
            'entropy_curve': entropy_curve,
            'value_curve': value_curve,
            'policy_curve': policy_curve,
            'lr_curve': lr_curve
        }
    except Exception as e:
        print(f"Warning: Could not parse TensorBoard logs: {e}")
        return {
            'final_reward': None,
            'final_length': None,
            'entropy_loss': None,
            'value_loss': None,
            'policy_loss': None,
            'rewards_curve': [],
            'lengths_curve': [],
            'entropy_curve': [],
            'value_curve': [],
            'policy_curve': [],
            'lr_curve': []
        }

def print_ascii_bar_chart(data, title, max_width=60):
    """Print ASCII bar chart in terminal"""
    print(f"\n{title}")
    print("=" * 80)

    if not data:
        print("No data available")
        return

    # Find max value for scaling
    max_val = max(v for _, v in data)
    min_val = min(v for _, v in data)

    # Handle edge case where all values are the same
    if max_val == min_val:
        scale = 1.0
    else:
        scale = max_width / (max_val - min_val)

    for label, value in data:
        # Calculate bar length
        if max_val == min_val:
            bar_len = max_width // 2
        else:
            bar_len = int((value - min_val) * scale)

        # Create bar
        bar = "█" * bar_len

        # Print with value
        print(f"{label:40s} {bar} {value:.2f}")

    print()


def plot_training_curves_ascii(all_curves, title, ylabel, max_width=70, max_height=20):
    """Plot training curves as ASCII art in terminal"""
    print(f"\n{title}")
    print("=" * 80)

    if not all_curves:
        print("No data available")
        return

    # Find global min/max for y-axis
    all_values = []
    for _, curve in all_curves.items():
        if curve:
            all_values.extend([v for _, v in curve])

    if not all_values:
        print("No data available")
        return

    y_min = min(all_values)
    y_max = max(all_values)

    # Handle edge case
    if y_max == y_min:
        y_max = y_min + 1

    # Create ASCII plot
    y_range = y_max - y_min
    y_step = y_range / max_height

    # Print y-axis labels and plot
    for row in range(max_height, -1, -1):
        y_val = y_min + (row * y_step)

        # Y-axis label
        print(f"{y_val:8.2f} │", end="")

        # Plot points for each curve
        line = [" "] * max_width

        for method_name, curve in all_curves.items():
            if not curve:
                continue

            # Get symbol for this method
            symbol = {"heuristic": "○", "astar": "△", "rrt": "□"}.get(method_name, "·")

            # Find max step for x-axis scaling
            max_step = max(step for step, _ in curve)
            x_scale = max_width / max_step if max_step > 0 else 1

            # Plot points
            for step, value in curve:
                x_pos = int(step * x_scale)
                y_pos = int((value - y_min) / y_step)

                if y_pos == row and 0 <= x_pos < max_width:
                    line[x_pos] = symbol

        print("".join(line))

    # X-axis
    print(f"{'':8s} └" + "─" * max_width)
    print(f"{'':9s} 0{' ' * (max_width - 10)}timesteps")

    # Legend
    print(f"\nLegend: ○ Heuristic  △ A*  □ RRT")
    print()


def save_plots_plotly(all_data, output_dir="training_plots"):
    """Save interactive plots using Plotly"""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        import pandas as pd

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\n[PLOTLY] Saving interactive plots to {output_path.absolute()}/")

        # Prepare data for plotting
        plot_data = []
        for data in all_data:
            if data['final_reward'] is not None:
                plot_data.append({
                    'Model': data['model'],
                    'Method': data['method'].capitalize(),
                    'Configuration': f"{data['grid']} - {data['cubes']} cubes",
                    'Grid': data['grid'],
                    'Cubes': data['cubes'],
                    'Final Reward': data['final_reward'],
                    'Entropy Loss': abs(data['entropy_loss']) if data['entropy_loss'] is not None else None,
                    'Value Loss': data['value_loss'],
                    'Policy Loss': abs(data['policy_loss']) if data['policy_loss'] is not None else None,
                    'Episode Length': data['final_length']
                })

        df_plot = pd.DataFrame(plot_data)

        # Plot 1: Interactive Grouped Bar Chart (Final Rewards)
        if not df_plot.empty:
            fig = px.bar(df_plot, x='Configuration', y='Final Reward', color='Method',
                        barmode='group',
                        title='Final Rewards Comparison Across Methods and Configurations',
                        labels={'Final Reward': 'Final Reward', 'Configuration': 'Configuration'},
                        color_discrete_map={'Heuristic': '#1f77b4', 'Astar': '#2ca02c', 'Rrt': '#d62728'},
                        hover_data=['Model', 'Episode Length'])

            fig.update_layout(
                font=dict(size=12),
                title_font=dict(size=16, family='Arial Black'),
                xaxis_title_font=dict(size=14, family='Arial', color='black'),
                yaxis_title_font=dict(size=14, family='Arial', color='black'),
                legend_title_font=dict(size=12),
                hovermode='closest',
                template='plotly_white'
            )

            fig.write_html(output_path / "final_rewards_interactive.html")
            print(f"  ✓ Saved: final_rewards_interactive.html (plotly interactive)")

        # Plot 2: Interactive Training Curves (Rewards)
        curve_data = []
        for data in all_data:
            if data['rewards_curve']:
                for step, value in data['rewards_curve']:
                    curve_data.append({
                        'Timesteps': step,
                        'Reward': value,
                        'Method': data['method'].capitalize(),
                        'Configuration': f"{data['grid']} - {data['cubes']}c",
                        'Model': data['model']
                    })

        if curve_data:
            df_curves = pd.DataFrame(curve_data)

            fig = px.line(df_curves, x='Timesteps', y='Reward', color='Method',
                         line_dash='Configuration',
                         title='Training Progress: Episode Rewards Over Time',
                         labels={'Reward': 'Episode Reward Mean', 'Timesteps': 'Training Timesteps'},
                         hover_data=['Model'])

            fig.update_layout(
                font=dict(size=12),
                title_font=dict(size=16, family='Arial Black'),
                xaxis_title_font=dict(size=14, family='Arial', color='black'),
                yaxis_title_font=dict(size=14, family='Arial', color='black'),
                legend_title_font=dict(size=12),
                hovermode='x unified',
                template='plotly_white'
            )

            fig.write_html(output_path / "reward_curves_interactive.html")
            print(f"  ✓ Saved: reward_curves_interactive.html (plotly interactive)")

        # Plot 3: Interactive Multi-panel Loss Curves
        entropy_data = []
        value_data = []
        policy_data = []

        for data in all_data:
            method_label = f"{data['method'].capitalize()} - {data['grid']} - {data['cubes']}c"

            if data['entropy_curve']:
                for step, value in data['entropy_curve']:
                    entropy_data.append({
                        'Timesteps': step,
                        'Loss': value,
                        'Model': method_label,
                        'Method': data['method'].capitalize()
                    })

            if data['value_curve']:
                for step, value in data['value_curve']:
                    value_data.append({
                        'Timesteps': step,
                        'Loss': value,
                        'Model': method_label,
                        'Method': data['method'].capitalize()
                    })

            if data['policy_curve']:
                for step, value in data['policy_curve']:
                    policy_data.append({
                        'Timesteps': step,
                        'Loss': value,
                        'Model': method_label,
                        'Method': data['method'].capitalize()
                    })

        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Entropy Loss', 'Value Loss', 'Policy Loss'),
            vertical_spacing=0.08,
            shared_xaxes=True
        )

        # Add entropy loss traces
        if entropy_data:
            df_entropy = pd.DataFrame(entropy_data)
            for method in df_entropy['Method'].unique():
                df_method = df_entropy[df_entropy['Method'] == method]
                for model in df_method['Model'].unique():
                    df_model = df_method[df_method['Model'] == model]
                    fig.add_trace(
                        go.Scatter(x=df_model['Timesteps'], y=df_model['Loss'],
                                 mode='lines', name=model, legendgroup=model,
                                 showlegend=True),
                        row=1, col=1
                    )

        # Add value loss traces
        if value_data:
            df_value = pd.DataFrame(value_data)
            for method in df_value['Method'].unique():
                df_method = df_value[df_value['Method'] == method]
                for model in df_method['Model'].unique():
                    df_model = df_method[df_method['Model'] == model]
                    fig.add_trace(
                        go.Scatter(x=df_model['Timesteps'], y=df_model['Loss'],
                                 mode='lines', name=model, legendgroup=model,
                                 showlegend=False),
                        row=2, col=1
                    )

        # Add policy loss traces
        if policy_data:
            df_policy = pd.DataFrame(policy_data)
            for method in df_policy['Method'].unique():
                df_method = df_policy[df_policy['Method'] == method]
                for model in df_method['Model'].unique():
                    df_model = df_method[df_method['Model'] == model]
                    fig.add_trace(
                        go.Scatter(x=df_model['Timesteps'], y=df_model['Loss'],
                                 mode='lines', name=model, legendgroup=model,
                                 showlegend=False),
                        row=3, col=1
                    )

        fig.update_layout(
            title_text='Training Losses Over Time',
            title_font=dict(size=16, family='Arial Black'),
            hovermode='x unified',
            template='plotly_white',
            height=900
        )

        fig.update_xaxes(title_text='Timesteps', row=3, col=1)
        fig.update_yaxes(title_text='Entropy Loss', row=1, col=1)
        fig.update_yaxes(title_text='Value Loss', row=2, col=1)
        fig.update_yaxes(title_text='Policy Loss', row=3, col=1)

        fig.write_html(output_path / "loss_curves_interactive.html")
        print(f"  ✓ Saved: loss_curves_interactive.html (plotly interactive)")

        # Plot 4: Interactive Heatmap
        if not df_plot.empty:
            heatmap_data = []
            for _, row in df_plot.iterrows():
                model_label = f"{row['Method']}\n{row['Configuration']}"
                heatmap_data.append({
                    'Model': model_label,
                    'Final Reward': row['Final Reward'],
                    'Entropy Loss': row['Entropy Loss'] if row['Entropy Loss'] is not None else 0,
                    'Value Loss': row['Value Loss'] if row['Value Loss'] is not None else 0,
                    'Policy Loss': row['Policy Loss'] if row['Policy Loss'] is not None else 0
                })

            df_heatmap = pd.DataFrame(heatmap_data)
            df_heatmap = df_heatmap.set_index('Model')

            fig = go.Figure(data=go.Heatmap(
                z=df_heatmap.T.values,
                x=df_heatmap.index,
                y=df_heatmap.columns,
                colorscale='RdYlGn_r',
                text=df_heatmap.T.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                hovertemplate='Model: %{x}<br>Metric: %{y}<br>Value: %{z:.2f}<extra></extra>'
            ))

            fig.update_layout(
                title='Metrics Comparison Heatmap',
                title_font=dict(size=16, family='Arial Black'),
                xaxis_title='Model Configuration',
                yaxis_title='Metric',
                template='plotly_white',
                height=600
            )

            fig.write_html(output_path / "metrics_heatmap_interactive.html")
            print(f"  ✓ Saved: metrics_heatmap_interactive.html (plotly interactive)")

        # Plot 5: Interactive 3D Scatter Plot (Reward vs Losses)
        if not df_plot.empty:
            fig = go.Figure(data=[go.Scatter3d(
                x=df_plot['Final Reward'],
                y=df_plot['Value Loss'],
                z=df_plot['Entropy Loss'],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color=df_plot['Final Reward'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Final Reward")
                ),
                text=df_plot['Method'],
                textposition='top center',
                hovertemplate='<b>%{text}</b><br>' +
                             'Final Reward: %{x:.2f}<br>' +
                             'Value Loss: %{y:.3f}<br>' +
                             'Entropy Loss: %{z:.2f}<br>' +
                             '<extra></extra>'
            )])

            fig.update_layout(
                title='3D Performance Space: Reward vs Losses',
                title_font=dict(size=16, family='Arial Black'),
                scene=dict(
                    xaxis_title='Final Reward',
                    yaxis_title='Value Loss',
                    zaxis_title='Entropy Loss (abs)'
                ),
                template='plotly_white',
                height=700
            )

            fig.write_html(output_path / "performance_3d_interactive.html")
            print(f"  ✓ Saved: performance_3d_interactive.html (plotly 3D scatter)")

        print(f"\n[PLOTLY] All interactive plots saved to: {output_path.absolute()}/\n")

        return True

    except ImportError:
        print("\n[PLOTLY] Plotly not available, skipping interactive plots")
        print("         Install with: pip install plotly")
        return False
    except Exception as e:
        print(f"\n[PLOTLY] Error generating interactive plots: {e}")
        return False


def save_plots_matplotlib(all_data, output_dir="training_plots"):
    """Save plots using matplotlib and seaborn (optional, if available)"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        # Try to import seaborn for better styling
        try:
            import seaborn as sns
            sns.set_theme(style="whitegrid", palette="husl")
            sns.set_context("notebook", font_scale=1.1)
            use_seaborn = True
            print(f"\n[PLOTS] Using seaborn for enhanced visualizations")
        except ImportError:
            use_seaborn = False
            print(f"\n[PLOTS] Seaborn not available, using matplotlib defaults")
            print(f"        Install with: pip install seaborn")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"[PLOTS] Saving plots to {output_path.absolute()}/")

        # Prepare data for seaborn
        import pandas as pd

        # Create DataFrame for easier plotting
        plot_data = []
        for data in all_data:
            if data['final_reward'] is not None:
                plot_data.append({
                    'Method': data['method'].capitalize(),
                    'Configuration': f"{data['grid']} - {data['cubes']} cubes",
                    'Grid': data['grid'],
                    'Cubes': data['cubes'],
                    'Final Reward': data['final_reward'],
                    'Entropy Loss': abs(data['entropy_loss']) if data['entropy_loss'] is not None else None,
                    'Value Loss': data['value_loss'],
                    'Policy Loss': abs(data['policy_loss']) if data['policy_loss'] is not None else None,
                    'Episode Length': data['final_length']
                })

        df_plot = pd.DataFrame(plot_data)

        # Plot 1: Final Rewards Comparison (Seaborn Barplot)
        if use_seaborn and not df_plot.empty:
            fig, ax = plt.subplots(figsize=(14, 7))

            sns.barplot(data=df_plot, x='Configuration', y='Final Reward', hue='Method', ax=ax, palette='Set2')
            ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
            ax.set_ylabel('Final Reward', fontsize=12, fontweight='bold')
            ax.set_title('Final Rewards Comparison Across Methods and Configurations', fontsize=14, fontweight='bold', pad=20)
            ax.legend(title='Method', title_fontsize=11, fontsize=10, loc='upper left')
            plt.xticks(rotation=45, ha='right')

            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f', padding=3, fontsize=9)

            plt.tight_layout()
            plt.savefig(output_path / "final_rewards.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: final_rewards.png (seaborn barplot)")
        elif not df_plot.empty:
            # Fallback to matplotlib
            fig, ax = plt.subplots(figsize=(12, 6))

            methods = df_plot['Method'].tolist()
            configs = df_plot['Configuration'].tolist()
            rewards = df_plot['Final Reward'].tolist()

            x = range(len(rewards))
            colors = {'Heuristic': 'blue', 'Astar': 'green', 'Rrt': 'red'}
            bar_colors = [colors.get(m, 'gray') for m in methods]

            ax.bar(x, rewards, color=bar_colors, alpha=0.7)
            ax.set_xlabel('Configuration')
            ax.set_ylabel('Final Reward')
            ax.set_title('Final Rewards Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels([f"{m}\n{c}" for m, c in zip(methods, configs)], rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path / "final_rewards.png", dpi=150)
            plt.close()
            print(f"  ✓ Saved: final_rewards.png")

        # Plot 2: Training Curves (Rewards) - Seaborn Lineplot
        if use_seaborn:
            # Prepare data for seaborn
            curve_data = []
            for data in all_data:
                if data['rewards_curve']:
                    for step, value in data['rewards_curve']:
                        curve_data.append({
                            'Timesteps': step,
                            'Reward': value,
                            'Method': data['method'].capitalize(),
                            'Configuration': f"{data['grid']} - {data['cubes']}c"
                        })

            if curve_data:
                df_curves = pd.DataFrame(curve_data)

                fig, ax = plt.subplots(figsize=(14, 7))

                sns.lineplot(data=df_curves, x='Timesteps', y='Reward', hue='Method',
                           style='Configuration', ax=ax, linewidth=2, alpha=0.8)
                ax.set_xlabel('Timesteps', fontsize=12, fontweight='bold')
                ax.set_ylabel('Episode Reward Mean', fontsize=12, fontweight='bold')
                ax.set_title('Training Progress: Episode Rewards Over Time', fontsize=14, fontweight='bold', pad=20)
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)

                plt.tight_layout()
                plt.savefig(output_path / "reward_curves.png", dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved: reward_curves.png (seaborn lineplot)")
        else:
            # Fallback to matplotlib
            fig, ax = plt.subplots(figsize=(12, 6))

            for data in all_data:
                if data['rewards_curve']:
                    steps, values = zip(*data['rewards_curve'])
                    label = f"{data['method']} - {data['grid']} - {data['cubes']} cubes"
                    ax.plot(steps, values, label=label, alpha=0.7)

            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Episode Reward Mean')
            ax.set_title('Training Progress: Episode Rewards')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path / "reward_curves.png", dpi=150)
            plt.close()
            print(f"  ✓ Saved: reward_curves.png")

        # Plot 3: Loss Curves - Seaborn Multi-panel
        if use_seaborn:
            # Prepare data for all three losses
            entropy_data = []
            value_data = []
            policy_data = []

            for data in all_data:
                method_label = f"{data['method'].capitalize()} - {data['grid']} - {data['cubes']}c"

                if data['entropy_curve']:
                    for step, value in data['entropy_curve']:
                        entropy_data.append({
                            'Timesteps': step,
                            'Loss': value,
                            'Model': method_label,
                            'Method': data['method'].capitalize()
                        })

                if data['value_curve']:
                    for step, value in data['value_curve']:
                        value_data.append({
                            'Timesteps': step,
                            'Loss': value,
                            'Model': method_label,
                            'Method': data['method'].capitalize()
                        })

                if data['policy_curve']:
                    for step, value in data['policy_curve']:
                        policy_data.append({
                            'Timesteps': step,
                            'Loss': value,
                            'Model': method_label,
                            'Method': data['method'].capitalize()
                        })

            fig, axes = plt.subplots(3, 1, figsize=(14, 12))

            # Entropy Loss
            if entropy_data:
                df_entropy = pd.DataFrame(entropy_data)
                sns.lineplot(data=df_entropy, x='Timesteps', y='Loss', hue='Method',
                           style='Model', ax=axes[0], linewidth=2, alpha=0.8)
                axes[0].set_ylabel('Entropy Loss', fontsize=11, fontweight='bold')
                axes[0].set_title('Training Losses Over Time', fontsize=14, fontweight='bold', pad=20)
                axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
                axes[0].set_xlabel('')

            # Value Loss
            if value_data:
                df_value = pd.DataFrame(value_data)
                sns.lineplot(data=df_value, x='Timesteps', y='Loss', hue='Method',
                           style='Model', ax=axes[1], linewidth=2, alpha=0.8, legend=False)
                axes[1].set_ylabel('Value Loss', fontsize=11, fontweight='bold')
                axes[1].set_xlabel('')

            # Policy Loss
            if policy_data:
                df_policy = pd.DataFrame(policy_data)
                sns.lineplot(data=df_policy, x='Timesteps', y='Loss', hue='Method',
                           style='Model', ax=axes[2], linewidth=2, alpha=0.8, legend=False)
                axes[2].set_ylabel('Policy Loss', fontsize=11, fontweight='bold')
                axes[2].set_xlabel('Timesteps', fontsize=11, fontweight='bold')

            plt.tight_layout()
            plt.savefig(output_path / "loss_curves.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: loss_curves.png (seaborn multi-panel)")
        else:
            # Fallback to matplotlib
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))

            # Entropy Loss
            for data in all_data:
                if data['entropy_curve']:
                    steps, values = zip(*data['entropy_curve'])
                    label = f"{data['method']} - {data['grid']} - {data['cubes']} cubes"
                    axes[0].plot(steps, values, label=label, alpha=0.7)
            axes[0].set_ylabel('Entropy Loss')
            axes[0].set_title('Training Losses')
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            axes[0].grid(alpha=0.3)

            # Value Loss
            for data in all_data:
                if data['value_curve']:
                    steps, values = zip(*data['value_curve'])
                    axes[1].plot(steps, values, alpha=0.7)
            axes[1].set_ylabel('Value Loss')
            axes[1].grid(alpha=0.3)

            # Policy Loss
            for data in all_data:
                if data['policy_curve']:
                    steps, values = zip(*data['policy_curve'])
                    axes[2].plot(steps, values, alpha=0.7)
            axes[2].set_xlabel('Timesteps')
            axes[2].set_ylabel('Policy Loss')
            axes[2].grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path / "loss_curves.png", dpi=150)
            plt.close()
            print(f"  ✓ Saved: loss_curves.png")

        # Plot 4: Metrics Comparison Heatmap (Seaborn only)
        if use_seaborn and not df_plot.empty:
            # Create pivot table for heatmap
            metrics_for_heatmap = ['Final Reward', 'Entropy Loss', 'Value Loss', 'Policy Loss']

            # Normalize metrics for better visualization
            df_normalized = df_plot.copy()
            for metric in metrics_for_heatmap:
                if metric in df_normalized.columns and df_normalized[metric].notna().any():
                    min_val = df_normalized[metric].min()
                    max_val = df_normalized[metric].max()
                    if max_val > min_val:
                        df_normalized[f'{metric}_norm'] = (df_normalized[metric] - min_val) / (max_val - min_val)

            # Create heatmap data
            heatmap_data = []
            for _, row in df_plot.iterrows():
                model_label = f"{row['Method']}\n{row['Configuration']}"
                heatmap_data.append({
                    'Model': model_label,
                    'Final Reward': row['Final Reward'],
                    'Entropy Loss': row['Entropy Loss'] if row['Entropy Loss'] is not None else 0,
                    'Value Loss': row['Value Loss'] if row['Value Loss'] is not None else 0,
                    'Policy Loss': row['Policy Loss'] if row['Policy Loss'] is not None else 0
                })

            df_heatmap = pd.DataFrame(heatmap_data)
            df_heatmap = df_heatmap.set_index('Model')

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df_heatmap.T, annot=True, fmt='.2f', cmap='RdYlGn_r',
                       linewidths=0.5, ax=ax, cbar_kws={'label': 'Metric Value'})
            ax.set_title('Metrics Comparison Heatmap', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Model Configuration', fontsize=11, fontweight='bold')
            ax.set_ylabel('Metric', fontsize=11, fontweight='bold')

            plt.tight_layout()
            plt.savefig(output_path / "metrics_heatmap.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: metrics_heatmap.png (seaborn heatmap)")

        # Plot 5: Loss Distribution Violin Plot (Seaborn only)
        if use_seaborn and not df_plot.empty:
            # Prepare data for violin plot
            loss_data = []
            for _, row in df_plot.iterrows():
                if row['Entropy Loss'] is not None:
                    loss_data.append({'Method': row['Method'], 'Loss Type': 'Entropy', 'Value': row['Entropy Loss']})
                if row['Value Loss'] is not None:
                    loss_data.append({'Method': row['Method'], 'Loss Type': 'Value', 'Value': row['Value Loss']})
                if row['Policy Loss'] is not None:
                    loss_data.append({'Method': row['Method'], 'Loss Type': 'Policy', 'Value': row['Policy Loss']})

            if loss_data:
                df_losses = pd.DataFrame(loss_data)

                fig, ax = plt.subplots(figsize=(12, 6))
                sns.violinplot(data=df_losses, x='Loss Type', y='Value', hue='Method',
                             ax=ax, palette='Set2', split=False, inner='box')
                ax.set_xlabel('Loss Type', fontsize=12, fontweight='bold')
                ax.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
                ax.set_title('Loss Distribution by Method and Type', fontsize=14, fontweight='bold', pad=20)
                ax.legend(title='Method', title_fontsize=11, fontsize=10)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

                plt.tight_layout()
                plt.savefig(output_path / "loss_distribution.png", dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved: loss_distribution.png (seaborn violin plot)")

        # Plot 6: Method Performance Comparison (Seaborn Boxplot)
        if use_seaborn and not df_plot.empty:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Final Reward by Method
            sns.boxplot(data=df_plot, x='Method', y='Final Reward', ax=axes[0, 0], palette='Set2')
            axes[0, 0].set_title('Final Reward by Method', fontsize=12, fontweight='bold')
            axes[0, 0].set_ylabel('Final Reward', fontsize=10, fontweight='bold')

            # Episode Length by Method
            if 'Episode Length' in df_plot.columns:
                sns.boxplot(data=df_plot, x='Method', y='Episode Length', ax=axes[0, 1], palette='Set2')
                axes[0, 1].set_title('Episode Length by Method', fontsize=12, fontweight='bold')
                axes[0, 1].set_ylabel('Episode Length', fontsize=10, fontweight='bold')

            # Value Loss by Method
            if 'Value Loss' in df_plot.columns:
                sns.boxplot(data=df_plot, x='Method', y='Value Loss', ax=axes[1, 0], palette='Set2')
                axes[1, 0].set_title('Value Loss by Method', fontsize=12, fontweight='bold')
                axes[1, 0].set_ylabel('Value Loss', fontsize=10, fontweight='bold')

            # Entropy Loss by Method
            if 'Entropy Loss' in df_plot.columns:
                sns.boxplot(data=df_plot, x='Method', y='Entropy Loss', ax=axes[1, 1], palette='Set2')
                axes[1, 1].set_title('Entropy Loss by Method', fontsize=12, fontweight='bold')
                axes[1, 1].set_ylabel('Entropy Loss (abs)', fontsize=10, fontweight='bold')

            plt.suptitle('Method Performance Comparison', fontsize=16, fontweight='bold', y=1.00)
            plt.tight_layout()
            plt.savefig(output_path / "method_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: method_comparison.png (seaborn boxplot)")

        print(f"\n[PLOTS] All plots saved to: {output_path.absolute()}/\n")

        return True

    except ImportError:
        print("\n[PLOTS] matplotlib not available, skipping plot generation")
        print("        Install with: pip install matplotlib")
        return False
    except Exception as e:
        print(f"\n[PLOTS] Error generating plots: {e}")
        return False


def main():
    """Compare all training results"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Compare RL training results")
    parser.add_argument("--save_plots", action="store_true", help="Save plots to files")
    parser.add_argument("--interactive", action="store_true", help="Use Plotly for interactive plots (requires --save_plots)")
    args = parser.parse_args()

    print("=" * 80)
    print("TRAINING RESULTS COMPARISON")
    print("=" * 80)
    print()

    # Find all metadata files
    model_dir = Path("models/object_selection")
    metadata_files = list(model_dir.glob("*_metadata.json"))

    if not metadata_files:
        print("No training results found!")
        print(f"Checked directory: {model_dir.absolute()}")
        return

    print(f"Found {len(metadata_files)} trained models\n")

    # Collect results
    all_data = []
    results_table = []

    for metadata_path in metadata_files:
        metadata = load_metadata(metadata_path)

        # Extract run name from metadata filename
        run_name = metadata_path.stem.replace("_metadata", "")

        # Try to find corresponding TensorBoard logs
        log_dir = Path("logs/object_selection") / run_name
        log_data = None
        if log_dir.exists():
            log_data = parse_tensorboard_logs(str(log_dir))
        else:
            log_data = parse_tensorboard_logs("")  # Returns empty dict

        # Store full data for plotting
        all_data.append({
            'model': run_name,
            'method': metadata.get("method", "unknown"),
            'grid': f"{metadata.get('training_grid_size', '?')}x{metadata.get('training_grid_size', '?')}",
            'cubes': metadata.get("num_cubes", "?"),
            'timesteps': metadata.get("total_timesteps", "?"),
            'final_reward': log_data['final_reward'],
            'final_length': log_data['final_length'],
            'entropy_loss': log_data['entropy_loss'],
            'value_loss': log_data['value_loss'],
            'policy_loss': log_data['policy_loss'],
            'rewards_curve': log_data['rewards_curve'],
            'lengths_curve': log_data['lengths_curve'],
            'entropy_curve': log_data['entropy_curve'],
            'value_curve': log_data['value_curve'],
            'policy_curve': log_data['policy_curve'],
            'lr_curve': log_data['lr_curve'],
            'policy': metadata.get("policy", "?"),
            'timestamp': metadata.get("timestamp", "?")
        })

        # Store table data
        results_table.append({
            "Model": run_name,
            "Method": metadata.get("method", "unknown"),
            "Grid": f"{metadata.get('training_grid_size', '?')}x{metadata.get('training_grid_size', '?')}",
            "Cubes": metadata.get("num_cubes", "?"),
            "Timesteps": metadata.get("total_timesteps", "?"),
            "Final Reward": f"{log_data['final_reward']:.2f}" if log_data['final_reward'] is not None else "N/A",
            "Final Ep Length": f"{log_data['final_length']:.2f}" if log_data['final_length'] is not None else "N/A",
            "Entropy Loss": f"{log_data['entropy_loss']:.2f}" if log_data['entropy_loss'] is not None else "N/A",
            "Value Loss": f"{log_data['value_loss']:.3f}" if log_data['value_loss'] is not None else "N/A",
            "Policy Loss": f"{log_data['policy_loss']:.4f}" if log_data['policy_loss'] is not None else "N/A",
            "Policy": metadata.get("policy", "?"),
            "Timestamp": metadata.get("timestamp", "?")
        })

    # Create DataFrame and sort by method, grid, cubes
    import pandas as pd
    df = pd.DataFrame(results_table)
    df = df.sort_values(by=["Method", "Grid", "Cubes"])

    # Display results table
    print(df.to_string(index=False))
    print()

    # ASCII Bar Charts for Final Metrics
    print("\n" + "=" * 80)
    print("VISUAL COMPARISON")
    print("=" * 80)

    # Final Rewards by Configuration
    reward_data = [(f"{d['method']:10s} {d['grid']} {d['cubes']}c", d['final_reward'])
                   for d in all_data if d['final_reward'] is not None]
    if reward_data:
        print_ascii_bar_chart(reward_data, "Final Rewards by Configuration")

    # Entropy Loss Comparison
    entropy_data = [(f"{d['method']:10s} {d['grid']} {d['cubes']}c", abs(d['entropy_loss']))
                    for d in all_data if d['entropy_loss'] is not None]
    if entropy_data:
        print_ascii_bar_chart(entropy_data, "Entropy Loss (absolute value)")

    # Value Loss Comparison
    value_data = [(f"{d['method']:10s} {d['grid']} {d['cubes']}c", d['value_loss'])
                  for d in all_data if d['value_loss'] is not None]
    if value_data:
        print_ascii_bar_chart(value_data, "Value Loss")

    # Policy Loss Comparison
    policy_data = [(f"{d['method']:10s} {d['grid']} {d['cubes']}c", abs(d['policy_loss']))
                   for d in all_data if d['policy_loss'] is not None]
    if policy_data:
        print_ascii_bar_chart(policy_data, "Policy Loss (absolute value)")

    # ASCII Training Curves
    # Group by configuration for cleaner plots
    configs = {}
    for d in all_data:
        config_key = f"{d['grid']}_{d['cubes']}cubes"
        if config_key not in configs:
            configs[config_key] = {}
        configs[config_key][d['method']] = d['rewards_curve']

    for config_name, curves in configs.items():
        plot_training_curves_ascii(curves,
                                   f"Reward Curves: {config_name.replace('_', ' ')}",
                                   "Reward")

    # Find best model for each configuration
    print("\n" + "=" * 80)
    print("BEST MODELS BY CONFIGURATION")
    print("=" * 80)
    print()

    for (grid, cubes), group in df.groupby(["Grid", "Cubes"]):
        print(f"\nConfiguration: {grid} grid, {cubes} cubes")
        print("-" * 60)

        # Convert Final Reward to numeric for comparison
        group_copy = group.copy()
        group_copy["Reward_Numeric"] = pd.to_numeric(group_copy["Final Reward"], errors='coerce')

        if not group_copy["Reward_Numeric"].isna().all():
            best_row = group_copy.loc[group_copy["Reward_Numeric"].idxmax()]
            print(f"  Best Method: {best_row['Method']}")
            print(f"  Final Reward: {best_row['Final Reward']}")
            print(f"  Entropy Loss: {best_row['Entropy Loss']}")
            print(f"  Value Loss: {best_row['Value Loss']}")
            print(f"  Policy Loss: {best_row['Policy Loss']}")
            print(f"  Model: {best_row['Model']}")

    # Save plots if requested
    if args.save_plots:
        if args.interactive:
            # Use Plotly for interactive plots
            plotly_success = save_plots_plotly(all_data)
            if not plotly_success:
                print("\n[INFO] Falling back to matplotlib/seaborn plots...")
                save_plots_matplotlib(all_data)
        else:
            # Use matplotlib/seaborn for static plots
            save_plots_matplotlib(all_data)

    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    print("To test a model in Isaac Sim:")
    print("  C:\\isaacsim\\python.bat franka_rrt_physXLidar_depth_camera_rl_standalone_v1.9.py \\")
    print("    --rl_model \"models/object_selection/<MODEL_NAME>_final.zip\" \\")
    print("    --num_cubes <NUM_CUBES> --training_grid_size <GRID_SIZE>")
    print()
    print("To save static plots (matplotlib/seaborn):")
    print("  py -3.11 compare_training_results.py --save_plots")
    print()
    print("To save interactive plots (plotly):")
    print("  py -3.11 compare_training_results.py --save_plots --interactive")
    print()

if __name__ == "__main__":
    main()

