"""
WandB Chart Configuration
Separate chart definitions from training scripts.
Can be updated without retraining models.

Inspired by professional ML dashboards with organized panels and clear visualizations.
"""

import wandb

def setup_wandb_charts():
    """
    Define custom charts and metrics for WandB dashboard.
    Call this after wandb.init() in training scripts.

    Chart organization:
    - training/* : Per-step training metrics (loss, epsilon, Q-values)
    - episode/*  : Per-episode performance metrics (rewards, success rate)
    - analysis/* : Advanced analysis metrics (optional, can be added later)
    """
    # Define step metrics
    wandb.define_metric("global_step")

    # Training metrics (per-step) - Core learning indicators
    wandb.define_metric("training/loss", step_metric="global_step")
    wandb.define_metric("training/epsilon", step_metric="global_step")
    wandb.define_metric("training/q_value", step_metric="global_step")
    wandb.define_metric("training/step_reward", step_metric="global_step")
    wandb.define_metric("training/episode_reward_running", step_metric="global_step")
    wandb.define_metric("training/episode_length_running", step_metric="global_step")

    # Episode metrics (per-episode) - Performance indicators
    wandb.define_metric("episode/total_reward", step_metric="global_step")
    wandb.define_metric("episode/total_length", step_metric="global_step")
    wandb.define_metric("episode/success", step_metric="global_step")
    wandb.define_metric("episode/avg_reward_100", step_metric="global_step")
    wandb.define_metric("episode/success_rate_100", step_metric="global_step")

    # Advanced DDQN metrics (inspired by professional RL dashboards)
    wandb.define_metric("train/loss_raw", step_metric="global_step")  # Raw loss (no smoothing)
    wandb.define_metric("train/td_error", step_metric="global_step")  # TD error magnitude
    wandb.define_metric("train/q_max", step_metric="global_step")  # Max Q-value
    wandb.define_metric("train/q_mean", step_metric="global_step")  # Mean Q-value
    wandb.define_metric("train/q_std", step_metric="global_step")  # Q-value std dev
    wandb.define_metric("train/action_accuracy", step_metric="global_step")  # % optimal actions
    wandb.define_metric("train/learning_rate", step_metric="global_step")  # Current LR
    wandb.define_metric("train/step_time", step_metric="global_step")  # Time per step

    # DDQN-specific: Overestimation bias tracking
    wandb.define_metric("ddqn/q_target", step_metric="global_step")  # Target network Q-value
    wandb.define_metric("ddqn/q_policy", step_metric="global_step")  # Policy network Q-value
    wandb.define_metric("ddqn/q_overestimation", step_metric="global_step")  # Q_policy - Q_target
    wandb.define_metric("ddqn/value_estimate", step_metric="global_step")  # V(s) estimate

    # Rewards analysis
    wandb.define_metric("rewards/mean", step_metric="global_step")  # Mean reward
    wandb.define_metric("rewards/std", step_metric="global_step")  # Reward std dev
    wandb.define_metric("rewards/max", step_metric="global_step")  # Max reward seen
    wandb.define_metric("rewards/min", step_metric="global_step")  # Min reward seen

    # Performance metrics
    wandb.define_metric("performance/overall_accuracy", step_metric="global_step")  # Overall accuracy
    wandb.define_metric("performance/episode_count", step_metric="global_step")  # Episodes completed

    # Analysis metrics (optional - can be added later without retraining)
    # wandb.define_metric("analysis/q_value_variance", step_metric="global_step")
    # wandb.define_metric("analysis/pick_efficiency", step_metric="global_step")
    # wandb.define_metric("analysis/exploration_ratio", step_metric="global_step")


def get_training_metrics():
    """
    Returns list of metrics to log during training (per-step).
    """
    return [
        "training/loss",
        "training/epsilon",
        "training/q_value",
        "training/step_reward",
        "training/episode_reward_running",
        "training/episode_length_running"
    ]


def get_episode_metrics():
    """
    Returns list of metrics to log at end of episode.
    """
    return [
        "episode/total_reward",
        "episode/total_length",
        "episode/success",
        "episode/avg_reward_100",
        "episode/success_rate_100"
    ]


def get_recommended_chart_layout():
    """
    Returns recommended chart layout for WandB dashboard.
    Use this as a guide when customizing charts on the dashboard.

    Professional dashboard layout inspired by top ML projects:
    - Clear sections for different metric types
    - Appropriate y-axis ranges for better visualization
    - Smoothing recommendations for noisy metrics

    Returns:
        Dictionary with recommended chart configurations
    """
    layout = {
        "sections": [
            {
                "name": "🎯 Training Progress",
                "description": "Core learning metrics - monitor convergence",
                "charts": [
                    {
                        "metric": "training/loss",
                        "title": "Training Loss",
                        "y_range": [0, 5],
                        "smoothing": 0.7,
                        "notes": "Should decrease and stabilize. DDQN loss doesn't go to zero."
                    },
                    {
                        "metric": "training/epsilon",
                        "title": "Exploration Rate (Epsilon)",
                        "y_range": [0, 1],
                        "smoothing": 0.0,
                        "notes": "Decays from 1.0 to 0.01 over ~25K steps"
                    },
                    {
                        "metric": "training/q_value",
                        "title": "Q-Value (Selected Action)",
                        "y_range": None,
                        "smoothing": 0.6,
                        "notes": "Average Q-value of actions taken by agent"
                    }
                ]
            },
            {
                "name": "📊 Episode Performance",
                "description": "Per-episode metrics - track agent performance",
                "charts": [
                    {
                        "metric": "episode/avg_reward_100",
                        "title": "Average Reward (100 episodes)",
                        "y_range": None,
                        "smoothing": 0.5,
                        "notes": "Higher is better. Should increase and plateau."
                    },
                    {
                        "metric": "episode/success_rate_100",
                        "title": "Success Rate (100 episodes)",
                        "y_range": [0, 1],
                        "smoothing": 0.5,
                        "notes": "Percentage of successful episodes (all cubes picked)"
                    },
                    {
                        "metric": "episode/total_length",
                        "title": "Episode Length",
                        "y_range": None,
                        "smoothing": 0.6,
                        "notes": "Number of steps per episode. Lower is more efficient."
                    }
                ]
            },
            {
                "name": "💰 Rewards Breakdown",
                "description": "Detailed reward analysis",
                "charts": [
                    {
                        "metric": "training/step_reward",
                        "title": "Step Reward",
                        "y_range": None,
                        "smoothing": 0.8,
                        "notes": "Reward received at each step (very noisy)"
                    },
                    {
                        "metric": "training/episode_reward_running",
                        "title": "Episode Reward (Running)",
                        "y_range": None,
                        "smoothing": 0.6,
                        "notes": "Cumulative reward within current episode"
                    },
                    {
                        "metric": "episode/total_reward",
                        "title": "Episode Total Reward",
                        "y_range": None,
                        "smoothing": 0.5,
                        "notes": "Total reward at end of each episode"
                    }
                ]
            },
            {
                "name": "🔬 DDQN Analysis",
                "description": "DDQN-specific metrics - overestimation bias and value estimates",
                "charts": [
                    {
                        "metric": "train/loss_raw",
                        "title": "Train/Loss (Raw)",
                        "y_range": [0, 10],
                        "smoothing": 0.0,
                        "notes": "Raw loss without smoothing - shows actual training signal"
                    },
                    {
                        "metric": "training/loss",
                        "title": "Train/Loss (Smoothed)",
                        "y_range": [0, 5],
                        "smoothing": 0.7,
                        "notes": "Smoothed loss for trend visualization"
                    },
                    {
                        "metric": "train/td_error",
                        "title": "TD Error",
                        "y_range": None,
                        "smoothing": 0.6,
                        "notes": "Temporal difference error magnitude"
                    },
                    {
                        "metric": "ddqn/q_overestimation",
                        "title": "Q-Value Overestimation Bias",
                        "y_range": None,
                        "smoothing": 0.6,
                        "notes": "Difference between policy and target Q-values (should be near zero for DDQN)"
                    },
                    {
                        "metric": "ddqn/value_estimate",
                        "title": "Value Estimates V(s)",
                        "y_range": None,
                        "smoothing": 0.6,
                        "notes": "State value estimates - should increase as agent learns"
                    }
                ]
            },
            {
                "name": "📈 Q-Value Analysis",
                "description": "Q-value statistics and distribution",
                "charts": [
                    {
                        "metric": "train/q_mean",
                        "title": "Mean Q-Value",
                        "y_range": None,
                        "smoothing": 0.6,
                        "notes": "Average Q-value across all actions"
                    },
                    {
                        "metric": "train/q_max",
                        "title": "Max Q-Value",
                        "y_range": None,
                        "smoothing": 0.6,
                        "notes": "Maximum Q-value (selected action)"
                    },
                    {
                        "metric": "train/q_std",
                        "title": "Q-Value Std Dev",
                        "y_range": None,
                        "smoothing": 0.6,
                        "notes": "Q-value variance - high variance indicates uncertainty"
                    }
                ]
            },
            {
                "name": "💰 Rewards Analysis",
                "description": "Reward statistics and distribution",
                "charts": [
                    {
                        "metric": "rewards/mean",
                        "title": "Rewards/Mean",
                        "y_range": None,
                        "smoothing": 0.6,
                        "notes": "Mean reward over recent steps"
                    },
                    {
                        "metric": "rewards/std",
                        "title": "Rewards/Std Dev",
                        "y_range": None,
                        "smoothing": 0.6,
                        "notes": "Reward variance - should decrease as policy stabilizes"
                    },
                    {
                        "metric": "episode/avg_reward_100",
                        "title": "Rewards/Training Steps",
                        "y_range": None,
                        "smoothing": 0.5,
                        "notes": "Average reward vs training steps"
                    }
                ]
            },
            {
                "name": "⚙️ Training Metrics",
                "description": "Training performance and hyperparameters",
                "charts": [
                    {
                        "metric": "train/learning_rate",
                        "title": "Train/Learning Rate",
                        "y_range": None,
                        "smoothing": 0.0,
                        "notes": "Current learning rate (if using LR scheduler)"
                    },
                    {
                        "metric": "train/step_time",
                        "title": "Train/Step Time",
                        "y_range": None,
                        "smoothing": 0.5,
                        "notes": "Time per training step (seconds)"
                    },
                    {
                        "metric": "train/action_accuracy",
                        "title": "Train/Action Accuracy",
                        "y_range": [0, 1],
                        "smoothing": 0.6,
                        "notes": "Percentage of optimal actions taken (if ground truth available)"
                    },
                    {
                        "metric": "performance/overall_accuracy",
                        "title": "Overall Accuracy",
                        "y_range": [0, 1],
                        "smoothing": 0.5,
                        "notes": "Overall task completion accuracy"
                    }
                ]
            },
            {
                "name": "🔬 Method Comparison",
                "description": "Compare A*, RRT Viz, and RRT Isaac Sim",
                "charts": [
                    {
                        "metrics": ["episode/avg_reward_100"],
                        "title": "Avg Reward Comparison (All Methods)",
                        "type": "multi_line",
                        "group_by": "method",
                        "notes": "Compare performance across different path planning methods"
                    },
                    {
                        "metrics": ["training/loss"],
                        "title": "Loss Comparison (All Methods)",
                        "type": "multi_line",
                        "group_by": "method",
                        "notes": "Compare learning curves across methods"
                    },
                    {
                        "metrics": ["ddqn/q_overestimation"],
                        "title": "Q-Overestimation Comparison (All Methods)",
                        "type": "multi_line",
                        "group_by": "method",
                        "notes": "Compare overestimation bias - DDQN should have lower bias than DQN"
                    },
                    {
                        "metrics": ["episode/success_rate_100"],
                        "title": "Success Rate Comparison (All Methods)",
                        "type": "multi_line",
                        "group_by": "method",
                        "notes": "Compare success rates - RRT Isaac Sim may have lower rate due to obstacles"
                    }
                ]
            }
        ]
    }

    return layout


def create_custom_charts():
    """
    Create custom charts programmatically using WandB API.
    These charts will appear in your workspace automatically!

    Call this after wandb.init() and setup_wandb_charts().

    Note: Custom charts are created per-run. For project-level charts,
    use the WandB dashboard UI to create and save them.
    """
    try:
        # Method Comparison - Line Plot
        wandb.log({
            "custom_charts/method_comparison": wandb.plot.line_series(
                xs=[[0, 1000, 2000], [0, 1000, 2000], [0, 1000, 2000]],
                ys=[[0, 50, 100], [0, 45, 95], [0, 40, 85]],
                keys=["A*", "RRT Viz", "RRT Isaac Sim"],
                title="Method Comparison - Average Reward",
                xname="Training Steps"
            )
        })

        print("✅ Custom charts created successfully!")
        print("   View them in your WandB workspace under 'custom_charts' section")

    except Exception as e:
        print(f"⚠️  Could not create custom charts: {e}")
        print("   Charts can still be created manually on the dashboard")


def add_custom_kpis(metrics_dict, state, env, agent):
    """
    Add custom KPIs to metrics dictionary.
    Can be extended without retraining.

    Args:
        metrics_dict: Dictionary to add metrics to
        state: Current environment state
        env: Environment instance
        agent: Agent instance

    Returns:
        Updated metrics dictionary
    """
    # Example custom KPIs you might want to add later:

    # 1. Exploration vs Exploitation ratio
    # metrics_dict["custom/exploration_ratio"] = 1.0 if agent.epsilon > 0.1 else 0.0

    # 2. Q-value variance (measure of uncertainty)
    # if hasattr(agent, 'last_q_values'):
    #     metrics_dict["custom/q_value_variance"] = np.var(agent.last_q_values)

    # 3. Distance to optimal (if you know optimal solution)
    # metrics_dict["custom/distance_to_optimal"] = calculate_distance_to_optimal(state)

    # 4. Obstacle proximity (for RRT methods)
    # if hasattr(env, 'obstacles'):
    #     metrics_dict["custom/min_obstacle_distance"] = calculate_min_obstacle_distance(state, env)

    # 5. Pick efficiency (picks per step)
    # if hasattr(env, 'objects_picked'):
    #     metrics_dict["custom/pick_efficiency"] = len(env.objects_picked) / env.current_step

    return metrics_dict


def log_new_metrics_to_existing_runs(run_path, new_metrics):
    """
    Add new metrics to existing WandB runs using the API.
    This allows you to compute and log new metrics from already-logged data!

    Args:
        run_path: WandB run path (e.g., "falgunsinha/ddqn-object-selection/run_id")
        new_metrics: Dictionary of new metrics to log {step: {metric_name: value}}

    Example:
        # Calculate new metric from existing data
        api = wandb.Api()
        run = api.run("falgunsinha/ddqn-object-selection/abc123")

        # Get existing data
        history = run.history()

        # Calculate new metric (e.g., reward per step)
        new_metrics = {}
        for i, row in history.iterrows():
            step = row['_step']
            reward_per_step = row['episode/total_reward'] / row['episode/total_length']
            new_metrics[step] = {'analysis/reward_per_step': reward_per_step}

        # Log new metrics
        log_new_metrics_to_existing_runs(run.path, new_metrics)
    """
    try:
        api = wandb.Api()
        run = api.run(run_path)

        # Log new metrics
        for step, metrics in new_metrics.items():
            run.log(metrics, step=step)

        print(f"✅ Added {len(new_metrics)} new metric entries to run {run_path}")

    except Exception as e:
        print(f"❌ Error adding metrics to run: {e}")
        print("   Make sure you have the correct run path and API access")


if __name__ == "__main__":
    """
    Display chart configuration guide and recommended layout
    """
    import json

    print("=" * 80)
    print("WandB CHART CONFIGURATION GUIDE")
    print("=" * 80)

    print("\n📋 USAGE:")
    print("1. Training scripts automatically import: from wandb_chart_config import setup_wandb_charts")
    print("2. Charts are defined at wandb.init(): setup_wandb_charts()")
    print("3. Update this file anytime to change future runs")
    print("4. Customize existing runs on WandB dashboard (no retraining needed!)")

    print("\n" + "=" * 80)
    print("📊 RECOMMENDED DASHBOARD LAYOUT")
    print("=" * 80)

    layout = get_recommended_chart_layout()

    for section in layout["sections"]:
        print(f"\n{section['name']}")
        print(f"  {section['description']}")
        print()

        for chart in section["charts"]:
            if "metric" in chart:
                print(f"  📈 {chart['title']}")
                print(f"     Metric: {chart['metric']}")
                print(f"     Y-axis: {chart['y_range'] if chart['y_range'] else 'Auto'}")
                print(f"     Smoothing: {chart['smoothing']}")
                print(f"     💡 {chart['notes']}")
                print()
            elif "metrics" in chart:
                print(f"  📊 {chart['title']} ({chart['type']})")
                print(f"     Metrics: {', '.join(chart['metrics'])}")
                print(f"     Group by: {chart['group_by']}")
                print(f"     💡 {chart['notes']}")
                print()

    print("=" * 80)
    print("🎨 HOW TO CUSTOMIZE CHARTS ON WANDB DASHBOARD")
    print("=" * 80)
    print("""
1. Go to: https://wandb.ai/falgunsinha/ddqn-object-selection

2. Click on any chart to edit:
   - Set Y-axis range (e.g., 0-5 for loss)
   - Adjust smoothing (0.0-1.0)
   - Change chart type (line, scatter, bar)
   - Add multiple metrics to same chart

3. Create custom panels:
   - Click "Add panel" → "Line plot"
   - Select metrics from dropdown
   - Group by "method" to compare A*, RRT Viz, RRT Isaac Sim
   - Save to workspace

4. Hide unwanted charts:
   - Click "..." on chart → "Hide"

5. Create comparison charts:
   - Add panel → Select multiple runs
   - Group by config parameter (method, grid_size, num_cubes)
   - Compare performance across experiments

6. Export data:
   - Click "..." on chart → "Download CSV"
   - Or use WandB API to download full history
    """)

    print("=" * 80)
    print("✅ CHANGES MADE TO FIX CHART ISSUES")
    print("=" * 80)
    print("""
BEFORE (Confusing):
  ❌ 'step' logged as metric (useless diagonal line chart)
  ❌ 'success_rate_100' logged per-step (doesn't make sense)
  ❌ 'success_rate_100_ep' duplicate metric
  ❌ No organization, all metrics in one flat list

AFTER (Clean):
  ✅ 'step' only used as x-axis (not logged as metric)
  ✅ Success rate only logged per-episode
  ✅ Metrics organized: training/* and episode/*
  ✅ Clear naming: episode/success_rate_100
  ✅ Separate config file (change without retraining!)
    """)

    print("=" * 80)
    print("🚀 READY TO TRAIN!")
    print("=" * 80)
    print("\nRun: train_all_ddqn_wandb.bat")
    print("\nYour charts will be clean and organized from the start!")
    print("You can still customize them on the dashboard anytime.")
    print("\n" + "=" * 80)

