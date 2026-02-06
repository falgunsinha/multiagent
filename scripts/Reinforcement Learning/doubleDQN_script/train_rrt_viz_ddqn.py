import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import json
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from src.rl.object_selection_env_rrt_viz import ObjectSelectionEnvRRTViz
from src.rl.doubleDQN import DoubleDQNAgent


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Double DQN agent with RRT Viz")
    
    # Training parameters
    parser.add_argument("--timesteps", type=int, default=None,
                       help="Total training timesteps (auto-set based on grid/cubes if not specified)")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate (default: 1e-3)")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size (default: 64)")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor (default: 0.99)")
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                       help="Initial epsilon (default: 1.0)")
    parser.add_argument("--epsilon_end", type=float, default=0.01,
                       help="Final epsilon (default: 0.01)")
    parser.add_argument("--epsilon_decay", type=float, default=0.995,
                       help="Epsilon decay rate (for multiplicative decay)")
    parser.add_argument("--epsilon_decay_type", type=str, default='exponential',
                       choices=['multiplicative', 'exponential'],
                       help="Type of epsilon decay (default: exponential)")
    parser.add_argument("--epsilon_decay_rate", type=int, default=2500,
                       help="Decay rate for exponential epsilon decay (default: 2500)")
    parser.add_argument("--target_update_tau", type=float, default=0.005,
                       help="Tau for soft target update (0 = hard update) (default: 0.005)")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                       help="Warmup steps before training (default: 1000)")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    
    # Environment parameters
    parser.add_argument("--grid_size", type=int, default=4,
                       help="Grid size (default: 4)")
    parser.add_argument("--num_cubes", type=int, default=9,
                       help="Number of cubes (default: 9)")
    parser.add_argument("--max_steps", type=int, default=50,
                       help="Max steps per episode (default: 50)")
    
 
    parser.add_argument("--save_freq", type=int, default=5000,
                       help="Save checkpoint every N steps (default: 5000)")
    parser.add_argument("--model_dir", type=str,
                       default=r"C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\models",
                       help="Directory to save models")
    parser.add_argument("--log_dir", type=str,
                       default=r"C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\logs",
                       help="Directory for logs")

    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from (e.g., models/checkpoint_step_25000.pt)")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    
    return parser.parse_args()


def main():
    args = parse_args()

    # Auto-set timesteps based on grid size and num cubes if not specified
    if args.timesteps is None:
        if args.grid_size == 3 and args.num_cubes == 4:
            args.timesteps = 5000
        elif args.grid_size == 4 and args.num_cubes == 6:
            args.timesteps = 7000
        elif args.grid_size == 4 and args.num_cubes == 9:
            args.timesteps = 10000
        else:
            args.timesteps = 10000  # Default
        print(f"Auto-set timesteps to {args.timesteps} based on grid_size={args.grid_size}, num_cubes={args.num_cubes}")


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project="ddqn-object-selection",
                name=f"rrt_viz_grid{args.grid_size}_cubes{args.num_cubes}",
                config={
                    "method": "rrt_viz",
                    "grid_size": args.grid_size,
                    "num_cubes": args.num_cubes,
                    "timesteps": args.timesteps,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "gamma": args.gamma,
                    "epsilon_start": args.epsilon_start,
                    "epsilon_end": args.epsilon_end,
                    "epsilon_decay": args.epsilon_decay,
                    "epsilon_decay_type": args.epsilon_decay_type,
                    "epsilon_decay_rate": args.epsilon_decay_rate,
                    "target_update_tau": args.target_update_tau,
                    "warmup_steps": args.warmup_steps,
                }
            )

            
            try:
                from wandb_chart_config import setup_wandb_charts
                setup_wandb_charts()
            except ImportError:
                # Fallback: define basic metrics if config file not found
                wandb.define_metric("training/loss", step_metric="global_step")
                wandb.define_metric("training/epsilon", step_metric="global_step")
                wandb.define_metric("episode/avg_reward_100", step_metric="global_step")

            print("W&B logging enabled")
        except ImportError:
            print("ERROR: wandb not installed. Install with: py -3.11 -m pip install wandb")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: W&B initialization failed: {e}")
            print("Please check your W&B login: py -3.11 -m wandb login")
            sys.exit(1)

    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    

    print("=" * 60)
    print("DOUBLE DQN TRAINING - RRT VIZ (PythonRobotics)")
    print("=" * 60)
    print(f"Method: RRT Viz (PythonRobotics RRT)")
    print(f"Timesteps: {args.timesteps}")
    print(f"Grid: {args.grid_size}x{args.grid_size}")
    print(f"Cubes: {args.num_cubes}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gamma: {args.gamma}")
    print(f"Epsilon: {args.epsilon_start} -> {args.epsilon_end} (decay: {args.epsilon_decay})")
    print("=" * 60)
    
    # Create environment
    max_objects = args.grid_size * args.grid_size
    env = ObjectSelectionEnvRRTViz(
        franka_controller=None,
        max_objects=max_objects,
        max_steps=args.max_steps,
        num_cubes=args.num_cubes,
        training_grid_size=args.grid_size,
        render_mode=None
    )
    
    # Create agent
    state_dim = max_objects * 6  # 6 features per object
    action_dim = max_objects

    agent = DoubleDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        epsilon_decay_type=args.epsilon_decay_type,
        epsilon_decay_rate=args.epsilon_decay_rate,
        batch_size=args.batch_size,
        buffer_capacity=100000,
        target_update_freq=1000,
        target_update_tau=args.target_update_tau,
        warmup_steps=args.warmup_steps
    )

    # Resume from checkpoint if specified
    total_steps = 0
    episode = 0
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"ERROR: Checkpoint not found: {args.resume}")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"RESUMING FROM CHECKPOINT")
        print(f"{'='*60}")
        print(f"Loading: {args.resume}")
        agent.load(args.resume)
        total_steps = agent.steps
        episode = agent.episodes
        print(f"Resuming from step {total_steps}, episode {episode}")
        print(f"Current epsilon: {agent.epsilon:.4f}")
        print(f"{'='*60}\n")

        # Extract run_name from checkpoint path
        checkpoint_name = os.path.basename(args.resume)
        # Format: ddqn_rrt_viz_grid4_cubes9_20251219_115447_step_25000.pt
        run_name = '_'.join(checkpoint_name.split('_')[:-2])  # Remove _step_XXXXX.pt
    else:
        # Training loop
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"ddqn_rrt_viz_grid{args.grid_size}_cubes{args.num_cubes}_{timestamp}"

    # Create enhanced log files
    log_file = os.path.join(args.log_dir, f"{run_name}_training.csv")
    episode_log_file = os.path.join(args.log_dir, f"{run_name}_episodes.csv")

    # Create or append to log files
    if args.resume:
        # Append mode - files should already exist
        if not os.path.exists(log_file):
            print(f"WARNING: Training log not found, creating new: {log_file}")
            with open(log_file, 'w') as f:
                f.write("step,episode,loss,step_reward,epsilon,q_value,episode_reward,episode_length,avg_reward_100,success_rate\n")
        if not os.path.exists(episode_log_file):
            print(f"WARNING: Episode log not found, creating new: {episode_log_file}")
            with open(episode_log_file, 'w') as f:
                f.write("episode,total_reward,length,success,avg_reward_100,success_rate_100\n")
    else:
        # Create new log files
        with open(log_file, 'w') as f:
            f.write("step,episode,loss,step_reward,epsilon,q_value,episode_reward,episode_length,avg_reward_100,success_rate\n")

        with open(episode_log_file, 'w') as f:
            f.write("episode,total_reward,length,success,avg_reward_100,success_rate_100\n")

    print("\nStarting training...")
    print(f"Checkpoints will be saved to: {args.model_dir}/{run_name}")
    print(f"Training log will be saved to: {log_file}")
    print(f"Episode log will be saved to: {episode_log_file}")
    print("=" * 60 + "\n")

    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    
    while total_steps < args.timesteps:
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and total_steps < args.timesteps:
            action_mask = info.get('action_mask', env.action_masks())
            
           
            action = agent.select_action(state, action_mask)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_action_mask = info.get('action_mask', env.action_masks())
            agent.store_transition(state, action, reward, next_state, done, action_mask, next_action_mask)
            loss = agent.train_step()
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            avg_reward_100 = np.mean(episode_rewards[-100:]) if episode_rewards else 0.0
            success_rate = np.mean(episode_successes[-100:]) if episode_successes else 0.0

            # Enhanced CSV logging (10 metrics)
            with open(log_file, 'a') as f:
                loss_val = f"{loss:.6f}" if loss is not None else ""
                q_val = f"{agent.last_q_value:.6f}"
                f.write(f"{total_steps},{episode},{loss_val},{reward:.6f},{agent.epsilon:.6f},"
                       f"{q_val},{episode_reward:.6f},{episode_length},{avg_reward_100:.6f},{success_rate:.6f}\n")

            # W&B logging (per-step metrics)
            if args.use_wandb:
                import wandb
                wandb.log({
                    # Step counter (required for custom step metric)
                    "global_step": total_steps,
                    "training/loss": loss if loss is not None else 0.0,
                    "train/loss_raw": loss if loss is not None else 0.0,
                    "training/epsilon": agent.epsilon,
                    "training/q_value": agent.last_q_value,
                    "training/step_reward": reward,
                    "training/episode_reward_running": episode_reward,
                    "training/episode_length_running": episode_length,
                    "train/q_mean": agent.q_mean,
                    "train/q_max": agent.q_max,
                    "train/q_std": agent.q_std,
                    "ddqn/q_policy": agent.q_max,
                    "ddqn/q_target": agent.value_estimate,
                    "ddqn/q_overestimation": agent.q_overestimation,
                    "ddqn/value_estimate": agent.value_estimate,

                    # TD error
                    "train/td_error": agent.td_error
                })

            # Print progress
            if total_steps % 1000 == 0:
                loss_str = f"{loss:.4f}" if loss else "0.0000"
                print(f"Steps: {total_steps}/{args.timesteps} | "
                      f"Episode: {episode} | "
                      f"Avg Reward (100 ep): {avg_reward_100:.2f} | "
                      f"Epsilon: {agent.epsilon:.4f} | "
                      f"Loss: {loss_str}")
            if total_steps % args.save_freq == 0:
                checkpoint_path = os.path.join(args.model_dir, f"{run_name}_step_{total_steps}.pt")
                agent.save(checkpoint_path)

        episode_success = 1.0 if len(env.objects_picked) == env.num_cubes else 0.0
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_successes.append(episode_success)
        avg_reward_100 = np.mean(episode_rewards[-100:])
        success_rate_100 = np.mean(episode_successes[-100:])

        # Log episode summary
        with open(episode_log_file, 'a') as f:
            f.write(f"{episode},{episode_reward:.6f},{episode_length},{int(episode_success)},"
                   f"{avg_reward_100:.6f},{success_rate_100:.6f}\n")

        # W&B logging (per-episode metrics)
        if args.use_wandb:
            import wandb
            wandb.log({
                "global_step": total_steps,
                "episode/total_reward": episode_reward,
                "episode/total_length": episode_length,
                "episode/success": episode_success,
                "episode/avg_reward_100": avg_reward_100,
                "episode/success_rate_100": success_rate_100
            })

        agent.episodes += 1
        episode += 1

    final_path = os.path.join(args.model_dir, f"{run_name}_final.pt")
    agent.save(final_path)
    metadata = {
        "method": "rrt_viz",
        "algorithm": "double_dqn",
        "training_grid_size": args.grid_size,
        "num_cubes": args.num_cubes,
        "max_objects": max_objects,
        "max_steps": args.max_steps,
        "timestamp": timestamp,
        "total_timesteps": args.timesteps,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay": args.epsilon_decay,
        "epsilon_decay_type": args.epsilon_decay_type,
        "epsilon_decay_rate": args.epsilon_decay_rate,
        "target_update_tau": args.target_update_tau,
        "warmup_steps": args.warmup_steps,
        "final_epsilon": agent.epsilon,
        "total_episodes": episode,
        "avg_reward_100": np.mean(episode_rewards[-100:]) if episode_rewards else 0.0,
        "success_rate_100": np.mean(episode_successes[-100:]) if episode_successes else 0.0
    }
    metadata_path = os.path.join(args.model_dir, f"{run_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total episodes: {episode}")
    print(f"Final model saved to: {final_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Average reward (last 100 ep): {metadata['avg_reward_100']:.2f}")
    print(f"Success rate (last 100 ep): {metadata['success_rate_100']:.2%}")
    print("=" * 60)

    # Finish W&B run
    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()

