"""
Train Double DQN Agent for Object Selection with RRT Viz (PythonRobotics)
Uses Double DQN algorithm with PythonRobotics RRT for visualization.

Usage:
    py -3.11 train_rrt_viz_ddqn.py --timesteps 50000 --grid_size 4 --num_cubes 9
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import json

# Add project root to path
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
    parser.add_argument("--timesteps", type=int, default=50000,
                       help="Total training timesteps (default: 50000)")
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
                       help="Epsilon decay rate (default: 0.995)")
    
    # Environment parameters
    parser.add_argument("--grid_size", type=int, default=4,
                       help="Grid size (default: 4)")
    parser.add_argument("--num_cubes", type=int, default=9,
                       help="Number of cubes (default: 9)")
    parser.add_argument("--max_steps", type=int, default=50,
                       help="Max steps per episode (default: 50)")
    
    # Saving
    parser.add_argument("--save_freq", type=int, default=5000,
                       help="Save checkpoint every N steps (default: 5000)")
    parser.add_argument("--model_dir", type=str,
                       default=r"C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\models",
                       help="Directory to save models")
    parser.add_argument("--log_dir", type=str,
                       default=r"C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script\logs",
                       help="Directory for logs")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Print configuration
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
        batch_size=args.batch_size,
        buffer_capacity=100000,
        target_update_freq=1000
    )
    
    # Training loop
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ddqn_rrt_viz_grid{args.grid_size}_cubes{args.num_cubes}_{timestamp}"

    # Create log file
    log_file = os.path.join(args.log_dir, f"{run_name}_training.csv")
    with open(log_file, 'w') as f:
        f.write("step,episode,loss,reward,epsilon\n")

    print("\nStarting training...")
    print(f"Checkpoints will be saved to: {args.model_dir}/{run_name}")
    print(f"Training log will be saved to: {log_file}")
    print("=" * 60 + "\n")

    episode_rewards = []
    episode_lengths = []
    total_steps = 0
    episode = 0
    
    while total_steps < args.timesteps:
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and total_steps < args.timesteps:
            # Get action mask
            action_mask = info.get('action_mask', env.action_masks())
            
            # Select action
            action = agent.select_action(state, action_mask)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Get next action mask
            next_action_mask = info.get('action_mask', env.action_masks())

            # Store transition
            agent.store_transition(state, action, reward, next_state, done, action_mask, next_action_mask)

            # Train
            loss = agent.train_step()

            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1

            # Log training data
            with open(log_file, 'a') as f:
                loss_val = f"{loss:.6f}" if loss is not None else ""
                f.write(f"{total_steps},{episode},{loss_val},{reward:.6f},{agent.epsilon:.6f}\n")

            # Print progress
            if total_steps % 1000 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                loss_str = f"{loss:.4f}" if loss else "0.0000"
                print(f"Steps: {total_steps}/{args.timesteps} | "
                      f"Episode: {episode} | "
                      f"Avg Reward (100 ep): {avg_reward:.2f} | "
                      f"Epsilon: {agent.epsilon:.4f} | "
                      f"Loss: {loss_str}")

            # Save checkpoint
            if total_steps % args.save_freq == 0:
                checkpoint_path = os.path.join(args.model_dir, f"{run_name}_step_{total_steps}.pt")
                agent.save(checkpoint_path)

        # Episode finished
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        agent.episodes += 1
        episode += 1

    # Save final model
    final_path = os.path.join(args.model_dir, f"{run_name}_final.pt")
    agent.save(final_path)

    # Save metadata
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
        "final_epsilon": agent.epsilon,
        "total_episodes": episode
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
    print(f"Average reward (last 100 ep): {np.mean(episode_rewards[-100:]):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

