"""
Train MAPPO Two-Agent System with A* Visualization

Trains:
- Agent 1 (DDQN): Pre-trained, frozen
- Agent 2 (MAPPO): Learns reshuffling policy

Environment: A* path planning with visualization
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import torch

# Add project paths
project_root = Path(r"C:\isaacsim\cobotproject")
mappo_root = project_root / "scripts" / "Reinforcement Learning" / "MAPPO"
rl_root = project_root / "scripts" / "Reinforcement Learning"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(mappo_root))
sys.path.insert(0, str(rl_root))

# Import environment
from astar_rl_episode_viz import ObjectSelectionEnvAStar

# Import DDQN agent
from src.rl.doubleDQN import DoubleDQNAgent

# Import MAPPO components
from envs.two_agent_env import TwoAgentEnv
from algorithms.mappo_policy import MAPPOPolicy
from algorithms.mappo_trainer import MAPPO
from utils.replay_buffer import RolloutBuffer
from utils.wandb_config import WandBLogger
from utils.detailed_logger import DetailedLogger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train MAPPO two-agent system with A* visualization")
    
    # Environment
    parser.add_argument("--grid_size", type=int, default=4, help="Grid size")
    parser.add_argument("--num_cubes", type=int, default=9, help="Number of cubes")
    parser.add_argument("--max_steps", type=int, default=50, help="Max steps per episode")
    
    # Training
    parser.add_argument("--timesteps", type=int, default=20000, help="Total training timesteps")
    parser.add_argument("--buffer_size", type=int, default=256, help="Rollout buffer size (reduced from 2048 to prevent slowdown)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (reduced to match buffer size)")
    parser.add_argument("--ppo_epochs", type=int, default=10, help="PPO epochs per update")
    parser.add_argument("--num_mini_batch", type=int, default=4, help="Number of mini-batches")
    
    # MAPPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_param", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--value_loss_coef", type=float, default=1.0, help="Value loss coefficient")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm")
    
    # Reshuffling
    parser.add_argument("--max_reshuffles", type=int, default=5, help="Max reshuffles per episode")
    parser.add_argument("--reshuffle_reward_scale", type=float, default=1.0, help="Reshuffle reward scale")
    
    # DDQN agent
    parser.add_argument("--ddqn_model_path", type=str, 
                        default=str(project_root / "models" / "grid4_cubes9" / "final.pt"),
                        help="Path to pre-trained DDQN model")
    
    # Logging
    parser.add_argument("--wandb_project", type=str, default="ddqn-mappo-object-selection-reshuffling", help="WandB project name")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")
    parser.add_argument("--config_name", type=str, default=None, help="Configuration name for grouping runs")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval (episodes)")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save interval (episodes)")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    return parser.parse_args()


def main():
    """Main training loop"""
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create base environment (A* visualization)
    print("Creating A* environment...")
    max_objects = args.grid_size * args.grid_size
    base_env = ObjectSelectionEnvAStar(
        franka_controller=None,
        max_objects=max_objects,
        max_steps=args.max_steps,
        num_cubes=args.num_cubes,
        training_grid_size=args.grid_size,
        render_mode=None
    )
    
    # Load pre-trained DDQN agent
    print(f"Loading DDQN agent from {args.ddqn_model_path}...")
    ddqn_agent = DoubleDQNAgent(
        state_dim=base_env.observation_space.shape[0],
        action_dim=base_env.action_space.n,
        device=device
    )
    ddqn_agent.load(args.ddqn_model_path)

    # Freeze DDQN agent (set to evaluation mode)
    ddqn_agent.epsilon = 0.0  # No exploration
    ddqn_agent.policy_net.eval()
    ddqn_agent.target_net.eval()
    for param in ddqn_agent.policy_net.parameters():
        param.requires_grad = False
    for param in ddqn_agent.target_net.parameters():
        param.requires_grad = False
    print("DDQN agent frozen (epsilon=0, networks in eval mode, gradients disabled)")
    
    # Create two-agent environment
    print("Creating two-agent environment...")
    env = TwoAgentEnv(
        base_env=base_env,
        ddqn_agent=ddqn_agent,
        grid_size=args.grid_size,
        num_cubes=args.num_cubes,
        max_reshuffles_per_episode=args.max_reshuffles,
        reshuffle_reward_scale=args.reshuffle_reward_scale,
    )
    
    # Create MAPPO policy
    print("Creating MAPPO policy...")
    policy = MAPPOPolicy(
        obs_dim=env.agent2_obs_dim,
        action_dim=env.agent2_action_dim,
        hidden_dim=256,
        lr=args.lr,
        device=device
    )
    
    # Create MAPPO trainer
    trainer = MAPPO(
        policy=policy,
        device=device,
        clip_param=args.clip_param,
        ppo_epoch=args.ppo_epochs,
        num_mini_batch=args.num_mini_batch,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
    )
    
    # Create rollout buffer
    buffer = RolloutBuffer(
        buffer_size=args.buffer_size,
        obs_dim=env.agent2_obs_dim,
        action_dim=env.agent2_action_dim,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        device=device
    )
    
    # Initialize WandB
    print("Initializing WandB...")
    # Model name includes 'mappo_' prefix for file naming
    # WandB run name matches DDQN exactly (no 'mappo_' prefix) for easy comparison
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"mappo_astar_grid{args.grid_size}_cubes{args.num_cubes}_{timestamp}"
    wandb_run_name = f"astar_grid{args.grid_size}_cubes{args.num_cubes}_{timestamp}"

    wandb_logger = WandBLogger(
        project_name=args.wandb_project,
        run_name=args.run_name or wandb_run_name,
        config=vars(args),
        tags=["mappo", "astar", f"grid{args.grid_size}", f"cubes{args.num_cubes}"],
    )

    # Initialize detailed logger (save directly in logs/, like DDQN)
    print("Initializing detailed logger...")
    log_dir = mappo_root / "logs"
    detailed_logger = DetailedLogger(
        log_dir=log_dir,
        run_name=model_name,  # Use model_name for log files (includes 'mappo_' prefix)
        wandb_logger=wandb_logger,
    )
    
    print("Starting training...")
    print(f"Total timesteps: {args.timesteps}")
    print(f"Buffer size: {args.buffer_size}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 80)

    # Training loop
    episode = 0
    total_steps = 0

    while total_steps < args.timesteps:
        # Reset environment
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        # Reset buffer for new rollout
        buffer.reset()

        # Episode loop
        while not done and buffer.ptr < args.buffer_size:
            # Update observation normalizer statistics
            policy.update_obs_normalizer(obs)

            # Get action mask
            action_mask = env.get_agent2_action_mask()

            # Get action from policy
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action_mask_tensor = torch.as_tensor(action_mask, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, value = policy.get_actions(obs_tensor, action_mask_tensor)
                action = action.item()
                log_prob = log_prob.item()
                value = value.item()

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
                action_mask=action_mask
            )

            # Update state
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            total_steps += 1

            # If episode done, finish path
            if done:
                buffer.finish_path(last_value=0.0)

                # Log episode
                if episode % args.log_interval == 0:
                    print(f"Episode {episode} | Steps {total_steps} | Reward {episode_reward:.2f} | "
                          f"Length {episode_length} | Reshuffles {info['reshuffles_performed']}")

                # Get cube distances for logging
                cube_positions = env.base_env.get_cube_positions()
                robot_pos = env.base_env.get_robot_position()
                cube_distances = {
                    i: float(np.linalg.norm(cube_positions[i] - robot_pos))
                    for i in range(len(cube_positions))
                }

                # Log detailed episode data (including distance/time metrics)
                detailed_logger.log_episode(
                    episode=episode,
                    total_reward=episode_reward,
                    episode_length=episode_length,
                    reshuffles_performed=info["reshuffles_performed"],
                    reshuffle_details=info.get("reshuffle_history", []),
                    cube_distances=cube_distances,
                    total_distance_reduced=info.get("total_distance_reduced", 0.0),  # NEW: Log to CSV
                    total_time_saved=info.get("total_time_saved", 0.0),  # NEW: Log to CSV
                )

                # Log system metrics to WandB (matches DDQN format)
                wandb_logger.log_system_metrics({
                    "total_reward": episode_reward,
                    "episode_length": episode_length,
                    "cubes_picked": info.get("cubes_picked", 0),
                    "reshuffles_performed": info["reshuffles_performed"],
                    "distance_reduced": info.get("total_distance_reduced", 0.0),  # NEW: Distance improvement
                    "time_saved": info.get("total_time_saved", 0.0),  # NEW: Time saved
                }, total_steps)

                episode += 1

                # Reset for next episode
                obs, info = env.reset()
                episode_reward = 0
                episode_length = 0

        # If buffer not full, finish current path
        if buffer.ptr > buffer.path_start_idx:
            # Get last value
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, _, last_value = policy.get_actions(obs_tensor)
                last_value = last_value.item()
            buffer.finish_path(last_value=last_value)

        # Update policy
        if buffer.ptr >= args.buffer_size:
            print(f"Updating policy at step {total_steps}...")

            # Get data from buffer
            data = buffer.get()

            # Perform PPO updates
            for epoch in range(args.ppo_epochs):
                # Shuffle data
                indices = torch.randperm(args.buffer_size, device=device)

                # Mini-batch updates
                for start in range(0, args.buffer_size, args.batch_size):
                    end = start + args.batch_size
                    if end > args.buffer_size:
                        continue

                    mb_indices = indices[start:end]

                    # Get mini-batch
                    mb_obs = data['observations'][mb_indices]
                    mb_actions = data['actions'][mb_indices]
                    mb_values = data['values'][mb_indices]
                    mb_log_probs = data['log_probs'][mb_indices]
                    mb_advantages = data['advantages'][mb_indices]
                    mb_returns = data['returns'][mb_indices]
                    mb_action_masks = data['action_masks'][mb_indices]

                    # PPO update
                    value_loss, policy_loss, entropy, ratio = trainer.ppo_update(
                        obs_batch=mb_obs,
                        actions_batch=mb_actions,
                        value_preds_batch=mb_values,
                        return_batch=mb_returns,
                        old_action_log_probs_batch=mb_log_probs,
                        adv_targ=mb_advantages,
                        action_mask_batch=mb_action_masks,
                    )

            # Log MAPPO training metrics
            wandb_logger.log_mappo_metrics({
                "value_loss": value_loss,
                "policy_loss": policy_loss,
                "entropy": entropy,
                "ratio": ratio,
            }, total_steps)

            # Also log as general training metrics (matches DDQN format)
            wandb_logger.log_training_metrics({
                "loss": value_loss + policy_loss,  # Combined loss
                "value_loss": value_loss,
                "policy_loss": policy_loss,
            }, total_steps)

            print(f"  Value Loss: {value_loss:.4f} | Policy Loss: {policy_loss:.4f} | "
                  f"Entropy: {entropy:.4f} | Ratio: {ratio:.4f}")

        # Save checkpoint (use DDQN naming convention)
        if episode % args.save_interval == 0 and episode > 0:
            models_dir = mappo_root / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = models_dir / f"{model_name}_step_{total_steps}.pt"
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': policy.optimizer.state_dict(),
                'episode': episode,
                'total_steps': total_steps,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Save final model (use DDQN naming convention)
    models_dir = mappo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    final_path = models_dir / f"{model_name}_final.pt"
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': policy.optimizer.state_dict(),
        'episode': episode,
        'total_steps': total_steps,
    }, final_path)
    print(f"Saved final model to {final_path}")

    # Close loggers
    detailed_logger.close()
    wandb_logger.finish()
    print("Training complete!")


if __name__ == "__main__":
    main()

