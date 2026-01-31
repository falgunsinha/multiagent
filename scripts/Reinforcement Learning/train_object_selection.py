"""
Train RL Agent for Intelligent Object Selection
Uses PPO from Stable-Baselines3 to train an agent that learns to select
which object to pick first for maximum efficiency.

Usage:
    python train_object_selection.py --timesteps 100000 --headless
    python train_object_selection.py --timesteps 500000 --save_freq 10000
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Stable-Baselines3 imports
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure

# Custom imports
from src.rl.object_selection_env import ObjectSelectionEnv
from src.rl.object_selection_env_astar import ObjectSelectionEnvAStar
from src.rl.object_selection_env_rrt import ObjectSelectionEnvRRT


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train object selection RL agent")
    
    # Training parameters
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Total training timesteps")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate for PPO")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for training")
    parser.add_argument("--n_steps", type=int, default=2048,
                       help="Number of steps per update")
    parser.add_argument("--n_epochs", type=int, default=10,
                       help="Number of epochs per update")
    
    # Environment parameters
    parser.add_argument("--max_objects", type=int, default=10,
                       help="Maximum number of objects in scene")
    parser.add_argument("--max_steps", type=int, default=50,
                       help="Maximum steps per episode")
    parser.add_argument("--num_cubes", type=int, default=4,
                       help="Number of cubes per episode (default: 4)")
    parser.add_argument("--training_grid_size", type=int, default=3,
                       help="Training grid size (e.g., 3 for 3x3 grid, default: 3)")
    parser.add_argument("--method", type=str, default="heuristic",
                       choices=["heuristic", "astar", "rrt"],
                       help="Training method: heuristic (Euclidean), astar (A* path), rrt (actual RRT)")
    
    # Saving and logging
    parser.add_argument("--save_freq", type=int, default=10000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--log_dir", type=str, default="logs/object_selection",
                       help="Directory for TensorBoard logs")
    parser.add_argument("--model_dir", type=str, default="models/object_selection",
                       help="Directory to save models")
    
    # Policy options
    parser.add_argument("--use_attention", action="store_true",
                       help="Use attention-based policy (default: simple MLP)")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--headless", action="store_true",
                       help="Run in headless mode (no rendering)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to model to resume training from")
    
    return parser.parse_args()


def mask_fn(env):
    """Extract action mask from environment for ActionMasker wrapper"""
    return env.action_masks()


def make_env(max_objects: int, max_steps: int, num_cubes: int, training_grid_size: int = 3, method: str = "heuristic", seed: int = 0):
    """
    Create and wrap environment with action masking.

    Args:
        max_objects: Maximum number of objects
        max_steps: Maximum steps per episode
        num_cubes: Number of cubes per episode
        training_grid_size: Training grid size (e.g., 3 for 3x3)
        method: Training method (heuristic, astar, rrt)
        seed: Random seed

    Returns:
        Wrapped environment with action masking
    """
    def _init():
        # Select environment class based on method
        if method == "astar":
            env_class = ObjectSelectionEnvAStar
        elif method == "rrt":
            env_class = ObjectSelectionEnvRRT
        else:  # heuristic
            env_class = ObjectSelectionEnv

        env = env_class(
            franka_controller=None,  # Will be set later when integrated
            max_objects=max_objects,
            max_steps=max_steps,
            num_cubes=num_cubes,
            training_grid_size=training_grid_size,
            render_mode=None
        )
        # Wrap with ActionMasker for action masking support
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def main():
    """Main training loop"""
    args = parse_args()

    # Validate: max_objects should match training_grid_size^2
    expected_max_objects = args.training_grid_size * args.training_grid_size
    if args.max_objects != expected_max_objects:
        print(f"[WARNING] max_objects ({args.max_objects}) doesn't match training_grid_size^2 ({expected_max_objects})")
        print(f"[WARNING] Setting max_objects to {expected_max_objects}")
        args.max_objects = expected_max_objects

    # Validate: num_cubes should not exceed grid capacity
    if args.num_cubes > args.max_objects:
        print(f"[ERROR] num_cubes ({args.num_cubes}) exceeds grid capacity ({args.max_objects})")
        print(f"[ERROR] Please reduce num_cubes or increase training_grid_size")
        return

    print("=" * 60)
    print("OBJECT SELECTION RL TRAINING")
    print("=" * 60)
    print(f"Method: {args.method.upper()}")
    if args.method == "heuristic":
        print("  - Reward based on Euclidean distance")
        print("  - Distance to robot, container, obstacles")
    elif args.method == "astar":
        print("  - Reward based on A* path length estimation")
        print("  - Grid-based pathfinding with obstacles")
    elif args.method == "rrt":
        print("  - Reward based on actual RRT planning")
    print(f"Timesteps: {args.timesteps}")
    print(f"Training grid: {args.training_grid_size}x{args.training_grid_size} ({args.max_objects} cells)")
    print(f"Cubes per episode: {args.num_cubes} ({100*args.num_cubes/args.max_objects:.1f}% filled)")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Policy: {'Attention-based' if args.use_attention else 'Simple MLP'}")
    print(f"Headless: {args.headless}")
    print("=" * 60)
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"object_selection_{args.method}_grid{args.training_grid_size}x{args.training_grid_size}_cubes{args.num_cubes}_{timestamp}"

    # Create environment
    print("\nCreating environment...")
    env = DummyVecEnv([make_env(args.max_objects, args.max_steps, args.num_cubes, args.training_grid_size, args.method, args.seed)])
    
    # Normalize observations and rewards
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    # Create or load model (using MaskablePPO for action masking)
    if args.resume:
        print(f"\nLoading model from {args.resume}...")
        model = MaskablePPO.load(args.resume, env=env)
        print("Model loaded successfully!")
    else:
        print("\nCreating new MaskablePPO model with action masking...")
        model = MaskablePPO(
            "MlpPolicy",  # Use built-in MlpPolicy (supports action masking)
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=args.log_dir,
            seed=args.seed
        )

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=os.path.join(args.model_dir, run_name),
        name_prefix="ppo_object_selection",
        save_vecnormalize=True
    )

    # Configure logger
    logger = configure(os.path.join(args.log_dir, run_name), ["stdout", "tensorboard"])
    model.set_logger(logger)

    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Monitor training with: tensorboard --logdir {args.log_dir}")
    print("=" * 60 + "\n")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=not bool(args.resume)
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")

    # Save final model
    final_model_path = os.path.join(args.model_dir, f"{run_name}_final")
    model.save(final_model_path)
    env.save(os.path.join(args.model_dir, f"{run_name}_vecnormalize.pkl"))

    # Save training metadata (grid size, max_objects, method, etc.)
    import json
    metadata = {
        "method": args.method,
        "training_grid_size": args.training_grid_size,
        "num_cubes": args.num_cubes,
        "max_objects": args.max_objects,
        "max_steps": args.max_steps,
        "timestamp": timestamp,
        "total_timesteps": args.timesteps,
        "learning_rate": args.learning_rate,
        "policy": "attention" if args.use_attention else "mlp"
    }
    metadata_path = os.path.join(args.model_dir, f"{run_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final model saved to: {final_model_path}")
    print(f"VecNormalize stats saved to: {run_name}_vecnormalize.pkl")
    print(f"Metadata saved to: {run_name}_metadata.json")
    print("=" * 60)

    # Close environment
    env.close()


if __name__ == "__main__":
    main()


