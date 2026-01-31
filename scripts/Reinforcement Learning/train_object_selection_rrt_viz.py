"""
Train RL Agent for Intelligent Object Selection with RRT Viz
Uses PPO from Stable-Baselines3 to train an agent with PythonRobotics RRT.
This is for visualizer training - uses standard RRT algorithm (not Isaac Sim).

Usage:
    py -3.11 train_object_selection_rrt_viz.py --timesteps 100000 --grid_size 4 --num_cubes 9
    py -3.11 train_object_selection_rrt_viz.py --timesteps 500000 --save_freq 10000
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
from src.rl.object_selection_env_rrt_viz import ObjectSelectionEnvRRTViz


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train object selection RL agent with RRT Viz")
    
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
    parser.add_argument("--grid_size", type=int, default=3,
                       help="Training grid size (e.g., 3 for 3x3 grid, default: 3)")
    
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
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to model to resume training from")
    
    return parser.parse_args()


def mask_fn(env):
    """Extract action mask from environment for ActionMasker wrapper"""
    return env.action_masks()


def make_env(max_objects: int, max_steps: int, num_cubes: int, grid_size: int = 3, seed: int = 0):
    """
    Create and wrap environment with action masking.

    Args:
        max_objects: Maximum number of objects
        max_steps: Maximum steps per episode
        num_cubes: Number of cubes per episode
        grid_size: Training grid size
        seed: Random seed

    Returns:
        Wrapped environment
    """
    def _init():
        # Create RRT Viz environment
        env = ObjectSelectionEnvRRTViz(
            franka_controller=None,
            max_objects=max_objects,
            max_steps=max_steps,
            num_cubes=num_cubes,
            render_mode=None,
            dynamic_obstacles=True,
            training_grid_size=grid_size
        )

        # IMPORTANT: Wrap with ActionMasker BEFORE Monitor
        # ActionMasker needs direct access to env.action_masks()
        env = ActionMasker(env, mask_fn)

        # Wrap with Monitor for logging (outermost wrapper)
        env = Monitor(env)

        # Set seed
        env.reset(seed=seed)

        return env

    return _init


def main():
    args = parse_args()
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Print configuration
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION - RRT VIZ")
    print("="*60)
    print(f"Method: RRT Viz (PythonRobotics RRT)")
    print(f"  - Reward based on PythonRobotics RRT path planning")
    print(f"  - Grid-based RRT with obstacles")
    print(f"Timesteps: {args.timesteps}")
    print(f"Training grid: {args.grid_size}x{args.grid_size} ({args.max_objects} cells)")
    print(f"Cubes per episode: {args.num_cubes}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"N steps: {args.n_steps}")
    print(f"N epochs: {args.n_epochs}")
    print(f"Seed: {args.seed}")
    print("="*60 + "\n")

    # Create vectorized environment
    env = DummyVecEnv([make_env(args.max_objects, args.max_steps, args.num_cubes, args.grid_size, args.seed)])

    # Wrap with VecNormalize for observation normalization
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Configure logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"rrt_viz_grid{args.grid_size}_cubes{args.num_cubes}_{timestamp}")
    new_logger = configure(log_path, ["stdout", "tensorboard"])

    # Create or load model
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        model = MaskablePPO.load(args.resume, env=env)
        model.set_logger(new_logger)
    else:
        # Policy kwargs (SB3 v1.8.0+ format: use dict directly, not list)
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )

        if args.use_attention:
            print("Using attention-based policy")
            policy_kwargs["features_extractor_class"] = "AttentionExtractor"

        # Create model
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            tensorboard_log=log_path
        )
        model.set_logger(new_logger)

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.model_dir,
        name_prefix=f"rrt_viz_grid{args.grid_size}_cubes{args.num_cubes}",
        save_vecnormalize=True
    )

    # Train model
    print(f"\nStarting training for {args.timesteps} timesteps...")
    print(f"Checkpoints will be saved to: {args.model_dir}")
    print(f"Logs will be saved to: {log_path}\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )

    # Save final model
    final_model_path = os.path.join(args.model_dir, f"rrt_viz_grid{args.grid_size}_cubes{args.num_cubes}_final")
    model.save(final_model_path)
    env.save(os.path.join(args.model_dir, f"rrt_viz_grid{args.grid_size}_cubes{args.num_cubes}_final_vecnormalize.pkl"))

    print(f"\nTraining complete!")
    print(f"Final model saved to: {final_model_path}")
    print(f"VecNormalize stats saved to: {final_model_path}_vecnormalize.pkl")

    # Close environment
    env.close()


if __name__ == "__main__":
    main()


