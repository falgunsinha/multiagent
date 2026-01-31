"""
Example: Training RL Agent with Action Masking (MaskablePPO)
This script demonstrates how to use action masking to prevent invalid picks.

IMPORTANT: This requires sb3-contrib to be installed:
    pip install sb3-contrib

Usage:
    python train_with_action_masking_example.py --timesteps 10000 --training_grid_size 4
"""

import argparse
import sys
from pathlib import Path
import os
import time
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Stable-Baselines3 imports
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure

# Custom imports
from src.rl.object_selection_env import ObjectSelectionEnv


def mask_fn(env):
    """
    Function to extract action mask from environment.
    Required by ActionMasker wrapper.
    """
    return env.action_masks()


def main():
    parser = argparse.ArgumentParser(description="Train with action masking (MaskablePPO)")
    parser.add_argument("--timesteps", type=int, default=10000,
                       help="Total training timesteps (default: 10000)")
    parser.add_argument("--training_grid_size", type=int, default=4,
                       help="Training grid size (e.g., 4 for 4x4 grid, default: 4)")
    parser.add_argument("--num_cubes", type=int, default=9,
                       help="Number of cubes per episode (default: 9)")
    parser.add_argument("--save_freq", type=int, default=2000,
                       help="Save checkpoint every N steps (default: 2000)")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate for PPO (default: 3e-4)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("Training RL Agent with Action Masking (MaskablePPO)")
    print("="*60)
    print(f"Grid: {args.training_grid_size}x{args.training_grid_size}")
    print(f"Cubes: {args.num_cubes}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Action Masking: ENABLED (invalid picks prevented)")
    print("="*60 + "\n")

    # Create environment
    max_objects = args.training_grid_size * args.training_grid_size
    env = ObjectSelectionEnv(
        franka_controller=None,  # Standalone training without Isaac Sim
        max_objects=max_objects,
        max_steps=50,
        num_cubes=args.num_cubes,
        training_grid_size=args.training_grid_size,
        dynamic_obstacles=True
    )

    # Wrap with ActionMasker (CRITICAL for action masking)
    env = ActionMasker(env, mask_fn)
    
    # Wrap with Monitor for logging
    env = Monitor(env)
    
    # Create vectorized environment
    vec_env = DummyVecEnv([lambda: env])
    
    # Wrap with VecNormalize for observation/reward normalization
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"masked_ppo_grid{args.training_grid_size}x{args.training_grid_size}_cubes{args.num_cubes}_{timestamp}"
    log_dir = os.path.join("logs", "object_selection_masked", run_name)
    model_dir = os.path.join("models", "object_selection_masked")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Create MaskablePPO model
    print("\n[TRAINER] Creating MaskablePPO model with action masking...")
    model = MaskablePPO(
        "MlpPolicy",  # Use built-in MlpPolicy (supports action masking)
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=log_dir,
        seed=42
    )

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=os.path.join(model_dir, run_name),
        name_prefix="masked_ppo",
        save_vecnormalize=True
    )

    # Train
    print(f"\n[TRAINER] Starting training for {args.timesteps} timesteps...")
    print(f"[TRAINER] Checkpoints saved to: {model_dir}/{run_name}")
    print(f"[TRAINER] Logs saved to: {log_dir}")
    
    start_time = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    training_time = time.time() - start_time

    # Save final model
    final_model_path = os.path.join(model_dir, f"{run_name}_final")
    model.save(final_model_path)
    vec_env.save(os.path.join(model_dir, f"{run_name}_vecnormalize.pkl"))
    
    print(f"\n[TRAINER] Training complete! Time: {training_time:.1f}s")
    print(f"[TRAINER] Final model saved to: {final_model_path}.zip")
    print(f"[TRAINER] VecNormalize saved to: {final_model_path}_vecnormalize.pkl")


if __name__ == "__main__":
    main()

