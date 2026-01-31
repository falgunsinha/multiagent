"""
Simple RL Model Tester
Tests the trained RL model without Isaac Sim - just shows the picking order.

Usage:
    C:\isaacsim\python.bat test_rl_model_simple.py --model_path models/object_selection/object_selection_20251128_122313_final.zip --grid_rows 2 --grid_cols 2
"""

import sys
from pathlib import Path
import numpy as np
import argparse

# Add project root to path
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rl.object_selection_env import ObjectSelectionEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def test_model(model_path: str, grid_rows: int, grid_cols: int, num_episodes: int = 5):
    """Test trained RL model"""
    
    print("\n" + "=" * 60)
    print("RL MODEL TESTER")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Grid: {grid_rows}x{grid_cols} ({grid_rows * grid_cols} objects)")
    print("=" * 60)
    
    # Create environment
    max_objects = grid_rows * grid_cols
    env = ObjectSelectionEnv(
        franka_controller=None,  # No Isaac Sim
        max_objects=max_objects,
        max_steps=50
    )
    
    # Wrap in DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])
    
    # Load VecNormalize if exists
    vecnorm_path = model_path.replace("_final.zip", "_vecnormalize.pkl")
    if Path(vecnorm_path).exists():
        print(f"Loading VecNormalize from: {vecnorm_path}")
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, env=vec_env)
    print("Model loaded successfully!\n")
    
    # Run episodes
    total_reward = 0
    for episode in range(num_episodes):
        obs = vec_env.reset()
        episode_reward = 0
        pick_order = []
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print("-" * 40)
        
        for step in range(max_objects):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            
            pick_order.append(int(action[0]))
            episode_reward += reward[0]
            
            print(f"  Step {step + 1}: Pick object {int(action[0])} (reward: {reward[0]:.2f})")
            
            if done[0]:
                break
        
        print(f"  Pick order: {pick_order}")
        print(f"  Total reward: {episode_reward:.2f}")
        total_reward += episode_reward
    
    avg_reward = total_reward / num_episodes
    print("\n" + "=" * 60)
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RL Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--grid_rows", type=int, default=2, help="Grid rows")
    parser.add_argument("--grid_cols", type=int, default=2, help="Grid columns")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
    args = parser.parse_args()
    
    test_model(args.model_path, args.grid_rows, args.grid_cols, args.episodes)

