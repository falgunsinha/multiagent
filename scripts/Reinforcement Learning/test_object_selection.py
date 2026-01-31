"""
Test Trained RL Agent for Object Selection
Evaluates a trained RL model and compares it with baseline strategies.

Usage:
    python test_object_selection.py --model_path models/object_selection/ppo_final.zip
    python test_object_selection.py --model_path models/object_selection/ppo_final.zip --episodes 100
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import time
from typing import Dict, List

# Add project root to path
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Stable-Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Custom imports
from src.rl.object_selection_env import ObjectSelectionEnv
from src.rl.reward_shaping import RewardShaper


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test object selection RL agent")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=20,
                       help="Number of test episodes")
    parser.add_argument("--max_objects", type=int, default=10,
                       help="Maximum number of objects")
    parser.add_argument("--max_steps", type=int, default=50,
                       help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--compare_baseline", action="store_true",
                       help="Compare with greedy baseline")
    
    return parser.parse_args()


def test_rl_agent(model, env, num_episodes: int = 20) -> Dict:
    """
    Test RL agent performance.
    
    Args:
        model: Trained PPO model
        env: Test environment
        num_episodes: Number of episodes to test
        
    Returns:
        Dictionary with performance metrics
    """
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    total_time = 0
    
    print("\n" + "=" * 60)
    print("TESTING RL AGENT")
    print("=" * 60)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        start_time = time.time()
        
        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                if info.get("success", False):
                    success_count += 1
                break
        
        episode_time = time.time() - start_time
        total_time += episode_time
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Length={episode_length}, "
              f"Success={info.get('success', False)}, "
              f"Time={episode_time:.2f}s")
    
    # Calculate statistics
    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "success_rate": success_count / num_episodes,
        "mean_time": total_time / num_episodes
    }
    
    print("\n" + "=" * 60)
    print("RL AGENT RESULTS")
    print("=" * 60)
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Length: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
    print(f"Success Rate: {results['success_rate'] * 100:.1f}%")
    print(f"Mean Time: {results['mean_time']:.2f}s")
    print("=" * 60)
    
    return results


def test_greedy_baseline(env, num_episodes: int = 20) -> Dict:
    """
    Test greedy baseline (always pick closest object).
    
    Args:
        env: Test environment
        num_episodes: Number of episodes to test
        
    Returns:
        Dictionary with performance metrics
    """
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    total_time = 0
    
    print("\n" + "=" * 60)
    print("TESTING GREEDY BASELINE")
    print("=" * 60)
    
    reward_shaper = RewardShaper()
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        start_time = time.time()
        
        # Greedy strategy: always pick closest unpicked object
        for step in range(env.max_steps):
            # Find closest unpicked object
            min_distance = float('inf')
            best_action = 0
            
            for i in range(env.total_objects):
                if i not in env.objects_picked:
                    # Distance to EE is the first feature (index 0) in 6-parameter observation
                    distance = obs[i * 6 + 0]  # Distance to EE feature
                    if distance < min_distance:
                        min_distance = distance
                        best_action = i
            
            obs, reward, terminated, truncated, info = env.step(best_action)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                if info.get("success", False):
                    success_count += 1
                break
        
        episode_time = time.time() - start_time
        total_time += episode_time
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Length={episode_length}, "
              f"Success={info.get('success', False)}, "
              f"Time={episode_time:.2f}s")
    
    # Calculate statistics
    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "success_rate": success_count / num_episodes,
        "mean_time": total_time / num_episodes
    }
    
    print("\n" + "=" * 60)
    print("GREEDY BASELINE RESULTS")
    print("=" * 60)
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Length: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
    print(f"Success Rate: {results['success_rate'] * 100:.1f}%")
    print(f"Mean Time: {results['mean_time']:.2f}s")
    print("=" * 60)
    
    return results


def main():
    """Main testing function"""
    args = parse_args()

    print("\n" + "=" * 60)
    print("OBJECT SELECTION RL TESTING")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Max objects: {args.max_objects}")
    print("=" * 60)

    # Create environment
    print("\nCreating test environment...")
    env = ObjectSelectionEnv(
        franka_controller=None,
        max_objects=args.max_objects,
        max_steps=args.max_steps,
        render_mode=None
    )

    # Load model
    print(f"Loading model from {args.model_path}...")

    # Create dummy env for loading
    dummy_env = DummyVecEnv([lambda: env])

    # Load VecNormalize if exists
    vecnorm_path = args.model_path.replace(".zip", "_vecnormalize.pkl")
    if Path(vecnorm_path).exists():
        print(f"Loading VecNormalize from {vecnorm_path}...")
        dummy_env = VecNormalize.load(vecnorm_path, dummy_env)
        dummy_env.training = False
        dummy_env.norm_reward = False

    # Load model
    model = PPO.load(args.model_path, env=dummy_env)
    print("Model loaded successfully!")

    # Test RL agent
    rl_results = test_rl_agent(model, env, args.episodes)

    # Test baseline if requested
    if args.compare_baseline:
        baseline_results = test_greedy_baseline(env, args.episodes)

        # Compare results
        print("\n" + "=" * 60)
        print("COMPARISON: RL vs GREEDY BASELINE")
        print("=" * 60)
        print(f"Reward Improvement: {rl_results['mean_reward'] - baseline_results['mean_reward']:.2f} "
              f"({((rl_results['mean_reward'] / baseline_results['mean_reward']) - 1) * 100:.1f}%)")
        print(f"Success Rate Improvement: {(rl_results['success_rate'] - baseline_results['success_rate']) * 100:.1f}%")
        print(f"Time Improvement: {baseline_results['mean_time'] - rl_results['mean_time']:.2f}s "
              f"({((baseline_results['mean_time'] / rl_results['mean_time']) - 1) * 100:.1f}%)")
        print("=" * 60)

    # Close environment
    env.close()

    print("\nTesting complete!")


if __name__ == "__main__":
    main()


