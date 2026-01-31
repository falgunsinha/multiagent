"""
Test Double DQN Agent
Evaluate trained Double DQN model on test episodes.

Usage:
    cd "C:\isaacsim\cobotproject\scripts\Reinforcement Learning\doubleDQN_script"
    py -3.11 test_ddqn.py --model_path "models\ddqn_astar_grid4_cubes9_YYYYMMDD_HHMMSS_final.pt" --episodes 100
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import json

# Add project root to path
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from src.rl.object_selection_env import ObjectSelectionEnv
from src.rl.object_selection_env_astar import ObjectSelectionEnvAStar
from src.rl.object_selection_env_rrt_viz import ObjectSelectionEnvRRTViz
from src.rl.doubleDQN import DoubleDQNAgent


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test Double DQN agent")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of test episodes (default: 100)")
    parser.add_argument("--render", action="store_true",
                       help="Render episodes (not implemented)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load metadata
    metadata_path = args.model_path.replace('.pt', '_metadata.json').replace('_final', '_metadata')
    if not os.path.exists(metadata_path):
        # Try alternative path
        base_path = args.model_path.replace('.pt', '').replace('_final', '')
        metadata_path = f"{base_path}_metadata.json"
    
    if not os.path.exists(metadata_path):
        print(f"ERROR: Metadata file not found: {metadata_path}")
        print("Please ensure metadata.json exists alongside the model checkpoint")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("=" * 60)
    print("DOUBLE DQN MODEL TESTING")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Method: {metadata.get('method', 'unknown')}")
    print(f"Grid: {metadata['training_grid_size']}x{metadata['training_grid_size']}")
    print(f"Cubes: {metadata['num_cubes']}")
    print(f"Test episodes: {args.episodes}")
    print("=" * 60)
    
    # Create environment
    method = metadata.get('method', 'heuristic')
    max_objects = metadata['max_objects']
    grid_size = metadata['training_grid_size']
    num_cubes = metadata['num_cubes']
    
    if method == 'astar':
        env = ObjectSelectionEnvAStar(
            franka_controller=None,
            max_objects=max_objects,
            max_steps=metadata['max_steps'],
            num_cubes=num_cubes,
            training_grid_size=grid_size,
            render_mode=None
        )
    elif method == 'rrt_viz':
        env = ObjectSelectionEnvRRTViz(
            franka_controller=None,
            max_objects=max_objects,
            max_steps=metadata['max_steps'],
            num_cubes=num_cubes,
            training_grid_size=grid_size,
            render_mode=None
        )
    else:
        env = ObjectSelectionEnv(
            franka_controller=None,
            max_objects=max_objects,
            max_steps=metadata['max_steps'],
            num_cubes=num_cubes,
            training_grid_size=grid_size,
            render_mode=None
        )
    
    # Create agent
    state_dim = max_objects * 6
    action_dim = max_objects
    
    agent = DoubleDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=metadata.get('learning_rate', 1e-3),
        gamma=metadata.get('gamma', 0.99),
        epsilon_start=0.0,  # No exploration during testing
        epsilon_end=0.0,
        epsilon_decay=1.0
    )
    
    # Load model
    agent.load(args.model_path)
    agent.epsilon = 0.0  # Ensure no exploration
    
    print("\nRunning test episodes...")
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(args.episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Get action mask
            action_mask = info.get('action_mask', env.action_masks())
            
            # Select action (greedy, no exploration)
            action = agent.select_action(state, action_mask)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if info.get('success', False):
            success_count += 1
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{args.episodes} | "
                  f"Avg Reward: {np.mean(episode_rewards):.2f} | "
                  f"Success Rate: {100*success_count/(episode+1):.1f}%")
    
    # Print results
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Success Rate: {100*success_count/args.episodes:.1f}%")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

