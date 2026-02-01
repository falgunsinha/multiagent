"""
Training Script for GAT + CVD

Main training loop for heterogeneous multi-agent RL with:
- DDQN (Agent 1): Object selection
- MASAC (Agent 2): Spatial manipulation
- CVD: Counterfactual credit assignment
"""

import os
import sys
import yaml
import torch
import numpy as np
from collections import deque
import argparse
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../src/rl'))

# Import components
from gat_cvd_agent import GATCVDAgent
from graph_utils import build_graph, compute_edge_features


class ReplayBuffer:
    """Simple replay buffer for storing transitions."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition):
        """Add transition to buffer."""
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        """Sample batch from buffer."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return batch
    
    def __len__(self):
        return len(self.buffer)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_graph_from_state(state, config, device):
    """
    Convert environment state to graph representation.
    
    Args:
        state: Environment state (dict or array)
        config: Configuration dict
        device: torch device
    
    Returns:
        graph: PyG Data object
    """
    # Extract positions from state
    # This is a placeholder - adapt to your actual state representation
    robot_positions = state.get('robot_positions', [[0, 0, 0], [1, 1, 0]])
    object_positions = state.get('object_positions', [[2, 2, 0], [3, 3, 0]])
    obstacles = state.get('obstacles', None)
    targets = state.get('targets', None)
    
    # Build graph
    graph = build_graph(
        obs=state,
        robot_positions=robot_positions,
        object_positions=object_positions,
        obstacles=obstacles,
        targets=targets,
        edge_threshold=config['graph']['edge_threshold'],
        device=device
    )
    
    # Compute edge features (if enabled)
    if config['graph']['use_reachability_edges'] or config['graph']['use_blocking_scores']:
        graph = compute_edge_features(graph, rrt_estimator=None)
    
    return graph


def train_episode(env, agent, config, replay_buffer, epsilon_ddqn, epsilon_masac, device):
    """
    Train for one episode.
    
    Args:
        env: Environment
        agent: GATCVDAgent
        config: Configuration dict
        replay_buffer: Replay buffer
        epsilon_ddqn: Exploration rate for DDQN
        epsilon_masac: Exploration rate for MASAC
        device: torch device
    
    Returns:
        episode_reward: Total episode reward
        episode_length: Episode length
    """
    state = env.reset()
    episode_reward = 0
    episode_length = 0
    done = False
    
    while not done and episode_length < config['training']['max_steps_per_episode']:
        # Convert state to graph
        graph = create_graph_from_state(state, config, device)
        
        # Select actions
        action_mask = state.get('action_mask', None)  # For DDQN
        action_ddqn, action_masac = agent.select_actions(
            graph, epsilon_ddqn, epsilon_masac, action_mask
        )
        
        # Execute actions in environment
        next_state, reward, done, info = env.step({
            'ddqn': action_ddqn,
            'masac': action_masac
        })
        
        # Convert next state to graph
        next_graph = create_graph_from_state(next_state, config, device)
        
        # Store transition
        transition = {
            'graph': graph,
            'action_ddqn': action_ddqn,
            'action_masac': action_masac,
            'reward': reward,
            'next_graph': next_graph,
            'done': done
        }
        replay_buffer.push(transition)
        
        # Update state
        state = next_state
        episode_reward += reward
        episode_length += 1
        
        # Train if enough samples
        if len(replay_buffer) >= config['training']['batch_size']:
            train_step(agent, replay_buffer, config, device)
    
    return episode_reward, episode_length


def train_step(agent, replay_buffer, config, device):
    """
    Perform one training step.
    
    Args:
        agent: GATCVDAgent
        replay_buffer: Replay buffer
        config: Configuration dict
        device: torch device
    """
    # Sample batch
    batch_data = replay_buffer.sample(config['training']['batch_size'])
    
    # Prepare batch
    batch = {
        'graphs': [t['graph'] for t in batch_data],
        'actions_ddqn': torch.tensor([t['action_ddqn'] for t in batch_data], device=device),
        'actions_masac': torch.tensor([t['action_masac'] for t in batch_data], device=device),
        'rewards': torch.tensor([[t['reward']] for t in batch_data], dtype=torch.float32, device=device),
        'next_graphs': [t['next_graph'] for t in batch_data],
        'dones': torch.tensor([[float(t['done'])] for t in batch_data], dtype=torch.float32, device=device)
    }
    
    # Train DDQN
    ddqn_loss = agent.train_step_ddqn(batch)
    
    # Train MASAC
    critic_loss, actor_loss, alpha_loss = agent.train_step_masac(batch)
    
    # Train CVD
    cvd_loss = agent.train_step_cvd(batch)
    
    # Soft update targets
    agent.soft_update_targets()
    
    return ddqn_loss, critic_loss, actor_loss, cvd_loss, alpha_loss


def evaluate(env, agent, config, n_episodes, device):
    """
    Evaluate agent.

    Args:
        env: Environment
        agent: GATCVDAgent
        config: Configuration dict
        n_episodes: Number of evaluation episodes
        device: torch device

    Returns:
        avg_reward: Average episode reward
        avg_length: Average episode length
    """
    total_reward = 0
    total_length = 0

    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done and episode_length < config['training']['max_steps_per_episode']:
            # Convert state to graph
            graph = create_graph_from_state(state, config, device)

            # Select actions (greedy)
            action_mask = state.get('action_mask', None)
            action_ddqn, action_masac = agent.select_actions(
                graph, epsilon_ddqn=0.0, epsilon_masac=0.0, action_mask=action_mask
            )

            # Execute actions
            next_state, reward, done, info = env.step({
                'ddqn': action_ddqn,
                'masac': action_masac
            })

            state = next_state
            episode_reward += reward
            episode_length += 1

        total_reward += episode_reward
        total_length += episode_length

    avg_reward = total_reward / n_episodes
    avg_length = total_length / n_episodes

    return avg_reward, avg_length


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train GAT + CVD Agent')
    parser.add_argument('--config', type=str, default='config_gat_cvd.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    config = load_config(config_path)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Create directories
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)

    # Initialize environment (placeholder - replace with your actual environment)
    # from object_selection_env_rrt import ObjectSelectionEnv
    # env = ObjectSelectionEnv(config)
    print("WARNING: Environment not initialized. Please import and initialize your environment.")
    env = None

    # Initialize agent
    agent_config = {
        'node_dim': config['graph']['node_dim'],
        'edge_dim': config['graph']['edge_dim'],
        'hidden_dim': config['gat']['hidden_dim'],
        'gat_output_dim': config['gat']['output_dim'],
        'num_layers': config['gat']['num_layers'],
        'num_heads': config['gat']['num_heads'],
        'n_actions_ddqn': config['ddqn']['n_actions'],
        'n_actions_masac': config['masac']['n_actions'],
        'gamma': config['training']['gamma'],
        'tau': config['training']['tau'],
        'lr': config['training']['learning_rate']
    }
    agent = GATCVDAgent(agent_config, device=device)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(config['training']['replay_buffer_size'])

    # Training loop
    epsilon_ddqn = config['ddqn']['epsilon_start']
    epsilon_masac = config['masac']['epsilon_start']
    best_reward = -float('inf')

    print(f"\nStarting training for {config['training']['n_episodes']} episodes...")
    print("=" * 80)

    for episode in range(config['training']['n_episodes']):
        # Train episode
        if env is not None:
            episode_reward, episode_length = train_episode(
                env, agent, config, replay_buffer, epsilon_ddqn, epsilon_masac, device
            )
        else:
            # Placeholder for testing without environment
            episode_reward, episode_length = 0, 0
            print(f"Episode {episode + 1}: Skipped (no environment)")
            continue

        # Decay epsilon
        epsilon_ddqn = max(config['ddqn']['epsilon_end'],
                          epsilon_ddqn * config['ddqn']['epsilon_decay'])
        epsilon_masac = max(config['masac']['epsilon_end'],
                           epsilon_masac * config['masac']['epsilon_decay'])

        # Logging
        if (episode + 1) % config['logging']['print_freq'] == 0:
            print(f"Episode {episode + 1}/{config['training']['n_episodes']}: "
                  f"Reward = {episode_reward:.2f}, Length = {episode_length}, "
                  f"Epsilon (DDQN) = {epsilon_ddqn:.3f}, Epsilon (MASAC) = {epsilon_masac:.3f}")

        # Evaluation
        if (episode + 1) % config['training']['eval_freq'] == 0:
            avg_reward, avg_length = evaluate(
                env, agent, config, config['training']['n_eval_episodes'], device
            )
            print(f"\n{'='*80}")
            print(f"Evaluation at episode {episode + 1}:")
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  Average Length: {avg_length:.2f}")
            print(f"{'='*80}\n")

            # Save best model
            if config['checkpoint']['save_best'] and avg_reward > best_reward:
                best_reward = avg_reward
                save_path = os.path.join(config['checkpoint']['save_dir'], 'best_model.pt')
                agent.save(save_path)
                print(f"Saved best model with reward {best_reward:.2f}")

        # Save checkpoint
        if (episode + 1) % config['training']['save_freq'] == 0:
            save_path = os.path.join(config['checkpoint']['save_dir'],
                                    f'checkpoint_ep{episode + 1}.pt')
            agent.save(save_path)
            print(f"Saved checkpoint at episode {episode + 1}")

    print("\nTraining completed!")
    print(f"Best reward: {best_reward:.2f}")


if __name__ == '__main__':
    main()


