"""
Utility functions for testing GAT+CVD model
"""

import torch
import numpy as np
from pathlib import Path


def load_gat_cvd_checkpoint(checkpoint_path, config, device):
    """
    Load GAT+CVD checkpoint for testing
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        device: torch device
        
    Returns:
        GATCVDAgent instance with loaded weights
    """
    from gat_cvd.gat_cvd_agent import GATCVDAgent
    
    # Create agent
    agent = GATCVDAgent(config, device=device)

    # Load checkpoint with weights_only=False for torch_geometric compatibility
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.ddqn_policy.load_state_dict(checkpoint['ddqn_policy'])
    agent.masac_policy.load_state_dict(checkpoint['masac_policy'])
    agent.cvd_module.load_state_dict(checkpoint['cvd_module'])
    agent.train_step = checkpoint.get('train_step', 0)
    agent.episodes = checkpoint.get('episodes', 0)

    # Set to evaluation mode (no training)
    agent.ddqn_policy.eval()
    agent.masac_policy.eval()
    agent.cvd_module.eval()
    
    print(f"✅ Loaded GAT+CVD checkpoint from: {checkpoint_path}")
    print(f"   Training steps: {agent.train_step}")
    print(f"   Episodes: {agent.episodes}")
    
    return agent


def create_gat_cvd_wrapper(gat_cvd_agent, trainer, config, device):
    """
    Create a wrapper that makes GAT+CVD compatible with TwoAgentEnv
    
    Args:
        gat_cvd_agent: GATCVDAgent instance
        trainer: FrankaRRTTrainer instance
        config: Configuration dictionary
        device: torch device
        
    Returns:
        Wrapper object with select_action() method
    """
    from gat_cvd.graph_utils import build_graph
    
    class GATCVDWrapper:
        """Wrapper to make GAT+CVD compatible with TwoAgentEnv"""
        def __init__(self, agent, trainer, config, device):
            self.agent = agent
            self.trainer = trainer
            self.config = config
            self.device = device
            self.epsilon = 0.0  # No exploration during testing
            
        def select_action(self, obs, action_mask):
            """Select action using DDQN from GAT+CVD"""
            # Build graph from observation
            robot_pos, _ = self.trainer.franka.get_world_pose()
            robot_positions = [robot_pos]
            object_positions = self.trainer.get_cube_positions()
            obstacle_positions = self.trainer.get_obstacle_positions()
            
            graph = build_graph(
                obs=obs,
                robot_positions=robot_positions,
                object_positions=object_positions,
                obstacles=obstacle_positions,
                edge_threshold=self.config['graph']['edge_threshold'],
                device=self.device
            )
            
            # Use GAT+CVD's DDQN to select action (no exploration)
            action, _ = self.agent.select_actions(
                graph,
                epsilon_ddqn=0.0,  # No exploration
                epsilon_masac=0.0,  # No exploration
                action_mask=action_mask
            )
            return action
    
    return GATCVDWrapper(gat_cvd_agent, trainer, config, device)


def create_gat_cvd_masac_wrapper(gat_cvd_agent, trainer, config, device, grid_size, num_cubes):
    """
    Create a wrapper for GAT+CVD's MASAC component (Agent 2)

    Args:
        gat_cvd_agent: GATCVDAgent instance
        trainer: FrankaRRTTrainer instance
        config: Configuration dictionary
        device: torch device
        grid_size: Grid size
        num_cubes: Number of cubes

    Returns:
        Wrapper object with select_action() method for MASAC
    """
    from gat_cvd.graph_utils import build_graph

    class GATCVDMASACWrapper:
        """Wrapper to make GAT+CVD's MASAC compatible with TwoAgentEnv"""
        def __init__(self, agent, trainer, config, device, grid_size, num_cubes):
            self.agent = agent
            self.trainer = trainer
            self.config = config
            self.device = device
            self.grid_size = grid_size
            self.num_cubes = num_cubes
            self.epsilon = 0.0  # No exploration during testing

        def select_action(self, obs, deterministic=True, valid_cubes=None):
            """Select reshuffling action using MASAC from GAT+CVD"""
            # Build graph from observation
            robot_pos, _ = self.trainer.franka.get_world_pose()
            robot_positions = [robot_pos]
            object_positions = self.trainer.get_cube_positions()
            obstacle_positions = self.trainer.get_obstacle_positions()

            graph = build_graph(
                obs=obs,
                robot_positions=robot_positions,
                object_positions=object_positions,
                obstacles=obstacle_positions,
                edge_threshold=self.config['graph']['edge_threshold'],
                device=self.device
            )

            # Use GAT+CVD's MASAC to select action (no exploration)
            _, action_masac = self.agent.select_actions(
                graph,
                epsilon_ddqn=0.0,  # No exploration
                epsilon_masac=0.0,  # No exploration
                action_mask=None
            )

            # Convert MASAC continuous action to dictionary format
            # action_masac is [cube_idx_normalized, grid_x_normalized, grid_y_normalized]
            cube_idx = int((action_masac[0] + 1) / 2 * (self.num_cubes - 1))
            grid_x = int((action_masac[1] + 1) / 2 * (self.grid_size - 1))
            grid_y = int((action_masac[2] + 1) / 2 * (self.grid_size - 1))

            # Clamp to valid ranges
            cube_idx = max(0, min(cube_idx, self.num_cubes - 1))
            grid_x = max(0, min(grid_x, self.grid_size - 1))
            grid_y = max(0, min(grid_y, self.grid_size - 1))

            return {
                'cube_idx': cube_idx,
                'target_grid_x': grid_x,
                'target_grid_y': grid_y
            }

    return GATCVDMASACWrapper(gat_cvd_agent, trainer, config, device, grid_size, num_cubes)


def load_baseline_ddqn(model_path, state_dim, action_dim):
    """
    Load baseline DDQN model for comparison
    
    Args:
        model_path: Path to DDQN checkpoint
        state_dim: State dimension
        action_dim: Action dimension
        
    Returns:
        DoubleDQNAgent instance
    """
    from src.rl.doubleDQN import DoubleDQNAgent
    
    agent = DoubleDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=0.0,  # No exploration during testing
        epsilon_end=0.0,
        epsilon_decay=1.0,
        buffer_capacity=10000,
        batch_size=64
    )
    
    # Load weights
    agent.load(model_path)
    print(f"✅ Loaded Baseline DDQN from: {model_path}")
    
    return agent


def calculate_efficiency_metrics(episode_data):
    """
    Calculate efficiency metrics from episode data

    Args:
        episode_data: Dictionary with episode metrics including:
            - total_distance_reduced: Distance saved by reshuffling (meters)
            - total_distance_traveled: Actual distance traveled by robot (meters)
            - total_time_saved: Time saved by reshuffling (seconds)
            - total_time_taken: Actual time taken for episode (seconds)
            - reshuffles_performed: Number of reshuffles executed
            - episode_length: Total steps in episode
            - cubes_picked: Number of cubes successfully picked
            - total_cubes: Total number of cubes available

    Returns:
        Dictionary with efficiency metrics:
            - distance_efficiency: (distance_reduced / distance_traveled) * 100 (percentage)
            - time_efficiency: (time_saved / time_taken) * 100 (percentage)
            - avg_distance_per_reshuffle: Avg distance reduced per reshuffle (m/reshuffle)
            - avg_time_per_reshuffle: Avg time saved per reshuffle (s/reshuffle)
            - success_rate: Cubes picked / Total cubes available (percentage)
            - steps_per_cube: Episode length / Cubes picked (lower is better)
    """
    # TRUE Distance Efficiency: What % of actual distance traveled was saved by reshuffling?
    distance_efficiency = 0.0
    if episode_data.get('total_distance_traveled', 0) > 0:
        distance_efficiency = (episode_data['total_distance_reduced'] / episode_data['total_distance_traveled']) * 100.0

    # TRUE Time Efficiency: What % of actual time taken was saved by reshuffling?
    time_efficiency = 0.0
    if episode_data.get('total_time_taken', 0) > 0:
        time_efficiency = (episode_data['total_time_saved'] / episode_data['total_time_taken']) * 100.0

    # Average distance reduced per reshuffle (for comparison)
    avg_distance_per_reshuffle = 0.0
    if episode_data['reshuffles_performed'] > 0:
        avg_distance_per_reshuffle = episode_data['total_distance_reduced'] / episode_data['reshuffles_performed']

    # Average time saved per reshuffle (for comparison)
    avg_time_per_reshuffle = 0.0
    if episode_data['reshuffles_performed'] > 0:
        avg_time_per_reshuffle = episode_data['total_time_saved'] / episode_data['reshuffles_performed']

    # Success rate: percentage of cubes picked
    success_rate = 0.0
    if episode_data.get('total_cubes', 0) > 0:
        success_rate = (episode_data['cubes_picked'] / episode_data['total_cubes']) * 100.0

    # Steps per cube: efficiency of picking (lower is better)
    steps_per_cube = 0.0
    if episode_data['cubes_picked'] > 0:
        steps_per_cube = episode_data['episode_length'] / episode_data['cubes_picked']

    return {
        'distance_efficiency': distance_efficiency,
        'time_efficiency': time_efficiency,
        'avg_distance_per_reshuffle': avg_distance_per_reshuffle,
        'avg_time_per_reshuffle': avg_time_per_reshuffle,
        'success_rate': success_rate,
        'steps_per_cube': steps_per_cube
    }

