"""
Double DQN Agent for Object Selection
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
import copy

from .dqn_network import DQNNetwork
from .replay_buffer import ReplayBuffer


class DoubleDQNAgent:
    """
    Double DQN agent with experience replay and target network.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize Double DQN agent.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate
            buffer_capacity: Replay buffer capacity
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            device: Device to use (cuda/cpu)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        
        # Networks
        self.policy_net = DQNNetwork(state_dim, action_dim).to(device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber loss
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # Training stats
        self.steps = 0
        self.episodes = 0
        self.losses = []
        
        print(f"[Double DQN] Initialized agent on {device}")
        print(f"[Double DQN] State dim: {state_dim}, Action dim: {action_dim}")
    
    def select_action(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            action_mask: Boolean mask for valid actions
        
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        action = self.policy_net.get_action(state_tensor, self.epsilon, action_mask)
        return action
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: Optional[np.ndarray] = None,
        next_action_mask: Optional[np.ndarray] = None
    ):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done, action_mask, next_action_mask)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones, action_masks, next_action_masks = \
            self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        next_action_masks = torch.BoolTensor(next_action_masks).to(self.device)
        
        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use policy network to select actions, target network to evaluate
        with torch.no_grad():
            # Get next actions from policy network
            next_q_values_policy = self.policy_net(next_states)
            
            # Apply action mask to next Q-values
            next_q_values_policy_masked = next_q_values_policy.clone()
            next_q_values_policy_masked[~next_action_masks] = -float('inf')
            
            next_actions = next_q_values_policy_masked.argmax(1)
            
            # Evaluate actions using target network
            next_q_values_target = self.target_net(next_states)
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Compute target Q-values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Store loss
        loss_value = loss.item()
        self.losses.append(loss_value)

        return loss_value

    def save(self, path: str):
        """
        Save agent state.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
            'losses': self.losses  # Save loss history
        }
        torch.save(checkpoint, path)
        print(f"[Double DQN] Saved checkpoint to {path}")

    def load(self, path: str):
        """
        Load agent state.

        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        # Load loss history if available (for backward compatibility)
        self.losses = checkpoint.get('losses', [])
        print(f"[Double DQN] Loaded checkpoint from {path}")
        print(f"[Double DQN] Epsilon: {self.epsilon:.4f}, Steps: {self.steps}, Episodes: {self.episodes}")
        if self.losses:
            print(f"[Double DQN] Loaded {len(self.losses)} loss values")

