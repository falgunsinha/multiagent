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
        epsilon_decay_type: str = 'multiplicative',  # NEW: 'multiplicative' or 'exponential'
        epsilon_decay_rate: int = 2500,  # NEW: For exponential decay
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        target_update_tau: float = 0.005,  # NEW: For soft target update
        warmup_steps: int = 1000,  # NEW: Warmup period before training
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
            epsilon_decay: Epsilon decay rate (for multiplicative decay)
            epsilon_decay_type: Type of epsilon decay ('multiplicative' or 'exponential')
            epsilon_decay_rate: Decay rate for exponential epsilon decay
            buffer_capacity: Replay buffer capacity
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates (for hard update)
            target_update_tau: Tau for soft target update (Polyak averaging)
            warmup_steps: Number of steps before training starts
            device: Device to use (cuda/cpu)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_type = epsilon_decay_type
        self.epsilon_decay_rate = epsilon_decay_rate
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.target_update_tau = target_update_tau
        self.warmup_steps = warmup_steps
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
        self.q_values = []  # Track Q-values for logging
        self.last_q_value = 0.0  # Last Q-value for current step

        # Advanced Q-value statistics (for WandB logging)
        self.q_mean = 0.0  # Mean Q-value across all actions
        self.q_max = 0.0  # Max Q-value
        self.q_std = 0.0  # Q-value standard deviation
        self.q_overestimation = 0.0  # DDQN overestimation bias
        self.value_estimate = 0.0  # State value estimate V(s)
        self.td_error = 0.0  # TD error magnitude

        print(f"[Double DQN] Initialized agent on {device}")
        print(f"[Double DQN] State dim: {state_dim}, Action dim: {action_dim}")
        print(f"[Double DQN] Epsilon decay: {epsilon_decay_type}")
        print(f"[Double DQN] Target update: {'soft (tau=' + str(target_update_tau) + ')' if target_update_tau > 0 else 'hard (freq=' + str(target_update_freq) + ')'}")
        print(f"[Double DQN] Warmup steps: {warmup_steps}")
    
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

        # Track Q-value for selected action (for logging)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor.unsqueeze(0))
            self.last_q_value = q_values[0, action].item()

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
        # Increment step counter (must be before warmup check)
        self.steps += 1

        # Warmup period: don't train until we have enough samples AND warmup is complete
        if len(self.replay_buffer) < self.batch_size or self.steps < self.warmup_steps:
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

        # Update target network (soft or hard update)
        if self.target_update_tau > 0:
            # Soft update (Polyak averaging)
            self._soft_update_target_network()
        else:
            # Hard update (copy weights every N steps)
            if self.steps % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon (multiplicative or exponential)
        if self.epsilon_decay_type == 'exponential':
            # Exponential decay: ε = ε_end + (ε_start - ε_end) * exp(-t / decay_rate)
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                          np.exp(-self.steps / self.epsilon_decay_rate)
        else:
            # Multiplicative decay: ε = max(ε_end, ε * decay)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Store loss and Q-value
        loss_value = loss.item()
        self.losses.append(loss_value)
        self.q_values.append(self.last_q_value)

        # Advanced Q-value statistics for WandB logging
        with torch.no_grad():
            # Get all Q-values for current states
            all_q_values_policy = self.policy_net(states)
            all_q_values_target = self.target_net(states)

            # Q-value statistics
            self.q_mean = all_q_values_policy.mean().item()
            self.q_max = all_q_values_policy.max(1)[0].mean().item()
            self.q_std = all_q_values_policy.std().item()

            # DDQN overestimation bias tracking
            # Policy network max Q (standard DQN - tends to overestimate)
            q_max_policy = all_q_values_policy.max(1)[0].mean().item()

            # Double Q-learning estimate (uses policy to select, target to evaluate)
            selected_actions = all_q_values_policy.max(1)[1]
            q_double = all_q_values_target.gather(1, selected_actions.unsqueeze(1)).squeeze(1).mean().item()

            # Overestimation bias (should be close to zero for DDQN)
            self.q_overestimation = q_max_policy - q_double

            # Value estimate V(s) - DDQN's estimate of state value
            self.value_estimate = q_double

            # TD error magnitude
            self.td_error = (current_q_values - target_q_values).abs().mean().item()

        return loss_value

    def _soft_update_target_network(self):
        """
        Soft update of target network using Polyak averaging.
        θ_target = τ * θ_policy + (1 - τ) * θ_target
        """
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                self.target_update_tau * policy_param.data +
                (1.0 - self.target_update_tau) * target_param.data
            )

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
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_decay_type': self.epsilon_decay_type,
            'epsilon_decay_rate': self.epsilon_decay_rate,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
            'target_update_tau': self.target_update_tau,
            'warmup_steps': self.warmup_steps,
            'losses': self.losses,  # Save loss history
            'q_values': self.q_values  # Save Q-value history
        }
        torch.save(checkpoint, path)
        print(f"[Double DQN] Saved checkpoint to {path}")

    def load(self, path: str, weights_only: bool = False):
        """
        Load agent state.

        Args:
            path: Path to load checkpoint from
            weights_only: If True, use weights_only=True for torch.load (safer but may fail on old checkpoints)
        """
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=weights_only)
        except Exception as e:
            if weights_only:
                # Retry without weights_only for backward compatibility
                print(f"[Double DQN] Loading with weights_only=False for compatibility")
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            else:
                raise e

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']

        # Load new parameters (with backward compatibility)
        self.epsilon_start = checkpoint.get('epsilon_start', self.epsilon_start)
        self.epsilon_decay_type = checkpoint.get('epsilon_decay_type', self.epsilon_decay_type)
        self.epsilon_decay_rate = checkpoint.get('epsilon_decay_rate', self.epsilon_decay_rate)
        self.target_update_tau = checkpoint.get('target_update_tau', self.target_update_tau)
        self.warmup_steps = checkpoint.get('warmup_steps', self.warmup_steps)

        # Load loss and Q-value history (for backward compatibility)
        self.losses = checkpoint.get('losses', [])
        self.q_values = checkpoint.get('q_values', [])

        print(f"[Double DQN] Loaded checkpoint from {path}")
        print(f"[Double DQN] Epsilon: {self.epsilon:.4f}, Steps: {self.steps}, Episodes: {self.episodes}")
        if self.losses:
            print(f"[Double DQN] Loaded {len(self.losses)} loss values")
        if self.q_values:
            print(f"[Double DQN] Loaded {len(self.q_values)} Q-value records")

