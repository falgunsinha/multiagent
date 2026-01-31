"""
Experience Replay Buffer for Double DQN
"""

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done, action_mask=None, next_action_mask=None):
        """
        Add transition to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
            action_mask: Valid actions in current state
            next_action_mask: Valid actions in next state
        """
        self.buffer.append((state, action, reward, next_state, done, action_mask, next_action_mask))
    
    def sample(self, batch_size: int):
        """
        Sample random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, action_masks, next_action_masks)
        """
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        action_masks = np.array([t[5] if t[5] is not None else np.ones(len(t[0])) for t in batch])
        next_action_masks = np.array([t[6] if t[6] is not None else np.ones(len(t[3])) for t in batch])
        
        return states, actions, rewards, next_states, dones, action_masks, next_action_masks
    
    def __len__(self):
        """Return current buffer size"""
        return len(self.buffer)
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()

