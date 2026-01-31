"""
DQN Network Architecture for Object Selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for object selection.
    
    Architecture:
    - Input: Object features (max_objects × 6 features)
    - Feature extraction: MLP layers with attention mechanism
    - Output: Q-values for each object (max_objects actions)
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        """
        Initialize DQN network.
        
        Args:
            input_dim: Input dimension (max_objects × 6)
            output_dim: Output dimension (max_objects)
            hidden_dim: Hidden layer dimension
        """
        super(DQNNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.features_per_object = 6
        self.num_objects = input_dim // self.features_per_object
        
        # Feature extraction layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Attention mechanism for object features
        self.attention_query = nn.Linear(hidden_dim, 64)
        self.attention_key = nn.Linear(hidden_dim, 64)
        self.attention_value = nn.Linear(hidden_dim, 64)
        
        # Output layer (Q-values for each action)
        self.fc4 = nn.Linear(hidden_dim + 64, hidden_dim)
        self.q_values = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            Q-values tensor (batch_size, output_dim)
        """
        # Feature extraction
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        
        # Attention mechanism
        query = self.attention_query(x3)
        key = self.attention_key(x3)
        value = self.attention_value(x3)

        # Handle both single sample and batch
        if query.dim() == 1:
            # Single sample: add batch dimension for processing
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            x3_batch = x3.unsqueeze(0)
            single_sample = True
        else:
            x3_batch = x3
            single_sample = False

        # Compute attention scores (batch_size, 1, 1)
        attention_scores = torch.bmm(query.unsqueeze(1), key.unsqueeze(2)) / np.sqrt(64)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention (batch_size, 64)
        attended = (attention_weights.squeeze(-1) * value).squeeze(1)

        # Concatenate features with attention
        combined = torch.cat([x3_batch, attended], dim=-1)

        # Remove batch dimension if single sample
        if single_sample:
            combined = combined.squeeze(0)
        
        # Output Q-values
        x4 = F.relu(self.fc4(combined))
        q_vals = self.q_values(x4)
        
        return q_vals
    
    def get_action(self, state, epsilon=0.0, action_mask=None):
        """
        Select action using epsilon-greedy policy with action masking.

        Args:
            state: Current state (numpy array or tensor)
            epsilon: Exploration rate
            action_mask: Boolean mask for valid actions (True = valid)

        Returns:
            Selected action index
        """
        if np.random.random() < epsilon:
            # Exploration: random action from valid actions
            if action_mask is not None:
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) > 0:
                    return np.random.choice(valid_actions)
            return np.random.randint(0, self.output_dim)
        else:
            # Exploitation: best action from valid actions
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    state = torch.FloatTensor(state).to(self.fc1.weight.device).unsqueeze(0)
                elif state.dim() == 1:
                    # If already a tensor but 1D, add batch dimension
                    state = state.unsqueeze(0)
                q_values = self.forward(state).squeeze(0).cpu().numpy()

                # Apply action mask
                if action_mask is not None:
                    q_values = np.where(action_mask, q_values, -np.inf)

                return np.argmax(q_values)

