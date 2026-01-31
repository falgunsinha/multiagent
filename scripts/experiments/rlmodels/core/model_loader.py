"""
Model Loader for loading pretrained RL models from DRL-Pytorch
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
import sys

# Add DRL-Pytorch to path
sys.path.append("cobotproject/scripts/Reinforcement Learning/DRL-Pytorch")


class ModelLoader:
    """Loads pretrained models from various RL algorithms"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize ModelLoader
        
        Args:
            device: Device to load models on ('cuda' or 'cpu')
        """
        self.device = device
        self.loaded_models = {}
        
    def load_model(self, model_path: str, model_type: str, 
                   state_dim: int, action_dim: int, 
                   model_config: Optional[Dict[str, Any]] = None) -> nn.Module:
        """
        Load a pretrained model
        
        Args:
            model_path: Path to model checkpoint
            model_type: Type of model ('duel_ddqn', 'per_ddqn', 'c51', 'sac', 'ppo', 'ddpg', 'td3')
            state_dim: State dimension
            action_dim: Action dimension
            model_config: Additional model configuration
            
        Returns:
            Loaded model
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Create model architecture based on type
        model = self._create_model_architecture(model_type, state_dim, action_dim, model_config)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
            
        model.to(self.device)
        model.eval()
        
        self.loaded_models[model_path.name] = model
        
        return model
    
    def _create_model_architecture(self, model_type: str, state_dim: int, 
                                  action_dim: int, config: Optional[Dict] = None) -> nn.Module:
        """
        Create model architecture based on type
        
        Args:
            model_type: Model type
            state_dim: State dimension
            action_dim: Action dimension
            config: Model configuration
            
        Returns:
            Model instance
        """
        config = config or {}
        
        if model_type in ['duel_ddqn', 'per_ddqn', 'ddqn']:
            return self._create_dqn_network(state_dim, action_dim)
        elif model_type == 'c51':
            return self._create_c51_network(state_dim, action_dim)
        elif model_type == 'sac_discrete':
            return self._create_sac_discrete_network(state_dim, action_dim)
        elif model_type == 'ppo_discrete':
            return self._create_ppo_discrete_network(state_dim, action_dim)
        elif model_type == 'ddpg':
            return self._create_ddpg_network(state_dim, action_dim)
        elif model_type == 'td3':
            return self._create_td3_network(state_dim, action_dim)
        elif model_type == 'sac_continuous':
            return self._create_sac_continuous_network(state_dim, action_dim)
        elif model_type == 'ppo_continuous':
            return self._create_ppo_continuous_network(state_dim, action_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_dqn_network(self, state_dim: int, action_dim: int) -> nn.Module:
        """Create DQN/DDQN network"""
        return nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def _create_c51_network(self, state_dim: int, action_dim: int, n_atoms: int = 51) -> nn.Module:
        """Create C51 network"""
        return nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * n_atoms)
        )
    
    def _create_sac_discrete_network(self, state_dim: int, action_dim: int) -> nn.Module:
        """Create SAC discrete actor network"""
        return nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def _create_ppo_discrete_network(self, state_dim: int, action_dim: int) -> nn.Module:
        """Create PPO discrete actor network"""
        return nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def _create_ddpg_network(self, state_dim: int, action_dim: int) -> nn.Module:
        """Create DDPG actor network"""
        return nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )
    
    def _create_td3_network(self, state_dim: int, action_dim: int) -> nn.Module:
        """Create TD3 actor network"""
        return nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
    
    def _create_sac_continuous_network(self, state_dim: int, action_dim: int) -> nn.Module:
        """Create SAC continuous actor network"""
        return nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
    
    def _create_ppo_continuous_network(self, state_dim: int, action_dim: int) -> nn.Module:
        """Create PPO continuous actor network"""
        return nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

