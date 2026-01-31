"""
Configuration Manager for loading and managing experiment configurations
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Manages loading and accessing experiment configurations from YAML files"""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize ConfigManager
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.configs = {}
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a configuration file
        
        Args:
            config_name: Name of config file (with or without .yaml extension)
            
        Returns:
            Dictionary containing configuration
        """
        if not config_name.endswith('.yaml'):
            config_name += '.yaml'
            
        config_path = self.config_dir / config_name
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self.configs[config_name] = config
        return config
    
    def get_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a previously loaded configuration
        
        Args:
            config_name: Name of config file
            
        Returns:
            Configuration dictionary or None if not loaded
        """
        if not config_name.endswith('.yaml'):
            config_name += '.yaml'
            
        return self.configs.get(config_name)
    
    def get_experiment_config(self, experiment_id: str) -> Dict[str, Any]:
        """
        Load experiment configuration by ID
        
        Args:
            experiment_id: Experiment ID (e.g., 'exp1', 'exp2')
            
        Returns:
            Experiment configuration
        """
        config_map = {
            'exp1': 'exp1_discrete_comparison.yaml',
            'exp2': 'exp2_continuous_comparison.yaml',
            'exp3': 'exp3_state_adapter_ablation.yaml',
            'exp4': 'exp4_training_progression.yaml',
            'exp5': 'exp5_per_variant_comparison.yaml',
            'exp6': 'exp6_action_space_analysis.yaml'
        }
        
        config_file = config_map.get(experiment_id)
        if not config_file:
            raise ValueError(f"Unknown experiment ID: {experiment_id}")
            
        return self.load_config(config_file)
    
    def get_model_paths(self) -> Dict[str, Any]:
        """
        Load model paths configuration
        
        Returns:
            Model paths dictionary
        """
        return self.load_config('model_paths.yaml')
    
    def save_config(self, config: Dict[str, Any], config_name: str):
        """
        Save a configuration to file
        
        Args:
            config: Configuration dictionary
            config_name: Name of config file
        """
        if not config_name.endswith('.yaml'):
            config_name += '.yaml'
            
        config_path = self.config_dir / config_name
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
        self.configs[config_name] = config
    
    def get_value(self, config_name: str, *keys) -> Any:
        """
        Get a nested value from configuration
        
        Args:
            config_name: Name of config file
            *keys: Nested keys to access
            
        Returns:
            Value at the specified path
        """
        config = self.get_config(config_name)
        if config is None:
            config = self.load_config(config_name)
            
        value = config
        for key in keys:
            value = value[key]
            
        return value

