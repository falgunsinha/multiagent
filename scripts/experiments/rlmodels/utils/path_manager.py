"""
Path Manager for handling file paths and directory creation
"""

from pathlib import Path
from typing import Optional
import os


class PathManager:
    """Manages file paths for models, logs, results, and visualizations"""
    
    def __init__(self, base_dir: str = "cobotproject/scripts/experiments/rlmodels"):
        """
        Initialize PathManager
        
        Args:
            base_dir: Base directory for all experiment files
        """
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "results"
        self.configs_dir = self.base_dir / "configs"
        self.models_dir = Path("cobotproject/scripts/Reinforcement Learning/DRL-Pytorch")
        self.custom_models_dir = Path("cobotproject/scripts/Reinforcement Learning/doubleDQN_script/models")
        
    def get_model_path(self, model_path: str) -> Path:
        """
        Get full path to a pretrained model
        
        Args:
            model_path: Relative path to model from DRL-Pytorch directory
            
        Returns:
            Full path to model file
        """
        # Check if it's a custom model
        if "doubleDQN_script" in model_path:
            return Path(model_path)
        
        # Otherwise it's a DRL-Pytorch model
        full_path = self.models_dir / model_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Model file not found: {full_path}")
            
        return full_path
    
    def get_results_dir(self, experiment_id: str, create: bool = True) -> Path:
        """
        Get results directory for an experiment
        
        Args:
            experiment_id: Experiment ID (e.g., 'exp1')
            create: Whether to create directory if it doesn't exist
            
        Returns:
            Path to results directory
        """
        results_path = self.results_dir / experiment_id
        
        if create:
            results_path.mkdir(parents=True, exist_ok=True)
            
        return results_path
    
    def get_visualization_dir(self, experiment_id: str, create: bool = True) -> Path:
        """
        Get visualization directory for an experiment
        
        Args:
            experiment_id: Experiment ID
            create: Whether to create directory if it doesn't exist
            
        Returns:
            Path to visualization directory
        """
        viz_path = self.get_results_dir(experiment_id, create) / "visualizations"
        
        if create:
            viz_path.mkdir(parents=True, exist_ok=True)
            
        return viz_path
    
    def get_results_file(self, experiment_id: str, filename: str, create_dir: bool = True) -> Path:
        """
        Get path to a results file
        
        Args:
            experiment_id: Experiment ID
            filename: Name of results file
            create_dir: Whether to create parent directory
            
        Returns:
            Path to results file
        """
        results_dir = self.get_results_dir(experiment_id, create=create_dir)
        return results_dir / filename
    
    def get_config_path(self, config_name: str) -> Path:
        """
        Get path to a configuration file
        
        Args:
            config_name: Name of config file
            
        Returns:
            Path to config file
        """
        if not config_name.endswith('.yaml'):
            config_name += '.yaml'
            
        return self.configs_dir / config_name
    
    def ensure_dir(self, path: Path) -> Path:
        """
        Ensure a directory exists
        
        Args:
            path: Directory path
            
        Returns:
            Path to directory
        """
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def list_experiment_results(self) -> list:
        """
        List all experiment result directories
        
        Returns:
            List of experiment IDs with results
        """
        if not self.results_dir.exists():
            return []
            
        return [d.name for d in self.results_dir.iterdir() if d.is_dir()]
    
    def get_checkpoint_dir(self, model_name: str) -> Path:
        """
        Get checkpoint directory for a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to checkpoint directory
        """
        return self.models_dir / model_name / "checkpoints"

