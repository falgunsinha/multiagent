"""
Checkpoint Manager for finding and loading model checkpoints
"""

from pathlib import Path
from typing import List, Dict, Optional
import re


class CheckpointManager:
    """Manages model checkpoints and training progression"""
    
    def __init__(self, models_dir: str = "cobotproject/scripts/Reinforcement Learning/DRL-Pytorch"):
        """
        Initialize CheckpointManager
        
        Args:
            models_dir: Base directory for pretrained models
        """
        self.models_dir = Path(models_dir)
        
    def find_checkpoints(self, model_dir: str, pattern: str = "*.pth") -> List[Path]:
        """
        Find all checkpoint files in a model directory
        
        Args:
            model_dir: Model directory (relative to models_dir)
            pattern: File pattern to match
            
        Returns:
            List of checkpoint file paths
        """
        full_dir = self.models_dir / model_dir
        
        if not full_dir.exists():
            return []
            
        # Look in both model/ and checkpoints/ subdirectories
        checkpoints = []
        
        for subdir in ['model', 'checkpoints', '.']:
            search_dir = full_dir / subdir
            if search_dir.exists():
                checkpoints.extend(search_dir.glob(pattern))
                
        return sorted(checkpoints)
    
    def get_checkpoint_by_episode(self, model_dir: str, episode: int) -> Optional[Path]:
        """
        Get checkpoint for a specific episode number
        
        Args:
            model_dir: Model directory
            episode: Episode number
            
        Returns:
            Path to checkpoint or None if not found
        """
        checkpoints = self.find_checkpoints(model_dir)
        
        for ckpt in checkpoints:
            # Try to extract episode number from filename
            match = re.search(r'_(\d+)\.pth', ckpt.name)
            if match and int(match.group(1)) == episode:
                return ckpt
                
        return None
    
    def get_checkpoint_progression(self, model_dir: str, 
                                  episodes: List[int]) -> Dict[int, Optional[Path]]:
        """
        Get checkpoints for multiple episode numbers
        
        Args:
            model_dir: Model directory
            episodes: List of episode numbers
            
        Returns:
            Dictionary mapping episode numbers to checkpoint paths
        """
        result = {}
        
        for ep in episodes:
            result[ep] = self.get_checkpoint_by_episode(model_dir, ep)
            
        return result
    
    def list_available_checkpoints(self, model_dir: str) -> List[Dict[str, any]]:
        """
        List all available checkpoints with metadata
        
        Args:
            model_dir: Model directory
            
        Returns:
            List of dictionaries with checkpoint info
        """
        checkpoints = self.find_checkpoints(model_dir)
        
        result = []
        for ckpt in checkpoints:
            info = {
                'path': ckpt,
                'name': ckpt.name,
                'size': ckpt.stat().st_size,
            }
            
            # Try to extract episode/step number
            match = re.search(r'_(\d+)\.pth', ckpt.name)
            if match:
                info['episode'] = int(match.group(1))
            
            # Try to extract step number
            match = re.search(r'(\d+)k', ckpt.name)
            if match:
                info['steps'] = int(match.group(1)) * 1000
                
            result.append(info)
            
        return result
    
    def get_latest_checkpoint(self, model_dir: str) -> Optional[Path]:
        """
        Get the most recent checkpoint
        
        Args:
            model_dir: Model directory
            
        Returns:
            Path to latest checkpoint or None
        """
        checkpoints = self.find_checkpoints(model_dir)
        
        if not checkpoints:
            return None
            
        # Sort by modification time
        return max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    def validate_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Validate that a checkpoint file exists and is readable
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            True if valid, False otherwise
        """
        if not checkpoint_path.exists():
            return False
            
        if not checkpoint_path.is_file():
            return False
            
        if checkpoint_path.stat().st_size == 0:
            return False
            
        return True
    
    def get_checkpoint_info(self, checkpoint_path: Path) -> Dict[str, any]:
        """
        Get information about a checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Dictionary with checkpoint metadata
        """
        if not self.validate_checkpoint(checkpoint_path):
            return {'valid': False}
            
        return {
            'valid': True,
            'path': str(checkpoint_path),
            'name': checkpoint_path.name,
            'size_mb': checkpoint_path.stat().st_size / (1024 * 1024),
            'modified': checkpoint_path.stat().st_mtime
        }

