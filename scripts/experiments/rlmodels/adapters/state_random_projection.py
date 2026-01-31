"""
Random Projection State Adapter (24D → 8D)
"""

import numpy as np
import pickle


class RandomProjectionAdapter:
    """Random projection for state compression"""
    
    def __init__(self, input_dim: int = 24, output_dim: int = 8, seed: int = 42):
        """
        Initialize Random Projection Adapter
        
        Args:
            input_dim: Input state dimension
            output_dim: Output state dimension
            seed: Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed
        
        # Create random projection matrix
        np.random.seed(seed)
        self.projection_matrix = np.random.randn(input_dim, output_dim) / np.sqrt(output_dim)
        
    def transform(self, state: np.ndarray) -> np.ndarray:
        """
        Transform state using random projection
        
        Args:
            state: 24D state vector
            
        Returns:
            8D compressed state
        """
        compressed = state @ self.projection_matrix
        return compressed.astype(np.float32)
    
    def save(self, path: str):
        """Save projection matrix"""
        save_dict = {
            'projection_matrix': self.projection_matrix,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'seed': self.seed
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"Random projection adapter saved to {path}")
    
    def load(self, path: str):
        """Load projection matrix"""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.projection_matrix = save_dict['projection_matrix']
        self.input_dim = save_dict['input_dim']
        self.output_dim = save_dict['output_dim']
        self.seed = save_dict['seed']
        
        print(f"Random projection adapter loaded from {path}")
    
    def reset(self):
        """Reset adapter state"""
        pass  # Random projection is stateless


if __name__ == "__main__":
    # Test the adapter
    adapter = RandomProjectionAdapter()
    
    # Transform a single state
    test_state = np.random.rand(24)
    compressed = adapter.transform(test_state)
    
    print(f"Input shape: {test_state.shape}")
    print(f"Output shape: {compressed.shape}")
    print(f"Compressed state: {compressed}")
    print(f"Projection matrix shape: {adapter.projection_matrix.shape}")

