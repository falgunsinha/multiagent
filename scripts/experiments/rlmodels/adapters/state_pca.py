"""
PCA State Adapter (NxD → 8D or 4D)
For Isaac Sim grid: input_dim = max_objects × 6 features
Example: 3x3 grid = 9 objects × 6 features = 54D
"""

import numpy as np
from sklearn.decomposition import PCA
import pickle
from pathlib import Path


class PCAStateAdapter:
    """PCA-based state compression"""

    def __init__(self, input_dim: int = 24, output_dim: int = 8):
        """
        Initialize PCA State Adapter

        Args:
            input_dim: Input state dimension (e.g., 54 for 9 objects × 6 features)
            output_dim: Output state dimension (8 for LunarLander, 4 for CartPole)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_objects = input_dim // 6  # Calculate number of objects
        self.pca = PCA(n_components=output_dim)
        self.fitted = False
        self.scaler_mean = None
        self.scaler_std = None
        
    def fit(self, states: np.ndarray):
        """
        Fit PCA on sample states

        Args:
            states: Array of states [N, input_dim]
        """
        # Normalize states first
        self.scaler_mean = np.mean(states, axis=0)
        self.scaler_std = np.std(states, axis=0) + 1e-8

        normalized_states = (states - self.scaler_mean) / self.scaler_std

        # Fit PCA
        self.pca.fit(normalized_states)
        self.fitted = True

        print(f"PCA fitted. Explained variance ratio: {self.pca.explained_variance_ratio_}")
        print(f"Total variance explained: {np.sum(self.pca.explained_variance_ratio_):.2%}")

    def transform(self, state: np.ndarray) -> np.ndarray:
        """
        Transform state using PCA

        Args:
            state: Input state vector (input_dim)

        Returns:
            Compressed state (output_dim)
        """
        if not self.fitted:
            # If not fitted, use simple feature aggregation
            # Reshape to (num_objects, 6 features)
            state_reshaped = state.reshape(self.num_objects, 6)
            # Find cubes not yet picked (picked flag = 0.0)
            has_cube = state_reshaped[:, 5] == 0.0
            cube_features = state_reshaped[has_cube]

            if len(cube_features) > 0:
                if self.output_dim == 8:
                    # For LunarLander (8D)
                    compressed = np.array([
                        np.min(cube_features[:, 0]),  # Min distance to EE
                        np.mean(cube_features[:, 0]), # Mean distance to EE
                        np.min(cube_features[:, 1]),  # Min distance to container
                        np.mean(cube_features[:, 1]), # Mean distance to container
                        np.mean(cube_features[:, 2]), # Mean obstacle score
                        np.sum(cube_features[:, 3]),  # Sum reachability
                        np.mean(cube_features[:, 4]), # Mean path clearance
                        np.sum(has_cube)              # Number of cubes remaining
                    ], dtype=np.float32)
                elif self.output_dim == 4:
                    # For CartPole (4D)
                    compressed = np.array([
                        np.min(cube_features[:, 0]),  # Min distance to EE
                        np.min(cube_features[:, 1]),  # Min distance to container
                        np.mean(cube_features[:, 2]), # Mean obstacle score
                        np.sum(has_cube)              # Number of cubes remaining
                    ], dtype=np.float32)
                elif self.output_dim == 3:
                    # For Pendulum (3D)
                    compressed = np.array([
                        np.min(cube_features[:, 0]),  # Min distance to EE
                        np.min(cube_features[:, 1]),  # Min distance to container
                        np.sum(has_cube)              # Number of cubes remaining
                    ], dtype=np.float32)
                elif self.output_dim == 24:
                    # For BipedalWalker (24D) - use more detailed features
                    # Aggregate features for all cubes (up to 4 cubes × 6 features = 24D)
                    compressed = np.zeros(24, dtype=np.float32)
                    for i, cube_feat in enumerate(cube_features[:4]):  # Max 4 cubes
                        compressed[i*6:(i+1)*6] = cube_feat
                else:
                    # Fallback: simple truncation
                    compressed = state.flatten()[:self.output_dim]
            else:
                # All cubes picked
                compressed = np.zeros(self.output_dim, dtype=np.float32)
            return compressed

        # Normalize
        normalized_state = (state - self.scaler_mean) / self.scaler_std

        # Transform
        compressed = self.pca.transform(normalized_state.reshape(1, -1))[0]

        return compressed.astype(np.float32)
    
    def save(self, path: str):
        """Save fitted PCA model"""
        if not self.fitted:
            raise ValueError("PCA not fitted yet")
        
        save_dict = {
            'pca': self.pca,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"PCA adapter saved to {path}")
    
    def load(self, path: str):
        """Load fitted PCA model"""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.pca = save_dict['pca']
        self.scaler_mean = save_dict['scaler_mean']
        self.scaler_std = save_dict['scaler_std']
        self.input_dim = save_dict['input_dim']
        self.output_dim = save_dict['output_dim']
        self.fitted = True
        
        print(f"PCA adapter loaded from {path}")
    
    def reset(self):
        """Reset adapter state"""
        pass  # PCA is stateless


if __name__ == "__main__":
    # Test the adapter with 24D input (4 cubes × 6 features)
    print("Testing PCA Adapter (24D → 8D)")
    adapter = PCAStateAdapter(input_dim=24, output_dim=8)

    # Test without fitting (uses feature aggregation)
    test_state = np.random.rand(24)
    # Simulate 4 cubes, 2 already picked
    test_state = test_state.reshape(4, 6)
    test_state[2:, 5] = 1.0  # Mark last 2 cubes as picked
    test_state = test_state.flatten()

    compressed = adapter.transform(test_state)

    print(f"Input shape: {test_state.shape}")
    print(f"Output shape: {compressed.shape}")
    print(f"Compressed state: {compressed}")

    # Test with fitting
    print("\nTesting with PCA fitting...")
    sample_states = np.random.rand(1000, 24)
    adapter.fit(sample_states)
    compressed_fitted = adapter.transform(test_state)
    print(f"Compressed state (fitted): {compressed_fitted}")

