"""
Continuous to Discrete Action Adapter
Maps continuous actions to discrete cube selections
"""

import numpy as np


class ContinuousToDiscreteAdapter:
    """Maps continuous actions to discrete cube selections"""

    def __init__(self, continuous_dim: int = 2, num_cubes: int = 9, cube_positions: np.ndarray = None):
        """
        Initialize Continuous to Discrete Adapter

        Args:
            continuous_dim: Dimension of continuous action (2 for x,y)
            num_cubes: Number of discrete cubes
            cube_positions: Optional actual cube positions from environment (N, 2 or 3).
                          If None, creates a default normalized grid based on num_cubes.
        """
        self.continuous_dim = continuous_dim
        self.num_cubes = num_cubes

        # Use provided cube positions or create default grid
        if cube_positions is not None:
            # Use actual cube positions from environment (take only x,y)
            self.cube_positions = self._normalize_positions(cube_positions[:, :2])
        else:
            # Create normalized grid based on num_cubes
            self.cube_positions = self._create_cube_grid(num_cubes)

    def _normalize_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        Normalize cube positions to [-1, 1] range

        Args:
            positions: Cube positions (N, 2) in world coordinates

        Returns:
            Normalized positions (N, 2) in range [-1, 1]
        """
        # Find min/max for normalization
        min_pos = positions.min(axis=0)
        max_pos = positions.max(axis=0)

        # Normalize to [-1, 1]
        range_pos = max_pos - min_pos
        range_pos = np.where(range_pos == 0, 1, range_pos)  # Avoid division by zero
        normalized = 2 * (positions - min_pos) / range_pos - 1

        return normalized

    def _create_cube_grid(self, num_cubes: int) -> np.ndarray:
        """
        Create normalized grid of cube positions (fallback if no positions provided)

        Args:
            num_cubes: Number of cubes to create positions for

        Returns:
            Array of shape [num_cubes, 2] with normalized positions in [-1, 1]
        """
        # Determine grid size (e.g., 9 cubes -> 3x3, 16 cubes -> 4x4)
        grid_size = int(np.ceil(np.sqrt(num_cubes)))

        positions = []
        cube_count = 0

        # Create grid positions normalized to [-1, 1]
        for row in range(grid_size):
            for col in range(grid_size):
                if cube_count >= num_cubes:
                    break
                # Normalize to [-1, 1] range
                x = (col / (grid_size - 1)) * 2 - 1 if grid_size > 1 else 0
                y = (row / (grid_size - 1)) * 2 - 1 if grid_size > 1 else 0
                positions.append([x, y])
                cube_count += 1
            if cube_count >= num_cubes:
                break

        return np.array(positions)
    
    def map_action(self, action: np.ndarray) -> int:
        """
        Map continuous action to nearest cube
        
        Args:
            action: Continuous action [x, y] in range [-1, 1]
            
        Returns:
            Cube index (0-8)
        """
        # Clip action to valid range
        action = np.clip(action, -1, 1)
        
        # Find nearest cube
        distances = np.linalg.norm(self.cube_positions - action[:2], axis=1)
        cube_idx = np.argmin(distances)
        
        return int(cube_idx)
    
    def map_action_probabilistic(self, action: np.ndarray, temperature: float = 1.0) -> int:
        """
        Map continuous action probabilistically using softmax
        
        Args:
            action: Continuous action [x, y]
            temperature: Softmax temperature (lower = more deterministic)
            
        Returns:
            Cube index (0-8)
        """
        # Clip action
        action = np.clip(action, -1, 1)
        
        # Calculate distances
        distances = np.linalg.norm(self.cube_positions - action[:2], axis=1)
        
        # Convert to probabilities (closer = higher probability)
        # Use negative distance for softmax
        logits = -distances / temperature
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        # Sample
        cube_idx = np.random.choice(self.num_cubes, p=probs)
        
        return int(cube_idx)


if __name__ == "__main__":
    # Test the adapter
    adapter = ContinuousToDiscreteAdapter()
    
    print("Cube positions:")
    print(adapter.cube_positions)
    print()
    
    # Test some actions
    test_actions = [
        np.array([0.0, 0.0]),    # Center
        np.array([-0.8, -0.8]),  # Top-left
        np.array([0.8, 0.8]),    # Bottom-right
        np.array([0.5, -0.5]),   # Random
    ]
    
    for action in test_actions:
        cube = adapter.map_action(action)
        print(f"Action {action} → Cube {cube} at {adapter.cube_positions[cube]}")

