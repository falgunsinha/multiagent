"""
Weighted Action Adapter
Combines distance and reachability for action selection
"""

import numpy as np


class WeightedActionAdapter:
    """Weighted action selection based on distance and reachability"""
    
    def __init__(self, num_cubes: int = 9, distance_weight: float = 0.6, reachability_weight: float = 0.4):
        """
        Initialize Weighted Action Adapter
        
        Args:
            num_cubes: Number of cubes
            distance_weight: Weight for distance component
            reachability_weight: Weight for reachability component
        """
        self.num_cubes = num_cubes
        self.distance_weight = distance_weight
        self.reachability_weight = reachability_weight
        
    def map_action(self, action: int, state: np.ndarray) -> int:
        """
        Map action using weighted combination of distance and reachability
        
        Args:
            action: Original action (can be ignored or used as hint)
            state: Current state containing cube positions and robot position
            
        Returns:
            Selected cube index
        """
        # Extract cube positions and robot position from state
        # Assuming state format: [cube_pos (18D), robot_pos (2D), ...]
        cube_positions = state[:18].reshape(9, 2)
        robot_pos = state[18:20]
        
        # Calculate distances
        distances = np.linalg.norm(cube_positions - robot_pos, axis=1)
        
        # Normalize distances to [0, 1] (closer = higher score)
        max_dist = np.max(distances) + 1e-8
        distance_scores = 1.0 - (distances / max_dist)
        
        # Calculate reachability scores (simple heuristic: prefer cubes in front)
        # Assuming robot faces positive x direction
        delta = cube_positions - robot_pos
        angles = np.arctan2(delta[:, 1], delta[:, 0])
        # Prefer cubes in front (angle close to 0)
        reachability_scores = np.cos(angles) * 0.5 + 0.5  # Map to [0, 1]
        
        # Combine scores
        combined_scores = (
            self.distance_weight * distance_scores +
            self.reachability_weight * reachability_scores
        )
        
        # Select cube with highest score
        selected_cube = np.argmax(combined_scores)
        
        return int(selected_cube)
    
    def map_action_probabilistic(self, action: int, state: np.ndarray, temperature: float = 1.0) -> int:
        """
        Probabilistic selection using softmax over weighted scores
        
        Args:
            action: Original action
            state: Current state
            temperature: Softmax temperature
            
        Returns:
            Selected cube index
        """
        # Extract positions
        cube_positions = state[:18].reshape(9, 2)
        robot_pos = state[18:20]
        
        # Calculate scores (same as deterministic)
        distances = np.linalg.norm(cube_positions - robot_pos, axis=1)
        max_dist = np.max(distances) + 1e-8
        distance_scores = 1.0 - (distances / max_dist)
        
        delta = cube_positions - robot_pos
        angles = np.arctan2(delta[:, 1], delta[:, 0])
        reachability_scores = np.cos(angles) * 0.5 + 0.5
        
        combined_scores = (
            self.distance_weight * distance_scores +
            self.reachability_weight * reachability_scores
        )
        
        # Apply softmax
        logits = combined_scores / temperature
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        # Sample
        selected_cube = np.random.choice(self.num_cubes, p=probs)
        
        return int(selected_cube)


if __name__ == "__main__":
    # Test the adapter
    adapter = WeightedActionAdapter()
    
    # Create sample state
    sample_state = np.random.rand(24)
    
    # Test deterministic selection
    cube = adapter.map_action(0, sample_state)
    print(f"Selected cube (deterministic): {cube}")
    
    # Test probabilistic selection
    samples = [adapter.map_action_probabilistic(0, sample_state) for _ in range(100)]
    unique, counts = np.unique(samples, return_counts=True)
    print(f"Probabilistic distribution: {dict(zip(unique, counts))}")

