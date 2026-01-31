"""
Feature Aggregation State Adapter (24D → 8D)
Semantic feature engineering for state compression
"""

import numpy as np
from typing import Optional


class FeatureAggregationAdapter:
    """
    Semantic feature engineering for state compression
    Transforms 24D Isaac Sim state to 8D LunarLander-like state
    """
    
    def __init__(self, input_dim: int = 24, output_dim: int = 8):
        """
        Initialize Feature Aggregation Adapter
        
        Args:
            input_dim: Input state dimension (24 for Isaac Sim)
            output_dim: Output state dimension (8 for LunarLander)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.last_robot_pos = None
        
    def transform(self, state: np.ndarray) -> np.ndarray:
        """
        Transform 24D Isaac Sim state to 8D LunarLander-like state
        
        State breakdown (24D):
        - Cube positions: 9 cubes × 2 coords = 18D (indices 0-17)
        - Robot position: 2D (indices 18-19)
        - Target cube index: 1D (index 20)
        - Gripper state: 1D (index 21)
        - Distance to target: 1D (index 22)
        - Collision flag: 1D (index 23)
        
        Output (8D):
        - Robot X, Y position: 2D
        - Velocity proxy (delta from last): 2D
        - Distance to nearest cube: 1D
        - Angle to nearest cube: 1D
        - Gripper state: 1D
        - Collision flag: 1D
        
        Args:
            state: 24D state vector
            
        Returns:
            8D compressed state
        """
        if len(state) != self.input_dim:
            raise ValueError(f"Expected state dimension {self.input_dim}, got {len(state)}")
        
        # Extract components
        cube_positions = state[:18].reshape(9, 2)
        robot_pos = state[18:20]
        gripper = state[21]
        collision = state[23]
        
        # Calculate velocity proxy (delta from last position)
        if self.last_robot_pos is not None:
            velocity = robot_pos - self.last_robot_pos
        else:
            velocity = np.zeros(2)
        self.last_robot_pos = robot_pos.copy()
        
        # Calculate distance to nearest cube
        distances = np.linalg.norm(cube_positions - robot_pos, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_dist = distances[nearest_idx]
        
        # Calculate angle to nearest cube
        delta = cube_positions[nearest_idx] - robot_pos
        angle = np.arctan2(delta[1], delta[0])
        
        # Normalize values to similar ranges as LunarLander
        # Robot position: already in reasonable range (0-1 typically)
        # Velocity: scale to [-1, 1]
        velocity = np.clip(velocity * 10, -1, 1)
        # Distance: normalize to [0, 1]
        nearest_dist = np.clip(nearest_dist / 2.0, 0, 1)
        # Angle: already in [-pi, pi], normalize to [-1, 1]
        angle = angle / np.pi
        
        # Construct 8D state
        compressed_state = np.array([
            robot_pos[0],      # Robot X
            robot_pos[1],      # Robot Y
            velocity[0],       # Velocity X
            velocity[1],       # Velocity Y
            nearest_dist,      # Distance to nearest cube
            angle,             # Angle to nearest cube
            gripper,           # Gripper state
            collision          # Collision flag
        ], dtype=np.float32)
        
        return compressed_state
    
    def reset(self):
        """Reset adapter state (e.g., velocity history)"""
        self.last_robot_pos = None


if __name__ == "__main__":
    # Test the adapter
    adapter = FeatureAggregationAdapter()
    
    # Create sample 24D state
    sample_state = np.random.rand(24)
    
    # Transform
    compressed = adapter.transform(sample_state)
    
    print(f"Input shape: {sample_state.shape}")
    print(f"Output shape: {compressed.shape}")
    print(f"Compressed state: {compressed}")

