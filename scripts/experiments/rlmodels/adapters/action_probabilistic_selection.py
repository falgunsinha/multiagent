"""
Probabilistic Action Adapter
Uses softmax over action logits for cube selection
"""

import numpy as np


class ProbabilisticActionAdapter:
    """Probabilistic action selection using softmax"""
    
    def __init__(self, num_cubes: int = 9, temperature: float = 1.0):
        """
        Initialize Probabilistic Action Adapter
        
        Args:
            num_cubes: Number of cubes
            temperature: Softmax temperature (lower = more deterministic)
        """
        self.num_cubes = num_cubes
        self.temperature = temperature
        
    def map_action(self, action_logits: np.ndarray) -> int:
        """
        Map action logits to cube selection using softmax
        
        Args:
            action_logits: Action logits or Q-values [num_cubes]
            
        Returns:
            Selected cube index
        """
        # Apply softmax with temperature
        logits = action_logits / self.temperature
        probs = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        probs = probs / np.sum(probs)
        
        # Sample from distribution
        selected_cube = np.random.choice(self.num_cubes, p=probs)
        
        return int(selected_cube)
    
    def map_action_deterministic(self, action_logits: np.ndarray) -> int:
        """
        Deterministic selection (argmax)
        
        Args:
            action_logits: Action logits or Q-values [num_cubes]
            
        Returns:
            Selected cube index
        """
        return int(np.argmax(action_logits))
    
    def get_probabilities(self, action_logits: np.ndarray) -> np.ndarray:
        """
        Get action probabilities
        
        Args:
            action_logits: Action logits or Q-values [num_cubes]
            
        Returns:
            Action probabilities [num_cubes]
        """
        logits = action_logits / self.temperature
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        
        return probs


if __name__ == "__main__":
    # Test the adapter
    adapter = ProbabilisticActionAdapter(temperature=0.5)
    
    # Create sample action logits
    action_logits = np.random.randn(9)
    
    print("Action logits:", action_logits)
    print()
    
    # Get probabilities
    probs = adapter.get_probabilities(action_logits)
    print("Probabilities:", probs)
    print("Sum:", np.sum(probs))
    print()
    
    # Test deterministic
    det_action = adapter.map_action_deterministic(action_logits)
    print(f"Deterministic action: {det_action}")
    print()
    
    # Test probabilistic
    samples = [adapter.map_action(action_logits) for _ in range(1000)]
    unique, counts = np.unique(samples, return_counts=True)
    print("Probabilistic distribution (1000 samples):")
    for cube, count in zip(unique, counts):
        print(f"  Cube {cube}: {count/10:.1f}%")

