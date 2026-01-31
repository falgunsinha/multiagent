"""
Discrete Action Mapper (4 actions → 9 cube selections)
Maps LunarLander actions to object selection
"""

import numpy as np


class DiscreteActionMapper:
    """Maps 4 discrete actions to 9 cube selections"""
    
    def __init__(self, source_actions: int = 4, target_actions: int = 9):
        """
        Initialize Discrete Action Mapper
        
        Args:
            source_actions: Number of source actions (4 for LunarLander)
            target_actions: Number of target actions (9 for cubes)
        """
        self.source_actions = source_actions
        self.target_actions = target_actions
        
        # Create mapping matrix
        self.mapping = self._create_mapping()
        
    def _create_mapping(self) -> np.ndarray:
        """
        Create action mapping matrix

        Strategy: Map source actions to target cubes using spatial logic

        3x3 grid layout (9 actions):
        0 1 2
        3 4 5
        6 7 8

        4x4 grid layout (16 actions):
        0  1  2  3
        4  5  6  7
        8  9  10 11
        12 13 14 15

        For 4 actions (LunarLander):
        - Action 0 (nothing/hover): → Center positions
        - Action 1 (left engine): → Left columns
        - Action 2 (main engine/down): → Center columns
        - Action 3 (right engine): → Right columns

        For 2 actions (CartPole):
        - Action 0 (left): → Left half
        - Action 1 (right): → Right half

        Returns:
            Mapping matrix [source_actions x target_actions]
        """
        mapping = np.zeros((self.source_actions, self.target_actions))

        if self.source_actions == 4 and self.target_actions == 9:
            # LunarLander → 3x3 grid mapping
            mapping[0, 4] = 1.0  # Action 0 → Center
            mapping[1, [0, 3, 6]] = [0.2, 0.5, 0.3]  # Action 1 → Left column
            mapping[2, [1, 4, 7]] = [0.3, 0.4, 0.3]  # Action 2 → Center column
            mapping[3, [2, 5, 8]] = [0.3, 0.5, 0.2]  # Action 3 → Right column

        elif self.source_actions == 4 and self.target_actions == 16:
            # LunarLander → 4x4 grid mapping
            mapping[0, [5, 6, 9, 10]] = [0.25, 0.25, 0.25, 0.25]  # Action 0 → Center 4 positions
            mapping[1, [0, 4, 8, 12]] = [0.2, 0.3, 0.3, 0.2]  # Action 1 → Left column
            mapping[2, [1, 2, 5, 6, 9, 10, 13, 14]] = [0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1]  # Action 2 → Center columns
            mapping[3, [3, 7, 11, 15]] = [0.2, 0.3, 0.3, 0.2]  # Action 3 → Right column

        elif self.source_actions == 2 and self.target_actions == 9:
            # CartPole → 3x3 grid mapping
            mapping[0, [0, 1, 3, 4, 6, 7]] = [0.15, 0.2, 0.2, 0.25, 0.1, 0.1]  # Action 0 → Left half
            mapping[1, [2, 4, 5, 7, 8]] = [0.2, 0.2, 0.25, 0.15, 0.2]  # Action 1 → Right half

        elif self.source_actions == 2 and self.target_actions == 16:
            # CartPole → 4x4 grid mapping
            mapping[0, [0, 1, 4, 5, 8, 9, 12, 13]] = [0.1, 0.15, 0.15, 0.2, 0.15, 0.15, 0.05, 0.05]  # Action 0 → Left half
            mapping[1, [2, 3, 6, 7, 10, 11, 14, 15]] = [0.15, 0.1, 0.2, 0.15, 0.15, 0.15, 0.05, 0.05]  # Action 1 → Right half

        else:
            # Fallback: uniform distribution
            mapping[:, :] = 1.0 / self.target_actions

        return mapping
        
    def map_action(self, action: int, state: np.ndarray = None, action_mask: np.ndarray = None) -> int:
        """
        Map source action to target action

        Args:
            action: Source action (0-3)
            state: Current state (optional, for context-aware mapping)
            action_mask: Boolean mask of valid actions (optional, for safety)

        Returns:
            Target action (0-8)
        """
        # Ensure action is in valid range
        if action >= self.source_actions:
            action = action % self.source_actions

        # Get probabilities for this action
        probs = self.mapping[action].copy()

        # CRITICAL FIX: Apply action mask to filter out invalid actions
        if action_mask is not None:
            # Zero out probabilities for invalid actions
            probs = probs * action_mask[:self.target_actions]

            # Renormalize probabilities
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                # No valid actions in this mapping, fallback to any valid action
                probs = action_mask[:self.target_actions].astype(float)
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                else:
                    # No valid actions at all, return 0 (will be caught by safety check)
                    return 0

        # Sample from distribution
        target_action = np.random.choice(self.target_actions, p=probs)

        return target_action
    
    def map_action_deterministic(self, action: int) -> int:
        """
        Map action deterministically (argmax)
        
        Args:
            action: Source action (0-3)
            
        Returns:
            Target action (0-8)
        """
        if action >= self.source_actions:
            action = action % self.source_actions
        
        probs = self.mapping[action]
        return np.argmax(probs)


if __name__ == "__main__":
    # Test the mapper
    mapper = DiscreteActionMapper()
    
    print("Action Mapping Matrix:")
    print(mapper.mapping)
    print()
    
    # Test each action
    for action in range(4):
        samples = [mapper.map_action(action) for _ in range(100)]
        print(f"Action {action} → Cubes: {np.unique(samples, return_counts=True)}")

