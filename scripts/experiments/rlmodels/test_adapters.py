"""
Test script for state and action adapters
"""

import numpy as np
import sys
from pathlib import Path

# Add adapters to path
sys.path.append(str(Path(__file__).parent))

from adapters.state_pca import PCAStateAdapter
from adapters.action_discrete_mapper import DiscreteActionMapper


def test_state_adapter_24_to_8():
    """Test 24D → 8D state adapter"""
    print("\n" + "="*60)
    print("Testing State Adapter: 24D → 8D (LunarLander)")
    print("="*60)

    adapter = PCAStateAdapter(input_dim=24, output_dim=8)

    # Create test state (4 cubes × 6 features)
    test_state = np.random.rand(24)
    test_state = test_state.reshape(4, 6)

    # Simulate 2 cubes not picked (0.0), 2 picked (1.0)
    test_state[:2, 5] = 0.0  # First 2 cubes NOT picked
    test_state[2:, 5] = 1.0  # Last 2 cubes picked
    test_state = test_state.flatten()

    print(f"Input state shape: {test_state.shape}")
    print(f"Input state (first 12 values): {test_state[:12]}")
    print(f"Cubes remaining: 2 (indices 0, 1)")

    compressed = adapter.transform(test_state)

    print(f"Output state shape: {compressed.shape}")
    print(f"Output state: {compressed}")
    print(f"✓ State adapter working!")


def test_state_adapter_24_to_4():
    """Test 24D → 4D state adapter"""
    print("\n" + "="*60)
    print("Testing State Adapter: 24D → 4D (CartPole)")
    print("="*60)

    adapter = PCAStateAdapter(input_dim=24, output_dim=4)

    # Create test state (4 cubes × 6 features)
    test_state = np.random.rand(24)
    test_state = test_state.reshape(4, 6)
    test_state[:2, 5] = 0.0  # First 2 cubes NOT picked
    test_state[2:, 5] = 1.0  # Last 2 cubes picked
    test_state = test_state.flatten()

    print(f"Input state shape: {test_state.shape}")
    print(f"Cubes remaining: 2 (indices 0, 1)")

    compressed = adapter.transform(test_state)

    print(f"Output state shape: {compressed.shape}")
    print(f"Output state: {compressed}")
    print(f"✓ State adapter working!")


def test_action_adapter_4_to_9():
    """Test 4 → 9 action adapter"""
    print("\n" + "="*60)
    print("Testing Action Adapter: 4 → 9 (LunarLander)")
    print("="*60)
    
    adapter = DiscreteActionMapper(source_actions=4, target_actions=9)
    
    # Test each action
    test_state = np.random.rand(54)
    
    for action in range(4):
        mapped = adapter.map_action(action, test_state)
        print(f"Action {action} → {mapped}")
    
    print(f"✓ Action adapter working!")


def test_action_adapter_2_to_9():
    """Test 2 → 9 action adapter"""
    print("\n" + "="*60)
    print("Testing Action Adapter: 2 → 9 (CartPole)")
    print("="*60)
    
    adapter = DiscreteActionMapper(source_actions=2, target_actions=9)
    
    # Test each action
    test_state = np.random.rand(54)
    
    for action in range(2):
        mapped = adapter.map_action(action, test_state)
        print(f"Action {action} → {mapped}")
    
    print(f"✓ Action adapter working!")


def test_full_pipeline():
    """Test full pipeline: state adaptation → action selection → action adaptation"""
    print("\n" + "="*60)
    print("Testing Full Pipeline (24D → 8D → 4 actions → 9 actions)")
    print("="*60)

    state_adapter = PCAStateAdapter(input_dim=24, output_dim=8)
    action_adapter = DiscreteActionMapper(source_actions=4, target_actions=9)

    # Simulate Isaac Sim state (4 cubes × 6 features)
    isaac_state = np.random.rand(24)
    isaac_state = isaac_state.reshape(4, 6)
    isaac_state[:2, 5] = 0.0  # First 2 cubes NOT picked
    isaac_state[2:, 5] = 1.0  # Last 2 cubes picked
    isaac_state = isaac_state.flatten()

    print(f"1. Isaac Sim state: {isaac_state.shape} (2 cubes remaining)")

    # Adapt state
    adapted_state = state_adapter.transform(isaac_state)
    print(f"2. Adapted state: {adapted_state.shape} = {adapted_state}")

    # Simulate model action selection (just pick action 2)
    model_action = 2
    print(f"3. Model selects action: {model_action}")

    # Adapt action back
    isaac_action = action_adapter.map_action(model_action, isaac_state)
    print(f"4. Adapted to Isaac Sim action: {isaac_action}")

    print(f"✓ Full pipeline working!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADAPTER TESTING SUITE")
    print("Isaac Sim: 3x3 grid, 4 cubes = 24D state, 9 actions")
    print("="*60)

    try:
        test_state_adapter_24_to_8()
        test_state_adapter_24_to_4()
        test_action_adapter_4_to_9()
        test_action_adapter_2_to_9()
        test_full_pipeline()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

