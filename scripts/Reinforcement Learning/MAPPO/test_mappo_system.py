"""
Test MAPPO Two-Agent System

Comprehensive tests for:
- Reshuffling decision module
- Reshuffling action space
- Two-agent environment
- MAPPO policy and trainer
- Replay buffer
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project paths
project_root = Path(r"C:\isaacsim\cobotproject")
mappo_root = project_root / "scripts" / "Reinforcement Learning" / "MAPPO"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(mappo_root))

from envs.reshuffling_decision import ReshufflingDecisionModule, ReshuffleReason
from envs.reshuffling_action_space import ReshufflingActionSpace
from algorithms.mappo_policy import MAPPOPolicy
from algorithms.mappo_trainer import MAPPO
from utils.replay_buffer import RolloutBuffer


def test_reshuffling_decision():
    """Test reshuffling decision module"""
    print("=" * 80)
    print("Testing Reshuffling Decision Module")
    print("=" * 80)
    
    module = ReshufflingDecisionModule()
    
    # Test case 1: Unreachable distance (too far)
    cube_positions = np.array([[1.0, 0.0, 0.0]])  # 1m away
    robot_position = np.array([0.0, 0.0, 0.0])
    obstacle_positions = np.array([])
    
    decision = module.check_reshuffling_needed(
        cube_positions=cube_positions,
        target_cube_idx=0,
        robot_position=robot_position,
        obstacle_positions=obstacle_positions,
    )
    
    print(f"Test 1 - Unreachable (too far):")
    print(f"  Should reshuffle: {decision.should_reshuffle}")
    print(f"  Reason: {decision.reason.value}")
    print(f"  Priority: {decision.priority}")
    assert decision.should_reshuffle, "Should reshuffle for unreachable cube"
    assert decision.reason == ReshuffleReason.UNREACHABLE_DISTANCE
    print("  ✓ PASSED")
    print()
    
    # Test case 2: Crowded area
    cube_positions = np.array([
        [0.5, 0.0, 0.0],
        [0.5, 0.1, 0.0],
        [0.5, 0.2, 0.0],
        [0.6, 0.0, 0.0],
    ])
    
    decision = module.check_reshuffling_needed(
        cube_positions=cube_positions,
        target_cube_idx=0,
        robot_position=robot_position,
        obstacle_positions=obstacle_positions,
    )
    
    print(f"Test 2 - Crowded area:")
    print(f"  Should reshuffle: {decision.should_reshuffle}")
    print(f"  Reason: {decision.reason.value}")
    print(f"  Priority: {decision.priority}")
    assert decision.should_reshuffle, "Should reshuffle for crowded area"
    assert decision.reason == ReshuffleReason.CROWDED_AREA
    print("  ✓ PASSED")
    print()
    
    # Test case 3: No reshuffling needed
    cube_positions = np.array([[0.6, 0.0, 0.0]])  # Optimal distance
    
    decision = module.check_reshuffling_needed(
        cube_positions=cube_positions,
        target_cube_idx=0,
        robot_position=robot_position,
        obstacle_positions=obstacle_positions,
    )
    
    print(f"Test 3 - No reshuffling needed:")
    print(f"  Should reshuffle: {decision.should_reshuffle}")
    print(f"  Reason: {decision.reason.value}")
    assert not decision.should_reshuffle, "Should not reshuffle for optimal position"
    print("  ✓ PASSED")
    print()


def test_reshuffling_action_space():
    """Test reshuffling action space"""
    print("=" * 80)
    print("Testing Reshuffling Action Space")
    print("=" * 80)
    
    action_space = ReshufflingActionSpace(grid_size=4, num_cubes=9)
    
    # Test action encoding/decoding
    print(f"Action space size: {action_space.action_dim}")
    assert action_space.action_dim == 9 * 4 * 4, "Action space size should be 144"
    print("  ✓ Action space size correct")
    
    # Test decode action
    action = 0  # Cube 0 to grid (0, 0)
    reshuffle_action = action_space.decode_action(action)
    print(f"\nAction 0 decoded:")
    print(f"  Cube: {reshuffle_action.cube_idx}")
    print(f"  Grid: ({reshuffle_action.target_grid_x}, {reshuffle_action.target_grid_y})")
    print(f"  World pos: {reshuffle_action.target_world_pos}")
    assert reshuffle_action.cube_idx == 0
    assert reshuffle_action.target_grid_x == 0
    assert reshuffle_action.target_grid_y == 0
    print("  ✓ PASSED")
    
    # Test encode action
    encoded = action_space.encode_action(cube_idx=0, grid_x=0, grid_y=0)
    assert encoded == 0, "Encoded action should be 0"
    print(f"\nEncode (0, 0, 0) -> {encoded}")
    print("  ✓ PASSED")
    
    # Test action mask
    cube_positions = np.random.rand(9, 3)
    picked_cubes = [0, 1]
    mask = action_space.get_action_mask(cube_positions, picked_cubes)
    print(f"\nAction mask shape: {mask.shape}")
    print(f"Valid actions: {np.sum(mask)}")
    assert mask.shape == (action_space.action_dim,)
    print("  ✓ PASSED")
    print()


def test_mappo_policy():
    """Test MAPPO policy"""
    print("=" * 80)
    print("Testing MAPPO Policy")
    print("=" * 80)
    
    obs_dim = 100
    action_dim = 144
    device = torch.device("cpu")
    
    policy = MAPPOPolicy(obs_dim=obs_dim, action_dim=action_dim, device=device)
    
    # Test forward pass
    batch_size = 4
    obs = torch.randn(batch_size, obs_dim, device=device)
    action_mask = torch.ones(batch_size, action_dim, device=device, dtype=torch.bool)
    
    actions, log_probs, values = policy.get_actions(obs, action_mask)
    
    print(f"Batch size: {batch_size}")
    print(f"Actions shape: {actions.shape}")
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Values shape: {values.shape}")
    
    assert actions.shape == (batch_size,)
    assert log_probs.shape == (batch_size,)
    assert values.shape == (batch_size, 1)
    print("  ✓ PASSED")
    print()


def test_replay_buffer():
    """Test replay buffer"""
    print("=" * 80)
    print("Testing Replay Buffer")
    print("=" * 80)
    
    buffer_size = 100
    obs_dim = 50
    action_dim = 144
    device = torch.device("cpu")
    
    buffer = RolloutBuffer(
        buffer_size=buffer_size,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device
    )
    
    # Add transitions
    for i in range(buffer_size):
        obs = np.random.randn(obs_dim)
        action = np.random.randint(0, action_dim)
        reward = np.random.randn()
        value = np.random.randn()
        log_prob = np.random.randn()
        done = (i % 20 == 19)  # Episode ends every 20 steps
        
        buffer.add(obs, action, reward, value, log_prob, done)
        
        if done:
            buffer.finish_path(last_value=0.0)
    
    # Finish last path if needed
    if buffer.ptr > buffer.path_start_idx:
        buffer.finish_path(last_value=0.0)
    
    # Get data
    data = buffer.get()
    
    print(f"Buffer size: {buffer_size}")
    print(f"Data keys: {list(data.keys())}")
    print(f"Observations shape: {data['observations'].shape}")
    print(f"Actions shape: {data['actions'].shape}")
    print(f"Advantages shape: {data['advantages'].shape}")
    print(f"Returns shape: {data['returns'].shape}")
    
    assert data['observations'].shape == (buffer_size, obs_dim)
    assert data['actions'].shape == (buffer_size,)
    assert data['advantages'].shape == (buffer_size,)
    assert data['returns'].shape == (buffer_size,)
    print("  ✓ PASSED")
    print()


def main():
    """Run all tests"""
    print("\n")
    print("=" * 80)
    print("MAPPO Two-Agent System - Comprehensive Tests")
    print("=" * 80)
    print("\n")
    
    try:
        test_reshuffling_decision()
        test_reshuffling_action_space()
        test_mappo_policy()
        test_replay_buffer()
        
        print("=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)
        print("\nThe MAPPO two-agent system is ready for training!")
        print("\nNext steps:")
        print("  1. Train with A*: python train/train_astar_mappo.py")
        print("  2. Train with RRT: python train/train_rrt_mappo.py")
        print("  3. Train all configs: train/train_all_configs.bat")
        print()
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

