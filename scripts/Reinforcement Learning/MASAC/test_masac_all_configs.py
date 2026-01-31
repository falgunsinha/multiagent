"""
Test MASAC on all 9 configurations (3 environments × 3 cube counts)
Tests for 10 episodes each and logs results locally (CSV + JSON)
"""

import sys
from pathlib import Path
import numpy as np
import json
import csv
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Add MASAC and MAPPO to path
masac_path = Path(__file__).parent
sys.path.insert(0, str(masac_path))
mappo_path = Path(__file__).parent.parent / "MAPPO"
sys.path.insert(0, str(mappo_path))

from src.rl.object_selection_env_astar import ObjectSelectionEnvAStar
from src.rl.object_selection_env_rrt_viz import ObjectSelectionEnvRRTViz
from src.rl.doubleDQN import DoubleDQNAgent
from agents.masac_continuous_wrapper import MASACContinuousWrapper

# Import TwoAgentEnv from MAPPO
sys.path.insert(0, str(Path(__file__).parent.parent / "MAPPO"))
from envs.two_agent_env import TwoAgentEnv


def load_ddqn_agent(model_path: str, env) -> DoubleDQNAgent:
    """Load pretrained DDQN agent"""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DoubleDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=0.0,  # No exploration during testing
        epsilon_end=0.0,
        epsilon_decay=1.0,
        batch_size=64,
        buffer_capacity=10000
    )

    agent.load(model_path)
    agent.epsilon = 0.0  # Ensure no exploration
    print(f"✅ Loaded DDQN model: {model_path}")
    return agent


def test_masac_configuration(
    env_type: str,
    grid_size: int,
    num_cubes: int,
    num_episodes: int = 10,
    log_dir: str = "cobotproject/scripts/Reinforcement Learning/MASAC/logs"
):
    """
    Test MASAC on a single configuration.

    Args:
        env_type: 'astar', 'rrt_viz', or 'rrt_isaacsim'
        grid_size: Grid size (3 or 4)
        num_cubes: Number of cubes (4, 6, or 9)
        num_episodes: Number of test episodes
        log_dir: Directory to save logs
    """
    print(f"\n{'='*80}")
    print(f"Testing MASAC: {env_type.upper()} | Grid {grid_size}x{grid_size} | {num_cubes} cubes")
    print(f"{'='*80}")

    # Skip Isaac Sim RRT in this script (handled by test_masac_isaacsim.py)
    if env_type == 'rrt_isaacsim':
        print(f"⚠️  Skipping {env_type} - handled by test_masac_isaacsim.py")
        return None

    # Calculate max_objects based on grid_size (same as MAPPO)
    max_objects = grid_size * grid_size

    # Create base environment
    if env_type == 'astar':
        base_env = ObjectSelectionEnvAStar(
            franka_controller=None,
            max_objects=max_objects,
            max_steps=50,
            num_cubes=num_cubes,
            training_grid_size=grid_size
        )
        cube_spacing = 0.26 if grid_size > 3 else 0.28
    elif env_type == 'rrt_viz':
        base_env = ObjectSelectionEnvRRTViz(
            franka_controller=None,
            max_objects=max_objects,
            max_steps=50,
            num_cubes=num_cubes,
            training_grid_size=grid_size
        )
        cube_spacing = 0.20 if grid_size > 3 else 0.22
    else:
        raise ValueError(f"Unknown env_type: {env_type}")
    
    # Load DDQN agent - use exact paths from MAPPO training
    ddqn_model_mapping = {
        # Grid 3x3, 4 cubes
        ('astar', 3, 4): 'ddqn_astar_grid3_cubes4_20251220_012015_final.pt',
        ('rrt_viz', 3, 4): 'ddqn_rrt_viz_grid3_cubes4_20251220_025425_final.pt',
        ('rrt_isaacsim', 3, 4): 'ddqn_rrt_isaacsim_grid3_cubes4_20251223_203144_final.pt',
        # Grid 4x4, 6 cubes
        ('astar', 4, 6): 'ddqn_astar_grid4_cubes6_20251220_014823_final.pt',
        ('rrt_viz', 4, 6): 'ddqn_rrt_viz_grid4_cubes6_20251220_054851_final.pt',
        ('rrt_isaacsim', 4, 6): 'ddqn_rrt_isaacsim_grid4_cubes6_20251224_122040_final.pt',
        # Grid 4x4, 9 cubes
        ('astar', 4, 9): 'ddqn_astar_grid4_cubes9_20251220_022000_final.pt',
        ('rrt_viz', 4, 9): 'ddqn_rrt_viz_grid4_cubes9_20251220_134808_final.pt',
        ('rrt_isaacsim', 4, 9): 'ddqn_rrt_isaacsim_grid4_cubes9_20251224_185752_final.pt',
    }

    config_key = (env_type, grid_size, num_cubes)
    if config_key not in ddqn_model_mapping:
        print(f"⚠️  No DDQN model mapping for {env_type} grid{grid_size} cubes{num_cubes}")
        return None

    ddqn_model_filename = ddqn_model_mapping[config_key]
    # Use absolute path from project root
    ddqn_models_dir = project_root / "scripts" / "Reinforcement Learning" / "doubleDQN_script" / "models"
    ddqn_model_path = ddqn_models_dir / ddqn_model_filename

    if not ddqn_model_path.exists():
        print(f"⚠️  DDQN model not found: {ddqn_model_path}")
        return None

    ddqn_agent = load_ddqn_agent(str(ddqn_model_path), base_env)

    # Calculate Agent 2 observation dimension
    # Agent 2 obs = cube_features + robot_pos + picked_mask + grid_distances + decision_features
    # = (num_cubes * 3) + 3 + num_cubes + (grid_size * grid_size) + 10
    agent2_state_dim = (num_cubes * 3) + 3 + num_cubes + (grid_size * grid_size) + 10

    # Create MASAC wrapper with pretrained Tennis models and dimension adapter
    # Tennis environment: state_dim=24, action_dim=2
    # Agent 2 environment: state_dim varies (38, 53, 65, etc.), action_dim=3
    # Use PCA to map between dimensions
    pretrained_path = project_root / "scripts" / "Reinforcement Learning" / "MASAC" / "pretrained_models"

    masac_agent = MASACContinuousWrapper(
        state_dim=agent2_state_dim,  # Agent 2 observation dimension
        grid_size=grid_size,
        num_cubes=num_cubes,
        cube_spacing=cube_spacing,
        pretrained_model_path=str(pretrained_path),  # Use Tennis pretrained models
        use_dimension_adapter=True,  # Enable dimension adaptation
        memory_size=10000,
        batch_size=64
    )

    # Create two-agent environment
    two_agent_env = TwoAgentEnv(
        base_env=base_env,
        ddqn_agent=ddqn_agent,
        grid_size=grid_size,
        num_cubes=num_cubes,
        max_reshuffles_per_episode=5,
        reshuffle_reward_scale=1.0,
        max_episode_steps=50,
        verbose=False  # Disable verbose logging during testing
    )

    # Fit PCA dimension adapter on sample states (silently)
    masac_agent.fit_dimension_adapter(two_agent_env, n_samples=500)

    # Set test mode
    masac_agent.set_test_mode(True)
    
    # Test for num_episodes
    episode_results = []

    for episode in range(num_episodes):
        obs, reset_info = two_agent_env.reset()  # TwoAgentEnv returns (obs, info)
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        reshuffles_performed = 0

        while not (done or truncated) and episode_length < 50:
            # Calculate valid cubes for action masking (not picked, not reshuffled 2+ times)
            valid_cubes = [
                i for i in range(num_cubes)
                if i not in two_agent_env.base_env.objects_picked
                and two_agent_env.reshuffle_count_per_cube.get(i, 0) < 2
            ]

            # MASAC selects reshuffling action with action masking
            action_dict = masac_agent.select_action(obs, deterministic=True, valid_cubes=valid_cubes)

            # Skip if no valid action
            if action_dict is None:
                print(f"  [WARNING] No valid cubes to reshuffle, skipping step")
                break

            # Execute action in environment
            next_obs, reward, done, truncated, info = two_agent_env.step(action_dict)  # 5 return values
            
            episode_reward += reward
            episode_length += 1

            if info.get('reshuffled_this_step', False):
                reshuffles_performed += 1

            obs = next_obs
        
        # Log episode results
        result = {
            'episode': episode + 1,
            'total_reward': float(episode_reward),
            'episode_length': episode_length,
            'reshuffles_performed': reshuffles_performed,
            'total_distance_reduced': float(two_agent_env.total_distance_reduced),
            'total_time_saved': float(two_agent_env.total_time_saved),
            'cubes_picked': len(two_agent_env.base_env.objects_picked)
        }
        episode_results.append(result)
        
        print(f"Episode {episode+1}/{num_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Reshuffles={reshuffles_performed}, "
              f"Distance={two_agent_env.total_distance_reduced:.3f}m, "
              f"Cubes={len(two_agent_env.base_env.objects_picked)}/{num_cubes}")
    
    # Save results
    save_results(env_type, grid_size, num_cubes, episode_results, log_dir)

    return episode_results


def save_results(env_type: str, grid_size: int, num_cubes: int, episode_results: list, log_dir: str):
    """Save test results to CSV and JSON"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"masac_{env_type}_grid{grid_size}_cubes{num_cubes}_{timestamp}"

    # Save CSV
    csv_path = log_dir / f"{base_filename}_episode_log.csv"
    with open(csv_path, 'w', newline='') as f:
        if episode_results:
            writer = csv.DictWriter(f, fieldnames=episode_results[0].keys())
            writer.writeheader()
            writer.writerows(episode_results)

    print(f"✅ Saved CSV log: {csv_path}")

    # Save JSON summary
    summary = {
        'env_type': env_type,
        'grid_size': grid_size,
        'num_cubes': num_cubes,
        'num_episodes': len(episode_results),
        'timestamp': timestamp,
        'avg_reward': np.mean([r['total_reward'] for r in episode_results]),
        'avg_reshuffles': np.mean([r['reshuffles_performed'] for r in episode_results]),
        'avg_distance_reduced': np.mean([r['total_distance_reduced'] for r in episode_results]),
        'avg_time_saved': np.mean([r['total_time_saved'] for r in episode_results]),
        'avg_cubes_picked': np.mean([r['cubes_picked'] for r in episode_results]),
        'episodes': episode_results
    }

    json_path = log_dir / f"{base_filename}_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Saved JSON summary: {json_path}")


def main():
    """Test MASAC on all 9 configurations"""
    parser = argparse.ArgumentParser(description='Test MASAC on all configurations')
    parser.add_argument('--episodes', type=int, default=5, help='Number of test episodes per config')
    parser.add_argument('--log_dir', type=str, default='cobotproject/scripts/Reinforcement Learning/MASAC/logs',
                        help='Directory to save logs')
    args = parser.parse_args()

    # Define all 9 configurations in order: A* → RRT Viz → Isaac Sim RRT
    configurations = [
        # A* (3 configs) - Native Python
        ('astar', 3, 4),
        ('astar', 4, 6),
        ('astar', 4, 9),
        # RRT Viz (3 configs) - Native Python
        ('rrt_viz', 3, 4),
        ('rrt_viz', 4, 6),
        ('rrt_viz', 4, 9),
        # RRT IsaacSim (3 configs) - Isaac Sim Python
        ('rrt_isaacsim', 3, 4),
        ('rrt_isaacsim', 4, 6),
        ('rrt_isaacsim', 4, 9),
    ]

    print(f"\n{'='*80}")
    print(f"MASAC TESTING - ALL 9 CONFIGURATIONS")
    print(f"Order: A* (3) → RRT Viz (3) → Isaac Sim RRT (3)")
    print(f"Episodes per config: {args.episodes}")
    print(f"Log directory: {args.log_dir}")
    print(f"{'='*80}\n")

    all_results = {}

    for env_type, grid_size, num_cubes in configurations:
        try:
            results = test_masac_configuration(
                env_type=env_type,
                grid_size=grid_size,
                num_cubes=num_cubes,
                num_episodes=args.episodes,
                log_dir=args.log_dir
            )

            if results:
                config_key = f"{env_type}_grid{grid_size}_cubes{num_cubes}"
                all_results[config_key] = results

        except Exception as e:
            print(f"❌ Error testing {env_type} grid{grid_size} cubes{num_cubes}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print(f"\n{'='*80}")
    print("MASAC TESTING COMPLETE")
    print(f"{'='*80}")
    print(f"Tested {len(all_results)}/{len(configurations)} configurations successfully")
    print(f"Results saved to: {args.log_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

