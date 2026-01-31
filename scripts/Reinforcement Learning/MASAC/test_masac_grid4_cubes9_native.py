"""
Test MASAC on Grid 4x4, 9 Cubes ONLY (A* and RRT Viz)
Simplified version that tests only one configuration with both planners

Usage:
    python test_masac_grid4_cubes9_native.py --episodes 5
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
from envs.two_agent_env import TwoAgentEnv


def load_ddqn_agent(model_path: str, env) -> DoubleDQNAgent:
    """Load pretrained DDQN agent"""
    agent = DoubleDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device='cuda'
    )
    agent.load(model_path)
    agent.epsilon = 0.01  # Set to test mode
    print(f"✅ Loaded DDQN model: {model_path}")
    return agent


def test_masac_configuration(
    env_type: str,
    grid_size: int,
    num_cubes: int,
    num_episodes: int,
    log_dir: str,
    seed: int = None,
    run_id: int = 1
):
    """
    Test MASAC on a specific configuration

    Args:
        env_type: Environment type ('astar' or 'rrt_viz')
        grid_size: Grid size
        num_cubes: Number of cubes
        num_episodes: Number of test episodes
        log_dir: Directory to save logs
        seed: Random seed for reproducibility
        run_id: Run identifier
    """
    algorithm = "MASAC"
    scenario = f"grid{grid_size}_cubes{num_cubes}_{env_type}"
    planner = "A*" if env_type == "astar" else "RRT Viz"

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)
        print(f"🎲 Random seed set to: {seed}")

    print(f"\n{'='*80}")
    print(f"Testing {algorithm}: {planner} | Grid {grid_size}x{grid_size} | {num_cubes} cubes")
    print(f"Scenario: {scenario} | Seed: {seed} | Run ID: {run_id}")
    print(f"{'='*80}")

    # Create base environment
    if env_type == 'astar':
        base_env = ObjectSelectionEnvAStar(
            max_objects=grid_size * grid_size,
            max_steps=50,
            num_cubes=num_cubes,
            training_grid_size=grid_size
        )
    elif env_type == 'rrt_viz':
        base_env = ObjectSelectionEnvRRTViz(
            max_objects=grid_size * grid_size,
            max_steps=50,
            num_cubes=num_cubes,
            training_grid_size=grid_size
        )
    else:
        raise ValueError(f"Unknown env_type: {env_type}")

    # Load DDQN agent
    config_key = f"{env_type}_grid{grid_size}_cubes{num_cubes}"
    ddqn_model_mapping = {
        'astar_grid4_cubes9': 'ddqn_astar_grid4_cubes9_20251220_022000_final.pt',
        'rrt_viz_grid4_cubes9': 'ddqn_rrt_viz_grid4_cubes9_20251220_134808_final.pt',
    }

    if config_key not in ddqn_model_mapping:
        print(f"⚠️  No DDQN model mapping for {config_key}")
        return None

    ddqn_model_filename = ddqn_model_mapping[config_key]
    ddqn_models_dir = project_root / "scripts" / "Reinforcement Learning" / "doubleDQN_script" / "models"
    ddqn_model_path = ddqn_models_dir / ddqn_model_filename

    if not ddqn_model_path.exists():
        print(f"⚠️  DDQN model not found: {ddqn_model_path}")
        return None

    ddqn_agent = load_ddqn_agent(str(ddqn_model_path), base_env)

    # Calculate Agent 2 observation dimension
    agent2_state_dim = (num_cubes * 3) + 3 + num_cubes + (grid_size * grid_size) + 10

    # Create MASAC wrapper with dimension adapter
    pretrained_path = project_root / "scripts" / "Reinforcement Learning" / "MASAC" / "pretrained_models"
    cube_spacing = 0.13 if grid_size > 3 else 0.15

    masac_agent = MASACContinuousWrapper(
        state_dim=agent2_state_dim,
        grid_size=grid_size,
        num_cubes=num_cubes,
        cube_spacing=cube_spacing,
        pretrained_model_path=str(pretrained_path),
        use_dimension_adapter=True,
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
        verbose=False
    )

    # Relax reshuffling thresholds for testing
    print("[TEST] Relaxing reshuffling thresholds for testing...")
    two_agent_env.reshuffle_decision.min_reachable_distance = 0.30
    two_agent_env.reshuffle_decision.max_reachable_distance = 0.90
    two_agent_env.reshuffle_decision.path_length_ratio_threshold = 1.5
    two_agent_env.reshuffle_decision.crowded_threshold = 2
    two_agent_env.reshuffle_decision.rrt_failure_window = 2
    two_agent_env.reshuffle_decision.min_clearance = 0.35
    two_agent_env.reshuffle_decision.far_cube_ratio = 1.1
    two_agent_env.reshuffle_decision.batch_reshuffle_count = 2
    print("[TEST] Reshuffling thresholds relaxed!")

    # Enable test mode to skip expensive reachability checks during PCA fitting
    print("[TEST] Enabling test mode for fast PCA fitting...")
    base_env.test_mode = True
    print("[TEST] Test mode enabled! Reachability checks will be skipped.")

    # Fit PCA dimension adapter
    masac_agent.fit_dimension_adapter(two_agent_env, n_samples=500)

    # Set test mode
    masac_agent.set_test_mode(True)

    # Test episodes
    episode_results = []
    timestep_results = []  # NEW: Track timestep-level data
    global_timestep = 0  # NEW: Global timestep counter across all episodes

    for episode in range(num_episodes):
        obs, _ = two_agent_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        reshuffles_performed = 0

        while not (done or truncated) and episode_length < 50:
            # Calculate valid cubes for action masking
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

            # Convert dictionary action to integer action for TwoAgentEnv
            action_int = two_agent_env.reshuffle_action_space.encode_action(
                cube_idx=action_dict['cube_idx'],
                grid_x=action_dict['target_grid_x'],
                grid_y=action_dict['target_grid_y']
            )

            # Execute action in environment
            next_obs, reward, done, truncated, info = two_agent_env.step(action_int)

            episode_reward += reward
            episode_length += 1
            global_timestep += 1

            if info.get('reshuffled_this_step', False):
                reshuffles_performed += 1

            # Log timestep-level data with algorithm/scenario metadata
            timestep_data = {
                'global_timestep': global_timestep,
                'episode': episode + 1,
                'step_in_episode': episode_length,
                'reward': float(reward),
                'cumulative_reward': float(episode_reward),
                'reshuffled': info.get('reshuffled_this_step', False),
                'distance_reduced': float(two_agent_env.total_distance_reduced),
                'time_saved': float(two_agent_env.total_time_saved),
                'cubes_picked': len(two_agent_env.base_env.objects_picked),
                'done': done,
                'truncated': truncated,
                'algorithm': algorithm,
                'scenario': scenario,
                'planner': planner,
                'seed': seed if seed is not None else -1,
                'run_id': run_id
            }
            timestep_results.append(timestep_data)

            obs = next_obs

        # Log episode results with success ratio and metadata
        cubes_picked_count = len(two_agent_env.base_env.objects_picked)
        success_ratio = cubes_picked_count / num_cubes  # Success ratio (e.g., 8/9 = 0.89)

        result = {
            'episode': episode + 1,
            'total_reward': float(episode_reward),
            'episode_length': episode_length,
            'reshuffles_performed': reshuffles_performed,
            'total_distance_reduced': float(two_agent_env.total_distance_reduced),
            'total_time_saved': float(two_agent_env.total_time_saved),
            'cubes_picked': cubes_picked_count,
            'success': success_ratio,  # Changed to ratio instead of boolean
            'algorithm': algorithm,
            'scenario': scenario,
            'planner': planner,
            'seed': seed if seed is not None else -1,
            'run_id': run_id
        }
        episode_results.append(result)

        print(f"Episode {episode+1}/{num_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Reshuffles={reshuffles_performed}, "
              f"Distance={two_agent_env.total_distance_reduced:.3f}m, "
              f"Cubes={len(two_agent_env.base_env.objects_picked)}/{num_cubes}")

    # Save results (both episode and timestep level)
    save_results(
        env_type=env_type,
        grid_size=grid_size,
        num_cubes=num_cubes,
        episode_results=episode_results,
        timestep_results=timestep_results,
        log_dir=log_dir,
        algorithm=algorithm,
        scenario=scenario,
        planner=planner,
        seed=seed,
        run_id=run_id
    )

    return episode_results


def save_results(env_type: str, grid_size: int, num_cubes: int, episode_results: list, timestep_results: list,
                 log_dir: str, algorithm: str = "MASAC", scenario: str = None, planner: str = None,
                 seed: int = None, run_id: int = 1):
    """
    Save test results to CSV and JSON with MAPPO-style formatting

    Args:
        env_type: Environment type (e.g., 'astar', 'rrt_viz')
        grid_size: Grid size
        num_cubes: Number of cubes
        episode_results: List of episode-level results
        timestep_results: List of timestep-level results
        log_dir: Directory to save logs
        algorithm: Algorithm name (e.g., 'MASAC', 'MAPPO')
        scenario: Scenario name (e.g., 'grid4_cubes9_astar')
        planner: Planner name (e.g., 'A*', 'RRT Viz')
        seed: Random seed
        run_id: Run identifier
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Save Episode-level CSV
    if not episode_results:
        print("⚠️  Warning: No episode results to save!")
        return

    csv_filename = f"masac_{env_type}_grid{grid_size}_cubes{num_cubes}_{timestamp}_episode_log.csv"
    csv_path = log_path / csv_filename

    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=episode_results[0].keys())
            writer.writeheader()
            writer.writerows(episode_results)
        print(f"✅ Saved episode CSV: {csv_path.name}")
    except Exception as e:
        print(f"❌ Error saving episode CSV: {e}")

    # Save Timestep-level CSV
    if timestep_results:
        timestep_csv_filename = f"masac_{env_type}_grid{grid_size}_cubes{num_cubes}_{timestamp}_timestep_log.csv"
        timestep_csv_path = log_path / timestep_csv_filename

        try:
            with open(timestep_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=timestep_results[0].keys())
                writer.writeheader()
                writer.writerows(timestep_results)
            print(f"✅ Saved timestep CSV: {timestep_csv_path.name}")
        except Exception as e:
            print(f"❌ Error saving timestep CSV: {e}")
    else:
        print("⚠️  Warning: No timestep results to save!")

    # Calculate statistics with variance
    rewards = [r['total_reward'] for r in episode_results]
    reshuffles = [r['reshuffles_performed'] for r in episode_results]
    distances = [r['total_distance_reduced'] for r in episode_results]
    times = [r['total_time_saved'] for r in episode_results]
    episode_lengths = [r['episode_length'] for r in episode_results]
    cubes_picked = [r['cubes_picked'] for r in episode_results]
    success_ratios = [r['success'] for r in episode_results]

    # Calculate average success rate (now it's a ratio, not boolean)
    avg_success_ratio = np.mean(success_ratios) if success_ratios else 0.0
    success_rate_pct = avg_success_ratio * 100  # Convert to percentage

    # Save summary JSON with mean, std, variance, median, min, max (MAPPO-style)
    summary = {
        # Metadata
        'algorithm': algorithm,
        'scenario': scenario if scenario else f"grid{grid_size}_cubes{num_cubes}_{env_type}",
        'planner': planner if planner else env_type,
        'env_type': env_type,
        'grid_size': grid_size,
        'num_cubes': num_cubes,
        'seed': seed if seed is not None else -1,
        'run_id': run_id,
        'num_episodes': len(episode_results),
        'total_timesteps': len(timestep_results),
        'timestamp': timestamp,

        # Reward statistics
        'reward': {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'variance': float(np.var(rewards)),
            'median': float(np.median(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards)),
        },

        # Success rate statistics (NEW - for MAPPO-style tables)
        'success_rate': {
            'mean': float(avg_success_ratio),
            'percentage': float(success_rate_pct),
            'std': float(np.std([float(s) * 100 for s in success_ratios])),
        },

        # Reshuffles statistics
        'reshuffles': {
            'mean': float(np.mean(reshuffles)),
            'std': float(np.std(reshuffles)),
            'variance': float(np.var(reshuffles)),
            'median': float(np.median(reshuffles)),
            'min': int(np.min(reshuffles)),
            'max': int(np.max(reshuffles)),
        },

        # Distance reduced statistics (meters)
        'distance_reduced_m': {
            'mean': float(np.mean(distances)),
            'std': float(np.std(distances)),
            'variance': float(np.var(distances)),
            'median': float(np.median(distances)),
            'min': float(np.min(distances)),
            'max': float(np.max(distances)),
        },

        # Time saved statistics (seconds)
        'time_saved_s': {
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'variance': float(np.var(times)),
            'median': float(np.median(times)),
            'min': float(np.min(times)),
            'max': float(np.max(times)),
        },

        # Episode length statistics
        'episode_length': {
            'mean': float(np.mean(episode_lengths)),
            'std': float(np.std(episode_lengths)),
            'variance': float(np.var(episode_lengths)),
            'median': float(np.median(episode_lengths)),
            'min': int(np.min(episode_lengths)),
            'max': int(np.max(episode_lengths)),
        },

        # Cubes picked statistics
        'cubes_picked': {
            'mean': float(np.mean(cubes_picked)),
            'std': float(np.std(cubes_picked)),
            'variance': float(np.var(cubes_picked)),
            'median': float(np.median(cubes_picked)),
            'min': int(np.min(cubes_picked)),
            'max': int(np.max(cubes_picked)),
        },

        # Units reference
        'units': {
            'distance': 'meters',
            'time': 'seconds',
            'reward': 'dimensionless',
            'success_rate': 'percentage'
        }
    }

    json_filename = f"masac_{env_type}_grid{grid_size}_cubes{num_cubes}_{timestamp}_summary.json"
    json_path = log_path / json_filename

    try:
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Saved JSON summary: {json_path.name}")
    except Exception as e:
        print(f"❌ Error saving JSON summary: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Test MASAC on Grid 4x4, 9 cubes (A* and RRT Viz)"""
    parser = argparse.ArgumentParser(description='Test MASAC on Grid 4x4, 9 cubes')
    parser.add_argument('--episodes', type=int, default=5, help='Number of test episodes per planner')
    parser.add_argument('--log_dir', type=str, default='cobotproject/scripts/Reinforcement Learning/MASAC/logs',
                        help='Directory to save logs')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--run_id', type=int, default=1, help='Run identifier for multiple runs')
    args = parser.parse_args()

    # Only Grid 4x4, 9 cubes (RRT Viz and A*)
    configurations = [
        ('rrt_viz', 4, 9),
        ('astar', 4, 9),
    ]

    print(f"MASAC TESTING - Grid 4x4, 9 Cubes (A* & RRT Viz)")
    print(f"Episodes per planner: {args.episodes}")
    print(f"Log directory: {args.log_dir}")
    print(f"Seed: {args.seed}")
    print(f"Run ID: {args.run_id}")
    print(f"{'='*80}\n")

    all_results = {}

    for env_type, grid_size, num_cubes in configurations:
        try:
            results = test_masac_configuration(
                env_type=env_type,
                grid_size=grid_size,
                num_cubes=num_cubes,
                num_episodes=args.episodes,
                log_dir=args.log_dir,
                seed=args.seed,
                run_id=args.run_id
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
    print("MASAC TESTING COMPLETE - Grid 4x4, 9 Cubes (A* & RRT Viz)")
    print(f"{'='*80}")
    print(f"Tested {len(all_results)}/{len(configurations)} planners successfully")
    print(f"Results saved to: {args.log_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()


