import sys
from pathlib import Path
import numpy as np
import json
import csv
from datetime import datetime
import argparse


project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
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
    import torch
    agent = DoubleDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    agent.load(model_path)
    agent.epsilon = 0.01
    print(f" Loaded DDQN model: {model_path}")
    return agent


def test_masac_configuration(
    env_type: str,
    grid_size: int,
    num_cubes: int,
    num_episodes: int = 5,
    log_dir: str = "cobotproject/scripts/Reinforcement Learning/MASAC/logs"
):
    """Test MASAC on a specific configuration"""
    
    print(f"\n{'='*80}")
    print(f"Testing MASAC: {env_type.upper()} | Grid {grid_size}x{grid_size} | {num_cubes} cubes")
    print(f"{'='*80}")
    ddqn_model_mapping = {
        "rrt_viz_grid3_cubes4": "ddqn_rrt_viz_grid3_cubes4_20251220_025425_final.pt",
        "rrt_viz_grid4_cubes6": "ddqn_rrt_viz_grid4_cubes6_20251220_054851_final.pt",
        "rrt_viz_grid4_cubes9": "ddqn_rrt_viz_grid4_cubes9_20251220_134808_final.pt",
        "astar_grid3_cubes4": "ddqn_astar_grid3_cubes4_20251220_012015_final.pt",
        "astar_grid4_cubes6": "ddqn_astar_grid4_cubes6_20251220_014823_final.pt",
        "astar_grid4_cubes9": "ddqn_astar_grid4_cubes9_20251220_022000_final.pt",
    }
    
    config_key = f"{env_type}_grid{grid_size}_cubes{num_cubes}"
    max_objects = 9 if grid_size == 3 else 16
    
    if config_key not in ddqn_model_mapping:
        print(f"  No DDQN model mapping for {config_key}")
        return None
    
    ddqn_model_filename = ddqn_model_mapping[config_key]
    ddqn_models_dir = project_root / "scripts" / "Reinforcement Learning" / "doubleDQN_script" / "models"
    ddqn_model_path = ddqn_models_dir / ddqn_model_filename
    
    if not ddqn_model_path.exists():
        print(f"  DDQN model not found: {ddqn_model_path}")
        return None
    
    # Create base environment
    if env_type == 'astar':
        base_env = ObjectSelectionEnvAStar(
            franka_controller=None,
            max_objects=max_objects,
            max_steps=50,
            num_cubes=num_cubes,
            training_grid_size=grid_size
        )
        cube_spacing = 0.20 if grid_size > 3 else 0.22  # Same spacing as RRT Viz
    elif env_type == 'rrt_viz':
        base_env = ObjectSelectionEnvRRTViz(
            franka_controller=None,
            max_objects=max_objects,
            max_steps=50,
            num_cubes=num_cubes,
            training_grid_size=grid_size
        )
        cube_spacing = 0.20 if grid_size > 3 else 0.22  # Keep original spacing for RRT Viz
    else:
        raise ValueError(f"Unknown env_type: {env_type}")
    ddqn_agent = load_ddqn_agent(str(ddqn_model_path), base_env)
    agent2_state_dim = (num_cubes * 3) + 3 + num_cubes + (grid_size * grid_size) + 10
    pretrained_path = project_root / "scripts" / "Reinforcement Learning" / "MASAC" / "pretrained_models"
    
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


    print("[TEST] Relaxing reshuffling thresholds for testing...")
    two_agent_env.reshuffle_decision.min_reachable_distance = 0.30  # Was 0.35
    two_agent_env.reshuffle_decision.max_reachable_distance = 0.90  # Was 0.85
    two_agent_env.reshuffle_decision.path_length_ratio_threshold = 1.5  # Was 1.8
    two_agent_env.reshuffle_decision.crowded_threshold = 2  # Was 3
    two_agent_env.reshuffle_decision.rrt_failure_window = 2  # Was 3
    two_agent_env.reshuffle_decision.min_clearance = 0.35  # Was 0.3
    two_agent_env.reshuffle_decision.far_cube_ratio = 1.1  # Was 1.2
    two_agent_env.reshuffle_decision.batch_reshuffle_count = 2  # Was 3
    print("[TEST] Reshuffling thresholds relaxed!")


    print("[TEST] Enabling test mode for fast PCA fitting...")
    base_env.test_mode = True
    print("[TEST] Test mode enabled! Reachability checks will be skipped.")

    masac_agent.fit_dimension_adapter(two_agent_env, n_samples=500)

    masac_agent.set_test_mode(True)


    episode_results = []

    for episode in range(num_episodes):
        obs, _ = two_agent_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        reshuffles_performed = 0

        while not (done or truncated) and episode_length < 50:
            valid_cubes = [
                i for i in range(num_cubes)
                if i not in two_agent_env.base_env.objects_picked
                and two_agent_env.reshuffle_count_per_cube.get(i, 0) < 2
            ]
            action_dict = masac_agent.select_action(obs, deterministic=True, valid_cubes=valid_cubes)
            if action_dict is None:
                print(f"  [WARNING] No valid cubes to reshuffle, skipping step")
                break

            action_int = two_agent_env.reshuffle_action_space.encode_action(
                cube_idx=action_dict['cube_idx'],
                grid_x=action_dict['target_grid_x'],
                grid_y=action_dict['target_grid_y']
            )


            next_obs, reward, done, truncated, info = two_agent_env.step(action_int)

            episode_reward += reward
            episode_length += 1

            if info.get('reshuffled_this_step', False):
                reshuffles_performed += 1

            obs = next_obs


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
    save_results(env_type, grid_size, num_cubes, episode_results, log_dir)

    return episode_results


def save_results(env_type: str, grid_size: int, num_cubes: int, episode_results: list, log_dir: str):
    """Save test results to CSV and JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    csv_filename = f"masac_{env_type}_grid{grid_size}_cubes{num_cubes}_{timestamp}_episode_log.csv"
    csv_path = log_path / csv_filename

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=episode_results[0].keys())
        writer.writeheader()
        writer.writerows(episode_results)

    print(f" Saved CSV log: {csv_path}")

    # Save JSON summary
    summary = {
        'env_type': env_type,
        'grid_size': grid_size,
        'num_cubes': num_cubes,
        'num_episodes': len(episode_results),
        'avg_reward': np.mean([r['total_reward'] for r in episode_results]),
        'avg_reshuffles': np.mean([r['reshuffles_performed'] for r in episode_results]),
        'avg_distance_reduced': np.mean([r['total_distance_reduced'] for r in episode_results]),
        'avg_cubes_picked': np.mean([r['cubes_picked'] for r in episode_results]),
        'episodes': episode_results
    }

    json_filename = f"masac_{env_type}_grid{grid_size}_cubes{num_cubes}_{timestamp}_summary.json"
    json_path = log_path / json_filename

    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f" Saved JSON summary: {json_path}")


def main():
    """Test MASAC on RRT Viz and A* configurations"""
    parser = argparse.ArgumentParser(description='Test MASAC on RRT Viz and A*')
    parser.add_argument('--episodes', type=int, default=5, help='Number of test episodes per config')
    parser.add_argument('--log_dir', type=str, default='cobotproject/scripts/Reinforcement Learning/MASAC/logs',
                        help='Directory to save logs')
    args = parser.parse_args()
    configurations = [
        ('astar', 3, 4),
        ('astar', 4, 6),
        ('astar', 4, 9),
        ('rrt_viz', 3, 4),
        ('rrt_viz', 4, 6),
        ('rrt_viz', 4, 9),
    ]

    print(f"MASAC TESTING - A* & RRT VIZ (6 CONFIGURATIONS)")
    print(f"Order: A* (3) → RRT Viz (3)")
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
            print(f" Error testing {env_type} grid{grid_size} cubes{num_cubes}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{'='*80}")
    print("MASAC TESTING COMPLETE")
    print(f"{'='*80}")
    print(f"Tested {len(all_results)}/{len(configurations)} configurations successfully")
    print(f"Results saved to: {args.log_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()


