"""
Quick Test: Heuristic Baseline Only
Tests just the heuristic agents to verify the pipeline works.
This is the fastest way to verify the system is working correctly.

Usage:
    C:\isaacsim\python.bat test_heuristic_only.py --episodes 3
"""

import sys
from pathlib import Path

# Add project root to path (absolute path for reliability)
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add MASAC and MAPPO to path (for agents/ and envs/ imports)
masac_path = Path(r"C:\isaacsim\cobotproject\scripts\Reinforcement Learning\MASAC")
if str(masac_path) not in sys.path:
    sys.path.insert(0, str(masac_path))
mappo_path = Path(r"C:\isaacsim\cobotproject\scripts\Reinforcement Learning\MAPPO")
if str(mappo_path) not in sys.path:
    sys.path.insert(0, str(mappo_path))

# Import Isaac Sim components (must be before other imports)
from isaacsim import SimulationApp
import argparse

# Parse arguments BEFORE creating SimulationApp
parser = argparse.ArgumentParser(description="Test Heuristic Baseline Only")
parser.add_argument("--episodes", type=int, default=3, help="Number of episodes (default: 3)")
parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
args = parser.parse_args()

# Create SimulationApp
simulation_app = SimulationApp({"headless": True})

# Now import everything else
import numpy as np
import time
from datetime import datetime

# Import Isaac Sim modules
from isaacsim.core.api import World
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage

# Import RL components
from src.rl.object_selection_env_rrt import ObjectSelectionEnvRRT
from envs.two_agent_env import TwoAgentEnv

# Import our new components
from two_agent_logger import TwoAgentLogger
from heuristic_agents import HeuristicAgent1, HeuristicAgent2


def create_isaacsim_environment(grid_size: int, num_cubes: int, max_steps: int = 50):
    """Create Isaac Sim environment with RRT planner"""
    from src.rl.franka_controller_rrt import FrankaControllerRRT
    
    # Create world
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    
    # Add Franka robot
    franka_prim_path = "/World/Franka"
    add_reference_to_stage(
        usd_path=get_assets_root_path() + "/Isaac/Robots/Franka/franka_alt_fingers.usd",
        prim_path=franka_prim_path
    )
    
    # Create Franka controller with RRT
    franka_controller = FrankaControllerRRT(
        world=world,
        prim_path=franka_prim_path,
        grid_size=grid_size,
        num_cubes=num_cubes
    )
    
    # Create environment
    env = ObjectSelectionEnvRRT(
        franka_controller=franka_controller,
        world=world,
        max_steps=max_steps,
        grid_size=grid_size,
        num_cubes=num_cubes
    )
    
    return env, world


def main():
    """Test heuristic baseline only"""
    print(f"\n{'='*80}")
    print(f"HEURISTIC BASELINE TEST")
    print(f"{'='*80}")
    print(f"Episodes: {args.episodes}")
    print(f"Seed: {args.seed}")
    print(f"{'='*80}\n")
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create logger
    logger = TwoAgentLogger(
        base_dir="heuristic_test_results",
        action_space="discrete",
        seed=args.seed
    )
    
    # Create Isaac Sim environment
    print("✓ Creating Isaac Sim environment...")
    grid_size = 4
    num_cubes = 9
    base_env, world = create_isaacsim_environment(
        grid_size=grid_size,
        num_cubes=num_cubes,
        max_steps=50
    )
    
    # Create heuristic agents
    print("✓ Creating heuristic agents...")
    state_dim = base_env.observation_space.shape[0]
    action_dim = base_env.action_space.n
    
    agent1 = HeuristicAgent1(state_dim=state_dim, action_dim=action_dim)
    agent2 = HeuristicAgent2(
        state_dim=(num_cubes * 3) + 3 + num_cubes + (grid_size * grid_size) + 10,
        action_dim=3,
        grid_size=grid_size,
        num_cubes=num_cubes,
        cube_spacing=0.13
    )
    
    # Create two-agent environment
    two_agent_env = TwoAgentEnv(
        base_env=base_env,
        ddqn_agent=agent1,
        grid_size=grid_size,
        num_cubes=num_cubes,
        max_reshuffles_per_episode=5,
        reshuffle_reward_scale=1.0,
        max_episode_steps=50,
        verbose=True
    )
    
    # Test episodes
    start_time = datetime.now().isoformat()
    
    for episode in range(args.episodes):
        # Randomize cube positions
        if hasattr(base_env, 'franka_controller') and base_env.franka_controller is not None:
            base_env.franka_controller.randomize_cube_positions()
        
        obs, _ = two_agent_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        agent1_reward = 0
        agent2_reward = 0
        episode_length = 0
        reshuffles_performed = 0
        episode_start_time = time.time()
        
        print(f"\n[Episode {episode+1}/{args.episodes}] Starting...")
        
        while not (done or truncated) and episode_length < 50:
            # Agent 2 selects reshuffling action (heuristic)
            action_continuous = agent2.select_action(obs, env=two_agent_env)
            
            # Convert continuous to dictionary
            cube_idx = int((action_continuous[0] + 1) / 2 * (num_cubes - 1))
            grid_x = int((action_continuous[1] + 1) / 2 * (grid_size - 1))
            grid_y = int((action_continuous[2] + 1) / 2 * (grid_size - 1))
            
            # Convert to integer action
            action_int = two_agent_env.reshuffle_action_space.encode_action(
                cube_idx=cube_idx,
                grid_x=grid_x,
                grid_y=grid_y
            )
            
            # Execute action
            next_obs, reward, done, truncated, info = two_agent_env.step(action_int)
            
            episode_reward += reward
            episode_length += 1
            
            if info.get('reshuffled_this_step', False):
                reshuffles_performed += 1
                agent2_reward += reward
            else:
                agent1_reward += reward
            
            # Log timestep
            timestep_data = {
                'episode': episode + 1,
                'step_in_episode': episode_length,
                'model': 'Heuristic',
                'reward': float(reward),
                'cumulative_reward': float(episode_reward),
                'reshuffled': info.get('reshuffled_this_step', False),
                'distance_reduced': float(info.get('distance_reduced', 0.0)),
                'time_saved': float(info.get('time_saved', 0.0)),
                'cubes_picked_so_far': len(two_agent_env.base_env.objects_picked),
                'done': done,
                'truncated': truncated,
                'timestamp': datetime.now().isoformat(),
                'planner': 'Isaac Sim RRT'
            }
            logger.log_timestep(timestep_data)
            
            obs = next_obs
        
        # Episode complete
        episode_duration = time.time() - episode_start_time
        cubes_picked = len(two_agent_env.base_env.objects_picked)
        success = (cubes_picked == num_cubes)
        
        # Log episode
        episode_data = {
            'episode': episode + 1,
            'model': 'Heuristic',
            'agent1_reward': float(agent1_reward),
            'success': success,
            'cubes_picked': cubes_picked,
            'pick_failures': 0,
            'successful_picks': cubes_picked,
            'unreachable_cubes': num_cubes - cubes_picked,
            'path_efficiency': 0.0,
            'action_entropy': 0.0,
            'agent2_reward': float(agent2_reward),
            'reshuffles_performed': reshuffles_performed,
            'total_distance_reduced': float(two_agent_env.total_distance_reduced),
            'total_time_saved': float(two_agent_env.total_time_saved),
            'total_reward': float(episode_reward),
            'episode_length': episode_length,
            'duration': episode_duration,
            'timestamp': datetime.now().isoformat(),
            'planner': 'Isaac Sim RRT',
            'grid_size': grid_size,
            'num_cubes': num_cubes
        }
        logger.log_episode(episode_data)
        
        print(f"✅ Episode {episode+1}/{args.episodes}: "
              f"Reward={episode_reward:.2f}, Cubes={cubes_picked}/{num_cubes}, "
              f"Reshuffles={reshuffles_performed}")
    
    # Write summary
    logger.write_summary_for_model('Heuristic', start_time)
    
    print(f"\n{'='*80}")
    print(f"HEURISTIC BASELINE TEST COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {logger.log_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[CLEANUP] Closing Isaac Sim...")
        try:
            from omni.isaac.core.utils.stage import clear_stage
            clear_stage()
        except:
            pass
        try:
            simulation_app.close()
            print("[CLEANUP] Isaac Sim closed successfully")
        except:
            pass

