"""
Test the TwoAgentLogger to ensure it works correctly
"""

from two_agent_logger import TwoAgentLogger
import numpy as np
from datetime import datetime
import time


def test_logger():
    """Test the logger with fake data"""
    print("Testing TwoAgentLogger...")
    
    # Create logger
    logger = TwoAgentLogger(
        base_dir="test_two_agent_results",
        action_space="discrete",
        seed=42
    )
    
    print("\n✅ Logger created successfully")
    print(f"   Log directory: {logger.log_dir}")
    
    # Simulate 2 models, 3 episodes each
    models = ["DDQN", "Heuristic"]
    num_episodes = 3
    
    for model in models:
        print(f"\n📊 Testing model: {model}")
        start_time = datetime.now().isoformat()
        
        for episode in range(num_episodes):
            print(f"  Episode {episode+1}/{num_episodes}...")
            
            # Simulate episode
            episode_length = np.random.randint(10, 30)
            cumulative_reward = 0
            
            for step in range(episode_length):
                # Simulate timestep
                reward = np.random.uniform(-1, 1)
                cumulative_reward += reward
                
                timestep_data = {
                    'episode': episode + 1,
                    'step_in_episode': step + 1,
                    'model': model,
                    'reward': float(reward),
                    'cumulative_reward': float(cumulative_reward),
                    'reshuffled': np.random.random() < 0.2,
                    'distance_reduced': float(np.random.uniform(0, 0.5)),
                    'time_saved': float(np.random.uniform(0, 2.0)),
                    'cubes_picked_so_far': min(step // 3, 9),
                    'done': (step == episode_length - 1),
                    'truncated': False,
                    'timestamp': datetime.now().isoformat(),
                    'planner': 'Isaac Sim RRT'
                }
                logger.log_timestep(timestep_data)
            
            # Log episode
            episode_data = {
                'episode': episode + 1,
                'model': model,
                'agent1_reward': float(np.random.uniform(5, 15)),
                'success': np.random.random() < 0.8,
                'cubes_picked': np.random.randint(7, 10),
                'pick_failures': np.random.randint(0, 3),
                'successful_picks': np.random.randint(7, 10),
                'unreachable_cubes': np.random.randint(0, 3),
                'path_efficiency': float(np.random.uniform(0.7, 1.0)),
                'action_entropy': float(np.random.uniform(0.5, 2.0)),
                'agent2_reward': float(np.random.uniform(0, 5)),
                'reshuffles_performed': np.random.randint(0, 5),
                'total_distance_reduced': float(np.random.uniform(0, 2.0)),
                'total_time_saved': float(np.random.uniform(0, 10.0)),
                'total_reward': float(cumulative_reward),
                'episode_length': episode_length,
                'duration': float(np.random.uniform(10, 30)),
                'timestamp': datetime.now().isoformat(),
                'planner': 'Isaac Sim RRT',
                'grid_size': 4,
                'num_cubes': 9
            }
            logger.log_episode(episode_data)
            
            time.sleep(0.1)  # Small delay to simulate real testing
        
        # Write summary for this model
        logger.write_summary_for_model(model, start_time)
        print(f"  ✅ Summary written for {model}")
    
    print("\n" + "="*80)
    print("LOGGER TEST COMPLETE")
    print("="*80)
    print(f"Results saved to: {logger.log_dir}")
    print(f"Files created:")
    print(f"  - {logger.episode_file}")
    print(f"  - {logger.timestep_file}")
    print(f"  - {logger.summary_file}")
    print("="*80)
    
    # Verify files exist
    assert logger.episode_file.exists(), "Episode file not created"
    assert logger.timestep_file.exists(), "Timestep file not created"
    assert logger.summary_file.exists(), "Summary file not created"
    
    print("\n✅ All files created successfully!")
    
    # Print summary
    import json
    with open(logger.summary_file, 'r') as f:
        summary = json.load(f)
    
    print("\n📊 Summary Statistics:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    test_logger()

