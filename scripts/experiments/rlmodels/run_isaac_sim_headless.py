"""
Standalone Headless Isaac Sim Execution for RL Model Testing
Runs all models in headless mode with real-time WandB logging
"""

import argparse
import sys
from pathlib import Path

# Parse arguments BEFORE importing Isaac Sim
parser = argparse.ArgumentParser(description="RL Model Testing in Isaac Sim (Headless)")
parser.add_argument("--experiment", type=str, required=True, 
                   help="Experiment name (exp1, exp2, exp3, exp4, exp5, exp6)")
parser.add_argument("--num_episodes", type=int, default=100,
                   help="Number of episodes per model")
parser.add_argument("--num_cubes", type=int, default=9,
                   help="Number of cubes in environment")
parser.add_argument("--grid_size", type=int, default=4,
                   help="Grid size for cube placement")
parser.add_argument("--wandb_project", type=str, default="exp-rl-models",
                   help="WandB project name")
parser.add_argument("--wandb_entity", type=str, default="falgunsinha",
                   help="WandB entity/username")
parser.add_argument("--save_results", action="store_true", default=True,
                   help="Save results to CSV")
args, unknown = parser.parse_known_args()

# Create SimulationApp BEFORE importing Isaac Sim modules
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})  # HEADLESS MODE

# NOW import Isaac Sim modules
import time
import numpy as np
import pandas as pd
import wandb
from datetime import datetime

# Isaac Sim imports
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats

# Add project root to path
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add rlmodels to path
rlmodels_root = Path(r"C:\isaacsim\cobotproject\scripts\experiments\rlmodels")
if str(rlmodels_root) not in sys.path:
    sys.path.insert(0, str(rlmodels_root))

# Import local modules
from utils import ConfigManager, DataLoader, PathManager
from core import ModelLoader, MetricsCollector
from adapters import (
    FeatureAggregationAdapter,
    PCAStateAdapter,
    RandomProjectionAdapter,
    DiscreteActionMapper,
    ContinuousToDiscreteAdapter,
    WeightedActionAdapter,
    ProbabilisticActionAdapter
)

# Import RL libraries
try:
    from stable_baselines3 import PPO, DQN, A2C, SAC, TD3
    from sb3_contrib import MaskablePPO
    import torch
    RL_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] Could not import RL libraries: {e}")
    RL_AVAILABLE = False
    simulation_app.close()
    sys.exit(1)


class IsaacSimEnvironment:
    """Isaac Sim environment for RL model testing"""
    
    def __init__(self, num_cubes=9, grid_size=4):
        """Initialize Isaac Sim environment"""
        self.num_cubes = num_cubes
        self.grid_size = grid_size
        
        # Create world
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        
        # Create cubes
        self.cubes = []
        self._create_cubes()
        
        # Robot (placeholder - you'll need to add your Franka robot)
        self.robot_position = np.array([0.0, 0.0, 0.0])
        
        print(f"[Isaac Sim] Environment created with {num_cubes} cubes")
    
    def _create_cubes(self):
        """Create cubes in grid layout"""
        cube_size = 0.05
        spacing = 0.15
        
        for i in range(self.num_cubes):
            row = i // self.grid_size
            col = i % self.grid_size
            
            x = col * spacing
            y = row * spacing
            z = cube_size / 2
            
            cube = DynamicCuboid(
                prim_path=f"/World/Cube_{i}",
                name=f"cube_{i}",
                position=np.array([x, y, z]),
                scale=np.array([cube_size, cube_size, cube_size]),
                color=np.array([1.0, 0.0, 0.0])
            )
            
            self.cubes.append(cube)
    
    def reset(self):
        """Reset environment"""
        self.world.reset()
        
        # Get initial state
        state = self._get_state()
        
        return state
    
    def _get_state(self):
        """Get current state (24D)"""
        # Cube positions (9 cubes × 2D = 18D)
        cube_positions = []
        for cube in self.cubes:
            pos, _ = cube.get_world_pose()
            cube_positions.extend([pos[0], pos[1]])
        
        # Robot position (2D)
        robot_pos = self.robot_position[:2]
        
        # Additional state info
        target_cube = 0  # Placeholder
        gripper_state = 0.0
        distance_to_target = 0.0
        collision = 0.0
        
        state = np.array(
            cube_positions + robot_pos.tolist() + 
            [target_cube, gripper_state, distance_to_target, collision],
            dtype=np.float32
        )
        
        return state
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        # Placeholder - implement actual action execution
        self.world.step(render=False)
        
        next_state = self._get_state()
        reward = 0.0
        done = False
        info = {}
        
        return next_state, reward, done, info


def run_experiment(experiment_name: str):
    """Run a single experiment"""
    
    print(f"\n{'='*60}")
    print(f"Running Experiment: {experiment_name}")
    print(f"{'='*60}\n")
    
    # Load configuration
    config_manager = ConfigManager()
    exp_config = config_manager.get_experiment_config(experiment_name)
    
    if exp_config is None:
        print(f"[ERROR] Could not load config for {experiment_name}")
        return
    
    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "experiment": experiment_name,
            "num_episodes": args.num_episodes,
            "num_cubes": args.num_cubes,
            "grid_size": args.grid_size,
            **exp_config
        }
    )
    
    # Create environment
    env = IsaacSimEnvironment(num_cubes=args.num_cubes, grid_size=args.grid_size)
    
    # Initialize model loader and metrics collector
    model_loader = ModelLoader()
    metrics_collector = MetricsCollector()
    
    # Get models to test
    models_config = exp_config.get('models', [])
    
    # Results storage
    all_results = []
    
    # Test each model
    for model_config in models_config:
        model_name = model_config['name']
        model_path = model_config['path']
        
        print(f"\n[Model] Testing: {model_name}")
        print(f"[Model] Path: {model_path}")
        
        # Load model
        try:
            model = model_loader.load_model(model_path, model_config.get('type', 'ppo'))
        except Exception as e:
            print(f"[ERROR] Could not load model {model_name}: {e}")
            continue
        
        # Initialize adapters if needed
        state_adapter = None
        action_adapter = None
        
        if 'state_adapter' in model_config:
            adapter_type = model_config['state_adapter']
            if adapter_type == 'feature_aggregation':
                state_adapter = FeatureAggregationAdapter()
            elif adapter_type == 'pca':
                state_adapter = PCAStateAdapter()
            elif adapter_type == 'random_projection':
                state_adapter = RandomProjectionAdapter()
        
        if 'action_adapter' in model_config:
            adapter_type = model_config['action_adapter']
            if adapter_type == 'discrete_mapper':
                action_adapter = DiscreteActionMapper()
            elif adapter_type == 'continuous_to_discrete':
                action_adapter = ContinuousToDiscreteAdapter()
        
        # Run episodes
        for episode in range(args.num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            episode_start_time = time.time()
            
            while not done and episode_steps < 1000:
                # Apply state adapter if needed
                if state_adapter:
                    adapted_state = state_adapter.transform(state)
                else:
                    adapted_state = state
                
                # Get action from model
                action, _ = model.predict(adapted_state, deterministic=True)
                
                # Apply action adapter if needed
                if action_adapter:
                    action = action_adapter.map_action(action, state)
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
            
            episode_time = time.time() - episode_start_time
            
            # Log to WandB
            wandb.log({
                f"{model_name}/episode_reward": episode_reward,
                f"{model_name}/episode_steps": episode_steps,
                f"{model_name}/episode_time": episode_time,
                "episode": episode
            })
            
            # Store results
            all_results.append({
                'model': model_name,
                'episode': episode,
                'reward': episode_reward,
                'steps': episode_steps,
                'time': episode_time
            })
            
            if (episode + 1) % 10 == 0:
                print(f"  Episode {episode + 1}/{args.num_episodes} - Reward: {episode_reward:.2f}")
    
    # Save results
    if args.save_results:
        results_df = pd.DataFrame(all_results)
        output_path = PathManager().get_results_path(experiment_name, 'results.csv')
        results_df.to_csv(output_path, index=False)
        print(f"\n[Results] Saved to: {output_path}")
    
    # Finish WandB
    wandb.finish()
    
    print(f"\n[Experiment] {experiment_name} completed!")


if __name__ == "__main__":
    try:
        run_experiment(args.experiment)
    finally:
        simulation_app.close()

