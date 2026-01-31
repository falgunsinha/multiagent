"""
Two-Agent System Logger - Fast Training-Time Logging
Logs raw episode and timestep data during training with minimal overhead.
Post-processing is done separately after training completes.
"""

import csv
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


class TwoAgentLogger:
    """Fast logger for two-agent system training"""
    
    def __init__(self, base_dir: str, action_space: str, seed: int):
        """
        Initialize logger for a specific action space and seed
        
        Args:
            base_dir: Base directory for all results (e.g., 'two_agent_results')
            action_space: 'discrete' or 'continuous'
            seed: Random seed for this run
        """
        self.base_dir = Path(base_dir)
        self.action_space = action_space
        self.seed = seed
        
        # Create directory structure
        self.log_dir = self.base_dir / action_space / f"seed_{seed}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.episode_file = self.log_dir / "episode_results.csv"
        self.timestep_file = self.log_dir / "timestep_results.csv"
        self.summary_file = self.log_dir / "summary.json"
        
        # Initialize CSV files with headers
        self._init_episode_csv()
        self._init_timestep_csv()
        
        # Counters
        self.global_timestep = 0
        self.episode_data_buffer = []  # Buffer for summary generation
        
        print(f"✅ TwoAgentLogger initialized:")
        print(f"   Action Space: {action_space}")
        print(f"   Seed: {seed}")
        print(f"   Log Directory: {self.log_dir}")
        print(f"   Episode CSV: {self.episode_file}")
        print(f"   Timestep CSV: {self.timestep_file}")
    
    def _init_episode_csv(self):
        """Initialize episode results CSV with headers"""
        if not self.episode_file.exists():
            with open(self.episode_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    # Episode info
                    'episode', 'model', 'seed',
                    # Agent 1 metrics (Pick Sequence)
                    'agent1_reward', 'success', 'cubes_picked',
                    'pick_failures', 'successful_picks', 'unreachable_cubes',
                    'path_efficiency', 'action_entropy',
                    # Agent 2 metrics (Reshuffling)
                    'agent2_reward', 'reshuffles_performed',
                    'total_distance_reduced', 'total_time_saved',
                    'total_distance_traveled', 'total_time_taken',
                    # Combined metrics
                    'total_reward', 'episode_length', 'duration', 'timeout',
                    # Efficiency metrics (NEW - GAT+CVD specific)
                    'distance_efficiency', 'time_efficiency',
                    'avg_distance_per_reshuffle', 'avg_time_per_reshuffle',
                    'success_rate', 'steps_per_cube',
                    # Metadata
                    'timestamp', 'planner', 'grid_size', 'num_cubes'
                ])
    
    def _init_timestep_csv(self):
        """Initialize timestep results CSV with headers"""
        if not self.timestep_file.exists():
            with open(self.timestep_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    # Timestep info
                    'global_timestep', 'episode', 'step_in_episode', 
                    'model', 'seed',
                    # Step metrics
                    'reward', 'cumulative_reward', 
                    'reshuffled', 'distance_reduced', 'time_saved',
                    'cubes_picked_so_far',
                    # Episode status
                    'done', 'truncated',
                    # Metadata
                    'timestamp', 'planner'
                ])
    
    def log_episode(self, episode_data: Dict[str, Any]):
        """
        Append episode data to CSV (FAST - no computation)
        
        Args:
            episode_data: Dictionary with episode metrics
        """
        # Ensure file exists (in case it was deleted)
        if not self.episode_file.exists():
            self._init_episode_csv()

        with open(self.episode_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                # Episode info (1-based indexing to match training)
                episode_data['episode'] + 1,  # Convert 0-based to 1-based
                episode_data['model'],
                self.seed,
                # Agent 1
                episode_data['agent1_reward'],
                episode_data['success'],
                episode_data['cubes_picked'],
                episode_data['pick_failures'],
                episode_data['successful_picks'],
                episode_data['unreachable_cubes'],
                episode_data.get('path_efficiency', 0.0),
                episode_data.get('action_entropy', 0.0),
                # Agent 2
                episode_data['agent2_reward'],
                episode_data['reshuffles_performed'],
                episode_data['total_distance_reduced'],
                episode_data['total_time_saved'],
                episode_data.get('total_distance_traveled', 0.0),
                episode_data.get('total_time_taken', 0.0),
                # Combined
                episode_data['total_reward'],
                episode_data['episode_length'],
                episode_data['duration'],
                episode_data.get('timeout', False),  # NEW: Timeout flag
                # Efficiency metrics (NEW - GAT+CVD specific)
                episode_data.get('distance_efficiency', 0.0),
                episode_data.get('time_efficiency', 0.0),
                episode_data.get('avg_distance_per_reshuffle', 0.0),
                episode_data.get('avg_time_per_reshuffle', 0.0),
                episode_data.get('success_rate', 0.0),
                episode_data.get('steps_per_cube', 0.0),
                # Metadata
                episode_data.get('timestamp', datetime.now().isoformat()),
                episode_data.get('planner', 'Isaac Sim RRT'),
                episode_data.get('grid_size', 4),
                episode_data.get('num_cubes', 9)
            ])
        
        # Buffer for summary generation
        self.episode_data_buffer.append(episode_data)
    
    def log_timestep(self, timestep_data: Dict[str, Any]):
        """
        Append timestep data to CSV (FAST - no computation)

        Args:
            timestep_data: Dictionary with timestep metrics
        """
        self.global_timestep += 1

        # Ensure file exists (in case it was deleted)
        if not self.timestep_file.exists():
            self._init_timestep_csv()

        with open(self.timestep_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                # Timestep info
                self.global_timestep,
                timestep_data['episode'] + 1,  # Convert 0-based to 1-based
                timestep_data['step_in_episode'],
                timestep_data['model'],
                self.seed,
                # Step metrics
                timestep_data['reward'],
                timestep_data['cumulative_reward'],
                timestep_data.get('reshuffled', False),
                timestep_data.get('distance_reduced', 0.0),
                timestep_data.get('time_saved', 0.0),
                timestep_data.get('cubes_picked_so_far', 0),
                # Episode status
                timestep_data['done'],
                timestep_data.get('truncated', False),
                # Metadata
                timestep_data.get('timestamp', datetime.now().isoformat()),
                timestep_data.get('planner', 'Isaac Sim RRT')
            ])

    def write_summary_for_model(self, model_name: str, start_time: str):
        """
        Write summary.json after a model completes all episodes

        Args:
            model_name: Name of the model that just completed
            start_time: ISO format start time
        """
        # Filter episodes for this model
        model_episodes = [ep for ep in self.episode_data_buffer if ep['model'] == model_name]

        if not model_episodes:
            return

        # Calculate statistics
        agent1_rewards = [ep['agent1_reward'] for ep in model_episodes]
        agent2_rewards = [ep['agent2_reward'] for ep in model_episodes]
        total_rewards = [ep['total_reward'] for ep in model_episodes]
        successes = [ep['success'] for ep in model_episodes]
        cubes_picked = [ep['cubes_picked'] for ep in model_episodes]
        pick_failures = [ep['pick_failures'] for ep in model_episodes]
        reshuffles = [ep['reshuffles_performed'] for ep in model_episodes]
        distances_reduced = [ep['total_distance_reduced'] for ep in model_episodes]
        times_saved = [ep['total_time_saved'] for ep in model_episodes]
        episode_lengths = [ep['episode_length'] for ep in model_episodes]

        model_summary = {
            "episodes": len(model_episodes),
            "agent1_metrics": {
                "avg_reward": float(np.mean(agent1_rewards)),
                "std_reward": float(np.std(agent1_rewards)),
                "avg_success": float(np.mean(successes)),
                "avg_cubes_picked": float(np.mean(cubes_picked)),
                "avg_pick_failures": float(np.mean(pick_failures)),
                "avg_path_efficiency": float(np.mean([ep.get('path_efficiency', 0.0) for ep in model_episodes])),
            },
            "agent2_metrics": {
                "avg_reward": float(np.mean(agent2_rewards)),
                "std_reward": float(np.std(agent2_rewards)),
                "avg_reshuffles": float(np.mean(reshuffles)),
                "avg_distance_reduced": float(np.mean(distances_reduced)),
                "avg_time_saved": float(np.mean(times_saved)),
            },
            "combined_metrics": {
                "avg_total_reward": float(np.mean(total_rewards)),
                "std_total_reward": float(np.std(total_rewards)),
                "avg_episode_length": float(np.mean(episode_lengths)),
            }
        }

        # Load existing summary or create new
        if self.summary_file.exists():
            with open(self.summary_file, 'r') as f:
                summary = json.load(f)
        else:
            summary = {
                "experiment_info": {
                    "seed": self.seed,
                    "total_episodes": 0,
                    "models_tested": 0,
                    "planner": "Isaac Sim RRT",
                    "grid_size": model_episodes[0].get('grid_size', 4),
                    "num_cubes": model_episodes[0].get('num_cubes', 9),
                    "start_time": start_time,
                    "end_time": None,
                    "total_duration_seconds": 0
                },
                "models": {}
            }

        # Update summary
        summary["models"][model_name] = model_summary
        summary["experiment_info"]["models_tested"] = len(summary["models"])
        summary["experiment_info"]["total_episodes"] = sum(m["episodes"] for m in summary["models"].values())
        summary["experiment_info"]["end_time"] = datetime.now().isoformat()

        # Calculate duration
        if start_time:
            start_dt = datetime.fromisoformat(start_time)
            end_dt = datetime.now()
            summary["experiment_info"]["total_duration_seconds"] = int((end_dt - start_dt).total_seconds())

        # Write summary
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✅ Summary updated for model: {model_name}")

