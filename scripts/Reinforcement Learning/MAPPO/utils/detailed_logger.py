"""
Detailed Logger for MAPPO Training

Logs comprehensive metrics to:
- WandB (online dashboard)
- CSV files (episode summaries)
- JSON files (detailed episode data)
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime


class DetailedLogger:
    """
    Comprehensive logger for MAPPO training.
    
    Logs:
    - Episode summaries (CSV)
    - Detailed episode data (JSON)
    - Reshuffle statistics
    - Reward component breakdowns
    - Distance improvements
    """
    
    def __init__(
        self,
        log_dir: Path,
        run_name: str,
        wandb_logger=None,
    ):
        """
        Initialize detailed logger.
        
        Args:
            log_dir: Directory to save logs
            run_name: Name of the run
            wandb_logger: WandB logger instance (optional)
        """
        self.log_dir = Path(log_dir)
        self.run_name = run_name
        self.wandb_logger = wandb_logger

        # Create log directory (no subdirectories, like DDQN)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CSV file (save directly in log_dir, like DDQN)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"{run_name}_{timestamp}_episode_log.csv"
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = None  # Will be initialized on first write

        # Episode history
        self.episode_history = []

        print(f"[LOGGER] Saving logs to: {self.log_dir}")
        print(f"[LOGGER] CSV: {self.csv_path}")
    
    def log_episode(
        self,
        episode: int,
        total_reward: float,
        episode_length: int,
        reshuffles_performed: int,
        reshuffle_details: List[Dict[str, Any]],
        cube_distances: Dict[int, float],
        reward_components: Optional[Dict[str, float]] = None,
        action_mask_stats: Optional[Dict[str, Any]] = None,
        additional_info: Optional[Dict[str, Any]] = None,
        total_distance_reduced: float = 0.0,  # NEW: Distance improvement metric
        total_time_saved: float = 0.0,  # NEW: Time saved metric
    ):
        """
        Log comprehensive episode data.

        Args:
            episode: Episode number
            total_reward: Total episode reward
            episode_length: Episode length
            reshuffles_performed: Number of reshuffles
            reshuffle_details: List of reshuffle dictionaries
            cube_distances: Final distances of each cube to robot
            reward_components: Breakdown of reward components
            action_mask_stats: Action mask statistics
            additional_info: Any additional info to log
            total_distance_reduced: Total distance improvement from reshuffling (meters)
            total_time_saved: Total time saved from reshuffling (seconds)
        """
        # Prepare episode data
        episode_data = {
            "episode": episode,
            "total_reward": total_reward,
            "episode_length": episode_length,
            "reshuffles_performed": reshuffles_performed,
            "total_distance_reduced": total_distance_reduced,  # NEW: Add to CSV
            "total_time_saved": total_time_saved,  # NEW: Add to CSV
            "timestamp": datetime.now().isoformat(),
        }

        # Add cube distances
        for cube_idx, dist in cube_distances.items():
            episode_data[f"cube_{cube_idx}_final_dist"] = dist

        # Add reward components
        if reward_components:
            for key, value in reward_components.items():
                episode_data[f"reward_{key}"] = value

        # Add action mask stats
        if action_mask_stats:
            for key, value in action_mask_stats.items():
                episode_data[f"mask_{key}"] = value

        # Add additional info
        if additional_info:
            episode_data.update(additional_info)
        
        # Write to CSV
        self._write_csv(episode_data)
        
        # Save detailed JSON
        detailed_data = {
            **episode_data,
            "reshuffle_details": reshuffle_details,
        }
        self._write_json(episode, detailed_data)

        # Note: WandB logging is now handled in the training scripts
        # to ensure proper step tracking with global_step

        # Store in history
        self.episode_history.append(episode_data)

    def _write_csv(self, episode_data: Dict[str, Any]):
        """Write episode data to CSV file."""
        # Initialize CSV writer with headers on first write
        if self.csv_writer is None:
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=episode_data.keys())
            self.csv_writer.writeheader()

        # Write row
        self.csv_writer.writerow(episode_data)
        self.csv_file.flush()  # Ensure data is written immediately

    def _write_json(self, episode: int, episode_data: Dict[str, Any]):
        """Write detailed episode data to JSON file (disabled to match DDQN format)."""
        # Disabled: DDQN doesn't save per-episode JSON files
        # Only save summary JSON at the end
        pass

    def log_reshuffle(
        self,
        cube_idx: int,
        reason: str,
        old_position: np.ndarray,
        new_position: np.ndarray,
        old_distance: float,
        new_distance: float,
        reward_components: Optional[Dict[str, float]] = None,
    ):
        """
        Log a single reshuffle action with detailed information.

        Args:
            cube_idx: Index of cube being reshuffled
            reason: Reason for reshuffle (e.g., "unreachable_distance", "blocked_by_obstacle")
            old_position: Old cube position
            new_position: New cube position
            old_distance: Old distance to robot
            new_distance: New distance to robot
            reward_components: Breakdown of improvement reward
        """
        distance_improvement = old_distance - new_distance

        # Print detailed reshuffle info
        print(f"[RESHUFFLE] Cube {cube_idx}: {reason.upper()}")
        print(f"            Old pos: {old_position} (dist={old_distance:.3f}m)")
        print(f"            New pos: {new_position} (dist={new_distance:.3f}m)")
        print(f"            Improvement: {distance_improvement:.3f}m ({distance_improvement/old_distance*100:.1f}%)")

        if reward_components:
            print(f"            Reward: {', '.join(f'{k}={v:.2f}' for k, v in reward_components.items())}")

    def save_summary(self):
        """Save final training summary."""
        if not self.episode_history:
            return

        summary = {
            "run_name": self.run_name,
            "total_episodes": len(self.episode_history),
            "avg_reward": np.mean([ep["total_reward"] for ep in self.episode_history]),
            "avg_episode_length": np.mean([ep["episode_length"] for ep in self.episode_history]),
            "avg_reshuffles": np.mean([ep["reshuffles_performed"] for ep in self.episode_history]),
            "best_episode": max(self.episode_history, key=lambda x: x["total_reward"]),
            "worst_episode": min(self.episode_history, key=lambda x: x["total_reward"]),
        }

        summary_path = self.log_dir / f"{self.run_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n[LOGGER] Training summary saved to: {summary_path}")
        print(f"[LOGGER] Total episodes: {summary['total_episodes']}")
        print(f"[LOGGER] Average reward: {summary['avg_reward']:.2f}")
        print(f"[LOGGER] Average reshuffles: {summary['avg_reshuffles']:.2f}")

    def close(self):
        """Close logger and save final summary."""
        self.save_summary()
        if self.csv_file:
            self.csv_file.close()
        print(f"[LOGGER] Logs saved successfully!")


