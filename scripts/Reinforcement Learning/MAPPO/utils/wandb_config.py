"""
WandB Configuration for MAPPO Experiments

Separate dashboard for MAPPO two-agent experiments.
"""

import wandb
import numpy as np
from typing import Dict, Any, Optional, List
from collections import deque


class WandBLogger:
    """
    WandB logger for MAPPO experiments.

    Tracks:
    - Agent 1 (DDQN) metrics
    - Agent 2 (MAPPO) metrics with variance
    - System metrics
    - Reshuffling statistics with variance
    """

    def __init__(
        self,
        project_name: str = "ddqn-mappo-object-selection-reshuffling",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        group: Optional[str] = None,
        variance_window: int = 100,
    ):
        """
        Initialize WandB logger.

        Args:
            project_name: WandB project name (default: "ddqn-mappo-object-selection-reshuffling")
            run_name: Run name (auto-generated if None)
            config: Configuration dictionary
            tags: List of tags
            notes: Run notes
            group: Group name for organizing related runs (e.g., "grid_4x4_9cubes")
            variance_window: Window size for computing variance/std (default: 100)
        """
        self.run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            tags=tags,
            notes=notes,
            group=group,
        )

        # Define custom metrics
        self._define_custom_metrics()

        # Track metrics for variance computation (using deque for efficiency)
        self.variance_window = variance_window
        self.reward_history = deque(maxlen=variance_window)
        self.reshuffle_history = deque(maxlen=variance_window)
        self.distance_reduced_history = deque(maxlen=variance_window)
        self.time_saved_history = deque(maxlen=variance_window)
        self.policy_loss_history = deque(maxlen=variance_window)
        self.value_loss_history = deque(maxlen=variance_window)
        self.entropy_history = deque(maxlen=variance_window)
    
    def _define_custom_metrics(self):
        """
        Define custom WandB metrics - matches DDQN format

        Note: For smooth curves and y-axis starting from 0:
        - In WandB UI: Settings > Smoothing > 0.6-0.9 (for smooth curves)
        - In WandB UI: Edit Chart > Y-Axis > Min: 0 (to start from 0)
        """
        # Use global_step as primary step metric (matches DDQN)
        wandb.define_metric("global_step")

        # Training metrics (matches DDQN)
        wandb.define_metric("training/*", step_metric="global_step")
        wandb.define_metric("train/*", step_metric="global_step")

        # Episode metrics (matches DDQN) - PRIMARY METRICS
        wandb.define_metric("episode/*", step_metric="global_step")

        # Reshuffling metrics (MAPPO-specific)
        wandb.define_metric("reshuffle/*", step_metric="global_step")

        # MAPPO-specific training metrics
        wandb.define_metric("mappo/*", step_metric="global_step")
    
    def log_training_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log training metrics (matches DDQN format).

        Args:
            metrics: Dictionary of metrics (e.g., loss, policy_loss, value_loss, entropy)
            step: Training step
        """
        log_dict = {"global_step": step}
        for key, value in metrics.items():
            log_dict[f"training/{key}"] = value

        wandb.log(log_dict)

    def log_mappo_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log MAPPO-specific metrics with variance.

        Args:
            metrics: Dictionary of metrics (e.g., policy_loss, value_loss, entropy, ratio)
            step: Training step
        """
        log_dict = {"global_step": step}

        # Track metrics for variance computation
        if "policy_loss" in metrics:
            self.policy_loss_history.append(metrics["policy_loss"])
        if "value_loss" in metrics:
            self.value_loss_history.append(metrics["value_loss"])
        if "entropy" in metrics:
            self.entropy_history.append(metrics["entropy"])

        # Log main metrics
        for key, value in metrics.items():
            log_dict[f"mappo/{key}"] = value

        # Log variance/std for key metrics (if enough data)
        if len(self.policy_loss_history) >= 10:
            log_dict["mappo/policy_loss_std"] = np.std(self.policy_loss_history)
        if len(self.value_loss_history) >= 10:
            log_dict["mappo/value_loss_std"] = np.std(self.value_loss_history)
        if len(self.entropy_history) >= 10:
            log_dict["mappo/entropy_std"] = np.std(self.entropy_history)

        wandb.log(log_dict)

    def log_system_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log episode-level metrics (matches DDQN format).
        Uses 'episode/' prefix to match DDQN naming convention.

        Args:
            metrics: Dictionary of metrics (e.g., total_reward, episode_length, cubes_picked, reshuffles_performed)
            step: Training step (global_step, not episode)
        """
        log_dict = {"global_step": step}

        # Map MAPPO metric names to DDQN format
        # NOTE: Removed episode/total_length (not useful) and global_step (redundant)
        metric_mapping = {
            "total_reward": "episode/total_reward",
            "cubes_picked": "episode/cubes_picked",
            "reshuffles_performed": "episode/reshuffles_performed",
        }

        for key, value in metrics.items():
            # Skip episode_length (not useful)
            if key == "episode_length":
                continue

            # Use mapped name if available, otherwise use episode/ prefix
            metric_name = metric_mapping.get(key, f"episode/{key}")
            log_dict[metric_name] = value

        wandb.log(log_dict)
    
    def log_reshuffle_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log reshuffling metrics.

        Args:
            metrics: Dictionary of metrics
            step: Training step
        """
        log_dict = {"global_step": step}
        for key, value in metrics.items():
            log_dict[f"reshuffle/{key}"] = value

        wandb.log(log_dict)

    def log_episode_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log episode-level metrics with variance (matches DDQN format).

        Args:
            metrics: Dictionary of metrics (e.g., total_reward, total_length, success)
            step: Training step (global_step)
        """
        log_dict = {"global_step": step}

        # Track metrics for variance computation
        if "total_reward" in metrics:
            self.reward_history.append(metrics["total_reward"])
        if "reshuffles_performed" in metrics:
            self.reshuffle_history.append(metrics["reshuffles_performed"])
        if "distance_reduced" in metrics:
            self.distance_reduced_history.append(metrics["distance_reduced"])
        if "time_saved" in metrics:
            self.time_saved_history.append(metrics["time_saved"])

        # Log main metrics
        for key, value in metrics.items():
            log_dict[f"episode/{key}"] = value

        # Log variance/std for key metrics (if enough data)
        if len(self.reward_history) >= 10:
            log_dict["episode/total_reward_std"] = np.std(self.reward_history)
            log_dict["episode/total_reward_min"] = np.min(self.reward_history)
            log_dict["episode/total_reward_max"] = np.max(self.reward_history)

        if len(self.reshuffle_history) >= 10:
            log_dict["episode/reshuffles_std"] = np.std(self.reshuffle_history)

        if len(self.distance_reduced_history) >= 10:
            log_dict["episode/distance_reduced_std"] = np.std(self.distance_reduced_history)

        if len(self.time_saved_history) >= 10:
            log_dict["episode/time_saved_std"] = np.std(self.time_saved_history)

        wandb.log(log_dict)

    def finish(self):
        """Finish WandB run"""
        wandb.finish()

