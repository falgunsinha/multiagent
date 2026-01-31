"""
Reward Shaping for Object Selection RL
Defines reward functions that encourage efficient and intelligent object picking.

Reward Components:
1. Distance-based reward: Pick closer objects first
2. Obstacle avoidance reward: Avoid objects near obstacles
3. Time efficiency reward: Complete task faster
4. Sequential optimality reward: Pick in optimal order
5. Success/failure rewards: Completion bonuses and penalties
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class RewardShaper:
    """
    Reward shaping for object selection task.
    """
    
    def __init__(
        self,
        distance_weight: float = 5.0,
        obstacle_weight: float = 3.0,
        time_penalty: float = -1.0,
        success_bonus: float = 20.0,
        failure_penalty: float = -10.0,
        sequential_bonus: float = 5.0,
        completion_time_bonus: float = 0.5
    ):
        """
        Initialize reward shaper with configurable weights.
        
        Args:
            distance_weight: Weight for distance-based reward
            obstacle_weight: Weight for obstacle avoidance reward
            time_penalty: Penalty per timestep (negative)
            success_bonus: Bonus for successful pick
            failure_penalty: Penalty for failed/invalid pick
            sequential_bonus: Bonus for picking in optimal order
            completion_time_bonus: Bonus per remaining step when completing
        """
        self.distance_weight = distance_weight
        self.obstacle_weight = obstacle_weight
        self.time_penalty = time_penalty
        self.success_bonus = success_bonus
        self.failure_penalty = failure_penalty
        self.sequential_bonus = sequential_bonus
        self.completion_time_bonus = completion_time_bonus
    
    def calculate_reward(
        self,
        action: int,
        object_position: np.ndarray,
        ee_position: np.ndarray,
        obstacle_score: float,
        all_object_positions: List[np.ndarray],
        objects_picked: List[int],
        is_first_pick: bool = False,
        is_last_pick: bool = False,
        remaining_steps: int = 0,
        pick_successful: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total reward for picking an object.
        
        Args:
            action: Index of object being picked
            object_position: Position of selected object
            ee_position: Current end-effector position
            obstacle_score: Obstacle proximity score (0-1)
            all_object_positions: Positions of all objects
            objects_picked: List of already picked object indices
            is_first_pick: Whether this is the first pick
            is_last_pick: Whether this is the last pick
            remaining_steps: Steps remaining in episode
            pick_successful: Whether the pick was successful
            
        Returns:
            total_reward: Total reward value
            reward_breakdown: Dictionary with individual reward components
        """
        reward_breakdown = {}
        
        # 1. Base success/failure reward
        if pick_successful:
            reward_breakdown["success"] = self.success_bonus
        else:
            reward_breakdown["failure"] = self.failure_penalty
            return self.failure_penalty, reward_breakdown
        
        # 2. Distance-based reward (exponential decay)
        distance = np.linalg.norm(object_position - ee_position)
        distance_reward = self.distance_weight * np.exp(-distance)
        reward_breakdown["distance"] = distance_reward
        
        # 3. Obstacle avoidance reward
        obstacle_reward = self.obstacle_weight * (1.0 - obstacle_score)
        reward_breakdown["obstacle"] = obstacle_reward
        
        # 4. Time penalty
        reward_breakdown["time"] = self.time_penalty
        
        # 5. Sequential optimality bonus
        if is_first_pick:
            # Bonus if first pick is the closest object
            distances = [np.linalg.norm(pos - ee_position) for pos in all_object_positions]
            unpicked_distances = [
                (i, dist) for i, dist in enumerate(distances)
                if i not in objects_picked
            ]
            if unpicked_distances:
                closest_idx = min(unpicked_distances, key=lambda x: x[1])[0]
                if action == closest_idx:
                    reward_breakdown["sequential"] = self.sequential_bonus
                else:
                    reward_breakdown["sequential"] = 0.0
        
        # 6. Completion time bonus
        if is_last_pick:
            time_bonus = remaining_steps * self.completion_time_bonus
            reward_breakdown["completion_time"] = time_bonus
        
        # Calculate total reward
        total_reward = sum(reward_breakdown.values())
        
        return total_reward, reward_breakdown
    
    def calculate_pick_difficulty(
        self,
        object_position: np.ndarray,
        ee_position: np.ndarray,
        obstacle_score: float,
        object_type: str = "cube"
    ) -> float:
        """
        Calculate difficulty score for picking an object.
        Lower score = easier to pick.
        
        Args:
            object_position: Position of object
            ee_position: Current end-effector position
            obstacle_score: Obstacle proximity score (0-1)
            object_type: Type of object (cube, cylinder, sphere)
            
        Returns:
            difficulty: Difficulty score (0-1, lower is easier)
        """
        # Distance component (normalized to 0-1)
        distance = np.linalg.norm(object_position - ee_position)
        distance_difficulty = min(distance / 1.0, 1.0)  # Normalize by 1m max distance
        
        # Obstacle component (already 0-1)
        obstacle_difficulty = obstacle_score
        
        # Object type difficulty
        type_difficulty = {
            "cube": 0.0,      # Easiest
            "cylinder": 0.2,  # Medium
            "sphere": 0.4     # Hardest (rolls)
        }.get(object_type, 0.0)
        
        # Weighted combination
        difficulty = (
            0.5 * distance_difficulty +
            0.3 * obstacle_difficulty +
            0.2 * type_difficulty
        )
        
        return difficulty
    
    def get_optimal_pick_order(
        self,
        object_positions: List[np.ndarray],
        ee_position: np.ndarray,
        obstacle_scores: List[float],
        object_types: List[str]
    ) -> List[int]:
        """
        Calculate optimal picking order based on difficulty.
        
        Args:
            object_positions: List of object positions
            ee_position: Current end-effector position
            obstacle_scores: List of obstacle scores
            object_types: List of object types
            
        Returns:
            optimal_order: List of object indices in optimal picking order
        """
        difficulties = []
        for i, (pos, obs_score, obj_type) in enumerate(
            zip(object_positions, obstacle_scores, object_types)
        ):
            difficulty = self.calculate_pick_difficulty(
                pos, ee_position, obs_score, obj_type
            )
            difficulties.append((i, difficulty))
        
        # Sort by difficulty (easiest first)
        optimal_order = [idx for idx, _ in sorted(difficulties, key=lambda x: x[1])]
        
        return optimal_order

