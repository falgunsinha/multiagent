"""
Reshuffling Decision Module

Determines when and why to reshuffle cubes based on:
- Essential conditions (must reshuffle)
- Efficiency conditions (should reshuffle)
- Strategic conditions (optimize globally)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ReshuffleReason(Enum):
    """Reasons for reshuffling"""
    # Essential (must reshuffle)
    UNREACHABLE_DISTANCE = "unreachable_distance"  # Distance > 0.85m or < 0.35m
    PATH_TOO_LONG = "path_too_long"  # RRT path > 1.8× Euclidean
    CROWDED_AREA = "crowded_area"  # 3+ nearby objects
    
    # Efficiency (should reshuffle)
    RRT_FAILURES = "rrt_failures"  # Recent planning failures
    LOW_CLEARANCE = "low_clearance"  # Path clearance < 0.3m
    FAR_CUBE = "far_cube"  # Distance > 1.2× average
    
    # Strategic (optimize globally)
    BATCH_RESHUFFLE = "batch_reshuffle"  # 3+ cubes too far
    GLOBAL_OPTIMIZATION = "global_optimization"  # Reduce total path length
    
    # No reshuffling needed
    NONE = "none"


@dataclass
class ReshuffleDecision:
    """Result of reshuffling decision"""
    should_reshuffle: bool
    reason: ReshuffleReason
    priority: int  # 0=none, 1=strategic, 2=efficiency, 3=essential
    target_cube_idx: Optional[int] = None
    target_position: Optional[np.ndarray] = None
    metadata: Dict = None  # Additional info (distances, clearances, etc.)


class ReshufflingDecisionModule:
    """
    Decides when to reshuffle cubes based on multiple conditions.
    
    This module implements rule-based logic that can be combined with
    learned MAPPO policy for optimal reshuffling decisions.
    """
    
    def __init__(
        self,
        # Essential thresholds
        min_reachable_distance: float = 0.35,
        max_reachable_distance: float = 0.85,
        path_length_ratio_threshold: float = 1.8,
        crowded_threshold: int = 3,
        
        # Efficiency thresholds
        rrt_failure_window: int = 3,
        min_clearance: float = 0.3,
        far_cube_ratio: float = 1.2,
        
        # Strategic thresholds
        batch_reshuffle_count: int = 3,
        global_optimization_threshold: float = 0.2,  # 20% improvement
    ):
        """
        Initialize reshuffling decision module.
        
        Args:
            min_reachable_distance: Minimum reachable distance (m)
            max_reachable_distance: Maximum reachable distance (m)
            path_length_ratio_threshold: Max ratio of RRT path to Euclidean distance
            crowded_threshold: Number of nearby objects to consider crowded
            rrt_failure_window: Number of recent RRT attempts to check
            min_clearance: Minimum path clearance (m)
            far_cube_ratio: Ratio to average distance to consider "far"
            batch_reshuffle_count: Number of far cubes to trigger batch reshuffle
            global_optimization_threshold: Min improvement ratio for global optimization
        """
        # Essential thresholds
        self.min_reachable_distance = min_reachable_distance
        self.max_reachable_distance = max_reachable_distance
        self.path_length_ratio_threshold = path_length_ratio_threshold
        self.crowded_threshold = crowded_threshold
        
        # Efficiency thresholds
        self.rrt_failure_window = rrt_failure_window
        self.min_clearance = min_clearance
        self.far_cube_ratio = far_cube_ratio
        
        # Strategic thresholds
        self.batch_reshuffle_count = batch_reshuffle_count
        self.global_optimization_threshold = global_optimization_threshold
        
        # History tracking
        self.rrt_failure_history = []  # List of (cube_idx, success) tuples
        self.path_clearance_history = {}  # cube_idx -> clearance
        
    def check_reshuffling_needed(
        self,
        cube_positions: np.ndarray,
        target_cube_idx: int,
        robot_position: np.ndarray,
        obstacle_positions: np.ndarray,
        rrt_path: Optional[np.ndarray] = None,
        rrt_success: bool = True,
        path_clearance: Optional[float] = None,
    ) -> ReshuffleDecision:
        """
        Check if reshuffling is needed for the target cube.
        
        Args:
            cube_positions: Array of cube positions (N, 3)
            target_cube_idx: Index of target cube
            robot_position: Current robot end-effector position (3,)
            obstacle_positions: Array of obstacle positions (M, 3)
            rrt_path: RRT path if available (P, 3)
            rrt_success: Whether RRT planning succeeded
            path_clearance: Minimum clearance along path
            
        Returns:
            ReshuffleDecision with should_reshuffle flag and reason
        """
        target_pos = cube_positions[target_cube_idx]
        
        # Calculate Euclidean distance
        euclidean_dist = np.linalg.norm(target_pos - robot_position)
        
        # ESSENTIAL CONDITION 1: Unreachable distance
        if euclidean_dist < self.min_reachable_distance or euclidean_dist > self.max_reachable_distance:
            return ReshuffleDecision(
                should_reshuffle=True,
                reason=ReshuffleReason.UNREACHABLE_DISTANCE,
                priority=3,
                target_cube_idx=target_cube_idx,
                metadata={"distance": euclidean_dist}
            )
        
        # ESSENTIAL CONDITION 2: Path too long
        if rrt_path is not None and len(rrt_path) > 0:
            path_length = self._calculate_path_length(rrt_path)
            path_ratio = path_length / euclidean_dist if euclidean_dist > 0 else float('inf')
            
            if path_ratio > self.path_length_ratio_threshold:
                return ReshuffleDecision(
                    should_reshuffle=True,
                    reason=ReshuffleReason.PATH_TOO_LONG,
                    priority=3,
                    target_cube_idx=target_cube_idx,
                    metadata={"path_length": path_length, "euclidean": euclidean_dist, "ratio": path_ratio}
                )
        
        # ESSENTIAL CONDITION 3: Crowded area
        nearby_count = self._count_nearby_objects(target_pos, cube_positions, obstacle_positions, radius=0.15)
        if nearby_count >= self.crowded_threshold:
            return ReshuffleDecision(
                should_reshuffle=True,
                reason=ReshuffleReason.CROWDED_AREA,
                priority=3,
                target_cube_idx=target_cube_idx,
                metadata={"nearby_count": nearby_count}
            )

        # EFFICIENCY CONDITION 1: Recent RRT failures
        self.rrt_failure_history.append((target_cube_idx, rrt_success))
        if len(self.rrt_failure_history) > 100:  # Keep last 100 attempts
            self.rrt_failure_history.pop(0)

        recent_failures = self._count_recent_failures(target_cube_idx, self.rrt_failure_window)
        if recent_failures >= self.rrt_failure_window:
            return ReshuffleDecision(
                should_reshuffle=True,
                reason=ReshuffleReason.RRT_FAILURES,
                priority=2,
                target_cube_idx=target_cube_idx,
                metadata={"recent_failures": recent_failures}
            )

        # EFFICIENCY CONDITION 2: Low path clearance
        if path_clearance is not None:
            self.path_clearance_history[target_cube_idx] = path_clearance
            if path_clearance < self.min_clearance:
                return ReshuffleDecision(
                    should_reshuffle=True,
                    reason=ReshuffleReason.LOW_CLEARANCE,
                    priority=2,
                    target_cube_idx=target_cube_idx,
                    metadata={"clearance": path_clearance}
                )

        # EFFICIENCY CONDITION 3: Far cube (compared to average)
        avg_distance = self._calculate_average_distance(cube_positions, robot_position)
        if euclidean_dist > self.far_cube_ratio * avg_distance:
            return ReshuffleDecision(
                should_reshuffle=True,
                reason=ReshuffleReason.FAR_CUBE,
                priority=2,
                target_cube_idx=target_cube_idx,
                metadata={"distance": euclidean_dist, "avg_distance": avg_distance}
            )

        # STRATEGIC CONDITION 1: Batch reshuffling (multiple far cubes)
        far_cubes = self._count_far_cubes(cube_positions, robot_position, avg_distance)
        if far_cubes >= self.batch_reshuffle_count:
            return ReshuffleDecision(
                should_reshuffle=True,
                reason=ReshuffleReason.BATCH_RESHUFFLE,
                priority=1,
                target_cube_idx=target_cube_idx,
                metadata={"far_cubes": far_cubes}
            )

        # STRATEGIC CONDITION 2: Global path optimization
        # (This would require more complex analysis - simplified here)
        # Check if reshuffling could reduce total remaining path length

        # No reshuffling needed
        return ReshuffleDecision(
            should_reshuffle=False,
            reason=ReshuffleReason.NONE,
            priority=0,
            target_cube_idx=target_cube_idx,
            metadata={}
        )

    def _calculate_path_length(self, path: np.ndarray) -> float:
        """Calculate total path length"""
        if len(path) < 2:
            return 0.0
        diffs = np.diff(path, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return np.sum(distances)

    def _count_nearby_objects(
        self,
        position: np.ndarray,
        cube_positions: np.ndarray,
        obstacle_positions: np.ndarray,
        radius: float
    ) -> int:
        """Count objects within radius of position"""
        count = 0

        # Count nearby cubes
        for cube_pos in cube_positions:
            if np.linalg.norm(cube_pos - position) < radius and not np.allclose(cube_pos, position):
                count += 1

        # Count nearby obstacles
        for obs_pos in obstacle_positions:
            if np.linalg.norm(obs_pos - position) < radius:
                count += 1

        return count

    def _count_recent_failures(self, cube_idx: int, window: int) -> int:
        """Count recent RRT failures for specific cube"""
        recent = [success for idx, success in self.rrt_failure_history[-window:] if idx == cube_idx]
        return sum(1 for success in recent if not success)

    def _calculate_average_distance(self, cube_positions: np.ndarray, robot_position: np.ndarray) -> float:
        """Calculate average distance to all cubes"""
        if len(cube_positions) == 0:
            return 0.0
        distances = [np.linalg.norm(pos - robot_position) for pos in cube_positions]
        return np.mean(distances)

    def _count_far_cubes(self, cube_positions: np.ndarray, robot_position: np.ndarray, avg_distance: float) -> int:
        """Count cubes that are far from robot"""
        count = 0
        for pos in cube_positions:
            dist = np.linalg.norm(pos - robot_position)
            if dist > self.far_cube_ratio * avg_distance:
                count += 1
        return count

    def get_far_cube_indices(self, cube_positions: np.ndarray, robot_position: np.ndarray, avg_distance: float, picked_cubes: set, reshuffle_count: dict, max_reshuffles: int = 2) -> List[int]:
        """
        Get indices of cubes that are far from robot and eligible for reshuffling.

        Args:
            cube_positions: Array of cube positions (N, 3)
            robot_position: Robot position (3,)
            avg_distance: Average distance to all cubes
            picked_cubes: Set of already picked cube indices
            reshuffle_count: Dict mapping cube_idx to reshuffle count
            max_reshuffles: Maximum reshuffles allowed per cube

        Returns:
            List of cube indices that are far and eligible for reshuffling
        """
        far_cube_indices = []
        for idx, pos in enumerate(cube_positions):
            # Skip if cube is already picked
            if idx in picked_cubes:
                continue
            # Skip if cube has been reshuffled max times
            if reshuffle_count.get(idx, 0) >= max_reshuffles:
                continue
            # Check if cube is far
            dist = np.linalg.norm(pos - robot_position)
            if dist > self.far_cube_ratio * avg_distance:
                far_cube_indices.append(idx)
        return far_cube_indices

    def reset(self):
        """Reset history tracking"""
        self.rrt_failure_history = []
        self.path_clearance_history = {}

    def get_statistics(self) -> Dict:
        """Get statistics about reshuffling decisions"""
        total_attempts = len(self.rrt_failure_history)
        failures = sum(1 for _, success in self.rrt_failure_history if not success)

        return {
            "total_rrt_attempts": total_attempts,
            "total_rrt_failures": failures,
            "failure_rate": failures / total_attempts if total_attempts > 0 else 0.0,
            "avg_clearance": np.mean(list(self.path_clearance_history.values())) if self.path_clearance_history else 0.0,
        }

