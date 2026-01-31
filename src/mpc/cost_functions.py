"""
Cost Functions for MPC
Defines various cost terms for trajectory optimization
"""

import numpy as np
from typing import List, Optional, Tuple


class MPCCostFunction:
    """
    Combined cost function for MPC trajectory optimization
    
    Includes:
    - Goal reaching cost (position + orientation)
    - Obstacle avoidance cost
    - Control effort cost (smoothness)
    - Joint limit cost
    """
    
    def __init__(
        self,
        goal_position_weight: float = 100.0,
        goal_orientation_weight: float = 50.0,
        obstacle_weight: float = 500.0,
        control_weight: float = 0.1,
        joint_limit_weight: float = 10.0,
        obstacle_margin: float = 0.15,  # Safety margin around obstacles (meters)
        joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        """
        Initialize cost function
        
        Args:
            goal_position_weight: Weight for position error
            goal_orientation_weight: Weight for orientation error
            obstacle_weight: Weight for obstacle avoidance
            control_weight: Weight for control effort (smoothness)
            joint_limit_weight: Weight for joint limit violations
            obstacle_margin: Safety margin around obstacles
            joint_limits: Tuple of (lower, upper) joint limits
        """
        self.goal_position_weight = goal_position_weight
        self.goal_orientation_weight = goal_orientation_weight
        self.obstacle_weight = obstacle_weight
        self.control_weight = control_weight
        self.joint_limit_weight = joint_limit_weight
        self.obstacle_margin = obstacle_margin
        
        # Franka joint limits
        if joint_limits is not None:
            self.joint_lower, self.joint_upper = joint_limits
        else:
            self.joint_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
            self.joint_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        
        # Goal state (set externally)
        self.goal_ee_position = None
        self.goal_ee_orientation = None
        
        # Obstacles (set externally)
        self.obstacles = []  # List of obstacle positions (x, y, z)
        
        # Forward kinematics function (set externally)
        self.fk_function = None
    
    def set_goal(self, position: np.ndarray, orientation: Optional[np.ndarray] = None):
        """Set goal end-effector pose"""
        self.goal_ee_position = position
        self.goal_ee_orientation = orientation
    
    def set_obstacles(self, obstacles: List[np.ndarray]):
        """Set obstacle positions"""
        self.obstacles = obstacles
    
    def set_fk_function(self, fk_function):
        """Set forward kinematics function"""
        self.fk_function = fk_function
    
    def compute_cost(self, trajectory: np.ndarray, controls: np.ndarray) -> float:
        """
        Compute total cost for a trajectory
        
        Args:
            trajectory: State trajectory (horizon+1, state_dim)
            controls: Control sequence (horizon, control_dim)
            
        Returns:
            Total cost (scalar)
        """
        total_cost = 0.0
        horizon = len(controls)
        
        # 1. Goal reaching cost (terminal cost on final state)
        if self.goal_ee_position is not None and self.fk_function is not None:
            final_state = trajectory[-1]
            final_ee_pos, final_ee_quat = self.fk_function(final_state)
            
            # Position error
            position_error = np.linalg.norm(final_ee_pos - self.goal_ee_position)
            total_cost += self.goal_position_weight * position_error ** 2
            
            # Orientation error (if goal orientation is specified)
            if self.goal_ee_orientation is not None:
                # Quaternion distance (1 - |q1 · q2|)
                quat_dot = np.abs(np.dot(final_ee_quat, self.goal_ee_orientation))
                orientation_error = 1.0 - quat_dot
                total_cost += self.goal_orientation_weight * orientation_error ** 2
        
        # 2. Obstacle avoidance cost (running cost over trajectory)
        if len(self.obstacles) > 0 and self.fk_function is not None:
            for t in range(horizon + 1):
                state = trajectory[t]
                ee_pos, _ = self.fk_function(state)
                
                # Check distance to each obstacle
                for obs_pos in self.obstacles:
                    distance = np.linalg.norm(ee_pos - obs_pos)
                    
                    # Exponential barrier cost (high cost when close to obstacle)
                    if distance < self.obstacle_margin:
                        # Exponential penalty
                        obstacle_cost = np.exp(-10.0 * (distance / self.obstacle_margin))
                        total_cost += self.obstacle_weight * obstacle_cost
        
        # 3. Control effort cost (smoothness)
        control_effort = np.sum(controls ** 2)
        total_cost += self.control_weight * control_effort
        
        # 4. Joint limit cost
        for t in range(horizon + 1):
            state = trajectory[t]
            
            # Soft barrier for joint limits
            lower_violation = np.maximum(0, self.joint_lower - state)
            upper_violation = np.maximum(0, state - self.joint_upper)
            
            joint_limit_cost = np.sum(lower_violation ** 2) + np.sum(upper_violation ** 2)
            total_cost += self.joint_limit_weight * joint_limit_cost
        
        return total_cost
    
    def __call__(self, trajectory: np.ndarray, controls: np.ndarray) -> float:
        """Allow cost function to be called directly"""
        return self.compute_cost(trajectory, controls)


def simple_dynamics(state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
    """
    Simple kinematic dynamics model for robot joints
    
    Assumes control is joint velocity command
    
    Args:
        state: Current joint positions (7,)
        control: Joint velocity command (7,)
        dt: Time step
        
    Returns:
        Next joint positions (7,)
    """
    # Simple Euler integration: q_next = q_current + v * dt
    next_state = state + control * dt
    return next_state

