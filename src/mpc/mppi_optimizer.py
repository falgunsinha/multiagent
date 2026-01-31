"""
MPPI (Model Predictive Path Integral) Optimizer
Sampling-based optimization for MPC without gradients
Based on information-theoretic control principles
"""

import numpy as np
from typing import Callable, Optional, Tuple


class MPPIOptimizer:
    """
    Model Predictive Path Integral (MPPI) optimizer
    
    MPPI is a sampling-based optimization method that:
    1. Samples multiple trajectory rollouts from a control distribution
    2. Evaluates cost for each rollout
    3. Weights rollouts by their cost using information-theoretic principles
    4. Updates control sequence as weighted average of samples
    
    Key advantages:
    - Derivative-free (no gradients needed)
    - Handles non-convex cost functions
    - Naturally incorporates stochasticity
    - Parallelizable (can run on GPU)
    """
    
    def __init__(
        self,
        horizon: int = 20,
        num_samples: int = 1000,
        lambda_: float = 1.0,
        noise_sigma: float = 0.5,
        control_dim: int = 7,
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        dt: float = 0.1,
        temperature: float = 1.0
    ):
        """
        Initialize MPPI optimizer
        
        Args:
            horizon: Planning horizon (number of timesteps)
            num_samples: Number of trajectory samples per iteration
            lambda_: Temperature parameter for cost weighting (lower = more exploitation)
            noise_sigma: Standard deviation of control noise
            control_dim: Dimension of control vector (7 for Franka joints)
            control_bounds: Tuple of (lower_bounds, upper_bounds) for controls
            dt: Time step between controls
            temperature: Temperature for softmax weighting
        """
        self.horizon = horizon
        self.num_samples = num_samples
        self.lambda_ = lambda_
        self.noise_sigma = noise_sigma
        self.control_dim = control_dim
        self.dt = dt
        self.temperature = temperature
        
        # Control bounds (joint limits)
        if control_bounds is not None:
            self.control_lower, self.control_upper = control_bounds
        else:
            # Default Franka joint limits (radians)
            self.control_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
            self.control_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        
        # Initialize control sequence (mean of distribution)
        self.control_sequence = np.zeros((horizon, control_dim))
        
        # Warm start: store previous solution
        self.prev_solution = None
        
    def optimize(
        self,
        current_state: np.ndarray,
        cost_function: Callable,
        dynamics_function: Callable,
        num_iterations: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run MPPI optimization to find optimal control sequence
        
        Args:
            current_state: Current robot state (joint positions)
            cost_function: Function that computes cost for a trajectory
                           Signature: cost_function(states, controls, obstacles) -> float
            dynamics_function: Function that simulates forward dynamics
                              Signature: dynamics_function(state, control, dt) -> next_state
            num_iterations: Number of MPPI iterations
            
        Returns:
            Tuple of (optimal_control_sequence, predicted_trajectory)
        """
        # Warm start: shift previous solution
        if self.prev_solution is not None:
            self.control_sequence[:-1] = self.prev_solution[1:]
            self.control_sequence[-1] = self.prev_solution[-1]
        
        for iteration in range(num_iterations):
            # Sample control noise
            noise = np.random.normal(0, self.noise_sigma, (self.num_samples, self.horizon, self.control_dim))
            
            # Generate perturbed control sequences
            perturbed_controls = self.control_sequence[np.newaxis, :, :] + noise
            
            # Clip to control bounds
            perturbed_controls = np.clip(perturbed_controls, self.control_lower, self.control_upper)
            
            # Rollout trajectories and compute costs
            costs = np.zeros(self.num_samples)
            trajectories = np.zeros((self.num_samples, self.horizon + 1, self.control_dim))
            
            for i in range(self.num_samples):
                state = current_state.copy()
                trajectories[i, 0] = state
                
                for t in range(self.horizon):
                    # Apply control and simulate dynamics
                    control = perturbed_controls[i, t]
                    state = dynamics_function(state, control, self.dt)
                    trajectories[i, t + 1] = state
                
                # Compute cost for this trajectory
                costs[i] = cost_function(trajectories[i], perturbed_controls[i])
            
            # Compute weights using softmax with temperature
            # Lower cost = higher weight
            min_cost = np.min(costs)
            exp_costs = np.exp(-(costs - min_cost) / (self.lambda_ * self.temperature))
            weights = exp_costs / np.sum(exp_costs)
            
            # Update control sequence as weighted average
            self.control_sequence = np.sum(weights[:, np.newaxis, np.newaxis] * perturbed_controls, axis=0)
            
            # Clip to bounds
            self.control_sequence = np.clip(self.control_sequence, self.control_lower, self.control_upper)
        
        # Store solution for warm start
        self.prev_solution = self.control_sequence.copy()
        
        # Compute final trajectory with optimal controls
        optimal_trajectory = np.zeros((self.horizon + 1, self.control_dim))
        state = current_state.copy()
        optimal_trajectory[0] = state
        
        for t in range(self.horizon):
            state = dynamics_function(state, self.control_sequence[t], self.dt)
            optimal_trajectory[t + 1] = state
        
        return self.control_sequence, optimal_trajectory
    
    def reset(self):
        """Reset optimizer (clear warm start)"""
        self.control_sequence = np.zeros((self.horizon, self.control_dim))
        self.prev_solution = None

