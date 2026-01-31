"""
GPU-Accelerated MPPI (Model Predictive Path Integral) Optimizer
Uses PyTorch + CUDA for fast parallel trajectory sampling and evaluation

This is similar to cuRobo's MPPI but without cuRobo dependency.
"""

import torch
import numpy as np
from typing import Callable, Optional, Tuple


class TorchMPPI:
    """
    GPU-Accelerated MPPI optimizer using PyTorch
    
    Samples and evaluates thousands of trajectories in parallel on GPU.
    Essential for real-time MPC performance.
    """
    
    def __init__(
        self,
        horizon: int = 20,
        num_samples: int = 500,
        lambda_: float = 1.0,
        noise_sigma: float = 0.3,
        control_dim: int = 7,
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        dt: float = 0.05,
        device: str = "cuda:0"
    ):
        """
        Initialize GPU-accelerated MPPI optimizer
        
        Args:
            horizon: Planning horizon (number of timesteps)
            num_samples: Number of trajectory samples per iteration
            lambda_: Temperature parameter for cost weighting
            noise_sigma: Standard deviation of control noise
            control_dim: Dimension of control vector (7 for Franka)
            control_bounds: Tuple of (lower_bounds, upper_bounds)
            dt: Time step between controls
            device: PyTorch device ("cuda:0" or "cpu")
        """
        self.horizon = horizon
        self.num_samples = num_samples
        self.lambda_ = lambda_
        self.noise_sigma = noise_sigma
        self.control_dim = control_dim
        self.dt = dt
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[TorchMPPI] Using device: {self.device}")
        print(f"[TorchMPPI] Horizon: {horizon}, Samples: {num_samples}")
        
        # Control bounds
        if control_bounds is not None:
            self.control_lower = torch.tensor(control_bounds[0], device=self.device, dtype=torch.float32)
            self.control_upper = torch.tensor(control_bounds[1], device=self.device, dtype=torch.float32)
        else:
            self.control_lower = None
            self.control_upper = None
        
        # Initialize control sequence (horizon, control_dim)
        self.control_sequence = torch.zeros((horizon, control_dim), device=self.device, dtype=torch.float32)
        
    def optimize(
        self,
        initial_state: torch.Tensor,
        cost_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        dynamics_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        num_iterations: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run MPPI optimization on GPU
        
        Args:
            initial_state: Initial state (control_dim,) on GPU
            cost_function: Function that computes cost for batch of trajectories
                          Input: states (num_samples, horizon+1, control_dim)
                                 controls (num_samples, horizon, control_dim)
                          Output: costs (num_samples,)
            dynamics_function: Function that computes next state
                              Input: state (num_samples, control_dim)
                                     control (num_samples, control_dim)
                              Output: next_state (num_samples, control_dim)
            num_iterations: Number of MPPI iterations
            
        Returns:
            Tuple of:
                - best_control_sequence: Optimized controls (horizon, control_dim)
                - best_trajectory: Optimized state trajectory (horizon+1, control_dim)
        """
        for iteration in range(num_iterations):
            # Sample control noise (num_samples, horizon, control_dim)
            noise = torch.randn(
                (self.num_samples, self.horizon, self.control_dim),
                device=self.device, dtype=torch.float32
            ) * self.noise_sigma
            
            # Perturbed controls = current_sequence + noise
            # Broadcast control_sequence to (num_samples, horizon, control_dim)
            perturbed_controls = self.control_sequence.unsqueeze(0) + noise
            
            # Clamp to control bounds
            if self.control_lower is not None:
                perturbed_controls = torch.clamp(
                    perturbed_controls,
                    self.control_lower.unsqueeze(0).unsqueeze(0),
                    self.control_upper.unsqueeze(0).unsqueeze(0)
                )
            
            # Rollout trajectories (num_samples, horizon+1, control_dim)
            trajectories = self._rollout_trajectories(
                initial_state, perturbed_controls, dynamics_function
            )
            
            # Compute costs for all trajectories (num_samples,)
            costs = cost_function(trajectories, perturbed_controls)
            
            # Compute weights using softmax with temperature
            weights = torch.softmax(-costs / self.lambda_, dim=0)
            
            # Update control sequence as weighted average
            # weights: (num_samples,) -> (num_samples, 1, 1)
            # perturbed_controls: (num_samples, horizon, control_dim)
            weighted_controls = weights.unsqueeze(1).unsqueeze(2) * perturbed_controls
            self.control_sequence = weighted_controls.sum(dim=0)

            # Disabled verbose logging - uncomment for debugging
            # if iteration == 0:
            #     print(f"[TorchMPPI] Iter {iteration}: min_cost={costs.min().item():.4f}, "
            #           f"mean_cost={costs.mean().item():.4f}")
        
        # Rollout best trajectory
        best_trajectory = self._rollout_single_trajectory(
            initial_state, self.control_sequence, dynamics_function
        )
        
        return self.control_sequence, best_trajectory
    
    def _rollout_trajectories(
        self,
        initial_state: torch.Tensor,
        controls: torch.Tensor,
        dynamics_function: Callable
    ) -> torch.Tensor:
        """
        Rollout batch of trajectories in parallel on GPU
        
        Args:
            initial_state: Initial state (control_dim,)
            controls: Control sequences (num_samples, horizon, control_dim)
            dynamics_function: Dynamics model
            
        Returns:
            State trajectories (num_samples, horizon+1, control_dim)
        """
        num_samples = controls.shape[0]
        trajectories = torch.zeros(
            (num_samples, self.horizon + 1, self.control_dim),
            device=self.device, dtype=torch.float32
        )
        
        # Set initial state for all samples
        trajectories[:, 0, :] = initial_state.unsqueeze(0).expand(num_samples, -1)
        
        # Rollout dynamics
        for t in range(self.horizon):
            trajectories[:, t + 1, :] = dynamics_function(
                trajectories[:, t, :], controls[:, t, :]
            )
        
        return trajectories

    def _rollout_single_trajectory(
        self,
        initial_state: torch.Tensor,
        controls: torch.Tensor,
        dynamics_function: Callable
    ) -> torch.Tensor:
        """
        Rollout single trajectory

        Args:
            initial_state: Initial state (control_dim,)
            controls: Control sequence (horizon, control_dim)
            dynamics_function: Dynamics model

        Returns:
            State trajectory (horizon+1, control_dim)
        """
        trajectory = torch.zeros(
            (self.horizon + 1, self.control_dim),
            device=self.device, dtype=torch.float32
        )

        trajectory[0, :] = initial_state

        for t in range(self.horizon):
            trajectory[t + 1, :] = dynamics_function(
                trajectory[t, :].unsqueeze(0), controls[t, :].unsqueeze(0)
            ).squeeze(0)

        return trajectory

    def get_action(self) -> torch.Tensor:
        """
        Get first control action from optimized sequence

        Returns:
            First control action (control_dim,)
        """
        return self.control_sequence[0, :]

    def shift_sequence(self):
        """
        Shift control sequence for receding horizon control
        Moves sequence forward by 1 timestep and adds zero at end
        """
        self.control_sequence = torch.roll(self.control_sequence, -1, dims=0)
        self.control_sequence[-1, :] = 0.0

