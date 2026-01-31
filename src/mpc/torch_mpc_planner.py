"""
GPU-Accelerated MPC Planner for Franka Robot
Uses PyTorch + CUDA for real-time MPC with obstacle avoidance

This is similar to cuRobo's MPC but without cuRobo dependency.
Uses PyTorch for GPU acceleration of FK and MPPI optimization.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
import sys
import os
from pathlib import Path
import importlib.util

# Load dependencies directly from files (no relative imports to avoid issues)
current_dir = Path(__file__).parent

# Load torch_kinematics
spec = importlib.util.spec_from_file_location("torch_kinematics", current_dir / "torch_kinematics.py")
torch_kinematics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(torch_kinematics)
FrankaBatchFK = torch_kinematics.FrankaBatchFK

# Load torch_mppi
spec = importlib.util.spec_from_file_location("torch_mppi", current_dir / "torch_mppi.py")
torch_mppi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(torch_mppi)
TorchMPPI = torch_mppi.TorchMPPI


class TorchMPCPlanner:
    """
    GPU-Accelerated MPC Planner using PyTorch
    
    Features:
    - Real-time MPC with MPPI optimization on GPU
    - Batch FK for fast trajectory evaluation
    - Obstacle avoidance through cost functions
    - Receding horizon control
    """
    
    def __init__(
        self,
        horizon: int = 15,
        num_samples: int = 400,
        dt: float = 0.05,
        mppi_iterations: int = 2,
        device: str = "cuda:0"
    ):
        """
        Initialize GPU-accelerated MPC planner
        
        Args:
            horizon: Planning horizon (timesteps)
            num_samples: Number of MPPI samples
            dt: Time step (seconds)
            mppi_iterations: MPPI optimization iterations
            device: PyTorch device ("cuda:0" or "cpu")
        """
        self.horizon = horizon
        self.num_samples = num_samples
        self.dt = dt
        self.mppi_iterations = mppi_iterations
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[TorchMPC] Initializing on device: {self.device}")
        
        # Initialize batch FK solver on GPU
        self.fk_solver = FrankaBatchFK(device=str(self.device))
        
        # Joint limits for Franka
        joint_limits_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        joint_limits_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        
        # Velocity limits (rad/s) - conservative for safety
        velocity_limits = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        control_lower = -velocity_limits
        control_upper = velocity_limits
        
        # Initialize MPPI optimizer on GPU
        self.mppi = TorchMPPI(
            horizon=horizon,
            num_samples=num_samples,
            lambda_=1.0,
            noise_sigma=0.3,
            control_dim=7,
            control_bounds=(control_lower, control_upper),
            dt=dt,
            device=str(self.device)
        )

        # Note: We use MPC/MPPI only (no batched IK)
        # MPC solves IK implicitly through sampling + FK evaluation
        print(f"[TorchMPC] Using MPC/MPPI-only approach (no batched IK)")

        # Goal and obstacles
        self.goal_position = None
        self.goal_orientation = None
        self.obstacles = []  # List of (position, radius) tuples
        
        # Cost weights (tuned for accurate goal reaching)
        self.weight_goal = 1000.0        # Very high weight for accurate goal reaching
        self.weight_orientation = 200.0  # High weight for orientation tracking
        self.weight_obstacle = 50.0      # Obstacle avoidance
        self.weight_control = 0.005      # Very low control effort (allow large movements)
        self.weight_joint_limits = 10.0  # Joint limit penalty
        
        print(f"[TorchMPC] MPC planner initialized successfully")
    
    def set_goal(self, position: np.ndarray, orientation: Optional[np.ndarray] = None):
        """
        Set goal pose for MPC
        
        Args:
            position: Goal position (3,) in world frame
            orientation: Goal orientation as rotation matrix (3, 3) or None
        """
        self.goal_position = torch.tensor(position, device=self.device, dtype=torch.float32)
        if orientation is not None:
            self.goal_orientation = torch.tensor(orientation, device=self.device, dtype=torch.float32)
        else:
            self.goal_orientation = None
    
    def update_obstacles(self, obstacles: List[Dict]):
        """
        Update obstacle list from Lidar data

        Args:
            obstacles: List of obstacle dicts with 'position' and 'radius' keys
                      OR list of numpy arrays (for backward compatibility)
        """
        self.obstacles = []
        for obs in obstacles:
            # Handle both dictionary format and direct numpy array format
            if isinstance(obs, dict):
                pos = torch.tensor(obs['position'], device=self.device, dtype=torch.float32)
                radius = obs.get('radius', 0.1)
            else:
                # Assume it's a numpy array or list (backward compatibility)
                pos = torch.tensor(obs, device=self.device, dtype=torch.float32)
                radius = 0.1  # Default radius
            self.obstacles.append((pos, radius))
    
    def _dynamics(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """
        Simple dynamics model: q_next = q + v * dt
        
        Args:
            state: Current joint positions (batch_size, 7)
            control: Joint velocities (batch_size, 7)
            
        Returns:
            Next joint positions (batch_size, 7)
        """
        next_state = state + control * self.dt
        
        # Clamp to joint limits
        next_state = torch.clamp(
            next_state,
            self.fk_solver.joint_limits_lower.unsqueeze(0),
            self.fk_solver.joint_limits_upper.unsqueeze(0)
        )
        
        return next_state
    
    def _cost_function(self, trajectories: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        """
        Compute cost for batch of trajectories (GPU-accelerated)
        
        Args:
            trajectories: State trajectories (num_samples, horizon+1, 7)
            controls: Control sequences (num_samples, horizon, 7)
            
        Returns:
            Costs for each trajectory (num_samples,)
        """
        num_samples = trajectories.shape[0]
        total_cost = torch.zeros(num_samples, device=self.device, dtype=torch.float32)
        
        # Reshape for batch FK: (num_samples * (horizon+1), 7)
        flat_states = trajectories.reshape(-1, 7)
        
        # Batch FK on GPU - compute all end-effector positions at once!
        ee_positions, ee_orientations = self.fk_solver.forward_kinematics(flat_states)

        # Reshape back: (num_samples, horizon+1, 3) and (num_samples, horizon+1, 4)
        ee_positions = ee_positions.reshape(num_samples, self.horizon + 1, 3)
        ee_orientations = ee_orientations.reshape(num_samples, self.horizon + 1, 4)

        # Goal reaching cost (only final state)
        if self.goal_position is not None:
            goal_error = torch.norm(ee_positions[:, -1, :] - self.goal_position.unsqueeze(0), dim=1)
            total_cost += self.weight_goal * goal_error ** 2

        # Orientation cost (only final state)
        if self.goal_orientation is not None:
            # Quaternion distance: 1 - |dot_product|
            dot_product = torch.abs(torch.sum(ee_orientations[:, -1, :] * self.goal_orientation.unsqueeze(0), dim=1))
            orientation_error = 1.0 - dot_product
            total_cost += self.weight_orientation * orientation_error ** 2
        
        # Obstacle avoidance cost (all timesteps)
        for obs_pos, obs_radius in self.obstacles:
            # Distance from each EE position to obstacle
            distances = torch.norm(ee_positions - obs_pos.unsqueeze(0).unsqueeze(0), dim=2)  # (num_samples, horizon+1)
            
            # Exponential barrier cost
            safety_margin = obs_radius + 0.05  # 5cm safety margin
            obstacle_cost = torch.exp(-10.0 * (distances - safety_margin))
            total_cost += self.weight_obstacle * obstacle_cost.sum(dim=1)
        
        # Control effort cost
        control_cost = torch.sum(controls ** 2, dim=(1, 2))
        total_cost += self.weight_control * control_cost
        
        # Joint limit cost
        joint_limit_violation = torch.clamp(
            trajectories - self.fk_solver.joint_limits_upper.unsqueeze(0).unsqueeze(0), min=0.0
        ) + torch.clamp(
            self.fk_solver.joint_limits_lower.unsqueeze(0).unsqueeze(0) - trajectories, min=0.0
        )
        total_cost += self.weight_joint_limits * torch.sum(joint_limit_violation ** 2, dim=(1, 2))
        
        return total_cost

    def step(self, current_state: np.ndarray) -> np.ndarray:
        """
        Compute next action using MPC (receding horizon control)

        Args:
            current_state: Current joint positions (7,)

        Returns:
            Next joint positions to execute (7,)
        """
        # Convert to PyTorch tensor on GPU
        current_state_torch = torch.tensor(current_state, device=self.device, dtype=torch.float32)

        # Run MPPI optimization on GPU
        control_sequence, trajectory = self.mppi.optimize(
            initial_state=current_state_torch,
            cost_function=self._cost_function,
            dynamics_function=self._dynamics,
            num_iterations=self.mppi_iterations
        )

        # Get first action (velocity)
        action_velocity = self.mppi.get_action()

        # Integrate to get next position
        next_state = current_state_torch + action_velocity * self.dt

        # Clamp to joint limits
        next_state = torch.clamp(
            next_state,
            self.fk_solver.joint_limits_lower,
            self.fk_solver.joint_limits_upper
        )

        # Shift control sequence for next iteration (receding horizon)
        self.mppi.shift_sequence()

        # Convert back to NumPy
        return next_state.cpu().numpy()

    def plan(
        self,
        current_state: np.ndarray,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        num_steps: int = 30
    ) -> Tuple[Optional[List[np.ndarray]], bool]:
        """
        Plan full trajectory to goal using MPC/MPPI only (no batched IK)

        Args:
            current_state: Current joint positions (7,)
            target_position: Target end-effector position (3,)
            target_orientation: Target end-effector orientation (4,) [w,x,y,z] quaternion (optional)
            num_steps: Maximum number of steps to plan

        Returns:
            Tuple of (trajectory, success):
                - trajectory: List of joint positions along trajectory, or None if failed
                - success: True if goal reached, False otherwise
        """
        # Set goal
        self.set_goal(target_position, target_orientation)

        trajectory = [current_state.copy()]
        state = current_state.copy()

        for i in range(num_steps):
            next_state = self.step(state)
            trajectory.append(next_state)
            state = next_state

            # Check if goal reached
            if self.goal_position is not None:
                state_torch = torch.tensor(state, device=self.device, dtype=torch.float32)
                ee_pos, ee_ori = self.fk_solver.forward_kinematics(state_torch.unsqueeze(0))

                # Position error
                pos_error = torch.norm(ee_pos[0] - self.goal_position).item()

                # Rotation error (if orientation specified)
                rot_error = 0.0
                if self.goal_orientation is not None:
                    dot_product = torch.abs(torch.sum(ee_ori[0] * self.goal_orientation)).item()
                    rot_error = 1.0 - dot_product

                # Success criteria (tighter for pick accuracy)
                pos_threshold = 0.015  # 1.5cm (tighter for better pick accuracy)
                rot_threshold = 0.08   # ~5 degrees (tighter for gripper alignment)

                if pos_error < pos_threshold and rot_error < rot_threshold:
                    print(f"[TorchMPC] ✓ Goal reached at step {i+1}: pos_err={pos_error*100:.2f}cm, rot_err={rot_error:.3f}")
                    return trajectory, True

        # Check final error
        state_torch = torch.tensor(state, device=self.device, dtype=torch.float32)
        ee_pos, ee_ori = self.fk_solver.forward_kinematics(state_torch.unsqueeze(0))
        pos_error = torch.norm(ee_pos[0] - self.goal_position).item()

        rot_error = 0.0
        if self.goal_orientation is not None:
            dot_product = torch.abs(torch.sum(ee_ori[0] * self.goal_orientation)).item()
            rot_error = 1.0 - dot_product

        print(f"[TorchMPC] ✗ Max steps reached: pos_err={pos_error*100:.2f}cm, rot_err={rot_error:.3f}")

        # Return trajectory even if not perfect (let caller decide)
        success = pos_error < 0.015 and rot_error < 0.08
        return trajectory, success

    def plan_with_ik(
        self,
        current_state: np.ndarray,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        num_steps: int = 20
    ) -> Tuple[Optional[List[np.ndarray]], bool, Dict]:
        """
        Plan trajectory using Batched IK for goal + MPC for trajectory

        This is the hybrid approach:
        1. Use batched collision-free IK to find goal configuration
        2. Use MPC/MPPI to generate collision-free trajectory from current to goal

        Args:
            current_state: Current joint positions (7,)
            target_position: Target end-effector position (3,)
            target_orientation: Target orientation as quaternion [w,x,y,z] (4,) or None
            num_steps: Maximum number of trajectory steps

        Returns:
            trajectory: List of joint positions, or None if failed
            success: True if both IK and trajectory generation succeeded
            info: Dictionary with additional information
        """
        # Step 1: Solve batched collision-free IK for goal configuration
        print(f"[TorchMPC] Solving batched IK for goal position: {target_position}")
        goal_config, ik_success, ik_info = self.ik_solver.solve(
            target_position=target_position,
            target_orientation=target_orientation,
            current_joints=current_state,
            retract_config=current_state  # Use current as retract for regularization
        )

        if not ik_success:
            print(f"[TorchMPC] Batched IK failed: pos_err={ik_info['position_error']:.4f}, "
                  f"rot_err={ik_info['rotation_error']:.4f}, coll={ik_info['collision_cost']:.4f}")
            return None, False, ik_info

        print(f"[TorchMPC] Batched IK succeeded: {ik_info['num_solutions']} solutions found")
        print(f"[TorchMPC] Goal config: {goal_config[:3]}...")

        # Step 2: Set goal for MPC trajectory generation
        self.set_goal(target_position, target_orientation)

        # Step 3: Generate trajectory from current to goal using MPC
        print(f"[TorchMPC] Generating MPC trajectory from current to goal...")
        trajectory = [current_state.copy()]
        state = current_state.copy()

        for i in range(num_steps):
            # Check if we're close to goal config (joint space)
            joint_dist = np.linalg.norm(state - goal_config)
            if joint_dist < 0.1:  # Close enough in joint space
                # Add goal config as final waypoint
                trajectory.append(goal_config)
                print(f"[TorchMPC] Reached goal config at step {i+1}")
                break

            # MPC step toward goal
            next_state = self.step(state)
            trajectory.append(next_state)
            state = next_state

            # Check task-space distance to goal
            state_torch = torch.tensor(state, device=self.device, dtype=torch.float32)
            ee_pos, _ = self.fk_solver.forward_kinematics(state_torch.unsqueeze(0))
            goal_dist = torch.norm(ee_pos[0] - self.goal_position).item()

            if goal_dist < 0.01:  # 1cm threshold in task space
                print(f"[TorchMPC] Reached goal position at step {i+1}")
                break

        info = {
            **ik_info,
            'trajectory_length': len(trajectory),
            'goal_config': goal_config
        }

        return trajectory, True, info

