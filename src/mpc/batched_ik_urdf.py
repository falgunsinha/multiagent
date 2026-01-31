"""
URDF-Based Batched IK Solver (cuRobo-style implementation from scratch)

This is a from-scratch implementation of cuRobo's batched IK solver:
1. Uses URDF for accurate FK (not DH parameters)
2. Two-stage optimization: MPPI + L-BFGS
3. GPU-accelerated batched operations
4. Collision-aware optimization

Based on cuRobo's IKSolver but without ROS dependencies.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, List
import time
import os
import sys
from pathlib import Path

# Handle both relative and absolute imports
try:
    from .urdf_kinematics import URDFKinematics
    from .robot_config_loader import RobotConfigLoader
except ImportError:
    # If relative import fails, try absolute import
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from urdf_kinematics import URDFKinematics
    from robot_config_loader import RobotConfigLoader


class BatchedIKSolverURDF:
    """
    GPU-Accelerated Batched IK Solver using URDF-based FK
    
    Implements cuRobo-style two-stage optimization:
    - Stage 1: MPPI (Model Predictive Path Integral) for exploration
    - Stage 2: L-BFGS for gradient-based refinement
    """
    
    def __init__(
        self,
        urdf_path: Optional[str] = None,
        yml_path: Optional[str] = None,
        num_seeds: int = 100,
        position_threshold: float = 0.005,  # 5mm (cuRobo default)
        rotation_threshold: float = 0.01,   # Adjusted for linear metric (1 - dot_product)
        mppi_iterations: int = 20,
        lbfgs_iterations: int = 100,
        device: str = "cuda:0"
    ):
        """
        Initialize URDF-based Batched IK Solver

        Args:
            urdf_path: Path to Franka URDF file (if yml_path not provided)
            yml_path: Path to robot YML config (cuRobo-style, contains URDF path + collision spheres)
            num_seeds: Number of parallel seeds (cuRobo default: 100)
            position_threshold: Position convergence threshold in meters
            rotation_threshold: Rotation convergence threshold (geodesic distance)
            mppi_iterations: MPPI iterations for exploration
            lbfgs_iterations: L-BFGS iterations for refinement
            device: PyTorch device
        """
        self.num_seeds = num_seeds
        self.position_threshold = position_threshold
        self.rotation_threshold = rotation_threshold
        self.mppi_iterations = mppi_iterations
        self.lbfgs_iterations = lbfgs_iterations
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load robot configuration (cuRobo-style)
        self.robot_config = None
        if yml_path is not None:
            # Load from YML (like cuRobo does)
            self.robot_config = RobotConfigLoader(yml_path, device=str(self.device))
            urdf_path = self.robot_config.urdf_path
            print(f"[BatchedIK-URDF] Loaded robot config from YML: {yml_path}")
            print(f"[BatchedIK-URDF] Collision spheres: {sum(len(s) for s in self.robot_config.collision_spheres.values())} total")
        elif urdf_path is None:
            raise ValueError("Either urdf_path or yml_path must be provided")

        # Initialize URDF-based FK
        self.fk = URDFKinematics(urdf_path, device=str(self.device))
        
        # Franka Panda joint limits
        self.joint_lower = torch.tensor(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            device=self.device, dtype=torch.float32
        )
        self.joint_upper = torch.tensor(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            device=self.device, dtype=torch.float32
        )
        
        # Default joint configuration (used for null-space optimization)
        self.default_joints = torch.tensor(
            [0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75],
            device=self.device, dtype=torch.float32
        )
        
        # Obstacles
        self.obstacles = []
        
        # MPPI parameters (from cuRobo)
        self.mppi_noise_std = 0.5  # Exploration noise
        self.mppi_temperature = 0.1  # Temperature for weighted averaging
        
        # L-BFGS parameters
        self.lbfgs_lr = 0.5  # Learning rate
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[BatchedIK-URDF] Initialized with {num_seeds} seeds on {self.device}")
            print(f"[BatchedIK-URDF] GPU: {gpu_name}, Memory: {gpu_memory:.1f} GB")
            print(f"[BatchedIK-URDF] Using URDF-based FK from: {urdf_path}")
        else:
            print(f"[BatchedIK-URDF] WARNING: CUDA not available, using CPU")

    def update_obstacles(self, obstacles: list):
        """
        Update obstacle list for collision checking

        Args:
            obstacles: List of obstacle dictionaries with 'position' and 'radius' keys
        """
        self.obstacles = obstacles
        if obstacles:
            print(f"[BatchedIK-URDF] Updated {len(obstacles)} obstacles for collision checking")
    
    def set_obstacles(self, obstacles: List[Dict]):
        """Update obstacle list for collision checking"""
        self.obstacles = []
        for obs in obstacles:
            if isinstance(obs, dict):
                pos = torch.tensor(obs['position'], device=self.device, dtype=torch.float32)
                radius = obs.get('radius', 0.1)
            else:
                pos = torch.tensor(obs, device=self.device, dtype=torch.float32)
                radius = 0.1
            self.obstacles.append((pos, radius))
    
    def _generate_seeds(self, current_joints: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Generate initial seed configurations
        
        Uses Halton sequence for better coverage (like cuRobo)
        Falls back to random sampling
        """
        seeds = torch.zeros((self.num_seeds, 7), device=self.device, dtype=torch.float32)
        
        # Seed 0: Current configuration (if provided)
        if current_joints is not None:
            seeds[0] = torch.tensor(current_joints, device=self.device, dtype=torch.float32)
            start_idx = 1
        else:
            start_idx = 0
        
        # Seed 1: Default configuration
        if start_idx < self.num_seeds:
            seeds[start_idx] = self.default_joints
            start_idx += 1
        
        # Remaining seeds: Random sampling within joint limits
        num_random = self.num_seeds - start_idx
        if num_random > 0:
            random_seeds = torch.rand((num_random, 7), device=self.device, dtype=torch.float32)
            seeds[start_idx:] = self.joint_lower + random_seeds * (self.joint_upper - self.joint_lower)
        
        return seeds

    def _compute_pose_error(
        self,
        current_pos: torch.Tensor,
        current_quat: torch.Tensor,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute position and rotation errors

        Args:
            current_pos: Current positions (batch_size, 3)
            current_quat: Current quaternions [w,x,y,z] (batch_size, 4)
            target_pos: Target position (3,)
            target_quat: Target quaternion [w,x,y,z] (4,)

        Returns:
            Tuple of (position_error, rotation_error)
        """
        # Position error (Euclidean distance)
        pos_error = torch.norm(current_pos - target_pos, dim=1)

        # Rotation error using quaternion distance
        # More gradient-friendly than geodesic distance
        # Formula: error = 1 - |<q1, q2>|
        # This is proportional to geodesic distance for small angles
        dot_product = torch.sum(current_quat * target_quat, dim=1)
        # Take absolute value to handle quaternion double cover (q and -q represent same rotation)
        dot_product = torch.abs(dot_product)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Numerical stability

        # Linear approximation of rotation error (better for gradients)
        rot_error = 1.0 - dot_product

        return pos_error, rot_error

    def _compute_cost(
        self,
        joint_positions: torch.Tensor,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute total cost for each configuration

        Cost = position_error + rotation_error + collision_cost + null_space_cost
        """
        batch_size = joint_positions.shape[0]

        # Forward kinematics
        current_pos, current_quat = self.fk.forward_kinematics(joint_positions)

        # Pose error
        pos_error, rot_error = self._compute_pose_error(
            current_pos, current_quat, target_pos, target_quat
        )

        # Weighted pose cost (increase rotation weight for better convergence)
        pose_cost = pos_error * 100.0 + rot_error * 200.0  # Weight rotation more

        # Null-space cost (prefer staying close to default configuration)
        null_space_cost = torch.norm(joint_positions - self.default_joints, dim=1) * 0.1

        # Collision cost (simple sphere-based)
        collision_cost = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        if len(self.obstacles) > 0:
            # Simplified: Check end-effector collision only
            for obs in self.obstacles:
                # Handle both dictionary and tuple formats
                if isinstance(obs, dict):
                    obs_pos = obs['position']
                    obs_radius = obs.get('radius', 0.15)  # Default 15cm radius
                else:
                    obs_pos, obs_radius = obs

                # Convert to tensor if needed
                if not isinstance(obs_pos, torch.Tensor):
                    obs_pos = torch.tensor(obs_pos, device=self.device, dtype=torch.float32)

                dist = torch.norm(current_pos - obs_pos, dim=1)
                # Soft collision cost (activates within 10cm of obstacle)
                activation_dist = obs_radius + 0.10
                collision_cost += torch.clamp(activation_dist - dist, min=0.0) * 100.0

        # Total cost
        total_cost = pose_cost + null_space_cost + collision_cost

        return total_cost

    def _mppi_step(
        self,
        joint_positions: torch.Tensor,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        iteration: int,
        total_iterations: int
    ) -> torch.Tensor:
        """
        Single MPPI optimization step with adaptive noise

        MPPI (Model Predictive Path Integral):
        1. Add noise to current solutions
        2. Evaluate costs
        3. Compute weighted average based on costs
        """
        batch_size = joint_positions.shape[0]

        # Adaptive noise: decrease over iterations
        noise_scale = self.mppi_noise_std * (1.0 - iteration / total_iterations)
        noise_scale = max(noise_scale, 0.1)  # Minimum noise

        # Generate noisy samples
        noise = torch.randn_like(joint_positions) * noise_scale
        noisy_positions = joint_positions + noise

        # Clamp to joint limits
        noisy_positions = torch.clamp(noisy_positions, self.joint_lower, self.joint_upper)

        # Compute costs for both original and noisy positions
        costs_original = self._compute_cost(joint_positions, target_pos, target_quat)
        costs_noisy = self._compute_cost(noisy_positions, target_pos, target_quat)

        # Keep better solutions (elitism)
        better_mask = costs_noisy < costs_original
        joint_positions[better_mask] = noisy_positions[better_mask]
        costs = torch.where(better_mask, costs_noisy, costs_original)

        # Compute weights using softmax with temperature
        weights = torch.softmax(-costs / self.mppi_temperature, dim=0)

        # Weighted average of top solutions
        top_k = max(batch_size // 4, 1)  # Top 25%
        top_indices = torch.topk(-costs, k=top_k)[1]

        best_positions = joint_positions[top_indices]
        best_weights = weights[top_indices]
        best_weights = best_weights / best_weights.sum()

        # Weighted average
        updated_positions = torch.sum(best_weights.unsqueeze(1) * best_positions, dim=0, keepdim=True)

        # Update all seeds towards the best weighted average
        alpha = 0.5  # Mixing factor
        joint_positions = alpha * updated_positions + (1 - alpha) * joint_positions

        return joint_positions

    def _lbfgs_step(
        self,
        joint_positions: torch.Tensor,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor
    ) -> torch.Tensor:
        """
        Gradient descent step with adaptive learning rate

        Uses PyTorch autograd for gradient computation
        Optimizes each seed independently for better convergence
        """
        batch_size = joint_positions.shape[0]

        # Process each solution independently for better convergence
        for i in range(batch_size):
            # Enable gradient computation for this seed
            seed = joint_positions[i:i+1].detach().requires_grad_(True)

            # Compute cost for this seed
            current_pos, current_quat = self.fk.forward_kinematics(seed)
            pos_error, rot_error = self._compute_pose_error(
                current_pos, current_quat, target_pos, target_quat
            )

            # Weighted cost
            cost = pos_error[0] * 100.0 + rot_error[0] * 10.0

            # Add null-space cost
            null_space_cost = torch.norm(seed - self.default_joints) * 0.1
            cost = cost + null_space_cost

            # Compute gradients
            cost.backward()

            # Gradient descent step with adaptive learning rate
            with torch.no_grad():
                # Use smaller learning rate for better convergence
                lr = 0.1 if pos_error[0] > 0.1 else 0.05
                joint_positions[i] = seed[0] - lr * seed.grad[0]

                # Clamp to joint limits
                joint_positions[i] = torch.clamp(
                    joint_positions[i],
                    self.joint_lower,
                    self.joint_upper
                )

        return joint_positions

    def solve(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        current_joints: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[Optional[np.ndarray], bool, Dict]:
        """
        Solve IK using two-stage optimization (MPPI + L-BFGS)

        Args:
            target_position: Target end-effector position (3,)
            target_orientation: Target quaternion [w,x,y,z] (4,)
            current_joints: Current joint configuration (7,)

        Returns:
            Tuple of (solution, success, info_dict)
        """
        start_time = time.time()

        # Convert to torch tensors
        target_pos = torch.tensor(target_position, device=self.device, dtype=torch.float32)
        target_quat = torch.tensor(target_orientation, device=self.device, dtype=torch.float32)

        # Normalize target quaternion
        target_quat = target_quat / torch.norm(target_quat)

        # Generate initial seeds
        joint_positions = self._generate_seeds(current_joints)

        # Stage 1: MPPI exploration
        for i in range(self.mppi_iterations):
            joint_positions = self._mppi_step(
                joint_positions, target_pos, target_quat, i, self.mppi_iterations
            )

        # Stage 2: Gradient refinement using Adam optimizer (faster than L-BFGS)
        # Evaluate current costs to find best seeds
        with torch.no_grad():
            current_pos, current_quat = self.fk.forward_kinematics(joint_positions)
            pos_errors, rot_errors = self._compute_pose_error(
                current_pos, current_quat, target_pos, target_quat
            )
            costs = pos_errors * 100.0 + rot_errors * 10.0

        # CRITICAL FIX: Batched refinement for GPU parallelization!
        # Previous approach: Sequential refinement of 5 seeds × 30 iterations = 150 sequential steps
        # New approach: Parallel refinement of all top seeds simultaneously on GPU
        top_k = min(5, self.num_seeds)
        top_indices = torch.topk(-costs, k=top_k)[1]

        # Extract top seeds for batched optimization
        top_seeds = joint_positions[top_indices].clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([top_seeds], lr=0.01)

        # Batched optimization: all seeds optimized in parallel on GPU
        max_refinement_iters = min(self.lbfgs_iterations, 20)  # Reduced from 30 for speed
        for i in range(max_refinement_iters):
            optimizer.zero_grad()

            # Compute costs for all top seeds in parallel (GPU accelerated!)
            current_pos, current_quat = self.fk.forward_kinematics(top_seeds)
            pos_errors, rot_errors = self._compute_pose_error(
                current_pos, current_quat, target_pos, target_quat
            )

            # Weighted costs for all seeds (batched)
            costs_batch = pos_errors * 100.0 + rot_errors * 200.0

            # Add null-space cost (batched)
            null_space_costs = torch.norm(top_seeds - self.default_joints, dim=1) * 0.1
            costs_batch = costs_batch + null_space_costs

            # Total cost (sum for backprop)
            total_cost = costs_batch.sum()

            # Backprop and optimize (all seeds updated in parallel!)
            total_cost.backward()
            optimizer.step()

            # Clamp to joint limits (batched)
            with torch.no_grad():
                top_seeds.data = torch.clamp(top_seeds.data, self.joint_lower, self.joint_upper)

        # Update the top solutions in joint_positions
        with torch.no_grad():
            joint_positions[top_indices] = top_seeds.detach()

        # Evaluate final solutions
        with torch.no_grad():
            final_pos, final_quat = self.fk.forward_kinematics(joint_positions)
            pos_errors, rot_errors = self._compute_pose_error(
                final_pos, final_quat, target_pos, target_quat
            )

        # Find best solution that meets thresholds
        success_mask = (pos_errors < self.position_threshold) & (rot_errors < self.rotation_threshold)

        if success_mask.any():
            # Get best successful solution
            successful_indices = torch.where(success_mask)[0]
            best_idx = successful_indices[torch.argmin(pos_errors[successful_indices])]

            solution = joint_positions[best_idx].cpu().numpy()
            pos_error = pos_errors[best_idx].item()
            rot_error = rot_errors[best_idx].item()
            success = True
        else:
            # No solution met thresholds, return best effort
            best_idx = torch.argmin(pos_errors)
            solution = joint_positions[best_idx].cpu().numpy()
            pos_error = pos_errors[best_idx].item()
            rot_error = rot_errors[best_idx].item()
            success = False

        solve_time = time.time() - start_time

        # Build info dictionary
        info = {
            'position_error': pos_error,
            'rotation_error': rot_error,
            'solve_time': solve_time,
            'num_successful': success_mask.sum().item(),
            'best_index': best_idx.item()
        }

        return solution, success, info


