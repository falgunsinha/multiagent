"""
Batched Collision-Free IK Solver using PyTorch + Isaac Sim Kinematics

This module provides GPU-accelerated batched IK similar to cuRobo's IK solver,
but without requiring URDF/YAML files. It uses:
- Isaac Sim's ArticulationKinematicsSolver for single IK solutions
- PyTorch for batched optimization (MPPI + gradient descent)
- GPU-accelerated batch FK from torch_kinematics.py
- Collision checking with obstacles
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
import importlib.util
from pathlib import Path

# Load torch_kinematics directly
current_dir = Path(__file__).parent
spec = importlib.util.spec_from_file_location("torch_kinematics", current_dir / "torch_kinematics.py")
torch_kinematics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(torch_kinematics)
FrankaBatchFK = torch_kinematics.FrankaBatchFK


class BatchedIKSolver:
    """
    GPU-Accelerated Batched IK Solver
    
    Features:
    - Solves IK for multiple seeds in parallel on GPU
    - Collision-aware optimization
    - Two-stage optimization: MPPI (exploration) + Gradient Descent (refinement)
    - Returns best collision-free solution
    """
    
    def __init__(
        self,
        num_seeds: int = 100,
        position_threshold: float = 0.02,  # 2cm (relaxed for practical use)
        rotation_threshold: float = 0.1,   # ~6 degrees (relaxed)
        max_iterations: int = 50,
        device: str = "cuda:0"
    ):
        """
        Initialize Batched IK Solver
        
        Args:
            num_seeds: Number of random seeds to optimize in parallel
            position_threshold: Position error threshold in meters
            rotation_threshold: Rotation error threshold (geodesic distance)
            max_iterations: Maximum optimization iterations
            device: PyTorch device to use
        """
        self.num_seeds = num_seeds
        self.position_threshold = position_threshold
        self.rotation_threshold = rotation_threshold
        self.max_iterations = max_iterations
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Initialize batch FK
        self.fk = FrankaBatchFK(device=str(self.device))
        
        # Joint limits for Franka Panda (7 DOF)
        self.joint_lower = torch.tensor(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            device=self.device
        )
        self.joint_upper = torch.tensor(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            device=self.device
        )
        
        # Obstacles (will be updated before each solve)
        self.obstacles = []  # List of (position, radius) tuples

        # Check if CUDA is actually available and working
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[BatchedIK] Initialized with {num_seeds} seeds on {self.device}")
            print(f"[BatchedIK] GPU: {gpu_name}, Memory: {gpu_memory:.1f} GB")
        else:
            print(f"[BatchedIK] WARNING: CUDA not available, using CPU (will be slow!)")
    
    def set_obstacles(self, obstacles: List[Dict]):
        """
        Update obstacle list for collision checking
        
        Args:
            obstacles: List of dicts with 'position' and 'radius' keys
        """
        self.obstacles = []
        for obs in obstacles:
            if isinstance(obs, dict):
                pos = torch.tensor(obs['position'], device=self.device, dtype=torch.float32)
                radius = obs.get('radius', 0.1)
            else:
                pos = torch.tensor(obs, device=self.device, dtype=torch.float32)
                radius = 0.1
            self.obstacles.append((pos, radius))
    
    def generate_seeds(self, current_joints: np.ndarray, retract_config: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Generate random seed configurations within joint limits
        
        Args:
            current_joints: Current joint configuration (7,)
            retract_config: Optional retract configuration for regularization (7,)
            
        Returns:
            Seed configurations (num_seeds, 7)
        """
        # Generate random seeds within joint limits
        seeds = torch.rand(self.num_seeds, 7, device=self.device)
        seeds = self.joint_lower + seeds * (self.joint_upper - self.joint_lower)
        
        # Replace first seed with current configuration
        seeds[0] = torch.tensor(current_joints, device=self.device, dtype=torch.float32)
        
        # If retract config provided, use it as second seed
        if retract_config is not None:
            seeds[1] = torch.tensor(retract_config, device=self.device, dtype=torch.float32)
        
        return seeds
    
    def compute_pose_error(
        self,
        positions: torch.Tensor,
        orientations: torch.Tensor,
        target_position: torch.Tensor,
        target_orientation: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute pose error for batched configurations
        
        Args:
            positions: End-effector positions (batch, 3)
            orientations: End-effector orientations as quaternions (batch, 4)
            target_position: Target position (3,)
            target_orientation: Target orientation as quaternion (4,) or None
            
        Returns:
            position_error: L2 distance (batch,)
            rotation_error: Geodesic distance (batch,) or zeros if no target orientation
        """
        # Position error (L2 norm)
        position_error = torch.norm(positions - target_position, dim=1)
        
        # Rotation error (geodesic distance)
        if target_orientation is not None:
            # Quaternion dot product
            dot_product = torch.abs(torch.sum(orientations * target_orientation, dim=1))
            # Clamp to avoid numerical issues
            dot_product = torch.clamp(dot_product, 0.0, 1.0)
            rotation_error = 1.0 - dot_product
        else:
            rotation_error = torch.zeros(positions.shape[0], device=self.device)
        
        return position_error, rotation_error

    def compute_collision_cost(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute collision cost for batched end-effector positions

        Args:
            positions: End-effector positions (batch, 3)

        Returns:
            collision_cost: Collision cost for each configuration (batch,)
        """
        if len(self.obstacles) == 0:
            return torch.zeros(positions.shape[0], device=self.device)

        collision_cost = torch.zeros(positions.shape[0], device=self.device)

        for obs_pos, obs_radius in self.obstacles:
            # Distance from end-effector to obstacle center
            dist = torch.norm(positions - obs_pos, dim=1)

            # Collision margin (activate cost within 10cm of obstacle)
            margin = 0.1
            safe_distance = obs_radius + margin

            # Smooth collision cost (exponential barrier)
            penetration = safe_distance - dist
            cost = torch.where(
                penetration > 0,
                torch.exp(10.0 * penetration) - 1.0,
                torch.zeros_like(dist)
            )
            collision_cost += cost

        return collision_cost

    def compute_total_cost(
        self,
        joint_configs: torch.Tensor,
        target_position: torch.Tensor,
        target_orientation: Optional[torch.Tensor] = None,
        retract_config: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute total cost for batched joint configurations

        Args:
            joint_configs: Joint configurations (batch, 7)
            target_position: Target end-effector position (3,)
            target_orientation: Target orientation as quaternion (4,) or None
            retract_config: Retract configuration for regularization (7,) or None

        Returns:
            total_cost: Total cost for each configuration (batch,)
        """
        # Forward kinematics
        positions, orientations = self.fk.forward_kinematics(joint_configs)

        # Pose error
        pos_error, rot_error = self.compute_pose_error(
            positions, orientations, target_position, target_orientation
        )

        # Collision cost
        collision_cost = self.compute_collision_cost(positions)

        # Regularization cost (distance from retract config)
        if retract_config is not None:
            reg_cost = torch.norm(joint_configs - retract_config, dim=1)
        else:
            reg_cost = torch.zeros(joint_configs.shape[0], device=self.device)

        # Weighted sum
        total_cost = (
            100.0 * pos_error +           # Position error (high weight)
            50.0 * rot_error +             # Rotation error (medium weight)
            10.0 * collision_cost +        # Collision avoidance
            0.1 * reg_cost                 # Regularization (low weight)
        )

        return total_cost

    def optimize_mppi(
        self,
        seeds: torch.Tensor,
        target_position: torch.Tensor,
        target_orientation: Optional[torch.Tensor] = None,
        retract_config: Optional[torch.Tensor] = None,
        num_iterations: int = 20
    ) -> torch.Tensor:
        """
        Optimize using MPPI (Model Predictive Path Integral)

        Args:
            seeds: Initial seed configurations (num_seeds, 7)
            target_position: Target position (3,)
            target_orientation: Target orientation (4,) or None
            retract_config: Retract configuration (7,) or None
            num_iterations: Number of MPPI iterations

        Returns:
            Optimized configurations (num_seeds, 7)
        """
        configs = seeds.clone()
        noise_std = 0.3  # Standard deviation for exploration noise

        for i in range(num_iterations):
            # Generate noisy samples
            noise = torch.randn_like(configs) * noise_std
            noisy_configs = configs + noise

            # Clamp to joint limits
            noisy_configs = torch.clamp(noisy_configs, self.joint_lower, self.joint_upper)

            # Compute costs
            costs = self.compute_total_cost(
                noisy_configs, target_position, target_orientation, retract_config
            )

            # Softmax weighting (lower cost = higher weight)
            weights = torch.softmax(-costs / 0.1, dim=0)

            # Weighted average update
            configs = torch.sum(weights.unsqueeze(1) * noisy_configs, dim=0, keepdim=True)
            configs = configs.expand(self.num_seeds, -1)

            # Decay noise over iterations
            noise_std *= 0.95

        return configs

    def optimize_gradient(
        self,
        seeds: torch.Tensor,
        target_position: torch.Tensor,
        target_orientation: Optional[torch.Tensor] = None,
        retract_config: Optional[torch.Tensor] = None,
        num_iterations: int = 50,
        learning_rate: float = 0.05
    ) -> torch.Tensor:
        """
        Optimize using gradient descent with automatic differentiation

        Args:
            seeds: Initial configurations (num_seeds, 7)
            target_position: Target position (3,)
            target_orientation: Target orientation (4,) or None
            retract_config: Retract configuration (7,) or None
            num_iterations: Number of gradient descent iterations
            learning_rate: Learning rate for gradient descent

        Returns:
            Optimized configurations (num_seeds, 7)
        """
        configs = seeds.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([configs], lr=learning_rate)

        for i in range(num_iterations):
            optimizer.zero_grad()

            # Clamp to joint limits (detach to avoid breaking gradient)
            configs_clamped = torch.clamp(configs, self.joint_lower, self.joint_upper)

            # Compute cost
            cost = self.compute_total_cost(
                configs_clamped, target_position, target_orientation, retract_config
            ).mean()

            # Backward pass
            cost.backward()

            # Gradient descent step
            optimizer.step()

            # Clamp after update
            with torch.no_grad():
                configs.clamp_(self.joint_lower, self.joint_upper)

        return configs.detach()

    def solve(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        current_joints: Optional[np.ndarray] = None,
        retract_config: Optional[np.ndarray] = None,
        return_all_solutions: bool = False
    ) -> Tuple[Optional[np.ndarray], bool, Dict]:
        """
        Solve batched collision-free IK

        Args:
            target_position: Target end-effector position (3,)
            target_orientation: Target orientation as quaternion [w,x,y,z] (4,) or None
            current_joints: Current joint configuration (7,) for seeding
            retract_config: Retract configuration (7,) for regularization
            return_all_solutions: If True, return all feasible solutions

        Returns:
            solution: Best joint configuration (7,) or None if failed
            success: True if solution found
            info: Dictionary with additional information
        """
        # Convert inputs to tensors
        target_pos = torch.tensor(target_position, device=self.device, dtype=torch.float32)
        target_ori = None
        if target_orientation is not None:
            target_ori = torch.tensor(target_orientation, device=self.device, dtype=torch.float32)

        retract_cfg = None
        if retract_config is not None:
            retract_cfg = torch.tensor(retract_config, device=self.device, dtype=torch.float32)

        # Generate seeds
        if current_joints is None:
            current_joints = np.zeros(7)
        seeds = self.generate_seeds(current_joints, retract_config)

        import time
        t0 = time.time()

        # Stage 1: MPPI optimization (exploration)
        configs = self.optimize_mppi(
            seeds, target_pos, target_ori, retract_cfg, num_iterations=15
        )
        t1 = time.time()

        # Stage 2: Gradient descent (refinement)
        configs = self.optimize_gradient(
            configs, target_pos, target_ori, retract_cfg, num_iterations=25
        )
        t2 = time.time()

        print(f"[BatchedIK] Timing: MPPI={t1-t0:.3f}s, Gradient={t2-t1:.3f}s, Total={t2-t0:.3f}s")

        # Evaluate final solutions
        positions, orientations = self.fk.forward_kinematics(configs)
        pos_error, rot_error = self.compute_pose_error(positions, orientations, target_pos, target_ori)
        collision_cost = self.compute_collision_cost(positions)

        # Check success criteria
        success_mask = (
            (pos_error < self.position_threshold) &
            (rot_error < self.rotation_threshold) &
            (collision_cost < 0.1)  # Low collision cost
        )

        # Find best solution
        if success_mask.any():
            # Among successful solutions, pick one with lowest cost
            total_cost = pos_error + rot_error + collision_cost
            total_cost[~success_mask] = float('inf')
            best_idx = torch.argmin(total_cost)

            solution = configs[best_idx].cpu().numpy()
            success = True

            info = {
                'position_error': pos_error[best_idx].item(),
                'rotation_error': rot_error[best_idx].item(),
                'collision_cost': collision_cost[best_idx].item(),
                'num_solutions': success_mask.sum().item()
            }

            if return_all_solutions:
                info['all_solutions'] = configs[success_mask].cpu().numpy()
        else:
            # No successful solution found, return best effort
            total_cost = pos_error + rot_error + collision_cost
            best_idx = torch.argmin(total_cost)

            solution = configs[best_idx].cpu().numpy()
            success = False

            info = {
                'position_error': pos_error[best_idx].item(),
                'rotation_error': rot_error[best_idx].item(),
                'collision_cost': collision_cost[best_idx].item(),
                'num_solutions': 0
            }

        return solution, success, info

