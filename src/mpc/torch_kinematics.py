"""
GPU-Accelerated Batch IK/FK Solver for Franka Panda Robot
Uses PyTorch + CUDA for fast batch operations (like cuRobo but without cuRobo dependency)

This module provides:
1. Batch Forward Kinematics (GPU-accelerated with PyTorch)
2. Essential for MPC/MPPI that needs thousands of FK evaluations

Key Features:
- GPU acceleration using PyTorch
- Batch FK: Compute FK for 1000s of configurations in parallel
- No cuRobo installation required
"""

import numpy as np
import torch
from typing import Tuple


class FrankaBatchFK:
    """
    GPU-Accelerated Batch Forward Kinematics for Franka Panda
    
    Computes FK for thousands of joint configurations in parallel on GPU.
    Essential for MPC/MPPI which needs to evaluate many trajectories.
    """
    
    def __init__(self, device: str = "cuda:0"):
        """
        Initialize FK solver on GPU
        
        Args:
            device: PyTorch device ("cuda:0" or "cpu")
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[BatchFK] Using device: {self.device}")
        
        # Modified DH parameters for Franka Panda (7-DOF)
        # [a, alpha, d, theta_offset]
        dh_params_np = np.array([
            [0.0,      0.0,        0.333,  0.0],      # Joint 1
            [0.0,     -np.pi/2,    0.0,    0.0],      # Joint 2
            [0.0,      np.pi/2,    0.316,  0.0],      # Joint 3
            [0.0825,   np.pi/2,    0.0,    0.0],      # Joint 4
            [-0.0825, -np.pi/2,    0.384,  0.0],      # Joint 5
            [0.0,      np.pi/2,    0.0,    0.0],      # Joint 6
            [0.088,    np.pi/2,    0.0,    0.0],      # Joint 7
        ], dtype=np.float32)
        
        self.dh_params = torch.tensor(dh_params_np, device=self.device)
        self.ee_offset = 0.107  # End-effector offset (meters)
        
        # Joint limits
        self.joint_limits_lower = torch.tensor(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            device=self.device, dtype=torch.float32
        )
        self.joint_limits_upper = torch.tensor(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            device=self.device, dtype=torch.float32
        )
    
    def _dh_transform(self, a: torch.Tensor, alpha: torch.Tensor, 
                      d: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute Modified DH transformation matrix (batched on GPU)
        
        Args:
            a: Link length (batch_size,)
            alpha: Link twist (batch_size,)
            d: Link offset (batch_size,)
            theta: Joint angle (batch_size,)
            
        Returns:
            Transformation matrices (batch_size, 4, 4)
        """
        batch_size = theta.shape[0]
        
        # Precompute trig functions
        ct = torch.cos(theta)
        st = torch.sin(theta)
        ca = torch.cos(alpha)
        sa = torch.sin(alpha)
        
        # Initialize transformation matrices
        T = torch.zeros((batch_size, 4, 4), device=self.device, dtype=torch.float32)
        
        # Fill transformation matrix (Modified DH convention)
        T[:, 0, 0] = ct
        T[:, 0, 1] = -st
        T[:, 0, 3] = a
        
        T[:, 1, 0] = st * ca
        T[:, 1, 1] = ct * ca
        T[:, 1, 2] = -sa
        T[:, 1, 3] = -d * sa
        
        T[:, 2, 0] = st * sa
        T[:, 2, 1] = ct * sa
        T[:, 2, 2] = ca
        T[:, 2, 3] = d * ca
        
        T[:, 3, 3] = 1.0
        
        return T
    
    def forward_kinematics(self, joint_positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute forward kinematics for batch of joint configurations (GPU-accelerated)
        
        Args:
            joint_positions: Joint angles (batch_size, 7) on GPU
            
        Returns:
            Tuple of:
                - positions: End-effector positions (batch_size, 3)
                - orientations: End-effector orientations as rotation matrices (batch_size, 3, 3)
        """
        batch_size = joint_positions.shape[0]
        
        # Initialize base transformation (identity)
        T = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Forward kinematics: multiply transformations for each joint
        for i in range(7):
            a = self.dh_params[i, 0].expand(batch_size)
            alpha = self.dh_params[i, 1].expand(batch_size)
            d = self.dh_params[i, 2].expand(batch_size)
            theta_offset = self.dh_params[i, 3].expand(batch_size)
            theta = joint_positions[:, i] + theta_offset
            
            # Compute DH transform for this joint
            T_i = self._dh_transform(a, alpha, d, theta)
            
            # Multiply: T = T @ T_i (batch matrix multiplication)
            T = torch.bmm(T, T_i)
        
        # Add end-effector offset
        T_ee = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
        T_ee[:, 2, 3] = self.ee_offset
        T = torch.bmm(T, T_ee)
        
        # Extract positions and orientations
        positions = T[:, :3, 3]
        rotation_matrices = T[:, :3, :3]

        # Convert rotation matrices to quaternions [w, x, y, z]
        quaternions = self._rotation_matrix_to_quaternion(rotation_matrices)

        return positions, quaternions

    def _rotation_matrix_to_quaternion(self, R: torch.Tensor) -> torch.Tensor:
        """
        Convert batch of rotation matrices to quaternions

        Args:
            R: Rotation matrices (batch, 3, 3)

        Returns:
            Quaternions [w, x, y, z] (batch, 4)
        """
        batch_size = R.shape[0]
        q = torch.zeros(batch_size, 4, device=self.device, dtype=torch.float32)

        # Compute quaternion components
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

        # Case 1: trace > 0
        mask1 = trace > 0
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * w
        q[mask1, 0] = 0.25 * s
        q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
        q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
        q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s

        # Case 2: R[0,0] is largest diagonal
        mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
        s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
        q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
        q[mask2, 1] = 0.25 * s
        q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
        q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s

        # Case 3: R[1,1] is largest diagonal
        mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
        s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
        q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
        q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
        q[mask3, 2] = 0.25 * s
        q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s

        # Case 4: R[2,2] is largest diagonal
        mask4 = (~mask1) & (~mask2) & (~mask3)
        s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
        q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
        q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
        q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
        q[mask4, 3] = 0.25 * s

        # Normalize quaternions
        q = q / torch.norm(q, dim=1, keepdim=True)

        return q

