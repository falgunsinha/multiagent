"""
URDF-Based Forward Kinematics for Franka Panda
Uses URDF file for accurate kinematics (like cuRobo)

This replaces DH-parameter based FK which can have errors.
"""

import numpy as np
import torch
from typing import Tuple, Dict, List
import xml.etree.ElementTree as ET


class URDFKinematics:
    """
    URDF-based Forward Kinematics using PyTorch for GPU acceleration
    
    Parses URDF file and computes FK using transformation matrices.
    Much more accurate than DH parameters.
    """
    
    def __init__(self, urdf_path: str, device: str = "cuda:0"):
        """
        Initialize URDF kinematics
        
        Args:
            urdf_path: Path to URDF file
            device: PyTorch device
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.urdf_path = urdf_path
        
        # Parse URDF
        self.links, self.joints = self._parse_urdf(urdf_path)
        
        # Build kinematic chain
        self.joint_chain = self._build_joint_chain()
        
        print(f"[URDF-FK] Loaded {len(self.joints)} joints from {urdf_path}")
        print(f"[URDF-FK] Joint chain: {[j['name'] for j in self.joint_chain]}")
    
    def _parse_urdf(self, urdf_path: str) -> Tuple[Dict, Dict]:
        """Parse URDF file and extract links and joints"""
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        links = {}
        joints = {}
        
        # Parse links
        for link in root.findall('link'):
            link_name = link.get('name')
            links[link_name] = {'name': link_name}
        
        # Parse joints
        for joint in root.findall('joint'):
            joint_name = joint.get('name')
            joint_type = joint.get('type')
            
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            
            # Parse origin (xyz, rpy)
            origin = joint.find('origin')
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]
            
            if origin is not None:
                if origin.get('xyz'):
                    xyz = [float(x) for x in origin.get('xyz').split()]
                if origin.get('rpy'):
                    rpy = [float(x) for x in origin.get('rpy').split()]
            
            # Parse axis
            axis = [0.0, 0.0, 1.0]  # Default Z axis
            axis_elem = joint.find('axis')
            if axis_elem is not None and axis_elem.get('xyz'):
                axis = [float(x) for x in axis_elem.get('xyz').split()]
            
            joints[joint_name] = {
                'name': joint_name,
                'type': joint_type,
                'parent': parent,
                'child': child,
                'xyz': xyz,
                'rpy': rpy,
                'axis': axis
            }
        
        return links, joints
    
    def _build_joint_chain(self) -> List[Dict]:
        """Build ordered joint chain from base to end-effector"""
        # For Franka: panda_joint1 -> panda_joint7
        joint_names = [f'panda_joint{i}' for i in range(1, 8)]

        chain = []
        for jname in joint_names:
            if jname in self.joints:
                chain.append(self.joints[jname])

        # Add end-effector transform (panda_link7 -> panda_hand)
        # This matches cuRobo's ee_link: "panda_hand"
        if 'panda_hand_joint' in self.joints:
            chain.append(self.joints['panda_hand_joint'])

        return chain
    
    def _rpy_to_rotation_matrix(self, roll: float, pitch: float, yaw: float) -> torch.Tensor:
        """Convert roll-pitch-yaw to rotation matrix"""
        # Rotation around X (roll)
        Rx = torch.tensor([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ], device=self.device, dtype=torch.float32)
        
        # Rotation around Y (pitch)
        Ry = torch.tensor([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ], device=self.device, dtype=torch.float32)
        
        # Rotation around Z (yaw)
        Rz = torch.tensor([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ], device=self.device, dtype=torch.float32)
        
        # Combined rotation: R = Rz * Ry * Rx
        return Rz @ Ry @ Rx

    def _axis_angle_to_rotation_matrix(self, axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """
        Convert axis-angle to rotation matrix (Rodrigues' formula)
        Batched version for GPU acceleration

        Args:
            axis: Rotation axis (batch_size, 3)
            angle: Rotation angle in radians (batch_size,)

        Returns:
            Rotation matrices (batch_size, 3, 3)
        """
        batch_size = angle.shape[0]

        # Normalize axis
        axis = axis / (torch.norm(axis, dim=1, keepdim=True) + 1e-8)

        # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
        # where K is the skew-symmetric matrix of axis

        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)

        # Create skew-symmetric matrix K
        K = torch.zeros((batch_size, 3, 3), device=self.device, dtype=torch.float32)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]

        # K²
        K2 = torch.bmm(K, K)

        # Identity matrix
        I = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0).expand(batch_size, 3, 3)

        # R = I + sin(θ)K + (1-cos(θ))K²
        R = I + sin_angle.view(-1, 1, 1) * K + (1 - cos_angle).view(-1, 1, 1) * K2

        return R

    def forward_kinematics(self, joint_positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute forward kinematics for batch of joint configurations

        Args:
            joint_positions: Joint angles (batch_size, 7)

        Returns:
            Tuple of:
                - positions: End-effector positions (batch_size, 3)
                - quaternions: End-effector orientations as quaternions [w,x,y,z] (batch_size, 4)
        """
        batch_size = joint_positions.shape[0]

        # Initialize transformation matrix (identity)
        T = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).expand(batch_size, 4, 4).clone()

        # Process each joint in the chain
        for i, joint in enumerate(self.joint_chain[:7]):  # Only first 7 joints (panda_joint1-7)
            # Get joint angle
            q = joint_positions[:, i]

            # Fixed transform from parent to joint (from URDF origin)
            xyz = torch.tensor(joint['xyz'], device=self.device, dtype=torch.float32)
            rpy = joint['rpy']

            # Create fixed transformation matrix
            R_fixed = self._rpy_to_rotation_matrix(rpy[0], rpy[1], rpy[2])
            T_fixed = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).expand(batch_size, 4, 4).clone()
            T_fixed[:, :3, :3] = R_fixed.unsqueeze(0).expand(batch_size, 3, 3)
            T_fixed[:, :3, 3] = xyz.unsqueeze(0).expand(batch_size, 3)

            # Create joint rotation (around axis)
            axis = torch.tensor(joint['axis'], device=self.device, dtype=torch.float32).unsqueeze(0).expand(batch_size, 3)
            R_joint = self._axis_angle_to_rotation_matrix(axis, q)

            T_joint = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).expand(batch_size, 4, 4).clone()
            T_joint[:, :3, :3] = R_joint

            # Combine: T = T * T_fixed * T_joint
            T = torch.bmm(T, T_fixed)
            T = torch.bmm(T, T_joint)

        # Add end-effector offset (panda_link7 -> panda_hand)
        if len(self.joint_chain) > 7:
            ee_joint = self.joint_chain[7]  # panda_hand_joint
            xyz_ee = torch.tensor(ee_joint['xyz'], device=self.device, dtype=torch.float32)
            rpy_ee = ee_joint['rpy']

            R_ee = self._rpy_to_rotation_matrix(rpy_ee[0], rpy_ee[1], rpy_ee[2])
            T_ee = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).expand(batch_size, 4, 4).clone()
            T_ee[:, :3, :3] = R_ee.unsqueeze(0).expand(batch_size, 3, 3)
            T_ee[:, :3, 3] = xyz_ee.unsqueeze(0).expand(batch_size, 3)

            T = torch.bmm(T, T_ee)

        # Extract position and orientation
        positions = T[:, :3, 3]
        rotation_matrices = T[:, :3, :3]

        # Convert rotation matrices to quaternions
        quaternions = self._rotation_matrix_to_quaternion(rotation_matrices)

        return positions, quaternions

    def _rotation_matrix_to_quaternion(self, R: torch.Tensor) -> torch.Tensor:
        """
        Convert rotation matrices to quaternions [w, x, y, z]

        Args:
            R: Rotation matrices (batch_size, 3, 3)

        Returns:
            Quaternions (batch_size, 4) in [w, x, y, z] format
        """
        batch_size = R.shape[0]
        q = torch.zeros((batch_size, 4), device=self.device, dtype=torch.float32)

        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

        # Case 1: trace > 0
        mask1 = trace > 0
        s = torch.sqrt(trace[mask1] + 1.0) * 2
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

        # Normalize
        q = q / (torch.norm(q, dim=1, keepdim=True) + 1e-8)

        return q


