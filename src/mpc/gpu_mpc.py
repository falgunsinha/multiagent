"""
GPU-Accelerated MPC - Single file wrapper to avoid import issues
This file re-exports all the GPU MPC classes
"""

# Import all torch-based MPC components
import sys
import os
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import the actual implementations
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable

# Load modules directly from files (no relative imports to avoid issues)
import importlib.util

current_dir = Path(__file__).parent

print(f"[GPU_MPC] Loading MPC modules from: {current_dir}")

# Load torch_kinematics
spec = importlib.util.spec_from_file_location(
    "torch_kinematics",
    current_dir / "torch_kinematics.py"
)
torch_kinematics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(torch_kinematics)
FrankaBatchFK = torch_kinematics.FrankaBatchFK

# Load torch_mppi
spec = importlib.util.spec_from_file_location(
    "torch_mppi",
    current_dir / "torch_mppi.py"
)
torch_mppi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(torch_mppi)
TorchMPPI = torch_mppi.TorchMPPI

# Load torch_mpc_planner
spec = importlib.util.spec_from_file_location(
    "torch_mpc_planner",
    current_dir / "torch_mpc_planner.py"
)
torch_mpc_planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(torch_mpc_planner)
TorchMPCPlanner = torch_mpc_planner.TorchMPCPlanner

# Load batched_ik
spec = importlib.util.spec_from_file_location(
    "batched_ik",
    current_dir / "batched_ik.py"
)
batched_ik = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batched_ik)
BatchedIKSolver = batched_ik.BatchedIKSolver

__all__ = ['FrankaBatchFK', 'TorchMPPI', 'TorchMPCPlanner', 'BatchedIKSolver']

print("✅ Successfully loaded torch MPC modules (including Batched IK) using direct file loading")

