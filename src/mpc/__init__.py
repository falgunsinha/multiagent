"""
MPC (Model Predictive Control) module for robot motion planning

This module provides GPU-accelerated MPC implementation using PyTorch.
"""

import sys
print(f"[MPC __init__] Loading MPC package from: {__file__}")
print(f"[MPC __init__] Package path: {__path__}")
print(f"[MPC __init__] sys.path[0]: {sys.path[0]}")

# Try to list what Python sees in this directory
import os
mpc_dir = os.path.dirname(__file__)
print(f"[MPC __init__] Directory contents:")
for f in os.listdir(mpc_dir):
    if f.endswith('.py'):
        print(f"  - {f}")

