"""
Test if PyTorch is available in Isaac Sim's Python environment
Run this script in Isaac Sim's Script Editor to verify PyTorch installation
"""

print("=" * 60)
print("Testing PyTorch Import in Isaac Sim")
print("=" * 60)

# Test 1: Import PyTorch
try:
    import torch
    print("✅ PyTorch imported successfully")
    print(f"   PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")
    print("   You need to install PyTorch in Isaac Sim's Python environment")
    exit(1)

# Test 2: Check CUDA availability
try:
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✅ CUDA is available")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        print(f"   Current GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA is NOT available - will use CPU (slower)")
except Exception as e:
    print(f"❌ CUDA check failed: {e}")

# Test 3: Test basic tensor operations
try:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.randn(10, 10, device=device)
    y = torch.matmul(x, x.T)
    print(f"✅ Tensor operations work on {device}")
except Exception as e:
    print(f"❌ Tensor operations failed: {e}")

# Test 4: Import our MPC modules
print("\n" + "=" * 60)
print("Testing MPC Module Imports")
print("=" * 60)

import sys
from pathlib import Path

project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"✅ Added project root to sys.path: {project_root}")

# First check if the files exist
import os
mpc_dir = project_root / "src" / "mpc"
print(f"\nChecking MPC directory: {mpc_dir}")
print(f"Directory exists: {mpc_dir.exists()}")
if mpc_dir.exists():
    print("Files in MPC directory:")
    for f in mpc_dir.glob("*.py"):
        print(f"  - {f.name} ({f.stat().st_size} bytes)")

# Check sys.path
print(f"\nPython sys.path:")
for i, p in enumerate(sys.path[:5]):
    print(f"  {i}: {p}")

# Try importing step by step
print("\nTrying imports step by step:")

try:
    import src
    print("✅ 'src' package imported")
except ImportError as e:
    print(f"❌ 'src' package import failed: {e}")

try:
    import src.mpc
    print("✅ 'src.mpc' package imported")
except ImportError as e:
    print(f"❌ 'src.mpc' package import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    import src.mpc.torch_kinematics
    print("✅ 'src.mpc.torch_kinematics' module imported")
    from src.mpc.torch_kinematics import FrankaBatchFK
    print("✅ FrankaBatchFK class imported")
except ImportError as e:
    print(f"❌ torch_kinematics import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from src.mpc.torch_mppi import TorchMPPI
    print("✅ TorchMPPI imported successfully")
except ImportError as e:
    print(f"❌ torch_mppi import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from src.mpc.torch_mpc_planner import TorchMPCPlanner
    print("✅ TorchMPCPlanner imported successfully (direct)")
except ImportError as e:
    print(f"❌ torch_mpc_planner import failed (direct): {e}")
    import traceback
    traceback.print_exc()

print("\n" + "-" * 60)
print("Trying wrapper import method:")
print("-" * 60)

try:
    from src.mpc.gpu_mpc import TorchMPCPlanner
    print("✅ TorchMPCPlanner imported successfully (via gpu_mpc wrapper)!")
    print(f"   Class: {TorchMPCPlanner}")
except ImportError as e:
    print(f"❌ gpu_mpc wrapper import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)

