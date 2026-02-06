
print("Checking dependencies...\n")

# Check PyTorch Geometric packages
try:
    import torch_geometric
    print(f" torch-geometric: INSTALLED - {torch_geometric.__version__}")
except ImportError:
    print(" torch-geometric: NOT INSTALLED")

try:
    import torch_scatter
    print(" torch-scatter: INSTALLED")
except ImportError:
    print(" torch-scatter: NOT INSTALLED")

try:
    import torch_sparse
    print(" torch-sparse: INSTALLED")
except ImportError:
    print("torch-sparse: NOT INSTALLED")

try:
    import torch_cluster
    print(" torch-cluster: INSTALLED")
except ImportError:
    print(" torch-cluster: NOT INSTALLED")

print("\n" + "="*60)

# Check other dependencies (should already be installed)
print("\nOther dependencies:")
try:
    import torch
    print(f" torch: {torch.__version__}")
except ImportError:
    print(" torch: NOT INSTALLED")

try:
    import numpy
    print(f" numpy: {numpy.__version__}")
except ImportError:
    print(" numpy: NOT INSTALLED")

try:
    import scipy
    print(f" scipy: {scipy.__version__}")
except ImportError:
    print(" scipy: NOT INSTALLED")

try:
    import pandas
    print(f" pandas: {pandas.__version__}")
except ImportError:
    print(" pandas: NOT INSTALLED")

try:
    import matplotlib
    print(f" matplotlib: {matplotlib.__version__}")
except ImportError:
    print(" matplotlib: NOT INSTALLED")

try:
    import seaborn
    print(f" seaborn: {seaborn.__version__}")
except ImportError:
    print(" seaborn: NOT INSTALLED")

try:
    import yaml
    print(f" PyYAML: INSTALLED")
except ImportError:
    print(" PyYAML: NOT INSTALLED")

print("="*60)

