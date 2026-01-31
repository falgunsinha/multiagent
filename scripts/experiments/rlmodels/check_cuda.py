"""
Quick CUDA/GPU Check Script
Verifies that CUDA is available for the experiments
"""

import sys

def check_cuda():
    """Check CUDA availability and GPU information"""
    print("\n" + "="*80)
    print("CUDA/GPU AVAILABILITY CHECK")
    print("="*80 + "\n")
    
    # Check PyTorch
    try:
        import torch
        print("✅ PyTorch installed")
        print(f"   Version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print("\n✅ CUDA IS AVAILABLE!")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"   Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"\n   GPU {i}:")
                print(f"      Name: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"      Memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"      Compute Capability: {props.major}.{props.minor}")
                print(f"      Multi-Processors: {props.multi_processor_count}")
            
            # Test GPU computation
            print("\n   Testing GPU computation...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("   ✅ GPU computation successful!")
            
            # Check memory
            print(f"\n   GPU Memory Usage:")
            print(f"      Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"      Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
            
        else:
            print("\n⚠️  CUDA IS NOT AVAILABLE")
            print("   Experiments will run on CPU (slower)")
            print("\n   Possible reasons:")
            print("   1. No NVIDIA GPU detected")
            print("   2. CUDA toolkit not installed")
            print("   3. PyTorch installed without CUDA support")
            print("\n   To fix:")
            print("   - Install CUDA toolkit from: https://developer.nvidia.com/cuda-downloads")
            print("   - Reinstall PyTorch with CUDA:")
            print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            
    except ImportError:
        print("❌ PyTorch not installed")
        print("   Install with: pip install torch torchvision torchaudio")
        return False
    
    # Check Stable-Baselines3
    print("\n" + "-"*80)
    try:
        import stable_baselines3 as sb3
        print("✅ Stable-Baselines3 installed")
        print(f"   Version: {sb3.__version__}")
    except ImportError:
        print("⚠️  Stable-Baselines3 not installed")
        print("   Install with: pip install stable-baselines3")
    
    # Check Isaac Sim
    print("\n" + "-"*80)
    try:
        from isaacsim import SimulationApp
        print("✅ Isaac Sim available")
    except ImportError:
        print("⚠️  Isaac Sim not available (this is OK if running standalone)")
    
    print("\n" + "="*80)
    
    if torch.cuda.is_available():
        print("✅ SYSTEM READY FOR GPU-ACCELERATED EXPERIMENTS!")
        print(f"   Expected speedup: 2-3x faster than CPU")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  SYSTEM WILL USE CPU (SLOWER)")
        print("   Expected runtime: 3-6 hours (vs 1-2 hours with GPU)")
    
    print("="*80 + "\n")
    
    return torch.cuda.is_available()

if __name__ == "__main__":
    cuda_available = check_cuda()
    sys.exit(0 if cuda_available else 1)

