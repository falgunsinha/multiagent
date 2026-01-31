"""
Quick test script for URDF-based batched IK solver
Run this to verify the implementation works before using in Isaac Sim
"""

import sys
from pathlib import Path
import numpy as np

# Add src/mpc to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src" / "mpc"))

# Import modules
from urdf_kinematics import URDFKinematics
from batched_ik_urdf import BatchedIKSolverURDF

def test_urdf_fk():
    """Test URDF-based forward kinematics"""
    print("\n" + "="*60)
    print("TEST 1: URDF Forward Kinematics")
    print("="*60)
    
    urdf_path = str(project_root / "assets" / "franka_panda.urdf")
    
    try:
        fk = URDFKinematics(urdf_path, device="cuda:0")
        print("✓ URDF kinematics loaded successfully")
        
        # Test with default joint configuration
        import torch
        default_joints = torch.tensor(
            [[0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75]],
            device=fk.device, dtype=torch.float32
        )
        
        pos, quat = fk.forward_kinematics(default_joints)
        
        print(f"\nDefault configuration FK result:")
        print(f"  Position: {pos[0].cpu().numpy()}")
        print(f"  Quaternion [w,x,y,z]: {quat[0].cpu().numpy()}")
        print("✓ Forward kinematics working!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batched_ik():
    """Test batched IK solver"""
    print("\n" + "="*60)
    print("TEST 2: Batched IK Solver")
    print("="*60)
    
    urdf_path = str(project_root / "assets" / "franka_panda.urdf")
    
    try:
        solver = BatchedIKSolverURDF(
            urdf_path=urdf_path,
            num_seeds=50,  # Use fewer seeds for quick test
            position_threshold=0.01,
            rotation_threshold=0.1,
            mppi_iterations=10,  # Fewer iterations for quick test
            lbfgs_iterations=50,
            device="cuda:0"
        )
        print("✓ Batched IK solver initialized")
        
        # Test IK solve
        target_position = np.array([0.4, 0.0, 0.5])  # Reachable position
        target_orientation = np.array([0.0, 1.0, 0.0, 0.0])  # [w,x,y,z]
        
        print(f"\nSolving IK for target:")
        print(f"  Position: {target_position}")
        print(f"  Orientation: {target_orientation}")
        
        solution, success, info = solver.solve(
            target_position=target_position,
            target_orientation=target_orientation,
            current_joints=None
        )
        
        if success:
            print(f"\n✓ IK Solution found!")
            print(f"  Joint angles: {solution}")
            print(f"  Position error: {info['position_error']*100:.2f} cm")
            print(f"  Rotation error: {info['rotation_error']:.4f}")
            print(f"  Solve time: {info['solve_time']:.3f} s")
            print(f"  Successful seeds: {info['num_successful']}/{solver.num_seeds}")
        else:
            print(f"\n⚠ IK did not converge to threshold")
            print(f"  Best position error: {info['position_error']*100:.2f} cm")
            print(f"  Best rotation error: {info['rotation_error']:.4f}")
            print(f"  Solve time: {info['solve_time']:.3f} s")
        
        return success
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("URDF-Based Batched IK Test Suite")
    print("="*60)
    
    # Check URDF file exists
    urdf_path = project_root / "assets" / "franka_panda.urdf"
    if not urdf_path.exists():
        print(f"✗ URDF file not found: {urdf_path}")
        print("  Please make sure franka_panda.urdf is in the assets folder")
        return
    
    print(f"✓ URDF file found: {urdf_path}")
    
    # Run tests
    test1_passed = test_urdf_fk()
    test2_passed = test_batched_ik()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"  URDF FK Test: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"  Batched IK Test: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Ready to use in Isaac Sim.")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

