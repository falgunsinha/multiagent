"""
Module Reloader for Isaac Sim
Clears Python module cache to force reload of custom modules
"""

import sys
import importlib

print("\n" + "=" * 60)
print("MODULE RELOADER")
print("=" * 60)

# List of modules to reload
modules_to_reload = [
    'src',
    'src.sensors.physx_lidar_sensor',
    'src.sensors.depth_camera_sensor',
    'src.sensors.collision_checker',
    'src.sensors.object_detector_owlv2',
    'src.sensors.femto_camera',
    'src.sensors.maskrcnn_detector',
    'src.sensors.custom_maskrcnn_detector',
    'src.sensors',
    'src.grippers.gripper',
    'src.grippers.parallel_gripper',
    'src.grippers.surface_gripper',
    'src.grippers',
    'src.manipulators.single_manipulator',
    'src.manipulators',
    'src.controllers.rrt_pick_place_controller',
    'src.controllers'
]

print("\nRemoving modules from cache...")
# Remove all modules that start with 'src'
modules_to_delete = [key for key in sys.modules.keys() if key.startswith('src')]
for module_name in modules_to_delete:
    print(f"  Removing: {module_name}")
    del sys.modules[module_name]

# Also remove specific modules
for module_name in modules_to_reload:
    if module_name in sys.modules:
        print(f"  Removing: {module_name}")
        del sys.modules[module_name]

print("\nReloading imports...")

# Test imports
try:
    from src.sensors import FemtoCamera
    print("✓ FemtoCamera imported successfully")
except Exception as e:
    print(f"✗ FemtoCamera import failed: {e}")

try:
    from src.controllers.rrt_pick_place_controller import RRTPickPlaceController
    print("✓ RRTPickPlaceController imported successfully")
except Exception as e:
    print(f"✗ RRTPickPlaceController import failed: {e}")

try:
    from src.sensors.object_detector_owlv2 import OwlV2ObjectDetector
    print("✓ OwlV2ObjectDetector imported successfully")
except Exception as e:
    print(f"✗ OwlV2ObjectDetector import failed: {e}")

try:
    from src.sensors.physx_lidar_sensor import PhysXLidarSensor
    print("✓ PhysXLidarSensor imported successfully")
except Exception as e:
    print(f"✗ PhysXLidarSensor import failed: {e}")

try:
    from src.sensors.collision_checker import CollisionChecker
    print("✓ CollisionChecker imported successfully")
except Exception as e:
    print(f"✗ CollisionChecker import failed: {e}")

try:
    from src.sensors.maskrcnn_detector import MaskRCNNDetector
    print("✓ MaskRCNNDetector imported successfully")
except Exception as e:
    print(f"✗ MaskRCNNDetector import failed: {e}")

try:
    from src.sensors.custom_maskrcnn_detector import CustomMaskRCNNDetector
    print("✓ CustomMaskRCNNDetector imported successfully")
except Exception as e:
    print(f"✗ CustomMaskRCNNDetector import failed: {e}")

print("\n" + "=" * 60)
print("MODULE RELOAD COMPLETE")
print("=" * 60)
print("\nYou can now run the main script:")
print("  franka_rrt_objDetection_pickplace_v2.0.py")

