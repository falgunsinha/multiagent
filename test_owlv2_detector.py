"""
Test OwlV2 Object Detector
Simple test script to verify OwlV2 detector works with Isaac Sim's Python.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.sensors.object_detector_owlv2 import OwlV2ObjectDetector


def test_detector():
    """Test OwlV2 detector with random image"""
    
    print("=" * 60)
    print("Testing OwlV2 Object Detector")
    print("=" * 60)
    
    # Initialize detector
    print("\n1. Initializing detector...")
    detector = OwlV2ObjectDetector(
        object_labels=["cube", "cylinder", "cuboid", "container"],
        confidence_threshold=0.20,
        device="auto"
    )
    print("   Detector initialized successfully!")
    
    # Create test image (random noise)
    print("\n2. Creating test image (640x480 RGB)...")
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"   Test image shape: {test_image.shape}")
    
    # Run detection
    print("\n3. Running detection (this will download the model on first run)...")
    detections = detector.detect(test_image, verbose=True)
    print(f"   Detection completed. Found {len(detections)} objects")
    
    # Test classification methods
    print("\n4. Testing classification methods...")
    targets = detector.get_targets(detections)
    obstacles = detector.get_obstacles(detections)
    containers = detector.get_containers(detections)
    
    print(f"   Targets: {len(targets)}")
    print(f"   Obstacles: {len(obstacles)}")
    print(f"   Containers: {len(containers)}")
    
    # Test label filtering
    print("\n5. Testing label filtering...")
    cubes = detector.get_objects_by_label(detections, "cube")
    cylinders = detector.get_objects_by_label(detections, "cylinder")
    cuboids = detector.get_objects_by_label(detections, "cuboid")
    
    print(f"   Cubes: {len(cubes)}")
    print(f"   Cylinders: {len(cylinders)}")
    print(f"   Cuboids: {len(cuboids)}")
    
    # Memory usage
    print("\n6. Checking memory usage...")
    memory = detector.get_memory_usage()
    print(f"   RSS: {memory['rss_mb']:.1f} MB")
    print(f"   VMS: {memory['vms_mb']:.1f} MB")
    if 'gpu_allocated_mb' in memory:
        print(f"   GPU Allocated: {memory['gpu_allocated_mb']:.1f} MB")
        print(f"   GPU Reserved: {memory['gpu_reserved_mb']:.1f} MB")
    
    # Clear cache
    print("\n7. Clearing GPU cache...")
    detector.clear_cache()
    print("   Cache cleared!")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    
    return detector, detections


if __name__ == "__main__":
    try:
        detector, detections = test_detector()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

