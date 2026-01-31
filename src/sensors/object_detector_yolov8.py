"""
YOLOv8 Object Detector Module with 3D Detection
Object detection using YOLOv8 (Ultralytics) + Depth Camera.
Detects cubes, cylinders, cuboids, and containers.

Classification:
- Cubes and cylinders are classified as 'target' (objects to pick)
- Cuboids are classified as 'obstacle' (objects to avoid)
- Containers are classified as 'container' (destination for placing)

Features:
- Non-Maximum Suppression (NMS) built into YOLOv8
- 3D position estimation using depth camera
- Automatic size estimation from bounding boxes + depth
- Configurable confidence threshold
- GPU acceleration support
- No network download required (uses local model)

Based on Robot Environment framework with Isaac Sim compatibility.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


class YOLOv8ObjectDetector:
    """YOLOv8-based object detector using custom trained model"""

    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = 0.20,
        device: str = "auto",
        camera_params: Dict = None
    ):
        """
        Initialize YOLOv8 object detector with 3D detection support.

        Args:
            model_path: Path to YOLOv8 model (.pt file). If None, uses default custom model.
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
            device: Device to use ("cuda", "cpu", or "auto")
            camera_params: Camera intrinsic parameters for 3D unprojection
                          {'fx': focal_x, 'fy': focal_y, 'cx': center_x, 'cy': center_y}
        """
        self.confidence_threshold = confidence_threshold

        # Default to custom trained model
        if model_path is None:
            model_path = r"C:\isaacsim\cobotproject\models\augmented_shapes_v2\weights\best.pt"
        
        self.model_path = model_path

        # Auto-detect device
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Camera parameters for 3D unprojection
        self.camera_params = camera_params

        # Initialize model (lazy loading)
        self.model = None
        self._model_loaded = False

        print(f"[YOLOV8 DETECTOR] Initialized")
        print(f"[YOLOV8 DETECTOR] Model path: {model_path}")
        print(f"[YOLOV8 DETECTOR] Confidence threshold: {confidence_threshold}")
        print(f"[YOLOV8 DETECTOR] Device: {self.device}")
        if camera_params:
            print(f"[YOLOV8 DETECTOR] 3D detection enabled with camera params")

    def _load_model(self):
        """Load YOLOv8 model (lazy loading)"""
        if self._model_loaded:
            return
        
        try:
            print("[YOLOV8 DETECTOR] Loading YOLOv8 model...")
            from ultralytics import YOLO
            
            # Load model
            self.model = YOLO(self.model_path)
            
            # Set device
            if self.device == "cuda":
                self.model.to('cuda')
            
            self._model_loaded = True
            print(f"[YOLOV8 DETECTOR] Model loaded successfully on {self.device}")
            print(f"[YOLOV8 DETECTOR] Classes: {self.model.names}")
            
        except Exception as e:
            print(f"[YOLOV8 DETECTOR ERROR] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def detect(self, rgb_image: np.ndarray, verbose: bool = False) -> List[Dict]:
        """
        Detect objects in RGB image using YOLOv8.

        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            verbose: Print detection details

        Returns:
            List of detected objects with classification
        """
        try:
            # Load model if not already loaded
            self._load_model()
            
            if verbose:
                print(f"[YOLOV8 DETECTOR] Input image shape: {rgb_image.shape}")
            
            # Run inference
            results = self.model(rgb_image, conf=self.confidence_threshold, verbose=False)
            
            detected_objects = []
            
            # Parse detections
            for result in results:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    
                    # Get confidence and class
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    # Calculate center and dimensions
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    
                    # Classify object type
                    object_type = self._classify_object_type(class_name)
                    
                    detection = {
                        'class': class_name,
                        'type': object_type,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'width': width,
                        'height': height,
                    }
                    
                    detected_objects.append(detection)
                    
                    if verbose:
                        print(f"[YOLOV8 DETECTOR] Detected {class_name} -> {object_type} (conf={confidence:.2%})")
            
            # Sort by confidence (highest first)
            detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
            
            return detected_objects
            
        except Exception as e:
            print(f"[YOLOV8 DETECTOR ERROR] Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _classify_object_type(self, class_name: str) -> str:
        """
        Classify object as target, obstacle, or container.

        Args:
            class_name: Object class name

        Returns:
            Object type: 'target', 'obstacle', or 'container'
        """
        class_lower = class_name.lower()

        # Container classification
        if 'container' in class_lower or 'box' in class_lower:
            return 'container'

        # Target classification (objects to pick)
        # Cubes and cylinders are targets
        if 'cube' in class_lower and 'cuboid' not in class_lower:
            return 'target'

        if 'cylinder' in class_lower:
            return 'target'

        # Obstacle classification (objects to avoid)
        # Cuboids are obstacles
        if 'cuboid' in class_lower:
            return 'obstacle'

        # Everything else is an obstacle
        return 'obstacle'

    def get_targets(self, detections: List[Dict]) -> List[Dict]:
        """Get only target objects (cubes/cylinders to pick)"""
        return [d for d in detections if d['type'] == 'target']

    def get_obstacles(self, detections: List[Dict]) -> List[Dict]:
        """Get only obstacle objects (to avoid)"""
        return [d for d in detections if d['type'] == 'obstacle']

    def get_containers(self, detections: List[Dict]) -> List[Dict]:
        """Get only container objects"""
        return [d for d in detections if d['type'] == 'container']

    def get_objects_by_label(self, detections: List[Dict], label: str) -> List[Dict]:
        """
        Get objects by specific label.

        Args:
            detections: List of detections
            label: Label to filter by (e.g., "cube", "cylinder")

        Returns:
            List of detections matching the label
        """
        return [d for d in detections if d['class'].lower() == label.lower()]

    def clear_cache(self):
        """Clear GPU cache to free memory"""
        if self.device == "cuda":
            import torch
            torch.cuda.empty_cache()
            print("[YOLOV8 DETECTOR] GPU cache cleared")

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage.

        Returns:
            Dictionary with memory usage info
        """
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()

        result = {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        }

        if self.device == "cuda":
            import torch
            if torch.cuda.is_available():
                result['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
                result['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024

        return result

    def detect_3d(self, rgb_image: np.ndarray, depth_image: np.ndarray, verbose: bool = False) -> List[Dict]:
        """
        Detect objects in 3D using RGB + Depth images.

        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            depth_image: Depth image as numpy array (H, W) in meters (radial distance from distance_to_camera)
            verbose: Print detection details

        Returns:
            List of detected objects with 3D positions and sizes
        """
        # First, run 2D detection
        detections_2d = self.detect(rgb_image, verbose=verbose)

        if len(detections_2d) == 0:
            return []

        # Debug: Reset unproject counter for each detection cycle
        if hasattr(self, '_unproject_debug_count'):
            self._unproject_debug_count = 0

        # Add 3D information to each detection
        detections_3d = []
        for i, det in enumerate(detections_2d):
            # Get 3D position from depth
            position_3d = self._unproject_to_3d(det['center'], depth_image)

            # Estimate 3D size from bounding box + depth
            size_3d = self._estimate_3d_size(det['bbox'], depth_image)

            # Add 3D information
            det['position_3d'] = position_3d
            det['size_3d'] = size_3d

            detections_3d.append(det)

            if verbose or i < 5:  # Print first 5 detections
                print(f"[3D DETECTION {i+1}] {det['class']}: bbox={det['bbox']}, center={det['center']}, pos_3d={position_3d}, size_3d={size_3d}")

        return detections_3d

    def _unproject_to_3d(self, image_point: Tuple[int, int], depth_image: np.ndarray) -> np.ndarray:
        """
        Unproject 2D image point to 3D camera frame using depth.

        SIMPLIFIED APPROACH:
        - depth_image contains radial distance from camera to object (distance_to_camera annotator)
        - We convert pixel (u,v) + radial_distance to 3D direction vector
        - Result is in camera frame (NOT world frame - transform happens later)

        Args:
            image_point: (x, y) in image coordinates
            depth_image: Depth image (H, W) in meters (radial distance from camera)

        Returns:
            3D position as np.array([x, y, z]) in camera frame
        """
        if self.camera_params is None:
            return None

        u, v = image_point

        # Get depth at this pixel (with bounds checking)
        h, w = depth_image.shape
        v = max(0, min(v, h - 1))
        u = max(0, min(u, w - 1))
        radial_distance = depth_image[v, u]

        # Debug: Print depth at detection point
        if not hasattr(self, '_unproject_debug_count'):
            self._unproject_debug_count = 0

        if self._unproject_debug_count < 5:
            print(f"\n[UNPROJECT DEBUG {self._unproject_debug_count+1}] Pixel ({u}, {v}): radial_distance={radial_distance:.3f}m")
            self._unproject_debug_count += 1

        # Check for invalid depth
        if radial_distance <= 0 or np.isnan(radial_distance) or np.isinf(radial_distance):
            if self._unproject_debug_count <= 5:
                print(f"  Invalid depth!")
            return None

        # Unproject using camera intrinsics
        fx = self.camera_params['fx']
        fy = self.camera_params['fy']
        cx = self.camera_params['cx']
        cy = self.camera_params['cy']

        # SIMPLIFIED: Convert pixel + radial distance to 3D direction
        # The radial distance is the distance from camera origin to the point
        # We need to find the 3D point (x, y, z) such that:
        # 1. It projects to pixel (u, v): u = fx*x/z + cx, v = fy*y/z + cy
        # 2. Its distance from origin is radial_distance: √(x² + y² + z²) = radial_distance

        # From projection equations:
        # x/z = (u - cx)/fx
        # y/z = (v - cy)/fy
        # Let's call these ratios: rx = (u-cx)/fx, ry = (v-cy)/fy

        rx = (u - cx) / fx
        ry = (v - cy) / fy

        # Now: x = rx*z, y = ry*z
        # Substitute into distance equation:
        # √((rx*z)² + (ry*z)² + z²) = radial_distance
        # √(z² * (rx² + ry² + 1)) = radial_distance
        # z * √(rx² + ry² + 1) = radial_distance
        # z = radial_distance / √(rx² + ry² + 1)

        scale = np.sqrt(rx**2 + ry**2 + 1.0)
        z_cam = radial_distance / scale
        x_cam = rx * z_cam
        y_cam = ry * z_cam

        if self._unproject_debug_count <= 5:
            print(f"  Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
            print(f"  Direction ratios: rx={rx:.3f}, ry={ry:.3f}, scale={scale:.3f}")
            print(f"  Camera frame: ({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f})")
            print(f"  Verification: distance = {np.linalg.norm([x_cam, y_cam, z_cam]):.3f}m (should be {radial_distance:.3f}m)")

        return np.array([x_cam, y_cam, z_cam])

    def _estimate_3d_size(self, bbox: Tuple[int, int, int, int], depth_image: np.ndarray) -> np.ndarray:
        """
        Estimate 3D size of object from 2D bounding box + depth.

        SIMPLIFIED APPROACH:
        - Get median depth in bounding box
        - Convert bbox pixel dimensions to 3D dimensions using depth
        - Assume object depth ≈ min(width, height) for cubes

        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            depth_image: Depth image (H, W) in meters (radial distance)

        Returns:
            3D size as np.array([width, height, depth]) in meters
        """
        if self.camera_params is None:
            return None

        u1, v1, u2, v2 = bbox

        # Get median depth in bounding box region (more robust than mean)
        h, w = depth_image.shape
        v1 = max(0, min(v1, h - 1))
        v2 = max(0, min(v2, h - 1))
        u1 = max(0, min(u1, w - 1))
        u2 = max(0, min(u2, w - 1))

        depth_roi = depth_image[v1:v2, u1:u2]
        valid_depths = depth_roi[(depth_roi > 0) & ~np.isnan(depth_roi) & ~np.isinf(depth_roi)]

        if len(valid_depths) == 0:
            return None

        median_depth = np.median(valid_depths)

        # Get camera intrinsics
        fx = self.camera_params['fx']
        fy = self.camera_params['fy']
        cx = self.camera_params['cx']
        cy = self.camera_params['cy']

        # Unproject bbox corners to 3D to get actual size
        # Top-left corner
        rx1 = (u1 - cx) / fx
        ry1 = (v1 - cy) / fy
        scale1 = np.sqrt(rx1**2 + ry1**2 + 1.0)
        z1 = median_depth / scale1
        x1_3d = rx1 * z1
        y1_3d = ry1 * z1

        # Bottom-right corner
        rx2 = (u2 - cx) / fx
        ry2 = (v2 - cy) / fy
        scale2 = np.sqrt(rx2**2 + ry2**2 + 1.0)
        z2 = median_depth / scale2
        x2_3d = rx2 * z2
        y2_3d = ry2 * z2

        # Calculate 3D dimensions
        width_3d = abs(x2_3d - x1_3d)
        height_3d = abs(y2_3d - y1_3d)

        # Depth estimation: assume object is roughly cubic
        # For cubes: depth ≈ min(width, height)
        # For cuboids: depth might be different, but this is a reasonable estimate
        depth_3d = min(width_3d, height_3d)

        return np.array([width_3d, height_3d, depth_3d])

