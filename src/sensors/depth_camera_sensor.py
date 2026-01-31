"""
Depth Camera Sensor Module
Handles depth camera data acquisition and processing.

Features:
- Depth data acquisition from SingleViewDepthSensor
- Data validation and error handling
"""

import numpy as np
from typing import Optional
import carb


class DepthCameraSensor:
    """
    Depth camera sensor wrapper for 3D detection.

    Provides depth data for 3D object detection and obstacle detection.
    """

    def __init__(self, depth_camera_sensor, verbose: bool = False):
        """
        Initialize depth camera sensor.

        Args:
            depth_camera_sensor: Isaac Sim SingleViewDepthSensor instance
            verbose: Print detailed information
        """
        self.depth_camera = depth_camera_sensor
        self.verbose = verbose

    def initialize_and_configure(
        self,
        min_distance: float = 0.1,
        max_distance: float = 3.0,
        baseline_mm: float = 55.0,
        focal_length_pixel: float = 320.0,
        confidence_threshold: float = 0.90
    ):
        """
        Initialize and configure depth camera sensor.

        Args:
            min_distance: Minimum distance in meters (default: 0.1m = 10cm)
            max_distance: Maximum distance in meters (default: 3.0m)
            baseline_mm: Baseline in millimeters (default: 55mm)
            focal_length_pixel: Focal length in pixels (default: 320.0)
            confidence_threshold: Confidence threshold (default: 0.90)
        """
        try:
            # Initialize with attach_rgb_annotator=False for better performance
            self.depth_camera.initialize(attach_rgb_annotator=False)

            # Attach depth annotator for distance measurements
            self.depth_camera.attach_annotator("DepthSensorDistance")

            # Configure depth sensor parameters
            self.depth_camera.set_enabled(enabled=True)
            self.depth_camera.set_min_distance(min_distance)
            self.depth_camera.set_max_distance(max_distance)
            self.depth_camera.set_baseline_mm(baseline_mm)
            self.depth_camera.set_focal_length_pixel(focal_length_pixel)
            self.depth_camera.set_confidence_threshold(confidence_threshold)

            if self.verbose:
                print(f"[DEPTH CAMERA] Initialized and configured:")
                print(f"  Min distance: {min_distance}m")
                print(f"  Max distance: {max_distance}m")
                print(f"  Baseline: {baseline_mm}mm")
                print(f"  Focal length: {focal_length_pixel}px")
                print(f"  Confidence threshold: {confidence_threshold}")

        except Exception as e:
            carb.log_warn(f"[DEPTH CAMERA] Error initializing: {e}")
    
    def get_depth_data(self) -> Optional[np.ndarray]:
        """
        Get depth data from depth camera.
        
        Returns:
            Depth image as numpy array (H, W) in meters, or None if unavailable
        """
        try:
            if self.depth_camera is None:
                return None
            
            # Get depth data from annotator
            depth_data = self.depth_camera.get_depth_data()
            
            if depth_data is None or len(depth_data.shape) != 2:
                return None
            
            return depth_data
            
        except Exception as e:
            if self.verbose:
                carb.log_warn(f"[DEPTH CAMERA] Error getting depth data: {e}")
            return None
    
    def get_rgb_data(self) -> Optional[np.ndarray]:
        """
        Get RGB data from depth camera (if available).
        
        Returns:
            RGB image as numpy array (H, W, 3), or None if unavailable
        """
        try:
            if self.depth_camera is None:
                return None
            
            # Get RGB data if available
            rgb_data = self.depth_camera.get_rgb_data()
            
            if rgb_data is None:
                return None
            
            return rgb_data
            
        except Exception as e:
            if self.verbose:
                carb.log_warn(f"[DEPTH CAMERA] Error getting RGB data: {e}")
            return None

