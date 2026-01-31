"""
Orbbec Femto Mega Camera Module

Provides RGB and Depth data from Femto Mega camera using Replicator API.
Handles camera setup, configuration, and data acquisition.
"""

import numpy as np
from pxr import Gf, UsdGeom
import omni.usd
import omni.replicator.core as rep
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.camera import Camera


class FemtoCamera:
    """Orbbec Femto Mega RGB-D Camera"""
    
    def __init__(self, prim_path="/World/Femto", position=None, rotation=None, scale=None):
        """
        Initialize Femto Mega camera
        
        Args:
            prim_path: USD prim path for camera
            position: Camera position [x, y, z] in meters (default: [0.8, 0.0, 0.8])
            rotation: Camera rotation [rx, ry, rz] in degrees (default: [-67.8, -0.1, 90.5])
            scale: Camera scale [sx, sy, sz] (default: [0.01, 0.01, 0.01])
        """
        self.prim_path = prim_path
        self.position = position if position is not None else [0.8, 0.0, 0.8]
        self.rotation = rotation if rotation is not None else [-67.8, -0.1, 90.5]
        self.scale = scale if scale is not None else [0.01, 0.01, 0.01]
        
        # Camera paths
        self.rgb_camera_prim_path = None
        self.depth_camera_prim_path = None
        
        # Replicator components
        self.render_product = None
        self.rgb_annotator = None
        self.depth_annotator = None
        
        # Camera intrinsics (default for Femto Mega)
        self.camera_params = {
            'fx': 320.0,
            'fy': 320.0,
            'cx': 320.0,
            'cy': 320.0
        }

        # Isaac Sim Camera wrapper for coordinate transformations
        self.camera_wrapper = None
        
    def setup(self, resolution=(640, 640), focal_length=2.3):
        """
        Setup Femto camera in the scene
        
        Args:
            resolution: Camera resolution (width, height)
            focal_length: RGB camera focal length in mm
        """
        # Add Femto USD to stage
        femto_assets_root = get_assets_root_path()
        femto_usd_path = femto_assets_root + "/Isaac/Sensors/Orbbec/FemtoMega/orbbec_femtomega_v1.0.usd"
        add_reference_to_stage(usd_path=femto_usd_path, prim_path=self.prim_path)
        
        # Configure Femto base transform
        stage = omni.usd.get_context().get_stage()
        femto_prim = stage.GetPrimAtPath(self.prim_path)
        
        if femto_prim.IsValid():
            self._clear_xform_ops(femto_prim)
            
            femto_xformable = UsdGeom.Xformable(femto_prim)
            translate_op = femto_xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(*self.position))
            
            rotate_op = femto_xformable.AddRotateXYZOp()
            rotate_op.Set(Gf.Vec3f(*self.rotation))
            
            scale_op = femto_xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
            scale_op.Set(Gf.Vec3d(*self.scale))
        
        # Configure FemtoMega body transform
        femto_mega_prim_path = self.prim_path + "/FemtoMega"
        femto_mega_prim = stage.GetPrimAtPath(femto_mega_prim_path)
        
        if femto_mega_prim.IsValid():
            self._clear_xform_ops(femto_mega_prim)
            
            xformable = UsdGeom.Xformable(femto_mega_prim)
            translate_op = xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
            
            rotate_op = xformable.AddRotateXYZOp()
            rotate_op.Set(Gf.Vec3f(0.0, 0.0, 180.0))
        
        # Setup RGB camera
        self.rgb_camera_prim_path = self.prim_path + "/FemtoMega/MegaBody/RgbSensor/at_rgb2114_module/lens_hold/lens_mc03/solid/solid/camera_rgb"
        rgb_camera_prim = stage.GetPrimAtPath(self.rgb_camera_prim_path)

        print(f"[FEMTO DEBUG] RGB camera prim path: {self.rgb_camera_prim_path}")
        print(f"[FEMTO DEBUG] RGB camera prim valid: {rgb_camera_prim.IsValid()}")

        if rgb_camera_prim.IsValid():
            rgb_camera_schema = UsdGeom.Camera(rgb_camera_prim)
            rgb_camera_schema.GetFocalLengthAttr().Set(focal_length)
            print(f"[FEMTO] RGB camera focal length set to {focal_length} mm")
        else:
            print(f"[FEMTO ERROR] RGB camera prim not found at {self.rgb_camera_prim_path}")
        
        # Setup Replicator annotators
        self.render_product = rep.create.render_product(self.rgb_camera_prim_path, resolution=resolution)

        self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb_annotator.attach([self.render_product])

        # Use distance_to_camera for RGB-D depth (Euclidean distance from camera origin)
        # This is the radial distance, which we'll convert to Z-depth in unprojection
        self.depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        self.depth_annotator.attach([self.render_product])

        # DEBUG: Print actual RGB camera world position and compute intrinsics
        print(f"[FEMTO DEBUG] About to compute camera intrinsics...")
        print(f"[FEMTO DEBUG] RGB camera prim valid (2nd check): {rgb_camera_prim.IsValid()}")

        if rgb_camera_prim.IsValid():
            print(f"[FEMTO DEBUG] Computing camera world position...")
            xformable = UsdGeom.Xformable(rgb_camera_prim)
            transform_matrix = xformable.ComputeLocalToWorldTransform(0)
            camera_world_pos = transform_matrix.ExtractTranslation()

            print(f"[FEMTO DEBUG] Getting camera intrinsics from USD...")
            # Get actual camera intrinsics from USD
            camera_schema = UsdGeom.Camera(rgb_camera_prim)
            focal_length_mm = camera_schema.GetFocalLengthAttr().Get()  # in mm
            h_aperture_mm = camera_schema.GetHorizontalApertureAttr().Get()  # in mm
            v_aperture_mm = camera_schema.GetVerticalApertureAttr().Get()  # in mm

            # Calculate intrinsics in pixels
            # fx = (focal_length / horizontal_aperture) * image_width
            # fy = (focal_length / vertical_aperture) * image_height
            # cx = image_width / 2
            # cy = image_height / 2
            fx = (focal_length_mm / h_aperture_mm) * resolution[0]
            fy = (focal_length_mm / v_aperture_mm) * resolution[1]
            cx = resolution[0] / 2.0
            cy = resolution[1] / 2.0

            # Update camera params with actual values
            self.camera_params = {
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy
            }

            print(f"\n[FEMTO DEBUG] RGB Camera Configuration:")
            print(f"  Femto parent position (set by user): {self.position}")
            print(f"  RGB camera actual world position: [{camera_world_pos[0]:.5f}, {camera_world_pos[1]:.5f}, {camera_world_pos[2]:.5f}]")
            print(f"  RGB camera prim path: {self.rgb_camera_prim_path}")
            print(f"  Resolution: {resolution}")
            print(f"  Focal length: {focal_length_mm:.3f} mm")
            print(f"  Horizontal aperture: {h_aperture_mm:.3f} mm")
            print(f"  Vertical aperture: {v_aperture_mm:.3f} mm")
            print(f"  Computed intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        else:
            print(f"[FEMTO ERROR] Cannot compute intrinsics - RGB camera prim not valid!")

        # Create Isaac Sim Camera wrapper for coordinate transformations
        try:
            self.camera_wrapper = Camera(prim_path=self.rgb_camera_prim_path)
            print(f"[FEMTO] Created Camera wrapper for transformations")
        except Exception as e:
            print(f"[FEMTO WARNING] Could not create Camera wrapper: {e}")
            self.camera_wrapper = None

        print(f"[FEMTO] Camera setup complete at {self.prim_path}")
        print(f"[FEMTO] RGB and Depth annotators attached (distance_to_camera - Euclidean)")
        
    def _clear_xform_ops(self, prim):
        """Clear existing transform operations from prim"""
        xformable = UsdGeom.Xformable(prim)
        existing_ops = xformable.GetOrderedXformOps()
        
        if len(existing_ops) > 0:
            xformable.ClearXformOpOrder()
            for op in existing_ops:
                prim.RemoveProperty(op.GetName())
        
        # Explicitly remove Orient property if it exists
        if prim.HasProperty("xformOp:orient"):
            prim.RemoveProperty("xformOp:orient")
    
    def get_rgb_data(self):
        """
        Get RGB image from camera
        
        Returns:
            np.array: RGB image (H, W, 3) or None if unavailable
        """
        if self.rgb_annotator is None:
            return None
        
        try:
            rgb_data = self.rgb_annotator.get_data()
            if rgb_data is None or len(rgb_data) == 0:
                return None
            
            # Convert RGBA to RGB if needed
            if len(rgb_data.shape) == 3 and rgb_data.shape[2] == 4:
                rgb_data = rgb_data[:, :, :3]
            
            return rgb_data
        
        except Exception as e:
            print(f"[FEMTO RGB ERROR] {e}")
            return None
    
    def get_depth_data(self):
        """
        Get depth data from camera
        
        Returns:
            np.array: Depth image (H, W) in meters or None if unavailable
        """
        if self.depth_annotator is None:
            return None
        
        try:
            depth_data = self.depth_annotator.get_data()
            if depth_data is None or len(depth_data) == 0:
                return None
            
            return depth_data
        
        except Exception as e:
            print(f"[FEMTO DEPTH ERROR] {e}")
            return None
    
    def get_camera_params(self):
        """Get camera intrinsic parameters"""
        return self.camera_params.copy()

    def transform_to_world(self, points_2d, depths):
        """
        Transform 2D image coordinates + depth to 3D world coordinates
        using Isaac Sim's built-in camera transformation.

        Args:
            points_2d: numpy array of shape (N, 2) with (u, v) pixel coordinates
            depths: numpy array of shape (N,) with depth values in meters

        Returns:
            numpy array of shape (N, 3) with (x, y, z) world coordinates
        """
        if self.camera_wrapper is None:
            raise RuntimeError("Camera wrapper not initialized. Cannot transform points.")

        # Use Isaac Sim's built-in transformation
        world_points = self.camera_wrapper.get_world_points_from_image_coords(
            points_2d, depths
        )

        return world_points

