# -*- coding: utf-8 -*-
# Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2023 PickNik, LLC. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This Isaac Sim example is derived from
# https://github.com/ros-planning/moveit2_tutorials/blob/efef1d3/doc/how_to_guides/isaac_panda/launch/isaac_moveit.py
# which in turn was derived from an example provided with Isaac Sim 2022.2.1, found at
# standalone_examples/api/omni.isaac.ros2_bridge/moveit.py
#
# flake8: noqa

import sys
import re
import os
import argparse

import carb
import numpy as np

# In older versions of Isaac Sim (prior to 4.0), SimulationApp is imported from
# omni.isaac.kit rather than isaacsim.
try:
    from isaacsim import SimulationApp
except:
    from omni.isaac.kit import SimulationApp

# Parse arguments before creating SimulationApp
parser = argparse.ArgumentParser()
parser.add_argument("--ros_domain_id", type=int, default=0, help="ROS_DOMAIN_ID")
args, unknown = parser.parse_known_args()

FRANKA_STAGE_PATH = "/Franka"
FRANKA_USD_PATH = "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
# Camera will be created programmatically, so path is simpler
CAMERA_PRIM_PATH = f"{FRANKA_STAGE_PATH}/panda_hand/realsense_camera"
BACKGROUND_STAGE_PATH = "/background"
BACKGROUND_USD_PATH = "/Isaac/Environments/Simple_Room/simple_room.usd"
GRAPH_PATH = "/ActionGraph"
REALSENSE_VIEWPORT_NAME = "realsense_viewport"

CONFIG = {"renderer": "RayTracedLighting", "headless": False}

simulation_app = SimulationApp(CONFIG)

from omni.isaac.version import get_version

# Check the major version number of Isaac Sim to see if it's four digits, corresponding
# to Isaac Sim 2023.1.1 or older.  The version numbering scheme changed with the
# Isaac Sim 4.0 release in 2024.
is_legacy_isaacsim = (len(get_version()[2]) == 4)

# More imports that need to compare after we create the app
from omni.isaac.core import SimulationContext  # noqa E402
from omni.isaac.core.utils.prims import set_targets
from omni.isaac.core.utils import (  # noqa E402
    extensions,
    nucleus,
    prims,
    rotations,
    stage,
    viewports,
)
from pxr import Gf, UsdGeom  # noqa E402
import omni.graph.core as og  # noqa E402
import omni

# enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")

# Enable Visual Scripting extension for Action Graph editing
carb.log_info("Enabling Visual Scripting extension...")
extensions.enable_extension("omni.kit.window.script_editor")
extensions.enable_extension("omni.graph.ui")
extensions.enable_extension("omni.graph.window.action")
extensions.enable_extension("omni.graph.window.generic")
carb.log_info("✓ Visual Scripting extensions enabled")

simulation_context = SimulationContext(stage_units_in_meters=1.0)

# Locate Isaac Sim assets folder to load environment and robot stages
assets_root_path = nucleus.get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

carb.log_info(f"Assets root path: {assets_root_path}")
carb.log_info(f"Loading Franka from: {assets_root_path + FRANKA_USD_PATH}")

# Preparing stage
viewports.set_camera_view(eye=np.array([1.2, 1.2, 0.8]), target=np.array([0, 0, 0.5]))

# Loading the simple_room environment
carb.log_info(f"Loading environment from: {assets_root_path + BACKGROUND_USD_PATH}")
stage.add_reference_to_stage(
    assets_root_path + BACKGROUND_USD_PATH, BACKGROUND_STAGE_PATH
)

# Get current stage
current_stage = stage.get_current_stage()

# Loading the franka robot from USD
carb.log_info("=" * 80)
carb.log_info(f"Creating Franka prim at: {FRANKA_STAGE_PATH}")
carb.log_info(f"Loading from USD: {assets_root_path + FRANKA_USD_PATH}")
franka_prim = prims.create_prim(
    FRANKA_STAGE_PATH,
    "Xform",
    position=np.array([0, -0.64, 0]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90)),
    usd_path=assets_root_path + FRANKA_USD_PATH,
)

franka_prim_check = current_stage.GetPrimAtPath(FRANKA_STAGE_PATH)
if franka_prim_check.IsValid():
    carb.log_info(f"✓ Franka prim loaded successfully at {FRANKA_STAGE_PATH}")
    carb.log_info(f"  Prim type: {franka_prim_check.GetTypeName()}")
    carb.log_info(f"  Has children: {len(franka_prim_check.GetChildren())}")
else:
    carb.log_error(f"✗ Failed to load Franka prim at {FRANKA_STAGE_PATH}")
    carb.log_error(f"  USD path attempted: {assets_root_path + FRANKA_USD_PATH}")
    carb.log_error("  This will cause articulation and camera errors later!")
    simulation_app.close()
    sys.exit()

carb.log_info("=" * 80)

# Create RealSense D435 camera programmatically BEFORE simulation starts
carb.log_info("Creating RealSense D435 camera programmatically...")

# Check if camera already exists (from USD file)
camera_prim_usd = current_stage.GetPrimAtPath(CAMERA_PRIM_PATH)

if not camera_prim_usd.IsValid():
    carb.log_info(f"Camera not found in USD, creating at: {CAMERA_PRIM_PATH}")

    # First, verify that panda_hand exists
    panda_hand_prim = current_stage.GetPrimAtPath(f"{FRANKA_STAGE_PATH}/panda_hand")
    if not panda_hand_prim.IsValid():
        carb.log_error(f"✗ panda_hand not found at {FRANKA_STAGE_PATH}/panda_hand")
        carb.log_error("  Cannot attach camera to panda_hand!")
        simulation_app.close()
        sys.exit()

    # Create camera prim as a child of panda_hand
    # This ensures it moves with the hand during simulation
    from pxr import UsdGeom, Gf

    # Define the camera prim
    camera_prim_usd = UsdGeom.Camera.Define(current_stage, CAMERA_PRIM_PATH)
    camera_prim = camera_prim_usd.GetPrim()

    # Set LOCAL transform (relative to panda_hand parent)
    # Position: 40mm offset in X and Z from panda_hand origin
    # Orientation: 90° rotation around Y axis to point camera forward
    xform = UsdGeom.Xformable(camera_prim)

    # Clear any existing transform ops
    xform.ClearXformOpOrder()

    # Add standard transform operations: translate, orient (rotation as quaternion), scale
    # This matches the standard USD transform stack shown in Isaac Sim UI
    translate_op = xform.AddTranslateOp()
    orient_op = xform.AddOrientOp()
    scale_op = xform.AddScaleOp()

    # Set the local transform values
    # These values were manually verified in Isaac Sim UI during simulation
    translate_op.Set(Gf.Vec3d(-0.2, -0.2, -0.2))  # Camera offset from panda_hand

    # Convert Euler angles (-180, 0, 90) to quaternion for Orient
    # Note: Orient expects GfQuatf (float), not GfQuatd (double)
    # Using omni.isaac.core.utils.rotations for Euler to quaternion conversion
    euler_angles = np.array([-180.0, 0.0, 90.0])  # degrees: (X, Y, Z)
    quat_np = rotations.euler_angles_to_quat(euler_angles, degrees=True)  # Returns [w, x, y, z]
    # Convert numpy array to USD Quatf
    quat = Gf.Quatf(float(quat_np[0]), float(quat_np[1]), float(quat_np[2]), float(quat_np[3]))
    orient_op.Set(quat)

    scale_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))  # Uniform scale

    carb.log_info(f"✓ Camera prim created at {CAMERA_PRIM_PATH}")
    carb.log_info(f"    Local Position: [0.04, 0.0, 0.04] (relative to panda_hand)")
    carb.log_info(f"    Local Orientation: 90° rotation around Y axis (pointing forward)")
    carb.log_info(f"    Camera will move with panda_hand during simulation")
else:
    carb.log_info(f"✓ Camera already exists at {CAMERA_PRIM_PATH} (from USD file)")
    # Get the camera object from the existing prim
    camera_prim_usd = UsdGeom.Camera(camera_prim_usd)

# Configure RealSense D435 camera parameters
# These values match the original franka_alt_fingers.usd camera settings
# Ensure we have a Camera object (either newly created or from existing prim)
if not isinstance(camera_prim_usd, UsdGeom.Camera):
    camera_prim_usd = UsdGeom.Camera(current_stage.GetPrimAtPath(CAMERA_PRIM_PATH))

if camera_prim_usd.GetPrim().IsValid():
    camera = camera_prim_usd

    # Camera parameters - manually verified in Isaac Sim UI
    horizontal_aperture = 20.955  # tenths of stage units
    vertical_aperture = 15.7      # tenths of stage units
    focal_length = 12.0           # tenths of stage units (manually verified)
    focus_distance = 400          # tenths of stage units (40 meters)

    # Set camera properties
    camera.GetHorizontalApertureAttr().Set(horizontal_aperture)
    camera.GetVerticalApertureAttr().Set(vertical_aperture)
    camera.GetFocalLengthAttr().Set(focal_length)
    camera.GetFocusDistanceAttr().Set(focus_distance)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000000.0))  # 1cm to very far

    carb.log_info("✓ Camera configured with parameters matching franka_alt_fingers.usd:")
    carb.log_info(f"    Horizontal aperture: {horizontal_aperture}")
    carb.log_info(f"    Vertical aperture: {vertical_aperture}")
    carb.log_info(f"    Focal length: {focal_length}")
    carb.log_info(f"    Focus distance: {focus_distance}")
else:
    carb.log_error(f"✗ Failed to create or access camera at {CAMERA_PRIM_PATH}")

carb.log_info("=" * 80)

# add some objects, spread evenly along the X axis
# with a fixed offset from the robot in the Y and Z
prims.create_prim(
    "/cracker_box",
    "Xform",
    position=np.array([-0.2, -0.25, 0.15]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
    usd_path=assets_root_path
    + "/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
)
prims.create_prim(
    "/sugar_box",
    "Xform",
    position=np.array([-0.07, -0.25, 0.1]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 1, 0), -90)),
    usd_path=assets_root_path
    + "/Isaac/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
)
prims.create_prim(
    "/soup_can",
    "Xform",
    position=np.array([0.1, -0.25, 0.10]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
    usd_path=assets_root_path
    + "/Isaac/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
)
prims.create_prim(
    "/mustard_bottle",
    "Xform",
    position=np.array([0.0, 0.15, 0.12]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
    usd_path=assets_root_path
    + "/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
)

simulation_app.update()

# Use ROS_DOMAIN_ID from command line argument or environment variable
if args.ros_domain_id != 0:
    ros_domain_id = args.ros_domain_id
    carb.log_info(f"Using ROS_DOMAIN_ID from command line: {ros_domain_id}")
else:
    try:
        ros_domain_id = int(os.environ["ROS_DOMAIN_ID"])
        carb.log_info(f"Using ROS_DOMAIN_ID from environment: {ros_domain_id}")
    except ValueError:
        carb.log_warn("Invalid ROS_DOMAIN_ID integer value. Setting value to 0")
        ros_domain_id = 0
    except KeyError:
        carb.log_info("ROS_DOMAIN_ID environment variable is not set. Setting value to 0")
        ros_domain_id = 0

# Create an action graph with ROS component nodes
try:
    # nvblox-compatible topic names (camera_0/*)
    # This matches the topic naming convention expected by nvblox
    og_keys_set_values = [
        ("Context.inputs:domain_id", ros_domain_id),
        # Set the /Franka target prim to Articulation Controller node
        ("ArticulationController.inputs:robotPath", FRANKA_STAGE_PATH),
        ("PublishJointState.inputs:topicName", "isaac_joint_states"),
        ("SubscribeJointState.inputs:topicName", "isaac_joint_commands"),
        # TF publishing configuration
        ("PublishTF.inputs:topicName", "tf"),
        ("PublishTF.inputs:targetPrims", [FRANKA_STAGE_PATH]),
        ("createViewport.inputs:name", REALSENSE_VIEWPORT_NAME),
        ("createViewport.inputs:viewportId", 1),
        # nvblox expects: camera_0/color/image, camera_0/color/camera_info,
        #                 camera_0/depth/image, camera_0/depth/camera_info
        # Note: ROS2CameraHelper automatically publishes camera_info alongside rgb/depth
        # So we only need 2 helpers: rgb and depth
        ("cameraHelperRgb.inputs:frameId", "camera_0_optical_frame"),
        ("cameraHelperRgb.inputs:topicName", "camera_0/color/image"),
        ("cameraHelperRgb.inputs:type", "rgb"),
        ("cameraHelperDepth.inputs:frameId", "camera_0_optical_frame"),
        ("cameraHelperDepth.inputs:topicName", "camera_0/depth/image"),
        ("cameraHelperDepth.inputs:type", "depth"),
    ]

    carb.log_info("=" * 80)
    carb.log_info("ROS 2 Topics (nvblox-compatible):")
    carb.log_info("  - camera_0/color/image")
    carb.log_info("  - camera_0/color/camera_info")
    carb.log_info("  - camera_0/depth/image")
    carb.log_info("  - camera_0/depth/camera_info")
    carb.log_info("  - isaac_joint_states")
    carb.log_info("  - isaac_joint_commands")
    carb.log_info("  - tf (TF transforms)")
    carb.log_info("  - clock")
    carb.log_info("=" * 80)

    # In older versions of Isaac Sim, the articulation controller node contained a
    # "usePath" checkbox input that should be enabled.
    if is_legacy_isaacsim:
        og_keys_set_values.insert(1, ("ArticulationController.inputs:usePath", True))

    og.Controller.edit(
        {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                (
                    "SubscribeJointState",
                    "isaacsim.ros2.bridge.ROS2SubscribeJointState",
                ),
                (
                    "ArticulationController",
                    "isaacsim.core.nodes.IsaacArticulationController",
                ),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                ("OnTick", "omni.graph.action.OnTick"),
                ("createViewport", "isaacsim.core.nodes.IsaacCreateViewport"),
                (
                    "getRenderProduct",
                    "isaacsim.core.nodes.IsaacGetViewportRenderProduct",
                ),
                ("setCamera", "isaacsim.core.nodes.IsaacSetCameraOnRenderProduct"),
                ("cameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("cameraHelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnImpulseEvent.outputs:execOut", "PublishJointState.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "SubscribeJointState.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "PublishTF.inputs:execIn"),
                (
                    "OnImpulseEvent.outputs:execOut",
                    "ArticulationController.inputs:execIn",
                ),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                ("Context.outputs:context", "PublishTF.inputs:context"),
                (
                    "ReadSimTime.outputs:simulationTime",
                    "PublishJointState.inputs:timeStamp",
                ),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
                (
                    "SubscribeJointState.outputs:jointNames",
                    "ArticulationController.inputs:jointNames",
                ),
                (
                    "SubscribeJointState.outputs:positionCommand",
                    "ArticulationController.inputs:positionCommand",
                ),
                (
                    "SubscribeJointState.outputs:velocityCommand",
                    "ArticulationController.inputs:velocityCommand",
                ),
                (
                    "SubscribeJointState.outputs:effortCommand",
                    "ArticulationController.inputs:effortCommand",
                ),
                ("OnTick.outputs:tick", "createViewport.inputs:execIn"),
                ("createViewport.outputs:execOut", "getRenderProduct.inputs:execIn"),
                ("createViewport.outputs:viewport", "getRenderProduct.inputs:viewport"),
                ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
                (
                    "getRenderProduct.outputs:renderProductPath",
                    "setCamera.inputs:renderProductPath",
                ),
                ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                ("setCamera.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
                ("Context.outputs:context", "cameraHelperRgb.inputs:context"),
                ("Context.outputs:context", "cameraHelperDepth.inputs:context"),
                (
                    "getRenderProduct.outputs:renderProductPath",
                    "cameraHelperRgb.inputs:renderProductPath",
                ),
                (
                    "getRenderProduct.outputs:renderProductPath",
                    "cameraHelperDepth.inputs:renderProductPath",
                ),
            ],
            og.Controller.Keys.SET_VALUES: og_keys_set_values,
        },
    )
except Exception as e:
    print(e)

simulation_app.update()

# Setting the /Franka target prim to Publish JointState node
set_targets(
    prim=stage.get_current_stage().GetPrimAtPath("/ActionGraph/PublishJointState"),
    attribute="inputs:targetPrim",
    target_prim_paths=[FRANKA_STAGE_PATH],
)

# Set camera target for the Action Graph
carb.log_info("=" * 80)
carb.log_info("Setting camera target for Action Graph...")
set_targets(
    prim=stage.get_current_stage().GetPrimAtPath(GRAPH_PATH + "/setCamera"),
    attribute="inputs:cameraPrim",
    target_prim_paths=[CAMERA_PRIM_PATH],
)

simulation_app.update()

# need to initialize physics getting any articulation..etc
simulation_context.initialize_physics()

simulation_context.play()

# Add camera_info publishers using Replicator
import omni.replicator.core as rep
from isaacsim.ros2.bridge import read_camera_info

carb.log_info("=" * 80)
carb.log_info("Setting up camera_info publishers...")

# Get the render product path from the OmniGraph
render_product_path = None
try:
    render_product_attr = og.Controller.attribute(GRAPH_PATH + "/getRenderProduct.outputs:renderProductPath")
    render_product_path = og.Controller.get(render_product_attr)
    carb.log_info(f"Render product path: {render_product_path}")
except Exception as e:
    carb.log_error(f"Failed to get render product path: {e}")

if render_product_path:
    try:
        # Read camera info from the render product
        camera_info, _ = read_camera_info(render_product_path=render_product_path)

        # Create camera_info publisher for color camera
        color_info_writer = rep.writers.get("ROS2PublishCameraInfo")
        color_info_writer.initialize(
            frameId="camera_0_optical_frame",
            nodeNamespace="",
            queueSize=1,
            topicName="camera_0/color/camera_info",
            width=camera_info.width,
            height=camera_info.height,
            projectionType=camera_info.distortion_model,
            k=camera_info.k.reshape([1, 9]),
            r=camera_info.r.reshape([1, 9]),
            p=camera_info.p.reshape([1, 12]),
            physicalDistortionModel=camera_info.distortion_model,
            physicalDistortionCoefficients=camera_info.d,
        )
        color_info_writer.attach([render_product_path])
        carb.log_info("✓ Created camera_0/color/camera_info publisher")

        # Create camera_info publisher for depth camera (same intrinsics)
        depth_info_writer = rep.writers.get("ROS2PublishCameraInfo")
        depth_info_writer.initialize(
            frameId="camera_0_optical_frame",
            nodeNamespace="",
            queueSize=1,
            topicName="camera_0/depth/camera_info",
            width=camera_info.width,
            height=camera_info.height,
            projectionType=camera_info.distortion_model,
            k=camera_info.k.reshape([1, 9]),
            r=camera_info.r.reshape([1, 9]),
            p=camera_info.p.reshape([1, 12]),
            physicalDistortionModel=camera_info.distortion_model,
            physicalDistortionCoefficients=camera_info.d,
        )
        depth_info_writer.attach([render_product_path])
        carb.log_info("✓ Created camera_0/depth/camera_info publisher")

    except Exception as e:
        carb.log_error(f"Failed to create camera_info publishers: {e}")
else:
    carb.log_warn("Could not create camera_info publishers - render product path not found")

carb.log_info("=" * 80)

# Dock the second camera window and set it to use the RealSense camera
# Wait for the viewport windows to be created (may take a few frames)
import time
max_retries = 10
retry_count = 0
viewport = None
rs_viewport = None

carb.log_info("Waiting for viewport windows to be created...")
while retry_count < max_retries:
    viewport = omni.ui.Workspace.get_window("Viewport")
    rs_viewport = omni.ui.Workspace.get_window(REALSENSE_VIEWPORT_NAME)

    if rs_viewport is not None and viewport is not None:
        carb.log_info(f"Found both viewports after {retry_count} retries")
        break

    carb.log_info(f"Retry {retry_count + 1}/{max_retries}: Viewport={viewport is not None}, RealSense={rs_viewport is not None}")
    simulation_app.update()
    time.sleep(0.1)
    retry_count += 1

if rs_viewport is not None and viewport is not None:
    # Dock the RealSense viewport to the right of the main viewport
    rs_viewport.dock_in(viewport, omni.ui.DockPosition.RIGHT)
    carb.log_info(f"✓ Docked {REALSENSE_VIEWPORT_NAME} viewport to the right of main Viewport")

    # Set the viewport to use the RealSense camera
    viewport_api = omni.kit.viewport.utility.get_viewport_from_window_name(REALSENSE_VIEWPORT_NAME)
    if viewport_api:
        viewport_api.set_active_camera(CAMERA_PRIM_PATH)
        carb.log_info(f"✓ Set {REALSENSE_VIEWPORT_NAME} to use camera: {CAMERA_PRIM_PATH}")
    else:
        carb.log_warn(f"Could not get viewport API for {REALSENSE_VIEWPORT_NAME}")
else:
    carb.log_warn(f"Could not dock viewport after {max_retries} retries.")
    carb.log_warn(f"  Viewport found: {viewport is not None}")
    carb.log_warn(f"  RealSense viewport found: {rs_viewport is not None}")


while simulation_app.is_running():

    # Run with a fixed step size
    simulation_context.step(render=True)

    # Tick the Publish/Subscribe JointState, Publish TF and Publish Clock nodes each frame
    og.Controller.set(
        og.Controller.attribute("/ActionGraph/OnImpulseEvent.state:enableImpulse"), True
    )

simulation_context.stop()
simulation_app.close()
