import sys
import re
import os
import argparse
import carb
import numpy as np

try:
    from isaacsim import SimulationApp
except:
    from omni.isaac.kit import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--ros_domain_id", type=int, default=0, help="ROS_DOMAIN_ID")
args, unknown = parser.parse_known_args()

FRANKA_STAGE_PATH = "/Franka"
FRANKA_USD_PATH = "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
CAMERA_PRIM_PATH = f"{FRANKA_STAGE_PATH}/panda_hand/realsense_camera"
BACKGROUND_STAGE_PATH = "/background"
BACKGROUND_USD_PATH = "/Isaac/Environments/Simple_Room/simple_room.usd"
GRAPH_PATH = "/ActionGraph"
REALSENSE_VIEWPORT_NAME = "realsense_viewport"

CONFIG = {"renderer": "RayTracedLighting", "headless": False}

simulation_app = SimulationApp(CONFIG)

from omni.isaac.version import get_version

is_legacy_isaacsim = (len(get_version()[2]) == 4)

from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.prims import set_targets
from omni.isaac.core.utils import extensions, nucleus, prims, rotations, stage, viewports
from pxr import Gf, UsdGeom
import omni.graph.core as og
import omni

extensions.enable_extension("omni.isaac.ros2_bridge")
extensions.enable_extension("omni.kit.window.script_editor")
extensions.enable_extension("omni.graph.ui")
extensions.enable_extension("omni.graph.window.action")
extensions.enable_extension("omni.graph.window.generic")

simulation_context = SimulationContext(stage_units_in_meters=1.0)

assets_root_path = nucleus.get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

viewports.set_camera_view(eye=np.array([1.2, 1.2, 0.8]), target=np.array([0, 0, 0.5]))

stage.add_reference_to_stage(assets_root_path + BACKGROUND_USD_PATH, BACKGROUND_STAGE_PATH)

current_stage = stage.get_current_stage()
franka_prim = prims.create_prim(
    FRANKA_STAGE_PATH,
    "Xform",
    position=np.array([0, -0.64, 0]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90)),
    usd_path=assets_root_path + FRANKA_USD_PATH,
)

franka_prim_check = current_stage.GetPrimAtPath(FRANKA_STAGE_PATH)
if not franka_prim_check.IsValid():
    carb.log_error(f"Failed to load Franka prim at {FRANKA_STAGE_PATH}")
    simulation_app.close()
    sys.exit()

camera_prim_usd = current_stage.GetPrimAtPath(CAMERA_PRIM_PATH)

if not camera_prim_usd.IsValid():
    panda_hand_prim = current_stage.GetPrimAtPath(f"{FRANKA_STAGE_PATH}/panda_hand")
    if not panda_hand_prim.IsValid():
        carb.log_error(f"panda_hand not found at {FRANKA_STAGE_PATH}/panda_hand")
        simulation_app.close()
        sys.exit()

    from pxr import UsdGeom, Gf

    camera_prim_usd = UsdGeom.Camera.Define(current_stage, CAMERA_PRIM_PATH)
    camera_prim = camera_prim_usd.GetPrim()

    xform = UsdGeom.Xformable(camera_prim)
    xform.ClearXformOpOrder()

    translate_op = xform.AddTranslateOp()
    orient_op = xform.AddOrientOp()
    scale_op = xform.AddScaleOp()

    translate_op.Set(Gf.Vec3d(-0.2, -0.2, -0.2))

    euler_angles = np.array([-180.0, 0.0, 90.0])
    quat_np = rotations.euler_angles_to_quat(euler_angles, degrees=True)
    quat = Gf.Quatf(float(quat_np[0]), float(quat_np[1]), float(quat_np[2]), float(quat_np[3]))
    orient_op.Set(quat)

    scale_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))
else:
    camera_prim_usd = UsdGeom.Camera(camera_prim_usd)

if not isinstance(camera_prim_usd, UsdGeom.Camera):
    camera_prim_usd = UsdGeom.Camera(current_stage.GetPrimAtPath(CAMERA_PRIM_PATH))

if camera_prim_usd.GetPrim().IsValid():
    camera = camera_prim_usd

    horizontal_aperture = 20.955
    vertical_aperture = 15.7
    focal_length = 12.0
    focus_distance = 400

    camera.GetHorizontalApertureAttr().Set(horizontal_aperture)
    camera.GetVerticalApertureAttr().Set(vertical_aperture)
    camera.GetFocalLengthAttr().Set(focal_length)
    camera.GetFocusDistanceAttr().Set(focus_distance)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000000.0))
else:
    carb.log_error(f"Failed to create or access camera at {CAMERA_PRIM_PATH}")
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

if args.ros_domain_id != 0:
    ros_domain_id = args.ros_domain_id
else:
    try:
        ros_domain_id = int(os.environ["ROS_DOMAIN_ID"])
    except (ValueError, KeyError):
        ros_domain_id = 0

try:
    og_keys_set_values = [
        ("Context.inputs:domain_id", ros_domain_id),
        ("ArticulationController.inputs:robotPath", FRANKA_STAGE_PATH),
        ("PublishJointState.inputs:topicName", "isaac_joint_states"),
        ("SubscribeJointState.inputs:topicName", "isaac_joint_commands"),
        ("PublishTF.inputs:topicName", "tf"),
        ("PublishTF.inputs:targetPrims", [FRANKA_STAGE_PATH]),
        ("createViewport.inputs:name", REALSENSE_VIEWPORT_NAME),
        ("createViewport.inputs:viewportId", 1),
        ("cameraHelperRgb.inputs:frameId", "camera_0_optical_frame"),
        ("cameraHelperRgb.inputs:topicName", "camera_0/color/image"),
        ("cameraHelperRgb.inputs:type", "rgb"),
        ("cameraHelperDepth.inputs:frameId", "camera_0_optical_frame"),
        ("cameraHelperDepth.inputs:topicName", "camera_0/depth/image"),
        ("cameraHelperDepth.inputs:type", "depth"),
    ]

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

set_targets(
    prim=stage.get_current_stage().GetPrimAtPath("/ActionGraph/PublishJointState"),
    attribute="inputs:targetPrim",
    target_prim_paths=[FRANKA_STAGE_PATH],
)

set_targets(
    prim=stage.get_current_stage().GetPrimAtPath(GRAPH_PATH + "/setCamera"),
    attribute="inputs:cameraPrim",
    target_prim_paths=[CAMERA_PRIM_PATH],
)

simulation_app.update()

simulation_context.initialize_physics()
simulation_context.play()

import omni.replicator.core as rep
from isaacsim.ros2.bridge import read_camera_info

render_product_path = None
try:
    render_product_attr = og.Controller.attribute(GRAPH_PATH + "/getRenderProduct.outputs:renderProductPath")
    render_product_path = og.Controller.get(render_product_attr)
except Exception as e:
    carb.log_error(f"Failed to get render product path: {e}")

if render_product_path:
    try:
        camera_info, _ = read_camera_info(render_product_path=render_product_path)

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

    except Exception as e:
        carb.log_error(f"Failed to create camera_info publishers: {e}")

import time
max_retries = 10
retry_count = 0
viewport = None
rs_viewport = None

while retry_count < max_retries:
    viewport = omni.ui.Workspace.get_window("Viewport")
    rs_viewport = omni.ui.Workspace.get_window(REALSENSE_VIEWPORT_NAME)

    if rs_viewport is not None and viewport is not None:
        break

    simulation_app.update()
    time.sleep(0.1)
    retry_count += 1

if rs_viewport is not None and viewport is not None:
    rs_viewport.dock_in(viewport, omni.ui.DockPosition.RIGHT)

    viewport_api = omni.kit.viewport.utility.get_viewport_from_window_name(REALSENSE_VIEWPORT_NAME)
    if viewport_api:
        viewport_api.set_active_camera(CAMERA_PRIM_PATH)

while simulation_app.is_running():
    simulation_context.step(render=True)
    og.Controller.set(
        og.Controller.attribute("/ActionGraph/OnImpulseEvent.state:enableImpulse"), True
    )

simulation_context.stop()
simulation_app.close()
