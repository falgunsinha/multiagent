import carb
import numpy as np
import os

# Isaac Sim Core APIs
from isaacsim.core.api import World
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation, XFormPrim
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.prims import delete_prim, get_prim_at_path
from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats
import omni.timeline
from omni.kit.async_engine import run_coroutine
import omni.ui as ui

# Motion generation imports
from isaacsim.robot_motion.motion_generation import (
    LulaCSpaceTrajectoryGenerator,
    LulaTaskSpaceTrajectoryGenerator,
    LulaKinematicsSolver,
    ArticulationTrajectory
)

import lula


class CobotTrajectoryGeneration():
    def __init__(self):
        self._c_space_trajectory_generator = None
        self._taskspace_trajectory_generator = None
        self._kinematics_solver = None

        self._action_sequence = []
        self._action_sequence_index = 0

        self._articulation = None
        self._end_effector_name = "ee_link"

    def load_example_assets(self, world):
        """Load the UR10 robot to the stage"""
        robot_prim_path = "/ur10"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"

        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = SingleArticulation(robot_prim_path, name="ur10")

        # Add articulation to the world scene
        world.scene.add(self._articulation)

        return self._articulation

    def setup(self):
        """Setup trajectory generators and kinematics solver"""
        # Config files for supported robots are stored in the motion_generation extension
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

        # Initialize a LulaCSpaceTrajectoryGenerator object
        self._c_space_trajectory_generator = LulaCSpaceTrajectoryGenerator(
            robot_description_path=rmp_config_dir + "/universal_robots/ur10/rmpflow/ur10_robot_description.yaml",
            urdf_path=rmp_config_dir + "/universal_robots/ur10/ur10_robot.urdf"
        )

        self._taskspace_trajectory_generator = LulaTaskSpaceTrajectoryGenerator(
            robot_description_path=rmp_config_dir + "/universal_robots/ur10/rmpflow/ur10_robot_description.yaml",
            urdf_path=rmp_config_dir + "/universal_robots/ur10/ur10_robot.urdf"
        )

        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path=rmp_config_dir + "/universal_robots/ur10/rmpflow/ur10_robot_description.yaml",
            urdf_path=rmp_config_dir + "/universal_robots/ur10/ur10_robot.urdf"
        )

    def setup_cspace_trajectory(self):
        """Setup C-space trajectory with multiple waypoints"""
        # Ensure articulation is initialized
        if not self._articulation.handles_initialized:
            carb.log_warn("Articulation not initialized. Please ensure world.reset() has been called.")
            self._action_sequence = []
            return

        c_space_points = np.array([
            [-0.41, 0.5, -2.36, -1.28, 5.13, -4.71],
            [-1.43, 1.0, -2.58, -1.53, 6.0, -4.74],
            [-2.83, 0.34, -2.11, -1.38, 1.26, -4.71],
            [-0.41, 0.5, -2.36, -1.28, 5.13, -4.71]
        ])

        timestamps = np.array([0, 5, 10, 13])

        trajectory_time_optimal = self._c_space_trajectory_generator.compute_c_space_trajectory(c_space_points)
        trajectory_timestamped = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(c_space_points, timestamps)

        # Visualize c-space targets in task space
        for i, point in enumerate(c_space_points):
            position, rotation = self._kinematics_solver.compute_forward_kinematics(self._end_effector_name, point)
            add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", f"/visualized_frames/target_{i}")
            frame = XFormPrim(f"/visualized_frames/target_{i}", scales=np.array([[.04, .04, .04]]))
            frame.set_world_poses(np.array([position]), np.array([rot_matrices_to_quats(rotation)]))

        if trajectory_time_optimal is None or trajectory_timestamped is None:
            carb.log_warn("No trajectory could be computed")
            self._action_sequence = []
        else:
            physics_dt = 1 / 60
            self._action_sequence = []

            # Follow both trajectories in a row
            articulation_trajectory_time_optimal = ArticulationTrajectory(self._articulation, trajectory_time_optimal, physics_dt)
            self._action_sequence.extend(articulation_trajectory_time_optimal.get_action_sequence())

            articulation_trajectory_timestamped = ArticulationTrajectory(self._articulation, trajectory_timestamped, physics_dt)
            self._action_sequence.extend(articulation_trajectory_timestamped.get_action_sequence())

    def setup_taskspace_trajectory(self):
        """Setup Task-space trajectory with position and orientation targets"""
        # Ensure articulation is initialized
        if not self._articulation.handles_initialized:
            carb.log_warn("Articulation not initialized. Please ensure world.reset() has been called.")
            self._action_sequence = []
            return

        task_space_position_targets = np.array([
            [0.3, -0.3, 0.1],
            [0.3, 0.3, 0.1],
            [0.3, 0.3, 0.5],
            [0.3, -0.3, 0.5],
            [0.3, -0.3, 0.1]
        ])

        task_space_orientation_targets = np.tile(np.array([0, 1, 0, 0]), (5, 1))

        trajectory = self._taskspace_trajectory_generator.compute_task_space_trajectory_from_points(
            task_space_position_targets, task_space_orientation_targets, self._end_effector_name
        )

        # Visualize task-space targets in task space
        for i, (position, orientation) in enumerate(zip(task_space_position_targets, task_space_orientation_targets)):
            add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", f"/visualized_frames/target_{i}")
            frame = XFormPrim(f"/visualized_frames/target_{i}", scales=np.array([[.04, .04, .04]]))
            frame.set_world_poses(np.array([position]), np.array([orientation]))

        if trajectory is None:
            carb.log_warn("No trajectory could be computed")
            self._action_sequence = []
        else:
            physics_dt = 1 / 60
            articulation_trajectory = ArticulationTrajectory(self._articulation, trajectory, physics_dt)

            # Get a sequence of ArticulationActions that are intended to be passed to the robot at 1/60 second intervals
            self._action_sequence = articulation_trajectory.get_action_sequence()

    def setup_advanced_trajectory(self):
        """Setup a complex trajectory combining c-space and task-space movements"""
        # Ensure articulation is initialized
        if not self._articulation.handles_initialized:
            carb.log_warn("Articulation not initialized. Please ensure world.reset() has been called.")
            self._action_sequence = []
            return

        # The following code demonstrates how to specify a complicated cspace and taskspace path
        # using the lula.CompositePathSpec object

        initial_c_space_robot_pose = np.array([0, 0, 0, 0, 0, 0])

        # Combine a cspace and taskspace trajectory
        composite_path_spec = lula.create_composite_path_spec(initial_c_space_robot_pose)

        #############################################################################
        # Demonstrate all the available movements in a taskspace path spec:

        # Lula has its own classes for Rotations and 6 DOF poses: Rotation3 and Pose3
        r0 = lula.Rotation3(np.pi / 2, np.array([1.0, 0.0, 0.0]))
        t0 = np.array([.3, -.1, .3])
        task_space_spec = lula.create_task_space_path_spec(lula.Pose3(r0, t0))

        # Add path linearly interpolating between r0,r1 and t0,t1
        t1 = np.array([.3, -.1, .5])
        r1 = lula.Rotation3(np.pi / 3, np.array([1, 0, 0]))
        task_space_spec.add_linear_path(lula.Pose3(r1, t1))

        # Add pure translation. Constant rotation is assumed
        task_space_spec.add_translation(t0)

        # Add pure rotation.
        task_space_spec.add_rotation(r0)

        # Add three-point arc with constant orientation.
        t2 = np.array([.3, .3, .3, ])
        midpoint = np.array([.3, 0, .5])
        task_space_spec.add_three_point_arc(t2, midpoint, constant_orientation=True)

        # Add three-point arc with tangent orientation.
        task_space_spec.add_three_point_arc(t0, midpoint, constant_orientation=False)

        # Add three-point arc with orientation target.
        task_space_spec.add_three_point_arc_with_orientation_target(lula.Pose3(r1, t2), midpoint)

        # Add tangent arc with constant orientation. Tangent arcs are circles that connect two points
        task_space_spec.add_tangent_arc(t0, constant_orientation=True)

        # Add tangent arc with tangent orientation.
        task_space_spec.add_tangent_arc(t2, constant_orientation=False)

        # Add tangent arc with orientation target.
        task_space_spec.add_tangent_arc_with_orientation_target(lula.Pose3(r0, t0))

        ###################################################
        # Demonstrate the usage of a c_space path spec:
        c_space_spec = lula.create_c_space_path_spec(np.array([0, 0, 0, 0, 0, 0]))

        c_space_spec.add_c_space_waypoint(np.array([0, 0.5, -2.0, -1.28, 5.13, -4.71]))

        ##############################################################
        # Combine the two path specs together into a composite spec:

        # specify how to connect initial_c_space and task_space points with transition_mode option
        transition_mode = lula.CompositePathSpec.TransitionMode.FREE
        composite_path_spec.add_task_space_path_spec(task_space_spec, transition_mode)

        transition_mode = lula.CompositePathSpec.TransitionMode.FREE
        composite_path_spec.add_c_space_path_spec(c_space_spec, transition_mode)

        # Transition Modes:
        # lula.CompositePathSpec.TransitionMode.LINEAR_TASK_SPACE:
        #      Connect cspace to taskspace points linearly through task space. This mode is only available when adding a task_space path spec.
        # lula.CompositePathSpec.TransitionMode.FREE:
        #      Put no constraints on how cspace and taskspace points are connected
        # lula.CompositePathSpec.TransitionMode.SKIP:
        #      Skip the first point of the path spec being added, using the last pose instead

        trajectory = self._taskspace_trajectory_generator.compute_task_space_trajectory_from_path_spec(
            composite_path_spec, self._end_effector_name
        )

        if trajectory is None:
            carb.log_warn("No trajectory could be computed")
            self._action_sequence = []
        else:
            physics_dt = 1 / 60
            articulation_trajectory = ArticulationTrajectory(self._articulation, trajectory, physics_dt)

            # Get a sequence of ArticulationActions that are intended to be passed to the robot at 1/60 second intervals
            self._action_sequence = articulation_trajectory.get_action_sequence()

    def update(self):
        """Update function to be called each frame"""
        if len(self._action_sequence) == 0:
            return

        if self._action_sequence_index >= len(self._action_sequence):
            self._action_sequence_index += 1
            self._action_sequence_index %= len(self._action_sequence) + 10  # Wait 10 frames before repeating trajectories
            return

        if self._action_sequence_index == 0:
            self._teleport_robot_to_position(self._action_sequence[0])

        self._articulation.apply_action(self._action_sequence[self._action_sequence_index])

        self._action_sequence_index += 1
        self._action_sequence_index %= len(self._action_sequence) + 10  # Wait 10 frames before repeating trajectories

    def reset(self):
        """Reset the trajectory execution"""
        # Delete any visualized frames
        if get_prim_at_path("/visualized_frames"):
            delete_prim("/visualized_frames")

        self._action_sequence = []
        self._action_sequence_index = 0

    def _teleport_robot_to_position(self, articulation_action):
        """Teleport robot to the initial position of the trajectory"""
        initial_positions = np.zeros(self._articulation.num_dof)
        initial_positions[articulation_action.joint_indices] = articulation_action.joint_positions

        self._articulation.set_joint_positions(initial_positions)
        self._articulation.set_joint_velocities(np.zeros_like(initial_positions))


# Global variables for UI control
g_world = None
g_trajectory_example = None
g_window = None
g_running_trajectory = None  # Tracks which trajectory is running: 'cspace', 'taskspace', 'advanced', or None
g_update_subscription = None


async def load_scene():
    """Load the scene with UR10 robot"""
    global g_world, g_trajectory_example

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        raise RuntimeError("Could not find Isaac Sim assets folder")

    # Stop timeline first
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()

    # Wait for timeline to stop
    await omni.kit.app.get_app().next_update_async()

    # Clear any existing World instance to ensure clean state
    print("Creating fresh World instance...")
    World.clear_instance()

    # Wait for cleanup to complete
    await omni.kit.app.get_app().next_update_async()

    # Create new World instance with 60Hz physics
    g_world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)

    # Wait for World to initialize
    await omni.kit.app.get_app().next_update_async()

    print("Adding ground plane...")
    g_world.scene.add_default_ground_plane()

    # Create trajectory example instance
    g_trajectory_example = CobotTrajectoryGeneration()

    # Load UR10 robot
    print("Loading UR10 robot...")
    g_trajectory_example.load_example_assets(g_world)

    # Setup trajectory generators
    print("Setting up trajectory generators...")
    g_trajectory_example.setup()

    print("Initializing physics...")
    g_world.initialize_physics()

    for _ in range(2):
        await omni.kit.app.get_app().next_update_async()

    print("Resetting world...")
    g_world.reset()

    for _ in range(2):
        await omni.kit.app.get_app().next_update_async()

    print("Scene loaded successfully!")


def on_load_scene_clicked():
    """Callback for Load Scene button"""
    print("=" * 80)
    print("Loading Scene...")
    print("=" * 80)
    run_coroutine(load_scene())


def on_cspace_trajectory_clicked():
    """Callback for Run C-space Trajectory button"""
    global g_running_trajectory, g_update_subscription

    if g_world is None or g_trajectory_example is None:
        carb.log_warn("Please load the scene first!")
        return

    if g_running_trajectory == 'cspace':
        # Stop the trajectory
        print("Stopping C-space trajectory...")
        stop_trajectory()
    else:
        # Start C-space trajectory
        print("=" * 80)
        print("Starting C-space Trajectory...")
        print("=" * 80)

        # Stop any currently running trajectory
        if g_running_trajectory is not None:
            stop_trajectory()

        # Reset world to ensure articulation is initialized
        g_world.reset()

        # Reset and setup c-space trajectory
        g_trajectory_example.reset()
        g_trajectory_example.setup_cspace_trajectory()

        print(f"C-space trajectory has {len(g_trajectory_example._action_sequence)} actions")

        # Start timeline
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        # Start update loop
        g_running_trajectory = 'cspace'
        g_update_subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            lambda e: update_trajectory()
        )


def on_taskspace_trajectory_clicked():
    """Callback for Run Task-space Trajectory button"""
    global g_running_trajectory, g_update_subscription

    if g_world is None or g_trajectory_example is None:
        carb.log_warn("Please load the scene first!")
        return

    if g_running_trajectory == 'taskspace':
        # Stop the trajectory
        print("Stopping Task-space trajectory...")
        stop_trajectory()
    else:
        # Start Task-space trajectory
        print("=" * 80)
        print("Starting Task-space Trajectory...")
        print("=" * 80)

        # Stop any currently running trajectory
        if g_running_trajectory is not None:
            stop_trajectory()

        # Reset world to ensure articulation is initialized
        g_world.reset()

        # Reset and setup task-space trajectory
        g_trajectory_example.reset()
        g_trajectory_example.setup_taskspace_trajectory()

        print(f"Task-space trajectory has {len(g_trajectory_example._action_sequence)} actions")

        # Start timeline
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        # Start update loop
        g_running_trajectory = 'taskspace'
        g_update_subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            lambda e: update_trajectory()
        )


def on_advanced_trajectory_clicked():
    """Callback for Run Advanced Trajectory button"""
    global g_running_trajectory, g_update_subscription

    if g_world is None or g_trajectory_example is None:
        carb.log_warn("Please load the scene first!")
        return

    if g_running_trajectory == 'advanced':
        # Stop the trajectory
        print("Stopping Advanced trajectory...")
        stop_trajectory()
    else:
        # Start Advanced trajectory
        print("=" * 80)
        print("Starting Advanced Trajectory...")
        print("=" * 80)

        # Stop any currently running trajectory
        if g_running_trajectory is not None:
            stop_trajectory()

        # Reset world to ensure articulation is initialized
        g_world.reset()

        # Reset and setup advanced trajectory
        g_trajectory_example.reset()
        g_trajectory_example.setup_advanced_trajectory()

        print(f"Advanced trajectory has {len(g_trajectory_example._action_sequence)} actions")

        # Start timeline
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        # Start update loop
        g_running_trajectory = 'advanced'
        g_update_subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            lambda e: update_trajectory()
        )


def stop_trajectory():
    """Stop the currently running trajectory"""
    global g_running_trajectory, g_update_subscription

    if g_update_subscription is not None:
        g_update_subscription.unsubscribe()
        g_update_subscription = None

    g_running_trajectory = None

    # Pause timeline
    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()

    print("Trajectory stopped.")


def update_trajectory():
    """Update function called each frame when trajectory is running"""
    global g_world, g_trajectory_example

    if g_world is None or g_trajectory_example is None:
        return

    if g_world.is_playing():
        g_trajectory_example.update()


def on_reset_scene_clicked():
    """Callback for Reset Scene button"""
    global g_world, g_trajectory_example, g_running_trajectory, g_update_subscription

    print("=" * 80)
    print("Resetting Scene...")
    print("=" * 80)

    # Stop any running trajectory
    if g_running_trajectory is not None:
        stop_trajectory()

    # Stop timeline
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()

    # Clear World instance
    if g_world is not None:
        World.clear_instance()
        g_world = None

    g_trajectory_example = None
    g_running_trajectory = None

    print("Scene reset complete. Click 'Load Scene' to start again.")


def create_ui():
    """Create the UI window with buttons"""
    global g_window

    # Create window
    g_window = ui.Window("Cobot Trajectory Generator", width=350, height=400)

    with g_window.frame:
        with ui.VStack(spacing=10):
            ui.Label("Cobot Trajectory Generation Control", alignment=ui.Alignment.CENTER, height=30)

            ui.Spacer(height=5)

            # Load Scene button
            ui.Button("Load Scene", clicked_fn=on_load_scene_clicked, height=40)

            ui.Spacer(height=5)

            # C-space Trajectory button
            ui.Button("Run C-space Trajectory", clicked_fn=on_cspace_trajectory_clicked, height=40)

            ui.Spacer(height=5)

            # Task-space Trajectory button
            ui.Button("Run Task-space Trajectory", clicked_fn=on_taskspace_trajectory_clicked, height=40)

            ui.Spacer(height=5)

            # Advanced Trajectory button
            ui.Button("Run Advanced Trajectory", clicked_fn=on_advanced_trajectory_clicked, height=40)

            ui.Spacer(height=5)

            ui.Label("Click a trajectory button again to stop it", alignment=ui.Alignment.CENTER, height=20)

            ui.Spacer(height=10)

            # Reset Scene button
            ui.Button("Reset Scene", clicked_fn=on_reset_scene_clicked, height=40)


# Create and show the UI
print("=" * 80)
print("Cobot Trajectory Generation")
print("=" * 80)

create_ui()

