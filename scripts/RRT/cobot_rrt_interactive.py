import asyncio
import time
import sys
import os
from pathlib import Path
import omni.ui as ui
from omni.kit.async_engine import run_coroutine
import omni.timeline

script_dir = None

try:
    if '__file__' in globals():
        script_dir = Path(__file__).parent.resolve()
except:
    pass

if script_dir is None:
    try:
        cwd = Path(os.getcwd())
        for parent in [cwd] + list(cwd.parents):
            if parent.name == "multiagent":
                script_dir = parent / "scripts" / "RRT"
                break
    except:
        pass

if script_dir is None:
    script_dir = Path.cwd() / "scripts" / "RRT"

if script_dir and str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Import our modules
from scene_setup import SceneSetup
from object_manager import ObjectManager
from rrt_controller import RRTController


class CobotRRTInteractive:
    """Main class for interactive RRT pick and place with UI"""

    def __init__(self, base_layer_count=4, num_layers=4):
        """
        Initialize interactive RRT system

        Args:
            base_layer_count: Number of objects in bottom layer of pyramid
            num_layers: Total number of layers in pyramid
        """
        self.base_layer_count = base_layer_count
        self.num_layers = num_layers

        self.scene_setup = None
        self.object_manager = None
        self.rrt_controller = None

        self.world = None
        self.cobot = None
        self.gripper = None
        self.container = None

        # UI elements
        self.window = None
        self.add_object_btn = None
        self.pick_place_btn = None
        self.reset_btn = None
        self.status_label = None
        self.fps_label = None
        self.frame_time_label = None
        self.steps_label = None

        # State
        self.is_running = False
        self.is_scene_loaded = False
        self.timeline = omni.timeline.get_timeline_interface()

        # Performance metrics
        self.step_count = 0
        self.last_frame_time = time.time()
        self.fps = 0.0
        self.frame_time_ms = 0.0
        self.fps_update_interval = 0.5  # Update FPS every 0.5 seconds
        self.last_fps_update = time.time()
        self.frame_count = 0
        
    def build_ui(self):
        """Build the UI window with buttons"""
        self.window = ui.Window("Cobot RRT Pick & Place", width=450, height=400)

        with self.window.frame:
            with ui.VStack(spacing=10, height=0):
                ui.Label("Cobot RRT Pick and Place Controller",
                        alignment=ui.Alignment.CENTER,
                        style={"font_size": 18})

                ui.Spacer(height=10)

                # Status label
                self.status_label = ui.Label("Status: Ready to load scene",
                                            alignment=ui.Alignment.CENTER,
                                            style={"font_size": 14, "color": 0xFF00FF00})

                ui.Spacer(height=10)

                # Performance metrics frame
                with ui.Frame(style={"background_color": 0x33000000}):
                    with ui.VStack(spacing=5, height=0):
                        ui.Label("Performance Metrics",
                                alignment=ui.Alignment.CENTER,
                                style={"font_size": 12, "color": 0xFFFFFFFF})

                        with ui.HStack(spacing=10):
                            ui.Spacer(width=10)
                            self.fps_label = ui.Label("FPS: 0.0",
                                                     alignment=ui.Alignment.LEFT,
                                                     style={"font_size": 11, "color": 0xFF00FFFF})
                            ui.Spacer()
                            self.frame_time_label = ui.Label("Frame Time: 0.0 ms",
                                                            alignment=ui.Alignment.CENTER,
                                                            style={"font_size": 11, "color": 0xFF00FFFF})
                            ui.Spacer()
                            self.steps_label = ui.Label("Steps: 0",
                                                       alignment=ui.Alignment.RIGHT,
                                                       style={"font_size": 11, "color": 0xFF00FFFF})
                            ui.Spacer(width=10)

                ui.Spacer(height=10)
                
                # Load Scene button
                with ui.HStack(spacing=5):
                    ui.Spacer()
                    ui.Button("Load Scene", 
                             width=150, height=40,
                             clicked_fn=self._on_load_scene_clicked,
                             style={"Button": {"background_color": 0xFF4CAF50}})
                    ui.Spacer()
                
                ui.Spacer(height=10)
                
                # Add Object button
                with ui.HStack(spacing=5):
                    ui.Spacer()
                    self.add_object_btn = ui.Button("Add Object Stack", 
                                                    width=150, height=40,
                                                    clicked_fn=self._on_add_object_clicked,
                                                    enabled=False,
                                                    style={"Button": {"background_color": 0xFF2196F3}})
                    ui.Spacer()
                
                ui.Spacer(height=10)
                
                # Pick and Place button
                with ui.HStack(spacing=5):
                    ui.Spacer()
                    self.pick_place_btn = ui.Button("Pick and Place", 
                                                    width=150, height=40,
                                                    clicked_fn=self._on_pick_place_clicked,
                                                    enabled=False,
                                                    style={"Button": {"background_color": 0xFFFF9800}})
                    ui.Spacer()
                
                ui.Spacer(height=10)
                
                # Reset button
                with ui.HStack(spacing=5):
                    ui.Spacer()
                    self.reset_btn = ui.Button("Reset Scene", 
                                               width=150, height=40,
                                               clicked_fn=self._on_reset_clicked,
                                               enabled=False,
                                               style={"Button": {"background_color": 0xFFF44336}})
                    ui.Spacer()
                
                ui.Spacer(height=20)
                
                ui.Label("Instructions:", alignment=ui.Alignment.LEFT)
                ui.Label("1. Load Scene - Creates Cobot, Container, Ground",
                        alignment=ui.Alignment.LEFT, word_wrap=True)
                ui.Label(f"2. Add Object Stack - Adds pyramid ({self.num_layers} layers: {self.base_layer_count}, {self.base_layer_count-1}, ..., 1)",
                        alignment=ui.Alignment.LEFT, word_wrap=True)
                ui.Label("3. Pick and Place - Starts pick/place automation",
                        alignment=ui.Alignment.LEFT, word_wrap=True)
                ui.Label("4. Reset Scene - Clears all objects",
                        alignment=ui.Alignment.LEFT, word_wrap=True)
    
    def _update_status(self, message, color=0xFF00FF00):
        """Update status label"""
        if self.status_label:
            self.status_label.text = f"Status: {message}"
            # Note: Color update might require style update

    def _update_performance_metrics(self):
        """Update FPS, Frame Time, and Steps display"""
        current_time = time.time()

        # Calculate frame time
        self.frame_time_ms = (current_time - self.last_frame_time) * 1000.0
        self.last_frame_time = current_time

        # Increment frame counter
        self.frame_count += 1

        # Update FPS every interval
        if current_time - self.last_fps_update >= self.fps_update_interval:
            elapsed = current_time - self.last_fps_update
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_update = current_time

        # Update UI labels
        if self.fps_label:
            self.fps_label.text = f"FPS: {self.fps:.1f}"

        if self.frame_time_label:
            self.frame_time_label.text = f"Frame Time: {self.frame_time_ms:.2f} ms"

        if self.steps_label:
            self.steps_label.text = f"Steps: {self.step_count}"
    
    def _on_load_scene_clicked(self):
        """Handle Load Scene button click"""
        print("\n=== Load Scene Button Clicked ===")
        run_coroutine(self._load_scene_async())
    
    async def _load_scene_async(self):
        """Load the scene asynchronously"""
        try:
            self._update_status("Loading scene...")

            self.scene_setup = SceneSetup()

            self.world, self.cobot, self.gripper, self.container = self.scene_setup.setup_complete_scene()

            self.object_manager = ObjectManager(self.world, self.base_layer_count, self.num_layers)

            container_info = self.scene_setup.get_container_info()
            self.rrt_controller = RRTController(self.cobot, self.gripper, container_info)
            
            # Setup RRT
            self.rrt_controller.setup_rrt()
            
            # Mark scene as loaded
            self.is_scene_loaded = True

            # Enable buttons
            self.add_object_btn.enabled = True
            self.reset_btn.enabled = True

            # Reset performance metrics
            self.step_count = 0
            self.last_frame_time = time.time()
            self.last_fps_update = time.time()
            self.frame_count = 0

            self._update_status("Scene loaded successfully!")
            print("=== Scene loaded successfully ===\n")

        except Exception as e:
            self._update_status(f"Error loading scene: {e}", color=0xFFFF0000)
            print(f"Error loading scene: {e}")
    
    def _on_add_object_clicked(self):
        """Handle Add Object button click"""
        print("\n=== Add Object Button Clicked ===")
        
        if not self.is_scene_loaded:
            self._update_status("Please load scene first!", color=0xFFFF0000)
            return
        
        # Add pyramid stack
        success = self.object_manager.add_stack()

        if success:
            stack_info = self.object_manager.get_stack_info()
            self._update_status(f"Pyramid added! Total: {stack_info['total_cuboids']} cuboids")

            # Enable pick and place button if we have cuboids
            if stack_info['total_cuboids'] > 0:
                self.pick_place_btn.enabled = True
        else:
            self._update_status("Pyramid stack already added!", color=0xFFFFAA00)
    
    def _on_pick_place_clicked(self):
        """Handle Pick and Place button click"""
        print("\n=== Pick and Place Button Clicked ===")
        
        if not self.is_scene_loaded:
            self._update_status("Please load scene first!", color=0xFFFF0000)
            return
        
        if self.object_manager.get_unpicked_count() == 0:
            self._update_status("No cuboids to pick!", color=0xFFFFAA00)
            return
        
        # Start pick and place automation
        if not self.is_running:
            self.is_running = True
            self.pick_place_btn.text = "Stop Pick & Place"
            run_coroutine(self._run_pick_place_async())
        else:
            # Stop automation
            self.is_running = False
            self.pick_place_btn.text = "Pick and Place"
            self._update_status("Pick and place stopped")
    
    async def _run_pick_place_async(self):
        """Run pick and place automation asynchronously"""
        try:
            # Play simulation if not already playing
            if not self.timeline.is_playing():
                self.timeline.play()
                await omni.kit.app.get_app().next_update_async()

            while self.is_running:
                # Update performance metrics
                self._update_performance_metrics()

                # Get next unpicked cuboid
                cuboid_info = self.object_manager.get_next_unpicked_cuboid()

                if cuboid_info is None:
                    # All cuboids picked
                    self._update_status("All cuboids picked and placed!")
                    self.is_running = False
                    self.pick_place_btn.text = "Pick and Place"
                    break

                # Set target
                self.rrt_controller.set_target_cuboid(cuboid_info)
                self._update_status(f"Picking: {cuboid_info['name']}")

                # Execute pick and place
                while not self.rrt_controller.is_done() and self.is_running:
                    self.rrt_controller.step()
                    self.step_count += 1
                    self._update_performance_metrics()
                    await omni.kit.app.get_app().next_update_async()

                if self.rrt_controller.is_done():
                    # Mark as picked
                    self.object_manager.mark_cuboid_picked(cuboid_info)

                    unpicked = self.object_manager.get_unpicked_count()
                    self._update_status(f"Placed! Remaining: {unpicked}")

                    # Reset for next cuboid
                    self.rrt_controller.reset_for_next_cuboid()

                    # Small delay between picks
                    for _ in range(30):
                        self.step_count += 1
                        self._update_performance_metrics()
                        await omni.kit.app.get_app().next_update_async()

                if not self.is_running:
                    self._update_status("Pick and place paused")
                    break

        except Exception as e:
            self._update_status(f"Error: {e}", color=0xFFFF0000)
            print(f"Error in pick and place: {e}")
            self.is_running = False
            self.pick_place_btn.text = "Pick and Place"
    
    def _on_reset_clicked(self):
        """Handle Reset button click"""
        print("\n=== Reset Button Clicked ===")
        run_coroutine(self._reset_scene_async())
    
    async def _reset_scene_async(self):
        """Reset the scene asynchronously"""
        try:
            self._update_status("Resetting scene...")

            # Stop any running automation
            self.is_running = False

            # Stop simulation
            if self.timeline.is_playing():
                self.timeline.stop()
                await omni.kit.app.get_app().next_update_async()

            # Clear all cuboids
            if self.object_manager:
                self.object_manager.clear_all()

            # Reset RRT controller
            if self.rrt_controller:
                self.rrt_controller.reset_for_next_cuboid()
                self.rrt_controller.placed_count = 0

            # Reset performance metrics
            self.step_count = 0
            self.last_frame_time = time.time()
            self.last_fps_update = time.time()
            self.frame_count = 0
            self.fps = 0.0
            self.frame_time_ms = 0.0
            self._update_performance_metrics()

            # Reset buttons
            self.pick_place_btn.text = "Pick and Place"
            self.pick_place_btn.enabled = False

            self._update_status("Scene reset complete!")
            print("=== Scene reset complete ===\n")

        except Exception as e:
            self._update_status(f"Error resetting: {e}", color=0xFFFF0000)
            print(f"Error resetting scene: {e}")


# Main execution
def main(base_layer_count=4, num_layers=4):
    """
    Main entry point

    Args:
        base_layer_count: Number of cuboids in bottom layer (default: 4 for 2x2 grid)
        num_layers: Total number of layers (default: 4 for layers of 4,3,2,1)
    """
    print("\n" + "="*60)
    print("Cobot RRT Interactive Pick and Place")
    print(f"Pyramid Configuration: {num_layers} layers")
    print(f"Base layer: {base_layer_count} objects")
    total_objects = sum(range(base_layer_count - num_layers + 1, base_layer_count + 1))
    print(f"Total objects: {total_objects}")
    print("="*60 + "\n")

    app = CobotRRTInteractive(base_layer_count, num_layers)
    app.build_ui()

    print("UI loaded! Use the buttons to control the robot.")
    print("="*60 + "\n")


# Run the application
if __name__ == "__main__":
    # You can customize the pyramid configuration here:
    # main(base_layer_count=4, num_layers=4)  # Default: 4,3,2,1 pyramid
    # main(base_layer_count=5, num_layers=5)  # 5,4,3,2,1 pyramid
    # main(base_layer_count=3, num_layers=3)  # 3,2,1 pyramid
    main()

