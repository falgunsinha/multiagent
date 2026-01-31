# Franka RRT Interactive Pick and Place

Interactive pick and place system using Lula RRT motion planning with UI buttons.

## 📁 File Structure

```
cobotproject/scripts/RRT/
├── franka_rrt_interactive.py    # Main UI and orchestration
├── scene_setup.py                # Scene creation (Franka, Container, Ground)
├── object_manager.py             # Dynamic cuboid management
├── rrt_controller.py             # RRT motion planning and pick/place logic
└── README.md                     # This file
```

## 🎯 Features

- **Modular Design**: Split into separate modules for better performance and maintainability
- **Interactive UI**: Four buttons to control the workflow
- **Performance Metrics**: Real-time FPS, Frame Time, and Steps display
- **No SimulationApp**: Executable directly from VS Code Script Editor
- **Lula RRT**: Fast motion planning without obstacle avoidance
- **Stack Management**: Automatically creates stacks of 4, 2, and 1 cuboids
- **Pause/Resume**: Can pause and resume pick and place operations

## 🚀 Usage

### 1. Run from Isaac Sim Script Editor

1. Open Isaac Sim
2. Open Script Editor (Window → Script Editor)
3. Load `franka_rrt_interactive.py`
4. Click "Run" or press Ctrl+Enter

### 2. Run from VS Code

1. Open `franka_rrt_interactive.py` in VS Code
2. Make sure Isaac Sim extensions are enabled in VS Code
3. Run the script

## 🎮 UI Controls

### **Performance Metrics Display**
- **FPS**: Frames per second (updated every 0.5 seconds)
- **Frame Time**: Time per frame in milliseconds
- **Steps**: Total simulation steps executed
- Real-time updates during pick and place operations
- Helps monitor performance and identify bottlenecks

### **Load Scene Button**
- Creates the base scene with:
  - Ground plane
  - Franka robot at default position
  - Warehouse container for placing objects
- Initializes RRT motion planner
- Resets performance metrics
- **Click once** at the start

### **Add Object Stack Button**
- Adds a **pyramid-style stack** of cuboids in layers:
  - **Bottom layer**: 4 cuboids in a 2x2 grid
  - **Layer 2**: 3 cuboids
  - **Layer 3**: 2 cuboids
  - **Top layer**: 1 cuboid
- Total: 10 cuboids (4+3+2+1)
- Cuboids rest on top of each other in pyramid formation
- **Click once** to add the entire pyramid
- Configuration is customizable (see Configuration section)

### **Pick and Place Button**
- Starts automated pick and place operation
- Picks cuboids one by one from stacks
- Places them in the container
- **Click again to pause** the operation
- **Click again to resume** from where it stopped
- Continues until all cuboids are placed

### **Reset Scene Button**
- Stops any running automation
- Clears all dynamic cuboids from the scene
- Resets the system to initial state
- **Does NOT** remove Franka, container, or ground plane
- After reset, you can add objects again

## 📊 Workflow

```
1. Load Scene
   ↓
2. Add Object Stack (click once to add pyramid)
   ↓
3. Pick and Place (starts automation)
   ↓
4. [Optional] Pause/Resume by clicking Pick and Place again
   ↓
5. Reset Scene (when done or to start over)
   ↓
6. Go back to step 2
```

## 🔧 Module Details

### **scene_setup.py**
- `SceneSetup` class manages static scene elements
- Creates World with optimized physics (60 Hz)
- Adds Franka robot with default joint configuration
- Adds warehouse container with physics
- Initializes Franka to RRT-compatible pose

### **object_manager.py**
- `ObjectManager` class manages dynamic cuboids in pyramid formation
- Pyramid configuration:
  - **Configurable base layer** (default: 4 cuboids in 2x2 grid)
  - **Configurable number of layers** (default: 4 layers)
  - Each layer has one fewer cuboid than the layer below
  - Example: 4 layers → 4,3,2,1 cuboids = 10 total
- Tracks picked/unpicked status
- Handles cuboid creation and deletion
- Calculates optimal positions for each layer

### **rrt_controller.py**
- `RRTController` class manages motion planning
- Uses Lula RRT for path planning
- State machine: IDLE → MOVING_TO_PICK → PICKING → MOVING_TO_PLACE → PLACING → DONE
- Plans paths using only 7 arm DOF (excludes gripper joints)
- Stacks cuboids in container with vertical offset

### **franka_rrt_interactive.py**
- Main UI and orchestration
- Creates omni.ui window with buttons
- Handles async operations
- Manages timeline (play/stop)
- Coordinates between modules

## ⚙️ Configuration

### Pyramid Configuration
Edit in `franka_rrt_interactive.py` at the bottom:
```python
# Default: 4 layers with 4 cuboids at base (4,3,2,1 = 10 total)
main(base_layer_count=4, num_layers=4)

# Larger pyramid: 5 layers with 5 cuboids at base (5,4,3,2,1 = 15 total)
main(base_layer_count=5, num_layers=5)

# Smaller pyramid: 3 layers with 3 cuboids at base (3,2,1 = 6 total)
main(base_layer_count=3, num_layers=3)

# Custom: 6 layers with 6 cuboids at base (6,5,4,3,2,1 = 21 total)
main(base_layer_count=6, num_layers=6)
```

### Pyramid Base Position
Edit in `object_manager.py`:
```python
self.base_position = np.array([0.5, 0.0, 0.0])  # Center of pyramid
```

### Cube Size
Edit in `object_manager.py`:
```python
self.cube_size = 0.0515  # 5.15 cm cubes
```

### RRT Parameters
Edit in `rrt_controller.py`:
```python
self.rrt.set_max_iterations(10000)  # Max planning iterations
```

### Physics Settings
Edit in `scene_setup.py`:
```python
self.setup_world(physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
```

## 🐛 Troubleshooting

### "Please load scene first!"
- Click "Load Scene" button before other operations

### "No cuboids to pick!"
- Click "Add Object Stack" to add cuboids first

### "All stacks already added!"
- All 3 stacks have been added
- Click "Reset Scene" to start over

### RRT Planning Fails
- Check that Franka is in valid configuration
- Increase max_iterations in `rrt_controller.py`
- Check that target positions are reachable

### Simulation Not Playing
- The "Pick and Place" button automatically starts simulation
- Check timeline status in Isaac Sim

## 📝 Notes

- **Performance**: Modular design prevents slowdowns by separating concerns
- **7 DOF Planning**: RRT uses only arm joints, gripper controlled separately
- **Async Operations**: All long-running operations use async/await
- **State Persistence**: Can pause and resume without losing progress
- **Clean Reset**: Reset only clears dynamic objects, keeps static scene

## 🔄 Advantages of Modular Design

1. **Faster Loading**: Only loads necessary components when needed
2. **Better Debugging**: Each module can be tested independently
3. **Easier Maintenance**: Changes isolated to specific modules
4. **Reusability**: Modules can be used in other projects
5. **Clear Separation**: UI, scene, objects, and control logic separated

## 📚 Dependencies

- Isaac Sim 5.0.0+
- isaacsim.core.api
- isaacsim.robot_motion.motion_generation
- omni.ui
- omni.timeline
- Local modules: src.manipulators, src.grippers

## 🎓 Learning Resources

- [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/)
- [Lula Motion Planning](https://docs.isaacsim.omniverse.nvidia.com/latest/features/motion_generation/index.html)
- [omni.ui Documentation](https://docs.omniverse.nvidia.com/kit/docs/omni.ui/latest/index.html)

