"""
Object Manager Module for RRT Pick and Place
Handles creation and management of dynamic cuboids in pyramid stacks
"""

import numpy as np
import time
from isaacsim.core.api.objects import DynamicCuboid


class ObjectManager:
    """Manages dynamic cuboid objects in pyramid-style stacking"""

    def __init__(self, world, base_layer_count=4, num_layers=4):
        """
        Initialize object manager with configurable pyramid stack

        Args:
            world: Isaac Sim World instance
            base_layer_count: Number of cuboids in the bottom layer (e.g., 4 for 2x2 grid)
            num_layers: Total number of layers in the pyramid (e.g., 4 for layers of 4,3,2,1)
        """
        self.world = world
        self.cuboids = []  # List of all created cuboids with metadata

        # Pyramid configuration
        self.base_layer_count = base_layer_count  # Number of cuboids in bottom layer
        self.num_layers = num_layers  # Total number of layers
        self.base_position = np.array([0.5, 0.0, 0.0])  # Center position of the pyramid

        # Stack state
        self.stack_added = False

        # Cube properties
        self.cube_size = 0.0515  # 5.15 cm cubes
        self.cube_half_size = self.cube_size / 2.0
        
    def _calculate_layer_positions(self, layer_index, cuboids_in_layer):
        """
        Calculate positions for cuboids in a specific layer

        Args:
            layer_index: Layer number (0 = bottom, 1 = second from bottom, etc.)
            cuboids_in_layer: Number of cuboids in this layer

        Returns:
            List of positions for each cuboid in the layer
        """
        positions = []

        # Calculate Z position (height) for this layer
        z_position = self.cube_half_size + (layer_index * self.cube_size) + 0.01

        # Arrange cuboids in a rectangular grid pattern
        # For example: 4 cuboids -> 2x2 grid, 3 cuboids -> 2x2 with one missing, etc.
        if cuboids_in_layer == 1:
            # Single cuboid at center
            positions.append(self.base_position + np.array([0.0, 0.0, z_position]))
        elif cuboids_in_layer == 2:
            # 2 cuboids in a line
            spacing = self.cube_size * 0.5
            positions.append(self.base_position + np.array([-spacing, 0.0, z_position]))
            positions.append(self.base_position + np.array([spacing, 0.0, z_position]))
        elif cuboids_in_layer == 3:
            # 3 cuboids in a triangle or line
            spacing = self.cube_size * 0.5
            positions.append(self.base_position + np.array([-spacing, -spacing, z_position]))
            positions.append(self.base_position + np.array([spacing, -spacing, z_position]))
            positions.append(self.base_position + np.array([0.0, spacing, z_position]))
        elif cuboids_in_layer == 4:
            # 4 cuboids in a 2x2 grid
            spacing = self.cube_size * 0.5
            positions.append(self.base_position + np.array([-spacing, -spacing, z_position]))
            positions.append(self.base_position + np.array([spacing, -spacing, z_position]))
            positions.append(self.base_position + np.array([-spacing, spacing, z_position]))
            positions.append(self.base_position + np.array([spacing, spacing, z_position]))
        else:
            # For larger numbers, create a grid pattern
            grid_size = int(np.ceil(np.sqrt(cuboids_in_layer)))
            spacing = self.cube_size * 0.5
            offset = (grid_size - 1) * spacing / 2.0

            idx = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if idx >= cuboids_in_layer:
                        break
                    x = i * spacing - offset
                    y = j * spacing - offset
                    positions.append(self.base_position + np.array([x, y, z_position]))
                    idx += 1
                if idx >= cuboids_in_layer:
                    break

        return positions

    def add_stack(self):
        """Add pyramid stack of cuboids with configurable layers"""
        if self.stack_added:
            print("Pyramid stack has already been added!")
            return False

        print(f"\n=== Adding Pyramid Stack ===")
        print(f"Base layer: {self.base_layer_count} cuboids")
        print(f"Total layers: {self.num_layers}")

        total_cuboids = 0

        # Create layers from bottom to top
        for layer_index in range(self.num_layers):
            # Calculate number of cuboids in this layer
            # Decreases by 1 each layer: base_layer_count, base_layer_count-1, ..., 1
            cuboids_in_layer = self.base_layer_count - layer_index

            if cuboids_in_layer <= 0:
                break

            print(f"\nLayer {layer_index + 1}: {cuboids_in_layer} cuboid(s)")

            # Get positions for this layer
            positions = self._calculate_layer_positions(layer_index, cuboids_in_layer)

            # Create cuboids for this layer
            for pos_index, position in enumerate(positions):
                # Generate unique name
                timestamp = int(time.time() * 1000) + total_cuboids  # Ensure uniqueness
                cube_name = f"cube_L{layer_index}_P{pos_index}_{timestamp}"
                cube_prim_path = f"/World/Cube_Layer{layer_index}_Pos{pos_index}_{timestamp}"

                # Create cuboid
                cuboid = self.world.scene.add(
                    DynamicCuboid(
                        name=cube_name,
                        position=position,
                        prim_path=cube_prim_path,
                        scale=np.array([self.cube_size, self.cube_size, self.cube_size]),
                        size=1.0,
                        color=np.array([0.0, 0.0, 1.0]),  # Blue
                    )
                )

                self.cuboids.append({
                    'object': cuboid,
                    'name': cube_name,
                    'prim_path': cube_prim_path,
                    'layer': layer_index,
                    'position_in_layer': pos_index,
                    'picked': False
                })

                print(f"  Added: {cube_name} at position {position}")
                total_cuboids += 1

        self.stack_added = True
        print(f"\nPyramid stack added successfully!")
        print(f"Total cuboids in pyramid: {total_cuboids}\n")

        return True
    
    def get_next_unpicked_cuboid(self):
        """Get the next cuboid that hasn't been picked yet"""
        for cuboid_info in self.cuboids:
            if not cuboid_info['picked']:
                return cuboid_info
        return None
    
    def mark_cuboid_picked(self, cuboid_info):
        """Mark a cuboid as picked"""
        cuboid_info['picked'] = True
        print(f"Marked {cuboid_info['name']} as picked")
    
    def get_unpicked_count(self):
        """Get count of unpicked cuboids"""
        return sum(1 for c in self.cuboids if not c['picked'])
    
    def get_total_count(self):
        """Get total count of cuboids"""
        return len(self.cuboids)
    
    def reset_all(self):
        """Reset all cuboids (mark as unpicked)"""
        for cuboid_info in self.cuboids:
            cuboid_info['picked'] = False
        print("All cuboids reset to unpicked state")
    
    def clear_all(self):
        """Clear all cuboids from the scene"""
        print("\n=== Clearing all cuboids ===")
        for cuboid_info in self.cuboids:
            try:
                # Remove from scene
                prim_path = cuboid_info['prim_path']
                from omni.isaac.core.utils.stage import get_current_stage
                stage = get_current_stage()
                prim = stage.GetPrimAtPath(prim_path)
                if prim.IsValid():
                    stage.RemovePrim(prim_path)
                    print(f"  Removed: {cuboid_info['name']}")
            except Exception as e:
                print(f"  Error removing {cuboid_info['name']}: {e}")

        self.cuboids.clear()
        self.stack_added = False
        print("All cuboids cleared!\n")

    def get_stack_info(self):
        """Get information about current pyramid stack"""
        info = {
            'base_layer_count': self.base_layer_count,
            'num_layers': self.num_layers,
            'stack_added': self.stack_added,
            'total_cuboids': len(self.cuboids),
            'unpicked_cuboids': self.get_unpicked_count()
        }
        return info

