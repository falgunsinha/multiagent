from isaacsim.util.debug_draw import _debug_draw


class VisualGrid:
    """
    Creates visual grid lines on the ground plane in Isaac Sim using debug draw.
    """

    def __init__(self, z_height=0.001):
        """
        Initialize the VisualGrid
        """
        self.draw_interface = _debug_draw.acquire_debug_draw_interface()
        self.line_height = z_height
        self.white_color = (1.0, 1.0, 1.0, 1.0)
        self.line_width = 2.0

    def create_grid(self, start_x, start_y, grid_extent_x, grid_extent_y, cell_size, num_rows, num_cols):
        """
        Create visual grid lines on the ground plane using debug draw.
        """
        line_count = 0
        z = self.line_height

     
        start_points = []
        end_points = []
        colors = []
        widths = []

        # Create horizontal lines (parallel to X-axis)
        num_horizontal_lines = num_rows + 1
        for i in range(num_horizontal_lines):
            y_pos = start_y + (i * cell_size)
   
            start_points.append((start_x, y_pos, z))
            end_points.append((start_x + grid_extent_x, y_pos, z))
            colors.append(self.white_color)
            widths.append(self.line_width)
            line_count += 1

        # Create vertical lines (parallel to Y-axis)
        num_vertical_lines = num_cols + 1
        for i in range(num_vertical_lines):
            x_pos = start_x + (i * cell_size)
        
            start_points.append((x_pos, start_y, z))
            end_points.append((x_pos, start_y + grid_extent_y, z))
            colors.append(self.white_color)
            widths.append(self.line_width)
            line_count += 1

        # Draw all lines in one batch
        self.draw_interface.draw_lines(start_points, end_points, colors, widths)

        return line_count

    def clear_grid(self):
        """Clear all debug draw lines."""
        self.draw_interface.clear_lines()


def create_visual_grid(start_x, start_y, grid_extent_x, grid_extent_y, cell_size, num_rows, num_cols, z_height=0.001):
    """
    Convenience function to create a visual grid using debug draw.
    """
    grid = VisualGrid(z_height=z_height)
    grid.create_grid(start_x, start_y, grid_extent_x, grid_extent_y, cell_size, num_rows, num_cols)
    return grid

