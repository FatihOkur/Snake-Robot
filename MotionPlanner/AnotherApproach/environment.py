import numpy as np
import math
from scipy.ndimage import distance_transform_edt, map_coordinates
import config

class DebrisMap:
    def __init__(self, width=70, height=70):
        self.width = width
        self.height = height
        self.raw_grid = np.zeros((height, width))
        self.half_width = config.SNAKE_WIDTH / 2.0
        self.create_chaos_field()
        self._build_distance_field()

    def create_chaos_field(self):
        # 1. Fill the entire map with impassable debris
        self.raw_grid[:, :] = 1

        # 2. Carve an incredibly tight 10-unit wide winding spiral maze
        # Path 1: Vertical Start (Bottom-Left)
        self.raw_grid[2:30, 10:20] = 0
        
        # Path 2: Sharp 90-deg Turn Right (Horizontal East)
        self.raw_grid[20:30, 10:50] = 0
        
        # Path 3: Sharp 90-deg Turn Left (Vertical North)
        self.raw_grid[20:50, 40:50] = 0
        
        # Path 4: Sharp 90-deg Turn Left (Horizontal West)
        self.raw_grid[40:50, 20:50] = 0
        
        # Path 5: Sharp 90-deg Turn Right to Goal (Vertical North)
        self.raw_grid[40:68, 20:30] = 0

    def _build_distance_field(self):
        self.distance_field = distance_transform_edt(1 - self.raw_grid)
        # Matplotlib imshow draws pixels from center-0.5 to center+0.5.
        # We must add 0.5 to our clearance to prevent the continuous robot envelope 
        # from touching the visual boundary of the discrete grid cells.
        self.safe_threshold = self.half_width + 0.5
        self.planning_grid = (self.distance_field < self.safe_threshold).astype(int)

    def is_collision(self, x, y):
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        # Bilinear interpolation for sub-pixel accuracy
        dist = map_coordinates(self.distance_field, [[y], [x]], order=1, mode='nearest')[0]
        return dist < self.safe_threshold
        
    def check_line_collision(self, p1, p2):
        """Check if any point along a line segment is too close to an obstacle."""
        x1, y1 = p1
        x2, y2 = p2
        dist = math.hypot(x2-x1, y2-y1)
        # 5 samples per unit ensures we don't skip over any grid cells
        steps = max(5, int(dist * 5.0))
        
        t = np.linspace(0, 1, steps+1)
        x_points = x1 + t * (x2 - x1)
        y_points = y1 + t * (y2 - y1)
        
        # Bounds check
        valid_mask = (x_points >= 0) & (x_points < self.width) & \
                     (y_points >= 0) & (y_points < self.height)
        
        if not np.all(valid_mask):
            return True
            
        # Vectorized bilinear interpolation check for all points along the segment
        dists = map_coordinates(self.distance_field, [y_points, x_points], order=1, mode='nearest')
        if np.any(dists < self.safe_threshold):
            return True
        return False

# Standalone helper for self-collision (geometry only)
def line_segment_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x4 - x3, y4 - y3
    det = dx1 * dy2 - dy1 * dx2
    if abs(det) < 1e-10: return False
    t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / det
    u = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / det
    return 0.01 < t < 0.99 and 0.01 < u < 0.99