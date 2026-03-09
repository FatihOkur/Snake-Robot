import numpy as np
import math
from scipy.ndimage import binary_dilation
import config

class DebrisMap:
    def __init__(self, width=70, height=70):
        self.width = width
        self.height = height
        self.raw_grid = np.zeros((height, width))
        self.planning_grid = np.zeros((height, width))
        self.create_chaos_field()
        self.inflate_obstacles(radius=config.INFLATION_RADIUS)

    def create_chaos_field(self):
        # Borders
        self.raw_grid[0, :] = 1
        self.raw_grid[-1, :] = 1
        self.raw_grid[:, 0] = 1
        self.raw_grid[:, -1] = 1
        
        # Bottom wall with right gap
        self.raw_grid[30:33, 0:45] = 1
        
        # Top wall with left gap
        self.raw_grid[50:53, 35:70] = 1

        # Pillars
        self.raw_grid[15:25, 55:60] = 1
        self.raw_grid[40:44, 20:25] = 1
        
        # Clear start and goal areas
        self.raw_grid[15:25, 30:40] = 0
        self.raw_grid[55:65, 55:65] = 0
        
    def inflate_obstacles(self, radius):
        structure = np.ones((radius*2, radius*2))
        self.planning_grid = binary_dilation(self.raw_grid, structure=structure).astype(int)

    def is_collision(self, x, y):
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        return self.planning_grid[int(y), int(x)] == 1
        
    def check_line_collision(self, p1, p2):
        """Coarse sampling for line collision check"""
        x1, y1 = p1
        x2, y2 = p2
        dist = math.hypot(x2-x1, y2-y1)
        steps = max(2, int(dist * 0.75))
        
        t = np.linspace(0, 1, steps+1)
        x_points = x1 + t * (x2 - x1)
        y_points = y1 + t * (y2 - y1)
        
        x_idx = x_points.astype(int)
        y_idx = y_points.astype(int)
        
        # Bounds check
        valid_mask = (x_idx >= 0) & (x_idx < self.width) & \
                     (y_idx >= 0) & (y_idx < self.height)
        
        if not np.all(valid_mask):
            return True
        if np.any(self.planning_grid[y_idx, x_idx] == 1):
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