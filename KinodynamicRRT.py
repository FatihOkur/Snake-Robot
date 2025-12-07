import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import math
import time
from scipy.ndimage import binary_dilation

# --- 1. CONFIGURATION ---
NUM_LINKS = 5           
SEGMENT_LENGTH = 3.0    # Length of one module
RRT_STEP_SIZE = 1.5     # Distance traveled in one simulation step (Keep < SEGMENT_LENGTH for resolution)
MAX_STEER_ANGLE =15    # Degrees change per STEP (determines turning radius)
ANIMATION_SPEED = 10    
GOAL_THRESHOLD = 3.0    # Distance to consider goal reached

# --- 2. ENVIRONMENT ---
class DebrisMap:
    def __init__(self, width=70, height=70, safe_zones=None):
        self.width = width
        self.height = height
        self.raw_grid = np.zeros((height, width)) 
        self.planning_grid = np.zeros((height, width))
        self.safe_zones = safe_zones if safe_zones else []
        
        self.create_hardcore_maze()
        self.carve_guaranteed_path(thickness=6)
        # Inflate slightly more for the body safety margin
        self.inflate_obstacles(radius=3)
        self.enforce_safe_zones()

    def create_hardcore_maze(self):
        self.raw_grid[0, :] = 1; self.raw_grid[-1, :] = 1
        self.raw_grid[:, 0] = 1; self.raw_grid[:, -1] = 1
        mid_x, mid_y = self.width // 2, self.height // 2
        self.raw_grid[mid_y-2:mid_y+2, :] = 1 
        self.raw_grid[:, mid_x-2:mid_x+2] = 1 
        self.raw_grid[mid_y-2:mid_y+2, 10:18] = 0 
        self.raw_grid[50:60, mid_x-2:mid_x+2] = 0
        self.raw_grid[mid_y-2:mid_y+2, 50:58] = 0
        self.raw_grid[10:20, 10:20] = 1
        self.raw_grid[25:30, 5:25] = 1 
        self.raw_grid[45:55, 10:20] = 1 
        self.raw_grid[40:42, 5:30] = 1 
        self.raw_grid[45:65, 45:47] = 1 
        self.raw_grid[40:42, 40:60] = 1 
        self.raw_grid[5:25, 45:47] = 1
        self.raw_grid[15:17, 45:60] = 1

    def carve_guaranteed_path(self, thickness=6):
        waypoints = [(5, 5), (14, 15), (14, 40), (14, 55), (35, 55), (54, 55), (54, 35), (54, 10), (65, 5)]
        for i in range(len(waypoints)-1):
            self.clear_line(waypoints[i], waypoints[i+1], thickness=thickness)

    def clear_line(self, p1, p2, thickness):
        x1, y1 = p1; x2, y2 = p2
        dist = math.hypot(x2-x1, y2-y1)
        if dist == 0: return
        steps = int(dist * 2) 
        for i in range(steps + 1):
            t = i / steps
            x = int(x1 + (x2 - x1) * t)
            y = int(y1 + (y2 - y1) * t)
            for ty in range(-thickness, thickness+1):
                for tx in range(-thickness, thickness+1):
                    if 0 <= x+tx < self.width and 0 <= y+ty < self.height:
                        self.raw_grid[y+ty, x+tx] = 0
                        self.planning_grid[y+ty, x+tx] = 0

    def inflate_obstacles(self, radius):
        self.planning_grid = binary_dilation(self.raw_grid, structure=np.ones((radius*2, radius*2))).astype(int)

    def enforce_safe_zones(self):
        for (sx, sy) in self.safe_zones:
            sy, sx = int(sy), int(sx)
            safe_area = (slice(max(0,sy-4), min(self.height,sy+5)), slice(max(0,sx-4), min(self.width,sx+5)))
            self.raw_grid[safe_area] = 0
            self.planning_grid[safe_area] = 0

    def is_collision(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height: return True
        return self.planning_grid[int(y)][int(x)] == 1

# --- 3. KINODYNAMIC RRT (Motion Primitive Planner) ---
class SnakeKinodynamicRRT:
    class Node:
        def __init__(self, x, y, yaw, cost=0.0, parent=None):
            self.x = x
            self.y = y
            self.yaw = yaw 
            self.parent = parent
            self.cost = cost
            self.dist_from_start = 0.0 if parent is None else parent.dist_from_start + RRT_STEP_SIZE

    def __init__(self, debris_map, start, goal):
        self.map = debris_map
        self.step_size = RRT_STEP_SIZE 
        self.max_steer = math.radians(MAX_STEER_ANGLE)
        
        # Start Node (Head facing East initially)
        self.start = self.Node(start[0], start[1], 0.0) 
        self.goal = self.Node(goal[0], goal[1], 0.0)
        
        self.nodes = [self.start]
        self.finished = False
        self.path = None

    def get_dist(self, n1, n2):
        return math.hypot(n1.x - n2.x, n1.y - n2.y)

    def get_angle_diff(self, a1, a2):
        diff = a2 - a1
        while diff > math.pi: diff -= 2 * math.pi
        while diff < -math.pi: diff += 2 * math.pi
        return diff

    # --- KINEMATICS & BODY TRACING ---
    def get_body_coordinates_at_node(self, node):
        """
        Reconstructs the full snake body by backtracking up the tree 
        from 'node' by fixed distances (SEGMENT_LENGTH).
        """
        body_coords = [(node.x, node.y)] # Head is at index 0
        
        current_trace_node = node
        remaining_dist_to_trace = 0.0
        
        # We need to find N links
        for i in range(NUM_LINKS):
            target_back_dist = SEGMENT_LENGTH
            
            # Walk back up the tree until we cover the distance
            while target_back_dist > 0:
                if current_trace_node.parent is None:
                    # If we run out of history (at start), just stack remaining links at start
                    body_coords.append((current_trace_node.x, current_trace_node.y))
                    target_back_dist = 0 # Done with this link
                else:
                    # Distance to parent
                    dist_to_parent = self.step_size # Approximation since we use fixed steps
                    
                    if dist_to_parent >= target_back_dist:
                        # The point is on this segment. Interpolate.
                        ratio = target_back_dist / dist_to_parent
                        px = current_trace_node.x + ratio * (current_trace_node.parent.x - current_trace_node.x)
                        py = current_trace_node.y + ratio * (current_trace_node.parent.y - current_trace_node.y)
                        
                        body_coords.append((px, py))
                        
                        # Prepare for next link. 
                        # We stay at 'current_trace_node' but logically we are at 'px, py'
                        # Actually simpler: just move the 'virtual cursor'
                        # For RRT simplicity, let's just use the parent node if close enough, or interpolate.
                        # Detailed interpolation is complex for the state; 
                        # let's assume the node resolution is high enough.
                        target_back_dist = 0 
                    else:
                        # Move fully to parent
                        target_back_dist -= dist_to_parent
                        current_trace_node = current_trace_node.parent

        return body_coords

    def check_full_body_collision(self, node):
        """Checks if ANY part of the snake body hits debris."""
        body_coords = self.get_body_coordinates_at_node(node)
        
        # Check Head
        if self.map.is_collision(node.x, node.y): return True
        
        # Check Joints/Links
        for (bx, by) in body_coords:
            if self.map.is_collision(bx, by): return True
            
        # Optional: Check lines between joints (Bresenham)
        for i in range(len(body_coords)-1):
            if self.check_line_collision(body_coords[i], body_coords[i+1]):
                return True
                
        return False

    def check_line_collision(self, p1, p2):
        x0, y0 = int(p1[0]), int(p1[1])
        x1, y1 = int(p2[0]), int(p2[1])
        dx = abs(x1 - x0); dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if self.map.is_collision(x, y): return True
                err -= dy
                if err < 0: y += sy; err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if self.map.is_collision(x, y): return True
                err -= dx
                if err < 0: x += sx; err += dy
                y += sy
        return False

    # --- STEP LOGIC ---
    def step(self):
        if self.finished: return False
        
        # 1. Sample Random Point (Geometric)
        if random.randint(0, 100) > 10: 
            rnd = [random.uniform(0, self.map.width), random.uniform(0, self.map.height)]
        else: 
            rnd = [self.goal.x, self.goal.y]

        # 2. Nearest Node (Euclidean Metric)
        # Note: A true Kinodynamic metric would consider heading, but Euclidean is fast and 'good enough' for this
        dists = [(n.x-rnd[0])**2 + (n.y-rnd[1])**2 for n in self.nodes]
        nearest = self.nodes[np.argmin(dists)]

        # 3. Motion Primitives Expansion
        # Instead of going straight to 'rnd', we try 3 steering inputs and see which one gets closer
        actions = [0, self.max_steer, -self.max_steer] # Straight, Left, Right
        
        best_child = None
        min_dist_to_sample = float('inf')

        for steer in actions:
            # Propagate State (Bicycle Model / Arc)
            new_yaw = nearest.yaw + steer
            new_x = nearest.x + self.step_size * math.cos(new_yaw)
            new_y = nearest.y + self.step_size * math.sin(new_yaw)
            
            # Distance to the random sample
            dist = math.hypot(new_x - rnd[0], new_y - rnd[1])
            
            # Heuristic: Prefer moving towards goal if random sample is goal
            if dist < min_dist_to_sample:
                # Create Candidate
                candidate = self.Node(new_x, new_y, new_yaw, nearest.cost + self.step_size, nearest)
                
                # Check Validity (Full Body & Bounds)
                if 0 <= new_x < self.map.width and 0 <= new_y < self.map.height:
                    if not self.check_full_body_collision(candidate):
                        min_dist_to_sample = dist
                        best_child = candidate
        
        # 4. Add Best Valid Child
        if best_child:
            self.nodes.append(best_child)
            
            # 5. Check Goal Reached
            dist_to_goal = self.get_dist(best_child, self.goal)
            if dist_to_goal < GOAL_THRESHOLD:
                # Add Final Node exactly at goal if angle allows? 
                # For Kinodynamic, usually 'close enough' is sufficient.
                print("Goal Reached!")
                self.path = self.extract_path(best_child)
                self.finished = True
                return True
                
        return False

    def extract_path(self, node):
        path = []
        while node:
            path.append(node)
            node = node.parent
        return path[::-1] # Return reversed list of Nodes

# --- 4. VISUALIZATION ---
def draw_snake_at_node(ax, rrt, node, color='blue'):
    """Draws the reconstructed body at a specific node state."""
    body_coords = rrt.get_body_coordinates_at_node(node)
    
    if len(body_coords) < 2: return
    bx, by = zip(*body_coords)
    
    # Body Links
    ax.plot(bx, by, color=color, linestyle='-', linewidth=4, alpha=0.8, zorder=5)
    # Joints
    ax.plot(bx[:-1], by[:-1], 'o', markerfacecolor='white', markeredgecolor='black', markersize=5, zorder=6)
    # Head
    ax.plot(bx[0], by[0], 'D', markerfacecolor='gold', markeredgecolor='black', markersize=7, zorder=7)

def draw_tree(ax, nodes):
    # Visualize the tree branches
    for node in nodes:
        if node.parent:
            ax.plot([node.parent.x, node.x], [node.parent.y, node.y], 
                    color='cyan', linestyle='-', linewidth=0.5, alpha=1.0)

if __name__ == "__main__":
    WIDTH, HEIGHT = 70, 70
    START = (5, 5)
    GOAL = (65, 5)
    
    debris_env = DebrisMap(WIDTH, HEIGHT, safe_zones=[START, GOAL])
    # Use Kinodynamic RRT
    snake_planner = SnakeKinodynamicRRT(debris_env, START, GOAL)
    
    print("Initialize Kinodynamic RRT...")
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    
    start_time = time.time()
    
    while not snake_planner.finished:
        for _ in range(ANIMATION_SPEED): 
            snake_planner.step()
            if snake_planner.finished: break
        
        ax.clear()
        ax.imshow(debris_env.raw_grid, cmap='Greys', origin='lower')
        # Show obstacles safe margin
        ax.imshow(debris_env.planning_grid, cmap='Reds', alpha=0.2, origin='lower')
        
        ax.plot(START[0], START[1], 'go', markersize=10, label="Start")
        ax.plot(GOAL[0], GOAL[1], 'ro', markersize=10, label="Goal")
        
        draw_tree(ax, snake_planner.nodes)
        
        # Draw the snake body at the newest node to show exploration
        if len(snake_planner.nodes) > 0:
            draw_snake_at_node(ax, snake_planner, snake_planner.nodes[-1], color='magenta')
            
        ax.set_title(f"Nodes: {len(snake_planner.nodes)} | Time: {time.time()-start_time:.1f}s")
        plt.pause(0.01)

    print(f"Path Found! Total Time: {time.time() - start_time:.2f}s")
    
    # Final Draw
    ax.clear()
    ax.imshow(debris_env.raw_grid, cmap='Greys', origin='lower')
    ax.imshow(debris_env.planning_grid, cmap='Reds', alpha=0.3, origin='lower')
    ax.plot(START[0], START[1], 'go', markersize=10)
    ax.plot(GOAL[0], GOAL[1], 'ro', markersize=10)
    
    # Draw Final Path Trace
    path_nodes = snake_planner.path
    px = [n.x for n in path_nodes]
    py = [n.y for n in path_nodes]
    ax.plot(px, py, 'lime', linewidth=2, label="Head Trajectory")
    
    # Draw Final Body Configuration
    draw_snake_at_node(ax, snake_planner, path_nodes[-1], color='blue')
    
    ax.legend(loc='upper right')
    ax.set_title("Final Kinodynamic Path (Arc-Based Primitives)")
    plt.ioff()
    plt.show()