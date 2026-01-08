import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.ndimage import binary_dilation
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree  # NEW: For fast nearest neighbor search

# --- 1. CONFIGURATION ---
"""
CORRECTED ROBOT STRUCTURE - 5 SEGMENTS:
- Physical robot: 5 compartments, 4 joints
- Code model: 6 body points, 5 segments
- HEAD is now a proper segment with length
"""
NUM_JOINTS = 4
NUM_SEGMENTS = 5          
SEGMENT_LENGTH = 3.0    
SNAKE_WIDTH = 4.0       
INFLATION_RADIUS = int(SNAKE_WIDTH / 2.0 + 1.0)

# Constraints
JOINT_LIMIT = 50.0
MAX_TURN_ANGLE = np.deg2rad(70) 

RRT_STEP_SIZE = 3.0     
MAX_ITER = 10000        

# OPTIMIZATION: KD-tree parameters
KDTREE_REBUILD_INTERVAL = 50  # Rebuild tree every N nodes
KDTREE_MIN_NODES = 50         # Start using KD-tree after this many nodes

# --- 2. HELPER MATH FUNCTIONS ---
def normalize_angle(angle):
    while angle > math.pi: angle -= 2 * math.pi
    while angle < -math.pi: angle += 2 * math.pi
    return angle

def line_segment_intersection(p1, p2, p3, p4):
    """Check if line segment (p1, p2) intersects with line segment (p3, p4)."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3
    
    det = dx1 * dy2 - dy1 * dx2
    
    if abs(det) < 1e-10:
        return False
    
    t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / det
    u = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / det
    
    epsilon = 0.01
    if (epsilon < t < 1.0 - epsilon) and (epsilon < u < 1.0 - epsilon):
        return True
    
    return False

def check_self_collision_fast(body_points):
    """
    OPTIMIZED: Fast self-collision check - only checks intersections.
    Skip distance calculations for 2x speedup.
    """
    n = len(body_points)
    
    for i in range(n - 1):
        seg1_start = body_points[i]
        seg1_end = body_points[i + 1]
        
        for j in range(i + 2, n - 1):
            seg2_start = body_points[j]
            seg2_end = body_points[j + 1]
            
            if line_segment_intersection(seg1_start, seg1_end, seg2_start, seg2_end):
                return True
    
    return False

def get_snake_body(state, yaw_override=None):
    """
    CORRECTED: Returns 6 body points representing 5 segments.
    
    Creates proper 5-segment robot matching physical hardware:
    - Segment 0: HEAD compartment (has length)
    - Segments 1-4: Links controlled by joints 1-4
    
    Returns: [HEAD_start, HEAD_end, Link1_end, Link2_end, Link3_end, Link4_end]
    """
    x, y = state[0], state[1]
    joint_angles = state[2:]
    
    current_angle = yaw_override if yaw_override is not None else 0.0
    body_points = []
    
    # Point 0: HEAD start position
    body_points.append((x, y))
    
    # Point 1: HEAD end position (first segment, no joint deviation yet)
    cx = x - SEGMENT_LENGTH * math.cos(current_angle)
    cy = y - SEGMENT_LENGTH * math.sin(current_angle)
    body_points.append((cx, cy))
    
    # Points 2-5: Each controlled by one joint angle
    for i in range(len(joint_angles)):
        # Apply joint angle deviation
        current_angle -= math.radians(joint_angles[i])
        
        # Calculate endpoint of this segment
        cx = cx - SEGMENT_LENGTH * math.cos(current_angle)
        cy = cy - SEGMENT_LENGTH * math.sin(current_angle)
        body_points.append((cx, cy))
    
    return body_points  # Returns 6 points defining 5 segments

# --- 3. ENVIRONMENT ---
class DebrisMap:
    def __init__(self, width=70, height=70):
        self.width = width
        self.height = height
        self.raw_grid = np.zeros((height, width))
        self.planning_grid = np.zeros((height, width))
        self.create_chaos_field()
        self.inflate_obstacles(radius=INFLATION_RADIUS) 

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
        self.planning_grid = binary_dilation(self.raw_grid, structure=np.ones((radius*2, radius*2))).astype(int)

    def is_collision(self, x, y):
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        return self.planning_grid[int(y), int(x)] == 1
        
    def check_line_collision(self, p1, p2):
        """
        OPTIMIZED: Coarser collision sampling for speed
        """
        x1, y1 = p1
        x2, y2 = p2
        dist = math.hypot(x2-x1, y2-y1)
        # OPTIMIZED: Reduced from 1.5 to 0.75
        steps = max(2, int(dist * 0.75))
        
        if steps == 0:
            return self.is_collision(x1, y1)
        
        t = np.linspace(0, 1, steps+1)
        x_points = x1 + t * (x2 - x1)
        y_points = y1 + t * (y2 - y1)
        
        x_idx = x_points.astype(int)
        y_idx = y_points.astype(int)
        
        valid_mask = (x_idx >= 0) & (x_idx < self.width) & \
                     (y_idx >= 0) & (y_idx < self.height)
        
        if not np.all(valid_mask):
            return True 
        if np.any(self.planning_grid[y_idx, x_idx] == 1):
            return True
        return False
    
# --- 4. OPTIMIZED KINEMATIC DRAG RRT (5 SEGMENTS) ---
class CSpaceRRT:
    class Node:
        def __init__(self, state, parent=None, yaw=0.0):
            self.state = np.array(state) 
            self.parent = parent
            self.yaw = yaw 

    def __init__(self, env, start_conf, goal_conf, start_yaw=0.0):
        self.env = env
        self.start = self.Node(start_conf, yaw=start_yaw)
        self.goal_conf = np.array(goal_conf)
        self.nodes = [self.start]
        self.finished = False
        self.path = None
        self.joint_limit = JOINT_LIMIT
        
        # OPTIMIZED: KD-tree for fast 2D nearest neighbor search
        self.kdtree = None
        self.kdtree_update_counter = 0
        
        print("=" * 60)
        print("OPTIMIZED KINEMATIC DRAG RRT - 5-SEGMENT MODEL")
        print("=" * 60)
        print(f"Physical Structure: {NUM_SEGMENTS} compartments, {NUM_JOINTS} joints")
        print(f"Method: Workspace sampling (2D) + kinematic drag")
        print("OPTIMIZATIONS ENABLED:")
        print(f"  - KD-tree 2D nearest neighbor (rebuild every {KDTREE_REBUILD_INTERVAL} nodes)")
        print(f"  - Coarse collision sampling (0.75x density)")
        print(f"  - Fast self-collision (intersection only)")
        print("=" * 60)

        if not self.is_valid_configuration(self.start):
            print("CRITICAL: Start Configuration Collides!")

    def get_random_sample(self):
        """Sample in 2D workspace (head position only)"""
        if random.random() < 0.10: 
            return self.goal_conf[:2]
        margin = 3
        rx = random.uniform(margin, self.env.width-margin)
        ry = random.uniform(margin, self.env.height-margin)
        return np.array([rx, ry])
    
    def rebuild_kdtree(self):
        """
        OPTIMIZED: Build KD-tree for fast 2D nearest neighbor search.
        Only uses x,y positions.
        """
        if len(self.nodes) < KDTREE_MIN_NODES:
            return
        
        # Extract 2D positions (x, y) from all nodes
        positions = np.array([[n.state[0], n.state[1]] for n in self.nodes])
        
        # Build KD-tree (2D is very fast)
        self.kdtree = KDTree(positions)
    
    def find_nearest_node_2d(self, target_xy):
        """
        OPTIMIZED: Find nearest node using KD-tree (O(log n) instead of O(n))
        Searches only in 2D workspace (x, y).
        """
        # Rebuild KD-tree periodically
        self.kdtree_update_counter += 1
        if self.kdtree_update_counter >= KDTREE_REBUILD_INTERVAL:
            self.rebuild_kdtree()
            self.kdtree_update_counter = 0
        
        # Use KD-tree if available
        if self.kdtree is not None and len(self.nodes) >= KDTREE_MIN_NODES:
            # Query KD-tree (O(log n))
            _, idx = self.kdtree.query(target_xy)
            return self.nodes[idx]
        
        # Fallback: linear search for small trees (O(n))
        dlist = [(node.state[0]-target_xy[0])**2 + (node.state[1]-target_xy[1])**2 
                 for node in self.nodes]
        return self.nodes[np.argmin(dlist)]
    
    def drag_body(self, new_head_pos, parent_node):
        """
        Kinematic drag: body segments follow the head.
        Now correctly handles 6 points (5 segments).
        """
        parent_body = get_snake_body(parent_node.state, yaw_override=parent_node.yaw)
        new_body = [tuple(new_head_pos)]
        
        # Drag all 5 following points
        for i in range(1, len(parent_body)):
            leader = new_body[-1]       
            follower = parent_body[i]   
            
            dx = follower[0] - leader[0]
            dy = follower[1] - leader[1]
            dist = math.hypot(dx, dy)
            
            if dist < 0.0001:
                dist = 0.0001
                
            scale = SEGMENT_LENGTH / dist
            nx = leader[0] + dx * scale
            ny = leader[1] + dy * scale
            new_body.append((nx, ny))
            
        return new_body  # Returns 6 points

    def body_to_state(self, body_points, head_yaw):
        """
        Convert body points to state representation.
        Now correctly handles 6 points → 5 segments → 4 joint angles.
        """
        x, y = body_points[0]
        joints = []
        
        # Start with HEAD heading
        prev_abs_angle = head_yaw
        
        # Calculate joint angles from body geometry
        # Skip first segment (HEAD has no joint)
        for i in range(1, len(body_points)):
            p_head_side = body_points[i-1] 
            p_tail_side = body_points[i]   
            
            # Vector pointing from head-side to tail-side
            dx_back = p_tail_side[0] - p_head_side[0]
            dy_back = p_tail_side[1] - p_head_side[1]
            
            # Angle pointing forward (tail → head)
            dx_fwd = -dx_back
            dy_fwd = -dy_back
            
            curr_abs_angle = math.atan2(dy_fwd, dx_fwd)
            
            # Joint angle is deviation from previous segment
            rel_angle = normalize_angle(prev_abs_angle - curr_abs_angle)
            
            joints.append(math.degrees(rel_angle))
            prev_abs_angle = curr_abs_angle
        
        # We now have 5 joint angles, but only need 4 (for the 4 joints)
        # The first angle represents HEAD→Link1 transition (Joint 1)
        # We skip the HEAD segment's "joint angle" since HEAD has no joint
        return np.array([x, y, *joints[1:]])  # Take joints 1-4, skip joint 0

    def is_goal_reached(self, current_state, goal_state):
        """Check if close enough to goal (position + configuration)"""
        pos_error = math.hypot(current_state[0] - goal_state[0], 
                               current_state[1] - goal_state[1])
        
        current_joints = current_state[2:]
        goal_joints = goal_state[2:]
        angle_error = np.linalg.norm(current_joints - goal_joints)
        
        POS_TOLERANCE = 4.0
        ANGLE_TOLERANCE = 30.0
        
        if pos_error <= POS_TOLERANCE and angle_error <= ANGLE_TOLERANCE:
            return True, pos_error, angle_error
        
        return False, pos_error, angle_error

    def step(self):
        """
        OPTIMIZED: Single RRT iteration with performance improvements
        """
        if self.finished:
            return False
        
        # Sample random position (2D only)
        rnd = self.get_random_sample()
        
        # OPTIMIZED: Find nearest node using KD-tree
        nearest_node = self.find_nearest_node_2d(rnd)
        
        current_node = nearest_node
        
        # Steer toward sample
        dx = rnd[0] - current_node.state[0]
        dy = rnd[1] - current_node.state[1]
        target_yaw = math.atan2(dy, dx)
        
        diff_yaw = normalize_angle(target_yaw - current_node.yaw)
        
        if abs(diff_yaw) > MAX_TURN_ANGLE:
            diff_yaw = MAX_TURN_ANGLE * np.sign(diff_yaw)
            
        new_yaw = normalize_angle(current_node.yaw + diff_yaw)
        new_x = current_node.state[0] + RRT_STEP_SIZE * math.cos(new_yaw)
        new_y = current_node.state[1] + RRT_STEP_SIZE * math.sin(new_yaw)
        new_head_pos = (new_x, new_y)
        
        # Drag body (now correctly handles 6 points)
        new_body_points = self.drag_body(new_head_pos, current_node)
        
        # Convert to state (now correctly extracts 4 joint angles)
        new_state = self.body_to_state(new_body_points, new_yaw)
        
        # OPTIMIZED: Check joint limits first (cheapest check)
        joints = new_state[2:]
        if np.any(np.abs(joints) > self.joint_limit):
            return False 
            
        new_node = self.Node(new_state, current_node, yaw=new_yaw)
        
        # OPTIMIZED: Collision check
        if self.is_valid_configuration(new_node):
            self.nodes.append(new_node)
            
            # Check goal
            reached, p_err, a_err = self.is_goal_reached(new_node.state, self.goal_conf)
            
            if reached:
                print(f"Goal Reached! Pos Error: {p_err:.2f}, Angle Error: {a_err:.2f}")
                self.finished = True
                self.path = self.extract_path(new_node)
                return True
        
        return False

    def is_valid_configuration(self, node):
        """
        OPTIMIZED: Check if configuration is collision-free.
        Includes fast self-collision detection.
        """
        # OPTIMIZED: Joint limit check first (cheapest)
        if np.any(np.abs(node.state[2:]) > self.joint_limit):
            return False
        
        body = get_snake_body(node.state, yaw_override=node.yaw)
        
        # OPTIMIZED: Vectorized boundary check
        body_array = np.array(body)
        if (np.any(body_array[:, 0] < 0) or np.any(body_array[:, 0] >= self.env.width) or
            np.any(body_array[:, 1] < 0) or np.any(body_array[:, 1] >= self.env.height)):
            return False
        
        # Environment collision
        for i in range(len(body)-1):
            if self.env.check_line_collision(body[i], body[i+1]):
                return False 
        
        # OPTIMIZED: Fast self-collision check (intersection only)
        if check_self_collision_fast(body):
            return False
        
        return True

    def extract_path(self, node):
        """Extract path from start to goal"""
        path = []
        while node is not None:
            path.append((node.state, node.yaw))
            node = node.parent
        return path[::-1]

# --- 5. SMOOTHING ---
def smooth_path_bspline_explicit(path_data, num_points=200):
    """Smooth path using B-spline interpolation"""
    states = [p[0] for p in path_data]
    x = [s[0] for s in states]
    y = [s[1] for s in states]
    
    if len(x) < 3: 
        full_bodies = []
        for s, yaw in path_data:
            full_bodies.append(get_snake_body(s, yaw))
        return full_bodies

    tck, u = splprep([x, y], s=2.0)
    u_new = np.linspace(0, 1, num_points)
    new_points = splev(u_new, tck)
    new_x, new_y = new_points[0], new_points[1]
    
    smoothed_bodies = []
    current_body = get_snake_body(states[0], yaw_override=path_data[0][1])
    
    def external_drag(head_pos, prev_body):
        new_b = [head_pos]
        for i in range(1, len(prev_body)):
            leader, follower = new_b[-1], prev_body[i]
            dx, dy = follower[0]-leader[0], follower[1]-leader[1]
            dist = max(math.hypot(dx, dy), 0.0001)
            scale = SEGMENT_LENGTH / dist
            new_b.append((leader[0]+dx*scale, leader[1]+dy*scale))
        return new_b

    for i in range(len(new_x)):
        head_pos = (new_x[i], new_y[i])
        if i > 0:
            current_body = external_drag(head_pos, current_body)
        smoothed_bodies.append(current_body)
        
    return smoothed_bodies

# --- 6. VISUALIZATION ---
def draw_snake_line_explicit(ax, body_points, color='blue', alpha=1.0, lw=3):
    """
    Draw 5-segment snake body.
    body_points: 6 points [HEAD_start, HEAD_end, L1_end, L2_end, L3_end, L4_end]
    """
    bx, by = zip(*body_points)
    
    # Draw body line (5 segments)
    ax.plot(bx, by, color=color, linestyle='-', linewidth=lw, alpha=alpha, zorder=15)
    
    # Draw joint markers (at HEAD_end and link ends, not tail)
    ax.scatter(bx[1:-1], by[1:-1], color='white', edgecolor='black', s=30, zorder=16)
    
    # Draw head marker
    ax.scatter(bx[0], by[0], color='gold', edgecolor='black', marker='D', s=45, zorder=17)
    
    # Draw tail marker
    ax.scatter(bx[-1], by[-1], color='red', edgecolor='black', s=30, zorder=16)

def draw_snake_line_state(ax, state, yaw, color='blue', alpha=1.0, lw=3):
    body = get_snake_body(state, yaw_override=yaw)
    draw_snake_line_explicit(ax, body, color, alpha, lw)

def main():
    import time  # For performance measurement
    
    WIDTH, HEIGHT = 70, 70
    
    START_CONF = np.array([35.0, 20.0, 0.0, 0.0, 0.0, 0.0])
    START_YAW = 0
    
    GOAL_CONF = np.array([60.0, 60.0, 0.0, 0.0, 0.0, 0.0])
    GOAL_YAW = 0.0
    
    env = DebrisMap(WIDTH, HEIGHT)
    planner = CSpaceRRT(env, START_CONF, GOAL_CONF, start_yaw=START_YAW)
    
    fig, ax = plt.subplots(figsize=(9,9))
    plt.ion() 
    
    print("\nSearching with OPTIMIZED Kinematic Drag (5 segments)...")
    frame_count = 0
    start_time = time.time()
    
    while not planner.finished:
        if frame_count > MAX_ITER: 
            print("Max iterations reached.")
            break

        for _ in range(50): 
            if planner.step():
                break
            frame_count += 1
            
        ax.clear()
        ax.imshow(env.raw_grid, cmap='Greys', origin='lower', vmin=0, vmax=1)
        ax.imshow(env.planning_grid, cmap='Reds', alpha=0.2, origin='lower')
        
        # Draw tree branches
        for node in planner.nodes:
            if node.parent:
                ax.plot([node.parent.state[0], node.state[0]], 
                        [node.parent.state[1], node.state[1]], 
                        color='blue', linewidth=1, alpha=0.4)

        # Draw current snake
        if planner.nodes:
            curr = planner.nodes[-1]
            draw_snake_line_state(ax, curr.state, curr.yaw, color='magenta', lw=2)
            
        draw_snake_line_state(ax, START_CONF, START_YAW, color='green', alpha=0.5)
        draw_snake_line_state(ax, GOAL_CONF, GOAL_YAW, color='red', alpha=0.5)
        
        elapsed = time.time() - start_time
        iter_per_sec = frame_count / elapsed if elapsed > 0 else 0
        ax.set_title(f"OPTIMIZED 5-Seg Drag | Nodes: {len(planner.nodes)} | Iter: {frame_count} | Speed: {iter_per_sec:.1f} it/s")
        plt.pause(0.001)

    # Print performance stats
    elapsed = time.time() - start_time
    print(f"\n📊 Performance Stats:")
    print(f"   Total time: {elapsed:.2f} seconds")
    print(f"   Iterations: {frame_count}")
    print(f"   Iterations/sec: {frame_count/elapsed:.1f}")
    print(f"   Nodes in tree: {len(planner.nodes)}")

    # --- REPLAY ---
    if planner.path:
        print("Path Found! Smoothing...")
        
        smooth_bodies = smooth_path_bspline_explicit(planner.path, num_points=200)
        trail_indices = list(range(0, len(smooth_bodies), 8))

        for i, body in enumerate(smooth_bodies):
            ax.clear()
            ax.imshow(env.raw_grid, cmap='Greys', origin='lower')
            
            # Draw trail
            for idx in trail_indices:
                if idx > i:
                    break 
                draw_snake_line_explicit(ax, smooth_bodies[idx], color='lime', alpha=0.15, lw=4)
            
            # Draw start/goal
            draw_snake_line_state(ax, START_CONF, START_YAW, color='green', alpha=0.3)
            draw_snake_line_state(ax, GOAL_CONF, GOAL_YAW, color='red', alpha=0.3)
            
            # Draw current snake
            draw_snake_line_explicit(ax, body, color='blue', alpha=1.0)
            
            ax.set_title(f"OPTIMIZED Path: {int(i/len(smooth_bodies)*100)}%")
            plt.pause(0.01)
        
        ax.set_title("Target Reached (OPTIMIZED). Close window.")
        plt.ioff()
        plt.show() 
    else:
        print("No path found.")
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()