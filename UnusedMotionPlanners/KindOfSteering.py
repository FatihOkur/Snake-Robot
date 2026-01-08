import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.ndimage import binary_dilation
from scipy.interpolate import splprep, splev

# --- 1. CONFIGURATION ---
NUM_JOINTS = 4          
SEGMENT_LENGTH = 3.0    
SNAKE_WIDTH = 4.0       
INFLATION_RADIUS = int(SNAKE_WIDTH / 2.0 + 1.0)

# Constraints
JOINT_LIMIT = 70.0      # Relaxed limit to allow natural movement
MAX_TURN_ANGLE = np.deg2rad(70) 

RRT_STEP_SIZE = 3.0     
MAX_ITER = 10000        

# --- 2. HELPER MATH FUNCTIONS ---
def normalize_angle(angle):
    while angle > math.pi: angle -= 2 * math.pi
    while angle < -math.pi: angle += 2 * math.pi
    return angle

def get_snake_body(state, yaw_override=None):
    """
    Returns the body points given state [x, y, j1, j2, j3, j4].
    CORRECTED: 
    1. Applies joint angle BEFORE calculating segment position (allows Link 1 to bend).
    2. Removes the extra phantom segment at the end.
    """
    x, y = state[0], state[1]
    joint_angles = state[2:]
    
    current_angle = yaw_override
    body_points = [(x, y)] # Segment 0 (Head)
    
    cx, cy = x, y
    
    for i in range(len(joint_angles)):
        # 1. Apply the joint angle deviation FIRST.
        # This defines the angle of the NEXT segment relative to the PREVIOUS one.
        # (e.g. Joint 1 defines angle of Link 1 relative to Head)
        current_angle -= math.radians(joint_angles[i])
        
        # 2. Calculate the end of this segment
        bx = cx - SEGMENT_LENGTH * math.cos(current_angle)
        by = cy - SEGMENT_LENGTH * math.sin(current_angle)
        
        body_points.append((bx, by))
        cx, cy = bx, by

    # (Phantom segment removed)
    return body_points

# --- 3. ENVIRONMENT ---
class DebrisMap:
    def __init__(self, width=70, height=70):
        self.width = width
        self.height = height
        self.raw_grid = np.zeros((height, width))
        self.planning_grid = np.zeros((height, width))
        self.create_chaos_field() # <--- NEW FUNCTION
        self.inflate_obstacles(radius=INFLATION_RADIUS) 

    def create_chaos_field(self):
        # 1. Borders
        self.raw_grid[0, :] = 1; self.raw_grid[-1, :] = 1
        self.raw_grid[:, 0] = 1; self.raw_grid[:, -1] = 1
        
        # --- STATIC OBSTACLES ---
        
        # Barrier 1: The "Low Wall" (Bottom Obstacle)
        # Forces the snake to go RIGHT to pass.
        # It blocks X=0 to X=45. The gap is on the Right (X > 45).
        self.raw_grid[30:33, 0:45] = 1
        
        # Barrier 2: The "High Wall" (Top Obstacle) - MODIFIED
        # Forces the snake to go LEFT to pass.
        # I shortened this wall. It now blocks X=35 to X=70.
        # The gap is on the Left (X < 35), which is now 35 pixels wide (Very easy).
        self.raw_grid[50:53, 35:70] = 1

        # Obstacle 3: The "Pillars"
        
        # Pillar A (Bottom Right): 
        # Forces a squeeze near the start, but leaves plenty of room.
        self.raw_grid[15:25, 55:60] = 1
        
        # Pillar B (Middle Corridor): 
        # A small island in the middle to force navigation.
        self.raw_grid[40:44, 20:25] = 1
        
        # (REMOVED Pillar C to ensure the path to goal is clear)

        # --- SAFETY CLEARING (CRITICAL) ---
        # Explicitly set the pixels around Start and Goal to 0 (Empty).
        # This overwrites any wall that might have accidentally been placed there.
        
        # Clear Start (35, 20)
        self.raw_grid[15:25, 30:40] = 0
        
        # Clear Goal (60, 60) - Expanded Radius
        # We clear a 10x10 box around the goal to ensure no inflation overlap.
        self.raw_grid[55:65, 55:65] = 0
        
    def inflate_obstacles(self, radius):
        # We increase inflation slightly to make "Narrow" really mean "Narrow" for the body
        # If a gap is 6 pixels wide and snake is 4, inflation leaves exactly 2 pixels of valid space.
        self.planning_grid = binary_dilation(self.raw_grid, structure=np.ones((radius*2, radius*2))).astype(int)

    def is_collision(self, x, y):
        if not (0 <= x < self.width and 0 <= y < self.height): return True
        return self.planning_grid[int(y), int(x)] == 1
        
    def check_line_collision(self, p1, p2):
        x1, y1 = p1; x2, y2 = p2
        dist = math.hypot(x2-x1, y2-y1)
        steps = int(dist * 1.5) 
        if steps == 0: return self.is_collision(x1, y1)
        
        t = np.linspace(0, 1, steps+1)
        x_points = x1 + t * (x2 - x1)
        y_points = y1 + t * (y2 - y1)
        
        x_idx = x_points.astype(int)
        y_idx = y_points.astype(int)
        
        valid_mask = (x_idx >= 0) & (x_idx < self.width) & \
                     (y_idx >= 0) & (y_idx < self.height)
        
        if not np.all(valid_mask): return True 
        if np.any(self.planning_grid[y_idx, x_idx] == 1): return True
        return False
    
# --- 4. PLANNER ---
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

        if not self.is_valid_configuration(self.start):
            print("CRITICAL: Start Configuration Collides!")

    def get_random_sample(self):
        if random.random() < 0.10: 
            return self.goal_conf[:2]
        margin = 3
        rx = random.uniform(margin, self.env.width-margin)
        ry = random.uniform(margin, self.env.height-margin)
        return np.array([rx, ry])
    
    def drag_body(self, new_head_pos, parent_node):
        parent_body = get_snake_body(parent_node.state, yaw_override=parent_node.yaw)
        new_body = [tuple(new_head_pos)]
        
        for i in range(1, len(parent_body)):
            leader = new_body[-1]       
            follower = parent_body[i]   
            
            dx = follower[0] - leader[0]
            dy = follower[1] - leader[1]
            dist = math.hypot(dx, dy)
            
            # Robustness: Prevent division by zero
            if dist < 0.0001: dist = 0.0001
                
            scale = SEGMENT_LENGTH / dist
            nx = leader[0] + dx * scale
            ny = leader[1] + dy * scale
            new_body.append((nx, ny))
            
        return new_body

    def body_to_state(self, body_points, head_yaw):
        x, y = body_points[0]
        joints = []
        
        # We start with the Head Heading
        prev_abs_angle = head_yaw
        
        # Calculate angle for each subsequent segment
        for i in range(1, len(body_points)):
            p_head_side = body_points[i-1] 
            p_tail_side = body_points[i]   
            
            # Vector pointing from Head-side to Tail-side
            dx_back = p_tail_side[0] - p_head_side[0]
            dy_back = p_tail_side[1] - p_head_side[1]
            
            # We need the angle pointing FORWARD (Tail -> Head) to match get_snake_body math
            dx_fwd = -dx_back
            dy_fwd = -dy_back
            
            curr_abs_angle = math.atan2(dy_fwd, dx_fwd)
            
            # The joint angle is the deviation from the previous segment's heading
            rel_angle = normalize_angle(prev_abs_angle - curr_abs_angle)
            
            joints.append(math.degrees(rel_angle))
            prev_abs_angle = curr_abs_angle
            
        return np.array([x, y, *joints])

    def step(self):
        if self.finished: return False
        
        rnd = self.get_random_sample()
        
        dlist = [(node.state[0]-rnd[0])**2 + (node.state[1]-rnd[1])**2 for node in self.nodes]
        nearest_node = self.nodes[np.argmin(dlist)]
        
        # Single Step (Non-Greedy)
        current_node = nearest_node
        
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
        
        # Drag Body
        new_body_points = self.drag_body(new_head_pos, current_node)
        
        # Convert to State
        new_state = self.body_to_state(new_body_points, new_yaw)
        
        # Joint Limit Check
        joints = new_state[2:]
        if np.any(np.abs(joints) > self.joint_limit):
            return False 
            
        new_node = self.Node(new_state, current_node, yaw=new_yaw)
        
        # Collision Check
        if self.is_valid_configuration(new_node):
            self.nodes.append(new_node)
            
            dist_to_goal = math.hypot(new_node.state[0]-self.goal_conf[0], new_node.state[1]-self.goal_conf[1])
            if dist_to_goal <= 4.0: 
                print(f"Goal Reached! Final Dist: {dist_to_goal:.2f}")
                self.finished = True
                self.path = self.extract_path(new_node)
                return True
        
        return False

    def is_valid_configuration(self, node):
        body = get_snake_body(node.state, yaw_override=node.yaw)
        for (bx, by) in body:
            if not (0 <= bx < self.env.width and 0 <= by < self.env.height): return False
        for i in range(len(body)-1):
            if self.env.check_line_collision(body[i], body[i+1]): return False 
        return True

    def extract_path(self, node):
        path = []
        while node is not None:
            path.append((node.state, node.yaw))
            node = node.parent
        return path[::-1]

# --- 5. SMOOTHING ---
def smooth_path_bspline_explicit(path_data, num_points=200):
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
    
    # Use the Start Configuration geometry
    current_body = get_snake_body(states[0], yaw_override=path_data[0][1])
    
    def external_drag(head_pos, prev_body):
        new_b = [head_pos]
        for i in range(1, len(prev_body)):
            leader, follower = new_b[-1], prev_body[i]
            dx, dy = follower[0]-leader[0], follower[1]-leader[1]
            dist = max(math.hypot(dx, dy), 0.0001)
            # STRICTLY ENFORCE SEGMENT LENGTH
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
    bx, by = zip(*body_points)
    ax.plot(bx, by, color=color, linestyle='-', linewidth=lw, alpha=alpha, zorder=15)
    ax.scatter(bx[:-1], by[:-1], color='white', edgecolor='black', s=30, zorder=16)
    # Head marker
    ax.scatter(bx[0], by[0], color='gold', edgecolor='black', marker='D', s=45, zorder=17)

def draw_snake_line_state(ax, state, yaw, color='blue', alpha=1.0, lw=3):
    body = get_snake_body(state, yaw_override=yaw)
    draw_snake_line_explicit(ax, body, color, alpha, lw)

def main():
    WIDTH, HEIGHT = 70, 70
    
    START_CONF = np.array([35.0, 20.0, 0.0, 0.0, 0.0, 0.0]) # 6D
    START_YAW = 0
    
    GOAL_CONF  = np.array([60.0, 60.0, 0.0, 0.0, 0.0, 0.0])
    GOAL_YAW = 0.0
    
    env = DebrisMap(WIDTH, HEIGHT)
    planner = CSpaceRRT(env, START_CONF, GOAL_CONF, start_yaw=START_YAW)
    
    fig, ax = plt.subplots(figsize=(9,9))
    plt.ion() 
    
    print("Searching... (Corrected Kinematics)")
    frame_count = 0
    while not planner.finished:
        if frame_count > MAX_ITER: 
            print("Max iterations reached.")
            break

        # Taking 50 steps per frame for faster Viz
        for _ in range(50): 
            if planner.step(): break
            frame_count += 1
            
        ax.clear()
        ax.imshow(env.raw_grid, cmap='Greys', origin='lower', vmin=0, vmax=1)
        ax.imshow(env.planning_grid, cmap='Reds', alpha=0.2, origin='lower')
        
        # Draw Branches
        for node in planner.nodes:
            if node.parent:
                ax.plot([node.parent.state[0], node.state[0]], 
                        [node.parent.state[1], node.state[1]], 
                        color='blue', linewidth=1, alpha=0.4)

        # Draw Active Snake Tip
        if planner.nodes:
            curr = planner.nodes[-1]
            draw_snake_line_state(ax, curr.state, curr.yaw, color='magenta', lw=2)
            
        draw_snake_line_state(ax, START_CONF, START_YAW, color='green', alpha=0.5)
        draw_snake_line_state(ax, GOAL_CONF, GOAL_YAW, color='red', alpha=0.5)
        
        ax.set_title(f"Nodes: {len(planner.nodes)} | Frames: {frame_count}")
        plt.pause(0.001)

    # --- REPLAY ---
    if planner.path:
        print("Path Found! Smoothing...")
        raw_path = planner.path
        
        # Get explicit body shapes
        smooth_bodies = smooth_path_bspline_explicit(raw_path, num_points=200)
        
        trail_indices = list(range(0, len(smooth_bodies), 8))

        for i, body in enumerate(smooth_bodies):
            ax.clear()
            ax.imshow(env.raw_grid, cmap='Greys', origin='lower')
            
            # 1. Draw Swept Volume
            for idx in trail_indices:
                if idx > i: break 
                draw_snake_line_explicit(ax, smooth_bodies[idx], color='lime', alpha=0.15, lw=4)
            
            # 2. Draw Start/Goal
            draw_snake_line_state(ax, START_CONF, START_YAW, color='green', alpha=0.3)
            draw_snake_line_state(ax, GOAL_CONF, GOAL_YAW, color='red', alpha=0.3)
            
            # 3. Draw Active Snake
            draw_snake_line_explicit(ax, body, color='blue', alpha=1.0)
            
            ax.set_title(f"Replaying Path: {int(i/len(smooth_bodies)*100)}%")
            plt.pause(0.01)
        
        ax.set_title("Target Reached. Close window to exit.")
        plt.ioff()
        plt.show() 
    else:
        print("No path found.")
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()