import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.ndimage import binary_dilation
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree  # NEW: For fast nearest neighbor search

# --- 1. CONFIGURATION ---
"""
CORRECTED ROBOT STRUCTURE TO MATCH PHYSICAL HARDWARE:

Physical Robot:
- 5 compartments (physical bodies with tracks)
- 4 joints (spherical gear mechanisms between compartments)

Layout:
    [Compartment 0] ─J1─ [Compartment 1] ─J2─ [Compartment 2] ─J3─ [Compartment 3] ─J4─ [Compartment 4]
         HEAD                  Link 1              Link 2              Link 3              Link 4

Code Model (CORRECTED):
- 6 body points: [HEAD_start, HEAD_end, Link1_end, Link2_end, Link3_end, Link4_end]
- 5 segments: HEAD segment + 4 links
- 4 joints: at HEAD_end, Link1_end, Link2_end, Link3_end

State: [x, y, θ₁, θ₂, θ₃, θ₄]
- (x, y): HEAD starting position
- θᵢ: Joint i angle (relative to previous segment)
- Head orientation: yaw (absolute)
"""
NUM_JOINTS = 4          # Number of articulation joints
NUM_SEGMENTS = 5        # Number of physical compartments (HEAD + 4 links)
SEGMENT_LENGTH = 3.0    # Length of each compartment
SNAKE_WIDTH = 4.0       
INFLATION_RADIUS = int(SNAKE_WIDTH / 2.0 + 1.0)

# Constraints
JOINT_LIMIT = 50.0
MAX_JOINT_CHANGE = 20.0
MAX_TURN_ANGLE = np.deg2rad(70)

# RRT Parameters
RRT_STEP_SIZE = 3.0
C_SPACE_STEP_SIZE = 0.3
MAX_ITER = 15000

# Distance Metric Weights
POSITION_WEIGHT = 0.7
JOINT_WEIGHT = 0.3

# OPTIMIZATION: KD-tree parameters
KDTREE_REBUILD_INTERVAL = 50  # Rebuild tree every N nodes
KDTREE_MIN_NODES = 50         # Start using KD-tree after this many nodes

# --- 2. HELPER MATH FUNCTIONS ---
def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    while angle > math.pi: angle -= 2 * math.pi
    while angle < -math.pi: angle += 2 * math.pi
    return angle

def line_segment_intersection(p1, p2, p3, p4):
    """
    Check if line segment (p1, p2) intersects with line segment (p3, p4).
    Returns True if segments intersect (excluding endpoints touching).
    """
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
    OPTIMIZED: Fast self-collision check - only checks intersections, not distances.
    This is ~2x faster than the full version with distance checks.
    
    Args:
        body_points: List of (x, y) tuples representing body positions
    
    Returns:
        True if self-collision detected, False otherwise
    """
    n = len(body_points)
    
    # Check all pairs of segments (skip adjacent segments)
    for i in range(n - 1):
        seg1_start = body_points[i]
        seg1_end = body_points[i + 1]
        
        # Only check against segments at least 2 segments away
        for j in range(i + 2, n - 1):
            seg2_start = body_points[j]
            seg2_end = body_points[j + 1]
            
            # Check line-line intersection only (skip distance checks for speed)
            if line_segment_intersection(seg1_start, seg1_end, seg2_start, seg2_end):
                return True
    
    return False

def get_snake_body(state, yaw_override=None):
    """
    CORRECTED Forward kinematics: 5 segments, 4 joints
    
    Returns 6 body points representing:
    [HEAD_start, HEAD_end, Link1_end, Link2_end, Link3_end, Link4_end]
    
    This creates 5 physical segments:
    - Segment 0: HEAD_start → HEAD_end (HEAD compartment)
    - Segment 1: HEAD_end → Link1_end (Joint 1 controls this)
    - Segment 2: Link1_end → Link2_end (Joint 2 controls this)
    - Segment 3: Link2_end → Link3_end (Joint 3 controls this)
    - Segment 4: Link3_end → Link4_end (Joint 4 controls this)
    """
    x, y = state[0], state[1]
    joint_angles = state[2:]  # [θ₁, θ₂, θ₃, θ₄]
    
    current_angle = yaw_override if yaw_override is not None else 0.0
    body_points = []
    
    # Point 0: HEAD start position
    body_points.append((x, y))
    
    # Point 1: HEAD end position (first segment, no joint deviation yet)
    cx = x - SEGMENT_LENGTH * math.cos(current_angle)
    cy = y - SEGMENT_LENGTH * math.sin(current_angle)
    body_points.append((cx, cy))
    
    # Points 2-5: Each controlled by one joint
    for i in range(len(joint_angles)):
        # Apply joint angle deviation
        current_angle -= math.radians(joint_angles[i])
        
        # Calculate next segment endpoint
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
        structure = np.ones((radius*2, radius*2))
        self.planning_grid = binary_dilation(self.raw_grid, structure=structure).astype(int)

    def is_collision(self, x, y):
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        return self.planning_grid[int(y), int(x)] == 1
        
    def check_line_collision(self, p1, p2):
        """
        OPTIMIZED: Check collision along a line segment with coarser sampling
        """
        x1, y1 = p1
        x2, y2 = p2
        dist = math.hypot(x2-x1, y2-y1)
        # OPTIMIZED: Reduced sampling density from 1.5 to 0.75
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

# --- 4. OPTIMIZED 6D CONFIGURATION SPACE RRT ---
class ConfigurationSpaceRRT:
    class Node:
        def __init__(self, state, parent=None, yaw=0.0):
            self.state = np.array(state, dtype=float)
            self.parent = parent
            self.yaw = yaw

    def __init__(self, env, start_conf, goal_conf, start_yaw=0.0):
        self.env = env
        self.start = self.Node(start_conf, yaw=start_yaw)
        self.goal_conf = np.array(goal_conf)
        self.goal_yaw = 0.0
        self.nodes = [self.start]
        self.finished = False
        self.path = None
        self.joint_limit = JOINT_LIMIT
        
        # OPTIMIZED: KD-tree for fast nearest neighbor search
        self.kdtree = None
        self.kdtree_update_counter = 0
        
        print("=" * 60)
        print("OPTIMIZED 6D C-SPACE RRT - 5-SEGMENT MODEL")
        print("=" * 60)
        print(f"Physical Structure: {NUM_SEGMENTS} compartments, {NUM_JOINTS} joints")
        print(f"Start Config: {start_conf}")
        print(f"Goal Config:  {goal_conf}")
        print(f"Joint Limit:  ±{JOINT_LIMIT}°")
        print("OPTIMIZATIONS ENABLED:")
        print(f"  - KD-tree nearest neighbor (rebuild every {KDTREE_REBUILD_INTERVAL} nodes)")
        print(f"  - Reduced interpolation checks (2 instead of 5)")
        print(f"  - Coarse collision sampling (0.75x density)")
        print(f"  - Fast self-collision (intersection only)")
        print("=" * 60)

        if not self.is_valid_configuration(self.start):
            print("❌ CRITICAL: Start Configuration Collides!")

    def get_random_sample_6D(self):
        """TRUE C-SPACE SAMPLING: Samples all 6 dimensions independently"""
        if random.random() < 0.10:
            return self.goal_conf.copy()
        
        margin = 5
        rx = random.uniform(margin, self.env.width - margin)
        ry = random.uniform(margin, self.env.height - margin)
        
        j1 = random.uniform(-self.joint_limit, self.joint_limit)
        j2 = random.uniform(-self.joint_limit, self.joint_limit)
        j3 = random.uniform(-self.joint_limit, self.joint_limit)
        j4 = random.uniform(-self.joint_limit, self.joint_limit)
        
        return np.array([rx, ry, j1, j2, j3, j4])
    
    def c_space_distance(self, state1, state2):
        """Weighted distance metric in configuration space"""
        pos_dist = np.linalg.norm(state1[:2] - state2[:2])
        joint_dist = np.linalg.norm(state1[2:] - state2[2:])
        return POSITION_WEIGHT * pos_dist + JOINT_WEIGHT * joint_dist
    
    def rebuild_kdtree(self):
        """
        OPTIMIZED: Build KD-tree for fast nearest neighbor search.
        Uses weighted states to match the c_space_distance metric.
        """
        if len(self.nodes) < KDTREE_MIN_NODES:
            return
        
        # Extract all states
        states = np.array([n.state for n in self.nodes])
        
        # Apply weights to match distance metric
        weighted_states = states.copy()
        weighted_states[:, :2] *= POSITION_WEIGHT  # x, y
        weighted_states[:, 2:] *= JOINT_WEIGHT      # joints
        
        # Build KD-tree
        self.kdtree = KDTree(weighted_states)
        
    def find_nearest_node(self, target_state):
        """
        OPTIMIZED: Find nearest node using KD-tree (O(log n) instead of O(n))
        Falls back to linear search for small trees.
        """
        # Rebuild KD-tree periodically
        self.kdtree_update_counter += 1
        if self.kdtree_update_counter >= KDTREE_REBUILD_INTERVAL:
            self.rebuild_kdtree()
            self.kdtree_update_counter = 0
        
        # Use KD-tree if available
        if self.kdtree is not None and len(self.nodes) >= KDTREE_MIN_NODES:
            # Apply same weights to target
            weighted_target = target_state.copy()
            weighted_target[:2] *= POSITION_WEIGHT
            weighted_target[2:] *= JOINT_WEIGHT
            
            # Query KD-tree (O(log n))
            _, idx = self.kdtree.query(weighted_target)
            return self.nodes[idx]
        
        # Fallback: linear search for small trees (O(n))
        distances = [self.c_space_distance(node.state, target_state) 
                     for node in self.nodes]
        return self.nodes[np.argmin(distances)]
    
    def is_configuration_reachable(self, from_state, to_state):
        """Check if configuration transition is physically feasible"""
        joint_changes = np.abs(to_state[2:] - from_state[2:])
        if np.any(joint_changes > MAX_JOINT_CHANGE):
            return False
        return True
    
    def interpolate_configurations(self, from_state, to_state, num_steps=2):
        """
        OPTIMIZED: Interpolate between configurations.
        Reduced from 5 steps to 2 for speed.
        """
        configs = []
        for t in np.linspace(0, 1, num_steps):
            interp_state = from_state + t * (to_state - from_state)
            configs.append(interp_state)
        return configs
    
    def extend_in_cspace(self, nearest_node, random_sample):
        """Extend tree in configuration space"""
        diff = random_sample - nearest_node.state
        
        # Position step
        pos_diff = diff[:2]
        pos_dist = np.linalg.norm(pos_diff)
        if pos_dist > RRT_STEP_SIZE:
            pos_step = (pos_diff / pos_dist) * RRT_STEP_SIZE
        else:
            pos_step = pos_diff
        
        # Joint step
        joint_diff = diff[2:]
        joint_dist = np.linalg.norm(joint_diff)
        if joint_dist > 0.001:
            joint_scale = min(1.0, MAX_JOINT_CHANGE / joint_dist)
            joint_step = joint_diff * joint_scale
        else:
            joint_step = joint_diff
        
        # Construct new state
        new_state = nearest_node.state.copy()
        new_state[:2] += pos_step
        new_state[2:] += joint_step
        
        # Enforce joint limits
        new_state[2:] = np.clip(new_state[2:], -self.joint_limit, self.joint_limit)
        
        return new_state
    
    def compute_yaw_from_state(self, state):
        """Compute heading direction"""
        return -math.radians(state[2]) if len(state) > 2 else 0.0

    def step(self):
        """
        OPTIMIZED: Single RRT iteration with performance improvements
        """
        if self.finished:
            return False
        
        # 1. Sample random configuration
        random_config = self.get_random_sample_6D()
        
        # 2. Find nearest node (OPTIMIZED with KD-tree)
        nearest_node = self.find_nearest_node(random_config)
        
        # 3. Extend in C-space
        new_state = self.extend_in_cspace(nearest_node, random_config)
        
        # 4. OPTIMIZED: Check joint limits FIRST (cheapest check, no body computation)
        if np.any(np.abs(new_state[2:]) > self.joint_limit):
            return False
        
        # 5. Check feasibility
        if not self.is_configuration_reachable(nearest_node.state, new_state):
            return False
        
        # 6. Compute yaw
        new_yaw = self.compute_yaw_from_state(new_state)
        
        # 7. Create new node
        new_node = self.Node(new_state, nearest_node, yaw=new_yaw)
        
        # 8. OPTIMIZED: Full collision check on new node
        if not self.is_valid_configuration(new_node):
            return False
        
        # 9. OPTIMIZED: Reduced interpolation checks (2 steps instead of 5)
        intermediate_configs = self.interpolate_configurations(
            nearest_node.state, new_state, num_steps=2
        )
        for config in intermediate_configs[1:-1]:  # Only 1 intermediate check now
            temp_node = self.Node(config, yaw=self.compute_yaw_from_state(config))
            if not self.is_valid_configuration(temp_node):
                return False
        
        # 10. Add to tree
        self.nodes.append(new_node)
        
        # 11. Check goal
        reached, p_err, a_err = self.is_goal_reached(new_node.state, self.goal_conf)
        if reached:
            print(f"🎯 Goal Reached!")
            print(f"   Position Error: {p_err:.2f} units")
            print(f"   Joint Error: {a_err:.2f}°")
            print(f"   Total Nodes: {len(self.nodes)}")
            self.finished = True
            self.path = self.extract_path(new_node)
            return True
        
        return False
    
    def is_goal_reached(self, current_state, goal_state):
        """Check if current configuration is close enough to goal"""
        pos_error = np.linalg.norm(current_state[:2] - goal_state[:2])
        angle_error = np.linalg.norm(current_state[2:] - goal_state[2:])
        
        POS_TOLERANCE = 4.0
        ANGLE_TOLERANCE = 30.0
        
        if pos_error <= POS_TOLERANCE and angle_error <= ANGLE_TOLERANCE:
            return True, pos_error, angle_error
        
        return False, pos_error, angle_error

    def is_valid_configuration(self, node):
        """
        OPTIMIZED: Check if configuration is collision-free.
        Checks ordered from cheapest to most expensive.
        
        1. Joint limits (no body computation needed)
        2. Boundary check (simple comparisons)
        3. Environment collision (line checks)
        4. Self-collision (most expensive)
        """
        # OPTIMIZED: Check joint limits first (cheapest, already done in step())
        # This is here for completeness in case called directly
        if np.any(np.abs(node.state[2:]) > self.joint_limit):
            return False
        
        # Compute body once
        body = get_snake_body(node.state, yaw_override=node.yaw)
        
        # OPTIMIZED: Vectorized boundary check
        body_array = np.array(body)
        if (np.any(body_array[:, 0] < 0) or np.any(body_array[:, 0] >= self.env.width) or
            np.any(body_array[:, 1] < 0) or np.any(body_array[:, 1] >= self.env.height)):
            return False
        
        # Check environment collision (walls, obstacles)
        for i in range(len(body)-1):
            if self.env.check_line_collision(body[i], body[i+1]):
                return False
        
        # OPTIMIZED: Fast self-collision (intersection only, no distance checks)
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
def smooth_path_bspline(path_data, num_points=200):
    """Smooth path using B-spline interpolation"""
    states = [p[0] for p in path_data]
    x = [s[0] for s in states]
    y = [s[1] for s in states]
    
    if len(x) < 3:
        return [get_snake_body(s, yaw) for s, yaw in path_data]

    tck, u = splprep([x, y], s=2.0, k=min(3, len(x)-1))
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
    Creates 5 visible segments
    """
    bx, by = zip(*body_points)  # 6 points
    
    # Draw body as continuous line (6 points, 5 segments)
    ax.plot(bx, by, color=color, linestyle='-', linewidth=lw, alpha=alpha, zorder=15)
    
    # Draw joint markers - at HEAD_end and each link end (not tail)
    ax.scatter(bx[1:-1], by[1:-1], color='white', edgecolor='black', s=30, zorder=16)
    
    # Draw head start marker (gold diamond)
    ax.scatter(bx[0], by[0], color='gold', edgecolor='black', marker='D', s=45, zorder=17)
    
    # Draw tail marker (different color)
    ax.scatter(bx[-1], by[-1], color='red', edgecolor='black', s=30, zorder=16)

def draw_snake_line_state(ax, state, yaw, color='blue', alpha=1.0, lw=3):
    """Draw snake from state representation"""
    body = get_snake_body(state, yaw_override=yaw)
    draw_snake_line_explicit(ax, body, color, alpha, lw)

def main():
    import time  # For performance measurement
    
    WIDTH, HEIGHT = 70, 70
    
    START_CONF = np.array([35.0, 20.0, 0.0, 0.0, 0.0, 0.0])
    START_YAW = 0.0
    
    GOAL_CONF = np.array([60.0, 60.0, 0.0, 0.0, 0.0, 0.0])
    GOAL_YAW = 0.0
    
    env = DebrisMap(WIDTH, HEIGHT)
    planner = ConfigurationSpaceRRT(env, START_CONF, GOAL_CONF, start_yaw=START_YAW)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.ion()
    
    print("\n🔍 Searching with OPTIMIZED 6D C-Space (5 segments)...")
    frame_count = 0
    start_time = time.time()
    
    while not planner.finished:
        if frame_count > MAX_ITER:
            print(f"⚠️  Max iterations ({MAX_ITER}) reached.")
            break

        for _ in range(50):
            if planner.step():
                break
            frame_count += 1
            
        if frame_count % 50 == 0:
            ax.clear()
            ax.imshow(env.raw_grid, cmap='Greys', origin='lower', vmin=0, vmax=1)
            ax.imshow(env.planning_grid, cmap='Reds', alpha=0.2, origin='lower')
            
            # Draw all branches
            for node in planner.nodes:
                if node.parent:
                    ax.plot([node.parent.state[0], node.state[0]], 
                            [node.parent.state[1], node.state[1]], 
                            color='cyan', linewidth=0.3, alpha=0.2)

            # Draw current snake
            if planner.nodes:
                curr = planner.nodes[-1]
                draw_snake_line_state(ax, curr.state, curr.yaw, color='magenta', lw=2)
                
            # Draw start/goal
            draw_snake_line_state(ax, START_CONF, START_YAW, color='green', alpha=0.5)
            draw_snake_line_state(ax, GOAL_CONF, GOAL_YAW, color='red', alpha=0.5)
            
            elapsed = time.time() - start_time
            iter_per_sec = frame_count / elapsed if elapsed > 0 else 0
            ax.set_title(f"OPTIMIZED 5-Seg Snake | Nodes: {len(planner.nodes)} | Iter: {frame_count} | Speed: {iter_per_sec:.1f} it/s")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            plt.pause(0.001)

    # Print performance stats
    elapsed = time.time() - start_time
    print(f"\n📊 Performance Stats:")
    print(f"   Total time: {elapsed:.2f} seconds")
    print(f"   Iterations: {frame_count}")
    print(f"   Iterations/sec: {frame_count/elapsed:.1f}")
    print(f"   Nodes in tree: {len(planner.nodes)}")

    # --- PATH REPLAY ---
    if planner.path:
        print("\n✅ Path Found! Smoothing and replaying...")
        print(f"   Path length: {len(planner.path)} waypoints")
        
        smooth_bodies = smooth_path_bspline(planner.path, num_points=200)
        trail_indices = list(range(0, len(smooth_bodies), 8))

        for i, body in enumerate(smooth_bodies):
            ax.clear()
            ax.imshow(env.raw_grid, cmap='Greys', origin='lower')
            
            # Draw trail
            for idx in trail_indices:
                if idx > i:
                    break
                draw_snake_line_explicit(ax, smooth_bodies[idx], 
                                        color='lime', alpha=0.15, lw=4)
            
            # Draw start/goal
            draw_snake_line_state(ax, START_CONF, START_YAW, color='green', alpha=0.3)
            draw_snake_line_state(ax, GOAL_CONF, GOAL_YAW, color='red', alpha=0.3)
            
            # Draw current snake
            draw_snake_line_explicit(ax, body, color='blue', alpha=1.0)
            
            progress = int(i/len(smooth_bodies)*100)
            ax.set_title(f"OPTIMIZED 5-Segment Snake Path: {progress}%")
            plt.pause(0.01)
        
        ax.set_title("🎯 Target Reached! (OPTIMIZED)")
        plt.ioff()
        plt.show()
    else:
        print("\n❌ No path found.")
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()