import numpy as np
import random
import math
from scipy.spatial import KDTree
import config
from robot_model import SnakeRobotModel

class Node:
    def __init__(self, state, parent=None):
        self.state = np.array(state, dtype=float)
        self.parent = parent
        self.yaw = state[2]
        self.direction = 1.0 

class TailBaseRRT:
    def __init__(self, env, start_state, goal_state):
        self.env = env
        self.start = Node(start_state)
        self.goal_state = np.array(goal_state)
        
        self.nodes = [self.start]
        self.kdtree = None
        self.rebuild_kdtree()
        
        self.finished = False
        self.path = None

    def rebuild_kdtree(self):
        # Increased Yaw weight so it favors nodes facing the right direction
        weights = np.array([config.W_POS, config.W_POS, 2.0] + 
                           [config.W_JOINT]*config.NUM_JOINTS)
        data = np.array([n.state * weights for n in self.nodes])
        self.kdtree = KDTree(data)

    def get_random_sample(self):
        # 20% bias towards goal
        if random.random() < 0.2:
            return self.goal_state
            
        rx = random.uniform(2, self.env.width - 2)
        ry = random.uniform(2, self.env.height - 2)
        ryaw = random.uniform(-math.pi, math.pi)
        
        # Gaussian sampling for joints: favors 0 (straight) but allows bending.
        # This prevents the snake from constantly tying itself into a collision knot.
        joints = [random.gauss(0, 15.0) for _ in range(config.NUM_JOINTS)]
        joints = np.clip(joints, -config.JOINT_LIMIT, config.JOINT_LIMIT).tolist()
        
        return np.array([rx, ry, ryaw] + joints)

    def normalize_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def extend(self, from_state, to_state):
        x, y, theta = from_state[0], from_state[1], from_state[2]
        dx = to_state[0] - x
        dy = to_state[1] - y
        dist_sq = dx*dx + dy*dy
        dist = math.sqrt(dist_sq)

        # 1. Local Frame for Direction & Curvature
        local_x = dx * math.cos(-theta) - dy * math.sin(-theta)
        local_y = dx * math.sin(-theta) + dy * math.cos(-theta)

        direction = 1.0
        if local_x < 0:
            direction = -1.0
            local_x = -local_x
            local_y = -local_y

        if dist_sq < 1e-6:
            curvature = 0.0
        else:
            curvature = 2.0 * local_y / dist_sq
            
        # Clamp curvature to the maximum physical capability of the joints
        max_curv = math.sin(math.radians(config.JOINT_LIMIT)) / config.SEGMENT_LENGTH
        curvature = max(-max_curv, min(max_curv, curvature))

        L = config.SEGMENT_LENGTH
        sin_q4_ideal = (curvature * L) / direction
        sin_q4_ideal = max(-0.999, min(0.999, sin_q4_ideal))
        q4_ideal_deg = math.degrees(math.asin(sin_q4_ideal))

        new_state = from_state.copy()

        # 2. Full Body Joint Stepping
        # Steer the front joints (q1, q2, q3) towards the randomly sampled RRT targets.
        SAFE_JOINT_STEP = 5.0 # Max degrees a joint can swing per step
        joint_diff = to_state[3:6] - from_state[3:6]
        max_j_front = np.max(np.abs(joint_diff))
        
        if max_j_front > 1e-6:
            scale_front = min(1.0, SAFE_JOINT_STEP / max_j_front)
            new_state[3:6] += joint_diff * scale_front

        # Steer q4 (the tail joint) towards the ideal spatial trajectory
        q4_diff = q4_ideal_deg - from_state[6]
        if abs(q4_diff) > 1e-6:
            scale_q4 = min(1.0, 10.0 / abs(q4_diff)) 
            new_state[6] += q4_diff * scale_q4

        new_state[3:] = np.clip(new_state[3:], -config.JOINT_LIMIT, config.JOINT_LIMIT)

        # 3. Step Base
        step_len = min(config.RRT_STEP_SIZE, dist)
        if step_len < 0.01: 
            step_len = config.RRT_STEP_SIZE

        q4_new_rad = math.radians(new_state[6])
        yaw_change = direction * (step_len / L) * math.sin(q4_new_rad)

        new_state[0] += direction * step_len * math.cos(theta + yaw_change / 2.0)
        new_state[1] += direction * step_len * math.sin(theta + yaw_change / 2.0)
        new_state[2] = self.normalize_angle(theta + yaw_change)

        return new_state, direction

    def run_local_controller(self, start_node):
        """
        Optimized Proportional Controller for smooth docking.
        Prevents jittering, curvature explosions, and reverse-gear thrashing.
        """
        current_node = start_node
        MAX_STEPS = 50 
        steps = 0
        
        # Lock the initial approach direction so we don't violently vibrate 
        # between forward and reverse if we overshoot by a millimeter.
        locked_direction = None

        while steps < MAX_STEPS:
            state = current_node.state.copy()
            dx = self.goal_state[0] - state[0]
            dy = self.goal_state[1] - state[1]
            dist = math.hypot(dx, dy)
            
            joint_diff = np.max(np.abs(state[3:] - self.goal_state[3:]))
            
            # Relaxed stopping condition. Millimeter perfection is physically 
            # unnecessary and causes endless wiggling. 
            if dist < 0.5 and joint_diff < 3.0:
                break
                
            # 1. Determine local coordinates
            theta = state[2]
            local_x = dx * math.cos(-theta) - dy * math.sin(-theta)
            local_y = dx * math.sin(-theta) + dy * math.cos(-theta)
            
            if locked_direction is None:
                locked_direction = 1.0 if local_x >= 0 else -1.0
            direction = locked_direction
            
            # --- FIX 1: Prevent Curvature Explosion ---
            # If we are very close to the goal, pure pursuit math explodes.
            # We enter "Terminal Approach": stop curving, just straighten out and coast.
            if dist < 1.5:
                q4_ideal_deg = self.goal_state[6] # Aim for final joint state (0)
                step_len = min(0.5, dist)         # Creep forward gently
            else:
                # Normal pure pursuit curvature
                curvature = 2.0 * local_y / (dist * dist)
                max_curv = math.sin(math.radians(config.JOINT_LIMIT)) / config.SEGMENT_LENGTH
                curvature = max(-max_curv, min(max_curv, curvature))
                
                L = config.SEGMENT_LENGTH
                sin_q4 = (curvature * L) / direction
                sin_q4 = max(-0.999, min(0.999, sin_q4))
                q4_ideal_deg = math.degrees(math.asin(sin_q4))
                
                step_len = min(config.RRT_STEP_SIZE, dist) 
                
            new_state = state.copy()
            
            # --- FIX 2: Smooth Proportional Joint Straightening ---
            # Instead of snapping at maximum speed, multiply the error by a fraction (0.4)
            # This makes the joints smoothly "ease" into the straight position without jittering.
            joint_error_front = self.goal_state[3:6] - state[3:6]
            for i in range(3):
                new_state[3+i] += joint_error_front[i] * 0.4 
                    
            # Smooth steering for the tail joint
            q4_error = q4_ideal_deg - state[6]
            new_state[6] += q4_error * 0.5 
                
            new_state[3:] = np.clip(new_state[3:], -config.JOINT_LIMIT, config.JOINT_LIMIT)
            
            # 3. Step Base
            q4_new_rad = math.radians(new_state[6])
            yaw_change = direction * (step_len / config.SEGMENT_LENGTH) * math.sin(q4_new_rad)
            
            new_state[0] += direction * step_len * math.cos(theta + yaw_change / 2.0)
            new_state[1] += direction * step_len * math.sin(theta + yaw_change / 2.0)
            new_state[2] = self.normalize_angle(theta + yaw_change)
            
            # Collision Check
            if not SnakeRobotModel.is_valid_state(new_state, self.env):
                print("   ⚠️ Docking trajectory blocked! Returning to RRT...")
                return None 
            
            new_node = Node(new_state, current_node)
            new_node.direction = direction
            current_node = new_node
            steps += 1
            
        return current_node

    def step(self):
        if self.finished: return False
        
        rnd = self.get_random_sample()
        
        weights = np.array([config.W_POS, config.W_POS, 2.0] + 
                           [config.W_JOINT]*config.NUM_JOINTS)
        _, idx = self.kdtree.query(rnd * weights)
        nearest = self.nodes[idx]
        
        new_state, direction_used = self.extend(nearest.state, rnd)
        
        if SnakeRobotModel.is_valid_state(new_state, self.env):
            new_node = Node(new_state, nearest)
            new_node.direction = direction_used
            self.nodes.append(new_node)
            
            if len(self.nodes) % 100 == 0:
                self.rebuild_kdtree()
            
            # --- GOAL CHECKING & HANDOFF ---
            d_pos = math.hypot(new_state[0] - self.goal_state[0], new_state[1] - self.goal_state[1])
            yaw_diff = abs(self.normalize_angle(new_state[2] - self.goal_state[2]))
            
            if d_pos < config.GOAL_POS_TOLERANCE and yaw_diff < np.deg2rad(config.GOAL_ANGLE_TOLERANCE):
                print(f"🎯 RRT reached tolerance boundary! Nodes: {len(self.nodes)}")
                print("🛬 Initiating Local Controller Docking Phase...")
                
                # Try to dock
                final_node = self.run_local_controller(new_node)
                
                # If docking succeeds (doesn't return None), we are done!
                if final_node is not None:
                    self.finished = True
                    self.path = self.get_path(final_node)
                    print("✅ Docking Complete! Path Generated.")
                    return True
                # If it fails, the RRT loop just continues naturally to find a better angle.
                
        return False

    def get_path(self, node):
        path = []
        while node:
            path.append((node.state, node.direction))
            node = node.parent
        return path[::-1]