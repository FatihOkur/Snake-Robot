import numpy as np
import random
import math
from scipy.spatial import KDTree
import config
from robot_model import SnakeRobotModel

class Node:
    def __init__(self, state, parent=None, edge_cost=0.0):
        self.state = np.array(state, dtype=float)
        self.parent = parent
        self.yaw = state[2]
        self.direction = 1.0 
        if parent is None:
            self.cost = 0.0
        else:
            self.cost = parent.cost + edge_cost

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
        self.final_node = None
        self.docking_cooldown = 0

    def rebuild_kdtree(self):
        # Increased Yaw weight so it favors nodes facing the right direction
        weights = np.array([config.W_POS, config.W_POS, 2.0] + 
                           [config.W_JOINT]*config.NUM_JOINTS)
        data = np.array([n.state * weights for n in self.nodes])
        self.kdtree = KDTree(data)

    def get_random_sample(self):
        # 20% bias towards goal, ONLY if not in cooldown
        if self.docking_cooldown <= 0 and random.random() < 0.2:
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

    def calculate_edge_cost(self, state1, state2):
        z1 = self.env.get_elevation(state1[0], state1[1])
        z2 = self.env.get_elevation(state2[0], state2[1])
        dist_moved_3d = math.sqrt((state2[0] - state1[0])**2 + (state2[1] - state1[1])**2 + (z2 - z1)**2)
        joint_changes = np.sum(np.abs(state2[3:] - state1[3:]))
        
        grid_x = max(0, min(self.env.width - 1, int(state2[0])))
        grid_y = max(0, min(self.env.height - 1, int(state2[1])))
        clearance = self.env.distance_field[grid_y, grid_x]
        obs_penalty = config.COST_OBS_WEIGHT / (clearance + 0.1)
        
        return (config.COST_DIST_WEIGHT * dist_moved_3d + 
                config.COST_JOINT_WEIGHT * joint_changes + 
                config.COST_PITCH_WEIGHT * abs(z2 - z1) + 
                obs_penalty)

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
        sin_q3_ideal = (curvature * L) / direction
        sin_q3_ideal = max(-0.999, min(0.999, sin_q3_ideal))
        q3_ideal_deg = math.degrees(math.asin(sin_q3_ideal))

        new_state = from_state.copy()

        # 2. Full Body Joint Stepping
        # Blend front joint targets 30% toward goal for faster convergence.
        SAFE_JOINT_STEP = 5.0 # Max degrees a joint can swing per step
        GOAL_JOINT_BLEND = 0.3
        joint_target = (1.0 - GOAL_JOINT_BLEND) * to_state[3:5] + GOAL_JOINT_BLEND * self.goal_state[3:5]
        joint_diff = joint_target - from_state[3:5]
        max_j_front = np.max(np.abs(joint_diff))
        
        if max_j_front > 1e-6:
            scale_front = min(1.0, SAFE_JOINT_STEP / max_j_front)
            new_state[3:5] += joint_diff * scale_front

        # Steer q3 (the tail joint) towards the ideal spatial trajectory
        q3_diff = q3_ideal_deg - from_state[5]
        if abs(q3_diff) > 1e-6:
            scale_q3 = min(1.0, 10.0 / abs(q3_diff)) 
            new_state[5] += q3_diff * scale_q3

        new_state[3:] = np.clip(new_state[3:], -config.JOINT_LIMIT, config.JOINT_LIMIT)

        # 3. Step Base
        step_len = min(config.RRT_STEP_SIZE, dist)
        if step_len < 0.01: 
            step_len = config.RRT_STEP_SIZE

        q3_new_rad = math.radians(new_state[5])
        yaw_change = direction * (step_len / L) * math.sin(q3_new_rad)

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
            
            # Calculate the yaw difference
            yaw_diff = abs(self.normalize_angle(state[2] - self.goal_state[2]))
            joint_diff = np.max(np.abs(state[3:] - self.goal_state[3:]))
            
            # Tighter stopping condition for high precision: MUST include yaw_diff
            if dist < 0.2 and joint_diff < 2.0 and yaw_diff < math.radians(3.0):
                # Snap exactly to the mathematical goal state for a perfect fit
                edge_cost = self.calculate_edge_cost(current_node.state, self.goal_state)
                exact_node = Node(self.goal_state, current_node, edge_cost)
                exact_node.direction = locked_direction if locked_direction is not None else 1.0
                
                # Return the exact goal node if it is collision-free
                if SnakeRobotModel.is_valid_state(self.goal_state, self.env):
                    return exact_node
                break
                
            # Determine local coordinates
            theta = state[2]
            local_x = dx * math.cos(-theta) - dy * math.sin(-theta)
            local_y = dx * math.sin(-theta) + dy * math.cos(-theta)
            
            if locked_direction is None:
                locked_direction = 1.0 if local_x >= 0 else -1.0
            direction = locked_direction
            
            # Pure pursuit curvature (safely clamped)
            dist_sq = max(dist * dist, 0.001)
            curvature = 2.0 * local_y / dist_sq
            
            # ANTI-ORBITING FIX
            if dist < 1.0:
                curvature *= dist
                
            max_curv = math.sin(math.radians(config.JOINT_LIMIT)) / config.SEGMENT_LENGTH
            curvature = max(-max_curv, min(max_curv, curvature))
            
            L = config.SEGMENT_LENGTH
            sin_q3 = (curvature * L) / direction
            sin_q3 = max(-0.999, min(0.999, sin_q3))
            pure_pursuit_angle = math.degrees(math.asin(sin_q3))
            
            # ADAPTIVE BLENDING
            if dist < 0.5:
                # Smoothly transition from position-seeking to angle-matching
                weight = dist / 0.5 
                q3_ideal_deg = (weight * pure_pursuit_angle) + ((1.0 - weight) * self.goal_state[5])
            else:
                q3_ideal_deg = pure_pursuit_angle
                
            # RATE-LIMITED BASE TRANSLATION
            # Move at max speed, unless the goal is closer than one max step.
            step_len = min(config.RRT_STEP_SIZE, dist)
                
            new_state = state.copy()
            
            # RATE-LIMITED JOINT CONTROL (No teleporting!)
            SAFE_JOINT_STEP = 5.0 # Max degrees a servo can swing per step
            
            # Front joints (q1, q2)
            joint_error_front = self.goal_state[3:5] - state[3:5]
            # Strictly clamp the error to physical limits
            joint_step_front = np.clip(joint_error_front, -SAFE_JOINT_STEP, SAFE_JOINT_STEP)
            new_state[3:5] += joint_step_front 
                    
            # Tail joint (q3)
            q3_error = q3_ideal_deg - state[5]
            q3_step = np.clip(q3_error, -SAFE_JOINT_STEP, SAFE_JOINT_STEP)
            new_state[5] += q3_step 
                
            new_state[3:] = np.clip(new_state[3:], -config.JOINT_LIMIT, config.JOINT_LIMIT)
            
            # Step Base
            q3_new_rad = math.radians(new_state[5])
            yaw_change = direction * (step_len / config.SEGMENT_LENGTH) * math.sin(q3_new_rad)
            
            new_state[0] += direction * step_len * math.cos(theta + yaw_change / 2.0)
            new_state[1] += direction * step_len * math.sin(theta + yaw_change / 2.0)
            new_state[2] = self.normalize_angle(theta + yaw_change)
            
            # Collision Check
            if not SnakeRobotModel.is_valid_state(new_state, self.env):
                print("   [WARN] Docking trajectory blocked! Returning to RRT...")
                return None 
            
            edge_cost = self.calculate_edge_cost(current_node.state, new_state)
            new_node = Node(new_state, current_node, edge_cost)
            new_node.direction = direction
            current_node = new_node
            steps += 1
            
        # If it runs out of steps without perfectly docking, fail and let RRT try a different approach angle
        print("   [WARN] Docking timed out before perfect alignment. Returning to RRT...")
        return None

    def step(self):
        if self.finished: return False
        
        # Decrement cooldown timer
        if self.docking_cooldown > 0:
            self.docking_cooldown -= 1
            
        rnd = self.get_random_sample()
        
        weights = np.array([config.W_POS, config.W_POS, 2.0] + 
                           [config.W_JOINT]*config.NUM_JOINTS)
        
        # RRT* specific logic
        k = min(getattr(config, 'RRT_STAR_K', 15), len(self.nodes))
        _, indices = self.kdtree.query(rnd * weights, k=k)
        
        # Handle edge case where KDTree returns a 1D array if k=1
        if k == 1:
            indices = [indices]
            
        best_parent = None
        best_cost = float('inf')
        best_state = None
        best_direction = None
        best_edge_cost = 0.0
        
        for idx in indices:
            near_node = self.nodes[idx]
            new_state, direction_used = self.extend(near_node.state, rnd)
            
            if SnakeRobotModel.is_valid_state(new_state, self.env):
                edge_cost = self.calculate_edge_cost(near_node.state, new_state)
                total_cost = near_node.cost + edge_cost
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_parent = near_node
                    best_state = new_state
                    best_direction = direction_used
                    best_edge_cost = edge_cost
                    
        if best_parent is not None:
            new_node = Node(best_state, best_parent, best_edge_cost)
            new_node.direction = best_direction
            self.nodes.append(new_node)
            
            # OPTIMIZATION: Dynamic KD-Tree Rebuild
            rebuild_interval = max(50, len(self.nodes) // 20) 
            if len(self.nodes) % rebuild_interval == 0:
                self.rebuild_kdtree()
            
            # --- GOAL CHECKING & HANDOFF ---
            d_pos = math.hypot(best_state[0] - self.goal_state[0], best_state[1] - self.goal_state[1])
            yaw_diff = abs(self.normalize_angle(best_state[2] - self.goal_state[2]))
            
            # Only attempt docking if we are NOT on cooldown
            if self.docking_cooldown <= 0 and d_pos < config.GOAL_POS_TOLERANCE and yaw_diff < np.deg2rad(config.GOAL_ANGLE_TOLERANCE):
                print(f"[TARGET] RRT* reached tolerance boundary! Nodes: {len(self.nodes)}")
                print("[DOCK] Initiating Local Controller Docking Phase...")
                
                final_node = self.run_local_controller(new_node)
                
                if final_node is not None:
                    self.finished = True
                    self.final_node = final_node
                    self.path = self.get_path(final_node)
                    print(f"[OK] Docking Complete! Final Path Cost: {final_node.cost:.2f}")
                    return True
                else:
                    # PENALTY: Docking failed. Put docking on cooldown to force RRT to explore elsewhere
                    self.docking_cooldown = 500
                    new_node.cost += 50.0  # Artificially inflate cost so RRT* prefers other branches
                
        return False

    def get_path(self, node):
        path = []
        while node:
            path.append((node.state, node.direction))
            node = node.parent
        return path[::-1]
