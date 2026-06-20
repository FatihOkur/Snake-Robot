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

    def calculate_edge_cost(self, state1, state2):
        dist_moved = math.hypot(state2[0] - state1[0], state2[1] - state1[1])
        joint_changes = np.sum(np.abs(state2[3:] - state1[3:]))
        
        grid_x = max(0, min(self.env.width - 1, int(state2[0])))
        grid_y = max(0, min(self.env.height - 1, int(state2[1])))
        clearance = self.env.distance_field[grid_y, grid_x]
        obs_penalty = config.COST_OBS_WEIGHT / (clearance + 0.1)
        
        return (config.COST_DIST_WEIGHT * dist_moved + 
                config.COST_JOINT_WEIGHT * joint_changes + 
                obs_penalty)

    def _is_docking_feasible(self, state):
        """Quick pre-check: is the docking corridor clear enough to bother trying?"""
        # 1. Check clearance along the straight line from current base to goal base
        n_samples = 10
        for i in range(n_samples + 1):
            t = i / n_samples
            x = state[0] + t * (self.goal_state[0] - state[0])
            y = state[1] + t * (self.goal_state[1] - state[1])
            gx = int(max(0, min(self.env.width - 1, x)))
            gy = int(max(0, min(self.env.height - 1, y)))
            clearance = self.env.distance_field[gy, gx]
            if clearance < self.env.safe_threshold + 1.0:
                return False
        
        # 2. Large joint differences mean wide sweeps → more collision risk
        joint_diff = np.max(np.abs(state[3:] - self.goal_state[3:]))
        if joint_diff > 25.0:
            return False
        
        return True

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
        max_curv = math.sin(math.radians(config.JOINT_LIMIT)) / config.L_SEG3
        curvature = max(-max_curv, min(max_curv, curvature))

        L = config.L_SEG3
        sin_q2_ideal = (curvature * L) / direction
        sin_q2_ideal = max(-0.999, min(0.999, sin_q2_ideal))
        q2_ideal_deg = math.degrees(math.asin(sin_q2_ideal))

        new_state = from_state.copy()

        # 2. Full Body Joint Stepping
        SAFE_JOINT_STEP = 5.0 # Max degrees a joint can swing per step
        GOAL_JOINT_BLEND = 0.3

        # q1 (index 3): Blend toward goal target
        q1_target = (1.0 - GOAL_JOINT_BLEND) * to_state[3] + GOAL_JOINT_BLEND * self.goal_state[3]
        q1_diff = q1_target - from_state[3]
        if abs(q1_diff) > 1e-6:
            scale_q1 = min(1.0, SAFE_JOINT_STEP / abs(q1_diff))
            new_state[3] += q1_diff * scale_q1

        # q3 (index 5): Blend toward goal target
        q3_target = (1.0 - GOAL_JOINT_BLEND) * to_state[5] + GOAL_JOINT_BLEND * self.goal_state[5]
        q3_diff = q3_target - from_state[5]
        if abs(q3_diff) > 1e-6:
            scale_q3 = min(1.0, SAFE_JOINT_STEP / abs(q3_diff))
            new_state[5] += q3_diff * scale_q3

        # q2 (index 4): Steer toward the ideal spatial trajectory (primary steering)
        q2_diff = q2_ideal_deg - from_state[4]
        if abs(q2_diff) > 1e-6:
            scale_q2 = min(1.0, 10.0 / abs(q2_diff)) 
            new_state[4] += q2_diff * scale_q2

        new_state[3:] = np.clip(new_state[3:], -config.JOINT_LIMIT, config.JOINT_LIMIT)

        # 3. Step Base — yaw change driven by q2 (the steering joint)
        step_len = min(config.RRT_STEP_SIZE, dist)
        if step_len < 0.01: 
            step_len = config.RRT_STEP_SIZE

        q2_new_rad = math.radians(new_state[4])
        yaw_change = direction * (step_len / config.L_SEG3) * math.sin(q2_new_rad)

        new_state[0] += direction * step_len * math.cos(theta + yaw_change / 2.0)
        new_state[1] += direction * step_len * math.sin(theta + yaw_change / 2.0)
        new_state[2] = self.normalize_angle(theta + yaw_change)

        return new_state, direction

    def run_local_controller(self, start_node):
        """
        Optimized Proportional Controller for smooth docking.
        q2 is the primary steering joint. q1 and q3 snap toward goal.
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
            max_curv = math.sin(math.radians(config.JOINT_LIMIT)) / config.L_SEG3
            curvature = max(-max_curv, min(max_curv, curvature))
            
            L = config.L_SEG3
            sin_q2 = (curvature * L) / direction
            sin_q2 = max(-0.999, min(0.999, sin_q2))
            pure_pursuit_angle = math.degrees(math.asin(sin_q2))
            
            # ADAPTIVE BLENDING for q2 (steering joint)
            if dist < 0.5:
                # Smoothly transition from position-seeking to angle-matching
                weight = dist / 0.5 
                q2_ideal_deg = (weight * pure_pursuit_angle) + ((1.0 - weight) * self.goal_state[4])
            else:
                q2_ideal_deg = pure_pursuit_angle
                
            # RATE-LIMITED BASE TRANSLATION
            # Move at max speed, unless the goal is closer than one max step.
            step_len = min(1.0, dist)  # Finer steps for docking precision
                
            new_state = state.copy()
            
            # RATE-LIMITED JOINT CONTROL (No teleporting!)
            SAFE_JOINT_STEP = 5.0 # Max degrees a servo can swing per step
            
            # q1 (index 3): Snap toward goal
            q1_error = self.goal_state[3] - state[3]
            q1_step = np.clip(q1_error, -SAFE_JOINT_STEP, SAFE_JOINT_STEP)
            new_state[3] += q1_step

            # q3 (index 5): Snap toward goal
            q3_error = self.goal_state[5] - state[5]
            q3_step = np.clip(q3_error, -SAFE_JOINT_STEP, SAFE_JOINT_STEP)
            new_state[5] += q3_step
                    
            # q2 (index 4): Steer toward the ideal docking trajectory
            q2_error = q2_ideal_deg - state[4]
            q2_step = np.clip(q2_error, -SAFE_JOINT_STEP, SAFE_JOINT_STEP)
            new_state[4] += q2_step 
                
            new_state[3:] = np.clip(new_state[3:], -config.JOINT_LIMIT, config.JOINT_LIMIT)
            
            # Step Base — yaw change driven by q2 (steering joint)
            q2_new_rad = math.radians(new_state[4])
            yaw_change = direction * (step_len / config.L_SEG3) * math.sin(q2_new_rad)
            
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
        
        rnd = self.get_random_sample()
        
        weights = np.array([config.W_POS, config.W_POS, 2.0] + 
                           [config.W_JOINT]*config.NUM_JOINTS)
        
        k = min(config.RRT_STAR_K, len(self.nodes))
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
            
            if len(self.nodes) % 50 == 0:
                self.rebuild_kdtree()
            
            # --- GOAL CHECKING & HANDOFF ---
            d_pos = math.hypot(best_state[0] - self.goal_state[0], best_state[1] - self.goal_state[1])
            yaw_diff = abs(self.normalize_angle(best_state[2] - self.goal_state[2]))
            
            if d_pos < config.GOAL_POS_TOLERANCE and yaw_diff < np.deg2rad(config.GOAL_ANGLE_TOLERANCE):
                # Quick pre-check: skip expensive docking if corridor is too tight
                if self._is_docking_feasible(best_state):
                    print(f"[TARGET] RRT reached tolerance boundary! Nodes: {len(self.nodes)}")
                    print("[DOCK] Initiating Local Controller Docking Phase...")
                    
                    # Try to dock
                    final_node = self.run_local_controller(new_node)
                    
                    # If docking succeeds (doesn't return None), we are done!
                    if final_node is not None:
                        self.finished = True
                        self.final_node = final_node
                        self.path = self.get_path(final_node)
                        print("[OK] Docking Complete! Path Generated.")
                        return True
                    # If it fails, the RRT loop just continues naturally to find a better angle.
                
        return False

    def get_path(self, node):
        path = []
        while node:
            path.append((node.state, node.direction))
            node = node.parent
        return path[::-1]
