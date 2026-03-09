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
        self.direction = 1.0 # 1.0 Forward, -1.0 Reverse

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
        # Weighting: [x, y, yaw, q1, q2, q3, q4]
        weights = np.array([config.W_POS, config.W_POS, config.W_YAW] + 
                           [config.W_JOINT]*config.NUM_JOINTS)
        data = np.array([n.state * weights for n in self.nodes])
        self.kdtree = KDTree(data)

    def get_random_sample(self):
        # Increased bias to 20% to help find the final slot
        if random.random() < 0.2:
            return self.goal_state
        rx = random.uniform(2, self.env.width - 2)
        ry = random.uniform(2, self.env.height - 2)
        ryaw = random.uniform(-math.pi, math.pi)
        joints = [random.uniform(-config.JOINT_LIMIT, config.JOINT_LIMIT) 
                  for _ in range(config.NUM_JOINTS)]
        return np.array([rx, ry, ryaw] + joints)

    def normalize_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def extend(self, from_state, to_state):
        """
        Hybrid Steering Extension:
        1. Far: Uses Non-Holonomic Arcs (Smooth, Safe).
        2. Near: Uses 'Parking Mode' (Turn-Drive-Turn) to snap to goal.
        """
        x, y, theta = from_state[0], from_state[1], from_state[2]
        dx = to_state[0] - x
        dy = to_state[1] - y
        dist_sq = dx*dx + dy*dy
        dist = math.sqrt(dist_sq)

        # --- MODE SELECTION ---
        # If we are closer than 1.5x the step size, we try a direct parking maneuver.
        # This abandons the "Safe" arc constraints to ensure we hit the goal.
        CLOSE_RANGE = config.RRT_STEP_SIZE * 1.5
        
        new_state = from_state.copy()
        direction = 1.0

        if dist < CLOSE_RANGE:
            # --- PARKING MODE (Aggressive Goal Snap) ---
            # 1. Turn to face target (Pivot)
            target_heading = math.atan2(dy, dx)
            
            # Decide Forward or Reverse
            diff_fwd = self.normalize_angle(target_heading - theta)
            diff_rev = self.normalize_angle(target_heading - (theta + math.pi))
            
            if abs(diff_fwd) < abs(diff_rev):
                # Forward
                direction = 1.0
                best_heading = target_heading
            else:
                # Reverse
                direction = -1.0
                best_heading = self.normalize_angle(target_heading + math.pi)

            # We assume we perform a "Turn In Place" here.
            # In RRT, we simulate the result: we are now at (x, y, best_heading).
            # Then we drive dist.
            
            new_state[0] += direction * dist * math.cos(best_heading)
            new_state[1] += direction * dist * math.sin(best_heading)
            
            # Finally, we align with the goal orientation (Another Pivot)
            # We assume we can rotate in place to the goal's exact yaw.
            new_state[2] = to_state[2] 
            
        else:
            # --- CRUISE MODE (Safe Arcs) ---
            # Standard non-holonomic drive for covering distance
            
            # Transform to local frame
            local_x = dx * math.cos(-theta) - dy * math.sin(-theta)
            local_y = dx * math.sin(-theta) + dy * math.cos(-theta)
            
            if local_x < 0:
                direction = -1.0
                local_x = -local_x
                local_y = -local_y

            curvature = 2.0 * local_y / dist_sq
            
            # Apply Curvature Limit (The "Safe" constraint)
            MAX_CURVATURE = 0.3 
            if abs(curvature) > MAX_CURVATURE:
                curvature = math.copysign(MAX_CURVATURE, curvature)

            step_len = config.RRT_STEP_SIZE
            if dist < step_len: step_len = dist

            if abs(curvature) < 0.001:
                new_state[0] += direction * step_len * math.cos(theta)
                new_state[1] += direction * step_len * math.sin(theta)
            else:
                radius = 1.0 / curvature
                yaw_change = step_len * curvature
                new_state[0] += direction * (math.sin(theta + yaw_change) - math.sin(theta)) * radius
                new_state[1] -= direction * (math.cos(theta + yaw_change) - math.cos(theta)) * radius
                new_state[2] += direction * yaw_change

        # --- Joint Interpolation (Always Active) ---
        joint_diff = to_state[3:] - from_state[3:]
        max_j = np.max(np.abs(joint_diff))
        if max_j > 1e-6:
            scale = min(1.0, config.MAX_JOINT_CHANGE / max_j)
            new_state[3:] += joint_diff * scale

        # Final constraints
        new_state[2] = self.normalize_angle(new_state[2])
        new_state[3:] = np.clip(new_state[3:], -config.JOINT_LIMIT, config.JOINT_LIMIT)
        
        return new_state, direction

    def step(self):
        if self.finished: return False
        
        rnd = self.get_random_sample()
        
        weights = np.array([config.W_POS, config.W_POS, config.W_YAW] + 
                           [config.W_JOINT]*config.NUM_JOINTS)
        _, idx = self.kdtree.query(rnd * weights)
        nearest = self.nodes[idx]
        
        new_state, direction_used = self.extend(nearest.state, rnd)
        
        if SnakeRobotModel.is_valid_state(new_state, self.env):
            new_node = Node(new_state, nearest)
            new_node.direction = direction_used
            self.nodes.append(new_node)
            
            if len(self.nodes) % 50 == 0:
                self.rebuild_kdtree()
            
            d_pos = np.linalg.norm(new_state[:2] - self.goal_state[:2])
            d_angles = np.linalg.norm(new_state[2:] - self.goal_state[2:])
            
            # Goal Check
            if d_pos < config.GOAL_POS_TOLERANCE and d_angles < np.deg2rad(config.GOAL_ANGLE_TOLERANCE):
                self.finished = True
                self.path = self.get_path(new_node)
                print(f"🎯 Goal Reached! Nodes: {len(self.nodes)}")
                return True
        return False

    def get_path(self, node):
        path = []
        while node:
            path.append((node.state, node.direction))
            node = node.parent
        return path[::-1]