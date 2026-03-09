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
        # State: [x_tail, y_tail, th_tail, q1, q2, q3, q4]

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
        # Weighting: Position is highest priority
        weights = np.array([config.W_POS, config.W_POS, config.W_YAW] + 
                           [config.W_JOINT]*config.NUM_JOINTS)
        data = np.array([n.state * weights for n in self.nodes])
        self.kdtree = KDTree(data)

    def get_random_sample(self):
        # 10% Goal Bias
        if random.random() < 0.1: 
            return self.goal_state
            
        rx = random.uniform(2, self.env.width - 2)
        ry = random.uniform(2, self.env.height - 2)
        rth = random.uniform(-math.pi, math.pi)
        
        joints = [random.uniform(-config.JOINT_LIMIT, config.JOINT_LIMIT) 
                  for _ in range(config.NUM_JOINTS)]
        
        return np.array([rx, ry, rth] + joints)

    def normalize_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def extend(self, from_state, to_state):
        """
        Non-Holonomic Steering for Tail (Base).
        """
        fx, fy, fth = from_state[0], from_state[1], from_state[2]
        tx, ty = to_state[0], to_state[1]
        
        dx = tx - fx
        dy = ty - fy
        dist = math.hypot(dx, dy)
        
        # 1. Determine Steering Direction (Forward/Reverse)
        target_heading = math.atan2(dy, dx)
        diff_fwd = self.normalize_angle(target_heading - fth)
        diff_rev = self.normalize_angle(target_heading - (fth + math.pi))
        
        if abs(diff_fwd) < abs(diff_rev):
            direction = 1.0
            steer_error = diff_fwd
        else:
            direction = -1.0
            steer_error = diff_rev
            
        # 2. Limit Turn Rate
        steer_step = np.clip(steer_error, -config.MAX_TURN_ANGLE, config.MAX_TURN_ANGLE)
        new_th = self.normalize_angle(fth + steer_step)
        
        # 3. Move
        step_dist = min(dist, config.RRT_STEP_SIZE)
        move_heading = new_th if direction == 1.0 else self.normalize_angle(new_th + math.pi)
        
        new_x = fx + step_dist * math.cos(move_heading)
        new_y = fy + step_dist * math.sin(move_heading)
        
        # 4. Interpolate Joints
        current_joints = from_state[3:]
        target_joints = to_state[3:]
        joint_diff = target_joints - current_joints
        
        max_j = np.max(np.abs(joint_diff))
        if max_j > 1e-6:
            scale = min(1.0, config.MAX_JOINT_CHANGE / max_j)
            new_joints = current_joints + joint_diff * scale
        else:
            new_joints = current_joints
            
        new_joints = np.clip(new_joints, -config.JOINT_LIMIT, config.JOINT_LIMIT)
        
        return np.array([new_x, new_y, new_th] + new_joints.tolist())

    def step(self):
        if self.finished: return False
        
        rnd = self.get_random_sample()
        
        weights = np.array([config.W_POS, config.W_POS, config.W_YAW] + 
                           [config.W_JOINT]*config.NUM_JOINTS)
        _, idx = self.kdtree.query(rnd * weights)
        nearest = self.nodes[idx]
        
        new_state = self.extend(nearest.state, rnd)
        
        if SnakeRobotModel.is_valid_state(new_state, self.env):
            new_node = Node(new_state, nearest)
            self.nodes.append(new_node)
            
            if len(self.nodes) % 500 == 0:
                self.rebuild_kdtree()
            
            # Relaxed Goal Check: Just check if Head or Tail is close
            d_pos = np.linalg.norm(new_state[:2] - self.goal_state[:2])
            
            if d_pos < config.GOAL_POS_TOLERANCE:
                self.finished = True
                self.path = self.get_path(new_node)
                print(f"🎯 Goal Reached! Nodes: {len(self.nodes)}")
                return True
        return False

    def get_path(self, node):
        path = []
        while node:
            path.append(node.state)
            node = node.parent
        return path[::-1]