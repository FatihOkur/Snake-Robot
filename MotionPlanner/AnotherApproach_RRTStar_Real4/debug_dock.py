"""Debug: what happens in the first docking attempt?"""
import numpy as np
import math
import sys
sys.path.insert(0, '.')

from environment import DebrisMap
from rrt_planner import TailBaseRRT, Node
from robot_model import SnakeRobotModel
import config

def calculate_straight_state_from_head(head_x, head_y, yaw_deg):
    return np.array([head_x, head_y, np.deg2rad(yaw_deg), 0, 0, 0])

env = DebrisMap(55, 85, map_type="complex_map")
START_STATE = calculate_straight_state_from_head(17.0, 22.0, 90)
GOAL_STATE = calculate_straight_state_from_head(26.0, 60.0, 90)

# Let's manually simulate docking from a close state to see what happens
# First let's find what kind of approach states the RRT produces
planner = TailBaseRRT(env, START_STATE, GOAL_STATE)

# Patch to capture the first docking attempt
first_dock_state = [None]
original_rlc = planner.run_local_controller.__func__

def capture_rlc(self, start_node):
    if first_dock_state[0] is None:
        state = start_node.state
        first_dock_state[0] = state.copy()
        print(f"\n[CAPTURE] First docking state:")
        print(f"  pos=({state[0]:.3f}, {state[1]:.3f}) yaw={math.degrees(state[2]):.2f}°")
        print(f"  q1={state[3]:.2f}° q2={state[4]:.2f}° q3={state[5]:.2f}°")
        print(f"  Goal: pos=({self.goal_state[0]:.3f}, {self.goal_state[1]:.3f}) yaw={math.degrees(self.goal_state[2]):.2f}°")
        dist = math.hypot(state[0]-self.goal_state[0], state[1]-self.goal_state[1])
        yaw_diff = abs(planner.normalize_angle(state[2] - self.goal_state[2]))
        print(f"  dist={dist:.3f} yaw_diff={math.degrees(yaw_diff):.2f}°")
        
        # Manually step through docking
        print(f"\n[DEBUG] Manual docking trace:")
        cur = state.copy()
        theta = cur[2]
        dx = self.goal_state[0] - cur[0]
        dy = self.goal_state[1] - cur[1]
        d = math.hypot(dx, dy)
        local_x = dx * math.cos(-theta) - dy * math.sin(-theta)
        locked_dir = 1.0 if local_x >= 0 else -1.0
        
        for step in range(20):
            dx = self.goal_state[0] - cur[0]
            dy = self.goal_state[1] - cur[1]
            d = math.hypot(dx, dy)
            theta = cur[2]
            yaw_err = planner.normalize_angle(self.goal_state[2] - cur[2])
            
            local_x = dx * math.cos(-theta) - dy * math.sin(-theta)
            local_y = dx * math.sin(-theta) + dy * math.cos(-theta)
            direction = locked_dir
            
            dist_sq = max(d*d, 0.001)
            curvature = 2.0 * local_y / dist_sq
            max_curv = math.sin(math.radians(config.JOINT_LIMIT)) / config.L_HEAD
            curvature = max(-max_curv, min(max_curv, curvature))
            
            L = config.L_HEAD
            sin_q1 = (curvature * L) / direction
            sin_q1 = max(-0.999, min(0.999, sin_q1))
            pp_angle = math.degrees(math.asin(sin_q1))
            
            yaw_error_signed = planner.normalize_angle(self.goal_state[2] - cur[2])
            if d < 1.0:
                pos_weight = d / 1.0
                yaw_corr = math.degrees(yaw_error_signed) * 0.8 / direction
                yaw_corr = max(-config.JOINT_LIMIT, min(config.JOINT_LIMIT, yaw_corr))
                q1_ideal = pos_weight * pp_angle + (1.0 - pos_weight) * yaw_corr
            else:
                q1_ideal = pp_angle
            
            step_len = min(0.6, max(0.1, d))
            
            new = cur.copy()
            q1_err = q1_ideal - cur[3]
            new[3] += np.clip(q1_err, -5.0, 5.0)
            new[4] += np.clip(self.goal_state[4]-cur[4], -5.0, 5.0)
            new[5] += np.clip(self.goal_state[5]-cur[5], -5.0, 5.0)
            new[3:] = np.clip(new[3:], -config.JOINT_LIMIT, config.JOINT_LIMIT)
            
            q1r = math.radians(new[3])
            yc = direction * (step_len / config.L_HEAD) * math.sin(q1r)
            new[0] += direction * step_len * math.cos(theta + yc/2)
            new[1] += direction * step_len * math.sin(theta + yc/2)
            new[2] = planner.normalize_angle(theta + yc)
            
            valid = SnakeRobotModel.is_valid_state(new, env)
            body = SnakeRobotModel.get_body_from_head_base(new)
            
            print(f"  step {step}: dist={d:.3f} yaw_err={math.degrees(yaw_err):.2f}° "
                  f"q1={new[3]:.2f}° pp={pp_angle:.2f}° q1_ideal={q1_ideal:.2f}° "
                  f"step_len={step_len:.3f} valid={valid}")
            if not valid:
                # Check which body segment collides
                for seg in range(len(body)-1):
                    if env.check_line_collision(body[seg], body[seg+1]):
                        print(f"    COLLISION on segment {seg}: {body[seg]} -> {body[seg+1]}")
                # Check bounds
                for idx, (bx, by) in enumerate(body):
                    if not (0 <= bx < env.width and 0 <= by < env.height):
                        print(f"    OUT OF BOUNDS at point {idx}: ({bx:.2f}, {by:.2f})")
                break
            cur = new
    
    return original_rlc(self, start_node)

import types
planner.run_local_controller = types.MethodType(capture_rlc, planner)

for i in range(30000):
    if planner.finished or first_dock_state[0] is not None:
        break
    planner.step()
    if i % 5000 == 0 and i > 0:
        print(f"   Iteration: {i} | Nodes: {len(planner.nodes)}")

if planner.finished:
    print("\n[OK] Path found!")
elif first_dock_state[0] is not None:
    print("\n[INFO] First docking attempt captured above.")
else:
    print("\n[FAIL] No docking attempt in 30000 iters")
