import numpy as np
import math
import config
from environment import line_segment_intersection

class SnakeRobotModel:
    @staticmethod
    def get_body_from_tail_state(state):
        """
        Reconstructs the body starting from the TAIL (Root).
        
        State Vector (7D): [x_tail, y_tail, theta_tail, q1, q2, q3, q4]
        
        Kinematic Chain (Strict Hierarchy):
        [Tail/Link4] --(J4)-- [Link3] --(J3)-- [Link2] --(J2)-- [Link1] --(J1)-- [Head]
        
        Constraints:
        - Tail is the Base.
        - theta_tail determines the orientation of the Tail segment.
        - Joints (q) add to the angle of the NEXT segment in the chain.
        """
        x_tail, y_tail, th_tail = state[0], state[1], state[2]
        joint_angles = state[3:] # [q1, q2, q3, q4]
        
        L = config.SEGMENT_LENGTH
        
        # 1. Define Tail Segment Geometry
        # We assume (x_tail, y_tail) is the FRONT (Joint 4 location) of the Tail segment.
        # The "End" of the tail is behind it.
        tail_end_x = x_tail - L * math.cos(th_tail)
        tail_end_y = y_tail - L * math.sin(th_tail)
        
        # Point 0: Tail End, Point 1: Tail Front (J4)
        chain_points = [(tail_end_x, tail_end_y), (x_tail, y_tail)]
        
        current_x, current_y = x_tail, y_tail
        current_yaw = th_tail
        
        # 2. Propagate Forward towards Head
        # Order: Tail(J4) -> Link3(J3) -> Link2(J2) -> Link1(J1) -> Head
        # Joint Indices: q4 (index 3), q3 (index 2), q2 (index 1), q1 (index 0)
        ordered_joints = [joint_angles[3], joint_angles[2], joint_angles[1], joint_angles[0]]
        
        for q in ordered_joints:
            # The joint angle rotates the NEXT segment relative to the CURRENT one.
            # Next_Yaw = Current_Yaw + q
            next_yaw = current_yaw + math.radians(q)
            
            # Calculate end of this new segment
            next_x = current_x + L * math.cos(next_yaw)
            next_y = current_y + L * math.sin(next_yaw)
            
            chain_points.append((next_x, next_y))
            
            # Update for next link
            current_x, current_y = next_x, next_y
            current_yaw = next_yaw
            
        # chain_points is [Tail_End, J4, J3, J2, J1, Head_Tip]
        # Reverse it so index 0 is Head (standard for visualization)
        return list(reversed(chain_points))

    @staticmethod
    def check_self_collision(body_points):
        n = len(body_points)
        for i in range(n - 2):
            for j in range(i + 2, n - 1):
                if line_segment_intersection(body_points[i], body_points[i+1], 
                                             body_points[j], body_points[j+1]):
                    return True
        return False

    @staticmethod
    def is_valid_state(state, env):
        # 1. Check Joint Limits
        if np.any(np.abs(state[3:]) > config.JOINT_LIMIT):
            return False

        # 2. Reconstruct Body
        body = SnakeRobotModel.get_body_from_tail_state(state)
        
        # 3. Check Map Boundaries
        for x, y in body:
            if not (0 <= x < env.width and 0 <= y < env.height):
                return False

        # 4. Check Environment Collision
        for i in range(len(body)-1):
            if env.check_line_collision(body[i], body[i+1]):
                return False
                
        # 5. Check Self Collision
        if SnakeRobotModel.check_self_collision(body):
            return False
            
        return True